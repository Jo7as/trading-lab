#!/usr/bin/env bash
# Fast health check for trading-lab (signal bot + evolution).
# Exit 0 = OK, 1 = alert. Designed for Cursor cron — finishes in <30s.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_HEARTBEAT_AGE_SEC=7200   # 2h without heartbeat update = stale
TMP_DIR="${TMPDIR:-/tmp}/hermes-monitor-$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

alerts=()
notes=()

if ! command -v railway >/dev/null 2>&1; then
  echo "ALERT: railway CLI not found"
  exit 1
fi

# --- hermes-signals ---
if ! railway logs -s hermes-signals --lines 15 2>&1 | tee "$TMP_DIR/signal.log" | rg -q "processed closed bar|next poll"; then
  if rg -qi "error|failed|traceback" "$TMP_DIR/signal.log"; then
    alerts+=("hermes-signals: errors in recent logs")
  else
    notes+=("hermes-signals: no recent bar processing in last 15 log lines")
  fi
fi

if railway volume files --volume hermes-signals-volume download \
    /app/live-state/heartbeat.json "$TMP_DIR/hb.json" --overwrite >/dev/null 2>&1; then
  hb_age="$(python3 - "$TMP_DIR/hb.json" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
age = time.time() - float(data.get("ts", 0))
print(int(age))
PY
)"
  price="$(python3 -c "import json; print(json.load(open('$TMP_DIR/hb.json')).get('price','?'))")"
  open_pos="$(python3 -c "import json; d=json.load(open('$TMP_DIR/hb.json')); print('yes' if d.get('open_position') else 'no')")"
  notes+=("gold heartbeat age=${hb_age}s price=${price} open=${open_pos}")
  if [[ "$hb_age" -gt "$MAX_HEARTBEAT_AGE_SEC" ]]; then
    alerts+=("hermes-signals: heartbeat stale (${hb_age}s)")
  fi
else
  alerts+=("hermes-signals: cannot download heartbeat.json")
fi

# --- sparkling-charm (Gold evolution — intentionally paused) ---
notes+=("sparkling-charm: paused (Gold evolution stopped)")

# --- hermes-jpy-evolution ---
JPY_BASELINE_OOS=12.9
STAGNATION_WARN=30
STAGNATION_ALERT=50
CHECKPOINT_MAX_AGE_SEC=14400   # 4h without best_oos progress → flag stagnation

railway logs -s hermes-jpy-evolution --lines 120 2>&1 | tee "$TMP_DIR/jpy_evol.log" >/dev/null || true

jpy_parse="$(python3 - "$TMP_DIR/jpy_evol.log" "$JPY_BASELINE_OOS" <<'PY'
import re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
baseline = float(sys.argv[2])

rounds = [int(m.group(1)) for m in re.finditer(r"Round #(\d+) complete", text)]
if not rounds:
    rounds = [int(m.group(1)) for m in re.finditer(r"Evolution round:\s*(\d+)", text)]
stags = [int(m.group(1)) for m in re.finditer(r"Stagnation: (\d+) rounds", text)]
oos_pnls = [
    float(m.group(1))
    for m in re.finditer(
        r"OUT-OF-SAMPLE \(holdout\) RESULTS\s+.*?Total PnL:\s+([-\d.]+)%",
        text,
        re.S,
    )
]
best_oos_log = re.findall(r"\[best-oos\] new record:\s+([-\d.]+)%", text)
long_only = re.findall(r"long_only=(True|False)", text)

round_n = rounds[-1] if rounds else None
stag_n = stags[-1] if stags else 0
oos_n = oos_pnls[-1] if oos_pnls else None
long_n = long_only[-1] if long_only else "?"

print(f"round={round_n or '?'}")
print(f"stagnation={stag_n}")
print(f"last_oos_pnl={oos_n if oos_n is not None else '?'}")
print(f"best_oos_log={best_oos_log[-1] if best_oos_log else '?'}")
print(f"long_only={long_n}")
print(f"below_baseline={1 if oos_n is not None and oos_n < baseline else 0}")
PY
)"

eval "$(echo "$jpy_parse" | sed 's/^/jpy_/')"

best_oos_pct="?"
best_oos_round="?"
best_oos_at="?"
if railway volume files --volume hermes-jpy-evolution-volume download \
    /app/state/best_oos_meta.json "$TMP_DIR/jpy_best_oos.json" --overwrite >/dev/null 2>&1; then
  read -r best_oos_pct best_oos_round best_oos_at < <(python3 - "$TMP_DIR/jpy_best_oos.json" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print(data.get("oos_pnl_pct", "?"), data.get("evolution_round", "?"), data.get("updated_at", "?"))
PY
)
fi

prev_best="?"
prev_round="?"
prev_ts=0
if railway volume files --volume hermes-jpy-evolution-volume download \
    /app/state/monitor_checkpoint.json "$TMP_DIR/jpy_checkpoint.json" --overwrite >/dev/null 2>&1; then
  read -r prev_best prev_round prev_ts < <(python3 - "$TMP_DIR/jpy_checkpoint.json" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print(data.get("best_oos_pct", "?"), data.get("round", "?"), int(data.get("ts", 0)))
PY
)
fi

now_ts="$(date +%s)"
python3 - "$TMP_DIR/jpy_checkpoint_out.json" "$now_ts" "$jpy_round" "$best_oos_pct" "$jpy_stagnation" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
payload = {
    "ts": int(sys.argv[2]),
    "round": sys.argv[3],
    "best_oos_pct": sys.argv[4],
    "stagnation": int(sys.argv[5]) if str(sys.argv[5]).isdigit() else 0,
}
out.write_text(json.dumps(payload))
PY
railway volume files --volume hermes-jpy-evolution-volume upload \
  "$TMP_DIR/jpy_checkpoint_out.json" /app/state/monitor_checkpoint.json --overwrite >/dev/null 2>&1 || true

if rg -qi "error|failed|traceback" "$TMP_DIR/jpy_evol.log"; then
  alerts+=("hermes-jpy-evolution: errors in recent logs")
elif ! echo "$jpy_parse" | rg -q "round=[0-9]+" \
  && ! rg -q "ST\+VOL EVOLUTION ROUND|Evolution round:|\[reflect|\[trade\]" "$TMP_DIR/jpy_evol.log"; then
  alerts+=("hermes-jpy-evolution: no evolution activity in recent logs")
fi

notes+=("jpy-evolution round=${jpy_round} stagnation=${jpy_stagnation} last_oos=${jpy_last_oos_pnl}% best_oos=${best_oos_pct}% (baseline ${JPY_BASELINE_OOS}%) long_only=${jpy_long_only}")

if [[ "$jpy_stagnation" =~ ^[0-9]+$ ]] && [[ "$jpy_stagnation" -ge "$STAGNATION_ALERT" ]]; then
  alerts+=("hermes-jpy-evolution: stagnation ${jpy_stagnation} rounds (>= ${STAGNATION_ALERT})")
elif [[ "$jpy_stagnation" =~ ^[0-9]+$ ]] && [[ "$jpy_stagnation" -ge "$STAGNATION_WARN" ]]; then
  notes+=("jpy-evolution WARN: stagnation ${jpy_stagnation} rounds")
fi

if [[ "$best_oos_pct" != "?" ]]; then
  below_baseline="$(python3 - "$best_oos_pct" "$JPY_BASELINE_OOS" <<'PY'
import sys
print(1 if float(sys.argv[1]) < float(sys.argv[2]) else 0)
PY
)"
  if [[ "$below_baseline" == "1" ]]; then
    if [[ "$jpy_stagnation" =~ ^[0-9]+$ ]] && [[ "$jpy_stagnation" -ge "$STAGNATION_WARN" ]]; then
      alerts+=("hermes-jpy-evolution: best_oos ${best_oos_pct}% below jpy_ls01 (${JPY_BASELINE_OOS}%) and stagnating")
    else
      notes+=("jpy-evolution: best_oos ${best_oos_pct}% still below jpy_ls01 (${JPY_BASELINE_OOS}%)")
    fi
  fi
fi

if [[ "$prev_ts" -gt 0 ]] && [[ "$best_oos_pct" != "?" ]] && [[ "$prev_best" != "?" ]]; then
  age=$((now_ts - prev_ts))
  best_oos_stuck="$(python3 - "$best_oos_pct" "$prev_best" "$age" "$CHECKPOINT_MAX_AGE_SEC" <<'PY'
import sys
cur, prev, age, max_age = sys.argv[1:5]
stuck = float(cur) == float(prev) and int(age) >= int(max_age)
print(1 if stuck else 0)
PY
)"
  if [[ "$best_oos_stuck" == "1" ]] \
     && [[ "$jpy_stagnation" =~ ^[0-9]+$ ]] && [[ "$jpy_stagnation" -ge "$STAGNATION_WARN" ]]; then
    alerts+=("hermes-jpy-evolution: no best_oos progress for ${age}s (stuck at ${best_oos_pct}%)")
  fi
fi

echo "=== trading-lab monitor ==="
if [[ ${#alerts[@]} -gt 0 ]]; then
  echo "STATUS: ALERT"
  for a in "${alerts[@]}"; do echo "  - $a"; done
  [[ ${#notes[@]} -gt 0 ]] && echo "Notes:" && for n in "${notes[@]}"; do echo "  - $n"; done
  exit 1
fi

echo "STATUS: OK"
for n in "${notes[@]}"; do echo "  - $n"; done
exit 0
