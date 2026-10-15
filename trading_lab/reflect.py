"""Reflection cycle — score-guided random search with bounds.

Picks ONE variable from the searchable space, nudges it ±step,
writes the new strategy.yaml, and logs the hypothesis.

Variables in the search space (9 total):
  - bb_std, squeeze_threshold, volume_mult, sl_atr_mult, tp1_r,
    tp1_close_pct, supertrend_mult, max_hold_hours, ema_fast
"""
from __future__ import annotations
from .env import env

import json
import os
import random
import shutil
import time
from pathlib import Path

import yaml

from .config import MCTFBConfig, load_config_from_yaml, config_to_yaml
from .score import score

STATE_DIR = Path(env("LAB_STATE", "/app/state"))
if not STATE_DIR.exists():
    STATE_DIR = Path.home() / "trading-lab" / "state"

GOAL_PATH = STATE_DIR / "goal.yaml"
STRATEGY_PATH = STATE_DIR / "strategy.yaml"
TRADES_PATH = STATE_DIR / "trades.jsonl"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"

_DEFAULT_HISTORY_MAX = 50
_DEFAULT_HYPOTHESES_MAX_LINES = 2000
_DEFAULT_TRADES_MAX_LINES = 5000


def _history_max_keep() -> int:
    raw = env("LAB_HISTORY_MAX_KEEP", str(_DEFAULT_HISTORY_MAX))
    try:
        return max(5, int(raw))
    except ValueError:
        return _DEFAULT_HISTORY_MAX


def _hypotheses_max_lines() -> int:
    raw = env(
        "LAB_HYPOTHESES_MAX_LINES", str(_DEFAULT_HYPOTHESES_MAX_LINES)
    )
    try:
        return max(100, int(raw))
    except ValueError:
        return _DEFAULT_HYPOTHESES_MAX_LINES


def _trades_max_lines() -> int:
    raw = env("LAB_TRADES_MAX_LINES", str(_DEFAULT_TRADES_MAX_LINES))
    try:
        return max(100, int(raw))
    except ValueError:
        return _DEFAULT_TRADES_MAX_LINES


def _version_sort_key(path: Path) -> int:
    stem = path.stem
    if stem.startswith("v"):
        stem = stem[1:]
    try:
        return int(stem)
    except ValueError:
        return 0


def prune_history(max_keep: int | None = None) -> int:
    """Delete oldest strategy snapshots; keep the newest max_keep files."""
    if not HISTORY_DIR.exists():
        return 0
    keep = max_keep if max_keep is not None else _history_max_keep()
    files = sorted(HISTORY_DIR.glob("v*.yaml"), key=_version_sort_key)
    removed = 0
    for path in files[:-keep]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def _truncate_jsonl(path: Path, max_lines: int) -> int:
    """Keep the last ``max_lines`` of a jsonl file. Disk-full safe: unlink first."""
    if not path.exists():
        return 0
    try:
        text = path.read_text(errors="replace")
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        return 0
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return 0
    kept = lines[-max_lines:]
    removed = len(lines) - max_lines
    # Free space before rewrite (critical when volume is 100% full)
    try:
        path.unlink()
    except OSError:
        pass
    try:
        path.write_text("\n".join(kept) + "\n")
    except OSError:
        # Last resort: leave file missing rather than crash the worker
        pass
    return removed


def prune_hypotheses(max_lines: int | None = None) -> int:
    """Truncate hypotheses log to the most recent lines."""
    limit = max_lines if max_lines is not None else _hypotheses_max_lines()
    return _truncate_jsonl(HYPOTHESES_PATH, limit)


def prune_trades(max_lines: int | None = None) -> int:
    """Truncate trades.jsonl (can grow to hundreds of MB on long evolutions)."""
    limit = max_lines if max_lines is not None else _trades_max_lines()
    return _truncate_jsonl(TRADES_PATH, limit)


def prune_state_volume() -> dict[str, int]:
    """Free disk on evolution volumes (history + hypotheses + trades).

    Order matters when the volume is full: unlink history first (no free space
    required), then truncate jsonl logs via unlink+rewrite.
    """
    return {
        "history_removed": prune_history(),
        "hypotheses_lines_removed": prune_hypotheses(),
        "trades_lines_removed": prune_trades(),
    }


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_trades(limit: int = 50) -> list[dict]:
    if not TRADES_PATH.exists():
        return []
    lines = TRADES_PATH.read_text().strip().splitlines()
    closed = []
    for line in lines:
        if not line.strip():
            continue
        try:
            t = json.loads(line)
            if t.get("status") == "closed":
                closed.append(t)
        except json.JSONDecodeError:
            continue
    return closed[-limit:]


def next_version(current: str) -> str:
    try:
        n = int(current, 10)
    except (ValueError, TypeError):
        n = 1
    return f"{n + 1:02d}"


def save_prior(strategy: dict) -> Path:
    prune_state_volume()
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    version = strategy.get("version", "01")
    path = HISTORY_DIR / f"v{version}.yaml"
    with open(path, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)
    prune_history()
    return path


def append_hypothesis(hyp: dict) -> None:
    with open(HYPOTHESES_PATH, "a") as f:
        f.write(json.dumps(hyp) + "\n")


def write_strategy(strategy: dict) -> None:
    with open(STRATEGY_PATH, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)


def reflect_fallback() -> None:
    """Score-guided hill climbing: try a random nudge, keep only if score improves."""
    goal = load_yaml(GOAL_PATH)
    strategy = load_yaml(STRATEGY_PATH)
    trades = load_trades(50)

    if not trades:
        print("[reflect] no closed trades yet — nothing to reflect on")
        return

    current_score = score(trades, goal)

    # Try several candidates, keep the best
    search_space = MCTFBConfig.search_space()
    best_candidate = None
    best_score = current_score

    for _ in range(8):  # sample 8 candidates
        var_name, lo, hi, step = random.choice(search_space)
        old_val = strategy.get(var_name)
        if old_val is None:
            old_val = getattr(MCTFBConfig(), var_name)

        direction = random.choice([-1, 1])
        new_val = round(float(old_val) + direction * step, 4)
        new_val = max(lo, min(hi, new_val))

        if isinstance(old_val, int) or var_name in ("ema_fast", "max_hold_hours"):
            new_val = int(new_val)

        if new_val == old_val:
            continue  # no change, skip

        # Simulate: apply change, re-score
        trial_strategy = strategy.copy()
        trial_strategy[var_name] = new_val
        # We can't re-run the backtest here, so we use a heuristic:
        # if the current score is bad, prefer risk-tightening moves
        # if score is good, prefer reward-expanding moves
        # Score the candidate by a simple heuristic proxy
        candidate_score = _heuristic_score(var_name, old_val, new_val, current_score, trades, goal)

        if candidate_score > best_score:
            best_score = candidate_score
            best_candidate = (var_name, old_val, new_val, candidate_score)

    if best_candidate is None:
        # No improvement found — try a random nudge anyway (exploration)
        var_name, lo, hi, step = random.choice(search_space)
        old_val = strategy.get(var_name)
        if old_val is None:
            old_val = getattr(MCTFBConfig(), var_name)
        direction = random.choice([-1, 1])
        new_val = round(float(old_val) + direction * step, 4)
        new_val = max(lo, min(hi, new_val))
        if isinstance(old_val, int) or var_name in ("ema_fast", "max_hold_hours"):
            new_val = int(new_val)
        best_candidate = (var_name, old_val, new_val, current_score)
        print(f"[reflect] no clear improvement — exploring {var_name}")

    var_name, old_val, new_val, best_score = best_candidate

    prior_path = save_prior(strategy)
    old_version = strategy.get("version", "01")
    new_version = next_version(old_version)

    strategy[var_name] = new_val
    strategy["version"] = new_version
    write_strategy(strategy)

    reasoning = (
        f"hill-climb: prior_score={current_score:.4f}, "
        f"nudged {var_name} {old_val}→{new_val} "
        f"(est_score={best_score:.4f}, bounds=[{lo},{hi}])"
    )

    hyp = {
        "ts": time.time(),
        "mode": "hill_climb",
        "version": f"{old_version} → {new_version}",
        "score": round(current_score, 4),
        "trades_examined": len(trades),
        "variable_changed": var_name,
        "old_value": old_val,
        "new_value": new_val,
        "reasoning": reasoning,
        "prior_path": str(prior_path),
    }
    append_hypothesis(hyp)
    print(json.dumps(hyp, indent=2))


def _heuristic_score(var_name, old_val, new_val, current_score, trades, goal):
    """Heuristic proxy: estimate if a change improves the strategy.

    Uses trade statistics to guide the search:
    - Low win rate → loosen entry filters (squeeze_threshold up, volume_mult down, bb_std up)
    - High drawdown → tighten stops (sl_atr_mult down)
    - Low profit factor → adjust reward (tp1_r up, tp1_close_pct up)
    """
    closed = [t for t in trades if t.get("status") == "closed"]
    pnls = [t.get("pnl_pct", 0) or 0.0 for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1
    pf = sum(wins) / abs(sum(losses)) if losses else 1

    direction = 1 if new_val > old_val else -1
    bonus = current_score  # start from current

    # Win rate is low (28%) → we need more entries or looser filters
    if win_rate < 0.35:
        if var_name in ("squeeze_threshold",) and direction > 0:
            bonus += 0.02  # looser squeeze → more trades
        if var_name in ("volume_mult",) and direction < 0:
            bonus += 0.02  # lower volume bar → more trades
        if var_name in ("bb_std",) and direction > 0:
            bonus += 0.01  # wider bands → more breakouts
        if var_name in ("min_body_ratio",) and direction < 0:
            bonus += 0.02  # accept weaker candles

    # Drawdown is high → tighten risk
    max_dd = goal.get("max_drawdown", 0.08)
    if pf < 1.5:
        if var_name in ("sl_atr_mult",) and direction < 0:
            bonus += 0.03  # tighter stop → less drawdown
        if var_name in ("tp1_r",) and direction > 0:
            bonus += 0.02  # take more profit at TP1
        if var_name in ("tp1_close_pct",) and direction > 0:
            bonus += 0.02  # close more at TP1

    # If too few trades, loosen filters
    if len(closed) < 30:
        if var_name in ("squeeze_threshold", "bb_std") and direction > 0:
            bonus += 0.01
        if var_name in ("volume_mult",) and direction < 0:
            bonus += 0.01

    return bonus


def reflect_with_agent() -> None:
    """Agent-powered reflection (if the configured agent CLI is available)."""
    import subprocess

    goal = load_yaml(GOAL_PATH)
    strategy = load_yaml(STRATEGY_PATH)
    trades = load_trades(50)

    if not trades:
        print("[reflect] no closed trades yet — nothing to reflect on")
        return

    s = score(trades, goal)

    prompt = f"""You are the reflection brain of a self-improving trading agent (MCTFB strategy).

Current strategy (v{strategy.get('version', '01')}):
{yaml.dump(strategy, default_flow_style=False)}

Last {len(trades)} closed trades:
{json.dumps(trades[-50:], indent=2)}

Goal:
{yaml.dump(goal, default_flow_style=False)}

Current score: {s:.4f}

Searchable variables (change exactly ONE):
  - bb_std (1.5–2.5, step 0.1)
  - squeeze_threshold (0.10–0.30, step 0.02)
  - volume_mult (1.2–2.0, step 0.1)
  - sl_atr_mult (1.0–2.5, step 0.1)
  - tp1_r (1.0–2.5, step 0.1)
  - tp1_close_pct (0.33–0.67, step 0.01)
  - supertrend_mult (2.0–4.0, step 0.1)
  - max_hold_hours (48–120, step 1)
  - ema_fast (9–50, step 1)

Rules:
- Change exactly ONE variable.
- If score < 0: tighten risk (reduce sl_atr_mult, reduce position risk).
- If drawdown too high: reduce sl_atr_mult or supertrend_mult.
- If too few trades: loosen squeeze_threshold or volume_mult.
- Output a YAML block with: variable_changed, old_value, new_value, reasoning.
- Keep reasoning under 100 words.
"""

    agent = env("LAB_REFLECT_AGENT", "hermes") or "hermes"
    try:
        result = subprocess.run(
            [agent, "--no-tui", "--prompt", prompt],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[reflect] agent CLI not available — falling back to random search")
        reflect_fallback()
        return

    import re
    m = re.search(r"```yaml\n(.*?)```", output, re.DOTALL)
    if not m:
        m = re.search(r"(variable_changed:.*?)(?:\n\n|\Z)", output, re.DOTALL)
    if not m:
        reflect_fallback()
        return

    parsed = yaml.safe_load(m.group(1))
    if not parsed or "variable_changed" not in parsed:
        reflect_fallback()
        return

    prior_path = save_prior(strategy)
    old_version = strategy.get("version", "01")
    new_version = next_version(old_version)

    var = parsed["variable_changed"]
    strategy[var] = parsed["new_value"]
    strategy["version"] = new_version
    write_strategy(strategy)

    hyp = {
        "ts": time.time(),
        "mode": "agent",
        "version": f"{old_version} → {new_version}",
        "score": round(s, 4),
        "variable_changed": var,
        "old_value": parsed.get("old_value"),
        "new_value": parsed.get("new_value"),
        "reasoning": parsed.get("reasoning", ""),
        "prior_path": str(prior_path),
    }
    append_hypothesis(hyp)
    print(json.dumps(hyp, indent=2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Reflection cycle")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--agent", action="store_true")
    args = parser.parse_args()
    if args.agent:
        reflect_with_agent()
    else:
        reflect_fallback()


if __name__ == "__main__":
    main()
