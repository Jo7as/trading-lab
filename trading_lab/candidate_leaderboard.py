"""Build and persist candidate leaderboard for evolution dashboard."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .adapters.market_data import get_market_adapter
from .exploration import JPY_RESEARCH_SEEDS, RESEARCH_SEEDS
from .strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest

BEST_OOS_STRATEGY = "best_oos.yaml"
BEST_OOS_META = "best_oos_meta.json"
COMBINED_LEADERBOARD = "evolution_leaderboard.json"


@dataclass(frozen=True)
class DashboardAssetProfile:
    """Per-asset dashboard configuration (Gold, JPY, …)."""

    key: str
    title: str
    asset: str
    state_dir: Path
    baseline_path: Path
    baseline_label: str
    signal_id: str
    seeds: tuple[tuple[str, SupertrendVolumeConfig], ...]
    frozen_paths: tuple[tuple[str, Path, str], ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    for candidate in (path, _repo_root() / path, Path("/app") / path.name):
        if candidate.exists():
            return candidate
    return path


def gold_dashboard_profile(state_dir: Path | None = None) -> DashboardAssetProfile:
    root = _repo_root()
    return DashboardAssetProfile(
        key="gold",
        title="Gold (XAU/USD)",
        asset="XAU/USD",
        state_dir=state_dir or root / "state",
        baseline_path=_resolve_path(root / "hf_baseline.yaml"),
        baseline_label="hf_ls01 (signal bot)",
        signal_id="hf_ls01",
        seeds=tuple(RESEARCH_SEEDS),
        frozen_paths=(
            ("gold_v12119 (export)", root / "gold_evolved_v12119.yaml", "champion"),
        ),
    )


def jpy_dashboard_profile(state_dir: Path | None = None) -> DashboardAssetProfile:
    root = _repo_root()
    return DashboardAssetProfile(
        key="jpy",
        title="USD/JPY",
        asset="USD/JPY",
        state_dir=state_dir or root / "jpy-state",
        baseline_path=_resolve_path(root / "jpy_baseline.yaml"),
        baseline_label="jpy_ls01 (baseline)",
        signal_id="jpy_ls01",
        seeds=tuple(JPY_RESEARCH_SEEDS),
        frozen_paths=(
            ("jpy_v48831 (export)", root / "jpy_evolved_v48831.yaml", "champion"),
        ),
    )


def dashboard_profile_for_asset(asset: str, state_dir: Path) -> DashboardAssetProfile | None:
    asset = asset.strip()
    if asset in ("GC=F", "XAU/USD"):
        return gold_dashboard_profile(state_dir)
    if asset in ("USD/JPY", "USDJPY=X"):
        return jpy_dashboard_profile(state_dir)
    return None


def default_dashboard_profiles() -> list[DashboardAssetProfile]:
    return [gold_dashboard_profile(), jpy_dashboard_profile()]


def load_strategy(path: Path) -> SupertrendVolumeConfig | None:
    path = _resolve_path(path)
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    try:
        return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})
    except TypeError:
        return None


def load_best_oos_meta(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / BEST_OOS_META
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def update_best_oos_tracker(
    state_dir: Path,
    strategy: dict[str, Any],
    oos_pnl_pct: float,
    *,
    evolution_round: int | None = None,
) -> bool:
    """Save strategy snapshot when OOS PnL beats the running best."""
    state_dir.mkdir(parents=True, exist_ok=True)
    meta_path = state_dir / BEST_OOS_META
    strat_path = state_dir / BEST_OOS_STRATEGY

    prev = 0.0
    existing = load_best_oos_meta(state_dir)
    if existing is not None:
        prev = float(existing.get("oos_pnl_pct", 0))

    if oos_pnl_pct <= prev:
        return False

    version = str(strategy.get("version", "?"))
    meta = {
        "version": version,
        "oos_pnl_pct": round(float(oos_pnl_pct), 4),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "evolution_round": evolution_round,
    }
    strat_path.write_text(
        yaml.dump(strategy, default_flow_style=False, sort_keys=False)
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


def _chart_label(label: str, tag: str, rank: int) -> str:
    """Short bar-chart label with rank prefix so x-axis never shows blank ticks."""
    aliases = {
        "gold_v12119 (export)": "v12119",
        "jpy_v48831 (export)": "v48831",
        "jpy_st10_m18_ema40_long": "st10_m18",
        "jpy_ls01 (baseline)": "jpy_ls01",
        "jpy_ls01_long": "ls01_long",
        "jpy_ls01_pullback": "ls01_pb",
        "jpy_hf_pullback_long": "hf_pb_L",
        "hf_ls01 (signal bot)": "hf_ls01",
        "hf_st7_m15_noema_longshort": "st7_L+S",
        "hf_st7_m15_noema_long": "st7_long",
        "hf_st7_m18_ema2150_longshort": "st7_m18_LS",
        "hf_st7_m18_ema2150_long": "st7_m18_L",
        "st7_m15_noema": "st7_noema",
        "st7_m15_ema21100": "st7_ema21",
        "st10_m2_longshort": "st10_L+S",
        "st10_m2_ema21100": "st10_ema21",
    }
    if label in aliases:
        short = aliases[label]
    elif tag == "champion":
        base = label.split("(")[0].strip()
        short = base.removeprefix("gold_").removeprefix("jpy_").removeprefix("hf_")[:10]
    elif tag == "baseline":
        short = label.split("(")[0].strip()[:10]
    elif tag == "best_oos":
        short = label.removeprefix("best oos ")[:10]
    else:
        short = label.removeprefix("jpy_").removeprefix("hf_").removeprefix("gold_")[:10]
    return f"#{rank} {short}"[:14]


def _format_best_oos_version(section_key: str, version: str) -> str:
    if version.startswith(("gold_", "jpy_", "hf_")):
        return version
    if section_key == "jpy" and version.isdigit():
        return f"jpy_v{version}"
    if section_key == "gold" and version.isdigit():
        return f"gold_v{version}"
    if section_key == "gold" and version in ("01", "hf_ls01"):
        return "hf_ls01"
    return version


def _metrics_row(
    label: str,
    cfg: SupertrendVolumeConfig,
    full: dict[str, Any],
    oos: dict[str, Any],
    *,
    months: float,
    oos_months: float,
    tag: str = "seed",
) -> dict[str, Any]:
    by_side = full.get("by_side", {})
    long_s = by_side.get("long", {})
    short_s = by_side.get("short", {})
    return {
        "id": label,
        "tag": tag,
        "label": label,
        "long_only": cfg.long_only,
        "st_period": cfg.supertrend_period,
        "st_mult": cfg.supertrend_mult,
        "vol_mult": cfg.volume_mult,
        "sl_atr": cfg.sl_atr_mult,
        "tp1_r": cfg.tp1_r,
        "tp1_close_pct": cfg.tp1_close_pct,
        "max_hold_hours": cfg.max_hold_hours,
        "pullback_entry": cfg.pullback_entry,
        "full_pnl_pct": round(full["total_pnl_pct"], 2),
        "full_trades": full["total_trades"],
        "full_tpw": round(full.get("trades_per_week", 0), 2),
        "full_wr": round(full["win_rate"], 1),
        "full_sharpe": round(full["sharpe"], 2),
        "full_dd_pct": round(full["max_drawdown_pct"], 2),
        "full_pf": round(full["profit_factor"], 2),
        "full_exp_r": round(full["expectancy_r"], 3),
        "pnl_per_month": round(full["total_pnl_pct"] / months, 2) if months else 0,
        "oos_pnl_pct": round(oos["total_pnl_pct"], 2),
        "oos_trades": oos["total_trades"],
        "oos_tpw": round(oos.get("trades_per_week", 0), 2),
        "oos_wr": round(oos["win_rate"], 1),
        "oos_sharpe": round(oos["sharpe"], 2),
        "oos_dd_pct": round(oos["max_drawdown_pct"], 2),
        "oos_pnl_per_month": round(oos["total_pnl_pct"] / oos_months, 2) if oos_months else 0,
        "long_pnl": round(float(long_s.get("pnl_pct", 0)), 2),
        "short_pnl": round(float(short_s.get("pnl_pct", 0)), 2),
        "long_trades": int(long_s.get("trades", 0)),
        "short_trades": int(short_s.get("trades", 0)),
    }


def _score_configs(
    configs: list[tuple[str, SupertrendVolumeConfig, str]],
    *,
    asset: str,
    asset_label: str,
    df,
    df_oos,
    months: float,
    oos_months: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, cfg, tag in configs:
        if cfg is None:
            continue
        cfg = dataclasses.replace(cfg, asset=asset, asset_label=asset_label)
        full = run_backtest(df, cfg, verbose=False)["metrics"]
        oos = run_backtest(df_oos, cfg, verbose=False)["metrics"]
        rows.append(
            _metrics_row(label, cfg, full, oos, months=months, oos_months=oos_months, tag=tag)
        )
    return rows


def _sync_best_oos_meta_measured(
    state_dir: Path,
    *,
    version: str,
    measured_oos_pnl_pct: float,
) -> dict[str, Any] | None:
    """Keep tracker PnL aligned with the current OOS window remeasure."""
    meta = load_best_oos_meta(state_dir)
    if meta is None:
        return None
    if str(meta.get("version", "?")) != version:
        return meta
    synced = {
        **meta,
        "oos_pnl_pct": round(float(measured_oos_pnl_pct), 4),
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    (state_dir / BEST_OOS_META).write_text(json.dumps(synced, indent=2) + "\n")
    return synced


def _best_oos_row_if_champion(
    state_dir: Path,
    *,
    top_oos: float,
    current_version: str | None,
    existing_labels: set[str],
    df,
    df_oos,
    months: float,
    oos_months: float,
) -> dict[str, Any] | None:
    """Add frozen best-oos snapshot only when its *measured* OOS beats #1.

    Always remeasure and sync the tracker to the current OOS window so the
    dashboard card matches the leaderboard (stale meta used to show +55% while
    #1 measured +53.5%). Skip adding a duplicate row when the same version is
    already present as champion/seed/evolved.
    """
    meta = load_best_oos_meta(state_dir)
    cfg = load_strategy(state_dir / BEST_OOS_STRATEGY)
    if meta is None or cfg is None:
        return None

    version = str(meta.get("version", "?"))
    label = f"best oos v{version}"

    full = run_backtest(df, cfg, verbose=False)["metrics"]
    oos = run_backtest(df_oos, cfg, verbose=False)["metrics"]
    measured = float(oos["total_pnl_pct"])
    _sync_best_oos_meta_measured(
        state_dir, version=version, measured_oos_pnl_pct=measured
    )

    if label in existing_labels:
        return None
    if any(version in existing for existing in existing_labels):
        return None
    if current_version is not None and version == current_version:
        evolved_label = f"evolved v{current_version}"
        if evolved_label in existing_labels:
            return None

    if measured <= top_oos:
        return None
    return _metrics_row(label, cfg, full, oos, months=months, oos_months=oos_months, tag="best_oos")


async def build_leaderboard(
    profile: DashboardAssetProfile,
    *,
    data_source: str = "dukascopy",
    start: str = "2020-01-01",
    oos_split: float = 0.30,
    interval: str = "1h",
) -> dict[str, Any]:
    state_dir = profile.state_dir
    adapter = get_market_adapter(data_source)
    start_dt = datetime.fromisoformat(start)
    df = await adapter.fetch_history(profile.asset, interval=interval, start=start_dt)
    if df.empty:
        raise RuntimeError(f"No data from {data_source} for {profile.asset}")

    split = int(len(df) * (1 - oos_split))
    df_oos = df.iloc[split:].copy()
    months = max((df.index[-1] - df.index[0]).days / 30.44, 1)
    oos_months = max(months * oos_split, 1)

    configs: list[tuple[str, SupertrendVolumeConfig, str]] = [
        (profile.baseline_label, load_strategy(profile.baseline_path), "baseline"),
    ]
    strat_path = state_dir / "strategy.yaml"
    evolved = load_strategy(strat_path)
    current_version: str | None = None
    strat_raw: dict[str, Any] | None = None
    if evolved is not None:
        strat_raw = yaml.safe_load(strat_path.read_text()) or {}
        current_version = str(strat_raw.get("version", "?"))
        configs.append((f"evolved v{current_version}", evolved, "evolved"))

    for label, path, tag in profile.frozen_paths:
        frozen = load_strategy(path)
        if frozen is not None:
            configs.append((label, frozen, tag))

    for label, cfg in profile.seeds:
        configs.append((label, cfg, "seed"))

    rows = _score_configs(
        configs,
        asset=profile.asset,
        asset_label="XAU/USD" if profile.asset == "XAU/USD" else "USD/JPY",
        df=df,
        df_oos=df_oos,
        months=months,
        oos_months=oos_months,
    )
    if strat_raw is not None and current_version is not None:
        evolved_row = next(
            (r for r in rows if r["label"] == f"evolved v{current_version}"),
            None,
        )
        if evolved_row is not None:
            update_best_oos_tracker(
                state_dir,
                strat_raw,
                float(evolved_row["oos_pnl_pct"]),
            )
    rows.sort(key=lambda r: r["oos_pnl_pct"], reverse=True)
    top_oos = rows[0]["oos_pnl_pct"] if rows else float("-inf")

    best_row = _best_oos_row_if_champion(
        state_dir,
        top_oos=top_oos,
        current_version=current_version,
        existing_labels={r["label"] for r in rows},
        df=df,
        df_oos=df_oos,
        months=months,
        oos_months=oos_months,
    )
    if best_row is not None:
        rows.append(best_row)

    # Drop transient evolution snapshots — only named seeds/baseline/champion/best_oos
    rows = [r for r in rows if r["tag"] != "evolved"]

    rows.sort(key=lambda r: r["oos_pnl_pct"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        row["chart_label"] = _chart_label(row["label"], row["tag"], i)

    best_meta = load_best_oos_meta(state_dir)
    if best_meta and "version" in best_meta:
        best_meta = {
            **best_meta,
            "display_version": _format_best_oos_version(
                profile.key, str(best_meta["version"])
            ),
        }

    baseline_row = next((r for r in rows if r["tag"] == "baseline"), None)
    return {
        "key": profile.key,
        "title": profile.title,
        "signal_id": profile.signal_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "asset": profile.asset,
        "interval": interval,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "candles": len(df),
        "oos_split": oos_split,
        "oos_start": str(df_oos.index[0]),
        "verified": {
            "asset": profile.asset,
            "baseline_label": profile.baseline_label,
            "baseline_oos_pnl_pct": baseline_row["oos_pnl_pct"] if baseline_row else None,
        },
        "best_oos": best_meta,
        "candidates": rows,
    }


def save_leaderboard(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_leaderboard(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_canvas(data: dict[str, Any], canvas_path: Path) -> None:
    """Regenerate multi-asset canvas with embedded leaderboard JSON."""
    payload = json.dumps(data, indent=2)
    content = '''import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const DASHBOARD = __DATA__ as const;

type AssetSection = (typeof DASHBOARD.assets)[number];
type CandidateRow = AssetSection["candidates"][number];

function fmtPct(v: number) {
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function rowTone(r: CandidateRow, signalId: string) {
  if (r.id.includes(signalId)) return "info" as const;
  if (r.tag === "best_oos" || r.tag === "champion") return "warning" as const;
  if (r.rank <= 3) return "success" as const;
  return undefined;
}

function chartCategory(r: CandidateRow) {
  return `#${r.rank} ${r.label}`;
}

function AssetLeaderboard({ section }: { section: AssetSection }) {
  const rows = [...section.candidates];
  const top = rows[0];
  const signal = rows.find((r) => r.id.includes(section.signal_id));
  const bestOos = section.best_oos as {
    version?: string;
    display_version?: string;
    oos_pnl_pct?: number;
  } | null;
  const chartRows = rows.slice(0, 8);

  return (
    <Stack gap={16}>
      <Stack gap={4}>
        <H2>{section.title}</H2>
        <Text tone="secondary" size="small">
          {section.data_source} · {section.asset} · {section.interval} · OOS from {section.oos_start}
        </Text>
      </Stack>

      <Grid columns={3} gap={12}>
        <Stat label="Top OOS PnL" value={top ? fmtPct(top.oos_pnl_pct) : "—"} />
        <Stat
          label={signal ? `${section.signal_id} · ${signal.oos_tpw}/wk` : section.signal_id}
          value={signal ? fmtPct(signal.oos_pnl_pct) : "—"}
        />
        <Stat
          label={
            bestOos?.display_version || bestOos?.version
              ? `Best OOS ${bestOos.display_version ?? bestOos.version}`
              : "Best OOS"
          }
          value={bestOos?.oos_pnl_pct != null ? fmtPct(bestOos.oos_pnl_pct) : "—"}
        />
      </Grid>

      <Stack gap={8}>
        <Text tone="tertiary" size="small">
          OOS PnL % — top {chartRows.length} configs (30% holdout)
        </Text>
        <BarChart
          horizontal
          categories={chartRows.map(chartCategory)}
          series={[{
            name: "OOS PnL %",
            data: chartRows.map((r) => r.oos_pnl_pct),
            tone: "info",
          }]}
          height={chartRows.length * 44 + 32}
          valueSuffix="%"
          showValues
        />
      </Stack>

      <Table
        striped
        stickyHeader
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right", "center"]}
        headers={[
          "Config",
          "OOS PnL",
          "OOS/wk",
          "Full PnL",
          "$/mo",
          "DD",
          "Sharpe",
          "WR",
          "PF",
          "L/S",
        ]}
        rowTone={rows.map((r) => rowTone(r, section.signal_id))}
        rows={rows.map((r) => [
          `#${r.rank} ${r.label}`,
          fmtPct(r.oos_pnl_pct),
          r.oos_tpw.toFixed(1),
          fmtPct(r.full_pnl_pct),
          fmtPct(r.pnl_per_month),
          `${r.full_dd_pct.toFixed(1)}%`,
          r.oos_sharpe.toFixed(2),
          `${r.oos_wr.toFixed(0)}%`,
          r.full_pf.toFixed(2),
          r.long_only ? "L" : "L+S",
        ])}
      />
    </Stack>
  );
}

export default function EvolutionLeaderboard() {
  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1100 }}>
      <Stack gap={6}>
        <H1>Evolution Leaderboard</H1>
        <Text tone="secondary" size="small">
          Gold + JPY candidates ranked by OOS return · updated {DASHBOARD.generated_at}
        </Text>
      </Stack>

      {DASHBOARD.assets.map((section, idx) => (
        <Stack key={section.key} gap={16}>
          {idx > 0 ? <Divider /> : null}
          <AssetLeaderboard section={section} />
        </Stack>
      ))}

      <Card variant="borderless">
        <CardHeader>Auto-update</CardHeader>
        <CardBody>
          <Text tone="secondary" size="small">
            Each asset stores best_oos.yaml in its state dir (state/ for Gold, jpy-state/ for JPY).
            Evolution refreshes leaderboard JSON every 5 rounds on Railway.
            Regenerate: uv run python list_candidates.py --refresh-canvas
          </Text>
        </CardBody>
      </Card>
    </Stack>
  );
}
'''.replace("__DATA__", payload)
    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_path.write_text(content)


async def refresh_dashboard(
    profile: DashboardAssetProfile,
    *,
    data_source: str = "dukascopy",
    start: str = "2020-01-01",
    json_path: Path | None = None,
) -> dict[str, Any]:
    data = await build_leaderboard(
        profile,
        data_source=data_source,
        start=start,
    )
    json_out = json_path or profile.state_dir / "candidate_leaderboard.json"
    save_leaderboard(data, json_out)
    return data


async def verify_dashboard_profiles(
    profiles: list[DashboardAssetProfile] | None = None,
    *,
    data_source: str = "dukascopy",
    start: str = "2020-01-01",
) -> list[dict[str, Any]]:
    """Cross-check each profile uses the correct market data and baseline metrics."""
    reports: list[dict[str, Any]] = []
    for profile in profiles or default_dashboard_profiles():
        section = await build_leaderboard(profile, data_source=data_source, start=start)
        baseline = next((r for r in section["candidates"] if r["tag"] == "baseline"), None)
        champion = next((r for r in section["candidates"] if r["tag"] == "champion"), None)
        reports.append(
            {
                "key": profile.key,
                "title": profile.title,
                "data_asset": section["asset"],
                "expected_asset": profile.asset,
                "asset_ok": section["asset"] == profile.asset,
                "candles": section["candles"],
                "oos_start": section.get("oos_start"),
                "baseline": baseline["label"] if baseline else None,
                "baseline_oos_pct": baseline["oos_pnl_pct"] if baseline else None,
                "champion": champion["label"] if champion else None,
                "champion_oos_pct": champion["oos_pnl_pct"] if champion else None,
                "top": section["candidates"][0]["label"] if section["candidates"] else None,
                "best_oos": section.get("best_oos"),
            }
        )
    return reports


async def refresh_multi_asset_dashboard(
    profiles: list[DashboardAssetProfile] | None = None,
    *,
    data_source: str = "dukascopy",
    start: str = "2020-01-01",
    combined_json_path: Path | None = None,
    canvas_path: Path | None = None,
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for profile in profiles or default_dashboard_profiles():
        section = await refresh_dashboard(
            profile,
            data_source=data_source,
            start=start,
        )
        sections.append(section)

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "assets": sections,
    }
    out = combined_json_path or _repo_root() / COMBINED_LEADERBOARD
    save_leaderboard(combined, out)
    if canvas_path is not None:
        write_canvas(combined, canvas_path)
    return combined
