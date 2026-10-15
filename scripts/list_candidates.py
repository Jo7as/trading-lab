#!/usr/bin/env python3
"""Refresh evolution candidate leaderboard + optional multi-asset canvas."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from trading_lab.candidate_leaderboard import (
    COMBINED_LEADERBOARD,
    default_dashboard_profiles,
    gold_dashboard_profile,
    jpy_dashboard_profile,
    refresh_dashboard,
    refresh_multi_asset_dashboard,
    verify_dashboard_profiles,
)
from trading_lab.env import env


def _default_state_dir() -> Path:
    state = env("LAB_STATE")
    if state:
        return Path(state)
    for candidate in (Path("state"), Path.home() / "trading-lab" / "state"):
        if candidate.exists():
            return candidate
    return Path("state")


def _default_jpy_state_dir() -> Path:
    state = env("LAB_JPY_STATE")
    if state:
        return Path(state)
    return Path("jpy-state")


def _default_canvas_path() -> Path:
    return Path.home() / ".cursor/projects/Users-jonas-trading-lab/canvases/evolution-leaderboard.canvas.tsx"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate leaderboard dashboard")
    parser.add_argument(
        "--asset",
        choices=("all", "gold", "jpy"),
        default="all",
        help="Which asset leaderboard to refresh (default: all)",
    )
    parser.add_argument("--state-dir", type=Path, default=None, help="Gold state dir")
    parser.add_argument("--jpy-state-dir", type=Path, default=None, help="JPY state dir")
    parser.add_argument(
        "--data-source",
        default=env("LAB_DATA_SOURCE", "dukascopy"),
        choices=["dukascopy", "yfinance"],
    )
    parser.add_argument(
        "--start",
        default=os.environ.get("EVOLUTION_START", "2020-01-01"),
    )
    parser.add_argument(
        "--refresh-canvas",
        action="store_true",
        help="Also regenerate evolution-leaderboard.canvas.tsx",
    )
    parser.add_argument("--canvas-path", type=Path, default=None)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print data/asset verification report after refresh",
    )
    args = parser.parse_args()

    gold_dir = args.state_dir or _default_state_dir()
    jpy_dir = args.jpy_state_dir or _default_jpy_state_dir()
    canvas_path = None
    if args.refresh_canvas:
        canvas_path = args.canvas_path or _default_canvas_path()

    if args.asset == "all":
        profiles = [
            gold_dashboard_profile(gold_dir),
            jpy_dashboard_profile(jpy_dir),
        ]
        data = await refresh_multi_asset_dashboard(
            profiles,
            data_source=args.data_source,
            start=args.start,
            combined_json_path=Path(COMBINED_LEADERBOARD),
            canvas_path=canvas_path,
        )
        for section in data["assets"]:
            print(
                f"  {section['title']}: {section['candidates'][0]['label']} "
                f"(OOS {section['candidates'][0]['oos_pnl_pct']:+.1f}%)"
            )
        print(f"Combined leaderboard: {COMBINED_LEADERBOARD}")
        if canvas_path:
            print(f"Canvas updated: {canvas_path}")
        if args.verify:
            reports = await verify_dashboard_profiles(profiles, data_source=args.data_source, start=args.start)
            print("\nVerification:")
            for r in reports:
                ok = "OK" if r["asset_ok"] else "FAIL"
                print(f"  [{ok}] {r['title']}: asset={r['data_asset']} candles={r['candles']}")
                print(f"       baseline {r['baseline']}: OOS {r['baseline_oos_pct']:+.1f}%")
                if r["champion"]:
                    print(f"       champion {r['champion']}: OOS {r['champion_oos_pct']:+.1f}%")
                bo = r.get("best_oos") or {}
                if bo.get("display_version"):
                    print(f"       best_oos: {bo['display_version']} ({bo.get('oos_pnl_pct'):+.2f}%)")
        return

    profile = (
        gold_dashboard_profile(gold_dir)
        if args.asset == "gold"
        else jpy_dashboard_profile(jpy_dir)
    )
    section = await refresh_dashboard(
        profile,
        data_source=args.data_source,
        start=args.start,
    )
    json_path = profile.state_dir / "candidate_leaderboard.json"
    print(f"Leaderboard saved: {json_path}")
    print(f"  Candidates: {len(section['candidates'])}")
    print(
        f"  Top: {section['candidates'][0]['label']} "
        f"(OOS {section['candidates'][0]['oos_pnl_pct']:+.1f}%)"
    )
    if section.get("best_oos"):
        bo = section["best_oos"]
        print(f"  Best OOS: v{bo.get('version')} ({bo.get('oos_pnl_pct'):+.2f}%)")


if __name__ == "__main__":
    asyncio.run(main())
