"""H10: do volume-profile levels on crypto perps attract price (MAGNET), stop it (HOLD), or
interact with SMC levels / taker flow - against random levels? Measurement only, no strategy.

Pre-registration: ``docs/specs/h10_volume_profile_PREREG.md`` (windows, levels, tests,
controls, thresholds and the verdict rule are fixed there). Library: ``trading_lab/volume_levels.py``.

    uv run python studies/crypto_volume_profile_study.py --year 2024      # discovery
    uv run python studies/crypto_volume_profile_study.py --year 2023      # confirmation
    uv run python studies/crypto_volume_profile_study.py --verdict        # both years -> verdict

Binance USD-M flow and prices, cached 10-s bars only (never downloads). 2025+ is refused.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_lab import costs  # noqa: E402
from trading_lab import level_flow as lf  # noqa: E402
from trading_lab import market_structure as ms  # noqa: E402
from trading_lab import orderflow as of  # noqa: E402
from trading_lab import volume_levels as vl  # noqa: E402
from trading_lab.adapters import crypto_trades as ct  # noqa: E402
from trading_lab.indicators import atr as atr_fn  # noqa: E402

MARKETS = ("BTC", "SOL")
VENUE = "binance_um"
HORIZONS = {"4h": "4h", "12h": "12h", "24h": "24h"}
PRIMARY = "24h"
WINDOWS = ["n96", "n288", "n672", "day", "week", "asia", "london", "ny", "leg"]
TYPES = ["poc", "vah", "val", "hvn", "lvn"]
MAGNET_CELLS = [f"{w}:{t}" for w in WINDOWS for t in TYPES] + ["naked:poc"]
HOLD_CELLS = [f"{w}:{t}" for w in WINDOWS if w != "day" for t in TYPES] + ["naked:poc"]
OUT = ct.CACHE.parent.parent / "crypto"


def fill_quiet_minutes(m1: pd.DataFrame) -> pd.DataFrame:
    """A minute without trades (no missing day - those abort the run) keeps the last price, so
    it is not mistaken for a data gap; its volume stays zero."""
    m1 = m1.copy()
    c = m1["close"].ffill()
    for col in ("open", "high", "low", "close"):
        m1[col] = m1[col].fillna(c)
    return m1


def all_levels(m1: pd.DataFrame, m15: pd.DataFrame, res: ms.StructureResult) -> list[lf.Level]:
    wins = (vl.fixed_windows(m1, 96) + vl.fixed_windows(m1, 288) + vl.fixed_windows(m1, 672)
            + vl.period_windows(m1, "day") + vl.period_windows(m1, "week")
            + vl.session_windows(m1, "asia") + vl.session_windows(m1, "london")
            + vl.session_windows(m1, "ny") + vl.leg_windows(m15, res))
    return vl.window_levels(m1, wins) + vl.naked_pocs(m1)


def run_year(year: int) -> None:
    if year >= 2025:
        raise SystemExit("2025+ is the holdout - not opened here")
    start, end = date(year, 1, 1), date(year, 12, 31)
    mag, hr, hc = [], [], []
    for market in MARKETS:
        bars = ct.load_bars(VENUE, market, start, end, allow_download=False)
        miss = bars.attrs.get("missing_days", [])
        if miss:
            raise SystemExit(f"{market} {year}: {len(miss)} days not cached (first {miss[0]}) - "
                             "wait for the download, nothing is fetched here")
        m1 = fill_quiet_minutes(of.resample(bars, "1min"))
        del bars
        m15 = of.resample(m1, "15min").dropna(subset=["close"])
        res = ms.analyze(m15)
        a15 = atr_fn(m15["high"], m15["low"], m15["close"], 14)
        path = vl.Path(m1, vl.atr_1h(m1))
        levels = all_levels(m1, m15, res)
        ref = vl.smc_reference(m15, res)
        print(f"{market} {year}: {len(levels)} levels", flush=True)
        ev = vl.magnet_events(path, levels, HORIZONS, seed=year)
        hold_lv = [lv for lv in levels if not lv.kind.startswith("day:")]
        real = vl.hold_events(path, hold_lv)
        ctl = pd.concat([vl.hold_events(path, lf.control_levels(hold_lv, seed=s)).assign(seed=s)
                         for s in (1, 2, 3)], ignore_index=True)
        for df in (real, ctl):
            df["confluent"] = vl.smc_confluence(df, m15, ref, a15) if len(df) else []
        for df in (ev, real, ctl):
            df["market"] = market
        mag.append(ev)
        hr.append(real)
        hc.append(ctl)
        print(f"  magnet {len(ev)}, hold real {len(real)}, hold control {len(ctl)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(mag, ignore_index=True).to_pickle(OUT / f"vp_{year}_magnet.pkl")
    pd.concat(hr, ignore_index=True).to_pickle(OUT / f"vp_{year}_hold.pkl")
    pd.concat(hc, ignore_index=True).to_pickle(OUT / f"vp_{year}_hold_ctl.pkl")
    text = report(year)
    (OUT / f"vp_{year}.md").write_text(text)
    print(text)
    print(f"\n-> {OUT / f'vp_{year}.md'}")


# ── tables ──────────────────────────────────────────────────────────────
def load(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_pickle(OUT / f"vp_{year}_magnet.pkl"), pd.read_pickle(OUT / f"vp_{year}_hold.pkl"),
            pd.read_pickle(OUT / f"vp_{year}_hold_ctl.pkl"))


def magnet_cell(ev: pd.DataFrame, cell: str, h: str = PRIMARY) -> dict:
    g = ev[ev["kind"] == cell]
    st = vl.cell_stats(g[f"eff_{h}"], g["ts"])
    st.update(hit=g[f"hit_{h}"].mean(), ctl=g[f"ctl_{h}"].mean(), mir=g[f"mir_{h}"].mean(),
              dist=g["dist_atr"].median() if len(g) else np.nan)
    return st


def hold_cell(real: pd.DataFrame, ctl: pd.DataFrame, cell: str) -> dict:
    r = real[(real["kind"] == cell) & real["hold"].notna()]
    c = ctl[(ctl["kind"] == cell) & ctl["hold"].notna()]
    return vl.diff_stats(r["hold"], r["ts"], c["hold"], c["ts"])


def _type(kind: pd.Series) -> pd.Series:
    return kind.str.split(":").str[1]


def interaction_cells(real: pd.DataFrame, ctl: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for t in TYPES:
        for name, flag in (("confluence", lambda d: d["confluent"]),
                           ("flow", lambda d: d["d"] * d["flow"] >= vl.FLOW_THR)):
            r = real[(_type(real["kind"]) == t) & real["hold"].notna()]
            c = ctl[(_type(ctl["kind"]) == t) & ctl["hold"].notna()]
            fr, fc = flag(r).astype(bool), flag(c).astype(bool)
            st = vl.diff_stats(r.loc[fr, "hold"], r.loc[fr, "ts"], r.loc[~fr, "hold"], r.loc[~fr, "ts"])
            sc = vl.diff_stats(c.loc[fc, "hold"], c.loc[fc, "ts"], c.loc[~fc, "hold"], c.loc[~fc, "ts"])
            st["ctl_diff"], st["share"] = sc["diff"], float(fr.mean()) if len(fr) else np.nan
            out[f"{t}:{name}"] = st
    return out


def f(x, fmt="{:+.3f}"):
    return "-" if x is None or not np.isfinite(x) else fmt.format(x)


def report(year: int) -> str:
    ev, real, ctl = load(year)
    cost = {m: costs.crypto_round_trip_bp(m, "bybit_perp") for m in MARKETS}
    L = [f"# H10 volume-profile levels, {year} (Binance perps BTC + SOL, 1-min bars)", "",
         "Pre-registration: docs/specs/h10_volume_profile_PREREG.md. t = smaller |t| of the "
         "day-clustered and the Newey-West (5 lags, per-day) t.", "",
         f"## (a) MAGNET, horizon {PRIMARY} (primary) - both markets", "",
         "| cell | n | hit real | hit control (same dist, other time) | hit mirror | effect | t | "
         "effect BTC | effect SOL | eff 4h | eff 12h | median dist ATR1h |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in MAGNET_CELLS:
        s = magnet_cell(ev, cell)
        bm = {m: magnet_cell(ev[ev["market"] == m], cell)["mean"] for m in MARKETS}
        L.append(f"| {cell} | {s['n']} | {f(s['hit'], '{:.1%}')} | {f(s['ctl'], '{:.1%}')} | "
                 f"{f(s['mir'], '{:.1%}')} | {f(s['mean'])} | {f(s['t'], '{:+.2f}')} | {f(bm['BTC'])} | "
                 f"{f(bm['SOL'])} | {f(magnet_cell(ev, cell, '4h')['mean'])} | "
                 f"{f(magnet_cell(ev, cell, '12h')['mean'])} | {f(s['dist'], '{:.2f}')} |")
    L += ["", "## (b) HOLD (bounce 0.5 ATR before break 0.5 ATR, 4 h) - both markets", "",
          "| cell | n real | hold real | hold control | diff | t | diff BTC | diff SOL | "
          "fade 4h gross bp | fade net bp | unresolved |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    avg_cost = float(np.mean(list(cost.values())))
    for cell in HOLD_CELLS:
        s = hold_cell(real, ctl, cell)
        bm = {m: hold_cell(real[real["market"] == m], ctl[ctl["market"] == m], cell)["diff"] for m in MARKETS}
        g = real[real["kind"] == cell]
        fade = g["fade_bp"].mean() if len(g) else np.nan
        L.append(f"| {cell} | {s['n_a']} | {f(s['a'], '{:.1%}')} | {f(s['b'], '{:.1%}')} | {f(s['diff'])} | "
                 f"{f(s['t'], '{:+.2f}')} | {f(bm['BTC'])} | {f(bm['SOL'])} | {f(fade, '{:+.1f}')} | "
                 f"{f(fade - avg_cost, '{:+.1f}')} | {f((g['outcome'] == 'unresolved').mean(), '{:.0%}')} |")
    L += ["", "## (c) Interactions on HOLD events (windows pooled, per level type)", "",
          "hold(flag) - hold(no flag); control = the same split on the shifted control levels.", "",
          "| cell | share flagged | hold flagged | hold not | diff | t | control diff |",
          "|---|---|---|---|---|---|---|"]
    for cell, s in interaction_cells(real, ctl).items():
        L.append(f"| {cell} | {f(s['share'], '{:.0%}')} | {f(s['a'], '{:.1%}')} | {f(s['b'], '{:.1%}')} | "
                 f"{f(s['diff'])} | {f(s['t'], '{:+.2f}')} | {f(s['ctl_diff'])} |")
    # practitioner claim: prior POC reached
    L += ["", "## Base rates the practitioner claim refers to (POC reached within h)", "",
          "| cell | 4h | 12h | 24h | control 24h |", "|---|---|---|---|---|"]
    for cell in ["day:poc", "week:poc", "n96:poc", "naked:poc"]:
        g = ev[ev["kind"] == cell]
        L.append(f"| {cell} | " + " | ".join(f(g[f'hit_{h}'].mean(), '{:.1%}') for h in HORIZONS)
                 + f" | {f(g[f'ctl_{PRIMARY}'].mean(), '{:.1%}')} |")
    L += ["", f"Events: magnet {len(ev)}, hold real {len(real)} (resolved {real['hold'].notna().sum()}), "
          f"hold control {len(ctl)} (3 seeds). Costs (fade only): "
          + ", ".join(f"{m} {c:.1f} bp" for m, c in cost.items())]
    return "\n".join(L)


def verdict() -> str:
    d_ev, d_r, d_c = load(2024)
    c_ev, c_r, c_c = load(2023)
    L = ["# H10 verdict (2024 discovery, 2023 confirmation)", "",
         "| test | cell | 2024 effect | 2024 t | 2023 effect | 2023 t | 2023 BTC | 2023 SOL | confirmed |",
         "|---|---|---|---|---|---|---|---|---|"]
    n_cells = n_conf = 0
    for cell in MAGNET_CELLS:
        a, b = magnet_cell(d_ev, cell), magnet_cell(c_ev, cell)
        bm = {m: magnet_cell(c_ev[c_ev["market"] == m], cell)["mean"] for m in MARKETS}
        ok = vl.cell_verdict(a, b, bm)
        n_cells, n_conf = n_cells + 1, n_conf + ok
        if ok or abs(a["t"] if np.isfinite(a["t"]) else 0) >= vl.T_DISCOVERY:
            L.append(f"| magnet | {cell} | {f(a['mean'])} | {f(a['t'], '{:+.2f}')} | {f(b['mean'])} | "
                     f"{f(b['t'], '{:+.2f}')} | {f(bm['BTC'])} | {f(bm['SOL'])} | {'YES' if ok else 'no'} |")
    for cell in HOLD_CELLS:
        a, b = hold_cell(d_r, d_c, cell), hold_cell(c_r, c_c, cell)
        bm = {m: hold_cell(c_r[c_r["market"] == m], c_c[c_c["market"] == m], cell)["diff"] for m in MARKETS}
        ok = vl.cell_verdict(a, b, bm, effect="diff")
        n_cells, n_conf = n_cells + 1, n_conf + ok
        if ok or abs(a["t"] if np.isfinite(a["t"]) else 0) >= vl.T_DISCOVERY:
            L.append(f"| hold | {cell} | {f(a['diff'])} | {f(a['t'], '{:+.2f}')} | {f(b['diff'])} | "
                     f"{f(b['t'], '{:+.2f}')} | {f(bm['BTC'])} | {f(bm['SOL'])} | {'YES' if ok else 'no'} |")
    ia, ib = interaction_cells(d_r, d_c), interaction_cells(c_r, c_c)
    ibm = {m: interaction_cells(c_r[c_r["market"] == m], c_c[c_c["market"] == m]) for m in MARKETS}
    n_int = 0
    for cell in ia:
        ok = vl.cell_verdict(ia[cell], ib[cell], {m: ibm[m][cell]["diff"] for m in MARKETS}, effect="diff")
        n_int += ok
        if ok or abs(ia[cell]["t"] if np.isfinite(ia[cell]["t"]) else 0) >= vl.T_DISCOVERY:
            L.append(f"| interaction | {cell} | {f(ia[cell]['diff'])} | {f(ia[cell]['t'], '{:+.2f}')} | "
                     f"{f(ib[cell]['diff'])} | {f(ib[cell]['t'], '{:+.2f}')} | "
                     f"{f(ibm['BTC'][cell]['diff'])} | {f(ibm['SOL'][cell]['diff'])} | {'YES' if ok else 'no'} |")
    L += ["", "(rows shown: every cell confirmed or with |t| >= 3 on 2024)", "",
          f"Deciding cells: {n_cells} magnet/hold + {len(ia)} interaction. Confirmed: {n_conf} magnet/hold, "
          f"{n_int} interaction.",
          "Verdict: " + ("volume-profile levels matter (confirmed cells above are spec candidates)"
                         if n_conf else "no better than random levels")]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int)
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild the year's report from saved events")
    a = ap.parse_args()
    if a.verdict:
        text = verdict()
        (OUT / "vp_verdict.md").write_text(text)
        print(text)
    elif a.year and a.report_only:
        print(report(a.year))
    elif a.year:
        run_year(a.year)
    else:
        ap.error("--year or --verdict")


if __name__ == "__main__":
    main()
