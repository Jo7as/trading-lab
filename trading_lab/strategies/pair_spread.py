"""Pair spread (statistical arbitrage) strategy — dollar-neutral reversion.

Long 1 $ of leg A / short 1 $ of leg B (or the reverse) when the log-ratio
``log(A) - log(B)`` deviates from its rolling mean by ``z_entry`` standard
deviations; exit when it comes back (``z_exit``), runs away (``z_stop``) or after
``max_hold_bars``.

Costs are modelled explicitly because that is where every other candidate died:
the long leg buys the ask / sells the bid, the short leg does the mirror image —
i.e. a full spread per leg per round trip, from the actual bid/ask of each leg.

Contract (shared): ``run_backtest(frames, cfg) -> {"trades","metrics","equity_curve","signals"}``
where ``frames`` is ``{asset: DataFrame}`` (bid OHLC + ``ask_*``).

**REJECTED — do not use as a candidate.** Measured on US100/SP500,
US30/SP500, GER40/UK100 and XAUUSD/XAGUSD (2019–2026, 15m): the gross (mid-price)
expectancy is ≈ 0 (profit factor 0.97–1.06) and with real spreads it is −141 % to
−691 %. The earlier "2σ_deviation/cost = 8…54" is the *wrong* quantity — σ of the
deviation is not the realized reversion. See
``docs/strategies/pair_spreads_REJECTED.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairSpreadConfig:
    lookback: int = 60              # bars for the rolling mean / std of the log ratio
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5
    max_hold_bars: int = 24         # 15m bars → ~6 h (intraday)
    notional: float = 1.0           # $ per leg as a fraction of the account
    risk_per_trade_pct: float = 0.5  # only for the reported R; sizing is dollar-neutral
    timeframe: str = "15min"
    flat_time: str | None = None    # e.g. "20:00" bar clock (UTC) → force intraday flat
    warmup: int = 80
    mid_prices: bool = False        # True = ignore the spread (gross variant, for the cost delta)
    random_entry: bool = False      # True = random side at random qualifying bars (baseline)
    seed: int = 7


def _merge(frames: dict[str, pd.DataFrame], a: str, b: str, tf: str) -> pd.DataFrame:
    out = {}
    for tag, name in (("a", a), ("b", b)):
        f = frames[name]
        r = f.resample(tf).agg({"close": "last", "ask_close": "last"}).dropna()
        out[tag] = r
    j = out["a"].join(out["b"], how="inner", lsuffix="_a", rsuffix="_b").dropna()
    return j


def run_backtest(frames: dict[str, pd.DataFrame], cfg: PairSpreadConfig, *,
                 pair: tuple[str, str], verbose: bool = False) -> dict:
    a, b = pair
    j = _merge(frames, a, b, cfg.timeframe)
    if len(j) < cfg.warmup + 10:
        return {"trades": [], "metrics": {}, "equity_curve": [1.0], "signals": []}

    close_a, close_b = j["close_a"].to_numpy(float), j["close_b"].to_numpy(float)
    ask_a, ask_b = j["ask_close_a"].to_numpy(float), j["ask_close_b"].to_numpy(float)
    ts = j.index

    lr = np.log(close_a) - np.log(close_b)
    s = pd.Series(lr)
    mu = s.rolling(cfg.lookback).mean().to_numpy()
    sd = s.rolling(cfg.lookback).std().to_numpy()
    z = (s - s.rolling(cfg.lookback).mean()).to_numpy() / sd

    trades: list[dict] = []
    signals: list[dict] = []
    pnl_stream: list[float] = []
    pos: dict | None = None
    n = len(j)

    def _entry(k: int, side: int) -> tuple[float, float]:
        """side=+1 → long A at ask / short B at bid; side=−1 → short A at bid / long B at ask."""
        if cfg.mid_prices:
            return close_a[k], close_b[k]
        if side > 0:
            return ask_a[k], close_b[k]
        return close_a[k], ask_b[k]

    def _exit(k: int, side: int) -> tuple[float, float]:
        """Mirror: buy back at the ask, sell at the bid."""
        if cfg.mid_prices:
            return close_a[k], close_b[k]
        if side > 0:                              # sell A at bid, buy B at ask
            return close_a[k], ask_b[k]
        return ask_a[k], close_b[k]               # buy A at ask, sell B at bid

    rng = np.random.default_rng(cfg.seed)
    entry_prob = float(np.mean(np.abs(z[cfg.warmup:]) >= cfg.z_entry)) if cfg.random_entry else 0.0

    def _open(i: int, side: int) -> dict | None:
        k = i + 1                                 # one bar of execution delay
        if k >= n:
            return None
        pa, pb = _entry(k, side)
        return {"side": side, "i_open": k, "pa": pa, "pb": pb}

    for i in range(cfg.warmup, n):
        if not np.isfinite(z[i]):
            continue
        if pos is None:
            if cfg.random_entry:
                side = int(rng.choice((-1, 1))) if rng.random() < entry_prob else 0
            elif z[i] >= cfg.z_entry:
                side = -1                          # ratio too high → short the spread
            elif z[i] <= -cfg.z_entry:
                side = +1
            else:
                side = 0
            if side:
                p = _open(i, side)
                if p:
                    pos = p
                    signals.append({"time": ts[pos["i_open"]].isoformat(), "side": side, "z": float(z[i])})
        else:
            hold = i - pos["i_open"]
            hit_exit = abs(z[i]) <= cfg.z_exit
            hit_stop = abs(z[i]) >= cfg.z_stop
            hit_time = hold >= cfg.max_hold_bars
            flat = (cfg.flat_time is not None and ts[i].strftime("%H:%M") >= cfg.flat_time)
            if hit_exit or hit_stop or hit_time or flat:
                pa, pb = _exit(i, pos["side"])
                if pos["side"] > 0:               # long A, short B
                    ret = (pa - pos["pa"]) / pos["pa"] - (pb - pos["pb"]) / pos["pb"]
                else:                             # short A, long B
                    ret = (pos["pa"] - pa) / pos["pa"] - (pos["pb"] - pb) / pos["pb"]
                pnl_pct = ret * cfg.notional * 100.0
                sd_i = float(sd[pos["i_open"]]) if np.isfinite(sd[pos["i_open"]]) else 0.0
                risk_pct = abs(cfg.z_stop - cfg.z_entry) * sd_i * 100.0 * cfg.notional
                reason = "exit" if hit_exit else ("stop" if hit_stop else ("time" if hit_time else "flat"))
                trades.append({
                    "day": ts[pos["i_open"]].strftime("%Y-%m-%d"),
                    "direction": "long_spread" if pos["side"] > 0 else "short_spread",
                    "status": "closed", "pnl_pct": round(pnl_pct, 4),
                    "entry_time": float(ts[pos["i_open"]].timestamp()),
                    "exit_time": float(ts[i].timestamp()),
                    "exit_reason": reason, "r": round(pnl_pct / risk_pct, 3) if risk_pct else 0.0,
                    "z_entry": round(float(z[i]), 2),
                })
                pnl_stream.append(pnl_pct)
                pos = None

    equity = [100.0]
    cum = 0.0
    for p in pnl_stream:
        cum += p
        equity.append(100.0 + cum)
    losses = abs(sum(p for p in pnl_stream if p < 0))
    metrics = {
        "total_trades": len(trades),
        "total_pnl_pct": round(cum, 2),
        "win_rate": round(100 * sum(1 for p in pnl_stream if p > 0) / len(pnl_stream), 1) if pnl_stream else 0.0,
        "profit_factor": round(sum(p for p in pnl_stream if p > 0) / losses, 2) if losses else float("inf"),
    }
    return {"trades": trades, "metrics": metrics, "equity_curve": equity, "signals": signals}
