"""Fair Pricing Theory — intraday NASDAQ strategy (JJ Simon / Chart Fanatics).

Algorithmic transcription of the "fair pricing" interview spec. The whole
document lives in ``fair_pricing_strategy_spec.md`` at the repo root; section
numbers below refer to it.

Scope of this first implementation
----------------------------------
Implemented: every setup (A continuation, B reversion, C pre-news reversion,
D1/D2 shock, E NY-PM), all four sessions (NY AM, NY PM, Asia, London), the
fair-price state machine (§4), the primitives of §2/§3 and the fixed-rule
risk of §6.2/§8 for **one account**.

Deliberately **not** implemented (a separate subsystem, later): the prop-firm
account model, multi-account routing and the offline optimizer of §6.3–6.5,
§7 and §9. Here a signal trades the single backtest account with the fixed
eval/continuation brackets; results come out of the shared backtest pipeline
(``_compute_metrics``, the acceptance path) in R and account-%.

Approximations (kept as config flags where possible)
----------------------------------------------------
- Instrument is the Dukascopy ``E_NQ-100`` 1m bid+ask series (Nasdaq-100 index
  CFD), the available proxy for NQ/MNQ futures. Point value is a config field.
- Asia/London start times were not in the interview (``[Ann]``) → config.
- Shock detection is the price proxy of §2.6 (range vs. ATR / fixed points),
  since tweets and speeches have no feed here.
- "Largest possible stop" of D1 uses the last consolidation before the entry.
- The news FP uses the scheduled event time from a local calendar file
  (see ``trading_lab.calendar_events``); no live news feed.
- §4.5 (the FP "structural adjust" on a break out of a consolidation) is not
  implemented; the FP is set by the session open / pre-news release / pre-open
  consolidation / post-shock consolidation only.
- Two interpretability add-ons beyond the transcript: one fixed-risk position at
  a time with a throttle (``min_bars_between_entries``), and one reversion cycle
  per FP touch (a new reversion needs the price to reach the FP again). The
  transcript is discretionary about re-entry; without this the same level is
  re-traded on every momentum bar.

Validation and self-review
--------------------------
Primitives and directionality are covered by ``tests/test_fair_pricing.py``
(displacement, swings/BOS, consolidation, a crafted Setup A short, a crafted
long and short reversion). Results are reported in R and account-% through the
shared backtest pipeline; no prop-firm simulation is applied here.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..calendar_events import CalendarEvent, events_in, high_impact_times
from ..indicators import atr as atr_indicator
from ..mr_common import ASK_COLS, ensure_utc, open_sized_position, with_ask
from ..risk import Position, RiskManager

_MIN_NS = 60 * 10**9


@dataclass(frozen=True)
class FairPricingConfig:
    """Parameters of the fair-pricing strategy (spec §1)."""

    asset: str = "US100"
    asset_label: str = "US100"
    timeframe: str = "1min"
    tz: str = "America/New_York"

    account_size: float = 50_000.0
    risk_per_trade: float = 0.01
    # RiskManager protocol fields (SL/TP are always overridden per setup)
    sl_atr_mult: float = 2.0
    tp1_r: float = 1.5
    tp1_close_pct: float = 1.0
    max_daily_loss: float = 1.0        # spec has its own session stop; neutral
    cooldown_hours: int = 0
    point_value: float = 20.0          # NQ 20 $/pt (MNQ 2) — for reporting only

    # Sessions (§1). Asia/London starts are [Ann].
    sessions: tuple[str, ...] = ("ny_am", "ny_pm", "asia", "london")
    session_len_min: int = 90
    ny_am: str = "09:30"
    ny_pm: str = "14:00"
    asia: str = "18:00"
    london: str = "03:00"

    # News (§4.2)
    news_time: str = "08:30"
    news_window_end: str = "11:00"

    # HTF bias (§3)
    htf_lookback_h: int = 12
    htf_neutral_pts: float = 0.0

    # Continuation (§5A)
    cont_sl_pts: float = 25.0
    cont_tp_pts: float = 38.0
    cont_big_candle_pts: float = 25.0
    cont_big_candle_measure: str = "range"   # "range" | "body"
    cont_big_sl_pts: float = 50.0
    cont_big_tp_pts: float = 76.0
    cont_extra_window_min: int = 5
    cont_extra_max: int = 1

    # Evaluation / fixed brackets (§6.1, §6.2)
    eval_sl_pts: float = 25.0
    eval_tp_pts: float = 38.0
    tp_reach_ratio: float = 0.80
    min_sl_pts_ny: float = 25.0

    # Primitives (§2)
    swing_strict: bool = True
    disp_lookback: int = 1
    disp_double: bool = False
    consol_min_bars: int = 10
    consol_max_range_pts: float = 30.0
    consol_level: str = "midpoint"     # "midpoint" | "last_close"
    shock_range_atr_mult: float = 4.0
    shock_min_pts: float = 60.0
    atr_period: int = 20

    # Reversion (§5B) and session control (§8.1)
    move_away_min_pts: float = 0.0     # 0 → tp_reach_ratio * eval_tp_pts
    be_on_fp_touch: bool = False
    be_at_0930_for_preopen_trades: bool = True
    max_consec_rev_losses: int = 3
    post_shock_reversion: bool = True  # Setup D2
    d1_max_hold_min: int = 240         # D1 has no TP; cap the hold to keep it intraday

    # Execution realism
    max_leverage: float = 20.0           # sizing cap; high enough to honour fixed-risk
    min_bars_between_entries: int = 3    # throttle: no re-entry within this many 1m bars

    # Costs
    slippage: float = 1.0
    slippage_pct: float = 0.0
    commission: float = 0.0
    fallback_spread_pct: float = 0.0

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("htf_lookback_h", 6, 12, 1),
            ("tp_reach_ratio", 0.75, 0.90, 0.05),
            ("cont_big_candle_pts", 15, 40, 5),
            ("eval_tp_pts", 25, 100, 5),
            ("eval_sl_pts", 25, 75, 12.5),
        ]

    def session_time(self, name: str) -> str:
        return {"ny_am": self.ny_am, "ny_pm": self.ny_pm,
                "asia": self.asia, "london": self.london}[name]

    def move_away_floor(self) -> float:
        return self.move_away_min_pts if self.move_away_min_pts > 0 else self.tp_reach_ratio * self.eval_tp_pts


# ── Primitive: displacement (§2.1) ──────────────────────────────────


def displacement_at(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    i: int, *, lookback: int = 1, double: bool = False, strict: bool = True,
) -> int:
    """±1 if bar ``i`` is a displacement candle in that direction, else 0."""
    if i < max(lookback, 2):
        return 0
    body = abs(close[i] - open_[i])
    color = np.sign(close[i] - open_[i])
    prev_color = np.sign(close[i - 1] - open_[i - 1])
    if double:
        prev_ok = prev_color == -color and color != 0
    else:
        prev_body = abs(close[i - 1] - open_[i - 1])
        bigger = body > prev_body if strict else body >= prev_body
        prev_ok = bigger and prev_color == -color and color != 0
    if not prev_ok:
        return 0
    ref_low = low[i - lookback:i].min()
    ref_high = high[i - lookback:i].max()
    if color < 0 and close[i] < ref_low:
        return -1
    if color > 0 and close[i] > ref_high:
        return 1
    return 0


class SwingTracker:
    """Causal 3-bar swing tracker with per-swing broken state (§2.2/§2.3).

    An older swing low stays the reference while every newer swing has been
    broken, so broken swings must not be pruned by count. The tracker keeps
    them, detects breaks with a price heap (each swing pops once, O(S log S)
    total) and keeps a backward pointer to the youngest unbroken swing
    (amortised O(1)). This preserves the "youngest outstanding swing" rule
    without the O(bars × swings) scan.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self.highs: list[list[float]] = []   # [idx, price, broken]
        self.lows: list[list[float]] = []
        self._low_heap: list[tuple[float, int]] = []   # (-price, list_idx)
        self._high_heap: list[tuple[float, int]] = []  # (price, list_idx)
        self._yl = -1
        self._yh = -1

    def _confirm(self, i: int, high: np.ndarray, low: np.ndarray) -> None:
        j = i - 1
        if j < 1:
            return
        lt = (lambda a, b: a < b) if self.strict else (lambda a, b: a <= b)
        gt = (lambda a, b: a > b) if self.strict else (lambda a, b: a >= b)
        if lt(low[j], low[j - 1]) and lt(low[j], low[j + 1]):
            self.lows.append([float(j), float(low[j]), 0.0])
            heapq.heappush(self._low_heap, (-float(low[j]), len(self.lows) - 1))
            self._yl = len(self.lows) - 1
        if gt(high[j], high[j - 1]) and gt(high[j], high[j + 1]):
            self.highs.append([float(j), float(high[j]), 0.0])
            heapq.heappush(self._high_heap, (float(high[j]), len(self.highs) - 1))
            self._yh = len(self.highs) - 1

    def _mark_broken(self, i: int, close: np.ndarray) -> None:
        c = close[i]
        while self._low_heap and -self._low_heap[0][0] > c:
            _, k = heapq.heappop(self._low_heap)
            self.lows[k][2] = 1.0
        while self._high_heap and self._high_heap[0][0] < c:
            _, k = heapq.heappop(self._high_heap)
            self.highs[k][2] = 1.0

    def update(self, i: int, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> None:
        self._mark_broken(i, close)
        self._confirm(i, high, low)

    def youngest_unbroken_low(self) -> float | None:
        swing = self.youngest_unbroken_low_swing()
        return swing[1] if swing is not None else None

    def youngest_unbroken_high(self) -> float | None:
        swing = self.youngest_unbroken_high_swing()
        return swing[1] if swing is not None else None

    def youngest_unbroken_low_swing(self) -> list[float] | None:
        while self._yl >= 0 and self.lows[self._yl][2]:
            self._yl -= 1
        return self.lows[self._yl] if self._yl >= 0 else None

    def youngest_unbroken_high_swing(self) -> list[float] | None:
        while self._yh >= 0 and self.highs[self._yh][2]:
            self._yh -= 1
        return self.highs[self._yh] if self._yh >= 0 else None

    def bos_dir(self, close: float) -> int:
        """+1 long BOS (close above youngest unbroken high), -1 short, else 0."""
        hi = self.youngest_unbroken_high()
        lo = self.youngest_unbroken_low()
        if hi is not None and close > hi:
            return 1
        if lo is not None and close < lo:
            return -1
        return 0


# ── Primitive: consolidation (§2.5) and shock (§2.6) ────────────────


def consolidation_arrays(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, cfg: FairPricingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar (ok, level) for the trailing-consolidation shape of §2.5."""
    n = len(high)
    ok = np.zeros(n, dtype=bool)
    level = np.full(n, np.nan)
    w = cfg.consol_min_bars
    if n < w:
        return ok, level
    h = pd.Series(high).rolling(w).max().to_numpy()
    l = pd.Series(low).rolling(w).min().to_numpy()
    span = h - l
    ok = np.isfinite(span) & (span <= cfg.consol_max_range_pts)
    if cfg.consol_level == "last_close":
        level = close.copy()
    else:
        level = (h + l) / 2.0
    return ok, level


# ── Primitive: HTF bias (§3) ────────────────────────────────────────


def htf_bias_array(index: pd.DatetimeIndex, close: np.ndarray, cfg: FairPricingConfig) -> np.ndarray:
    """Per-1m-bar HTF bias: reversion sign of the last ``htf_lookback_h`` hours."""
    s = pd.Series(close, index=index)
    tf5 = s.resample("5min", label="left", closed="left").last().dropna()
    lookback = max(int(round(cfg.htf_lookback_h * 12)), 1)
    delta = tf5 - tf5.shift(lookback)
    bias5 = -np.sign(delta.to_numpy())
    bias5[np.abs(delta.to_numpy()) <= cfg.htf_neutral_pts] = 0.0
    bias5[~np.isfinite(delta.to_numpy())] = 0.0
    end5 = (tf5.index + pd.Timedelta(minutes=5)).as_unit("ns").asi8
    pos = np.searchsorted(end5, index.as_unit("ns").asi8, side="right") - 1
    out = np.zeros(len(index))
    valid = pos >= 0
    out[valid] = bias5[pos[valid]]
    return out


# ── Prepared data ───────────────────────────────────────────────────


@dataclass
class _Session:
    name: str
    start_ns: int
    window_end_ns: int
    open_idx: int = -1
    cash_open_idx: int = -1        # first NY-cash bar (09:30); == open_idx unless news
    is_news: bool = False
    fp: float = 0.0
    fp_source: str = "SESSION_OPEN"
    fp2: float | None = None
    open_price: float = 0.0
    cont_mode: bool = False
    cont_dir: int = 0
    extra_cont_taken: int = 0
    consec_rev_losses: int = 0
    stopped: bool = False
    news_fp: float | None = None
    news_dir: int = 0
    last_fp_touch_ns: int = 0
    post_shock: bool = False
    active: bool = False
    last_entry_idx: int = -10**9
    rev_block: bool = False        # set after a reversion entry; cleared on FP touch
    event_cutoff_ns: int = 0       # §8.1: no new entries from a scheduled event on


@dataclass
class Prepared:
    ts: np.ndarray
    local: pd.DatetimeIndex
    bid: dict[str, np.ndarray]
    ask: dict[str, np.ndarray]
    spread: np.ndarray
    atr: np.ndarray
    bias: np.ndarray
    consol_ok: np.ndarray
    consol_level: np.ndarray
    sessions: list[_Session]


def _session_start_ns(d: date, hhmm: str, tz: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    local = pd.Timestamp(datetime(d.year, d.month, d.day, h, m), tz=tz)
    return int(local.tz_convert("UTC").value)


def prepare(
    data: pd.DataFrame,
    cfg: FairPricingConfig,
    *,
    events: Iterable[CalendarEvent] | None = None,
) -> Prepared:
    """Precompute arrays, sessions and per-bar context for the engine.

    When a Red Folder event falls on ``news_time`` of a day, the NY-AM session
    is opened at the event (``08:30``, §4.2) so the pre-news reversion (Setup C)
    exists; its cash open (09:30) is tracked separately for the open decision.
    """
    if not all(c in data.columns for c in ASK_COLS):
        data = with_ask(data, cfg.fallback_spread_pct)
    df = ensure_utc(data)
    tz = ZoneInfo(cfg.tz)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    bid = {"open": o, "high": h, "low": l, "close": c}
    ask = {k: df[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close")}
    ts = df.index.as_unit("ns").asi8
    local = df.index.tz_convert(tz)

    atr = atr_indicator(df["high"], df["low"], df["close"], cfg.atr_period).to_numpy(float)
    bias = htf_bias_array(df.index, c, cfg)
    consol_ok, consol_level = consolidation_arrays(h, l, c, cfg)

    red_days: set[date] = set()
    if events:
        for t in high_impact_times(events):
            red_days.add(datetime.fromtimestamp(t, tz=tz).date())
    hi_times = high_impact_times(events) if events else []

    def event_cutoff(start_ns: int, end_ns: int) -> int:
        """§8.1: first scheduled Red Folder event inside a session, if any.

        The release that anchors the pre-news FP (at ``news_time``) is skipped,
        so Setup C is not blocked by its own event.
        """
        for t in hi_times:
            t_ns = int(t * 1e9)
            if start_ns + 5 * _MIN_NS < t_ns < end_ns:
                return t_ns
        return 0

    def bar_near(start_ns: int) -> int | None:
        i = int(np.searchsorted(ts, start_ns, side="left"))
        if i >= len(ts) or ts[i] - start_ns > 15 * _MIN_NS:
            return None
        return i

    days = pd.DatetimeIndex(local.tz_localize(None).normalize().unique())
    sessions: list[_Session] = []
    for name in cfg.sessions:
        for d in days:
            day = d.date()
            if name == "ny_am" and day in red_days:
                news_ns = _session_start_ns(day, cfg.news_time, cfg.tz)
                cash_ns = _session_start_ns(day, cfg.ny_am, cfg.tz)
                i_news, i_cash = bar_near(news_ns), bar_near(cash_ns)
                if i_news is not None and i_cash is not None:
                    end_ns = _session_start_ns(day, cfg.news_window_end, cfg.tz)
                    sessions.append(_Session(
                        name=name, start_ns=news_ns, window_end_ns=end_ns,
                        open_idx=i_news, cash_open_idx=i_cash, is_news=True,
                        event_cutoff_ns=event_cutoff(news_ns, end_ns)))
                continue
            start = _session_start_ns(day, cfg.session_time(name), cfg.tz)
            i = bar_near(start)
            if i is None:
                continue
            end_ns = start + cfg.session_len_min * _MIN_NS
            sessions.append(_Session(
                name=name, start_ns=start, window_end_ns=end_ns,
                open_idx=i, cash_open_idx=i,
                event_cutoff_ns=event_cutoff(start, end_ns)))
    sessions.sort(key=lambda s: s.open_idx)
    return Prepared(ts=ts, local=local, bid=bid, ask=ask,
                    spread=ask["close"] - bid["close"], atr=atr, bias=bias,
                    consol_ok=consol_ok, consol_level=consol_level, sessions=sessions)


# ── Cost / fill helpers ─────────────────────────────────────────────


def _slip(cfg: FairPricingConfig, price: float) -> float:
    return cfg.slippage + cfg.slippage_pct * price


def _entry_price(p: Prepared, cfg: FairPricingConfig, i: int, direction: int) -> float:
    """Market at the bar open, paying the spread + slippage against us."""
    if direction > 0:
        return float(p.ask["open"][i]) + _slip(cfg, p.bid["open"][i])
    return float(p.bid["open"][i]) - _slip(cfg, p.bid["open"][i])


# ── Engine ──────────────────────────────────────────────────────────


class _Engine:
    """Bar-by-bar fair-pricing engine for a single account."""

    def __init__(self, p: Prepared, cfg: FairPricingConfig, events: Iterable[CalendarEvent] | None):
        self.p = p
        self.cfg = cfg
        self.rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
        self.swings = SwingTracker(strict=cfg.swing_strict)
        self.events = list(events or [])
        self.trades: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.pos: Position | None = None
        self.sess: _Session | None = None
        self._n = len(p.ts)
        self._pending: dict[str, Any] | None = None   # entry armed for next bar
        self._shock_dir: int = 0
        self._be_done = False
        self._pending_preopen: tuple[float, int, _Session] | None = None

    # ── session lifecycle ───────────────────────────────────────
    def _init_fp(self, s: _Session, i: int) -> None:
        s.fp = float(self.p.bid["open"][i])
        s.open_price = s.fp
        s.fp_source = "SESSION_OPEN"
        if s.is_news:
            # §4.2 — the last close before the release is fair until the open decides
            j = int(np.searchsorted(self.p.ts, s.start_ns, side="left")) - 1
            if j >= 0:
                s.fp = float(self.p.bid["close"][j])
                s.news_fp = s.fp
                s.fp_source = "PRE_NEWS"
        if s.name == "ny_pm":
            # §4.6 — the 09:30 open is the secondary reversion target (FP2)
            am_ns = _session_start_ns(self.p.local[i].date(), self.cfg.ny_am, self.cfg.tz)
            j = int(np.searchsorted(self.p.ts, am_ns, side="left"))
            if j < len(self.p.ts) and self.p.ts[j] - am_ns <= 15 * _MIN_NS:
                s.fp2 = float(self.p.bid["open"][j])
        if s.name == "ny_am":
            self._init_preopen_consolidation(s, i)

    def _apply_open_decision(self, s: _Session, i: int) -> None:
        """§4.2 — at the 09:30 cash open on a news day, open vs. pre-news level."""
        if not s.is_news or s.news_fp is None:
            return
        k0 = int(np.searchsorted(self.p.ts, s.start_ns, side="left"))
        if i <= k0:
            return
        post = self.p.bid["close"][k0:i]
        news_fp = s.news_fp
        news_dir = int(np.sign(float(self.p.bid["close"][i - 1]) - news_fp))
        s.news_dir = news_dir
        s.open_price = float(self.p.bid["open"][i])
        if news_dir == 0:
            return
        extremum = post.max() if news_dir > 0 else post.min()
        o, c = float(self.p.bid["open"][i]), float(self.p.bid["close"][i])
        beyond = (c > extremum) if news_dir > 0 else (c < extremum)
        if int(np.sign(c - o)) == news_dir and beyond:
            s.fp = s.open_price            # news more important than expected → open is fair
            s.fp_source = "SESSION_OPEN"

    def _init_preopen_consolidation(self, s: _Session, i: int) -> None:
        """§4.3 — a pre-open consolidation whose level the open traverses wins."""
        cfg = self.cfg
        lo_ns = s.start_ns - 90 * _MIN_NS
        j0, j1 = int(np.searchsorted(self.p.ts, lo_ns, "left")), int(np.searchsorted(self.p.ts, s.start_ns, "left"))
        best: float | None = None
        for j in range(j1 - 1, j0 - 1, -1):
            if self.p.consol_ok[j]:
                best = float(self.p.consol_level[j])
                break
        if best is not None:
            self._pending_preopen = (best, i + 15, s)

    def _manage_preopen(self, i: int) -> None:
        pend = getattr(self, "_pending_preopen", None)
        if not pend:
            return
        level, until, s = pend
        if i >= until:
            self._pending_preopen = None
            return
        hi, lo = self.p.bid["high"][i], self.p.bid["low"][i]
        if lo <= level <= hi:
            s.fp = level
            s.fp_source = "PREOPEN_CONSOL"
            self._pending_preopen = None

    # ── signals ─────────────────────────────────────────────────
    def _reversion_signal(self, i: int, s: _Session, fp: float) -> int:
        c = float(self.p.bid["close"][i])
        if fp <= 0 or c == fp:
            return 0
        d = 1 if fp > c else -1
        disp = displacement_at(self.p.bid["open"], self.p.bid["high"], self.p.bid["low"],
                               self.p.bid["close"], i, lookback=self.cfg.disp_lookback,
                               double=self.cfg.disp_double)
        bos = self.swings.bos_dir(c)
        return d if (disp == d or bos == d) else 0

    # ── entry / exit ────────────────────────────────────────────
    def _arm(self, i: int, direction: int, setup: str, sl_pts: float, tp_pts: float,
             size_mult: float = 1.0, sl_abs: float | None = None,
             tp_abs: float | None = None, no_tp: bool = False) -> None:
        """Enter at the open of bar ``i+1`` (spec FILL_MODEL)."""
        self._pending = {
            "i": i + 1, "direction": direction, "setup": setup,
            "sl_pts": sl_pts, "tp_pts": tp_pts, "size_mult": size_mult,
            "sl_abs": sl_abs, "tp_abs": tp_abs, "no_tp": no_tp,
        }

    def _try_fill_pending(self, i: int) -> None:
        pend = self._pending
        if not pend or pend["i"] != i or self.pos is not None:
            if pend and pend["i"] < i:
                self._pending = None
            return
        self._pending = None
        cfg = self.cfg
        d = pend["direction"]
        entry = _entry_price(self.p, cfg, i, d)
        if pend["sl_abs"] is not None:
            sl = float(pend["sl_abs"])
        else:
            sl = entry - d * pend["sl_pts"]
        tp = None if pend["no_tp"] else (
            float(pend["tp_abs"]) if pend["tp_abs"] is not None else entry + d * pend["tp_pts"]
        )
        if abs(entry - sl) <= 0:
            return
        pos = open_sized_position(
            self.rm, entry_price=entry, stop_loss=sl, tp1=tp if tp is not None else entry + d * pend["tp_pts"],
            tp2=None, direction="long" if d > 0 else "short",
            entry_time=float(self.p.ts[i]) / 1e9, entry_idx=i, atr=float(self.p.atr[i]),
            max_leverage=cfg.max_leverage, size_mult=pend["size_mult"],
            meta={"setup": pend["setup"], "tp_abs": tp, "no_tp": pend["no_tp"],
                  "sl_pts": pend["sl_pts"], "tp_pts": pend["tp_pts"]},
        )
        if pos is None:
            return
        pos.tp2 = None
        pos.meta["risk_points"] = abs(entry - sl)
        self.pos = pos
        self._be_done = False
        if self.sess is not None:
            self.sess.last_entry_idx = i
        self.signals.append({"i": i, "time": pd.Timestamp(self.p.ts[i], tz="UTC").isoformat(),
                             "setup": pend["setup"], "direction": d, "entry": round(entry, 4)})

    def _close(self, i: int, price: float, reason: str) -> None:
        pos = self.pos
        assert pos is not None
        self.rm.close_position(pos, price, reason, float(self.p.ts[i]) / 1e9)
        if pos.meta is not None:
            entry = pos.entry_price
            rp = pos.meta.get("risk_points") or 0.0
            pos.meta["r"] = round(((price - entry) if pos.direction == "long" else (entry - price)) / rp, 4) if rp else 0.0
        self.trades.append(_pos_to_dict(pos))
        if pos.meta and pos.meta.get("setup") in ("B", "C", "D2") and self.sess is not None:
            if (pos.pnl_pct or 0.0) < 0:
                self.sess.consec_rev_losses += 1
                if self.sess.consec_rev_losses >= self.cfg.max_consec_rev_losses:
                    self.sess.stopped = True
            else:
                self.sess.consec_rev_losses = 0
        self.pos = None

    def _manage_position(self, i: int, s: _Session | None) -> None:
        pos = self.pos
        if pos is None:
            return
        cfg = self.cfg
        d = 1 if pos.direction == "long" else -1
        exit_px = self.p.bid if d > 0 else self.p.ask
        hi, lo, opn = exit_px["high"][i], exit_px["low"][i], exit_px["open"][i]

        # break-even rules (§8.2)
        if cfg.be_at_0930_for_preopen_trades and not self._be_done and s is not None:
            open930_s = _session_start_ns(self.p.local[i].date(), cfg.ny_am, cfg.tz) / 1e9
            if pos.entry_time < open930_s <= self.p.ts[i] / 1e9:
                pos.stop_loss = pos.entry_price
                pos.sl_moved_to_be = True
                self._be_done = True
        if cfg.be_on_fp_touch and not self._be_done and s is not None and s.fp > 0:
            tp = pos.tp1
            beyond = (tp > s.fp) if d > 0 else (tp < s.fp)
            if beyond and lo <= s.fp <= hi:
                pos.stop_loss = pos.entry_price
                pos.sl_moved_to_be = True
                self._be_done = True

        # hard bracket (§8.2): stop first
        hit_sl = lo <= pos.stop_loss if d > 0 else hi >= pos.stop_loss
        if hit_sl:
            lvl = min(pos.stop_loss, opn) if d > 0 else max(pos.stop_loss, opn)
            self._close(i, float(lvl), "breakeven_stop" if pos.sl_moved_to_be else "stop_loss")
            return
        tp = pos.meta.get("tp_abs") if pos.meta else None
        if tp is not None:
            hit_tp = hi >= tp if d > 0 else lo <= tp
            if hit_tp:
                self._close(i, float(tp), "tp")
                return

        # D1: no TP. Exit on the youngest unbroken swing created after entry
        # (O(1)), and cap the hold so the intraday backtest cannot sit in one
        # trade for years.
        if pos.meta and pos.meta.get("setup") == "D1":
            c = float(self.p.bid["close"][i])
            if d > 0:
                sw = self.swings.youngest_unbroken_low_swing()
                if sw is not None and sw[0] > pos.entry_idx and c < sw[1]:
                    self._close(i, c, "opposing_bos")
                    return
            else:
                sw = self.swings.youngest_unbroken_high_swing()
                if sw is not None and sw[0] > pos.entry_idx and c > sw[1]:
                    self._close(i, c, "opposing_bos")
                    return
            if (i - pos.entry_idx) >= self.cfg.d1_max_hold_min:
                self._close(i, c, "max_hold")
                return

    # ── main loop ───────────────────────────────────────────────
    def run(self, verbose: bool = False) -> dict[str, Any]:
        p, cfg = self.p, self.cfg
        equity: list[float] = [self.rm.current_equity]
        sess_by_open: dict[int, _Session] = {s.open_idx: s for s in p.sessions}
        move_floor = cfg.move_away_floor()

        for i in range(self._n):
            self._try_fill_pending(i)
            s_start = sess_by_open.get(i)
            if s_start is not None:
                self.sess = s_start
                s_start.active = True
                self._init_fp(s_start, i)
                if s_start.cash_open_idx == i:      # non-news day: open == session start
                    self._try_setup_a(s_start, i)
            else:
                s = self.sess
                if s is not None and p.ts[i] >= s.window_end_ns:
                    s.active = False
                    self.sess = None
                    s = None
                if s is not None:
                    if i == s.cash_open_idx and s.open_idx != s.cash_open_idx:
                        self._apply_open_decision(s, i)   # §4.2 at 09:30
                        self._try_setup_a(s, i)
                    self._manage_preopen(i)
                    self._fp_update(i, s)
                    if self._detect_shock(i, s):
                        self._try_setup_d1(s, i)
                    elif not s.stopped and self._in_window(s, i) and not self._event_blackout(i, s):
                        if s.cont_mode:
                            self._try_extra_continuation(s, i)
                        self._try_reversion(s, i, move_floor)

            self._manage_position(i, self.sess)
            # structure is updated after the bar's signals: the close of bar i
            # breaks an outstanding swing, then the swing is marked broken.
            self.swings.update(i, p.bid["high"], p.bid["low"], p.bid["close"])
            equity.append(self.rm.current_equity)

        if self.pos is not None:
            self._close(self._n - 1, float(self.p.bid["close"][-1]), "data_end")

        return {"trades": self.trades, "signals": self.signals,
                "equity_curve": equity, "metrics": _compute_metrics(self.trades, equity, cfg)}

    def _in_window(self, s: _Session, i: int) -> bool:
        return self.p.ts[i] < s.window_end_ns

    def _event_blackout(self, i: int, s: _Session | None = None) -> bool:
        if s is not None and s.event_cutoff_ns and self.p.ts[i] >= s.event_cutoff_ns:
            return True
        if not self.events:
            return False
        t = float(self.p.ts[i]) / 1e9
        return bool(events_in(self.events, t, t + 60))

    def _detect_shock(self, i: int, s: _Session) -> bool:
        cfg = self.cfg
        rng = float(self.p.bid["high"][i] - self.p.bid["low"][i])
        atr = float(self.p.atr[i]) if np.isfinite(self.p.atr[i]) else 0.0
        thresh = max(cfg.shock_min_pts, cfg.shock_range_atr_mult * atr)
        news_ns = _session_start_ns(self.p.local[i].date(), cfg.news_time, cfg.tz) if s.name == "ny_am" else None
        in_news = news_ns is not None and 0 <= (self.p.ts[i] - news_ns) <= 2 * _MIN_NS
        if rng >= thresh and not in_news and i != s.open_idx:
            self._shock_dir = int(np.sign(self.p.bid["close"][i] - self.p.bid["open"][i]))
            s.post_shock = True
            return True
        return False

    def _try_setup_d1(self, s: _Session, i: int) -> None:
        if self.pos is not None or i + 1 >= self._n:
            return
        d = self._shock_dir
        if d == 0:
            return
        # largest possible stop: opposite side of the last consolidation before entry
        lo = hi = None
        for j in range(i - 1, max(s.open_idx, i - 240) - 1, -1):
            if self.p.consol_ok[j]:
                w = self.cfg.consol_min_bars
                lo = float(self.p.bid["low"][j - w + 1:j + 1].min())
                hi = float(self.p.bid["high"][j - w + 1:j + 1].max())
                break
        entry = _entry_price(self.p, self.cfg, i + 1, d)
        sl_abs = (lo if d > 0 else hi)
        if sl_abs is None:
            sl_abs = entry - d * self.cfg.eval_sl_pts * 2
        if abs(entry - sl_abs) <= 0:
            return
        self._arm(i, d, "D1", abs(entry - sl_abs), 0.0, sl_abs=sl_abs, no_tp=True)

    def _try_setup_a(self, s: _Session, i: int) -> None:
        if self.pos is not None:
            return
        cfg = self.cfg
        o, c = float(self.p.bid["open"][i]), float(self.p.bid["close"][i])
        color = int(np.sign(c - o))
        if color == 0:
            return
        if int(self.p.bias[i]) != color:           # §5A filter 1
            return
        sl_sw = self.swings.youngest_unbroken_high() if color > 0 else self.swings.youngest_unbroken_low()
        if sl_sw is None:
            j = max(i - 1, 0)
            prev = self.p.bid["high"][j] if color > 0 else self.p.bid["low"][j]
            beyond = (c > prev) if color > 0 else (c < prev)
        else:
            beyond = (c > sl_sw) if color > 0 else (c < sl_sw)
        if not beyond:
            return
        measure = (float(self.p.bid["high"][i] - self.p.bid["low"][i])
                   if cfg.cont_big_candle_measure == "range" else abs(c - o))
        if measure <= cfg.cont_big_candle_pts:
            self._arm(i, color, "A", cfg.cont_sl_pts, cfg.cont_tp_pts, size_mult=1.0)
        else:
            self._arm(i, color, "A", cfg.cont_big_sl_pts, cfg.cont_big_tp_pts, size_mult=0.5)
        s.cont_mode = True
        s.cont_dir = color

    def _try_extra_continuation(self, s: _Session, i: int) -> None:
        cfg = self.cfg
        if s.extra_cont_taken >= cfg.cont_extra_max or self.pos is not None:
            return
        if (self.p.ts[i] - int(self.p.ts[s.cash_open_idx])) > cfg.cont_extra_window_min * _MIN_NS:
            return
        d = s.cont_dir
        disp = displacement_at(self.p.bid["open"], self.p.bid["high"], self.p.bid["low"],
                               self.p.bid["close"], i, lookback=cfg.disp_lookback)
        if disp == d or self.swings.bos_dir(float(self.p.bid["close"][i])) == d:
            s.extra_cont_taken += 1
            self._arm(i, d, "A", cfg.cont_sl_pts, cfg.cont_tp_pts)

    def _try_reversion(self, s: _Session, i: int, move_floor: float) -> None:
        cfg = self.cfg
        if self.pos is not None or i <= s.open_idx or s.rev_block:
            return
        if i - s.last_entry_idx < cfg.min_bars_between_entries:
            return
        c = float(self.p.bid["close"][i])
        gate = max(move_floor, cfg.tp_reach_ratio * cfg.eval_tp_pts)
        refs = [s.fp]
        if s.name == "ny_pm" and s.fp2 is not None and s.fp2 != s.fp:
            refs.append(s.fp2)          # §4.6 secondary target
        chosen: tuple[float, int] | None = None
        for fp in refs:
            d = self._reversion_signal(i, s, fp)
            if d == 0:
                continue
            runway = abs(fp - c)
            if runway < gate:
                continue
            # §5B exclusion: the close must not have reached/crossed the FP yet
            if (d > 0 and c >= fp) or (d < 0 and c <= fp):
                continue
            chosen = (fp, d)
            break
        if chosen is None:
            return
        fp, d = chosen
        setup = "C" if s.fp_source == "PRE_NEWS" else ("D2" if s.post_shock else "B")
        # §6.2: the take-profit is a fixed point target (38), not the FP itself;
        # D >= 0.8 * TP only qualifies the trade (the FP is a waypoint).
        self._arm(i, d, setup, cfg.eval_sl_pts, cfg.eval_tp_pts)
        s.rev_block = True

    def _fp_update(self, i: int, s: _Session) -> None:
        """Post-shock consolidation becomes the FP (§4.4) and cycle bookkeeping."""
        c = float(self.p.bid["close"][i])
        spread = float(self.p.spread[i]) if np.isfinite(self.p.spread[i]) else 0.0
        if abs(c - s.fp) <= max(spread, 0.25):
            # FP reached → a new reversion cycle may start
            s.last_fp_touch_ns = int(self.p.ts[i])
            s.rev_block = False
        if s.post_shock and self.p.consol_ok[i]:
            new_fp = float(self.p.consol_level[i])
            if new_fp != s.fp:
                s.rev_block = False
            s.fp = new_fp
            s.fp_source = "POST_SHOCK_CONSOL"


# ── Public entry point ──────────────────────────────────────────────


def run_backtest(
    data: pd.DataFrame,
    cfg: FairPricingConfig,
    *,
    events: Iterable[CalendarEvent] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the fair-pricing strategy over a 1m bid+ask frame.

    ``events`` is the economic calendar (from ``load_events``); without it the
    news setups (C, and the 08:30 FP rule) stay inactive. Returns the shared
    contract: ``{"trades", "metrics", "equity_curve", "signals"}``.
    """
    prep = prepare(data, cfg, events=events)
    return _Engine(prep, cfg, events).run(verbose=verbose)
