"""Live signal loop for Supertrend+Volume HF strategy.

Polls 1h candles (Dukascopy XAU/USD spot by default), manages paper
positions, sends Telegram alerts. Frozen config — no reflection or evolution.
"""
from __future__ import annotations
from .env import env

import asyncio
import dataclasses
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adapters.market_data import get_market_adapter
from .backtest import _pos_to_dict
from .notify.telegram import TelegramNotifier
from .risk import Position, RiskManager
from .strategies.supertrend_volume import (
    SupertrendVolumeConfig,
    generate_signals,
    manage_position,
)

MAX_CONSECUTIVE_FAILURES = 5

_BAR_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "10m": timedelta(minutes=10),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "60m": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _next_entry_poll_at(now: datetime, offset_sec: int) -> datetime:
    """Next UTC poll time: hour boundary + offset (e.g. offset=60 → :01:00)."""
    hour_start = _hour_floor(now)
    target = hour_start + timedelta(seconds=offset_sec)
    if now < target:
        return target
    return hour_start + timedelta(hours=1, seconds=offset_sec)


def _in_entry_retry_window(
    now: datetime, offset_sec: int, window_sec: int
) -> bool:
    hour_start = _hour_floor(now)
    target = hour_start + timedelta(seconds=offset_sec)
    return target <= now < target + timedelta(seconds=window_sec)


def bar_timedelta(interval: str) -> timedelta:
    key = (interval or "1h").strip().lower()
    if key not in _BAR_DELTAS:
        raise ValueError(f"Unsupported candle interval for signal loop: {interval!r}")
    return _BAR_DELTAS[key]


def _ts_utc(ts: Any) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC").to_pydatetime()
    return t.tz_convert("UTC").to_pydatetime()


def latest_closed_bar_index(
    index: pd.DatetimeIndex,
    now: datetime,
    bar_delta: timedelta,
) -> int:
    """Newest bar whose period has ended (open + bar_delta <= now).

    Dukascopy historical feeds often omit the still-forming bar, so ``iloc[-1]``
    is already closed. Never assume ``iloc[-2]`` is the signal bar.
    """
    if len(index) == 0:
        return -1
    if now.tzinfo is None:
        now_utc = now.replace(tzinfo=timezone.utc)
    else:
        now_utc = now.astimezone(timezone.utc)
    for i in range(len(index) - 1, -1, -1):
        if _ts_utc(index[i]) + bar_delta <= now_utc:
            return i
    return -1


def _load_st_config(path: Path) -> SupertrendVolumeConfig:
    if not path.exists():
        return SupertrendVolumeConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return SupertrendVolumeConfig(**kwargs)


def _position_from_dict(data: dict[str, Any]) -> Position:
    fields = {f.name for f in dataclasses.fields(Position)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    return Position(**kwargs)


class SignalLoop:
    """HF signal worker: Supertrend+Volume on 1h with Telegram alerts."""

    def __init__(
        self,
        state_dir: Path,
        strategy_path: Path,
        *,
        interval_seconds: int = 60,
        hour_aligned: bool = True,
        entry_poll_offset_seconds: int = 60,
        entry_retry_interval_seconds: int = 20,
        entry_retry_window_seconds: int = 900,
        data_period: str = "60d",
        candle_interval: str = "1h",
        mode: str = "paper",
        notifier: TelegramNotifier | None = None,
        data_source: str | None = None,
        state_suffix: str = "",
        log_tag: str = "signal",
    ):
        self.state_dir = Path(state_dir)
        self.strategy_path = Path(strategy_path)
        self.interval = interval_seconds
        self.hour_aligned = hour_aligned
        self.entry_poll_offset = entry_poll_offset_seconds
        self.entry_retry_interval = entry_retry_interval_seconds
        self.entry_retry_window = entry_retry_window_seconds
        self.data_period = data_period
        self.candle_interval = candle_interval
        self.mode = mode
        self.notifier = notifier or TelegramNotifier()
        self.data_source = (
            data_source
            or env("LAB_DATA_SOURCE", "dukascopy").strip().lower()
        )
        self.adapter = get_market_adapter(self.data_source)
        self.consecutive_failures = 0
        self._last_heartbeat_day: str | None = None
        self.state_suffix = state_suffix
        self.log_tag = log_tag

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.rm: RiskManager | None = None
        self.open_pos: Position | None = None

    def _state_name(self, base: str) -> str:
        if not self.state_suffix:
            return base
        stem, dot, ext = base.partition(".")
        return f"{stem}{self.state_suffix}{dot}{ext}" if dot else f"{base}{self.state_suffix}"

    @property
    def _position_path(self) -> Path:
        return self.state_dir / self._state_name("position.json")

    @property
    def _last_bar_path(self) -> Path:
        return self.state_dir / self._state_name("last_signal_bar.txt")

    @property
    def _trades_path(self) -> Path:
        return self.state_dir / self._state_name("live_trades.jsonl")

    @property
    def _heartbeat_path(self) -> Path:
        return self.state_dir / self._state_name("heartbeat.json")

    @property
    def _heartbeat_day_path(self) -> Path:
        return self.state_dir / self._state_name("last_heartbeat_day.txt")

    def _read_last_heartbeat_day(self) -> str | None:
        if not self._heartbeat_day_path.exists():
            return None
        return self._heartbeat_day_path.read_text().strip() or None

    def _write_last_heartbeat_day(self, day: str) -> None:
        self._heartbeat_day_path.write_text(day)

    def _load_persisted_state(self, cfg: SupertrendVolumeConfig) -> None:
        self.rm = RiskManager(
            cfg, account_size=cfg.account_size, backtest_mode=True
        )
        if not self._position_path.exists():
            self.open_pos = None
            return
        try:
            raw = json.loads(self._position_path.read_text())
            pos_data = raw.get("position")
            rm_data = raw.get("risk_state")
            if pos_data:
                self.open_pos = _position_from_dict(pos_data)
                if self.open_pos.status != "open":
                    self.open_pos = None
            if rm_data and self.rm:
                for key, val in rm_data.items():
                    if hasattr(self.rm.state, key):
                        setattr(self.rm.state, key, val)
        except Exception as exc:
            print(f"[{self.log_tag}] WARN could not load position.json: {exc}")
            self.open_pos = None

    def _save_persisted_state(self) -> None:
        if self.rm is None:
            return
        payload = {
            "position": asdict(self.open_pos) if self.open_pos else None,
            "risk_state": asdict(self.rm.state),
            "saved_at": time.time(),
        }
        self._position_path.write_text(json.dumps(payload, indent=2, default=str))

    def _read_last_bar(self) -> str | None:
        if not self._last_bar_path.exists():
            return None
        return self._last_bar_path.read_text().strip() or None

    def _write_last_bar(self, bar_ts: str) -> None:
        self._last_bar_path.write_text(bar_ts)

    def _config_version(self) -> str:
        if not self.strategy_path.exists():
            return "?"
        raw = yaml.safe_load(self.strategy_path.read_text()) or {}
        return str(raw.get("version", "?"))

    def _append_trade(self, pos: Position) -> None:
        with open(self._trades_path, "a") as f:
            f.write(json.dumps(_pos_to_dict(pos), default=str) + "\n")

    async def _fetch_ohlcv(self, asset: str) -> pd.DataFrame:
        """Warm history + short tip refresh (long-range Dukascopy can lag ~1h)."""
        df = await self.adapter.fetch_history(
            asset, interval=self.candle_interval, period=self.data_period
        )
        tip_start = _utc_now() - timedelta(days=3)
        tip_end = _utc_now() + timedelta(minutes=5)
        try:
            tip = await self.adapter.fetch_history(
                asset,
                interval=self.candle_interval,
                start=tip_start,
                end=tip_end,
            )
        except TypeError:
            # Adapters that only accept period=
            tip = pd.DataFrame()
        except Exception as exc:
            print(f"[{self.log_tag}] WARN tip refresh failed: {exc}")
            tip = pd.DataFrame()

        if tip.empty:
            return df
        if df.empty:
            return tip
        merged = pd.concat([df, tip])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged

    def _sleep_seconds_after_tick(
        self, *, waiting_for_closed_bar: bool
    ) -> float:
        if not self.hour_aligned:
            return float(self.interval)

        now = _utc_now()
        if self.open_pos is not None and self.open_pos.status == "open":
            return float(self.interval)

        if waiting_for_closed_bar and _in_entry_retry_window(
            now, self.entry_poll_offset, self.entry_retry_window
        ):
            return float(self.entry_retry_interval)

        target = _next_entry_poll_at(now, self.entry_poll_offset)
        return max(1.0, (target - now).total_seconds())

    async def run_forever(self) -> None:
        if self.hour_aligned:
            offset = self.entry_poll_offset
            print(
                f"[{self.log_tag}] hour-aligned entry poll "
                f"(UTC :00+{offset}s, retry={self.entry_retry_interval}s)"
            )
            print(f"[{self.log_tag}] manage interval (open pos): {self.interval}s")
        else:
            print(f"[{self.log_tag}] starting loop interval={self.interval}s")
        print(f"[{self.log_tag}] strategy={self.strategy_path}")
        print(f"[{self.log_tag}] data_source={self.data_source}")
        print(
            f"[{self.log_tag}] telegram={'on' if self.notifier.enabled else 'off'} "
            f"(token_set={getattr(self.notifier, 'token_set', False)}, "
            f"chat_id_set={getattr(self.notifier, 'chat_id_set', False)})"
        )
        while True:
            waiting_for_closed_bar = False
            try:
                waiting_for_closed_bar = await self._tick()
                self.consecutive_failures = 0
            except Exception as exc:
                self.consecutive_failures += 1
                print(f"[{self.log_tag}] error: {exc}")
                await self.notifier.notify_error(str(exc))
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"[{self.log_tag}] circuit-breaker tripped")
                    await self.notifier.notify_error("Circuit breaker: too many failures")
                    return
            sleep_s = self._sleep_seconds_after_tick(
                waiting_for_closed_bar=waiting_for_closed_bar
            )
            if self.hour_aligned:
                wake_at = (_utc_now() + timedelta(seconds=sleep_s)).strftime(
                    "%H:%M:%S UTC"
                )
                print(f"[{self.log_tag}] next poll in {sleep_s:.0f}s (at {wake_at})")
            await asyncio.sleep(sleep_s)

    async def _tick(self) -> bool:
        """Run one poll. Returns True if flat and still waiting for a new closed bar."""
        cfg = _load_st_config(self.strategy_path)
        if self.rm is None:
            self._load_persisted_state(cfg)
        else:
            self.rm.cfg = cfg  # type: ignore[assignment]

        df = await self._fetch_ohlcv(cfg.asset)
        if df.empty or len(df) < 3:
            print(f"[{self.log_tag}] insufficient data from {self.data_source}")
            return self.open_pos is None

        df_ind = generate_signals(df, cfg)
        df_ind = df_ind.dropna(
            subset=["atr", "ema_slow_val", "supertrend", "st_dir"]
        ).copy()
        if len(df_ind) < 2:
            print(f"[{self.log_tag}] warmup not complete")
            return self.open_pos is None

        now = _utc_now()
        bar_delta = bar_timedelta(self.candle_interval)
        closed_idx = latest_closed_bar_index(df_ind.index, now, bar_delta)
        if closed_idx < 0:
            print(f"[{self.log_tag}] no closed bar yet (warmup/clock)")
            return self.open_pos is None

        last_bar_before = self._read_last_bar()
        closed = df_ind.iloc[closed_idx]
        closed_ts = str(closed.name)

        # Forming = bar after closed when present and still open; else reuse closed tip
        if closed_idx + 1 < len(df_ind):
            forming_idx = closed_idx + 1
            forming = df_ind.iloc[forming_idx]
        else:
            forming_idx = closed_idx
            forming = closed

        forming_ts = (
            float(forming.name.timestamp())
            if hasattr(forming.name, "timestamp")
            else time.time()
        )
        forming_price = float(forming["close"])
        forming_high = float(forming["high"])
        forming_low = float(forming["low"])

        assert self.rm is not None
        pending_trades: list[dict] = []
        tp1_before = self.open_pos.tp1_hit if self.open_pos else False

        # Manage open position every tick (intrabar SL/TP on tip candle)
        if self.open_pos is not None and self.open_pos.status == "open":
            manage_position(
                self.open_pos,
                forming,
                forming_idx,
                forming_ts,
                forming_price,
                forming_high,
                forming_low,
                cfg,
                self.rm,
                pending_trades,
            )
            if self.open_pos.status == "closed":
                await self.notifier.notify_close(self.open_pos, cfg)
                self._append_trade(self.open_pos)
                self.open_pos = None
            elif not tp1_before and self.open_pos.tp1_hit:
                await self.notifier.notify_tp1(self.open_pos, cfg)

        # New closed bar — evaluate entry once
        last_bar = self._read_last_bar()
        if closed_ts != last_bar:
            closed_ts_float = (
                float(closed.name.timestamp())
                if hasattr(closed.name, "timestamp")
                else forming_ts
            )
            closed_price = float(closed["close"])
            closed_high = float(closed["high"])
            closed_low = float(closed["low"])

            # If still open after tip manage, also run manage on closed bar
            if self.open_pos is not None and self.open_pos.status == "open":
                tp1_before_closed = self.open_pos.tp1_hit
                manage_position(
                    self.open_pos,
                    closed,
                    closed_idx,
                    closed_ts_float,
                    closed_price,
                    closed_high,
                    closed_low,
                    cfg,
                    self.rm,
                    pending_trades,
                )
                if self.open_pos.status == "closed":
                    await self.notifier.notify_close(self.open_pos, cfg)
                    self._append_trade(self.open_pos)
                    self.open_pos = None
                elif not tp1_before_closed and self.open_pos.tp1_hit:
                    await self.notifier.notify_tp1(self.open_pos, cfg)

            # Entry on newly closed bar
            if self.open_pos is None:
                allowed, reason = self.rm.can_trade(closed_ts_float)
                if allowed:
                    atr_val = float(closed.get("atr", 0))
                    if atr_val > 0:
                        direction: str | None = None
                        if closed.get("long_signal", 0) > 0:
                            direction = "long"
                        elif (
                            closed.get("short_signal", 0) > 0
                            and not cfg.long_only
                        ):
                            direction = "short"
                        if direction:
                            pos = self.rm.open_position(
                                closed_price,
                                atr_val,
                                direction,
                                closed_ts_float,
                                closed_idx,
                            )
                            if pos:
                                self.open_pos = pos
                                await self.notifier.notify_open(
                                    pos, cfg, self._config_version()
                                )
                                print(
                                    f"[{self.log_tag}] OPEN {direction} @ {closed_price:.2f} "
                                    f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}"
                                )
                elif reason:
                    print(f"[{self.log_tag}] entry blocked: {reason}")

            self._write_last_bar(closed_ts)
            print(
                f"[{self.log_tag}] processed closed bar {closed_ts} "
                f"(idx={closed_idx}/{len(df_ind)-1})"
            )

        self._save_persisted_state()
        self._write_heartbeat(forming, cfg)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        last_sent = self._read_last_heartbeat_day()
        if last_sent != today:
            self._write_last_heartbeat_day(today)
            await self.notifier.notify_heartbeat(
                {
                    "asset": cfg.asset_label or cfg.asset,
                    "price": forming_price,
                    "equity": self.rm.current_equity,
                    "open_side": (
                        self.open_pos.direction if self.open_pos else "none"
                    ),
                    "last_bar": str(forming.name),
                }
            )

        return (
            self.open_pos is None
            and closed_ts == (last_bar_before or "")
        )

    def _write_heartbeat(self, row: pd.Series, cfg: SupertrendVolumeConfig) -> None:
        hb = {
            "ts": time.time(),
            "mode": self.mode,
            "worker": "signal",
            "asset": cfg.asset,
            "price": float(row.get("close", 0)),
            "atr": float(row.get("atr", 0)),
            "st_dir": float(row.get("st_dir", 0)),
            "last_closed_bar": self._read_last_bar(),
            "open_position": (
                asdict(self.open_pos) if self.open_pos else None
            ),
            "equity": self.rm.current_equity if self.rm else cfg.account_size,
            "config_version": self._config_version(),
        }
        self._heartbeat_path.write_text(
            json.dumps(hb, indent=2, default=str)
        )
