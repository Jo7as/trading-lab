"""Telegram Bot API notifier for HF trading signals."""
from __future__ import annotations
from ..env import env

import os
from typing import Any

import httpx

from ..risk import Position
from ..strategies.supertrend_volume import SupertrendVolumeConfig


class TelegramNotifier:
    """Send formatted trade alerts via Telegram Bot API."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        *,
        enabled: bool = True,
    ):
        self.token = token or env("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or env("TELEGRAM_CHAT_ID", "")
        self.token_set = bool(self.token)
        self.chat_id_set = bool(self.chat_id)
        self.enabled = enabled and self.token_set and self.chat_id_set

    async def send(self, text: str) -> bool:
        if not self.enabled:
            print(f"[telegram] (disabled) {text[:120]}...")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            return True
        except Exception as exc:
            print(f"[telegram] send failed: {exc}")
            return False

    @staticmethod
    def _pct_move(entry: float, level: float, direction: str) -> float:
        if entry <= 0:
            return 0.0
        if direction == "long":
            return (level - entry) / entry * 100
        return (entry - level) / entry * 100

    def format_open(
        self, pos: Position, cfg: SupertrendVolumeConfig, version: str = "?"
    ) -> str:
        side = pos.direction.upper()
        emoji = "🟢" if pos.direction == "long" else "🔴"
        risk_usd = cfg.account_size * cfg.risk_per_trade
        sl_pct = self._pct_move(pos.entry_price, pos.stop_loss, pos.direction)
        label = cfg.asset_label or "XAUUSD"
        return (
            f"{emoji} OPEN {side} {label}\n"
            f"Entry:  {pos.entry_price:.2f}\n"
            f"SL:     {pos.stop_loss:.2f}  ({sl_pct:+.2f}%)\n"
            f"TP1:    {pos.tp1:.2f}  ({cfg.tp1_r:.1f}R, {cfg.tp1_close_pct:.0%} close)\n"
            f"Risk:   {cfg.risk_per_trade:.0%} (${risk_usd:,.0f} on ${cfg.account_size:,.0f})\n"
            f"Config: {version} | st{cfg.supertrend_period}/{cfg.supertrend_mult}"
        )

    def format_tp1(self, pos: Position, cfg: SupertrendVolumeConfig) -> str:
        label = cfg.asset_label or "XAUUSD"
        return (
            f"🎯 TP1 PARTIAL {pos.direction.upper()} {label}\n"
            f"TP1:    {pos.tp1:.2f}\n"
            f"Close:  {cfg.tp1_close_pct:.0%} of position\n"
            f"SL → breakeven ({pos.stop_loss:.2f})"
        )

    def format_close(self, pos: Position, cfg: SupertrendVolumeConfig) -> str:
        pnl = pos.pnl_pct or 0.0
        emoji = "✅" if pnl >= 0 else "❌"
        reason = pos.exit_reason or "close"
        label = cfg.asset_label or "XAUUSD"
        exit_px = pos.exit_price or 0.0
        return (
            f"{emoji} CLOSE {pos.direction.upper()} {label}\n"
            f"Exit:   {exit_px:.2f} ({reason})\n"
            f"PnL:    {pnl:+.2f}%"
        )

    async def notify_open(
        self, pos: Position, cfg: SupertrendVolumeConfig, version: str = "?"
    ) -> None:
        await self.send(self.format_open(pos, cfg, version))

    async def notify_tp1(self, pos: Position, cfg: SupertrendVolumeConfig) -> None:
        await self.send(self.format_tp1(pos, cfg))

    async def notify_close(self, pos: Position, cfg: SupertrendVolumeConfig) -> None:
        await self.send(self.format_close(pos, cfg))

    async def notify_heartbeat(self, stats: dict[str, Any]) -> None:
        await self.send(
            "📊 DAILY HEARTBEAT\n"
            f"Asset:   {stats.get('asset', '?')}\n"
            f"Price:   {stats.get('price', 0):.2f}\n"
            f"Equity:  ${stats.get('equity', 0):,.2f}\n"
            f"Open:    {stats.get('open_side', 'none')}\n"
            f"Bar:     {stats.get('last_bar', '?')}"
        )

    async def notify_error(self, message: str) -> None:
        await self.send(f"⚠️ SIGNAL ERROR\n{message}")
