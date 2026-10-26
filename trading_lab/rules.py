"""Plain-language trading rules for a concrete strategy variant.

A frozen portfolio must say, for a human, exactly when a trade is opened, where the
stop and the target sit and when it is closed - with the variant's real numbers, not
"see the code". ``describe(strategy, cfg)`` builds that text from the config; the
logic it paraphrases lives in ``trading_lab/strategies/<module>.py`` (whose SHA-256
is frozen next to it, see ``trading_lab/portfolio.py``).

Every text was written against the code on 2026-10-26; if a strategy module changes,
its hash changes, the portfolio lock fails and this text must be re-checked.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

#: strategy name in the registry -> source module that holds its logic
MODULES = {
    "momentum": "trading_lab/strategies/momentum.py",
    "pullback": "trading_lab/strategies/pullback.py",
    "orb_2r": "trading_lab/strategies/orb_2r.py",
    "mean_reversion": "trading_lab/strategies/mean_reversion.py",
    "session_range_fade": "trading_lab/strategies/session_range_fade.py",
    "bb_squeeze": "trading_lab/strategies/bb_squeeze_breakout.py",
    "supertrend": "trading_lab/strategies/supertrend_volume.py",
    "orb_classic": "trading_lab/strategies/opening_range_breakout.py",
}
#: shared code every variant depends on (fills, sizing, normalisation, costs)
SHARED = ("trading_lab/risk.py", "trading_lab/mr_common.py", "trading_lab/indicators.py",
          "trading_lab/optimizer.py", "trading_lab/registry.py")

_SESS = {"ny": "New York 09:30-15:55 (NY-Zeit)", "london": "London 08:00-16:30 (London-Zeit)",
         "both": "03:00-15:55 NY-Zeit"}


def _momentum(c: dict) -> dict:
    reg = {"none": "kein Trendfilter",
           "er": f"Efficiency Ratio({c['regime_lookback']}) >= {c['regime_min']}",
           "adx": f"ADX(14) >= {20 + 20 * c['regime_min']:.0f}",
           "ema_stack": f"EMA-Abstand + Efficiency Ratio >= {c['regime_min']}"}[c["regime"]]
    entry = {
        "breakout": f"Mid-Kurs steigt über das Hoch der letzten {c['entry_lookback']} {c['tf']}-Kerzen "
                    f"+ {c['entry_buffer_atr']} x ATR -> Long (Short spiegelbildlich unter das Tief); "
                    "Fill als Market-Order (Long zum Ask, Short zum Bid)",
        "momentum": f"Rendite der letzten {c['entry_lookback']} {c['tf']}-Kerzen positiv UND Mid über deren "
                    "Hoch -> Long (negativ und unter dem Tief -> Short); Fill zum Mid (ruhende Order)",
        "market": f"Vorzeichen der Rendite der letzten {c['entry_lookback']} {c['tf']}-Kerzen; Fill Market",
        "pullback": "Trend aus EMA schnell > EMA langsam, Einstieg wenn der Mid die schnelle EMA berührt; Fill zum Mid",
    }[c["entry"]]
    stop = {"atr": f"Einstieg -/+ {c['stop_atr']} x ATR(14, {c['tf']})",
            "swing": f"tiefstes Tief / höchstes Hoch der letzten {c['entry_lookback']} Minuten",
            "session_range": f"Einstieg -/+ max({c['stop_atr']} x ATR, {2 * c['stop_min_atr']} x ATR)"}[c["stop"]]
    exit_ = {"target": f"Ziel = Einstieg +/- {c['target_r']} x Risiko (R:R 1:{c['target_r']}); sonst Stop, "
                       f"Zeit ({c['max_hold_minutes']} min) oder Session-Ende",
             "time": f"KEIN festes Ziel: Ausstieg zum Mid nach {c['max_hold_minutes']} Minuten oder bei "
                     "Session-Ende, vorher nur über den Stop (target_r hat hier keine Wirkung)",
             "trail_atr": f"Trailing-Stop {c['trail_atr']} x ATR hinter dem besten Kurs; Zeit {c['max_hold_minutes']} min",
             "be_trail": f"Stop auf Einstand ab +{c['be_trigger_r']} R, danach Trailing {c['trail_atr']} x ATR",
             }[c["exit"]]
    return {
        "zeitrahmen": f"Signal auf {c['tf']}-Kerzen (nur abgeschlossene Kerzen, um eine Kerze verzögert), "
                      "Ausführung, Stop und Ziel auf 1-Minuten-Kerzen",
        "session": f"{_SESS[c['session']]}; Einstiege von Minute {c['trade_minute_start']} bis "
                   f"{c['trade_minute_end']} nach Session-Beginn; höchstens 1 Trade pro Tag",
        "filter": f"{reg}; Spread <= {c['spread_filter']} x Tagesmedian; Kosten <= {c['max_cost_r']} R; "
                  f"Stop mindestens {c['stop_min_atr']} x ATR",
        "entry": entry, "stop_loss": stop, "take_profit_und_ausstieg": exit_,
        "kosten": "Einstieg zum Mid (Market-/Breakout-Typen: Long zum Ask, Short zum Bid), Stop und Ziel "
                  "zum jeweiligen Kurslevel; je Trade wird ein voller gemessener Spread (beim Einstieg) "
                  "abgezogen. Stop gewinnt, wenn Stop und Ziel in derselben Minute liegen.",
    }


def _supertrend(c: dict) -> dict:
    sig = (f"Supertrend({c['supertrend_period']}, {c['supertrend_mult']}) dreht auf Long (bzw. Short) UND Schluss "
           f"über (unter) der Linie UND Volumen > Median({c['volume_ma_period']}) x {c['volume_mult']} UND "
           f"EMA({c['ema_fast']}) > EMA({c['ema_slow']}) (Short: <) UND ATR/Kurs > {c['min_atr_pct']}")
    if c["pullback_entry"]:
        sig += f"; zusätzlich Rücksetzer-Einstieg an die Linie (Toleranz {c['pullback_tolerance']})"
    return {
        "zeitrahmen": f"{c['timeframe']}-Kerzen",
        "filter": f"max. Tagesverlust {c['max_daily_loss'] * 100:.0f} %, Pause {c['cooldown_hours']} h nach Verlust"
                  + ("; nur Long" if c["long_only"] else ""),
        "entry": f"Signal am Kerzenschluss: {sig}. Einstieg zur ERÖFFNUNG der nächsten Kerze "
                 f"(+{c['slippage_pct'] * 1e4:.0f} bp Slippage).",
        "stop_loss": f"Einstieg -/+ {c['sl_atr_mult']} x ATR({c['atr_period']})",
        "take_profit_und_ausstieg": f"TP1 = Einstieg +/- {c['tp1_r']} x Stop-Abstand, dort {c['tp1_close_pct']:.0%} "
                                    "schließen und Stop auf Einstand; Rest wird an der Supertrend-Linie "
                                    "nachgezogen und bei Supertrend-Wechsel geschlossen; "
                                    f"Zeitstopp nach {c['max_hold_hours']} h",
        "kosten": "Engine rechnet nur auf dem Bid; der gemessene Spread wird je Trade vom Optimierer abgezogen",
    }


def _bb_squeeze(c: dict) -> dict:
    return {
        "zeitrahmen": "1h-Kerzen",
        "filter": (f"Trendfilter Schluss über/unter EMA({c['ema_trend_period']})" if c["use_trend_filter"]
                   else "kein Trendfilter")
                  + f"; Aktivität > Median({c['activity_period']}) x {c['activity_mult']}; Kerzenkörper >= "
                    f"{c['min_body_ratio']:.0%} der Spanne; Bandbreite >= {c['min_bb_bandwidth']}"
                  + (f"; nur {c['session_start_utc']}-{c['session_end_utc']} UTC" if c["session_filter"] else "")
                  + ("; nur Long" if c["long_only"] else ""),
        "entry": f"Vorherige Kerze im Squeeze (Bandbreite von BB({c['bb_period']}, {c['bb_std']}) im untersten "
                 f"{c['squeeze_threshold']:.0%} der letzten {c['squeeze_lookback']} Kerzen) UND Schluss über dem "
                 "oberen Band -> Long (unter dem unteren -> Short). Einstieg zum SCHLUSSKURS dieser Kerze.",
        "stop_loss": f"Einstieg -/+ {c['sl_atr_mult']} x ATR({c['atr_period']})",
        "take_profit_und_ausstieg": f"TP1 = Einstieg +/- {c['tp1_r']} x Stop-Abstand, dort {c['tp1_close_pct']:.0%} "
                                    "schließen und Stop auf Einstand"
                                    + ("; Ausstieg wenn der Schluss zurück ins Band fällt" if c["fail_back_inside_bands"] else "")
                                    + f"; Zeitstopp nach {c['max_hold_hours']} Kerzen",
        "kosten": "Engine rechnet nur auf dem Bid; der gemessene Spread wird je Trade vom Optimierer abgezogen",
    }


def _orb_classic(c: dict) -> dict:
    side = "nur Long" if c["long_only"] else ("nur Short" if c["short_only"] else "Long und Short")
    return {
        "zeitrahmen": f"{c['bar_minutes']}-Minuten-Kerzen, NY-Zeit",
        "filter": f"Tag auslassen, wenn Spanne/Mitte > {c['max_orb_pct']:.2%}; {side}; höchstens 1 Trade pro Tag",
        "entry": f"Opening Range = Hoch/Tief der ersten {c['orb_minutes']} Minuten ab {c['rth_open']}. "
                 f"Erster Kerzenschluss über Hoch + {c['breakout_buffer']} Punkte (Short: unter Tief - Puffer) "
                 f"vor {c['entry_cutoff']} -> Einstieg zum Schlusskurs dieser Kerze.",
        "stop_loss": f"gegenüberliegende Seite der Range -/+ {c['sl_buffer']} Punkte",
        "take_profit_und_ausstieg": f"Ziel = Einstieg +/- {c['tp_range_mult']} x Range-Breite"
                                    + (f", {c['tp1_close_pct']:.0%} dort schließen, Rest bis {c['tp2_range_mult']} x Breite"
                                       if c["tp2_range_mult"] > 0 and c["tp1_close_pct"] < 1 else " (ganze Position)")
                                    + f"; spätestens {c['eod_flat']} glattstellen",
        "kosten": "Engine rechnet nur auf dem Bid; der gemessene Spread wird je Trade vom Optimierer abgezogen",
    }


_DESCRIBE = {"momentum": _momentum, "supertrend": _supertrend, "bb_squeeze": _bb_squeeze,
             "orb_classic": _orb_classic}


def describe(strategy: str, cfg: Any) -> dict:
    """Rules of one variant in words. Raises for strategies without a text yet."""
    if strategy not in _DESCRIBE:
        raise KeyError(f"keine Regelbeschreibung für {strategy!r} - in trading_lab/rules.py ergänzen")
    out = _DESCRIBE[strategy](asdict(cfg))
    out["risiko_einheit"] = "1 R = Abstand Einstieg -> Stop; Positionsgröße so, dass 1 R = festes Risiko je Trade"
    return out
