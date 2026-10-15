"""Markov 2.0 Regime Filter — replaces EMA200 trend filter in MCTFB.

Classifies each candle into BULL / BEAR / SIDEWAYS regime using:
  1. Rolling cumulative return over N bars
  2. Threshold-based state labeling
  3. Hysteresis (state must persist N bars before confirming switch)

This is a FILTER, not a standalone strategy. It gates MCTFB entries:
  BULL     -> long entries allowed
  BEAR     -> short entries allowed
  SIDEWAYS -> no entries (chop protection)

No look-ahead bias: rolling window and hysteresis are backward-looking.
The transition matrix is computed once for analysis; the filter itself
only uses the state label at each candle.

Calibrated for Gold (GC=F) on 1h candles:
  - 24-bar window (~1 trading day)
  - ±1.5% thresholds (achievable in a day of strong directional move)
  - 3-bar hysteresis (prevents whipsaw)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# State encoding
SIDEWAYS = 0
BULL = 1
BEAR = 2


class MarkovRegimeFilter:
    """Regime detection filter using rolling return + hysteresis."""

    def __init__(
        self,
        window: int = 24,
        bull_threshold: float = 0.015,
        bear_threshold: float = -0.015,
        hysteresis: int = 3,
    ):
        self.window = window
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.hysteresis = hysteresis

    def _label_states(self, closes: pd.Series) -> np.ndarray:
        """Label each bar as BULL/BEAR/SIDEWAYS from rolling return."""
        rolling_ret = closes.pct_change(self.window)
        states = np.full(len(closes), SIDEWAYS, dtype=int)

        valid = rolling_ret.notna().values
        rets = rolling_ret.values

        states[valid & (rets >= self.bull_threshold)] = BULL
        states[valid & (rets <= self.bear_threshold)] = BEAR
        # NaN period stays SIDEWAYS

        return states

    def _apply_hysteresis(self, states: np.ndarray) -> np.ndarray:
        """State must persist N bars before confirming a switch.

        This is the single biggest improvement per the Markov 2.0 docs:
        cuts max drawdown dramatically by filtering whipsaws.
        """
        if len(states) == 0:
            return states

        confirmed = states[0]
        hold_count = 1
        result = states.copy()

        for i in range(1, len(states)):
            if states[i] == confirmed:
                hold_count += 1
            else:
                hold_count += 1
                if hold_count >= self.hysteresis:
                    confirmed = states[i]
                    hold_count = 1
                else:
                    result[i] = confirmed

        return result

    def compute_regimes(self, closes: pd.Series) -> pd.Series:
        """Compute regime labels for the full series (vectorized, no look-ahead).

        Returns a pd.Series of ints: 0=SIDEWAYS, 1=BULL, 2=BEAR.
        """
        raw_states = self._label_states(closes)
        final_states = self._apply_hysteresis(raw_states)
        return pd.Series(final_states, index=closes.index, name="regime")

    def get_current_regime(self, closes: pd.Series) -> int:
        """Get the regime label for the last bar (for live mode)."""
        if len(closes) < self.window + self.hysteresis:
            return SIDEWAYS
        regimes = self.compute_regimes(closes)
        return int(regimes.iloc[-1])

    @staticmethod
    def build_transition_matrix(states: np.ndarray, stride: int | None = None) -> np.ndarray:
        """Build transition probability matrix (3x3, rows sum to 1).

        If stride=None: overlapping (every bar is a transition).
        If stride=N: non-overlapping (stride-sampled, statistically honest).
        """
        if stride is not None:
            sampled = states[::stride]
        else:
            sampled = states

        transitions = np.zeros((3, 3), dtype=int)
        for i in range(len(sampled) - 1):
            transitions[sampled[i]][sampled[i + 1]] += 1

        prob = np.zeros((3, 3), dtype=float)
        for i in range(3):
            row_sum = transitions[i].sum()
            prob[i] = transitions[i] / row_sum if row_sum > 0 else np.ones(3) / 3

        return prob

    def regime_stats(self, closes: pd.Series) -> dict:
        """Return regime distribution + transition matrix for analysis."""
        regimes = self.compute_regimes(closes)
        states = regimes.values

        counts = {
            "SIDEWAYS": int((states == SIDEWAYS).sum()),
            "BULL": int((states == BULL).sum()),
            "BEAR": int((states == BEAR).sum()),
        }
        total = len(states)
        distribution = {k: round(v / total * 100, 1) for k, v in counts.items()} if total > 0 else counts

        matrix = self.build_transition_matrix(states, stride=self.window)

        return {
            "counts": counts,
            "distribution_pct": distribution,
            "transition_matrix_stride": matrix.tolist(),
            "stickiness": {
                "SIDEWAYS": round(float(matrix[0][0]), 3),
                "BULL": round(float(matrix[1][1]), 3),
                "BEAR": round(float(matrix[2][2]), 3),
            },
            "current_regime": int(regimes.iloc[-1]) if len(regimes) > 0 else SIDEWAYS,
        }


def regime_to_trend(regimes: pd.Series) -> pd.DataFrame:
    """Convert regime labels to trend_up/trend_down booleans for MCTFB.

    BULL     -> trend_up=True, trend_down=False
    BEAR     -> trend_up=False, trend_down=True
    SIDEWAYS -> trend_up=False, trend_down=False (no trades)
    """
    trend_up = (regimes == BULL).astype(float)
    trend_down = (regimes == BEAR).astype(float)
    return pd.DataFrame({
        "trend_up": trend_up,
        "trend_down": trend_down,
    }, index=regimes.index)
