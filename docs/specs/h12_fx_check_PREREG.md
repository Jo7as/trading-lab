# H12 FX check: the confirmed combination on FX pairs - PRE-REGISTRATION

Written 2026-10-30 at the user's request, before any FX outcome was computed. Measurement only.
Code: `studies/h12_fx_check.py`. The candidate comes from `docs/specs/h12_pattern_mining_PREREG.md`:
5-min `cca` family + 1-h trend up -> down (5 fixed variants). FX took no part in the mining, so
this is an out-of-sample test on new markets. Nothing is re-mined or tuned.

* **Pairs (fixed):** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD (the majors `data_audit.SESSIONS`
  knows; USDCHF is not in it). FX has no finished data audit; `build_study_frame` is applied and its
  drop share reported.
* **Years:** fit 2019-2020 (move threshold, size / volume terciles, volatility quintiles, per pair, no
  outcome of the signal used), **test 2021-2024**. Holdout 2025+ closed.
* Same definitions as H12 track A: decision grid, 60-min first passage, control per test period
  (pair x clock x last-1-h range quintile).
* **Primary:** directional excess `(y_down - y_up) - (p_down - p_up)` pooled over the 6 pairs.
  **Passes iff excess > 0, day-clustered z >= 2.0, excess > 0 in >= 4 of 6 pairs.**
* **Secondary, no weight:** the mirror (5-min `aac` family + 1-h trend down -> up), because the sign
  of an FX pair is arbitrary and the original pattern came from assets with a down-move asymmetry.
* Cost, reported: FTMO profile - median liquid-hours spread + $2.50/lot/side, in ATR1h.
