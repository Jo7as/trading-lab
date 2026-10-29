"""Tests for the screening funnel (loose first, strict later, nothing dropped silently)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.screening import (
    S1_MIN_TRADES,
    Funnel,
    run_funnel,
    s1_explore,
    s2_cost,
    s3_robust,
    s4_significant,
    s5_oos,
)


def _variant(**kw) -> dict:
    base = {"n": 1000, "gross_mean": 0.10, "placebo_gross": 0.02,
            "gross_bp": 5.0, "cost_bp": 2.0,
            "pos_years": 6, "n_years": 8, "neighbour_positive": 1.0,
            "t_daily": 4.0, "placebo_edge": 0.05, "pbo": 0.1, "oos_pass": True}
    base.update(kw)
    return base


def test_s1_keeps_gross_positive_that_beats_placebo() -> None:
    df = pd.DataFrame([
        _variant(name="good"),
        _variant(name="worse_than_placebo", gross_mean=0.01, placebo_gross=0.05),
        _variant(name="negative", gross_mean=-0.05),
        _variant(name="too_few", n=S1_MIN_TRADES - 1),
    ])
    s = s1_explore(df)
    assert list(s.kept["name"]) == ["good"]
    assert s.n_in == 4 and len(s.dropped) == 3


def test_s1_does_not_look_at_cost() -> None:
    """Stage 1 must be blind to cost - that is the whole point of the ordering."""
    df = pd.DataFrame([_variant(gross_bp=1.0, cost_bp=99.0)])
    assert len(s1_explore(df).kept) == 1


def test_s2_only_then_applies_cost() -> None:
    df = pd.DataFrame([_variant(name="pays"), _variant(name="does_not",
                                                       gross_bp=1.0, cost_bp=2.0)])
    s = s2_cost(df)
    assert list(s.kept["name"]) == ["pays"]


def test_s3_needs_the_year_sign_and_the_neighbourhood() -> None:
    df = pd.DataFrame([
        _variant(name="robust"),
        _variant(name="one_year", pos_years=1),
        _variant(name="lone_spike", neighbour_positive=0.0),
    ])
    s = s3_robust(df)
    assert list(s.kept["name"]) == ["robust"]


def test_s4_uses_the_daily_t_and_not_the_per_trade_one() -> None:
    df = pd.DataFrame([_variant(name="ok"), _variant(name="weak_t", t_daily=2.0)])
    assert list(s4_significant(df).kept["name"]) == ["ok"]


def test_s5_is_a_binary_holdout() -> None:
    df = pd.DataFrame([_variant(name="passed"), _variant(name="failed", oos_pass=False)])
    assert list(s5_oos(df).kept["name"]) == ["passed"]


def test_funnel_runs_in_order_and_reports_every_cut() -> None:
    df = pd.DataFrame([
        _variant(name="full_pass"),
        _variant(name="dies_s1", gross_mean=-1.0),
        _variant(name="dies_s3", pos_years=0),
        _variant(name="dies_s5", oos_pass=False),
    ])
    # Default stoppt bei S4 (Holdout gesperrt) -> explizite Freigabe fuer den vollen Lauf
    f = run_funnel(df, up_to=None, holdout_freigabe="2026-10-26")
    assert isinstance(f, Funnel)
    assert list(f.survivors["name"]) == ["full_pass"]
    txt = f.summary()
    assert "S1 Exploration" in txt and "S5 Out-of-Sample" in txt
    assert "von" in txt
    # nothing is ever dropped silently: every stage keeps + drops = the input it saw
    for s in f.stages:
        assert len(s.kept) + len(s.dropped) == s.n_in


def test_funnel_stops_early_before_the_holdout() -> None:
    df = pd.DataFrame([_variant()])
    f = run_funnel(df, up_to="S3")
    assert [s.name.split()[0] for s in f.stages] == ["S1", "S2", "S3"]


def test_a_stage_that_cuts_nothing_is_flagged() -> None:
    """If S2 removes nothing, the row is already cost-filtered - worth saying out loud."""
    df = pd.DataFrame([_variant(gross_bp=50.0, cost_bp=1.0)])
    f = run_funnel(df, up_to="S2")
    assert "cuttet nichts" in f.summary()


# ── D1: die nachtraeglich ergaenzten Stufen-Bausteine ───────────────────────

def test_daily_series_collapses_trades_to_days() -> None:
    from trading_lab.screening import daily_series
    trades = [{"exit_time": f"2026-01-{d:02d}T12:00:00Z", "r": 0.5}
              for d in (1, 1, 2, 2, 2)]
    s = daily_series(trades)
    assert len(s) == 2
    assert s.iloc[0] == pytest.approx(1.0)     # zwei Trades am Tag 1
    assert s.iloc[1] == pytest.approx(1.5)     # drei am Tag 2


def test_t_stat_is_zero_for_a_single_observation() -> None:
    from trading_lab.screening import t_stat
    assert t_stat([1.0]) == 0.0
    assert t_stat([]) == 0.0


def test_daily_aggregation_does_not_add_power() -> None:
    """Re-binning to days keeps the t (it is the honest series, not a stronger one).

    The previous version of this test asserted ``t_daily > 2 * t_trade`` on one
    seed; across seeds that holds ~28 % of the time, and the means are equal.
    """
    from trading_lab.screening import t_stat
    td, tt = [], []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        per_trade = rng.normal(0.05, 4.0, 2000)
        tt.append(t_stat(per_trade))
        td.append(t_stat(per_trade.reshape(-1, 40).sum(axis=1)))
    assert abs(np.mean(td) - np.mean(tt)) < 0.15


def test_pbo_detects_a_real_edge() -> None:
    """Ein echtes Edge muss PBO gegen 0 treiben - sonst ist die Kennzahl nutzlos."""
    from trading_lab.screening import pbo
    rng = np.random.default_rng(3)
    m = pd.DataFrame(rng.normal(0.0, 1.0, size=(10, 640)))
    m.iloc[0] += 0.5                                  # eine Variante mit echtem Edge
    assert pbo(m) <= 0.10
    # und bei reinem Rauschen ist sie nicht vertrauenswuerdig -> nan bei zu wenig Daten
    small = pd.DataFrame(rng.normal(0.0, 1.0, size=(3, 640)))
    assert np.isnan(pbo(small))


def test_holdout_is_locked_without_a_release() -> None:
    """S5 darf nicht still laufen - der Holdout wird nur einmal verbraucht."""
    from trading_lab.screening import HOLDOUT_SPERRE, run_funnel
    df = pd.DataFrame([_variant()])
    f = run_funnel(df, up_to=None)                    # bis zum Ende angefragt
    last = f.stages[-1]
    assert last.name.startswith("S5")
    assert last.note == HOLDOUT_SPERRE
    assert len(last.dropped) == 0                     # nichts wurde verworfen


def test_holdout_runs_with_a_release() -> None:
    from trading_lab.screening import run_funnel
    df = pd.DataFrame([_variant(name="passed"), _variant(name="failed", oos_pass=False)])
    f = run_funnel(df, up_to=None, holdout_freigabe="2026-10-26")
    assert [s.name.split()[0] for s in f.stages] == ["S1", "S2", "S3", "S4", "S5"]
    assert list(f.survivors["name"]) == ["passed"]


def test_default_stops_before_the_holdout() -> None:
    from trading_lab.screening import run_funnel
    f = run_funnel(pd.DataFrame([_variant()]))
    assert [s.name.split()[0] for s in f.stages] == ["S1", "S2", "S3", "S4"]


# ── D2: Meta-Test - der Trichter darf weder zu streng noch zu locker sein ────

def _edges(n_variants: int, *, edge: bool, seed: int = 11) -> pd.DataFrame:
    """Varianten mit bekanntem Edge (oder reines Rauschen) durch den Trichter."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_variants):
        # der wahre Vorteil: mit Edge ~0.12 R, ohne Edge 0.0
        true_edge = 0.12 if edge else 0.0
        n = int(rng.integers(700, 1500))
        gross = true_edge + rng.normal(0, 0.01)          # je Trade, in R
        placebo = rng.normal(0, 0.01)                    # Zufallsrichtung, nur Exit-Skew
        # Die Tages-Reihe ist die Summe der Trades eines Tages. Der Backtest laeuft
        # ueber BACKTEST_DAYS Tage, also ist die Dichte n/BACKTEST_DAYS - NICHT n/40.
        # Vorher stand hier eine Dichte von 40 Trades/Tag (die ganze Historie in einem
        # Monat), was t_daily voellig verzerrt hat. An dieser Groesse haengt, was S4
        # ueberhaupt nachweisen kann: t_daily ~ edge * dichte * sqrt(Tage) / sqrt(dichte).
        BACKTEST_DAYS, SD_TRADE = 1500, 1.0            # 6 Jahre, sd 1 R je Trade
        per_day = n / BACKTEST_DAYS
        t_daily = ((true_edge * per_day) * np.sqrt(BACKTEST_DAYS)
                   / (SD_TRADE * np.sqrt(per_day)))
        rows.append({
            "name": f"v{i}",
            "n": n,
            "gross_mean": gross,
            "placebo_gross": placebo,
            "gross_bp": 4.0 + true_edge * 20,
            "cost_bp": 1.2,
            "pos_years": 6 if edge else int(rng.integers(2, 6)),
            "n_years": 8,
            "neighbour_positive": 1.0 if edge else 0.4,
            "t_daily": t_daily,
            "placebo_edge": gross - placebo,
            "pbo": 0.05 if edge else 0.6,
            "oos_pass": True,
        })
    return pd.DataFrame(rows)


def test_meta_a_real_edge_survives_the_funnel() -> None:
    """Der Kern des Auftrags: ein echter Edge darf NICHT an einer Prognose sterben."""
    from trading_lab.screening import run_funnel
    df = _edges(50, edge=True)
    f = run_funnel(df, up_to=None, holdout_freigabe="2026-10-26")
    kept = [len(s.kept) for s in f.stages]
    assert kept[0] == 50, "S1 hat einen echten Edge aussortiert"
    assert f.survivors.shape[0] >= 40, f"nur {f.survivors.shape[0]} ueberleben: {kept}"


def test_meta_pure_noise_dies_at_s1() -> None:
    """Und Rauschen darf nicht durchrutschen."""
    from trading_lab.screening import run_funnel
    df = _edges(50, edge=False, seed=5)
    f = run_funnel(df, up_to=None, holdout_freigabe="2026-10-26")
    assert len(f.stages[0].kept) < len(df), "S1 hat reines Rauschen durchgelassen"
    assert f.survivors.shape[0] < 10, f"zu viel Rauschen ueberlebt: {f.survivors.shape[0]}"


def test_meta_a_small_sample_is_flagged_not_killed_in_s4() -> None:
    """S4 urteilt auf dem Tages-t - ein kleineres Trade-n darf nicht der Grund sein."""
    from trading_lab.screening import run_funnel
    df = _edges(20, edge=True).copy()
    df["n"] = 150                      # klein, aber t_daily stark
    df["t_daily"] = 5.0
    f = run_funnel(df, up_to="S4")
    assert len(f.stages[-1].kept) == 20


def test_meta_low_power_lands_in_dropped_not_nowhere() -> None:
    """Zu wenig Daten heisst 'nicht nachweisbar', nicht 'verschwunden'.

    Das ist die Lehre aus dem Audit: die alten Karten starben an fehlender
    Teststaerke. Wer in S4 an einem zu kleinen Tages-t scheitert, muss in
    ``dropped`` auffindbar bleiben - sonst wiederholt sich genau das.
    """
    from trading_lab.screening import run_funnel
    df = _edges(30, edge=True).copy()
    df["n"] = 200                      # 200 Trades / 6 Jahre -> t_daily ~ 1.7
    df["t_daily"] = 1.7
    f = run_funnel(df, up_to="S4")
    s4 = f.stages[-1]
    assert len(s4.kept) == 0
    assert len(s4.dropped) == 30, "die schwach belegten Varianten fehlen im Report"


def test_meta_the_funnel_never_cuts_without_saying_so() -> None:
    """Jede Stufe muss ihre Schnitte berichten - kein stiller Ausfall."""
    from trading_lab.screening import run_funnel
    f = run_funnel(_edges(40, edge=True), up_to=None, holdout_freigabe="2026-10-26")
    for s in f.stages:
        assert len(s.kept) + len(s.dropped) == s.n_in
        assert s.criterion, f"{s.name} hat kein Kriterium benannt"


# ── skipped: was eine Stufe nicht beurteilen kann, wird NICHT verworfen ─────

def test_rows_without_values_are_skipped_not_dropped() -> None:
    """Der Kern der Audit-Lehre: fehlende Werte duerfen keine Zeile entfernen.

    Genau so sind die alten Karten gestorben - ein fehlender oder prognostizierter
    Wert hat sie aus dem Verfahren gekippt, statt sie weiterlaufen zu lassen.
    """
    df = pd.DataFrame([
        _variant(name="completes"),
        {"name": "no_values", "n": None, "gross_mean": None, "placebo_gross": None,
         "gross_bp": None, "cost_bp": None, "pos_years": None, "n_years": None,
         "neighbour_positive": None, "t_daily": None, "placebo_edge": None,
         "pbo": None, "oos_pass": None},
    ])
    f = run_funnel(df, up_to="S4")
    s1 = f.stages[0]
    assert list(s1.skipped["name"]) == ["no_values"]
    assert "no_values" not in list(s1.dropped["name"])
    # sie bleibt bis S4 sichtbar, wird dort aber nicht zum Ueberlebenden:
    # ohne Nachweis kein Ueberleben (Review 2026-10-26: diese Luecke liess eine
    # Variante ohne Placebo bis vor den Holdout laufen)
    assert "no_values" in list(f.stages[-1].skipped["name"])
    assert "no_values" not in list(f.survivors["name"])


def test_summary_reports_skipped_rows() -> None:
    df = pd.DataFrame([_variant(), {"name": "blank"}])
    txt = run_funnel(df, up_to="S1").summary()
    assert "nicht beurteilen" in txt


def test_stage_counts_add_up_with_skipped() -> None:
    df = pd.DataFrame([_variant(name="a"), _variant(name="b", gross_mean=-1.0),
                       {"name": "c"}])
    f = run_funnel(df, up_to="S2")
    for s in f.stages:
        assert len(s.kept) + len(s.dropped) + len(s.skipped) == s.n_in


# ── C2: turning a study's trade/event log into a funnel row ─────────────────

def test_daily_series_reads_unix_seconds() -> None:
    """The Position log stores exit_time as unix seconds; the day must not collapse to 1970."""
    from trading_lab.screening import daily_series
    t0 = 1_700_000_000                       # 2023-11-14 UTC
    trades = [{"exit_time": t0, "r": 1.0}, {"exit_time": t0 + 3600, "r": 2.0},
              {"exit_time": t0 + 90_000, "r": 3.0}]     # +25 h → next day
    s = daily_series(trades)
    assert len(s) == 2 and s.iloc[0] == 3.0 and s.iloc[1] == 3.0


def test_funnel_row_from_trades_computes_the_columns() -> None:
    from datetime import datetime, timedelta, timezone
    from trading_lab.screening import funnel_row_from_trades
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    trades = [{"exit_time": (base + timedelta(days=i)).isoformat(), "r": 1.0} for i in range(5)]
    trades += [{"exit_time": (base + timedelta(days=400 + i)).isoformat(), "r": -0.2}
               for i in range(5)]
    row = funnel_row_from_trades(trades, name="x", gross_bp=5.0, cost_bp=1.0)
    assert row["n"] == 10
    assert row["gross_mean"] == pytest.approx(0.4)
    assert row["n_years"] == 2 and row["pos_years"] == 1     # 2020 +5R, 2021 -1R
    assert row["t_daily"] is not None
    assert row["gross_bp"] == 5.0 and row["cost_bp"] == 1.0
    # values a log cannot prove stay None so the funnel SKIPS, never drops
    assert row["placebo_gross"] is None and row["pbo"] is None


def test_funnel_row_from_trades_leaves_uncomputable_fields_none() -> None:
    from trading_lab.screening import funnel_row_from_trades
    row = funnel_row_from_trades([{"r": 1.0}] * 20)          # no clock
    assert row["n"] == 20 and row["gross_mean"] == 1.0
    assert row["t_daily"] is None and row["pos_years"] is None and row["n_years"] is None


def test_funnel_row_from_trades_survives_s4() -> None:
    """A row built from a log with a real daily edge must clear S4, not be skipped."""
    from datetime import datetime, timedelta, timezone
    from trading_lab.screening import funnel_row_from_trades, run_funnel
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    trades = [{"exit_time": (base + timedelta(days=i)).isoformat(),
               "r": 0.5 + (0.05 if i % 2 else -0.05)} for i in range(200)]
    row = funnel_row_from_trades(trades, name="edge", gross_bp=50.0, cost_bp=1.0,
                                 placebo_gross=0.0)
    f = run_funnel(pd.DataFrame([_variant(**row)]), up_to="S4")
    assert list(f.stages[-1].kept["name"]) == ["edge"]


# ── Nachbesserungen nach dem Review 2026-10-26 ─────────────────────────────

def test_s1_drops_negative_gross_even_if_the_placebo_is_worse() -> None:
    df = pd.DataFrame([_variant(name="neg", gross_mean=-0.05, placebo_gross=-0.20)])
    assert len(s1_explore(df).kept) == 0


def test_s1_without_placebo_still_needs_positive_gross() -> None:
    df = pd.DataFrame([_variant(name="pos", placebo_gross=None),
                       _variant(name="neg", gross_mean=-0.01, placebo_gross=None)])
    assert list(s1_explore(df).kept["name"]) == ["pos"]


def test_s2_prefers_the_measured_net_result() -> None:
    df = pd.DataFrame([
        _variant(name="net_neg", net_mean=-0.02),            # bp sagt positiv, gemessen negativ
        _variant(name="net_pos", net_mean=0.03, gross_bp=None, cost_bp=None),
    ])
    assert list(s2_cost(df).kept["name"]) == ["net_pos"]


def test_required_t_grows_with_the_search() -> None:
    from trading_lab.screening import S4_MIN_T_DAILY, required_t
    assert required_t(1) == pytest.approx(1.645, abs=0.01)
    assert required_t(100) == pytest.approx(3.29, abs=0.02)
    assert required_t(1000) > required_t(100)
    assert required_t(None) == S4_MIN_T_DAILY


def test_s4_bar_depends_on_n_trials() -> None:
    df = pd.DataFrame([_variant(name="prereg", t_daily=2.0, n_trials=1),
                       _variant(name="searched", t_daily=2.0, n_trials=500)])
    assert list(s4_significant(df).kept["name"]) == ["prereg"]


def test_s4_uses_dsr_when_present() -> None:
    df = pd.DataFrame([_variant(name="a", t_daily=10.0, dsr=0.5),
                       _variant(name="b", t_daily=0.5, dsr=0.99)])
    assert list(s4_significant(df).kept["name"]) == ["b"]


def test_deflated_sharpe_punishes_the_best_of_many_noise_variants() -> None:
    from trading_lab.screening import deflated_sharpe
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, size=(200, 750))
    sr = noise.mean(axis=1) / noise.std(axis=1, ddof=1)
    best = noise[int(np.argmax(sr))]
    assert deflated_sharpe(best, n_trials=1) > 0.95           # naiv: "signifikant"
    assert deflated_sharpe(best, n_trials=200, trial_sharpes=sr) < 0.95
    edge = rng.normal(0.25, 1, 750)
    assert deflated_sharpe(edge, n_trials=200, trial_sharpes=sr) > 0.95


# ── literature track (agreed 2026-10-29) ────────────────────────────────────

def _lit_row(**kw) -> dict:
    row = {"name": "x", "n": 1600, "gross_mean": 0.15, "placebo_gross": 0.0, "net_mean": 0.09,
           "pos_years": 13, "n_years": 21, "neighbour_positive": 1.0, "t_daily": 2.37, "dsr": 0.906,
           "placebo_edge": 0.15, "pbo": float("nan"), "n_trials": 4,
           "track": "literatur", "reference_ok": True, "paper_base": True, "holding_days": 40.0}
    row.update(kw)
    return row


def test_literature_track_relaxes_s3_and_s4_only_for_the_paper_rule() -> None:
    import pandas as pd
    from trading_lab import screening as S
    ok = pd.DataFrame([_lit_row()])
    assert len(S.s3_robust(ok).kept) == 1 and len(S.s4_significant(ok).kept) == 1
    normal = pd.DataFrame([_lit_row(track=None)])
    assert len(S.s3_robust(normal).kept) == 0 and len(S.s4_significant(normal).kept) == 0
    no_ref = pd.DataFrame([_lit_row(reference_ok=False)])
    assert len(S.s3_robust(no_ref).kept) == 0 and len(S.s4_significant(no_ref).kept) == 0
    tuned = pd.DataFrame([_lit_row(paper_base=False)])                  # a variant, not the paper rule
    assert len(S.s4_significant(tuned).kept) == 0
    fast = pd.DataFrame([_lit_row(holding_days=0.2)])                   # intraday keeps 70 % years
    assert len(S.s3_robust(fast).kept) == 0


def test_literature_track_keeps_the_placebo_and_t_floor() -> None:
    import pandas as pd
    from trading_lab import screening as S
    assert len(S.s4_significant(pd.DataFrame([_lit_row(t_daily=1.9)])).kept) == 0
    assert len(S.s4_significant(pd.DataFrame([_lit_row(placebo_edge=-0.01)])).kept) == 0
    assert len(S.s3_robust(pd.DataFrame([_lit_row(pos_years=12)])).kept) == 0     # 57 % < 60 %
