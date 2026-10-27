"""Geführte Strategie-Prüfung: ein fester Ablauf, bei dem man jeden Schritt sieht.

Warum es das gibt: ein Skript, das "einfach durchläuft", liefert am Ende eine Zahl,
und niemand weiß, welche Entscheidung sie erzeugt hat. Hier läuft **jede** Strategie
durch dieselben 12 Schritte, in derselben Reihenfolge. Nach jedem Schritt:

* was gerade passiert ist (in einfachen Worten),
* das Zwischenergebnis (Tabelle),
* was es bedeutet,
* und - an den Stellen, an denen es etwas zu entscheiden gibt - eine Rückfrage.

Alles wird mitgeschrieben (``artifacts/pruefung/<zeit>_<strategie>_<instrument>/``):
jedes Zwischenergebnis als JSON, jede Entscheidung im Protokoll, am Ende ``bericht.md``.

Der Holdout (die Jahre ab 2025, die die Suche nie sieht) wird **nur** nach einer
ausdrücklichen Bestätigung angeschaut, nur für Varianten, die S1-S4 bestanden haben,
und jeder Blick wird in ``artifacts/pruefung/holdout_ledger.json`` vermerkt - ein
zweiter Blick auf denselben Holdout ist kein unabhängiger Test mehr, und das sagt
der Ablauf dann auch.

Aufruf:

    uv run python studies/pruefen.py                       # fragt alles ab
    uv run python studies/pruefen.py --strategie momentum --instrument US100

Die Logik steckt in :class:`GuidedCheck`; Ein-/Ausgabe laufen über ``say``/``ask``,
damit der Ablauf testbar ist (``tests/test_guided.py``).
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import screening, strategy_spec
from .optimizer import (MIN_TRADES, build_rows, evaluate_holdout, grid_size, run_variant,
                        per_year, prop_profile, search, shortlist)
from .registry import (ASSETS, BUDGET_SCALE, DESCRIPTIONS, HOLDOUT_START, NOT_ADAPTED,
                       REGISTRY, SHORTLIST, load_cached)

OUT_ROOT = Path("artifacts/pruefung")
LEDGER = OUT_ROOT / "holdout_ledger.json"
BUDGETS = {"s": ("schnell", 40), "n": ("normal", 150), "g": ("gruendlich", 400)}

#: The fixed sequence. Every run, every strategy: these steps, this order.
STEPS = (
    "Umgebung prüfen",
    "Strategie und Instrument wählen",
    "Daten laden und aufteilen",
    "Basis-Backtest mit Standard-Parametern",
    "Suchraum und Suchbudget festlegen",
    "Optimierung auf den Trainingsjahren",
    "Stufe S1 - Brutto und Placebo",
    "Stufe S2 - Kosten",
    "Stufe S3 - Robustheit",
    "Stufe S4 - Signifikanz",
    "Stufe S5 - Holdout (einmalig)",
    "Prop-Firm-Einschätzung und Bericht",
)


class Abbruch(Exception):
    """The user stopped the run; the protocol so far is still written."""


def _console_ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


@dataclass
class GuidedCheck:
    strategy: str | None = None
    asset: str | None = None
    budget: int | None = None
    say: Callable[[str], None] = print
    ask: Callable[[str], str] = _console_ask
    out_root: Path = OUT_ROOT
    loader: Callable[[str], pd.DataFrame] = load_cached
    holdout_start: pd.Timestamp = HOLDOUT_START
    cost_profile: str | None = None
    #: None = ask; True = continue without an agreed spec (report marked); False = abort
    allow_unchecked: bool | None = None
    #: decisions and results, in order - becomes the protocol in the report
    log: list[dict] = field(default_factory=list)

    # ── small I/O helpers ────────────────────────────────────────────────
    def _header(self, k: int) -> None:
        self.say("")
        self.say("=" * 78)
        self.say(f"  SCHRITT {k}/{len(STEPS)}: {STEPS[k - 1]}")
        self.say("=" * 78)

    def _explain(self, text: str) -> None:
        for line in text.strip().splitlines():
            self.say("  " + line.strip())

    def _pause(self, what: str = "weiter") -> None:
        a = self.ask(f"\n  [Enter] = {what},  a = abbrechen > ").strip().lower()
        if a == "a":
            raise Abbruch()

    def _choose(self, question: str, options: dict[str, str], default: str) -> str:
        opts = "  ".join(f"{k} = {v}" for k, v in options.items())
        while True:
            a = self.ask(f"\n  {question}\n  {opts}   [Enter = {default}] > ").strip().lower()
            if a == "":
                return default
            if a == "a":
                raise Abbruch()
            if a in options:
                return a
            self.say(f"  Bitte eine der Optionen eingeben: {', '.join(options)}")

    def _record(self, step: int, **data: Any) -> None:
        entry = {"schritt": step, "name": STEPS[step - 1], "zeit": datetime.now().isoformat(timespec="seconds"), **data}
        self.log.append(entry)
        if getattr(self, "run_dir", None) is not None:
            (self.run_dir / f"schritt_{step:02d}.json").write_text(json.dumps(entry, indent=1, default=str))

    # ── the run ──────────────────────────────────────────────────────────
    def run(self) -> Path | None:
        self.run_dir = None
        try:
            self.step1_environment()
            self.step2_choose()
            self.step3_data()
            self.step4_baseline()
            self.step5_space()
            self.step6_optimise()
            self.step7_s1()
            self.step8_s2()
            self.step9_s3()
            self.step10_s4()
            self.step11_holdout()
            return self.step12_report()
        except Abbruch:
            self.say("\n  Abgebrochen. Das Protokoll bis hierhin wird gespeichert.")
            return self._write_report(aborted=True)

    # 1 ─────────────────────────────────────────────────────────────────
    def step1_environment(self) -> None:
        self._header(1)
        self._explain(f"""
        Ziel des Ablaufs: herausfinden, ob eine Strategie einen echten Vorteil (Edge) hat -
        nicht, ob sie im Rückblick gut aussieht. Dafür läuft jede Strategie durch dieselben
        {len(STEPS)} Schritte: erst locker (Brutto), dann immer strenger (Kosten, Robustheit,
        Signifikanz, Holdout). Eine Strategie scheitert nur an einem ERGEBNIS, nie an einer
        Prognose. Du siehst nach jedem Schritt, was passiert ist, und entscheidest mit.
        """)
        py = sys.version.split()[0]
        venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        self.say(f"\n  Python {py}   virtuelle Umgebung aktiv: {'ja' if venv else 'NEIN'}")
        if not venv:
            self._explain("""
            Hinweis: ohne aktivierte .venv fehlen oft Pakete. Starte besser mit
            'uv run python studies/pruefen.py' (uv sync legt die .venv an).
            """)
        self._record(1, python=py, venv=venv)
        self._pause()

    # 2 ─────────────────────────────────────────────────────────────────
    def step2_choose(self) -> None:
        self._header(2)
        names = list(REGISTRY)
        if self.strategy is None:
            self.say("  Verfügbare Strategien:")
            for i, n in enumerate(names, 1):
                self.say(f"   {i}. {n:20s} {DESCRIPTIONS.get(n, '')[:90]}")
            if NOT_ADAPTED:
                self.say("\n  (nicht im Ablauf, mit Grund: " + ", ".join(NOT_ADAPTED) + ")")
            opts = {str(i): n for i, n in enumerate(names, 1)}
            self.strategy = opts[self._choose("Welche Strategie?", opts, "1")]
        if self.strategy not in REGISTRY:
            raise ValueError(f"unbekannte Strategie {self.strategy!r}; bekannt: {names}")
        assets = REGISTRY[self.strategy][1]
        if self.asset is None:
            opts = {str(i): a for i, a in enumerate(assets, 1)}
            self.asset = opts[self._choose(f"Instrument für {self.strategy}?", opts, "1")]
        self.spec = REGISTRY[self.strategy][0](self.asset)
        self.say(f"\n  Gewählt: {self.strategy} auf {self.asset}")
        self._explain(DESCRIPTIONS.get(self.strategy, ""))
        # Pflicht vor jedem Test: eine abgestimmte Spezifikation (docs/specs/<strategie>.yaml).
        # Die Pipeline prüft nur, was man ihr gibt - eine schlecht definierte Strategie
        # scheitert zu Recht und beweist trotzdem nichts über die Idee (H-012).
        self.spec_check = strategy_spec.load(self.strategy)
        if self.spec_check.agreed:
            self.say(f"\n  Spezifikation: abgestimmt ({self.spec_check.path})")
        else:
            self.say(f"\n  SPEZIFIKATION FEHLT ODER IST NICHT ABGESTIMMT:")
            for err in self.spec_check.errors[:12]:
                self.say(f"   - {err}")
            self._explain(f"""
            Ohne abgestimmte Spezifikation (Entry, Stop-Loss, Take-Profit, Haltedauer, Kosten,
            Referenz, Tests - Vorlage docs/specs/TEMPLATE.yaml) ist ein Ergebnis keine Aussage
            über die Idee. Weiter geht es nur bewusst; der Bericht trägt dann '{strategy_spec.UNCHECKED}'.
            """)
            if self.allow_unchecked is None:
                a = self._choose("Trotzdem weiter?", {"a": "abbrechen", "w": f"weiter als {strategy_spec.UNCHECKED}"}, "a")
                self.allow_unchecked = a == "w"
            if not self.allow_unchecked:
                raise Abbruch()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.out_root / f"{stamp}_{self.strategy}_{self.asset}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._record(2, strategie=self.strategy, instrument=self.asset,
                     spezifikation=("abgestimmt" if self.spec_check.agreed else strategy_spec.UNCHECKED),
                     spec_fehler=self.spec_check.errors)
        self._pause()

    # 3 ─────────────────────────────────────────────────────────────────
    def step3_data(self) -> None:
        self._header(3)
        self._explain(f"""
        Die Daten sind 1-Minuten-Kerzen mit Geld- UND Briefkurs (Bid/Ask), damit Kosten echt
        sind und nicht geschätzt. Sie werden in zwei Teile geteilt:
          * TRAIN (bis {self.holdout_start.date()}): hier wird gesucht und geprüft.
          * HOLDOUT (ab {self.holdout_start.date()}): bleibt verschlossen, bis eine Variante
            alle Stufen bestanden hat. Er ist der eine ehrliche Test am Ende.
        """)
        t0 = time.time()
        m1 = self.loader(self.asset)
        if m1.empty:
            self._explain("""
            Für dieses Instrument liegen keine Daten im Cache (artifacts/cache/m1).
            Der Download von Dukascopy dauert je nach Instrument 10-60 Minuten, weil der
            Server Massenabfragen bremst. Er passiert nur einmal, danach liegt alles lokal.
            """)
            if self._choose("Jetzt herunterladen?", {"j": "ja", "n": "nein, abbrechen"}, "n") != "j":
                raise Abbruch()
            from .minute_data import load_minute_bid_ask
            from .registry import YEAR_START
            load_minute_bid_ask(self.asset, YEAR_START)
            m1 = self.loader(self.asset)
            if m1.empty:
                raise Abbruch()
        from . import costs
        if self.cost_profile is None:
            self._explain("""
            Kosten: Mit welchen Konditionen soll gerechnet werden? Das ändert den Spread
            (bei manchen Profilen auf die typischen Werte des Handelsplatzes skaliert) und die
            Kommission je Trade. 'ftmo' ist der Standard, weil das Ziel ein Prop-Konto ist.
            """)
            opts = {str(i): n for i, n in enumerate(costs.PROFILES, 1)}
            for k, n in opts.items():
                self.say(f"   {k}. {n:10s} {costs.PROFILES[n].description}")
            default = next(k for k, n in opts.items() if n == costs.DEFAULT_PROFILE)
            self.cost_profile = opts[self._choose("Kostenprofil?", opts, default)]
        self.spec.cost_profile = self.cost_profile
        raw_spread = (m1["ask_close"] - m1["close"]) / m1["close"] * 1e4
        m1 = costs.adjust_frame(m1, self.asset, self.cost_profile)
        self.m1 = m1
        self.m1_train = m1[m1.index < self.holdout_start]
        spread = (m1["ask_close"] - m1["close"]) / m1["close"] * 1e4
        info = {
            "von": str(m1.index[0].date()), "bis": str(m1.index[-1].date()),
            "kerzen": int(len(m1)), "train_kerzen": int(len(self.m1_train)),
            "holdout_kerzen": int(len(m1) - len(self.m1_train)),
            "spread_median_bp_daten": round(float(raw_spread.median()), 2),
            "spread_median_bp": round(float(spread.median()), 2),
            "kostenprofil": self.cost_profile,
            "kommission_bp_je_seite": round(costs.commission_bp_side(self.asset, float(m1["close"].iloc[-1]),
                                                                   self.cost_profile), 3),
        }
        self.say("")
        for k, v in info.items():
            self.say(f"  {k:18s} {v}")
        self._explain(f"""
        Ein Round Trip kostet hier etwa {info['spread_median_bp']} bp Spread (Median) plus
        2 x {info['kommission_bp_je_seite']} bp Kommission. Das zahlt jeder Trade mindestens - je kleiner
        die erwartete Bewegung pro Trade, desto mehr frisst es.
        (geladen in {time.time() - t0:.0f}s)
        """)
        self._record(3, **info)
        self._pause()

    # 4 ─────────────────────────────────────────────────────────────────
    def step4_baseline(self) -> None:
        self._header(4)
        self._explain("""
        Erst einmal die Strategie so, wie sie im Code steht (Standard-Parameter), auf den
        Trainingsjahren. Das ist der Ausgangspunkt - damit man später sieht, was die
        Optimierung überhaupt verändert hat.
        Einheit "R": 1 R = das Risiko eines Trades (Abstand Einstieg -> Stop). +0,1 R je Trade
        heißt: im Schnitt gewinnt jeder Trade ein Zehntel dessen, was er riskiert.
        """)
        t0 = time.time()
        tr = run_variant(self.spec, self.spec.base, self.m1_train, adjusted=True)
        self.baseline = tr
        if tr.empty:
            self.say("\n  Keine Trades mit den Standard-Parametern.")
            self._record(4, n=0)
            self._pause()
            return
        yrs = per_year(tr)
        res = {"n": int(len(tr)), "brutto_R": round(float(tr["r_gross"].mean()), 3),
               "netto_R": round(float(tr["r"].mean()), 3),
               "trefferquote": round(float((tr["r"] > 0).mean()), 3),
               "netto_R_summe": round(float(tr["r"].sum()), 1),
               "jahre_positiv": f"{int((yrs > 0).sum())}/{len(yrs)}",
               "sekunden": round(time.time() - t0, 1)}
        self.say("")
        for k, v in res.items():
            self.say(f"  {k:18s} {v}")
        self.say("\n  Netto-R je Jahr: " + "  ".join(f"{y}: {v:+.1f}" for y, v in yrs.items()))
        self._explain("""
        "brutto" = vor Spread, "netto" = mit echtem Bid/Ask. Ist schon brutto nichts da,
        kann keine Kostenoptimierung das retten - ist brutto etwas da, lohnt die Suche.
        """)
        self._record(4, **res, netto_je_jahr={int(k): round(float(v), 2) for k, v in yrs.items()})
        self._pause()

    # 5 ─────────────────────────────────────────────────────────────────
    def step5_space(self) -> None:
        self._header(5)
        grid = dict(self.spec.grid)
        self._explain("""
        Die Optimierung probiert Kombinationen dieser Einstellungen (Entry, Filter, Stop,
        Ziel, Haltedauer, Session). Wichtig: JEDE ausprobierte Kombination wird gezählt.
        Wer 500 Varianten probiert, findet immer eine, die zufällig gut aussieht - deshalb
        wird die Signifikanz-Hürde in S4 umso höher, je mehr wirklich verschiedene Varianten
        getestet wurden. Breit suchen ist erlaubt, es kostet nur Beweiskraft.
        """)
        self.say("")
        for p, v in grid.items():
            self.say(f"  {p:24s} {', '.join(map(str, v))}")
        self.say(f"\n  Kombinationen insgesamt: {grid_size(grid):,}".replace(",", "."))
        if self._choose("Einen Parameter fest einstellen (verkleinert die Suche)?",
                        {"n": "nein, so übernehmen", "j": "ja"}, "n") == "j":
            while True:
                a = self.ask("  parameter=wert (leer = fertig) > ").strip()
                if not a:
                    break
                if "=" not in a or a.split("=")[0].strip() not in grid:
                    self.say("  Format: name=wert, name aus der Liste oben")
                    continue
                k, v = (x.strip() for x in a.split("=", 1))
                match = [x for x in grid[k] if str(x) == v]
                if not match:
                    self.say(f"  {v} ist kein Wert von {k}: {grid[k]}")
                    continue
                grid[k] = (match[0],)
                self.say(f"  {k} fest auf {match[0]}")
            self.spec.grid = grid
            self.spec.base = replace(self.spec.base, **{k: v[0] for k, v in grid.items() if len(v) == 1})
        if self.budget is None:
            scale = BUDGET_SCALE.get(self.strategy, 1.0)
            opts = {k: f"{name} (~{max(20, int(b * scale))} Varianten)" for k, (name, b) in BUDGETS.items()}
            b = self._choose("Wie gründlich suchen?", opts, "n")
            self.budget = max(20, int(BUDGETS[b][1] * scale))
        self._record(5, grid={k: list(v) for k, v in self.spec.grid.items()}, budget=self.budget)

    # 6 ─────────────────────────────────────────────────────────────────
    def step6_optimise(self) -> None:
        self._header(6)
        self._explain(f"""
        Die Suche läuft NUR auf den Trainingsjahren. Ziel ist nicht das höchste R:R und nicht
        der höchste Gewinn, sondern der höchste Erwartungswert je Trade NACH Kosten, abzüglich
        seiner Unsicherheit (untere Konfidenzgrenze). So gewinnt eine solide Variante mit
        +0,15 R aus 800 Trades gegen eine Glückstreffer-Variante mit +0,40 R aus 30 Trades.
        Das R:R ist dabei nur ein Rädchen (Ziel- und Stop-Parameter), kein Ziel.
        Budget: {self.budget} Varianten. Danach werden für die besten Varianten auch die
        Nachbar-Einstellungen gerechnet (für die Robustheit in S3).
        """)
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if time.time() - last[0] > 5:
                last[0] = time.time()
                self.say(f"    ... {done} Varianten gerechnet")

        t0 = time.time()
        s = search(self.spec, self.m1_train, budget=self.budget, verbose=False, progress=progress)
        top_net, top_gross = SHORTLIST.get(self.strategy, (6, 3))
        short = shortlist(s, top_net=top_net, top_gross=top_gross)
        self.rows, self.stats = build_rows(s, short)
        self.search_ = s
        self.say(f"\n  {len(s.trials)} Einstellungen gerechnet, davon {self.stats['n_trials']} mit "
                 f"unterschiedlichem Ergebnis, effektiv unabhängig: {self.stats['n_eff']:.0f} "
                 f"({time.time() - t0:.0f}s)")
        df = self._table(self.rows)
        self.say("\n  Die besten Varianten (Train, netto):\n")
        self.say(df[["#", "n", "brutto_R", "netto_R", "LCB", "Parameter"]].head(8).to_string(index=False))
        base_n = float(self.baseline["r"].mean()) if len(self.baseline) else float("nan")
        self._explain(f"""
        Zum Vergleich der Basis-Backtest: {base_n:+.3f} R netto je Trade.
        Ab jetzt prüft der Trichter diese Varianten Stufe für Stufe. Eine Variante, die eine
        Stufe nicht schafft, bleibt im Bericht sichtbar - mit dem Grund.
        """)
        self._record(6, varianten=len(s.trials), n_trials=self.stats["n_trials"],
                     n_eff=self.stats["n_eff"], pbo=self.stats["pbo"], shortlist=self.rows)
        self._pause()

    def _table(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["#"] = df["name"].str.split("#").str[-1]
        df["brutto_R"] = df["gross_mean"].round(3)
        df["netto_R"] = df["net_mean"].round(3)
        df["LCB"] = df["objective_lcb"].round(3)
        base = {k: v for k, v in self._asdict(self.spec.base).items() if k in self.spec.grid}
        df["Parameter"] = df["params"].map(
            lambda p: ", ".join(f"{k}={v}" for k, v in p.items() if base.get(k) != v)[:70] or "(Standard)")
        return df

    @staticmethod
    def _asdict(cfg: Any) -> dict:
        from dataclasses import asdict
        return asdict(cfg)

    @staticmethod
    def _reason(tag: str, r: pd.Series) -> str:
        """Why a row failed a stage - in words, one item per broken criterion."""
        def num(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                return None
        out = []
        if tag == "S1":
            if (num(r.get("n")) or 0) < screening.S1_MIN_TRADES:
                out.append(f"zu wenige Trades (<{screening.S1_MIN_TRADES})")
            g, pl = num(r.get("gross_mean")), num(r.get("placebo_gross"))
            if g is not None and g <= 0:
                out.append("brutto <= 0")
            if g is not None and pl is not None and g <= pl:
                out.append("nicht besser als Placebo")
        elif tag == "S2":
            if (num(r.get("net_mean")) or 0) <= 0:
                out.append("netto <= 0 (Kosten fressen den Vorteil)")
        elif tag == "S3":
            py, ny = num(r.get("pos_years")), num(r.get("n_years"))
            if py is not None and ny and py / ny < screening.S3_MIN_POS_YEARS:
                out.append(f"nur {int(py)}/{int(ny)} Jahre positiv")
            nb = num(r.get("neighbour_positive"))
            if nb is not None and nb < screening.S3_NEIGHBOUR_MIN:
                out.append(f"Nachbarn nur {nb:.0%} positiv (Spitze)")
        elif tag == "S4":
            d = num(r.get("dsr"))
            if d is not None and d < screening.S4_MIN_DSR:
                out.append(f"DSR {d:.2f} < {screening.S4_MIN_DSR}")
            pe = num(r.get("placebo_edge"))
            if pe is not None and pe <= screening.S4_MIN_PLACEBO_EDGE:
                out.append("kein Placebo-Vorsprung")
            pb = num(r.get("pbo"))
            if pb is not None and pb > screening.S4_MAX_PBO:
                out.append(f"PBO {pb:.2f} > {screening.S4_MAX_PBO}")
        return "; ".join(out) or "-"

    def _stage(self, k: int, fn, df: pd.DataFrame, text_before: str, cols: list[str],
               text_after: Callable[[screening.Stage], str]) -> pd.DataFrame:
        self._header(k)
        self._explain(text_before)
        st = fn(df)
        self.say(f"\n  {st.summary_line().strip()}")
        show = pd.concat([st.kept.assign(Ergebnis="bestanden"), st.dropped.assign(Ergebnis="raus"),
                          st.skipped.assign(Ergebnis="nicht beurteilbar")])
        if len(show):
            t = self._table(show.to_dict("records")).assign(Ergebnis=show["Ergebnis"].to_numpy())
            for c in cols:
                if c not in t:
                    t[c] = show[c].to_numpy() if c in show else np.nan
            tag = st.name.split()[0]
            t["Grund"] = [("" if e == "bestanden" else ("Wert fehlt" if e == "nicht beurteilbar"
                           else self._reason(tag, row)))
                          for e, (_, row) in zip(show["Ergebnis"], show.iterrows())]
            self.say("")
            self.say(t[["#"] + cols + ["Ergebnis", "Grund"]].round(3).to_string(index=False))
        self._explain(text_after(st))
        self._record(k, bestanden=list(st.kept.get("name", [])), raus=list(st.dropped.get("name", [])),
                     nicht_beurteilbar=list(st.skipped.get("name", [])), kriterium=st.criterion)
        carry = st.name.split()[0] in screening.CARRY_SKIPPED
        nxt = pd.concat([st.kept, st.skipped]) if len(st.skipped) and carry else st.kept
        if len(st.skipped) and not carry:
            self.say(f"\n  {len(st.skipped)} Variante(n) waren hier nicht beurteilbar (Wert fehlt). In dieser "
                     "Stufe heißt das: kein Nachweis - sie gehen NICHT weiter.")
        if nxt.empty:
            self.say("\n  Keine Variante ist mehr übrig. Die weiteren Stufen werden der Vollständigkeit "
                     "halber angezeigt, prüfen aber nichts mehr.")
        self._pause()
        return nxt

    # 7-10 ──────────────────────────────────────────────────────────────
    def step7_s1(self) -> None:
        self.cur = pd.DataFrame(self.rows)
        self.cur = self._stage(7, screening.s1_explore, self.cur, f"""
        Locker: Ist BRUTTO (vor Spread) überhaupt etwas da, und schlägt das Signal einen
        Münzwurf? Der Placebo spielt jeden Trade noch einmal in Gegenrichtung durch (gleiche
        Zeit, gleicher Stop, gleiches Ziel). Was die Strategie mehr verdient als der Durchschnitt
        beider Richtungen, kommt von ihrer Richtungswahl - der Rest käme auch per Zufall.
        Mindestens {screening.S1_MIN_TRADES} Trades, sonst ist ein Mittelwert nichts wert.
        """, ["n", "brutto_R", "placebo_gross", "placebo_edge"],
            lambda st: ("Raus heißt hier: kein Brutto-Vorteil oder nicht besser als Zufall. "
                        "Kosten spielen in S1 bewusst noch keine Rolle."))

    def step8_s2(self) -> None:
        self.cur = self._stage(8, screening.s2_cost, self.cur, """
        Jetzt erst die Kosten - gemessen, nicht geschätzt: derselbe Backtest mit echten
        Bid/Ask-Fills. Bleibt netto ein positiver Erwartungswert je Trade?
        """, ["brutto_R", "netto_R"],
            lambda st: ("Wer hier rausfliegt, hat einen echten Brutto-Effekt, der die Kosten nicht "
                        "trägt - ein Befund, kein Prognose-Tod. Ideen dafür: längere Haltedauer, "
                        "Limit-Einstieg, günstigeres Instrument."))

    def step9_s3(self) -> None:
        self.cur = self._stage(9, screening.s3_robust, self.cur, f"""
        Robustheit: Ein Edge, der nur in einem Jahr existiert, ist ein Zufallstreffer.
          * Vorzeichen netto positiv in mindestens {screening.S3_MIN_POS_YEARS:.0%} der Jahre,
          * die Nachbar-Einstellungen (jeder Parameter eine Stufe hoch/runter) zu mindestens
            {screening.S3_NEIGHBOUR_MIN:.0%} positiv - ein Plateau, keine einzelne Spitze.
        """, ["netto_R", "pos_years", "n_years", "neighbour_positive"],
            lambda st: "Spitzen ohne Plateau sind typisch für Überanpassung an die Vergangenheit.")

    def step10_s4(self) -> None:
        pbo = self.stats["pbo"]
        self.cur = self._stage(10, screening.s4_significant, self.cur, f"""
        Signifikanz - die strengste Stufe vor dem Holdout. Die Frage: Ist das Ergebnis der
        besten Variante größer als das, was die beste von {self.stats['n_eff']:.0f} reinen
        Zufalls-Varianten zeigen würde?
          * DSR (Deflated Sharpe) >= {screening.S4_MIN_DSR}: Wahrscheinlichkeit, dass der Edge echt ist,
            NACH Abzug des Suchaufwands. Bei 1 Variante entspricht das t >= 1,65, bei vielen
            Varianten entsprechend mehr.
          * Placebo-Vorsprung > 0 und PBO <= {screening.S4_MAX_PBO} (PBO = Wahrscheinlichkeit, dass die
            beste Trainings-Variante außerhalb nur Mittelmaß ist; hier {pbo:.2f}).
        """, ["netto_R", "t_daily", "dsr", "placebo_edge", "pbo"],
            lambda st: ("Ein DSR knapp unter der Schwelle heißt: vielversprechend, aber noch nicht "
                        "bewiesen. Der Holdout wird dafür NICHT geöffnet - sonst wäre er kein Test mehr."))

    # 11 ────────────────────────────────────────────────────────────────
    def step11_holdout(self) -> None:
        self._header(11)
        survivors = set(self.cur.get("name", [])) if len(self.cur) else set()
        self.holdout_used = False
        if not survivors:
            self._explain("""
            Keine Variante hat S1-S4 bestanden, also gibt es nichts, was der Holdout prüfen
            könnte. Er bleibt unberührt - für eine spätere, bessere Idee.
            """)
            self._record(11, holdout="nicht geöffnet (keine S4-Überlebenden)")
            self._pause()
            return
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        key = f"{self.strategy}:{self.asset}"
        self._explain(f"""
        {len(survivors)} Variante(n) haben alle Trainings-Stufen bestanden. Jetzt kommt der eine
        ehrliche Test: die Jahre ab {self.holdout_start.date()}, die die Suche nie gesehen hat.
        WICHTIG: der Holdout ist nur EINMAL aussagekräftig. Wer danach Parameter ändert und
        wieder schaut, hat ihn zu Trainingsdaten gemacht.
        """)
        if key in ledger:
            self.say(f"\n  ACHTUNG: Der Holdout für {key} wurde bereits am {ledger[key]['datum']} geöffnet.")
            self._explain("Ein zweiter Blick ist kein unabhängiger Test mehr - das wird im Bericht vermerkt.")
        a = self.ask("\n  Zum Öffnen 'HOLDOUT' eintippen, [Enter] = nicht öffnen > ").strip()
        if a != "HOLDOUT":
            self._record(11, holdout="nicht geöffnet (Entscheidung)")
            return
        evaluate_holdout(self.spec, self.m1, self.rows, survivors, holdout_start=self.holdout_start)
        self.holdout_used = True
        ledger[key] = {"datum": datetime.now().isoformat(timespec="seconds"),
                       "varianten": sorted(survivors), "wiederholt": key in ledger}
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(ledger, indent=1))
        df = pd.DataFrame([r for r in self.rows if r["name"] in survivors])
        t = self._table(df.to_dict("records"))
        t["holdout_n"], t["holdout_netto_R"] = df["oos_n"].to_numpy(), df["oos_net_mean"].round(3).to_numpy()
        t["bestanden"] = df["oos_pass"].to_numpy()
        self.say("")
        self.say(t[["#", "netto_R", "holdout_n", "holdout_netto_R", "bestanden", "Parameter"]].to_string(index=False))
        self._explain("""
        Bestanden heißt: im Holdout mindestens 20 Trades und netto positiv. Liegt der Holdout
        deutlich unter dem Training, ist ein Teil des Trainings-Ergebnisses Überanpassung.
        """)
        self._record(11, holdout="geöffnet", ergebnisse=[{k: r.get(k) for k in ("name", "oos_n", "oos_net_mean", "oos_pass")}
                                                          for r in self.rows if r["name"] in survivors])
        self._pause()

    # 12 ────────────────────────────────────────────────────────────────
    def step12_report(self) -> Path:
        self._header(12)
        passed = [r for r in self.rows if r.get("oos_pass")]
        s4 = set(self.cur.get("name", [])) if len(self.cur) else set()
        if passed:
            best, status = max(passed, key=lambda r: r["oos_net_mean"]), "alle Stufen inkl. Holdout bestanden"
        elif s4:
            best, status = next(r for r in self.rows if r["name"] in s4), "S4 bestanden, Holdout nicht geöffnet/bestanden"
        else:
            best = max(self.rows, key=lambda r: r["objective_lcb"]) if self.rows else None
            status = "NICHT bestanden - nur zur Information"
        self.best, self.status = best, status
        if best is None:
            self.say("  Keine Variante.")
            return self._write_report()
        p = best.get("prop_holdout") if best.get("prop_holdout") else best["prop_train"]
        src = "Holdout" if best.get("prop_holdout") else "Train"
        self._explain(f"""
        Was hieße das für ein Prop-Firm-Konto mit Ziel 1-2 % pro Monat? Gerechnet mit der
        besten Variante ({status}), Datenbasis: {src}.
        """)
        self.say("")
        labels = {"r_per_month": "R pro Monat", "trades_per_month": "Trades pro Monat",
                  "pos_months": "Anteil positiver Monate", "max_dd_r": "max. Drawdown in R",
                  "worst_day_r": "schlechtester Tag in R", "risk_for_target_pct": "Risiko/Trade für 1,5 %/Monat (%)",
                  "dd_at_target_pct": "Drawdown bei diesem Risiko (%)",
                  "worst_day_at_target_pct": "schlechtester Tag bei diesem Risiko (%)",
                  "prop_ok": "passt zu 10 % Gesamt- / 5 % Tageslimit"}
        for k, lab in labels.items():
            if k in p:
                self.say(f"  {lab:42s} {p[k]}")
        if "nicht" in status.lower() and "bestanden -" in status:
            self.say("\n  ACHTUNG: Diese Variante hat den Trichter NICHT bestanden. Die Zahlen zeigen nur,")
            self.say("  wie weit sie vom Ziel entfernt ist - sie sind keine Empfehlung.")
        self._explain("""
        Lesart: Risiko/Trade ist das, was man pro Trade riskieren müsste, damit der
        durchschnittliche Monat das Ziel trifft. Ist der Drawdown dabei größer als das
        Gesamt-Limit der Firma, scheitert das Konto im Schnitt, bevor es das Ziel erreicht.
        Keine Anlageberatung - ein Forschungsergebnis auf historischen Daten.
        """)
        self._record(12, beste=best["name"], status=status, prop=p, prop_quelle=src)
        path = self._write_report()
        self.say(f"\n  Bericht: {path}")
        return path

    def _write_report(self, aborted: bool = False) -> Path | None:
        if getattr(self, "run_dir", None) is None:
            return None
        lines = [f"# Prüfbericht {self.strategy} auf {self.asset}", "",
                 f"Erstellt {datetime.now():%Y-%m-%d %H:%M}. Fester Ablauf, {len(STEPS)} Schritte."
                 + (" **ABGEBROCHEN.**" if aborted else ""), ""]
        sc = getattr(self, "spec_check", None)
        if sc is not None and not sc.agreed:
            lines += [f"> **{strategy_spec.UNCHECKED}** - keine abgestimmte Spezifikation "
                      f"(docs/specs/{self.strategy}.yaml). Das Ergebnis sagt etwas über diese Umsetzung, "
                      "nicht über die Idee.", ""]
        if hasattr(self, "status"):
            lines += [f"**Ergebnis:** {self.status}", ""]
        if getattr(self, "rows", None):
            df = self._table(self.rows)
            stage = {}
            for e in self.log:
                for n in e.get("bestanden", []):
                    stage[n] = e["name"].split(" - ")[0]
            df["erreicht"] = df["name"].map(lambda n: stage.get(n, "-"))
            cols = ["#", "n", "brutto_R", "netto_R", "pos_years", "n_years", "neighbour_positive",
                    "t_daily", "dsr", "erreicht", "Parameter"]
            lines += ["## Varianten", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for _, r in df.iterrows():
                lines.append("| " + " | ".join(
                    f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
            lines.append("")
        lines += ["## Protokoll (jeder Schritt, jede Entscheidung)", ""]
        for e in self.log:
            d = {k: v for k, v in e.items() if k not in ("schritt", "name", "zeit", "shortlist")}
            lines.append(f"- **{e['schritt']}. {e['name']}** ({e['zeit']}): "
                         + json.dumps(d, ensure_ascii=False, default=str)[:600])
        path = self.run_dir / "bericht.md"
        path.write_text("\n".join(lines))
        (self.run_dir / "varianten.json").write_text(json.dumps(getattr(self, "rows", []), indent=1, default=str))
        return path
