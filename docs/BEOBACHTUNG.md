# Beobachtungsliste

Automatisch aus `studies/watchlist.py` (nur Trainingszahlen). Eine Familie = Strategie x Markt, beste
Variante korrigiert für die Suche in der Familie, Benjamini-Hochberg bei 20 % FDR.

**Die Liste gibt nichts frei** - weder Handel noch Korb. Sie nennt nur, was eine Prüfung auf frischen
Daten oder im Korb verdient.

60 Familien geprüft, 0 auf der Liste.

| Familie | t | eff. Versuche | p (korrigiert) | DSR | Holdout |
|---|---|---|---|---|---|

## Am nächsten dran (nicht auf der Liste, nur zur Information)

| Familie | t | eff. Versuche | p (korrigiert) | Holdout |
|---|---|---|---|---|
| momentum:US100 | 3.74 | 77 | 0.007 | TK6, nicht bestanden |
| pre_fomc:INDIZES (S-009) | 2.11 | 1 | 0.018 | unberührt |
| noise_area_momentum:US100 | 1.95 | 1 | 0.031 | formal, t 0,41 |
| noise_area_momentum:SP500 | 1.86 | 1 | 0.035 | nicht bestanden |
| supertrend:XAUUSD | 3.24 | 123 | 0.074 | unberührt |
| supertrend:US100 | 3.14 | 103 | 0.088 | unberührt |
