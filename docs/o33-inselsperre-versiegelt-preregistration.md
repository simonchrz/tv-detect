> **ABGESCHLOSSEN 2026-10-05 — ERFÜLLT (R1 und R2), aber getragen von EINER Aufnahme.**
>
> | | Seed 0 | 1 | 2 | 3 | 4 | Median |
> |---|---|---|---|---|---|---|
> | Δ IoU primär (39) | +0.0000 | +0.0024 | +0.0024 | +0.0024 | +0.0024 | **+0.0024** |
> | Δ IoU alle (63) | +0.0000 | +0.0015 | +0.0015 | +0.0015 | +0.0015 | +0.0015 |
>
> Verluste > 0.10: 0. Auf allen anderen 62 Aufnahmen ändert I90 in keinem Seed
> irgendetwas. Der ganze Effekt ist dvr-rtl-1788097500 (Die Beet-Brüder):
> K teilt den ersten Werbeblock durch eine 56-s-Sendungsinsel (23:51–24:47),
> IoU 0.899 → 0.993 — genau das Muster aus O32 (Hot oder Schrott). Ehrliche
> Lesart: die Sperre schadet nirgends und behebt diesen Fehlertyp, wo er
> auftritt; er ist selten (1 von 63). Log `~/Library/Logs/o33-lauf.log`.

# O33 — Hebt die Inselsperre (90 s) die Blockgüte auf dem versiegelten Satz? (Vorab-Registrierung)

**Geschrieben 2026-10-05, vor dem ersten Datenpunkt auf dem versiegelten Satz.**
Simon hat den versiegelten Satz für genau diese Frage geöffnet (05.10.). Danach
ist er für weitere Entscheidungen teilweise verbraucht; der Ledger vermerkt es.
Skript: `scripts/o33-inselsperre-versiegelt.py`.

## Woher die Frage kommt

O32 (VERFEHLT, R1) fand als Nebenbefund: mit `--hsmm-inner-show-min 90` stieg
die IoU des Kandidaten in allen 5 Seeds (+0.0016 bis +0.0059, Median +0.0038)
ohne einen einzigen Verlust > 0.10. Das war nicht die registrierte Frage und
wurde auf denselben test-Aufnahmen gefunden, auf denen jetzt nicht mehr
bestätigt werden darf. Bestätigung nur auf Daten, die noch nie für eine
Entscheidung ausgewertet wurden: dem versiegelten Satz (63 Aufnahmen, alle mit
Blöcken).

## Labels im versiegelten Satz (VOR der Messung festgestellt)

`massstab-audit.py`: 0 nachweislich menschlich, 24 maschinell (auto-confirm =
Ausgabe des damaligen Dekoders, also K), 39 unbekannt (Archiv `merged`, deckt
Mensch UND auto-confirm). Maschinelle Labels begünstigen K per Konstruktion.
Deshalb:

* **Primär:** die 39 nicht-maschinellen Aufnahmen.
* **Sekundär (nur Bericht):** alle 63.

## Behandlung

Arme K (Gate-Dekoder: hsmm, DurW 15) und I90 (`--hsmm-inner-show-min 90`).
5 Seeds; je Seed EIN Kopf, trainiert auf ALLEN train-Aufnahmen wie in O31/O32
(gleiche Zeilen, Spalten, Architektur, MLP7). Beide Arme dekodieren DIESELBEN
Wahrscheinlichkeiten; Unterschied nur der Dekoder. Ausgewertet wie das Gate:
`_replay_blocks` (ohne Decode-Spur `_replay_ohne_spur`), `block_iou` gegen die
Archiv-Labels.

## Regel

Auf dem PRIMÄREN Satz, gepaart nach Seed:
* **R1 Gewinn:** Median(IoU_I90 − IoU_K) ≥ +0.002 UND Δ > 0 in ≥ 4 von 5 Seeds.
* **R2 Einzelschaden:** über alle Seeds höchstens 2 Fälle, in denen eine
  Aufnahme mit I90 mehr als 0.10 gegen K verliert — ausgenommen Aufnahmen,
  deren Label eine Sendungslücke < 90 s zwischen zwei Blöcken enthält (einzeln
  aufgeführt, Simon zur Prüfung vorgelegt, nicht gezählt).

ERFÜLLT, wenn R1 und R2 halten.

**Konsequenz bei ERFÜLLT:** Vorschlag an Simon, `--hsmm-inner-show-min 90` in
Detect (`tv-thumbs-daemon.py`) UND `EVAL_DECODER` (`train-head.py`)
gleichzeitig zu setzen. Golden-/Per-Rec-Zahlen davor und danach sind nicht
vergleichbar (`decoder`-Feld in golden-trend.jsonl). Ohne Simons OK keine
Produktionsänderung.
**Bei VERFEHLT:** Option bleibt aus; der O32-Nebenbefund gilt als nicht bestätigt.

```regel
{"id": "O33", "frage": "Hebt die Inselsperre (90 s) die Blockguete auf dem versiegelten Satz?",
 "nicht_in_serienabschluss": true,
 "name": "O33 Inselsperre versiegelt", "art": "offline",
 "skript": "scripts/o33-inselsperre-versiegelt.py",
 "r1": "median(IoU_I90 - IoU_K) >= +0.002 und Delta > 0 in >= 4/5 Seeds (primaer: nicht-maschinelle Labels)",
 "r2": "<= 2 Verluste > 0.10 ueber alle Seeds, ausser Label mit innerer Luecke < 90 s",
 "seeds": 5, "arme": {"K": 0, "I90": 90}}
```
