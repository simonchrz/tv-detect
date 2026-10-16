# Optimierer — Kopf-Versuche ohne Nacht

Ein Kreislauf aus **Planer → Prüfer → Läufer → Urteil**, der Versuchsarme
gegen den Kontrollkopf misst, ohne auf das Nachttraining zu warten. Ein Lauf
dauert 1–3 Minuten. Er ersetzt NICHT das Gate und liefert nie aus: ein
bestätigter Arm ist ein **Kandidat für Simon**, die Übertragung in
`train-head.py` und das Nachtgate entscheiden danach (L3, L5).

## Messgerät (`scripts/optimierer/opt_lauf.py`)

* **Kontrolle K** = Abbild des Produktionskopfs MLP7: Backbone 1280 + Logo +
  Audio + 3 OCR-Spalten (`ocr_spalten.py`) + SigLIP 64 + Indikator
  (`siglip_spalten.py`, Projektion aus dem Trainingsteil), MLP hidden 96,
  12 Epochen, Klassengewichte N/(2·n_k), **Logit-Mittel aus 3 Köpfen** je
  Wiederholung, Platt-Kalibrierung auf einem eigenen Kalibrierteil.
* **Bewertung wie die Produktion:** Wahrscheinlichkeit je Sekunde →
  per-Frame-CSV → `tv-detect --replay-signals --replay-nn-csv` mit dem
  Produktions-HSMM (`--hsmm-dur-w 15 --hsmm-inner-show-min 90`) → Block-IoU
  gegen das Menschenlabel (`backbone-sonde.menschlabels`).
* **Abbild-Lücke** (16.10., 22 ungesehene test-Aufnahmen): K 0.970 gegen
  Produktion 0.979 (gepaart −0.009, fast ganz aus toggo plus 1780640400 und
  nick 1781550300); Mediane 0.990 gegen 0.992. K fehlen die Gewichtungen
  der Produktion (Alter, Herkunft, Anker, Hygiene) und 25 % der Daten.
* **Rauschen** (A/A): eine Wiederholung sd ~0.001; drei Sätze à 5
  Wiederholungen im Median 0.9294 / 0.9304 / 0.9306.

## Sätze — fest, kein Arm wählt sie

| Satz | Inhalt | Zweck |
|---|---|---|
| training | train-Eimer ohne die drei unten (609 Aufnahmen, jede 4. Sekunde) | Fit |
| kalibrierung | 40 train-Aufnahmen per Hash (34 mit Merkmalen) | nur Platt |
| entwicklung | Messsatz ∩ train (76, Menschenlabel, Signal-Dump) | Suche |
| bestaetigung | Messsatz ∩ test (22) | **einmal** je Lauf, nur nach erfülltem Entwicklungs-Urteil |

Golden-Satz und versiegelter Satz kommen nicht vor. Ein Arm sieht von
Entwicklung/Bestätigung nur Merkmale, nie Labels.

## Was ein Arm ist (`scripts/optimierer/arme/<name>.py`)

Alles optional, mindestens eins davon:

```python
BESCHREIBUNG = "…"
HYPER = {"hidden": 96, "epochen": 12, "lr": 1e-3, "batch": 8192,
         "dropout": 0.0, "weight_decay": 0.0, "ensemble": 3}
def spalten(teil, X, ktx): ...      # Merkmale je Teil umbauen (training|kalibrierung|eval)
def gewichte(ktx): ...              # Gewicht je Trainingszeile (>= 0)
def ziel(ktx): ...                  # Ziel je Trainingszeile (0/1)
def modell(n_in, hyper): ...        # torch.nn.Module mit 2 Logits
def nachbearbeiten(p, rec, ktx): ...# je Sekunde, vor der Kalibrierung
```

`ktx` = `{teil, uuids, rec, meta (Archiv-Meta je uuid: title, slug, which,
ads NUR bei training, cohort, start_ts …), schritt, spalten_namen, y (nur
training)}`. Spalten: `backbone0..1279, logo, audio, ocr_hinweis,
ocr_werbung, ocr_da, siglip0..63, siglip_da`. Achtung: im Trainingsteil
ist jede 4. Sekunde geladen (`schritt=4`), in Kalibrierung und Eval jede
Sekunde — zeitliche Merkmale müssen `ktx["schritt"]` berücksichtigen.

**Verboten** (opt_registrieren.py prüft grob, der Prüfer genau): Labels,
Cutlists, Archiv, Ledger, Satzwahl, Netz, Prozesse, Dateien schreiben,
löschen; `meta["ads"]` oder `which` von eval-Aufnahmen benutzen (dort
fehlen die ads, aber `which` und `title` sind da — Titel/Kanal als
Merkmal ist erlaubt, Herkunft des Labels nicht).

## Ablauf eines Laufs

```sh
cd ~/src/tv-detect; PY=/Users/simon/ml/tv-classifier/.venv/bin/python
$PY scripts/optimierer/opt_registrieren.py --arm <name> --regel gewinn|nicht_schlechter \
    --hypothese "…" --begruendung "…"          # committet, VOR jeder Messung
$PY scripts/optimierer/opt_lauf.py --arm <name> --lauf NNN-<name>
$PY scripts/optimierer/opt_urteil.py --lauf NNN-<name>
# nur bei Status bestaetigung_faellig:
$PY scripts/optimierer/opt_lauf.py --arm <name> --lauf NNN-<name> --satz bestaetigung
$PY scripts/optimierer/opt_urteil.py --lauf NNN-<name>
```

Schwellen (fest in `opt_urteil.py`): Gewinn = Median Δ ≥ +0.003, ≥ 4/5
Wiederholungen positiv, höchstens eine Aufnahme mit Δ < −0.10.
Bestätigung: Median Δ > 0 und ≥ 4/5 positiv. Nichts davon ist verhandelbar;
eine Variante eines verworfenen Arms ist ein **neuer** Arm mit neuem Namen
und zählt als weiterer Versuch (Journal zählt mit).

Schutz: kein Lauf 02:15–05:45 (Nachttraining), nicht neben train-head.py,
nicht unter 24 GB freiem Speicher, nie zwei gleichzeitig. Bei „Nachtfenster“
endet der Kreislauf, er wartet nicht.

## Wo noch etwas zu holen ist (Fehlerbudget 16.10., 98 Aufnahmen)

Produktion 0.983, perfektes NN 0.999 (NN-Deckel 0.017), Dekoder-Deckel
0.001. Verlust-Sekunden: Kante 51 %, NN verpasst 22 %, NN erfindet 20 %,
Dekoder 7 %. Bekannte NN-Schwächen: Trailer/Programmhinweise (Kopf 0.586,
§3aa), Countdown-Einzelspots (0.08, §3ax), Kinder-Sender (toggo, nick:
größte Abbild-Lücke), Fehler häufen sich, wo Backbone-Nachbarn das
Gegenteil-Label tragen (13 % der Sekunden, 45 % der Fehler).

## Schon entschieden — nicht noch einmal (Ledger §4 und §3)

* Kapazität: MLP-64/128 (O6, zweimal Rauschen).
* Minute-Prior (inert), Boundary-Head (negativ), Kontextspalten als
  Kantenhebel (O16), Whisper-Spalte (O1: kostet), weiches Champion-Ziel
  (O31), Inselsperre über den Kopf (O32), Klassen-Split Spot/Trailer über
  Anker (O20), Herkunfts-Gewichte (O17, O18), Cluster-Anker im Training
  (O24), Defekt-Archiv entfernen (O25), Backbone durch SigLIP ersetzen /
  mehr SigLIP-Komponenten (O30), Flankenauswahl (O14), Dekoder-Sweeps.
* Wer etwas davon wieder anfasst, braucht einen Grund, der die alte Messung
  entwertet — und schreibt ihn in `--begruendung`.

Erfolgreich und schon in Produktion: OCR-Spalten (O26), SigLIP-64 (O28),
Inselsperre 90 s im Dekoder (O33), Seed-Ensemble.
