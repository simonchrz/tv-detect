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
* **Seit 17.10. ohne O25:** K trainiert wie die Produktion ohne die 90
  Aufnahmen aus `docs/archiv-ausschluss-o25.json` (Lauf 014 = Kandidat,
  nicht schlechter). Kalibrierteil unverändert. Läufe bis 014 maßen gegen
  das alte K; ihre Deltas gelten, die absoluten Werte sind nicht mit
  späteren vergleichbar.
* **Rauschen** (A/A, Entwicklungssatz, Seeds 0/50/100 je 5 Wiederholungen):
  * K ohne O25 (17.10.): Mediane 0.9340 / 0.9340 / 0.9335, eine
    Wiederholung sd 0.0012–0.0026.
  * altes K (16.10.): 0.9304 / 0.9306 / 0.9294, sd ~0.001.
  Das neue K liegt +0.0036 höher, streut je Wiederholung aber etwa doppelt
  so stark; die Mediane der drei Sätze liegen weiterhin innerhalb 0.0005.

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
Wiederholungen positiv, höchstens eine Aufnahme mit Δ < −0.10, und (seit
16.10.) der Median der Aufnahme-IoU darf nicht sinken.
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

## Vorgabe ab Durchgang 2 (Simon, 16.10.)

Durchgang 1 hat allgemeines Umgewichten abgehakt: Sender-Ausgleich (001,
+0.0003), Kanten-Saum (002, +0.0005), Backbone-Dropout (003, −0.0005) —
alles Rauschen. **Keine weiteren globalen Gewichts-, Regularisierungs- oder
Kapazitätsvarianten.** Jeder Arm greift eine BENANNTE Schwäche an:

1. **Trailer mit Sendetermin-Einblendung** („Montag 20:15 joyn“, „Mo–Fr
   19:05“): Kopf gab Trailern 0.586 (§3aa); Menschen verwechseln sie auch.
   Träger: `ocr_hinweis`, SigLIP, Logo (Sender-Logo fehlt oft im Trailer).
2. **Countdown-Einzelspots** („Werbung – Jetzt geht's weiter GALILEO“, §3ax):
   Logo fehlt, `ocr_werbung` hoch, NN 0.06–0.26. Seit 16.10. sind 36
   Galileo-Kanten im Archiv korrigiert — der Trainingsteil enthält die Form
   jetzt als Werbung.
3. **Verwechslungszonen**: 13 % der Sekunden, deren Backbone-Nachbarn das
   Gegenteil-Label tragen, tragen 45 % der Fehler
   (`scripts/backbone-sonde.py`, Memory fehler_sitzen_in_der_repraesentation).
   Ansatz z. B. Merkmale, die dort trennen (OCR, Logo-Verlauf, Zeitkontext
   über Nachbarzeilen), nicht mehr Gewicht.

Diagnose vorher auf dem Trainingsteil: zeigen, dass die Schwäche dort
messbar ist und der Arm sie adressiert (eine Zahl in `--begruendung`).

## Lehre aus 006–008 (16.10.)

Gestapelter Kopf (Stufe 1 out-of-fold, Stufe 2 mit Logo/Audio/OCR): 006
+0.0052, Gegenproben mit anderen Seeds je +0.0027. Echt, aber getragen von
drei South-Park-Aufnahmen (+0.07 bis +0.24), während der Median der
Aufnahmen SANK. Befund daraus: der Kopf nutzt das Logo außerhalb seiner
Trainingsdaten zu wenig (Logo ≥ 0.8 und p ≥ 0.8: nur 53 % Werbung). Ein Arm,
der das gezielt nutzt, ohne die typische Aufnahme zu verschlechtern, ist
willkommen. Keine weitere Seed-/Falten-Variante des Stapels.
Jeder Arm mit Zufall in der Merkmalsbildung zieht diesen Zufall aus dem
Wiederholungs-Seed (ktx enthält ihn nicht — dann eine feste Liste von
Seeds je Aufruf benutzen und das in der Begründung sagen), sonst misst die
Streuung nur einen Teil.

## Fehlerarten im Urteil (seit 16.10.)

Jedes Ergebnis zählt je Wiederholung die Fehlersekunden nach Ursache (Regel
wie `fehlerbudget.py`: ≤ 15 s an einer wahren Kante = Kante, sonst NN bzw.
Dekoder erfindet/verpasst); das Urteil führt `ursachen_sekunden` mit
Kontrolle und Δ, `opt_stand.py` zeigt sie. Kontrolle K auf Entwicklung
(Seed 0): Kante 1590, NN erfindet 1329, NN verpasst 1100, Dekoder erfindet
194, Dekoder verpasst 62 Sekunden — das Abbild erfindet doppelt so viel wie
die Produktion (636 auf dem ganzen Messsatz). Ein Arm, der eine Zeile senkt
und eine andere hebt, verschiebt Fehler, statt sie zu beheben.

## Ideenklassen und Banditen-Auswahl (seit 16.10.)

Jeder Arm gehört zu einer Klasse (`opt_registrieren.py --klasse`):
`merkmale` (spalten), `training` (HYPER/gewichte/modell), `nachbearbeitung`,
`daten` (Aufnahmen herausnehmen oder umgewichten nach Herkunft/Zustand),
`labels` (Trainingslabels in der Kopie umschreiben, z. B. eine Regel für
Countdown-Spots, Nachlauf, Trenner). **Vor jeder Runde zieht
`opt_bandit.py` die Klasse** (Thompson-Sampling über die Urteile: Belohnung
0 bei Δ ≤ −0.003, 0.5 bei 0, 1 ab der Schwelle, halbiert bei sinkendem
Aufnahme-Median, 1 für Kandidat, 0 für nicht bestätigt). Der Planer bleibt in
der gezogenen Klasse. 001–011 sind nachträglich eingeordnet (001–003
training, 004–011 merkmale).

**Daten- und Label-Arme sind die Simulation für Labelentscheidungen:**
`ziel()`/`gewichte()` ändern nur die Trainingskopie, gemessen wird gegen
die echten Menschenlabels. Damit lässt sich beziffern, was eine
Labelkorrektur bringen WÜRDE, bevor jemand das Archiv anfasst. Ein
bestätigter Label-Arm ist eine Empfehlung an Simon (L2: Stichprobe im
Bild, dann Freigabe), keine Korrektur. `ktx["sek"]` gibt die absolute
Sekunde je Zeile; im Trainingsteil liegt in `meta[uuid]` u. a. `ads`,
`which`, `title`, `slug`, `start_ts`, `label_korrektur`.
Achtung Entwicklungssatz: dessen Aufnahmen stehen nicht im Trainingsteil —
ein Label-Arm wirkt nur über das, was der Kopf aus den ANDEREN Aufnahmen lernt.

## Schiedsrichter Qwen — Eichung 17.10. (`scripts/optimierer/opt_schiedsrichter.py`)

Ziel: Arme ohne Menschenlabels beurteilen. Qwen3.8-27B urteilt nur an den
Sekunden, an denen Kontrolle und Arm verschieden entscheiden (Raster 12 s,
Rand 6 s, höchstens 40 je Aufnahme, sicher = beide Durchgänge gleich und
≥ 0.95). Geeicht an den Armen 015–022 auf dem Entwicklungssatz, 320
verschiedene Punkte, 4.5 s je Punkt.

* Sicher sind nur 43 % der Punkte (139 von 320); Uneinigkeitsstellen sind schwere Stellen.
* Gegen die Menschenlabels stimmen 101 von 139 sicheren Punkten (73 %).
  Die Abweichungen sind fast nur Konventionsfälle, im Bild geprüft:
  * 23 Punkte Split-Screen bei Let's Dance (`dvr-rtl-1780078500`, ab
    14364 s): Spot groß mit „Werbung“, Show im kleinen Fenster. Label sagt
    Sendung, Qwen Werbung. Konvention offen.
  * Gewinnspiel am Blockrand (`dvr-prosieben-1781095127` @864/876,
    `dvr-kabel-eins-1779292601` @1500) und Trailer am Blockrand
    (disney, nick, super-rtl): Konvention sagt Werbung, das Label beginnt
    erst danach. Hier liegt Qwen richtig, das Label daneben.
  * Ohne die Split-Screen-Aufnahme 87 % (0.95) bzw. 79 % (Schwelle 0.5,
    dann 240 statt 116 Punkte).
* Urteil je Arm (Median über 5 Seeds, Sekunden Arm besser als K):

| Arm | Qwen | Mensch gleiche Punkte | Mensch voll |
|---|---|---|---|
| 015 bildwechsel_dichte | −276 | +216 | +250 |
| 016 logo_blind_aufnahmen_gewicht | −24 | +24 | −5 |
| 017 logit_weichkappe | −36 | −168 | −163 |
| 018 klammer_wiederkehr | +60 | +156 | +109 |
| 019 werbe_dubletten_gewicht | +48 | −132 | −118 |
| 020 aufnahme_pegel_stapel | −468 | −720 | −716 |
| 021 klammer_kette | 0 | +120 | +104 |
| 022 siglip_text_fenster | −48 | −132 | −145 |

  Gleiches Vorzeichen bei 5 von 8. Die beiden Gegenläufer erklären sich
  ganz aus den Konventionsfällen: 015 hängt an der Split-Screen-Aufnahme
  (Qwen −1008, Mensch +1380; ohne sie beide negativ), 019 am Gewinnspiel
  in `dvr-prosieben-1781095127` (Qwen +120, Mensch −120, Qwen hat recht).
* Folgerung: Qwen taugt als Schiedsrichter, wenn die Split-Screen-
  Konvention entschieden ist. Noch nicht im Kreislauf; offen sind die
  Konvention und die geringe Abdeckung (Schwelle senken kostet ~8 Punkte
  Trefferquote).

### Seit 17.10.: Schiedsrichter im Kreislauf

Simon: Split-Screen-Werbung ist Werbung, Ausnahme Let's Dance (dort ist der Split-Screen Teil der Sendung, die Labels bleiben). Steht in der Qwen-Frage (
`~/ml/vlm-exp/vlm_wisch.py`, Sicherung `.vor-splitscreen-20261017`).

* **Schiedsrichter-Satz** `~/.cache/tvd-train-archive/schiri-satz-2026-10-17.json`:
  Ledger test, nicht im Messsatz, mit Features, ohne Menschenlabel (119;
  davon nur die mit Quelle bekommen einen Dump unter
  `~/.cache/tv-detect-daemon/emit-signals-schiri`). Nie im Training des
  Optimierers, nie im Golden.
* **Ablauf:** Nach erfüllter Entwicklung misst der Läufer wie bisher die
  menschliche Bestätigung (22) und dann `opt_schiedsrichter.py lauf NNN`
  (fortsetzbar, Rückgabe 3 = erneut aufrufen). Ergebnis
  `laeufe/NNN/schiedsrichter.json`, sein `status` ist das Endurteil.
* **Regel (vorläufig, aus der Eichung):** Kandidat, wenn Qwen den Arm in
  4/5 Seeds vorn sieht, Median ≥ +0.5 s je Aufnahme, und die menschliche
  Bestätigung nicht unter −0.002 liegt. Kandidat heißt weiter nur:
  Simon entscheidet, das Nachtgate bleibt menschlich.
