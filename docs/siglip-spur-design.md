# SigLIP-Spur — Produktionsweg für SigLIP 2 je Sekunde (Entwurf, NICHT gebaut)

**Stand 2026-09-26. Wartet auf Simons Freigabe.** Der Header-Wechsel (Schritt 5) ist
eine L5-Änderung und braucht ein ausdrückliches OK. Nichts hiervon ist gebaut.

## Warum

O28 und O29 (`docs/o28-…`, `docs/o29-…`, Ledger): SigLIP 2 NaFlex je Sekunde, als
PCA-64 + Indikator angehängt, hebt den Produktionszustand (inkl. OCR-Spalten) von
F1 **0.9236 auf 0.9644** (+0.040, 5/5 Seeds, sd 0.0025). Billige Träger ersetzen es
nicht (SigLIP-Mittel je Aufnahme +0.004, Sendungs-Kennung +0.002). Grundlage sind 23
menschlich gelabelte test-Aufnahmen. Das ist konsistent, aber schmal, siehe „Risiken“.

## Leitidee: die OCR-Spur kopieren

Die OCR-Spur (O26 → MLP6) hat genau dieses Problem schon einmal gelöst, und der Weg hat
sich bewährt. SigLIP bekommt denselben Aufbau, keine neue Architektur:

| | OCR-Spur (heute) | SigLIP-Spur (neu) |
|---|---|---|
| Erzeuger | `tv-ocr-spur` (Go, Vision) | `scripts/siglip-spur.py` in `~/ml/siglip-exp/.venv` |
| Ablage | `~/.cache/tvd-ocr-spur/<uuid>.json` | `~/.cache/tvd-siglip2/<uuid>.npy` (float16, n×768) + `<uuid>.json` (Quelle bytes/mtime, Modell, Stand) |
| Frische | `_ocr_spur_frisch` (Quelle bytes + mtime) | dasselbe: veraltete Spur wird NIE mitgegeben |
| Detect | Daemon erzeugt vor dem Detect, `--ocr-spur` | Daemon erzeugt vor dem Detect, `--siglip-spur` |
| Spalten | `ocr_spalten.py` ↔ `ocrspalten.go` + Paritätstest | `siglip_spalten.py` ↔ `siglipspalten.go` + Paritätstest |
| Fehlt die Spur | Spalten 0 wie im Training | Spalten 0, `siglip_da=0`, wie im Training |

**Dieselbe Spur für Training und Detect.** Train/Serve-Parität entsteht durch Bauart
(ein Erzeuger), nicht durch Nachrechnen. Die O27-Falle (Training voll, Detect halb)
kann so nicht entstehen: der Erzeuger dekodiert selbst, unabhängig von
`DETECT_DECODE_SCALE`.

## Die Schritte

1. **Erzeuger `scripts/siglip-spur.py`.** Er ist aus `o28-siglip-merkmale.py`
   hervorgegangen, mit zwei Änderungen:
   - *Gekachelt* in 180-s-Fenster mit eigenem `-ss` wie `tv-ocr-spur`. Eine Stunde
     `fps=1` an einem Stück driftet bei .ts (Memory `frames_tragen_erwartete_zeit`).
     Die Offline-Merkmale wurden noch am Stück gerechnet und nur auf ±2 Zeilen geprüft.
     Deshalb gilt ein **Paritätsschritt**: für ~20 Aufnahmen gekachelt gegen am Stück
     vergleichen und die Abweichung je Sekunde nennen. Ist sie groß, wird der Cache neu
     gerechnet (~1.5 h für 255 Aufnahmen).
   - Eine Frische-Beilage (`quelle_bytes`, `quelle_mtime`, `modell`, `max_patches`).
2. **Daemon.** `_siglip_spur_fuer()` nach dem Muster von `_ocr_spur_fuer`. Sie darf den
   Detect nie aufhalten: bei Fehler laufen die Spalten mit 0 weiter. Aufruf in einem
   eigenen Dienst-Prozess? Nein, als Kind mit Timeout reicht. Aber die launchd-Grenzen
   (256 Dateien, Memory `umgebung_ist_teil_des_laufs`) im Probelauf per `kickstart`
   prüfen, nicht im Terminal.
3. **Go-Seite (`internal/signals`).** `LadeSigLIPSpur` + `SetSigLIPSpalten`. Die
   **Projektion 768→64 rechnet Go** aus Werten, die IM Kopf stehen (Schritt 4). Die
   Spur bleibt roh und kopfunabhängig, genau wie die OCR-Spur rohe Treffer hält.
4. **Kopf-Format MLP7** = v6 + `n_siglip` (0 oder 65). Mittelwert (768) und
   Projektionsmatrix (768×64) stehen **im Körper von head.bin**, hinter den Gewichten,
   und NICHT als eigene Beilage. Begründung: die Audio-Spalte hing an einer
   Beilage-Datei, und der Transport riss dreimal (Memory `audio_spalte_traegt_zweierlei`).
   Was nicht getrennt werden kann, kann nicht auseinanderlaufen. v6-Köpfe liest derselbe
   Lader unverändert.
5. **Training (`train-head.py`).** PCA NUR auf train-Zeilen mit Spur (wie O28). SigLIP-
   Spalten stehen ganz hinten (Präfix-Vertrag), Aufnahmen ohne Spur haben Nullen und
   `siglip_da=0`. Geschrieben wird v7 **nur hinter einem Schalter** (`--siglip`), bis
   Schritt 6 entscheidet.
6. **Einführung wie MLP6, mit L5-OK.** Reihenfolge, damit nie ein Kopf ankommt, den der
   Detect nicht lesen kann:
   1. Go-Lader v7 + Spur-Laden deployen (liest v6 unverändert) und die Detect-Zeit messen.
   2. Daemon-Erzeugung an: Spuren entstehen, der v6-Kopf ignoriert sie.
   3. Nightly rechnet einen v7-Herausforderer. Er muss das Tor gegen den v6-Champion
      bestehen (Golden-Boden + Kopf-an-Kopf). Das Tor ist nur relativ (Memory
      `train_gate_is_relative_only_ratchet`), deshalb zusätzlich den O29-Wert auf den
      23 Aufnahmen nennen.
   4. Erst dann ausliefern.

## Kosten

- **Rechenzeit:** gemessen 15–25 s je 30 min Video, 60–90 s je Stunde, auf der Mac-GPU
  (MPS, Stapel 64). Der Detect braucht heute 4–13 min je Aufnahme, also rund +10–20 %.
  Im echten Detect-Pfad neu messen: dort teilt sich SigLIP die GPU mit dem
  CoreML-Backbone.
- **Speicher:** 768 × float16 = 1.5 kB je Sekunde, also ~5.5 MB je Stunde. Die 255
  Aufnahmen heute liegen bei rund 0.6 GB.
- **Abhängigkeit:** eine zweite Python-Umgebung (torch + transformers) im
  Produktionsweg. Ein späterer CoreML-Export (feste 16:9-Patchzahl statt NaFlex) würde
  sie ablösen. Das gehört nicht zu diesem Entwurf.

## Risiken und was ich vorher prüfen würde

- **Schmale Grundlage:** 23 test-Aufnahmen. Der Herausforderer im Nightly (Schritt 6.3)
  ist die zweite, unabhängige Probe. Besteht er das Tor nicht, wird nichts ausgeliefert.
- **Abdeckung im Training:** nur ~30 % des Korpus hat noch eine Quelle. Neue Aufnahmen
  bekommen die Spur ab Schritt 2, der Anteil wächst also. Zeilen mit `siglip_da=0` sind im
  Training gut vertreten; ein Detect ohne Spur ist also kein Fremdzustand.
- **Zeitachse:** siehe Schritt 1. Das ist der wahrscheinlichste stille Fehler (Memory
  `frames_tragen_erwartete_zeit`, `groessenpruefung_hielt_den_kopf_fest`).
- **Live-Detect:** bekommt keine Spur (die braucht die fertige Datei), sieht also
  `siglip_da=0`, einen im Training vertretenen Zustand. Sollte der Live-Pfad messbar
  leiden, bekommt er einen eigenen Kopf. Vor der Einführung prüfen.
- **Ungeklärt:** die Hälfte des Gewinns überlebt das zeitliche Mischen (O29-Gegenprobe).
  Für den Bau ändert das nichts, beide Hälften brauchen Einzelbilder. Es bleibt aber
  eine offene Frage für später.
