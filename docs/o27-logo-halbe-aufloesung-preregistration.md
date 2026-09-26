# O27 — Hilft es, die Logo-Spalte bei der Produktions-Aufloesung zu trainieren? (Vorab-Registrierung)

**Geschrieben 2026-09-26, vor dem ersten Behandlungs-Datenpunkt.** Bauart wie
[`o26-ocr-spalte-preregistration.md`](o26-ocr-spalte-preregistration.md):
offline, gleiche Zeilen, gleiche Seeds, gleiche Architektur; die Arme
unterscheiden sich NUR im Inhalt der Logo-Spalte (Index 1280). Skript:
`scripts/o27-logo-halb.py`.

## Woher die Frage kommt

Sweep 2026-09-25: das Training extrahiert die Logo-Spalte mit dem Template
bei VOLLER Aufloesung (`train-head.py extract_logo_per_second`), die
Produktion dekodiert HALB (`DETECT_DECODE_SCALE=0.5`, skaliertes Template).
Die Werte weichen ab (MAD Median 0.058–0.064, max 0.585).

Vormessung (`scripts/o27-vormessung.py`, ohne Behandlung): derselbe
Kontrollarm-Fit, ausgewertet einmal mit der Trainings-, einmal mit der
Produktionsspalte. Auf den 23 menschlich gelabelten test-Aufnahmen mit
Quelle ist die Produktionsspalte BESSER: Delta F1 +0.079 (5/5 Seeds). Die
Dreiweg-Gegenprobe schliesst den Template-Stand als Ursache aus: VOLL frisch
mit dem heutigen Template trifft das Archiv exakt (MAD ≤ 0.002, F1
identisch). Die halbe Aufloesung ist also selbst das bessere Signal — die
Frage ist, ob ein Kopf, der sie auch im TRAINING sieht, noch mehr gewinnt.

## Die Arme

| Arm | Logo-Spalte in train | Logo-Spalte in test |
|---|---|---|
| kontrolle | Archiv (voll) | Archiv (voll) |
| versuch | halb, wo die Quelle noch liegt; sonst Archiv (voll) | halb, wo die Quelle liegt |

Halbe Werte kommen aus `~/.cache/tvd-o27-logo-halb/<uuid>.npy`, erzeugt mit
derselben Funktion wie das Training (`extract_logo_per_second`), nur mit
`--decode-width/--decode-height` und dem skalierten Template wie im Daemon.
NaN → 0.5 wie im Loader. Rund 70 % der train-Aufnahmen haben keine Quelle
mehr; der Versuchsarm ist also gemischt — genau so saehe ein Produktionsweg
aus. Ein Erfolg hier ist eine Untergrenze.

## Rauschen, VOR der Behandlung gemessen

Kontrollarm (= "voll" der Vormessung, `--nur-mensch`, 5 Seeds): F1 0.8304 /
0.7877 / 0.8382 / 0.7991 / 0.8179, **Median 0.8179, sd 0.0211**. Hoch, weil
nur 23 Aufnahmen die primaere Grundgesamtheit bilden.

## Regel

```regel
{
  "id": "O27",
  "frage": "Hilft es, die Logo-Spalte bei der Produktions-Aufloesung (halb) zu trainieren?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf den MENSCHLICH gelabelten test-Aufnahmen mit Quelle (label_herkunft.mensch_aus_markern is True), geglaettet 10s; Nebenwert: alle test-Aufnahmen mit Quelle",
  "arme": {"kontrolle": "Logo-Spalte voll (Archiv) in train und test", "versuch": "Logo-Spalte halb, wo die Quelle liegt (train und test)"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0211,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.02,
    "positive_seeds_mindestens": 4
  },
  "konsequenz_bei_erfuellt": "Produktionsweg VORSCHLAGEN: train-head extrahiert die Logo-Spalte kuenftig bei halber Aufloesung (wie der Daemon), neu extrahiert wo die Quelle liegt; die Semantik der Spalte kommt wie bei O22 in eine Beilage, damit Detect und Kopf nicht auseinanderlaufen. Nicht bauen ohne OK.",
  "konsequenz_bei_verfehlt": "Nichts aendern. Die Produktion profitiert laut Vormessung schon heute von der halben Aufloesung; ein Umbau des Trainings lohnt nicht."
}
```

## Warum die Schwelle so hoch ist

+0.02 ≈ 1 sd des Kontrollarms. Kleiner waere bei 23 Aufnahmen nicht von
Fit-Zufall zu trennen, und ein Erfolg haette Kosten (Neu-Extraktion,
Beilage, Train/Serve-Vertrag).

## Was diese Frage NICHT beantwortet

* Nicht den Golden-Median (22 der 23 primaeren Aufnahmen sind golden, aber
  der Golden-Median misst den ausgelieferten Nightly-Kopf, nicht diesen Fit).
* Nicht, WARUM die halbe Aufloesung besser ist (Vermutung: Glaettung der
  Kanten macht das Template-Signal stabiler).
* Nicht die maschinell gelabelten Aufnahmen als Hauptmass: deren Labels
  stammen vom Detektor, der selbst halb dekodiert (Zirkularitaet).
