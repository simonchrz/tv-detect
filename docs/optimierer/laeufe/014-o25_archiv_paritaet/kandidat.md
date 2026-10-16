# Kandidat 014-o25_archiv_paritaet: o25_archiv_paritaet

**Entscheidung:** „übertragen“ oder „nein“. Ein Kandidat ist NICHT
ausgeliefert. Übertragen heißt: in `train-head.py` nachbauen, dann entscheidet
das Nachtgate wie immer (L3).

## Hypothese

Die Produktion trainiert seit 24.09. ohne die 90 Aufnahmen aus O25 (which=auto, nur über den rec_dir-Defekt ins Archiv gekommen). Diesen Ausschluss bildet K nicht nach: 86 davon liegen im Trainingsteil und stellen 12 % der Zeilen. Der Arm nimmt sie wie in der Produktion heraus (Gewicht 0). Er prüft so, ob diese Hygiene den Kopf nicht verschlechtert. Fällt er deutlich durch, kostet der O25-Ausschluss die Produktion Qualität.

**Begründung:** Fehlerposten: die Abbild-Lücke. K fehlt laut README die Hygiene der Produktion, und dieser Teil davon ist genau benannt. Die Liste kommt wörtlich aus docs/archiv-ausschluss-o25.json, der Produktionsdatei vom 24.09. Sie enthält 86 Trainings- und 2 Kalibrieraufnahmen, keine aus Entwicklung oder Bestätigung, und der Arm liest keine Labels. Ein Kriterium aus den Meta-Daten gibt es nicht, weil cluster_anchored bei diesen Aufnahmen True ist. Deshalb steht die Liste fest im Arm.

Warum das kein Friedhof-Fall ist: O25 wurde im alten Tagesserien-Regime gemessen, mit einem Rauschboden von etwa 0.008 bei einer Schwelle von 0.010. Das Ergebnis war +0.0005 und 2/5 negativ. Der Ausschluss wurde als Hygiene eingebaut, ohne Nachweis in die eine oder andere Richtung. Dieses Messgerät hat eine Streuung von etwa 0.001 und kann die Frage zum ersten Mal entscheiden.

Gezogene Klasse: daten.

Diagnose auf dem Trainingsteil:
- Proxy mit Hälften nach Aufnahme, K-MLP, 3 Seeds je Richtung, Bewertung auf nicht-maschinellen Aufnahmen: Ohne die O25-Aufnahmen sinkt die AUC von 0.9674 auf 0.9659 (6 von 6 Läufen schlechter). Der Median je Aufnahme steigt von 0.0176 auf 0.0181.
- Andere Daten-Ideen fielen im Proxy durch:
  - alle 147 auto-Aufnahmen weg: Median 0.0178 → 0.0230
  - Gewicht 0 auf auto-Zeilen an Label-Kanten: Kantenfehler unverändert, 0.230 → 0.231
  - Galileo-Countdown-Konvention bei den unkorrigierten Aufnahmen: nur 5 von 95 Kanten betroffen
Befund: /Users/simon/src/tv-detect/scripts/optimierer/arme/o25_archiv_paritaet.befund.md

**Beschreibung im Arm:** Daten: 90 O25-Archiv-Aufnahmen (rec_dir-Defekt, which=auto) wie in der Produktion aus dem Training nehmen

## Zahlen (Block-IoU über den Produktions-HSMM, gepaart gegen die Kontrolle)

| Satz | Aufnahmen | Kontrolle | Median Δ | Δ je Wiederholung |
|---|---|---|---|---|
| Entwicklung (Messsatz ∩ train) | 76 | 0.9301 | +0.0050 | [0.0058, 0.005, 0.0052, -0.0001, 0.0001] |
| Bestätigung (Messsatz ∩ test) | 22 | 0.9701 | +0.0014 | [-0.002, 0.0002, 0.004, 0.0014, 0.0039] |

Rauschen zum Vergleich: eine Wiederholung schwankt um ±0.001. Abbild-Lücke
der Kontrolle zur Produktion: −0.009 (16.10.).

**Größte Gewinne (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
| dvr-comedy-central-1778764500 | South Park (comedy-central) | +0.109 |
| dvr-nick-1782930900 | South Park (nick) | +0.062 |
| dvr-comedy-central-1781026800 | South Park (comedy-central) | +0.048 |

**Verluste über 0.10 (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
| – | – | – |

## Wie belastbar

Bis zu diesem Kandidaten wurde der Entwicklungssatz 14-mal und der
Bestätigungssatz 2-mal befragt. Je höher die erste Zahl, desto eher
ist ein knapper Entwicklungsgewinn Glück; die Bestätigung auf ungesehenen
Aufnahmen ist deshalb das eigentliche Kriterium.

## Was die Übertragung kostet

* **gewichte:** nur Training: Gewichtung in train-head.py nachbauen, kein Go.

## Code des Arms (`scripts/optimierer/arme/o25_archiv_paritaet.py`, Abdruck 13becc308180)

```python
"""Daten: die 90 O25-Archiv-Aufnahmen aus dem Training nehmen (Produktions-Paritaet).

Die Produktion trainiert seit 24.09. mit --archiv-ausschluss ohne diese 90
Aufnahmen (docs/archiv-ausschluss-o25.json: which=auto, kamen nur ueber den
rec_dir-Defekt 05-03..09-23 ins Archiv, keine eigenen Anker). Die Kontrolle K
traegt 86 davon im Trainingsteil (12 % der Zeilen) — ein benannter Teil der
Abbild-Luecke. O25 wurde damals im alten Tagesserien-Regime (Rauschen ~0.008) gemessen
und als Hygiene ohne Nachweis eingebaut; dieses Messgeraet (sd ~0.001) kann
die Frage zum ersten Mal entscheiden.

Proxy auf dem Trainingsteil (Haelften nach Aufnahme, K-MLP, 3 Seeds je
Richtung, Bewertung auf nicht-maschinellen Aufnahmen der anderen Haelfte):
AUC 0.9674 -> 0.9659 (6/6 schlechter), Sekundenfehler +0.0003, Median je
Aufnahme +0.0005. Erwartet ist also ein kleiner VERLUST; faellt der Arm
deutlich durch, ist die Empfehlung an Simon die Umkehr (Ausschluss im
Nightly aufheben), haelt er, ist O25 als kostenlos belegt.

Die Liste stammt aus einer Produktions-Datei vom 24.09., enthaelt nur
Trainings-/Kalibrier-Aufnahmen (keine aus Entwicklung/Bestaetigung) und
nutzt keine Labels.
"""
import numpy as np

BESCHREIBUNG = ("Daten: 90 O25-Archiv-Aufnahmen (rec_dir-Defekt, which=auto) wie in "
                "der Produktion aus dem Training nehmen")

O25 = frozenset([
    "dvr-anixe-1781518500",
    "dvr-comedy-central-1781199600",
    "dvr-comedy-central-1781286000",
    "dvr-comedy-central-1781545200",
    "dvr-comedy-central-1785153600",
    "dvr-comedy-central-1785844800",
    "dvr-deluxe-music-1781470800",
    "dvr-disney-channel-1780947600",
    "dvr-disney-channel-1780955100",
    "dvr-disney-channel-1781125200",
    "dvr-disney-channel-1781125800",
    "dvr-disney-channel-1781151000",
    "dvr-disney-channel-1781237400",
    "dvr-disney-channel-1781468100",
    "dvr-disney-channel-1781470200",
    "dvr-disney-channel-1781472000",
    "dvr-disney-channel-1781473800",
    "dvr-disney-channel-1781475600",
    "dvr-disney-channel-1787371800",
    "dvr-kabel-eins-1780671469",
    "dvr-kabel-eins-1780675029",
    "dvr-kabel-eins-1780930639",
    "dvr-kabel-eins-1780934189",
    "dvr-kabel-eins-1781017046",
    "dvr-kabel-eins-1781020626",
    "dvr-kabel-eins-1781103440",
    "dvr-kabel-eins-1781107020",
    "dvr-kabel-eins-1781189915",
    "dvr-kabel-eins-1781193495",
    "dvr-kabel-eins-1781279761",
    "dvr-kabel-eins-1781535410",
    "dvr-kabel-eins-1781539018",
    "dvr-kabel-eins-1781621730",
    "dvr-nick-1780945500",
    "dvr-nick-1780947000",
    "dvr-nick-1781118300",
    "dvr-nick-1781204700",
    "dvr-nick-1781376000",
    "dvr-prosieben-1780043100",
    "dvr-prosieben-1780377480",
    "dvr-prosieben-1780666677",
    "dvr-prosieben-1780669800",
    "dvr-prosieben-1780679125",
    "dvr-prosieben-1780811400",
    "dvr-prosieben-1780851927",
    "dvr-prosieben-1780915593",
    "dvr-prosieben-1780922368",
    "dvr-prosieben-1780927353",
    "dvr-prosieben-1780938332",
    "dvr-prosieben-1780987477",
    "dvr-prosieben-1781002211",
    "dvr-prosieben-1781074090",
    "dvr-prosieben-1781088532",
    "dvr-prosieben-1781100189",
    "dvr-prosieben-1781101860",
    "dvr-prosieben-1781111128",
    "dvr-prosieben-1781175203",
    "dvr-prosieben-1781246588",
    "dvr-prosieben-1781283901",
    "dvr-prosieben-1781456612",
    "dvr-prosieben-1781532287",
    "dvr-prosieben-1781543122",
    "dvr-prosieben-1781618739",
    "dvr-prosieben-1781620347",
    "dvr-prosieben-1785085457",
    "dvr-rtl-1779865200",
    "dvr-rtl-1780235100",
    "dvr-rtl-1785003300",
    "dvr-rtlzwei-1780229700",
    "dvr-rtlzwei-1781123700",
    "dvr-rtlzwei-1781338500",
    "dvr-rtlzwei-1781435700",
    "dvr-rtlzwei-1781439600",
    "dvr-rtlzwei-1784974500",
    "dvr-rtlzwei-1785075300",
    "dvr-sat-1-1779890400",
    "dvr-sat-1-1779976800",
    "dvr-sixx-1779867600",
    "dvr-vox-1780934400",
    "dvr-vox-1781024400",
    "dvr-vox-1781107200",
    "dvr-vox-1781110800",
    "dvr-vox-1781193600",
    "dvr-vox-1781197200",
    "dvr-vox-1781280000",
    "dvr-vox-1781283600",
    "dvr-vox-1781453400",
    "dvr-vox-1781542800",
    "dvr-vox-1781629200",
    "dvr-vox-1785082200",
])


def gewichte(ktx):
    rec = ktx["rec"]
    raus = np.array([u in O25 for u in ktx["uuids"]], bool)
    return np.where(raus[rec], 0.0, 1.0).astype(np.float32)
```
