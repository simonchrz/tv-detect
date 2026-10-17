# Kandidat 046-ton_dynamik_paritaet: ton_dynamik_paritaet

**Entscheidung:** „übertragen“ oder „nein“. Ein Kandidat ist NICHT
ausgeliefert. Übertragen heißt: in `train-head.py` nachbauen, dann entscheidet
das Nachtgate wie immer (L3).

## Hypothese

K rechnet mit dem rohen Ton-Pegel. Der deployte Kopf ersetzt diese Spalte seit O22 durch ihre gleitende Standardabweichung über etwa 30 s (head.audio.json dynamik=true, Fenster 30). Werbung ist stark komprimiert und hält ihren Pegel, Sendung schwankt durch Dialog, Musik und Stille. Ersetzt man die Spalte audio in allen Teilen durch die Streuung des Pegels über t-12..t+12 s (7 Stützstellen im Abstand 4 s, identisch bei schritt 4 und schritt 1), schließt K diese Abbild-Lücke zur Produktion. Die NN-Fehler (erfindet und verpasst) sinken dabei leicht, und die Block-IoU wird nicht schlechter.

**Begründung:** Gezogen wurde die Klasse labels (Zug 0.81, nur weil dort 0 Läufe stehen). Ein Label-Arm ist hier nicht sinnvoll. Elf frühere Befunde fanden die Trainingslabels sauber: 032, 040, 041, 044, 045, blockende_logo_nachlauf_kuerzen, logo_werbebloecke_raus, auto_blockende_logo, rand_einblendung_ziel und weitere. Die übrigen Konventionsfälle (Rand-Trailer, Gewinnspiel, Nachlauf) stehen im Entwicklungssatz genauso, ein Label-Arm würde also gegen die Messlatte arbeiten. Ich bin deshalb auf merkmale ausgewichen, die Klasse mit dem zweithöchsten Zug (0.43).

Fehlerposten: Abbild-Lücke sowie NN erfindet/verpasst. Belege auf dem Trainingsteil (524 Aufnahmen, read-only):
- Ton-AUC je Aufnahme, Median: Pegel 0.559, Schwankung 0.596. Nützlich (AUC > 0.55) ist der Pegel in 56 % der Aufnahmen, die Schwankung in 63 %.
- K-Proxy mit vollen Spalten (baue_k, Hälften nach sha1, 2 Seeds): Sekundenfehler 0.0807→0.0803, 0.0813→0.0807, 0.0629→0.0616 und 0.0639→0.0636, also 4/4 besser. FP und FN sinken beide.

Warum nicht Friedhof: 036 und 037 haben den 60-s-Median des Pegels als Nachbearbeitung gemessen, nie die Streuung als Spalte. O22 wurde vor OCR/SigLIP gegen die Produktion gemessen, nie gegen K. Die Produktion hat das schon, deshalb ist dies ein Paritäts-Arm wie 014, und die Regel lautet nicht_schlechter. Der Arm nutzt keinen Zufall und kein Label. Der Schätzer ist in Training und Eval bitgleich, im synthetischen Test geprüft.

**Beschreibung im Arm:** Ton-Dynamik (Produktions-Paritaet O22): Spalte audio durch die Standardabweichung des Pegels ueber t-12..t+12 s (Stuetzstellen alle 4 s) ersetzen

## Zahlen (Block-IoU über den Produktions-HSMM, gepaart gegen die Kontrolle)

| Satz | Aufnahmen | Kontrolle | Median Δ | Δ je Wiederholung |
|---|---|---|---|---|
| Entwicklung (Messsatz ∩ train) | 76 | 0.9341 | +0.0002 | [-0.0001, -0.0006, 0.0017, 0.0002, 0.0014] |
| Bestätigung (Messsatz ∩ test) | 22 | 0.9717 | -0.0006 | [0.0025, -0.0034, -0.0006, -0.0006, 0.0012] |

Rauschen zum Vergleich: eine Wiederholung schwankt um ±0.001. Abbild-Lücke
der Kontrolle zur Produktion: −0.009 (16.10.).

**Größte Gewinne (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
| dvr-nick-1782930900 | South Park (nick) | +0.031 |
| dvr-comedy-central-1781026800 | South Park (comedy-central) | +0.024 |
| dvr-kabel-eins-1779119815 | Mein Lokal, Dein Lokal - Der Profi kommt (kabel-eins) | +0.004 |

**Verluste über 0.10 (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
| – | – | – |

## Wie belastbar

Bis zu diesem Kandidaten wurde der Entwicklungssatz 46-mal und der
Bestätigungssatz 6-mal befragt. Je höher die erste Zahl, desto eher
ist ein knapper Entwicklungsgewinn Glück; die Bestätigung auf ungesehenen
Aufnahmen ist deshalb das eigentliche Kriterium.

## Was die Übertragung kostet

* **spalten:** NEUE EINGABESPALTEN: Training UND Go-Inferenz (nn.go), Header-Bump, Paritaets-Fixture — L5, braucht dein ausdrueckliches OK.

## Code des Arms (`scripts/optimierer/arme/ton_dynamik_paritaet.py`, Abdruck 5ea0419ea080)

```python
"""Ton-Dynamik wie die Produktion: Spalte `audio` durch ihre gleitende
Standardabweichung ueber ~30 s ersetzen (O22, head.audio.json dynamik=true).

K rechnet mit dem rohen Ton-Pegel, der deployte Kopf seit O22 mit der
Schwankung (train-head.audio_dynamik, Fenster 30 s, ersetzt die Spalte,
Breite bleibt). Das ist eine Abbild-Luecke, die das README nicht fuehrt.

Damit Training (jede 4. s) und Kalibrierung/Eval (jede s) denselben
Schaetzer sehen, wird die Streuung in beiden Teilen aus denselben 7 Stuetz-
stellen im Abstand 4 s (t-12 .. t+12 s) gerechnet; Raender werden
abgeschnitten, nicht aufgefuellt (wie die Produktion). Kein Zufall, kein Label.
"""
import numpy as np

BESCHREIBUNG = ("Ton-Dynamik (Produktions-Paritaet O22): Spalte audio durch die "
                "Standardabweichung des Pegels ueber t-12..t+12 s (Stuetzstellen alle 4 s) ersetzen")

HALB_S = 12
STUETZ_S = 4


def _dynamik(a, schritt):
    n = len(a)
    k = max(1, int(round(STUETZ_S / schritt)))
    h = int(round(HALB_S / schritt))
    offs = np.arange(-h, h + 1, k)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    cnt = np.zeros(n)
    idx0 = np.arange(n)
    for o in offs:
        idx = idx0 + o
        ok = (idx >= 0) & (idx < n)
        v = a[idx[ok]]
        s1[ok] += v
        s2[ok] += v * v
        cnt[ok] += 1
    mu = s1 / np.maximum(cnt, 1)
    var = np.maximum(0.0, s2 / np.maximum(cnt, 1) - mu * mu)
    return np.sqrt(var)


def spalten(teil, X, ktx):
    namen = ktx["spalten_namen"]
    ia = namen.index("audio")
    rec = np.asarray(ktx["rec"])
    sek = ktx.get("sek")
    schritt = int(ktx["schritt"])
    X = np.array(X, dtype=np.float32, copy=True)
    a_alt = X[:, ia].astype(np.float64)
    neu = np.zeros(len(a_alt), np.float64)
    for r in np.unique(rec):
        zeilen = np.where(rec == r)[0]
        if sek is not None:
            zeilen = zeilen[np.argsort(np.asarray(sek)[zeilen], kind="stable")]
        neu[zeilen] = _dynamik(a_alt[zeilen], schritt)
    X[:, ia] = neu.astype(np.float32)
    return X
```
