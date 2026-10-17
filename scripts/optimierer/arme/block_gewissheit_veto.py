"""Block-Gewissheit-Veto: ein langer Werbe-Lauf ohne sichere Werbesekunden
ist meist erfunden.

Befund (Trainingsteil, 527 Aufnahmen mit Labels, out-of-fold-Abbild nach
Haelften, jede Sekunde): Laeufe mit 31-s-Mittel p > 0.5 von mindestens
120 s sind zu 93 % echte Werbebloecke; bei diesen liegt der Anteil der
Sekunden mit p > 0.9 bei q10 0.69 bis 0.82. Die 60 falschen Laeufe dieser
Laenge (16 000 s erfundene Werbung) haben dagegen im Median nur 0.31 bis
0.50 sichere Sekunden: der Kopf ist dort ueberall halb ueberzeugt, nirgends
sicher (gestoerte Quellen, Spielfilm-Szenen, Zeichentrick ohne Logo).
Der Dekoder sieht nur die Einzelsekunden und baut daraus einen Block.

Regel je Aufnahme (vor der Kalibrierung, gleich in Kalibrierung und Eval):
Lauf = zusammenhaengende Sekunden mit 31-s-Mittel p > 0.5, Laenge >= 120 s.
Ist darin der Anteil p > 0.9 kleiner als 0.4, sinkt jeder Logit des Laufs
um 4. Kuerzere Laeufe bleiben (dort entscheidet der Dekoder schon heute).
Affine Vorkonditionierung A*z + B wie werbe_nachhall (Platt-Startpunkt),
Information unveraendert. Kein Zufall, keine Labels, keine Merkmale.
"""
import numpy as np

BESCHREIBUNG = ("Block-Gewissheit-Veto: Werbe-Laeufe (31-s-Mittel p > 0.5) ab 120 s "
                "mit weniger als 40 % Sekunden p > 0.9 bekommen Logit -4 vor der Kalibrierung")

LAENGE = 120.0   # Sekunden
HOCH = 0.9       # sichere Werbesekunde
ANTEIL = 0.4     # Mindestanteil sicherer Sekunden in einem echten Block
C = 4.0          # Abzug im Logit
FENSTER = 15.0   # Halbbreite des Mittels in Sekunden
A, B = 0.63, -1.10
GRENZE = 13.8


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _mittel(x, k):
    c = np.cumsum(np.r_[0.0, x])
    n = len(x)
    i = np.arange(n)
    a = np.maximum(0, i - k)
    b = np.minimum(n, i + k + 1)
    return (c[b] - c[a]) / (b - a)


def _veto(p, z, schritt):
    n = len(p)
    if n == 0:
        return z
    k = max(1, int(round(FENSTER / schritt)))
    g = (_mittel(p, k) > 0.5).astype(np.int8)
    d = np.diff(np.r_[0, g, 0])
    aus = z.copy()
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (b - a) * schritt < LAENGE:
            continue
        if (p[a:b] > HOCH).mean() < ANTEIL:
            aus[a:b] -= C
    return aus


def nachbearbeiten(p, rec, ktx):
    p = np.asarray(p, np.float64)
    z = np.clip(_logit(p), -GRENZE, GRENZE)
    schritt = float(ktx.get("schritt") or 1)
    aus = z.copy()
    for i in np.unique(rec):
        m = np.flatnonzero(rec == i)
        aus[m] = _veto(p[m], z[m], schritt)
    aus = np.clip(A * np.clip(aus, -GRENZE, GRENZE) + B, -GRENZE, GRENZE)
    return 1.0 / (1.0 + np.exp(-aus))
