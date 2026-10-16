"""Kanten-Saum: Trainingszeilen nahe einer Label-Kante bekommen mehr Gewicht,
die Zeilen direkt auf der Kante weniger.

Je Trainingsaufnahme werden die Kanten aus den Werbebloecken des
Trainingslabels genommen (Blockstart und Blockende, Aufnahme-Raender
ausgenommen). Abstand jeder Zeile zur naechsten Kante in Sekunden
(Zeilenindex * ktx["schritt"]):

* <= KERN s       -> Gewicht KERN_W (Konventions-/Lagerauschen des Labels)
* KERN < d <= SAUM -> Gewicht SAUM_W (Uebergangsmaterial: Trailer,
                      Programmhinweise, Rueckkehr-Teaser, Split-Screen)
* sonst           -> 1

Danach Mittel 1. Klassengewichte der Kontrolle bleiben.

Grund: Kante = 51 % der Verlust-Sekunden; das Mittelband 0.15-0.85 an
schlechten Kanten traegt die meiste Fehlermasse (Ledger 3af). Im Training
liegen nur ~6 % der Sekunden 3-30 s neben einer Kante, der Kopf lernt den
Uebergang also kaum, waehrend die Lagefehler des Labels (Median 2-4 s)
direkt auf der Kante als Rauschen wirken.
"""
import numpy as np

BESCHREIBUNG = ("Kanten-Saum: Zeilen 4-30 s neben einer Label-Kante x2.5, "
                "Zeilen <=3 s auf der Kante x0.3, Mittel 1")

KERN = 3.0
SAUM = 30.0
KERN_W = 0.3
SAUM_W = 2.5


def gewichte(ktx):
    rec = np.asarray(ktx["rec"])
    schritt = float(ktx["schritt"])
    meta = ktx["meta"]
    w = np.ones(len(rec), np.float64)
    for i, u in enumerate(ktx["uuids"]):
        m = rec == i
        k = int(m.sum())
        if k == 0:
            continue
        dauer = k * schritt
        bloecke = (meta.get(u) or {}).get("ads") or []
        kanten = []
        for a, b in bloecke:
            a, b = float(a), float(b)
            if a > 1.0:
                kanten.append(a)
            if b < dauer - 1.0:
                kanten.append(b)
        if not kanten:
            continue
        t = np.arange(k, dtype=np.float64) * schritt
        kv = np.sort(np.asarray(kanten, np.float64))
        j = np.searchsorted(kv, t)
        links = np.abs(t - kv[np.clip(j - 1, 0, len(kv) - 1)])
        rechts = np.abs(kv[np.clip(j, 0, len(kv) - 1)] - t)
        d = np.minimum(links, rechts)
        wi = np.ones(k, np.float64)
        wi[(d > KERN) & (d <= SAUM)] = SAUM_W
        wi[d <= KERN] = KERN_W
        w[m] = wi
    w = w / w.mean()
    return w.astype(np.float32)
