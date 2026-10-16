"""Selbstaehnlichkeit innerhalb der Aufnahme: wie sehr gleicht diese Sekunde
dem Rest DERSELBEN Aufnahme?

Sendung wiederholt sich in sich selbst (gleiches Studio, gleiche Figuren,
gleiche Kulisse ueber die ganze Folge), Werbung ist bildlich einmalig — ausser
ein Spot oder Trailer laeuft in einem zweiten Block noch einmal, dann ist die
Aehnlichkeit fast 1. Ein einzelner Backbone-Vektor kann das nicht zeigen, der
Kopf sieht jede Sekunde fuer sich. Das zielt auf die Verwechslungszonen:
Sekunden, deren Backbone-Nachbarn in ANDEREN Aufnahmen das Gegenteil-Label
tragen; die Frage hier richtet sich nur an die eigene Aufnahme.

Anders als 010 (Wiederkehr in fremden Titeln, quer ueber das Archiv) und 005
(Aufnahme-Mittel abziehen) ein relationales Merkmal innerhalb der Aufnahme;
anders als 004/O16 kein Zeitfenster um die Sekunde, Nachbarn unter 90 s sind
ausdruecklich ausgeschlossen.

Spalten (3), auf dem zentrierten Backbone (Mittel aus dem Trainingsteil),
Kosinus gegen einen Vergleichsvorrat mit Abstand >= 90 s:
* max      groesste Aehnlichkeit
* top5     Mittel der fuenf groessten
* anteil   Anteil des Vorrats mit Aehnlichkeit > 0.6

Schritt: der Vorrat ist IMMER jede 4. Sekunde der Aufnahme (im Training
sind das alle geladenen Zeilen, in Kalibrierung/Eval Zeilen mit Sekunde % 4
== 0), damit Maximum und Anteil in allen Teilen aus gleich dichten Vorraeten
kommen. Abstand in Sekunden = Zeilenabstand * ktx["schritt"]. Kein Zufall.
"""
import numpy as np
import torch

BESCHREIBUNG = ("Selbstaehnlichkeit in der Aufnahme: max/top5/Anteil>0.6 Kosinus "
                "des zentrierten Backbones gegen dieselbe Aufnahme, Abstand >= 90 s")

ABSTAND_S = 90
VORRAT_S = 4
_MITTEL = {}


def _merkmale(E, sek, vorrat):
    e = torch.from_numpy(np.ascontiguousarray(E))
    e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    pv = e[vorrat]
    ts = torch.from_numpy(sek.astype(np.int64))
    tv = ts[vorrat]
    aus = np.zeros((len(E), 3), np.float32)
    for a in range(0, len(E), 4096):
        s = e[a:a + 4096] @ pv.T
        nah = (ts[a:a + 4096, None] - tv[None, :]).abs() < ABSTAND_S
        s[nah] = -1.0
        gueltig = (~nah).sum(1).clamp(min=1).float()
        if s.shape[1] == 0:
            continue
        k = min(5, s.shape[1])
        aus[a:a + 4096, 0] = s.max(1).values.numpy()
        aus[a:a + 4096, 1] = s.topk(k, 1).values.mean(1).numpy()
        aus[a:a + 4096, 2] = ((s > 0.6).sum(1).float() / gueltig).numpy()
    # Aufnahmen kuerzer als der Abstand: keine Vergleichssekunde -> -1 bleibt
    aus[:, :2] = np.maximum(aus[:, :2], -1.0)
    return aus


def spalten(teil, X, ktx):
    B = np.asarray(X[:, :1280], np.float32)
    if teil == "training" or "mu" not in _MITTEL:
        _MITTEL["mu"] = B.mean(0)
    mu = _MITTEL["mu"]
    schritt = int(ktx["schritt"])
    rec = np.asarray(ktx["rec"])
    zus = np.zeros((len(X), 3), np.float32)
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        sek = np.arange(len(m)) * schritt
        vorrat = np.flatnonzero(sek % VORRAT_S == 0)
        zus[m] = _merkmale(B[m] - mu, sek, vorrat)
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
