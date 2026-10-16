"""Kanal-Ausgleich: seltene Sender bekommen im Training mehr Gewicht.

Jede Trainingszeile wird mit (Gesamtzeilen / Zeilen ihres Senders) ** 0.5
gewichtet, gedeckelt auf das 4-fache des Medians, und auf Mittel 1
normiert. Die Klassengewichte der Kontrolle bleiben (sie wirken im Verlust
zusaetzlich). Sender ohne slug bilden eine eigene Gruppe.

Grund: im Trainingsteil stellen vox/kabel-eins/prosieben/rtlzwei/rtl rund
82 % der Stunden, Kinder- und Nischensender (nick 1.2 %, disney 3.1 %,
sixx 0.9 %, comedy-central 0.9 %, sat-1 0.5 %) zusammen unter 7 %. Der
Kopf passt sich an die grossen Sender an; die Abbild-Luecke sitzt bei den
Kinder-Sendern.
"""
import numpy as np

BESCHREIBUNG = ("Kanal-Ausgleich: Zeilengewicht ~ 1/sqrt(Senderanteil im Training), "
                "Deckel 4x Median, Mittel 1")

ALPHA = 0.5
DECKEL = 4.0


def gewichte(ktx):
    rec = np.asarray(ktx["rec"])
    uuids = ktx["uuids"]
    meta = ktx["meta"]
    gruppe_je_rec = []
    for u in uuids:
        s = (meta.get(u) or {}).get("slug") or "?"
        gruppe_je_rec.append(s)
    gruppen = sorted(set(gruppe_je_rec))
    gidx = {g: i for i, g in enumerate(gruppen)}
    g_je_rec = np.array([gidx[g] for g in gruppe_je_rec], np.int64)
    g_je_zeile = g_je_rec[rec]
    zeilen = np.bincount(g_je_zeile, minlength=len(gruppen)).astype(np.float64)
    zeilen[zeilen < 1] = 1
    wg = (zeilen.sum() / zeilen) ** ALPHA
    wg = np.minimum(wg, DECKEL * np.median(wg))
    w = wg[g_je_zeile]
    w = w / w.mean()
    return w.astype(np.float32)
