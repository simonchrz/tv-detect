"""Daten: Trainingsaufnahmen mit nachweislich gebrochenem Label-Zustand herausnehmen.

Nicht nach Herkunft (das waere auto_rest bzw. O17/O18), sondern nach Zustand:
nur Aufnahmen, deren Label im Trainingsteil selbst als falsch erkennbar ist.

1. Zeitachsen-Versatz: Label (VOD-Achse) und Merkmale (Quell-.ts) liegen
   gegeneinander verschoben. Die Produktion quarantaeniert > 15 s
   (zeitachsen-versatz.json, train-head.zeitachsen_quarantaene), K traegt
   beide noch: dvr-rtl-1779473700 (+1269 s, Let's Dance) und
   dvr-rtl-1781199600 (+40 s). Liste fest aus der Produktionsdatei vom
   Stand 17.10. (wie die O25-Liste in 014).
2. Erfundene Werbung unter sichtbarem Logo: reine Maschinenlabels
   (which=auto) mit mindestens einem Werbeblock >= 60 s, dessen mittlere
   Logo-Spalte > 0.6 ist. Im Trainingsteil 8 Aufnahmen (Beet-Brueder x4,
   Bella Italia x2, Let's Dance, Mein Lokal), 16 von 168 auto-Bloecken
   (9.5 %) gegen 12 von 911 menschlich gepruefte (1.3 %); out-of-fold liegt
   der Fehler dort fast ganz auf "Label Werbung, Kopf Sendung" (fn 0.13-0.30
   bei fp ~0). Beet-Brueder-Autolabels mit 47 % Werbung sind schon im
   August als kaputt belegt (Memory tv_detect_rtl_is_a_show_problem).

Zusammen 9 Aufnahmen, ~15k von 506k Trainingszeilen (3 %). Menschlich
gepruefte Aufnahmen bleiben unberuehrt (Dauerwerbung mit Logo ist dort
Konvention). Kalibrierung und Eval unberuehrt; benutzt werden nur ktx["y"],
die Logo-Spalte und meta["which"] der TRAININGS-Aufnahmen. Kein Zufall.
"""
import numpy as np

BESCHREIBUNG = ("Daten: 9 Trainingsaufnahmen mit gebrochenem Label-Zustand raus "
                "(Zeitachsen-Versatz > 15 s; auto-Label mit Werbeblock >= 60 s unter "
                "sichtbarem Logo)")

VERSATZ = frozenset([
    "dvr-rtl-1779473700",   # +1269.3 s
    "dvr-rtl-1781199600",   # +40.3 s
])
MIN_BLOCK_S = 60
LOGO_SCHWELLE = 0.6

_stand = {}


def _bruch(X, ktx):
    j = ktx["spalten_namen"].index("logo")
    rec = np.asarray(ktx["rec"])
    y = np.asarray(ktx["y"])
    schritt = int(ktx["schritt"])
    meta = ktx["meta"]
    uuids = ktx["uuids"]
    raus = np.zeros(len(uuids), bool)
    for k, u in enumerate(uuids):
        if u in VERSATZ:
            raus[k] = True
            continue
        if (meta.get(u) or {}).get("which") != "auto":
            continue
        m = rec == k
        if not m.any():
            continue
        yy = y[m].astype(np.int8)
        lg = np.asarray(X[m, j], np.float64)
        d = np.diff(np.r_[0, yy, 0])
        anf = np.flatnonzero(d == 1)
        end = np.flatnonzero(d == -1)
        for a, b in zip(anf, end):
            if (b - a) * schritt >= MIN_BLOCK_S and lg[a:b].mean() > LOGO_SCHWELLE:
                raus[k] = True
                break
    return raus


def spalten(teil, X, ktx):
    if teil == "training":
        _stand["raus"] = _bruch(X, ktx)
    return X


def gewichte(ktx):
    raus = _stand.get("raus")
    if raus is None or len(raus) != len(ktx["uuids"]):
        raise RuntimeError("spalten(training) muss vor gewichte laufen")
    return np.where(raus[np.asarray(ktx["rec"])], 0.0, 1.0).astype(np.float32)
