"""Daten nach Quellen-Zustand: Trainingszeilen in langen logo-losen Strecken aufwerten.

Fehlerposten: NN erfindet lange Phantombloecke ueber gestoerten Quellen
(dvr-super-rtl-1781112000 im Schiedsrichter-Satz: Logo 0.00 ueber die ganze
Aufnahme, K erfindet in jedem Seed einen 15-min-Block). Gestoerte Quellen
zeigen sich in den Merkmalen nur als minutenlanger Logo-Ausfall (Befunde
039/042 und leer_aufnahmen_ersatz: keine NaN-Backbones, keine Standbild-
Strecken, kein flacher Ton).

Zustand im Trainingsteil (jede Sekunde, Logo 31 s median-geglaettet < 0.15):
logo-lose Strecken >= 10 min sind 0.48 % der Sekunden (10 Strecken, 7 von
527 Aufnahmen, Werbeanteil ~0.2), in Entwicklung 1.5 %, Bestaetigung 1.8 %.
Die Produktion trainiert zusaetzlich die 44 leeren auto-confirm-Aufnahmen,
die K nicht laden kann; davon tragen 3 zusammen 3835 s solcher Strecken.
Der Zustand ist im Fit also unterrepraesentiert gegenueber Produktion und
Erkennung (Kovariaten-Verschiebung).

Arm (Klasse daten, kein neues Merkmal, kein Zufall, kein Label): Zeilen, die
in einer logo-losen Strecke von >= 10 min liegen, bekommen Gewicht FAKTOR,
unabhaengig vom Label (Importance-Weighting nach Zustand). Werbebloecke ohne
Logo (3-7 min) bleiben bei 1. Strecken werden aus der Logo-Spalte der
Trainingszeilen berechnet (spalten() gibt X unveraendert zurueck und merkt
sich nur die Maske), Fenster und Laenge mit ktx["schritt"] umgerechnet.
"""
import numpy as np

BESCHREIBUNG = ("Daten: Trainingszeilen in logo-losen Strecken >= 10 min (gestoerte/"
                "logo-lose Quelle) labelunabhaengig auf Gewicht 4")

AUS = 0.15
GLATT_S = 31
MIN_S = 600.0
FAKTOR = 4.0

_stand = {}


def _median_glatt(x, k):
    if k <= 1 or len(x) < 2:
        return x.copy()
    h = k // 2
    p = np.pad(x, h, mode="edge")
    w = np.lib.stride_tricks.sliding_window_view(p, k)
    return np.median(w, axis=1)[:len(x)]


def _lange_aus(lg, schritt):
    k = max(3, int(GLATT_S // schritt) | 1)
    aus = _median_glatt(lg, k) < AUS
    m = np.zeros(len(lg), bool)
    if not aus.any():
        return m
    d = np.diff(np.concatenate([[0], aus.astype(np.int8), [0]]))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if (b - a) * schritt >= MIN_S:
            m[a:b] = True
    return m


def spalten(teil, X, ktx):
    if teil == "training":
        i = ktx["spalten_namen"].index("logo")
        rec = np.asarray(ktx["rec"])
        schritt = int(ktx["schritt"])
        lg = np.asarray(X[:, i], np.float32)
        maske = np.zeros(len(rec), bool)
        for r in np.unique(rec):
            j = np.where(rec == r)[0]
            maske[j] = _lange_aus(lg[j], schritt)
        _stand["maske"] = maske
    return X


def gewichte(ktx):
    rec = np.asarray(ktx["rec"])
    m = _stand.get("maske")
    if m is None or len(m) != len(rec):
        return np.ones(len(rec), np.float32)
    return np.where(m, FAKTOR, 1.0).astype(np.float32)
