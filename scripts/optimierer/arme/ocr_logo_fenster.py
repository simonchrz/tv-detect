"""OCR- und Logo-Fenster: gerichteter Zeitkontext nur ueber Bildinhalt.

Je Aufnahme kommen 6 Spalten dazu, jeweils als ANTEIL der Sekunden im
Fenster (Anteil statt Maximum, damit die Spalte bei schritt=4 im Training
und schritt=1 in Kalibrierung/Eval dieselbe Verteilung hat):

* ocr_hinweis > 0.5 in den letzten / naechsten FENSTER s
* ocr_werbung > 0.5 in den letzten / naechsten FENSTER s
* Mittel des Logo-Werts in den letzten / naechsten FENSTER s

Die aktuelle Zeile zaehlt nicht mit (sie steht schon als Spalte da).

Grund: Trailer mit Sendetermin-Einblendung tragen die Karte ("Montag
20:15") erst am Ende, Countdown-Einzelspots die Karte "Werbung" am Anfang;
die Sekunden dazwischen sehen pointwise aus wie Sendung ohne Logo.
Trainingsteil (61 Aufnahmen mit OCR-Spur, jede 4. s): bei Logo < 0.5 und
ohne Karte jetzt ist der Werbeanteil 0.55; mit Hinweis-Karte in den
naechsten 30 s 0.90, mit Werbung-Karte in den letzten 30 s 0.91. Bei
Logo >= 0.5: 0.022 gegen 0.20. O16 hatte Kontext aus NN-Signal-Spalten,
nicht aus OCR/Logo (Ledger: Wiedervorlage nur ueber Bildinhalt).
"""
import numpy as np

BESCHREIBUNG = ("OCR-/Logo-Fenster: Anteil ocr_hinweis/ocr_werbung-Karten und "
                "Logo-Mittel in den letzten/naechsten 30 s (6 Spalten)")

FENSTER = 30.0


def _vor_zurueck(x, w, leer=0.0):
    """Mittel von x ueber die w Zeilen davor bzw. danach (ohne die Zeile)."""
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    i = np.arange(n)
    a0, b0 = np.maximum(0, i - w), i
    a1, b1 = i + 1, np.minimum(n, i + 1 + w)
    zur = np.where(b0 > a0, (c[b0] - c[a0]) / np.maximum(b0 - a0, 1), leer)
    vor = np.where(b1 > a1, (c[b1] - c[a1]) / np.maximum(b1 - a1, 1), leer)
    return zur, vor


def spalten(teil, X, ktx):
    namen = ktx["spalten_namen"]
    ih = namen.index("ocr_hinweis")
    iw = namen.index("ocr_werbung")
    il = namen.index("logo")
    rec = np.asarray(ktx["rec"])
    w = max(1, int(round(FENSTER / float(ktx["schritt"]))))
    zus = np.zeros((len(X), 6), np.float32)
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        h = (X[m, ih] > 0.5).astype(np.float64)
        wb = (X[m, iw] > 0.5).astype(np.float64)
        lg = np.nan_to_num(X[m, il].astype(np.float64), nan=0.5)
        hz, hv = _vor_zurueck(h, w)
        wz, wv = _vor_zurueck(wb, w)
        lz, lv = _vor_zurueck(lg, w, leer=0.5)
        zus[m] = np.stack([hz, hv, wz, wv, lz, lv], 1)
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
