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
