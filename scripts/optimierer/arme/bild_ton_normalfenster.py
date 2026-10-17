"""Bild-Ton-Normalfenster: Werbeminuten liegen in Bild UND Ton eng im Normalband.

Ein Werbeblock ist ein Gemisch vieler kurzer, lautheitsregulierter Spots.
Ueber eine Minute gemittelt landet er deshalb in zwei Groessen in einem
engen Band: der Ton nahe am Normalpegel der Aufnahme (R128, Lauf 036) und
das Bild nahe am globalen Mittelbild des Backbone-Raums (Fenstermedian der
PCA-Abweichung). Sendungsstrecken (eine Szene, ein Stil, Musik, Stille,
gestoerte Quelle) liegen in beiden Groessen breit daneben. K sieht beides
nur je Sekunde.

Nachbearbeitung vor der Platt-Kalibrierung, auf dem rohen Ensemble-Logit z:

    Ar = |Median60s(rerr) - 105.5| / 10      Rest ausserhalb PCA-64 (Backbone)
    Am = |Median60s(maha) -  65.7| / 10      Mahalanobis in PCA-64
    A  = |Median60s(audio) - Median(audio, Aufnahme)|
    z1 = z  - 0.27*Ar - 0.35*Am - 0.12*Am*max(z,0)        (Bild)
    z' = z1 - 20*A - 5*max(z1,0)*A                         (Ton, wie 036)

PCA (Mittel, 64 Achsen, Varianzen) aus dem Trainingsteil, feste
gleichabstaendige Zeilenauswahl, kein Zufall. Zentren = Median der
Fenstermediane in Trainings-Werbezeilen; Koeffizienten aus logistischer
Regression auf out-of-fold-Logits eines K-gleichen Kopfs (Haelften nach
sha1, 2 Seeds, stabil). Ton-Term unveraendert aus Lauf 036. Platt wird
danach auf der ebenso behandelten Kalibrierung neu angepasst.
Keine Labels ausserhalb des Trainingsteils, nur Spalten der Aufnahme selbst.
"""
import numpy as np
from scipy.ndimage import median_filter

BESCHREIBUNG = ("Bild-Ton-Normalfenster: 60-s-Fenstermedian der Backbone-PCA-Abweichung "
                "und des Tonpegels ausserhalb des Werbebands senkt den Werbe-Logit")

FENSTER_S = 60.0
N_PCA = 64
N_STICH = 60000
Z_RERR, Z_MAHA, SKALA = 105.5, 65.7, 10.0
C_R, C_M, C_MZ = -0.27, -0.35, -0.12
C_A, C_AZ = -20.0, -5.0

_PCA = {}
_TEIL = {}


def spalten(teil, X, ktx):
    namen = ktx["spalten_namen"]
    B = np.asarray(X[:, :1280], np.float64)
    if teil == "training" or not _PCA:
        idx = np.linspace(0, len(B) - 1, min(N_STICH, len(B))).astype(np.int64)
        mu = B[idx].mean(0)
        _, s, Vt = np.linalg.svd(B[idx] - mu, full_matrices=False)
        _PCA.update(mu=mu, V=Vt[:N_PCA], var=np.maximum(s[:N_PCA] ** 2 / len(idx), 1e-9))
    Bc = B - _PCA["mu"]
    P = Bc @ _PCA["V"].T
    rerr = (Bc ** 2).sum(1) - (P ** 2).sum(1)
    maha = ((P ** 2) / _PCA["var"]).sum(1)
    _TEIL[teil] = {"rerr": rerr, "maha": maha,
                   "audio": np.asarray(X[:, namen.index("audio")], np.float64).copy()}
    return X


def _fenster(x, rec, w, rel=False):
    o = np.zeros(len(x), np.float64)
    for r in np.unique(rec):
        j = np.flatnonzero(rec == r)
        v = x[j]
        m = median_filter(v, size=min(w, max(1, len(v) | 1)), mode="nearest")
        o[j] = m - np.median(v) if rel else m
    return o


def nachbearbeiten(p, rec, ktx):
    d = _TEIL.get(ktx["teil"])
    if d is None or len(d["audio"]) != len(p):
        return p
    rec = np.asarray(rec)
    w = max(1, int(round(FENSTER_S / max(1, ktx["schritt"]))))
    if w % 2 == 0:
        w += 1
    Ar = np.abs(_fenster(d["rerr"], rec, w) - Z_RERR) / SKALA
    Am = np.abs(_fenster(d["maha"], rec, w) - Z_MAHA) / SKALA
    A = np.abs(_fenster(d["audio"], rec, w, rel=True))
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    z1 = z + C_R * Ar + C_M * Am + C_MZ * Am * np.maximum(z, 0.0)
    z2 = z1 + C_A * A + C_AZ * np.maximum(z1, 0.0) * A
    return (1.0 / (1.0 + np.exp(-z2))).astype(np.asarray(p).dtype)
