"""Aufnahme-Nachbar-Glaettung: Logit-Ausgleich ueber bildgleiche Sekunden derselben Aufnahme.

Eine Sendung zeigt dieselben Kulissen, Figuren und Einstellungen ueber die
ganze Folge, ein Spot, der zweimal laeuft, sieht beim zweiten Mal gleich aus.
Der Kopf urteilt jede Sekunde einzeln; liegt er an einer Stelle daneben,
urteilt er an den bildgleichen Stellen derselben Aufnahme meist richtig.
Der HSMM glaettet nur zeitlich, nicht ueber Bildgleichheit.

Nachbearbeitung vor Platt, auf dem rohen Ensemble-Logit z:
  Graph je Aufnahme: zentriertes, normiertes Backbone; je Sekunde die 16
  aehnlichsten Sekunden mit Zeitabstand >= 45 s (die Zeitnachbarn regelt der
  Dekoder). Kanten mit Kosinus >= 0.5 zaehlen, Gewicht = Kosinus.
  m = gewichtetes Mittel der geclippten (+-8) Nachbar-Logits
  z' = z + 0.3 * (m - clip(z))      (nur wo es Nachbarn gibt)
Platt wird danach auf der ebenso behandelten Kalibrierung neu angepasst.
Keine Labels, kein Zufall, keine Aufnahme-Metadaten: nur Backbone und z
derselben Aufnahme. Der Trainingsteil wird nicht beruehrt.
"""
import numpy as np

BESCHREIBUNG = ("Aufnahme-Nachbar-Glaettung: Logit 30 % zum Mittel der bildaehnlichsten "
                "Sekunden derselben Aufnahme (Kosinus >= 0.5, Abstand >= 45 s)")

K_NACHBARN = 16
MIN_ABSTAND_S = 45.0
TAU = 0.5
ALPHA = 0.3
CLIP = 8.0

_GRAPH = {}


def _graph(B, rec, schritt):
    n = len(rec)
    I = np.zeros((n, K_NACHBARN), np.int64)
    W = np.zeros((n, K_NACHBARN), np.float32)
    min_d = max(1, int(np.ceil(MIN_ABSTAND_S / max(1, schritt))))
    for r in np.unique(rec):
        j = np.flatnonzero(rec == r)
        if len(j) < 2:
            continue
        X = np.asarray(B[j], np.float32)
        X = X - X.mean(0, keepdims=True)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
        t = np.arange(len(j))
        k = min(K_NACHBARN, len(j) - 1)
        for s in range(0, len(j), 1024):
            S = X[s:s + 1024] @ X.T
            S[np.abs(t[s:s + 1024, None] - t[None, :]) < min_d] = -2.0
            ix = np.argpartition(-S, k - 1, axis=1)[:, :k]
            I[j[s:s + 1024], :k] = j[ix]
            W[j[s:s + 1024], :k] = np.take_along_axis(S, ix, 1)
    return I, W


def spalten(teil, X, ktx):
    if teil != "training":
        namen = ktx["spalten_namen"]
        a, b = namen.index("backbone0"), namen.index("backbone1279") + 1
        _GRAPH[teil] = _graph(X[:, a:b], np.asarray(ktx["rec"]), ktx["schritt"])
    return X


def nachbearbeiten(p, rec, ktx):
    g = _GRAPH.get(ktx["teil"])
    if g is None or len(g[0]) != len(p):
        return p
    I, W = g
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    zc = np.clip(z, -CLIP, CLIP)
    w = np.where(W >= TAU, W, 0.0).astype(np.float64)
    sw = w.sum(1)
    hat = sw > 0
    m = np.where(hat, (w * zc[I]).sum(1) / np.maximum(sw, 1e-9), zc)
    z2 = z + ALPHA * (m - zc) * hat
    return (1.0 / (1.0 + np.exp(-z2))).astype(np.asarray(p).dtype)
