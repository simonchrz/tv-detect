"""Bilddynamik: wie stark sich das Bild ueber +-30 s veraendert.

Merkmal je Sekunde: d = 1 - Kosinus zwischen dem Backbone dieser Sekunde und
dem 4 s davor, auf dem ROHEN Backbone. Als Spalten kommen dazu:

* bw_fenster  Mittel von d ueber +-30 s (die Spalte, die im Proxy traegt)
* bw_still    Anteil der Sekunden mit d < STILL im selben Fenster

Werbung schneidet schnell, Sendung verweilt: im Trainingsteil (191 Aufnahmen,
Befund docs/optimierer/abgelehnt/bildwechsel_dichte.befund.md) trennt das
Fenster-Mittel allein mit AUC 0.868 (punktweise d nur 0.667).

Vorpruefung 17.10. (nur Trainingsteil, 96 bzw. 102 Aufnahmen, jede 4. s):
Zentrierung entscheidet. Global zentriert (wie im Befund) AUC 0.874, je
Aufnahme zentriert 0.702, ROH 0.917/0.934. Warum roh so viel besser
trennt, ist nicht geklaert; das Urteil faellt ohnehin am vollen Kopf.
Still-Schwelle 0.1: AUC 0.77, Korrelation zum Fenster-Mittel nur -0.72
(ab 0.2 ueber -0.88, dann waere es dieselbe Spalte zweimal). Im Proxy
(HistGB auf Backbone-PCA48 + logo + audio, Aufnahmen haelftig geteilt) sinkt
der Fehler in den Verwechslungszonen um ein Drittel, und der Gewinn bleibt
nach Glaettung der Vorhersage ueber +-32 s fast voll -- also neue Information,
kein verkappter Dekoder. An Kanten hilft es im Proxy nicht.

Abgrenzung: O8/O8b ("temporal") waren Spalten aus dem NN-Signal bzw. der
Position, 004 (ocr_logo_fenster) Fenster ueber OCR-Karten und Logo. Hier ist
es eine eigene Groesse aus dem Bildinhalt; der Kopf sieht sonst nur die
einzelne Sekunde.

Schritt: im Training ist jede Zeile 4 s, in Kalibrierung/Eval 1 s. d nimmt
deshalb immer den Abstand zu t-4 s (Training: Vorzeile, sonst 4 Zeilen
zurueck), das Fenster immer +-30 s; die Spaltenmittel stimmen zwischen
beiden Schritten auf 3 Stellen ueberein (Befund, 12 Aufnahmen). Nutzt keine
Labels.
"""
import numpy as np

BESCHREIBUNG = ("Merkmale: Bilddynamik (1-cos Backbone zu t-4 s) als Mittel und "
                "Still-Anteil ueber +-30 s (2 Spalten)")

FENSTER = 30.0
ABSTAND = 4.0
STILL = 0.1


def _fenster_mittel(x, w):
    """Mittel von x ueber die Zeilen i-w..i+w (am Rand verkuerzt)."""
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    i = np.arange(n)
    a, b = np.maximum(0, i - w), np.minimum(n, i + w + 1)
    return (c[b] - c[a]) / (b - a)


def spalten(teil, X, ktx):
    schritt = float(ktx["schritt"])
    lag = max(1, int(round(ABSTAND / schritt)))
    w = max(1, int(round(FENSTER / schritt)))
    rec = np.asarray(ktx["rec"])
    zus = np.zeros((len(X), 2), np.float32)
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        B = np.nan_to_num(np.asarray(X[m, :1280], np.float64))
        B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-8)
        d = np.empty(len(m))
        if len(m) > lag:
            d[lag:] = 1.0 - np.sum(B[lag:] * B[:-lag], 1)
            d[:lag] = d[lag]
        else:
            d[:] = 0.0
        zus[m, 0] = _fenster_mittel(d, w)
        zus[m, 1] = _fenster_mittel((d < STILL).astype(np.float64), w)
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
