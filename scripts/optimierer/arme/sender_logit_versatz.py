"""Sender-Logit-Versatz: fester, halbierter Logit-Versatz je Sender vor Platt.

K kennt den Sender nicht. Out-of-fold (K-gleicher Kopf, Haelften nach
sha1(uuid), 2 Seeds, Trainingsteil) liegt der Kopf je Sender systematisch
daneben, und zwar auch im Blockinneren (> 16 s von jeder Labelkante), also
nicht als Kanten-Konvention: kabel-eins und rtlzwei verpassen Werbung
(Inneres: FN 1964/1757 gegen FP 290/289), prosieben und disney erfinden.
Der logloss-optimale Versatz je Sender ist in beiden Haelften gleich
gerichtet: kabel-eins +1.12/+1.39, rtlzwei +1.22/+1.02, vox +0.44/+0.47,
prosieben -0.79/-0.48, disney-channel -1.20/-0.81.

Ein Kanal als Eingangsspalte (O7) lernt die Lage IN-SAMPLE, wo der Kopf
die Trainingsaufnahmen ohnehin trifft; die Verzerrung zeigt sich erst auf
ungesehenen Aufnahmen. Darum hier ein Versatz aus out-of-fold-Logits, nur
fuer Sender mit >= 20 Trainingsaufnahmen und gleichem Vorzeichen in beiden
Haelften, halbiert (Faktor 0.5 war im Produktions-HSMM-Proxy robust,
Faktor 1.0 nicht). Alle anderen Sender: 0. Platt wird danach auf der
ebenso behandelten Kalibrierung neu angepasst. Kein Zufall, keine Labels:
nur der Sender aus der uuid (bzw. meta slug).
"""
import numpy as np

BESCHREIBUNG = ("Sender-Logit-Versatz: halbierter out-of-fold-Versatz je Sender "
                "(kabel-eins/rtlzwei/vox hoch, prosieben/disney runter) vor Platt")

VERSATZ = {
    "kabel-eins": 0.64,
    "rtlzwei": 0.58,
    "vox": 0.22,
    "prosieben": -0.33,
    "disney-channel": -0.56,
}


def _sender(u, meta):
    u = str(u)
    if u.startswith("dvr-") and "-" in u[4:]:
        return u[4:].rsplit("-", 1)[0]
    m = meta.get(u) or {}
    return m.get("slug") or ""


def nachbearbeiten(p, rec, ktx):
    rec = np.asarray(rec)
    p = np.asarray(p)
    o = np.zeros(len(p), np.float64)
    for i, u in enumerate(ktx["uuids"]):
        v = VERSATZ.get(_sender(u, ktx.get("meta") or {}), 0.0)
        if v:
            o[rec == i] = v
    q = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q)) + o
    return (1.0 / (1.0 + np.exp(-z))).astype(p.dtype)
