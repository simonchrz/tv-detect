"""Fremdtitel-Wiederkehr: lief dieses Bild schon in ANDEREN Sendungen?

Werbespots und Trailer laufen in vielen fremden Titeln, Sendungsbilder fast
nur in der eigenen Serie. Der Kopf sieht jede Sekunde allein; ob genau diese
Bildfolge auch in drei anderen Titeln lief, weiss er nicht. Das trifft die
Verwechslungszonen (Zeichentrick-Sendung gegen Zeichentrick-Werbung: der
Stil ist gleich, aber nur der Spot kehrt in fremden Titeln wieder) und
Trailer mit Sendetermin-Einblendung (laufen ueber viele Sendungen).

Je Zeile kommen 3 Spalten dazu: die beste Aehnlichkeit zum besten, zweit-
und drittbesten FREMDEN Titel im Trainingsteil (Titel der Zeile selbst
ausgeschlossen, also auch die eigene Aufnahme). Aehnlichkeit wie
scripts/wiederholung.py: Backbone ZENTRIERT (roher Cosinus hat einen
Sockel von ~0.68), Deskriptor = Mittel der Sekunden t und t+4, PCA 64,
L2-normiert, Cosinus.

Gleiche Verteilung in allen Teilen: die Bank sind die Trainingszeilen
(jede 4. Sekunde, beliebige Phase je Aufnahme); der Deskriptor nimmt in
jedem Teil die Sekunden t und t+4 (Training: Zeile i und i+1; Kalibrierung
und Eval: Zeile t und t+4, ueber ktx["schritt"]). Die Bank enthaelt keine
Labels, nur Merkmale; Kalibrierung und Eval sehen nur die Trainingsbank.
Kein Zufall: PCA aus einer festen Teilmenge (jede 7. Zeile), deterministisch.

Diagnose auf dem Trainingsteil (748 Aufnahmen, jede 4. s, 573k Zeilen,
Grundrate Werbung 0.239): dritter fremder Titel >= 0.85 bei 10 % der
Zeilen, davon 84 % Werbung; >= 0.90 bei 6.8 %, 96 % Werbung. Kopf K gegen
K + diese 3 Spalten auf einem Drittel der Trainingsaufnahmen als Holdout
(Hash), zwei Seeds: Sekundenfehler 0.0796/0.0820 -> 0.0756/0.0771, Median
des Fehlers je Aufnahme 0.0430/0.0432 -> 0.0382/0.0382, logloss
0.245/0.252 -> 0.239/0.241. Rechenzeit ~0.07 s je 1000 Zeilen (MPS).

Anders als O24 (Wiederholungs-Anker als LABEL, dazu defekt eingelesen):
hier ist die Wiederholung ein Eingang, kein Ziel; der Kopf lernt selbst,
wie viel er ihr glaubt (Vorspann/Abspann der eigenen Serie zaehlen nicht,
der Titel ist ausgeschlossen).
"""
import numpy as np

BESCHREIBUNG = ("Fremdtitel-Wiederkehr: beste zentrierte Backbone-Aehnlichkeit "
                "zum 1./2./3. fremden Titel im Trainingsteil (3 Spalten)")

N_PCA = 64
STUECK = 512
CHUNK = 512
ABSTAND_S = 4.0

_BANK = {}


def _titel(meta, u):
    m = meta.get(u) or {}
    t = (m.get("title") or "").strip().lower()
    return t or ("?" + u)


def _deskriptor(B, rec, schritt):
    """Mittel der Zeilen t und t + 4 s innerhalb derselben Aufnahme."""
    k = max(1, int(round(ABSTAND_S / float(schritt))))
    n = len(B)
    j = np.minimum(np.arange(n) + k, n - 1)
    gleich = rec[j] == rec
    j = np.where(gleich, j, np.arange(n))
    return 0.5 * (B[:n] + B[j])


def _projiziere(D, mu, V):
    Z = (D - mu) @ V
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-6)
    return Z.astype(np.float32)


def _bank_bauen(X, ktx):
    rec = np.asarray(ktx["rec"])
    B = np.asarray(X[:, :1280], np.float32)
    D = _deskriptor(B, rec, ktx["schritt"])
    mu = D.mean(0, keepdims=True)
    S = D[::7] - mu
    _, _, Vt = np.linalg.svd(S.astype(np.float64), full_matrices=False)
    V = Vt[:N_PCA].T.astype(np.float32)
    Z = _projiziere(D, mu, V)
    namen = sorted({_titel(ktx["meta"], u) for u in ktx["uuids"]})
    tid = {t: i for i, t in enumerate(namen)}
    rec_t = np.array([tid[_titel(ktx["meta"], u)] for u in ktx["uuids"]], np.int64)
    zt = rec_t[rec]
    # Bank nach Titel ordnen und je Titel in Stuecke zu STUECK Zeilen teilen
    # (Rest mit Nullvektoren aufgefuellt, Aehnlichkeit 0). Je Anfrage reicht
    # dann ein Maximum je Stueck und ein reduceat je Titel, kein topk.
    stuecke, stueck_t = [], []
    for t in range(len(namen)):
        zeilen = Z[zt == t]
        for s in range(0, len(zeilen), STUECK):
            st = zeilen[s:s + STUECK]
            if len(st) < STUECK:
                st = np.concatenate([st, np.zeros((STUECK - len(st), Z.shape[1]), np.float32)])
            stuecke.append(st)
            stueck_t.append(t)
    stueck_t = np.asarray(stueck_t, np.int64)
    anfang = np.flatnonzero(np.r_[True, stueck_t[1:] != stueck_t[:-1]])
    _BANK.clear()
    _BANK.update({"mu": mu, "V": V, "Zp": np.concatenate(stuecke), "n_st": len(stueck_t),
                  "anfang": anfang, "titel_der_gruppe": stueck_t[anfang], "tid": tid})


def _merkmale(Zq, qt):
    """Je Anfrage: beste Aehnlichkeit zum 1./2./3. fremden Titel."""
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Zb = torch.from_numpy(_BANK["Zp"]).to(dev)
    n_st = _BANK["n_st"]
    anfang, tg = _BANK["anfang"], _BANK["titel_der_gruppe"]
    aus = np.zeros((len(Zq), 3), np.float32)
    for s in range(0, len(Zq), CHUNK):
        q = torch.from_numpy(Zq[s:s + CHUNK]).to(dev)
        m = (q @ Zb.T).view(len(q), n_st, STUECK).amax(2).cpu().numpy()
        jt = np.maximum.reduceat(m, anfang, axis=1)
        jt[tg[None, :] == qt[s:s + CHUNK, None]] = -2.0
        k = min(3, jt.shape[1])
        top = -np.sort(-jt, axis=1)[:, :k]
        aus[s:s + len(q), :k] = np.maximum(top, 0.0)
    return aus


def spalten(teil, X, ktx):
    rec = np.asarray(ktx["rec"])
    if teil == "training":
        _bank_bauen(X, ktx)
    if not _BANK:
        raise RuntimeError("Bank fehlt: spalten('training') muss zuerst laufen")
    B = np.asarray(X[:, :1280], np.float32)
    Zq = _projiziere(_deskriptor(B, rec, ktx["schritt"]), _BANK["mu"], _BANK["V"])
    tid = _BANK["tid"]
    rec_t = np.array([tid.get(_titel(ktx["meta"], u), -1) for u in ktx["uuids"]], np.int64)
    zus = _merkmale(Zq, rec_t[rec])
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
