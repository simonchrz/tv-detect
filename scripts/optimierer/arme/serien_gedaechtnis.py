"""Serien-Gedaechtnis: wie aehnlich ist diese Sekunde der Sendung bzw. der
Werbung in ANDEREN Folgen derselben Serie?

Verwechslungszonen (13 % der Sekunden, 45 % der Fehler) sind Sekunden,
deren Backbone-Nachbarn ueber den ganzen Korpus das Gegenteil-Label tragen:
Zeichentrick-Sendung gegen Zeichentrick-Spot, Doku-Sendung gegen
Doku-Trailer. Global sieht die Sekunde aus wie das Falsche. Innerhalb der
eigenen Serie ist das anders: Studio, Moderatoren, Figuren und Kulisse
kehren in jeder Folge wieder, und die Werbebloecke einer Serie haben ihre
eigenen Spots. Ein Gedaechtnis nur ueber die Geschwister-Folgen trennt dort,
wo der globale Kopf verwechselt.

Je Zeile kommen 4 Spalten dazu:
* s_sendung: bester Cosinus zu einer Sendungs-Zeile einer ANDEREN Aufnahme
  mit gleichem Titel im Trainingsteil
* s_werbung: dasselbe zu Werbe-Zeilen
* s_werbung - s_sendung
* hat_geschwister (1, wenn der Titel im Trainingsteil eine andere Aufnahme hat)
Aehnlichkeit: Backbone ZENTRIERT (roher Cosinus hat einen Sockel ~0.68),
PCA 64 aus jeder 7. Trainingszeile (deterministisch), L2-normiert.

Ehrlich wie bei der Erkennung: im Training ist die eigene Aufnahme aus der
Bank ausgeschlossen (leave-one-recording-out), Kalibrierung und Eval sehen
nur die Trainingsbank. Die Bank traegt die Trainingslabels (ktx["y"] im
Trainingsteil), Eval-Labels kommen nie vor. Die Bank ist jede 4. Sekunde
der Trainingsaufnahmen; die Anfrage ist eine einzelne Sekunde, also in allen
Teilen dieselbe Verteilung (kein Nachbarzeilen-Kontext, schritt egal).
Kein Zufall.

Anders als 010 (Fremdtitel-Wiederkehr, label-frei, fremde Titel: Delta
-0.0001) und O29-T (Sendungs-Kennung one-hot, +0.0017): hier ist es ein
label-bedingtes Gedaechtnis der eigenen Serie, keine Identitaet und keine
Wiederholungszaehlung. Anders als O24 (Anker als Label): Eingang, kein Ziel.

Diagnose (Trainingsteil 527 Aufn., jede 4. s, 506k Zeilen, leave-one-out):
91 % der Zeilen haben Geschwister; AUC (s_werbung - s_sendung) 0.946
(Logo allein 0.906), bei Logo > 0.6 0.865, bei Logo < 0.2 0.938.
Proxy MLP-96 auf Backbone+Logo+Audio, Haelften nach Aufnahme, 2 Seeds x 2
Richtungen: AUC +0.006..+0.013 in 4/4, Sekundenfehler (geglaettet)
0.039/0.051/0.055/0.049 -> 0.028/0.050/0.039/0.047, Median je Aufnahme in
4/4 kleiner.
"""
import numpy as np

BESCHREIBUNG = ("Serien-Gedaechtnis: bester zentrierter Backbone-Cosinus zu Sendungs- "
                "bzw. Werbezeilen anderer Folgen desselben Titels (4 Spalten)")

N_PCA = 64
CHUNK = 1024

_BANK = {}


def _titel(meta, u):
    m = meta.get(u) or {}
    return (m.get("title") or "").strip().lower()


def _geraet():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _projiziere(B, mu, V):
    Z = (B - mu) @ V
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-6)
    return Z.astype(np.float32)


def _bank_bauen(X, ktx):
    rec = np.asarray(ktx["rec"])
    y = np.asarray(ktx["y"]).astype(np.int64)
    B = np.asarray(X[:, :1280], np.float32)
    mu = B.mean(0, keepdims=True)
    S = (B[::7] - mu).astype(np.float64)
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt[:N_PCA].T.astype(np.float32)
    Z = _projiziere(B, mu, V)
    titel = np.array([_titel(ktx["meta"], u) for u in ktx["uuids"]], object)
    gruppen = {}
    for t in set(titel.tolist()):
        if not t:
            continue
        idx = np.flatnonzero(titel[rec] == t)
        gruppen[t] = {"Z": Z[idx], "y": y[idx], "rec": rec[idx].astype(np.int64),
                      "uuid": [ktx["uuids"][i] for i in np.unique(rec[idx])]}
    _BANK.clear()
    _BANK.update({"mu": mu, "V": V, "gruppen": gruppen})


def _abfrage(Zq, eigen, g):
    """Je Anfragezeile: max Cosinus zu Sendung / Werbung der Gruppe, eigene
    Aufnahme (eigen, Bank-rec-Nummer oder -1) ausgeschlossen."""
    import torch
    dev = _geraet()
    Zb = torch.from_numpy(g["Z"]).to(dev)
    rb = torch.from_numpy(g["rec"]).to(dev)
    yb = torch.from_numpy(g["y"]).to(dev)
    aus = np.zeros((len(Zq), 2), np.float32)
    for s in range(0, len(Zq), CHUNK):
        q = torch.from_numpy(Zq[s:s + CHUNK]).to(dev)
        e = torch.from_numpy(eigen[s:s + CHUNK]).to(dev)
        sim = q @ Zb.T
        sim = sim.masked_fill(e[:, None] == rb[None, :], -2.0)
        for k in (0, 1):
            v = sim.masked_fill(yb[None, :] != k, -2.0).amax(1)
            v = torch.where(v > -1.5, v, torch.zeros_like(v))
            aus[s:s + len(q), k] = v.cpu().numpy()
    return aus


def spalten(teil, X, ktx):
    rec = np.asarray(ktx["rec"])
    if teil == "training":
        _bank_bauen(X, ktx)
    if not _BANK:
        raise RuntimeError("Bank fehlt: spalten('training') muss zuerst laufen")
    Zq = _projiziere(np.asarray(X[:, :1280], np.float32), _BANK["mu"], _BANK["V"])
    zus = np.zeros((len(X), 4), np.float32)
    titel = [_titel(ktx["meta"], u) for u in ktx["uuids"]]
    for t in set(titel):
        g = _BANK["gruppen"].get(t)
        if g is None:
            continue
        aufn = [i for i, tt in enumerate(titel) if tt == t]
        idx = np.flatnonzero(np.isin(rec, aufn))
        if teil == "training":
            eigen = rec[idx].astype(np.int64)
            anderer = len(g["uuid"]) >= 2
        else:
            eigen = np.full(len(idx), -1, np.int64)
            anderer = len(g["uuid"]) >= 1
        if not anderer:
            continue
        r = _abfrage(Zq[idx], eigen, g)
        zus[idx, 0] = r[:, 0]
        zus[idx, 1] = r[:, 1]
        zus[idx, 2] = r[:, 1] - r[:, 0]
        zus[idx, 3] = 1.0
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
