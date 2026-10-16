"""Aufnahme-zentrierte Bildmerkmale gegen die Verwechslungszonen.

Die Verwechslungszonen (13 % der Sekunden, 45 % der Fehler) sind Sekunden,
deren Backbone-Nachbarn in ANDEREN Aufnahmen das Gegenteil-Label tragen:
Zeichentrick-Sendung gegen Zeichentrick-Werbung, Actionfilm gegen
Kinotrailer. Global sieht die Sekunde aus wie das Falsche; relativ zur
eigenen Aufnahme nicht, denn die Sendung stellt rund drei Viertel jeder
Aufnahme, der Median je Aufnahme ist also ein Sendungs-Prototyp.

Je Aufnahme kommen 130 Spalten dazu:

* PCA-64 des Backbones minus Backbone-Median der Aufnahme
  (Projektion nur aus dem Trainingsteil angepasst)
* SigLIP-64 minus SigLIP-Median der Aufnahme
* Cosinus-Abstand Backbone bzw. SigLIP zum Aufnahme-Median
  (beides nach Abzug des globalen Trainingsmittels, sonst Sockel ~0.68)

Der Median ueber die Zeilen ist bei schritt=4 (Training) und schritt=1
(Kalibrierung/Eval) gleich verteilt; es wird nichts ueber Nachbarzeilen
gerechnet. Benutzt wird nur, was die Erkennung einer fertigen Aufnahme
auch hat: die Merkmale derselben Aufnahme.

Diagnose (Trainingsteil, 213 Aufnahmen, jede 8. s, haelftig nach
Aufnahme getrennt): MLP-96 auf Backbone+Logo+Audio Fehler 0.090 ->
0.067 mit diesen Spalten, AUC 0.937 -> 0.951; Kinder-Sender 0.072 -> 0.058.
"""
import numpy as np

BESCHREIBUNG = ("Aufnahme-zentriert: Backbone-PCA64 und SigLIP64 minus Median der "
                "Aufnahme + 2 Abstaende zum Aufnahme-Median (130 Spalten)")

N_PCA = 64
_ZUSTAND = {}


def _bloecke(namen):
    ib = [namen.index(f"backbone{i}") for i in range(1280)]
    isg = [namen.index(f"siglip{i}") for i in range(64)]
    return np.array(ib), np.array(isg)


def _cos_abstand(a, m):
    na = np.linalg.norm(a, axis=1) * (np.linalg.norm(m) + 1e-6) + 1e-6
    return (1.0 - (a @ m) / na).astype(np.float32)


def spalten(teil, X, ktx):
    X = np.asarray(X, np.float32)
    ib, isg = _bloecke(ktx["spalten_namen"])
    rec = np.asarray(ktx["rec"])
    recs = np.unique(rec)
    if teil == "training" or "pca" not in _ZUSTAND:
        gb = X[:, ib].mean(0)
        gs = X[:, isg].mean(0)
        # Kovarianz der zentrierten Backbone-Zeilen aus einer Stichprobe
        cov = np.zeros((1280, 1280), np.float64)
        n = 0
        summe = np.zeros(1280, np.float64)
        for r in recs:
            m = np.flatnonzero(rec == r)
            Bm = X[m][:, ib]
            c = Bm - np.median(Bm, 0)
            c = c[::4].astype(np.float64)
            cov += c.T @ c
            summe += c.sum(0)
            n += len(c)
        mu = summe / max(n, 1)
        cov = cov / max(n, 1) - np.outer(mu, mu)
        w, V = np.linalg.eigh(cov)
        V = V[:, np.argsort(w)[::-1][:N_PCA]].astype(np.float32)
        _ZUSTAND.update(gb=gb, gs=gs, mu=mu.astype(np.float32), pca=V)
    gb, gs, mu, V = (_ZUSTAND[k] for k in ("gb", "gs", "mu", "pca"))
    zus = np.zeros((len(X), N_PCA + 64 + 2), np.float32)
    for r in recs:
        m = np.flatnonzero(rec == r)
        Bm = X[m][:, ib] - gb
        Sm = X[m][:, isg] - gs
        mb = np.median(Bm, 0)
        ms = np.median(Sm, 0)
        zus[m, :N_PCA] = (Bm - mb - mu) @ V
        zus[m, N_PCA:N_PCA + 64] = Sm - ms
        zus[m, -2] = _cos_abstand(Bm, mb)
        zus[m, -1] = _cos_abstand(Sm, ms)
    return np.concatenate([X, zus], 1)
