"""Nachbearbeitung: Whisper-Text-Wiederkehr schiebt den Werbe-Logit an.

Fehlerposten: NN verpasst (1099 s) und NN erfindet (1262 s) dort, wo das
Bild mehrdeutig ist. Werbespots und Programmhinweise laufen woertlich quer
durch fremde Sendungen, Sendungsdialog fast nie (040: AUC 0.87-0.91 allein).

Warum nicht noch einmal 040: dort war die Wiederkehr eine SPALTE, die K aus
dem Trainingsteil lernen musste, in dem nur 117 von 527 Aufnahmen (22 %)
ein Transkript haben; fuer 78 % der Zeilen war die Spalte 0 und damit von
"keine Wiederkehr" nicht zu unterscheiden. Eval hat zu 100 % ein Transkript.
Hier lernt K nichts Neues; die Wiederkehr wird nach dem Ensemble auf den
rohen Logit addiert, mit einer festen Staerke aus einem out-of-fold-Proxy
im Trainingsteil (K-gleicher Kopf, Haelften nach sha1(uuid), siehe Befund):

    z' = z + C * W,   W = Mittel der Wiederkehr der ueberdeckenden 60-s-Fenster

Platt wird danach wie immer auf der ebenso behandelten Kalibrierung
angepasst. Ohne Transkript bleibt p unveraendert.

Referenzkorpus = Whisper-Fenster des Trainingsteils (Titel aus der Meta),
gebaut im Aufruf spalten("training"); Merkmale bleiben unveraendert.
Kein Label, kein Zufall; Meta nur Titel.
"""
import json
import os
import re

import numpy as np

BESCHREIBUNG = ("Whisper-Text-Wiederkehr (Wort-5-Gramme in fremden Trainingstiteln) als "
                "fester Logit-Zuschlag nach dem Ensemble, vor Platt (C=5)")

QUELLE = os.path.expanduser("~/.cache/tv-whisper")
K_GRAMM = 5
C = 5.0
_WORT = re.compile(r"\w+")
_korpus = {}   # gramm -> set(titel), nur aus dem Trainingsteil


def _fenster(u):
    f = os.path.join(QUELLE, f"{u}.whisper.json")
    if not os.path.isfile(f):
        return None
    try:
        with open(f) as fh:
            return json.load(fh).get("windows") or None
    except Exception:
        return None


def _gramme(text):
    w = _WORT.findall((text or "").lower())
    return {" ".join(w[i:i + K_GRAMM]) for i in range(len(w) - K_GRAMM + 1)}


def _titel(ktx, u):
    m = (ktx.get("meta") or {}).get(u) or {}
    return (m.get("title") or "").strip().lower() or u


def _wiederkehr(u, n, titel):
    ws = _fenster(u)
    if not ws or n <= 0:
        return None
    summe = np.zeros(n)
    cnt = np.zeros(n)
    for w in ws:
        a = max(0, int(w.get("t", 0)))
        b = min(n, a + 60)
        if b <= a:
            continue
        gs = _gramme(w.get("text"))
        if len(gs) >= 3:
            fremd = sum(1 for g in gs if (_korpus.get(g, ()) and (_korpus[g] - {titel})))
            v = fremd / len(gs)
        else:
            v = 0.0
        summe[a:b] += v
        cnt[a:b] += 1
    return np.where(cnt > 0, summe / np.maximum(cnt, 1), 0.0)


def spalten(teil, X, ktx):
    if teil == "training":
        _korpus.clear()
        for u in ktx["uuids"]:
            ws = _fenster(u)
            if not ws:
                continue
            ti = _titel(ktx, u)
            for w in ws:
                for g in _gramme(w.get("text")):
                    _korpus.setdefault(g, set()).add(ti)
    return X


def nachbearbeiten(p, rec, ktx):
    p = np.asarray(p, np.float64)
    rec = np.asarray(rec)
    s = int(ktx["schritt"])
    q = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    for i, u in enumerate(ktx["uuids"]):
        j = np.flatnonzero(rec == i)
        if len(j) == 0:
            continue
        W = _wiederkehr(u, len(j) * s, _titel(ktx, u))
        if W is None:
            continue
        z[j] = z[j] + C * W[::s][:len(j)]
    return 1 / (1 + np.exp(-z))
