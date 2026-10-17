"""Merkmale: Text-Wiederkehr aus dem Whisper-Transkript -- laeuft dieser
Wortlaut auch in Aufnahmen FREMDER Titel?

Fehlerposten: NN erfindet (1262 s) und NN verpasst (1099 s) ueber laengere
Strecken, wo das Bild mehrdeutig ist (Sendung ohne Logo, Zeichentrick-Sendung
gegen Zeichentrick-Spot, Logo im Spot). Werbespots und Programmhinweise
werden woertlich wiederholt, und zwar quer durch Sendungen; Dialog einer
Sendung kehrt in fremden Titeln fast nie wieder. 039 hat Wortrate,
Werbewort-Anteil und Wortvielfalt gemessen (univariat AUC 0.64), nicht die
Wiederkehr; 010/018/021/023 haben Wiederkehr nur im Bild (Backbone) gemessen.

Quelle: ~/.cache/tv-whisper/<uuid>.whisper.json, Fenster [t, t+60), Schritt
30 s. Referenzkorpus = die Whisper-Fenster des TRAININGSTEILS (Titel aus
ktx["meta"]), aufgebaut beim Aufruf fuer "training" und danach fuer
Kalibrierung und Eval unveraendert benutzt. Je Fenster: Anteil seiner
Wort-5-Gramme, die in mindestens einer Trainingsaufnahme mit ANDEREM Titel
vorkommen (eigener Titel, also auch die eigene Aufnahme und Folgen derselben
Serie, zaehlt nicht -> Vorspann/Abspann einer Serie bleiben Sendung).
Fenster mit weniger als 3 Fuenf-Grammen (Stille, Musik) -> 0.

Spalten je Sekunde:
* wiederkehr      Mittel der ueberdeckenden Fenster
* wiederkehr_max  Maximum der ueberdeckenden Fenster
* whisper_da      1, wenn ein Fenster die Sekunde deckt (sonst alles 0)

Kein Label, kein Zufall, kein Archiv; Meta nur Titel.
"""
import json
import os
import re

import numpy as np

BESCHREIBUNG = ("Whisper-Text-Wiederkehr: Anteil der Wort-5-Gramme eines Fensters, die in "
                "Trainingsaufnahmen fremder Titel vorkommen (Mittel, Max) + whisper_da")

QUELLE = os.path.expanduser("~/.cache/tv-whisper")
K_GRAMM = 5
N_SP = 3
_WORT = re.compile(r"\w+")
_korpus = {}   # gramm -> set(titel), nur aus dem Trainingsteil


def _fenster(u):
    f = os.path.join(QUELLE, f"{u}.whisper.json")
    if not os.path.isfile(f):
        return None
    try:
        return json.load(open(f)).get("windows") or None
    except Exception:
        return None


def _gramme(text):
    w = _WORT.findall((text or "").lower())
    return {" ".join(w[i:i + K_GRAMM]) for i in range(len(w) - K_GRAMM + 1)}


def _titel(ktx, u):
    m = (ktx.get("meta") or {}).get(u) or {}
    return (m.get("title") or "").strip().lower() or u


def _korpus_bauen(ktx):
    _korpus.clear()
    for u in ktx["uuids"]:
        ws = _fenster(u)
        if not ws:
            continue
        ti = _titel(ktx, u)
        for w in ws:
            for g in _gramme(w.get("text")):
                _korpus.setdefault(g, set()).add(ti)


def _je_sekunde(u, n, titel):
    aus = np.zeros((n, N_SP), np.float64)
    ws = _fenster(u)
    if not ws:
        return aus
    summe = np.zeros(n)
    mx = np.zeros(n)
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
        mx[a:b] = np.maximum(mx[a:b], v)
        cnt[a:b] += 1
    da = cnt > 0
    aus[:, 0] = np.where(da, summe / np.maximum(cnt, 1), 0.0)
    aus[:, 1] = np.where(da, mx, 0.0)
    aus[:, 2] = da
    return aus


def spalten(teil, X, ktx):
    if teil == "training":
        _korpus_bauen(ktx)
    s = int(ktx["schritt"])
    rec = ktx["rec"]
    Z = np.zeros((X.shape[0], N_SP), np.float64)
    for i, u in enumerate(ktx["uuids"]):
        m = rec == i
        k = int(m.sum())
        if k == 0:
            continue
        Z[m] = _je_sekunde(u, k * s, _titel(ktx, u))[::s][:k]
    return np.concatenate([X, Z.astype(np.float32)], 1)
