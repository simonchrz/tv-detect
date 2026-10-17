"""Merkmale: Sprechdichte aus dem Whisper-Text -- was gesagt wird, ohne
den gelernten Whisper-Klassifikator.

Fehlerposten: NN verpasst/erfindet ueber laengere Strecken (zusammen 42 %
der Verlust-Sekunden), wo das Bild mehrdeutig ist. Der Ton steht in K nur
als ein Pegelwert; WAS gesprochen wird (dichter Werbe-Sprecher gegen Dialog,
Wortwiederholung, Werbe-Vokabular) sieht K nicht. O1 hat die frueher
gelernte Whisper-Wahrscheinlichkeit (`prob`) gemessen, nicht den Text;
`prob` bleibt hier draussen (der Klassifikator wurde auf gelabelten Fenstern
trainiert, also im Messgeraet nicht sauber).

Quelle: ~/.cache/tv-whisper/<uuid>.whisper.json, Fenster [t, t+60) mit
Schritt 30 s (Ausrichtung am Trainingsteil geprueft: AUC von prob gegen
Label am hoechsten bei Fenstermitte t+30). Je Sekunde Mittel der
ueberdeckenden Fenster:

* wortrate      Woerter je Sekunde
* werbewort     Anteil fester Werbe-Woerter (Liste unten, nicht gesucht)
* wortvielfalt  verschiedene Woerter / Woerter
* wortrate_rel, wortvielfalt_rel: relativ zur eigenen Aufnahme
  (Median, IQR), was die Erkennung einer fertigen Aufnahme auch hat
* whisper_da    1, wenn ein Fenster die Sekunde deckt

Ohne JSON alle Spalten 0 (Abdeckung: training 143/665, kalibrierung 4/40,
entwicklung 76/76, bestaetigung 22/22). Standardisierung der 5 Werte auf
den Whisper-Zeilen des Trainingsteils, mal whisper_da, damit "fehlt" als
eigener Punkt (0) bleibt. Kein Zufall, kein Label, kein Meta.
"""
import json
import os
import re

import numpy as np

BESCHREIBUNG = ("Whisper-Text: Wortrate, Werbewort-Anteil, Wortvielfalt (+2 relativ "
                "zur Aufnahme) + whisper_da, ohne gelernte prob")

QUELLE = os.path.expanduser("~/.cache/tv-whisper")
WERBEWORT = re.compile(
    r"\b(euro|prozent|jetzt|gratis|angebot|nur|sparen|erh[aä]ltlich|apotheke|www|"
    r"bestellen|testen|neu|rabatt|kaufen|shop|preis)\b", re.I)
N_SP = 6
_stand = {}


def _je_sekunde(u, n):
    f = os.path.join(QUELLE, f"{u}.whisper.json")
    aus = np.zeros((n, N_SP), np.float64)
    if not os.path.isfile(f):
        return aus
    try:
        d = json.load(open(f))
    except Exception:
        return aus
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    for w in d.get("windows", []):
        a = max(0, int(w.get("t", 0)))
        b = min(n, a + 60)
        if b <= a:
            continue
        ws = (w.get("text") or "").split()
        nw = len(ws)
        acc[a:b] += [nw / 60.0,
                     len(WERBEWORT.findall(" ".join(ws))) / max(nw, 1),
                     len(set(x.lower() for x in ws)) / max(nw, 1)]
        cnt[a:b] += 1
    da = cnt > 0
    if not da.any():
        return aus
    acc /= np.maximum(cnt, 1)[:, None]
    rel = []
    for c in (0, 2):
        v = acc[da, c]
        q75, q25 = np.percentile(v, [75, 25])
        rel.append((acc[:, c] - np.median(v)) / (q75 - q25 + 1e-3))
    aus[:, :3] = acc
    aus[:, 3] = rel[0]
    aus[:, 4] = rel[1]
    aus[:, 5] = da
    aus[~da, :5] = 0.0
    return aus


def spalten(teil, X, ktx):
    s = int(ktx["schritt"])
    rec = ktx["rec"]
    Z = np.zeros((X.shape[0], N_SP), np.float64)
    for i, u in enumerate(ktx["uuids"]):
        m = rec == i
        k = int(m.sum())
        if k == 0:
            continue
        Z[m] = _je_sekunde(u, k * s)[::s][:k]
    da = Z[:, 5] > 0
    if teil == "training" or "mu" not in _stand:
        if da.sum() > 10:
            _stand["mu"] = Z[da, :5].mean(0)
            _stand["sd"] = Z[da, :5].std(0) + 1e-6
        else:
            _stand["mu"] = np.zeros(5)
            _stand["sd"] = np.ones(5)
    Z[:, :5] = (Z[:, :5] - _stand["mu"]) / _stand["sd"] * Z[:, 5:6]
    return np.concatenate([X, Z.astype(np.float32)], 1)
