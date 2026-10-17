"""Sender-Massstab fuer das Logo: der Kopf erfaehrt, auf welcher Skala das Logo
dieses Senders liegt.

Befund (Trainingsteil, 527 Aufnahmen mit Werbung, jede 4. s): der Logo-Wert
in Sendung hat je Sender einen ganz anderen Median -- vox 0.89, prosieben
0.80, kabel-eins/rtlzwei 1.00, aber disney-channel 0.53, nick 0.33,
comedy-central 0.29. Werbung liegt ueberall bei 0.03-0.15. Ein Logo von 0.3
heisst bei vox "Werbung", bei comedy-central "Sendung". K kennt den Sender
nicht (die Produktion traegt seit mlp32 keine Kanal-Spalten) und muss die
Skala aus dem Bild erraten.

Zwei Spalten, beide aus dem Trainingsteil geschaetzt und je Sender fest:

* logo_sender  (logo - Werbe-Median des Senders) / (Sendungs-Median -
               Werbe-Median), auf [-1, 2] gekappt
* sender_show  Sendungs-Median des Logos fuer diesen Sender (Konstante je
               Sender; beschreibt die Skala, keine Label-Rate)

Medianwerte nur fuer Sender mit >= 2000 Sendungs- und >= 500 Werbezeilen im
Trainingsteil; sonst (auch Sender ohne Trainingsaufnahme, z. B. toggo,
super-rtl, und Aufnahmen ohne slug) gelten die globalen Mediane. Der Sender
ist bei der Erkennung bekannt (slug), es wird nichts vom Label der
bewerteten Aufnahme benutzt. Kein Zeitfenster, kein Zufall. Die Statistik
entsteht beim Aufruf fuer den Trainingsteil, der im Messgeraet immer zuerst
kommt.
"""
import numpy as np

BESCHREIBUNG = ("Logo auf Sender-Skala: (logo - Werbe-Median) / (Sendungs- - Werbe-Median) "
                "je Sender plus Sendungs-Median des Senders (2 Spalten, aus dem Trainingsteil)")

MIN_SENDUNG = 2000
MIN_WERBUNG = 500

_STAT = {}


def _slugs(ktx):
    meta = ktx["meta"]
    return [((meta.get(u) or {}).get("slug") or "") for u in ktx["uuids"]]


def _logo(X, ktx):
    il = ktx["spalten_namen"].index("logo")
    return np.nan_to_num(np.asarray(X[:, il], np.float64), nan=0.5)


def _anpassen(X, ktx):
    lg = _logo(X, ktx)
    y = np.asarray(ktx["y"])
    rec = np.asarray(ktx["rec"])
    sl = np.array(_slugs(ktx), dtype=object)[rec]
    glob = (float(np.median(lg[y == 0])), float(np.median(lg[y == 1])))
    je = {}
    for s in set(sl.tolist()):
        if not s:
            continue
        m = sl == s
        ms, mw = m & (y == 0), m & (y == 1)
        if ms.sum() >= MIN_SENDUNG and mw.sum() >= MIN_WERBUNG:
            je[s] = (float(np.median(lg[ms])), float(np.median(lg[mw])))
    _STAT.clear()
    _STAT.update({"glob": glob, "je": je})


def spalten(teil, X, ktx):
    if teil == "training":
        _anpassen(X, ktx)
    if not _STAT:
        raise RuntimeError("Sender-Statistik fehlt: Trainingsteil muss zuerst kommen")
    lg = _logo(X, ktx)
    rec = np.asarray(ktx["rec"])
    zus = np.zeros((len(X), 2), np.float32)
    for i, s in enumerate(_slugs(ktx)):
        m = rec == i
        ss, sw = _STAT["je"].get(s, _STAT["glob"])
        zus[m, 0] = np.clip((lg[m] - sw) / max(ss - sw, 0.05), -1.0, 2.0)
        zus[m, 1] = ss
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
