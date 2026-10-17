"""Kinder-Sender-Gewicht: Trainingszeilen von Aufnahmen der Kinder- und
Zeichentrick-Sender (nick, disney-channel, comedy-central, super-rtl,
toggo, kika) zaehlen doppelt.

Fehlerposten: Kinder-Sender sind die groesste Abbild-Luecke (README) und
tragen out-of-fold doppelt so viele Fehlersekunden je Aufnahme wie der Rest
(Proxy 0.067-0.087 gegen 0.033-0.042), fast alles erfundene Werbung ueber
Zeichentrick. Sie stellen nur 4.3 % der Zeilen. 016 (Gewicht 3 nach
Logo-Trennschaerfe, ein Label-abgeleitetes Kriterium) holte seine +0.0029
fast ganz auf comedy-central, nick und super-rtl; hier entscheidet allein
der Sender, der auch bei der Erkennung bekannt ist.

Nur gewichte(); keine Merkmale, kein Ziel, kein Zufall. Sender aus
meta.slug, sonst aus der uuid (dvr-<sender>-<zeit>), sonst aus dem letzten
Eintrag der Kohorte (z. B. ['SpongeBob Schwammkopf', 'Nick']).
"""
import re

import numpy as np

BESCHREIBUNG = ("Daten: Trainingszeilen der Kinder-/Zeichentrick-Sender "
                "(nick, disney, comedy-central, super-rtl, toggo, kika) mit Gewicht 2")

FAKTOR = 2.0
KINDER = ("nick", "disney", "comedy-central", "super-rtl", "toggo", "kika")


def _sender(u, m):
    m = m or {}
    s = str(m.get("slug") or "").lower()
    if not s and str(u).startswith("dvr-"):
        s = re.sub(r"-\d+$", "", str(u)[4:]).lower()
    if not s:
        k = m.get("cohort")
        if isinstance(k, (list, tuple)) and k:
            s = str(k[-1]).lower()
    return s


def _ist_kind(u, m):
    s = _sender(u, m)
    return any(k in s for k in KINDER)


def gewichte(ktx):
    rec = np.asarray(ktx["rec"])
    meta = ktx.get("meta") or {}
    kind = np.array([_ist_kind(u, meta.get(u)) for u in ktx["uuids"]], bool)
    w = np.ones(len(rec), np.float32)
    if len(kind):
        w[kind[rec]] = FAKTOR
    return w
