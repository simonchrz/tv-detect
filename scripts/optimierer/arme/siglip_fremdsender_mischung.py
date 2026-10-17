"""SigLIP nur halb trauen, wo der Kopf den Sender nie mit SigLIP gesehen hat.

Fehlerposten: NN erfindet auf Kinder-/Nischensendern (Abbild-Luecke toggo,
nick; super-rtl-1781112000 im Schiedsrichter-Satz: 15-min-Phantomblock).
Die Bildmerkmale dort sind NICHT gestoert (Backbone-Flackern, Norm, SigLIP-
Nachbarkosinus wie im Training); was abweicht, ist die Abdeckung: zwei
Drittel des Trainingsteils haben keine SigLIP-Spur, und genau die Kinder-
und Nischensender fast nur ohne (nick 3, comedy-central 3, sat-1 3,
sat-1-gold 3, disney 5, sixx 5 Aufnahmen mit Spur; super-rtl, toggo 0).
In Eval und Betrieb haben sie alle eine Spur -> der Kopf extrapoliert die
64 SigLIP-Spalten auf Bilder, die er so nie gesehen hat.

Regel: Spalte `spur_fremd` = 1 fuer jede Zeile einer Aufnahme, deren Sender
im Trainingsteil weniger als MIN_SPUR Aufnahmen mit SigLIP-Spur hat (nur
Kalibrierung und Eval; im Training immer 0, Sender aus meta/uuid, kein
Label). Das Modell ist der K-MLP auf allen Spalten ausser `spur_fremd`;
bei spur_fremd = 1 ist der Ausgang das Mittel aus K(x) und K(x ohne SigLIP)
(SigLIP-Spalten und siglip_da auf den standardisierten Wert von 0, also den
spurlosen Modus, den K aus dem Training kennt). Training unveraendert
(keine markierte Zeile), kein Zufall, keine Labels.
"""
import re

import numpy as np
import torch
import torch.nn as nn

BESCHREIBUNG = ("SigLIP-Fremdsender: Logit-Mittel aus K(x) und K(x ohne SigLIP) fuer "
                "Sender mit < 6 SigLIP-Aufnahmen im Training (Spalte spur_fremd)")

MIN_SPUR = 6
_st = {}


def _slug(u, meta):
    s = (meta.get(u) or {}).get("slug") or ""
    if not s and u.startswith("dvr-"):
        s = re.sub(r"^dvr-|-\d+$", "", u)
    return s


def spalten(teil, X, ktx):
    X = np.asarray(X, np.float32)
    namen = ktx["spalten_namen"]
    sg = [namen.index(f"siglip{i}") for i in range(64)] + [namen.index("siglip_da")]
    da = namen.index("siglip_da")
    rec = np.asarray(ktx["rec"])
    uu = ktx["uuids"]
    meta = ktx.get("meta") or {}
    flag = np.zeros(len(X), np.float32)
    if teil == "training":
        zahl = {}
        for r, u in enumerate(uu):
            m = rec == r
            if m.any() and float(X[m, da].mean()) > 0.5:
                s = _slug(u, meta)
                zahl[s] = zahl.get(s, 0) + 1
        _st["zahl"] = zahl
        # Standardisierter Wert von 'keine Spur' (roh 0), wie opt_lauf ihn nach
        # spalten() berechnet: Mittel/Std des Trainingsteils, sd < 1e-6 -> 1.
        mu = X[:, sg].mean(0)
        sd = X[:, sg].std(0)
        sd[sd < 1e-6] = 1.0
        _st["sg"] = sg
        _st["null"] = ((0.0 - mu) / sd).astype(np.float32)
    else:
        zahl = _st.get("zahl", {})
        for r, u in enumerate(uu):
            s = _slug(u, meta)
            if s and zahl.get(s, 0) < MIN_SPUR:
                flag[rec == r] = 1.0
    return np.concatenate([X, flag[:, None]], 1)


class _Misch(nn.Module):
    def __init__(self, n_in, hidden, sg, null):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in - 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.register_buffer("sg", torch.tensor(sg, dtype=torch.long))
        self.register_buffer("null", torch.tensor(null, dtype=torch.float32))

    def forward(self, x):
        xin, fl = x[:, :-1], x[:, -1]
        a = self.f(xin)
        # Flag-Spalte ist im Training konstant 0 -> standardisiert 0; in
        # Kalibrierung/Eval 1 fuer markierte Zeilen.
        m = fl > 0.5
        if bool(m.any()):
            x0 = xin[m].clone()
            x0[:, self.sg] = self.null
            b = 0.5 * (a[m] + self.f(x0))
            a = a.clone()
            a[m] = b
        return a


def modell(n_in, hyper):
    sg, null = _st.get("sg"), _st.get("null")
    assert sg is not None, "spalten('training') muss vor modell() laufen"
    return _Misch(n_in, hyper["hidden"], sg, null)
