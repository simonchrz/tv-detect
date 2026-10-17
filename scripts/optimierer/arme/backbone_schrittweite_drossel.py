"""Training: Backbone-Eingang um Faktor 0.35 vorkonditionieren -- der Kopf
lernt den 1280er-Backbone langsamer und die kleinen Spalten (Logo, Audio,
OCR, SigLIP) relativ schneller.

Fehlerposten: der Kopf stellt sich bei sichtbarem Logo gegen das Logo
(Lehre 006-008: Logo >= 0.8 und p >= 0.8 nur 53 % Werbung) und uebertraegt
schlecht auf Sender, die im Trainingsteil kaum vorkommen: nick, comedy-
central, sixx, disney, super-rtl, toggo stellen 29 % der Bewertungs-
aufnahmen, aber nur ~4 % der Trainingszeilen (vox allein 28 %). In-sample
erklaert der Backbone fast alles, er lernt Sender-Stil mit; ausserhalb
dieses Stils fehlt dem Kopf die Stimme der kleinen Spalten.

Mechanismus: im forward (Training UND Erkennung gleich) werden die
standardisierten Backbone-Spalten mit SKALA multipliziert. Fuer die
Funktionsklasse ist das bedeutungslos (erste Schicht linear), es aendert
nur die Lern-Dynamik: Adam macht je Gewicht Schritte ~lr, ein Backbone-
Gewicht braucht also 1/SKALA-mal so lange fuer dieselbe Wirkung, und die
Startwerte der Backbone-Wirkung sind 0.35-mal so gross. Effektiv eine
Schrittweiten-Drossel nur fuer die Backbone-Gruppe, keine Aenderung an
Merkmalen, Gewichten, Zielen, Kapazitaet oder Epochen.

Abgrenzung: 003 (Backbone-Dropout) rauscht die Eingabe, 012 (Zwei-Turm)
schaltet den Backbone-Turm ab, 024 (Logo-Steigungs-Gatter) baut einen
eigenen Logo-Pfad. Hier bleibt das K-MLP unveraendert, nur die relative
Lerngeschwindigkeit der Gruppen verschiebt sich.

Kein Zufall ausser torch (Startwerte/Reihenfolge aus dem Wiederholungs-Seed
via trainiere()). Kein Label, kein Meta; nur Spaltennamen.
"""
import numpy as np

BESCHREIBUNG = ("Training: Backbone-Spalten im Kopf mit 0.35 vorkonditioniert "
                "(Schrittweiten-Drossel nur fuer den Backbone, K-MLP sonst gleich)")

SKALA = 0.35
_stand = {}


def spalten(teil, X, ktx):
    namen = ktx["spalten_namen"]
    idx = np.array([i for i, n in enumerate(namen) if n.startswith("backbone")], np.int64)
    if teil == "training":
        _stand["idx"] = idx
        _stand["n"] = len(namen)
    else:
        # Eval und Kalibrierung muessen dieselbe Spaltenlage haben.
        assert _stand.get("n", len(namen)) == len(namen)
    return X


def modell(n_in, hyper):
    import torch
    import torch.nn as nn

    idx = _stand.get("idx")
    maske = np.ones(n_in, np.float32)
    if idx is not None and len(idx):
        maske[idx] = SKALA
    else:
        maske[:1280] = SKALA

    class BackboneDrossel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("maske", torch.from_numpy(maske))
            self.f = nn.Sequential(nn.Linear(n_in, hyper["hidden"]), nn.ReLU(),
                                   nn.Linear(hyper["hidden"], 2))

        def forward(self, x):
            return self.f(x * self.maske)

    return BackboneDrossel()
