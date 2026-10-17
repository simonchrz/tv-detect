"""Gewichtsmittel (EMA) je Kopf: vorhergesagt wird mit dem gleitenden Mittel
der Parameter ueber die letzten ~2-3 Epochen, nicht mit dem Zustand nach dem
letzten Adam-Schritt.

Schwaeche: K ist zwischen Seeds instabil und erfindet dort, wo die Koepfe
uneinig sind, lange Phantombloecke (Seed-Uneinigkeit in Phantomlaeufen 3.7x
hoeher als in wahren Bloecken, Befund leer_aufnahmen_ersatz; K-Seeds mit
1.3-1.8 h mehr Werbung im Schiedsrichter-Durchgang). Ein Teil dieser Streuung
ist Rauschen des letzten Optimierer-Schritts (konstante lr 1e-3, 12 Epochen
ohne Abkuehlen). Das Parameter-Mittel glaettet genau diesen Anteil, ohne
Merkmale, Gewichte, Ziel, Kapazitaet oder Ensemble-Groesse zu aendern.

Architektur und Training identisch zu K (Linear 96 - ReLU - Linear 2, gleiche
Initialisierung, gleiche Schritte). Einziger Unterschied: im Trainingsmodus
fuehrt jeder forward()-Aufruf vor dem Rechnen ein EMA der aktuellen Parameter
nach (Schritt k sieht die Parameter nach Schritt k-1), im eval-Modus rechnet
der Kopf mit den EMA-Parametern. Abklingen 0.995 (Horizont ~200 Schritte,
~3 Epochen bei 506k Zeilen / 8192), mit Anlaufkorrektur min(d, (1+n)/(10+n)).
Kein eigener Zufall: Initialisierung ueber torch, das der Laeufer je Kopf
mit dem Wiederholungs-Seed setzt.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BESCHREIBUNG = ("K-MLP mit Parameter-EMA (Abklingen 0.995, ~3 Epochen): "
                "Vorhersage mit gemittelten Gewichten statt letztem Schritt")

ABKLINGEN = 0.995


class _EmaMLP(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        # gleiche Reihenfolge der Initialisierung wie K (nn.Sequential)
        self.l1 = nn.Linear(n_in, hidden)
        self.l2 = nn.Linear(hidden, 2)
        for name, p in (("w1", self.l1.weight), ("b1", self.l1.bias),
                        ("w2", self.l2.weight), ("b2", self.l2.bias)):
            self.register_buffer("e_" + name, p.detach().clone())
        self.register_buffer("n", torch.zeros(()))

    def _paare(self):
        return ((self.e_w1, self.l1.weight), (self.e_b1, self.l1.bias),
                (self.e_w2, self.l2.weight), (self.e_b2, self.l2.bias))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.n += 1
                n = float(self.n.item())
                d = min(ABKLINGEN, (1.0 + n) / (10.0 + n))
                for e, p in self._paare():
                    e.mul_(d).add_(p.detach(), alpha=1.0 - d)
            return self.l2(torch.relu(self.l1(x)))
        h = torch.relu(F.linear(x, self.e_w1, self.e_b1))
        return F.linear(h, self.e_w2, self.e_b2)


def modell(n_in, hyper):
    return _EmaMLP(n_in, hyper["hidden"])
