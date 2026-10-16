"""Backbone-Spalten-Dropout: Regularisierung gegen das Auswendiglernen von
Filmmaterial.

Im Training wird jede der 1280 Backbone-Spalten mit Wahrscheinlichkeit
P_BB unabhaengig auf 0 gesetzt (nach der Standardisierung des Laeufers =
Spaltenmittel) und der Rest mit 1/(1-P_BB) skaliert (inverted dropout).
Die kleinen Zusatzspalten (Logo, Audio, OCR, SigLIP, Indikatoren) bleiben
unberuehrt. Zusaetzlich Dropout P_H auf der versteckten Schicht. Weil das
Signal pro Schritt verrauschter ist, 16 statt 12 Epochen. Bei der
Vorhersage (eval-Modus) ist alles aus: die Architektur ist zur Laufzeit
dieselbe Linear-ReLU-Linear-Kette wie die Kontrolle.

Grund (Ledger 3aa): die Ueberanpassungs-Luecke train->test ist bei
Trailern +0.251, zehnmal so gross wie bei Sendung (+0.023) oder Werbung
(+0.007). Der Kopf merkt sich gesehenes Filmmaterial ueber die
hochdimensionale Backbone-Einbettung statt verallgemeinerbarer Merkmale.
Die OCR-Spalten tragen das verallgemeinerbare Merkmal (Wochentag+Uhrzeit,
10/11 Trailer), werden aber vom 1280er-Block ueberstimmt. Spalten-Dropout
nur auf dem Backbone zwingt den Kopf, sich auf die kleinen Spalten und auf
redundante Backbone-Richtungen zu stuetzen.
"""
import torch
import torch.nn as nn

BESCHREIBUNG = ("Backbone-Spalten-Dropout 0.25 (nur Spalten 0..1279) + "
                "Hidden-Dropout 0.2, 16 Epochen")

N_BB = 1280
P_BB = 0.25
P_H = 0.2

HYPER = {"epochen": 16}


class _BlockDropout(nn.Module):
    def __init__(self, n_block, p):
        super().__init__()
        self.n_block = n_block
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        n = min(self.n_block, x.shape[1])
        keep = (torch.rand(x.shape[0], n, device=x.device) >= self.p).to(x.dtype)
        bb = x[:, :n] * keep / (1.0 - self.p)
        return torch.cat([bb, x[:, n:]], dim=1)


def modell(n_in, hyper):
    return nn.Sequential(
        _BlockDropout(N_BB, P_BB),
        nn.Linear(n_in, hyper["hidden"]),
        nn.ReLU(),
        nn.Dropout(P_H),
        nn.Linear(hyper["hidden"], 2),
    )
