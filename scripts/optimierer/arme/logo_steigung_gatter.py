"""Logo als kontextabhaengige Steigung: der Kopf bekommt neben dem K-MLP
einen eigenen, im Logo LINEAREN Pfad, dessen Steigung aus dem Kontext
der Zeile kommt und dessen Vorzeichen fest ist (mehr Logo -> weniger
Werbung):

  d(x)      = softplus(MLP16(x)) * logo_norm(x)       (>= 0 je Einheit Logo)
  logit_ad  = f(x)[1] - d/2,   logit_show = f(x)[0] + d/2

f = K-MLP (hidden 96, alle Spalten wie K). Der Steigungs-Turm sieht alle
Spalten und entscheidet je Zeile, wie viel das Logo hier zaehlt (z. B.
wenig, wo das Bild nach Spielfilm/Kindersender aussieht und das Logo
ohnehin nicht trennt; viel, wo das Bild in einer Verwechslungszone liegt).
Das K-MLP muss das Logo nicht mehr ueber ReLU-Knicke stueckweise
nachbauen; die Wirkung des Logos bleibt auch ausserhalb der
Trainingsverteilung linear und gleichgerichtet.

Keine Merkmals-, Gewichts- oder Zielaenderung, kein eigener Zufall
(torch-Init ueber den Wiederholungs-Seed des Laeufers).
"""
import torch
import torch.nn as nn

BESCHREIBUNG = ("Logo-Steigungs-Gatter: logit = K-MLP(alle) + softplus(MLP16(alle)) * logo, "
                "Vorzeichen fest (mehr Logo = weniger Werbung)")

LOGO = 1280      # Spalte 'logo' in K (backbone0..1279, logo, audio, ...), normiert
N_K = 1350


class _LogoSteigung(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        assert n_in == N_K, f"Spaltenzahl {n_in} != {N_K}"
        self.f = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.s = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.f(x)
        d = -nn.functional.softplus(self.s(x)).squeeze(1) * x[:, LOGO]
        return torch.stack([z[:, 0] - d / 2, z[:, 1] + d / 2], 1)


def modell(n_in, hyper):
    return _LogoSteigung(n_in, hyper["hidden"])
