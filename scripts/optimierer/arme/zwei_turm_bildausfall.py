"""Zwei-Turm-Kopf mit Bild-Ausfall: Logo/OCR/Audio bekommen eine eigene,
additive Stimme, die auch ohne Bildmeinung tragen muss.

Befund (Trainingsteil, haelftig nach Aufnahme, jede 8. s, Platt 2-fach auf
Haelfte B): bei K steht der Kopf bei Logo >= 0.8 oft gegen das Logo auf
Werbung (751-826 Zeilen mit p >= 0.8, davon 20-26 % Sendung). Der Grund:
in-sample erklaert der 1280er-Backbone fast alles, der Gradient zu den
kleinen Spalten versiegt, ihr Gewicht bleibt klein. Hier:

  logit = f(alle Spalten) + g(logo, audio, ocr_hinweis, ocr_werbung, ocr_da, siglip_da)

f = K-MLP (hidden 96), g = kleines MLP (hidden 16). Im Training wird f je
ZEILE mit Wahrscheinlichkeit P_AUS ganz ausgeschaltet (Logit 0), dann muss
g allein entscheiden und lernt die volle Staerke der kleinen Signale.
Bei der Vorhersage sind beide Tuerme an (keine Skalierung: g ist auf
'f fehlt' wie auf 'f da' trainiert, Platt gleicht den Massstab aus).
Zufall nur ueber torch, das der Laeufer je Kopf mit dem Wiederholungs-Seed
setzt. Keine Merkmals-, Gewichts- oder Zielaenderung.
"""
import torch
import torch.nn as nn

BESCHREIBUNG = ("Zwei-Turm: logit = K-MLP(alle) + MLP16(logo/audio/ocr/Indikator), "
                "K-Turm je Zeile mit p=0.5 ausgeschaltet im Training")

P_AUS = 0.5
KLEIN = (1280, 1281, 1282, 1283, 1284, 1349)   # logo, audio, ocr_hinweis, ocr_werbung, ocr_da, siglip_da
N_K = 1350


class _ZweiTurm(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        assert n_in == N_K, f"Spaltenzahl {n_in} != {N_K}"
        self.register_buffer("ki", torch.tensor(KLEIN, dtype=torch.long))
        self.f = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.g = nn.Sequential(nn.Linear(len(KLEIN), 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        a = self.f(x)
        b = self.g(x.index_select(1, self.ki))
        if self.training and P_AUS > 0:
            an = (torch.rand(x.shape[0], 1, device=x.device) >= P_AUS).to(x.dtype)
            a = a * an
        return a + b


def modell(n_in, hyper):
    return _ZweiTurm(n_in, hyper["hidden"])
