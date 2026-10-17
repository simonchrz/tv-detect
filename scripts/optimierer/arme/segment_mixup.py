"""Training: Segment-Mixup -- Trainingszeilen werden mit einer zeitlichen
Nachbarzeile GLEICHEN Labels derselben Aufnahme gemischt (Eingaberaum).

Fehlerposten: Verwechslungszonen (13 % der Sekunden, deren Backbone-Nachbarn
das Gegenteil-Label tragen, tragen 45 % der Fehler; NN erfindet/verpasst).
K lernt jede Sekunde als isolierten Punkt. Eine schwer unterscheidbare
Sekunde (Trailer-Bild in Sendung, Sendungsbild im Spot) liegt im
Merkmalsraum nahe am Gegenteil, ihre Nachbarn 4-32 s davor/danach aber
meist eindeutig. Mischt man sie mit einem solchen Nachbarn
(x' = lam*x_t + (1-lam)*x_{t+o}, lam in [LMIN, 1], Label bleibt, weil beide
dasselbe tragen), lernt der Kopf, dass die ganze Strecke zwischen
zusammengehoerigen Sekunden auf einer Seite der Grenze liegt. Das zielt auf
das, was der HSMM liest: die Evidenz ueber eine Strecke, nicht jede
Einzelsekunde.

Abgrenzung: kein Fenster-/Kontextmerkmal (O16, 015/018/021/022) -- die
Auswertung sieht nur die Einzelsekunde, unveraendert wie K. Kein Gewicht,
kein Ziel, keine Kapazitaet; das Modell ist das K-MLP. Getestet und
verworfen im Proxy: Fenster-Verlust auf dem Logit-Mittel (Einzel-AUC -0.01),
Focal- und GCE-Verlust, weiche Kanten-Ziele, mehr Optimiererschritte,
Hygiene wie die Produktion (siehe Befund).

Nachbarn: +-4/8/16/32 s (ktx["schritt"] beruecksichtigt), selbe Aufnahme,
selbes Trainingslabel. Je Zeile mit Wahrscheinlichkeit PMIX ein zufaelliger
gueltiger Nachbar. Zufall: torch-Generator, den der Laeufer je Kopf mit
seed*100+j setzt -- die Streuung ueber Wiederholungen misst ihn mit.

Technik: spalten() haengt im Trainingsteil den Zeilenindex als zwei Spalten
(hi/lo) an (sonst Nullen); das Modell entfernt sie und liest die
Nachbarzeilen aus einer standardisierten float16-Kopie des Trainingsteils
(gleiche Standardisierung wie der Laeufer).
"""
import numpy as np

BESCHREIBUNG = ("Training: Segment-Mixup, 50 % der Zeilen mit einem Nachbarn "
                "+-4..32 s gleichen Labels gemischt (lam 0.5..1)")

PMIX = 0.5
LMIN = 0.5
OFFSETS_S = (4, 8, 16, 32)
_BASIS = 1000

_stand = {}


def spalten(teil, X, ktx):
    n = len(X)
    zus = np.zeros((n, 2), np.float32)
    if teil == "training":
        idx = np.arange(n)
        zus[:, 0] = idx // _BASIS
        zus[:, 1] = idx % _BASIS
        rec = np.asarray(ktx["rec"])
        y = np.asarray(ktx["y"])
        schritt = int(ktx["schritt"])
        offs = sorted({max(1, int(round(s / schritt))) for s in OFFSETS_S})
        offs = [-o for o in offs[::-1]] + offs
        N = np.full((n, len(offs)), -1, np.int64)
        for k, o in enumerate(offs):
            j = idx + o
            ok = (j >= 0) & (j < n)
            jj = np.clip(j, 0, n - 1)
            ok &= (rec[jj] == rec) & (y[jj] == y)
            N[ok, k] = jj[ok]
        X2 = np.concatenate([np.asarray(X, np.float32), zus], 1)
        _stand.clear()
        _stand["X"] = X2
        _stand["N"] = N
        return X2
    return np.concatenate([np.asarray(X, np.float32), zus], 1)


def _vorbereiten(dev):
    import torch
    if "Z" in _stand:
        if _stand["Z"].device != dev:
            _stand["Z"] = _stand["Z"].to(dev)
            _stand["Nt"] = _stand["Nt"].to(dev)
        return
    X2 = _stand["X"]
    mu = X2.mean(0, keepdims=True)
    sd = X2.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    _stand["mu_idx"] = mu[0, -2:].astype(np.float64)
    _stand["sd_idx"] = sd[0, -2:].astype(np.float64)
    Z = np.empty((X2.shape[0], X2.shape[1] - 2), np.float16)
    for s in range(0, len(X2), 65536):
        Z[s:s + 65536] = ((X2[s:s + 65536, :-2] - mu[:, :-2]) / sd[:, :-2]).astype(np.float16)
    _stand["Z"] = torch.from_numpy(Z).to(dev)
    _stand["Nt"] = torch.from_numpy(_stand["N"]).to(dev)
    _stand.pop("X", None)


def modell(n_in, hyper):
    import torch
    import torch.nn as nn

    class SegmentMixup(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(n_in - 2, hyper["hidden"]), nn.ReLU(),
                                   nn.Linear(hyper["hidden"], 2))

        def forward(self, x):
            z = x[:, :-2]
            if not self.training or "N" not in _stand:
                return self.f(z)
            _vorbereiten(x.device)
            mu, sd = _stand["mu_idx"], _stand["sd_idx"]
            hi = torch.round(x[:, -2].float() * float(sd[0]) + float(mu[0]))
            lo = torch.round(x[:, -1].float() * float(sd[1]) + float(mu[1]))
            idx = (hi * _BASIS + lo).long().clamp(0, _stand["Nt"].shape[0] - 1)
            nb = _stand["Nt"][idx]
            b, k = nb.shape
            wahl = torch.randint(0, k, (b,), device=x.device)
            j = nb.gather(1, wahl[:, None])[:, 0]
            ok = (j >= 0) & (torch.rand(b, device=x.device) < PMIX)
            lam = LMIN + (1 - LMIN) * torch.rand(b, 1, device=x.device)
            zn = _stand["Z"][j.clamp(min=0)].float()
            z = torch.where(ok[:, None], lam * z + (1 - lam) * zn, z)
            return self.f(z)

    return SegmentMixup()
