"""Training: Logo-Abwesenheit als Augmentation -- Sendung ohne Logo kennenlernen.

Fehlerposten: NN erfindet Werbung in Sendung ohne sichtbares Logo
(Spielfilme, Kinder-Sender toggo/nick, ausgewaschene oder versteckte Logos;
K erfindet doppelt so viel wie die Produktion). Im Trainingsteil liegt das
Logo bei Sendung fast immer an (q10 0.31, Median 0.91), bei Werbung fast nie
(Median 0.06). Der Kopf lernt daraus "kein Logo = Werbung" als Abkuerzung
und muss Sendung nie am Bild erkennen, wenn das Logo fehlt. 016 hat die 36
logo-blinden Trainingsaufnahmen hochgewichtet (+0.0029, knapp verfehlt) --
das wirkt nur ueber die wenigen Aufnahmen, die zufaellig logo-blind sind.

Mechanismus: im Training wird bei zufaelligen 15 % der Zeilen die Logo-Spalte
auf "kein Logo" (Rohwert 0) gedrueckt (nur nach unten: min(x, 0)). Werbezeilen
aendern sich dabei kaum (Logo dort ohnehin ~0), Sendungszeilen werden zu
"Sendung ohne Logo": der Kopf muss sie am Bild (Backbone, SigLIP, OCR, Ton)
erkennen. Logo-Anwesenheit bleibt starke Sendungs-Evidenz, Abwesenheit wird
schwaecher. Kalibrierung und Eval sehen die echten Werte (eval-Modus).

Zufall: torch.rand im forward; trainiere() setzt torch.manual_seed aus dem
Wiederholungs-Seed (seed*100+j), die Streuung misst also den ganzen Zufall.
Kein Label, kein Meta, nur Spaltennamen und Trainingsstatistik der Logo-Spalte.
"""
import numpy as np

BESCHREIBUNG = ("Training: bei 15 % der Trainingszeilen Logo-Spalte auf 'kein Logo' "
                "druecken (Augmentation gegen 'kein Logo = Werbung')")

ANTEIL = 0.15

_stand = {}


def spalten(teil, X, ktx):
    if teil == "training":
        i = ktx["spalten_namen"].index("logo")
        sp = np.asarray(X[:, i], np.float64)
        mu, sd = sp.mean(), sp.std()
        if sd < 1e-6:
            sd = 1.0
        _stand["i"] = i
        # opt_lauf standardisiert danach mit Mittel/Streuung des Trainingsteils:
        # Rohwert 0 entspricht dann (0 - mu) / sd.
        _stand["null"] = float((0.0 - mu) / sd)
    return X


def modell(n_in, hyper):
    import torch
    import torch.nn as nn

    i, null, anteil = _stand["i"], _stand["null"], ANTEIL

    class LogoAbwesenheit(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(n_in, hyper["hidden"]), nn.ReLU(),
                                   nn.Linear(hyper["hidden"], 2))

        def forward(self, x):
            if self.training and anteil > 0:
                m = torch.rand(x.shape[0], device=x.device) < anteil
                spalte = x[:, i]
                neu = torch.where(m, torch.clamp(spalte, max=null), spalte)
                x = torch.cat([x[:, :i], neu[:, None], x[:, i + 1:]], 1)
            return self.f(x)

    return LogoAbwesenheit()
