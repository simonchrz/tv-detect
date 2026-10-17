"""Daten: wiederkehrende Werbezeilen abwerten (Gegenrichtung zu 019).

Fehlerposten: NN verpasst Werbung und Kanten. Im Trainingsteil sind 52 % der
Werbezeilen Nahezu-Duplikate (Backbone-Kosinus > 0.9, zentriert) von Zeilen
in ANDEREN Aufnahmen, 5 % haben mehr als 20 solcher Treffer -- dieselben
Spots stehen dutzend- bis hundertfach im Gradienten. Out-of-fold verpasst ein
K-aehnlicher Kopf einmalige Werbezeilen zu 14-15 %, haeufig wiederholte zu
0.5 %: der Kopf lernt Spots auswendig, die seltene Werbeform (Trailer, neue
Spots, Countdown-Spots) bleibt unterbelichtet.

019 hat die Gegenrichtung im Messgeraet gemessen: Duplikate aufwerten
((1+n)^+0.5) kostete -0.0039 bei 5/5 negativen Wiederholungen, vor allem
ueber mehr verpasste Werbung und Kanten (nn_verpasst +32, kante +44). Der
Proxy hatte das Gegenteil vorhergesagt -- er taugt fuer diese Achse nicht,
das Messgeraet zeigte eine klare Steigung. Dieser Arm geht die andere
Richtung: Gewicht je Werbezeile (1 + n_dup)^-0.5, so skaliert, dass die
Summe der Werbegewichte gleich bleibt (Klassenverhaeltnis wie K).

Zaehlung nur auf Backbone-Merkmalen und Trainingslabels des Trainingsteils;
Kalibrierung und Eval unberuehrt. Kein Zufall.
"""
import numpy as np

BESCHREIBUNG = ("Daten: Werbezeilen mit Nahezu-Duplikaten in anderen "
                "Trainingsaufnahmen abwerten ((1+n)^-0.5, Klassensumme gleich)")

ALPHA = -0.5
SCHWELLE = 0.9

_stand = {}


def _zaehle(E, rec, frage):
    """Je Zeile aus `frage`: Anzahl Zeilen anderer Aufnahmen mit Kosinus > SCHWELLE."""
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    E = E - E.mean(0, keepdims=True)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
    T = torch.tensor(E, dtype=torch.float16, device=dev)
    R = torch.tensor(rec.astype(np.int64), device=dev)
    out = np.zeros(len(frage), np.int64)
    for i in range(0, len(frage), 4096):
        idx = torch.tensor(frage[i:i + 4096], device=dev)
        S = T[idx] @ T.T
        S[R[idx][:, None] == R[None, :]] = -1
        out[i:i + 4096] = (S > SCHWELLE).sum(1).cpu().numpy()
    return out


def spalten(teil, X, ktx):
    if teil == "training":
        n = ktx["spalten_namen"].index("backbone1279") + 1
        y = np.asarray(ktx["y"])
        frage = np.where(y == 1)[0]
        _stand["frage"] = frage
        _stand["n_dup"] = _zaehle(np.asarray(X[:, :n], np.float32),
                                  np.asarray(ktx["rec"]), frage)
    return X


def gewichte(ktx):
    y = np.asarray(ktx["y"])
    w = np.ones(len(y), np.float64)
    frage = _stand["frage"]
    if len(frage):
        d = (1.0 + _stand["n_dup"].astype(np.float64)) ** ALPHA
        d *= len(d) / d.sum()
        w[frage] = d
    return w.astype(np.float32)
