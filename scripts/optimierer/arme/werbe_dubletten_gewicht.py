"""Daten: Werbezeilen, deren Bild in anderen Trainingsaufnahmen wiederkehrt, aufwerten.

Fehlerposten: NN erfindet Werbung in Sendung ohne Logo (Spielfilme,
Kindersender, logo-blinde Aufnahmen). Befund auf dem Trainingsteil: 63 % der
Werbezeilen haben ein Nahezu-Duplikat (Backbone-Kosinus > 0.9, zentriert) in
einer ANDEREN Aufnahme (Median 1, q90 21, q99 185 Treffer), bei Sendung nur
4.7 %. Die Werbeklasse ist damit zu grossen Teilen ein Satz auswendig
gelernter Spots, die hundertfach im Gradienten stehen. Was der Kopf ueber
"neues, ungesehenes Bild" lernt, kommt fast nur aus der Sendung und aus dem
Drittel einmaliger Werbung -- und genau dort (Film ohne Logo) erfindet er.

Gemessen im Proxy (Trainingsteil, Haelften): die naheliegende Entzerrung
(Duplikate ABwerten) schadet monoton -- die einmaligen Werbezeilen (Trailer,
neue Spots, filmartiges Material) sind es, die dem Kopf "ungesehenes Bild
ohne Logo = Werbung" beibringen. Die Gegenrichtung hilft: wiederkehrende
Spots aufwerten, einmalige relativ abwerten, Werbeklasse in Summe gleich.

Gewicht je Werbezeile: (1 + n_dup)^ALPHA, danach so skaliert, dass die
Summe der Werbegewichte unveraendert bleibt (Klassenverhaeltnis wie K). Die
Zaehlung nutzt nur Backbone-Merkmale der Trainingsaufnahmen und deren
Trainingslabels; Kalibrierung und Eval bleiben unberuehrt. Kein Zufall.
"""
import numpy as np

BESCHREIBUNG = ("Daten: Werbezeilen mit Nahezu-Duplikaten in anderen "
                "Trainingsaufnahmen aufwerten ((1+n)^0.5, Klassensumme gleich)")

ALPHA = 0.5
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
        _stand["n_dup"] = _zaehle(np.asarray(X[:, :n], np.float32), np.asarray(ktx["rec"]), frage)
    return X


def gewichte(ktx):
    y = np.asarray(ktx["y"])
    w = np.ones(len(y), np.float64)
    frage = _stand["frage"]
    d = (1.0 + _stand["n_dup"]) ** ALPHA
    d *= len(d) / d.sum()
    w[frage] = d
    return w.astype(np.float32)
