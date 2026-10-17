"""Daten: Trainingsaufnahmen, in denen das Logo Werbung und Sendung nicht trennt, dreifach gewichten.

Fehlerposten: NN erfindet (K erfindet etwa doppelt so viel wie die Produktion)
und die Verwechslungszonen. Out-of-fold auf dem Trainingsteil tragen die 36
von 527 Aufnahmen, deren Logo-Spalte innerhalb der Aufnahme kaum trennt
(AUC Logo-Abwesenheit gegen Label < 0.75; Call Me Kat, Lenssen hilft, Beet-
Brueder, Micky Maus, SpongeBob, Filme), 39 % der Fehlerzeilen bei 7.5 % der
Zeilen. Dort ordnet das NN die Sekunden meist richtig (AUC 0.96), liegt aber
als Ganzes auf der Werbeseite: ohne Logo liest der Kopf Sendung als Werbung.
Mit 7.5 % der Zeilen lernt er das kaum. Gewicht 3 auf diese Aufnahmen soll ihm
beibringen, dass fehlendes Logo bei Sendungsbild keine Werbung ist.

Die Auswahl nutzt nur Trainingslabels (ktx["y"]) und die Logo-Spalte der
Trainingsaufnahmen; Kalibrierung und Eval bleiben unberuehrt. Kein Zufall.
"""
import numpy as np

BESCHREIBUNG = ("Daten: Trainingsaufnahmen mit nicht trennender Logo-Spur "
                "(Logo-AUC < 0.75 je Aufnahme) dreifach gewichten")

SCHWELLE = 0.75
FAKTOR = 3.0

_stand = {}


def _auc(score, y):
    """AUC von score fuer y==1 (Rang-Formel, Bindungen gemittelt)."""
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spalten(teil, X, ktx):
    if teil == "training":
        j = ktx["spalten_namen"].index("logo")
        rec = ktx["rec"]
        y = np.asarray(ktx["y"])
        schwach = np.zeros(len(ktx["uuids"]), bool)
        for k in range(len(ktx["uuids"])):
            m = rec == k
            if not m.any():
                continue
            # hohe Logo-Spalte = Logo da = Sendung; daher Abwesenheit als Werbe-Score
            a = _auc(-np.asarray(X[m, j], np.float64), y[m])
            if a is not None and a < SCHWELLE:
                schwach[k] = True
        _stand["schwach"] = schwach
    return X


def gewichte(ktx):
    schwach = _stand.get("schwach")
    if schwach is None or len(schwach) != len(ktx["uuids"]):
        raise RuntimeError("spalten(training) muss vor gewichte laufen")
    return np.where(schwach[ktx["rec"]], FAKTOR, 1.0).astype(np.float32)
