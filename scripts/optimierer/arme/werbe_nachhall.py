"""Werbe-Nachhall: nach einem sicheren Werbeblock klingt die Werbe-Evidenz
einige Sekunden nach, statt sofort auf Sendung zu fallen.

Befund (Trainingsteil, 134 Aufnahmen mit Signal-Dump, K-Abbild out-of-fold
je Sekunde, Produktions-HSMM wie im Laeufer): der Block-IoU-Verlust
sitzt an den Blockenden. Wo ein Block zu frueh endet (> 10 s, 57 Bloecke,
Median 34 s), steht das Logo im verpassten Schwanz schon wieder (Median
0.75) und der Kopf gibt dort p 0.25 (q25/q75 0.16/0.40): mittlere, nicht
sendungssichere Evidenz direkt hinter starker Werbung (Trailer, Countdown,
Sponsor am Blockende). Der Dekoder schneidet am letzten starken Werbebild.

Regel je Aufnahme auf den Kopf-Logits z (vor der Kalibrierung, gleich in
Kalibrierung und Eval): ein Werbeanker ist eine Sekunde, deren 5-s-Mittel
z > STARK ist. In den W Sekunden nach dem letzten Anker bekommt jede
Sekunde, die nicht sendungssicher ist (z > UNTEN), einen linear
abklingenden Zuschlag C * (1 - dt/W). Sendungssichere Sekunden bleiben
unberuehrt, ebenso alles vor einem Block (Blockanfaenge sind nicht schief:
Startversatz Median 0 s).

Die Ausgabe wird affin vorkonditioniert (A*z' + B, A/B = Platt-Loesung von K
auf dem Trainingsteil). Grund: der ungedaempfte Newton in opt_lauf.platt
startet bei (1, 0) und divergiert bei schon leicht umgeformten Logits
(a ~ 1e11, also harte 0/1-Evidenz mit verschobener Schwelle) — dann misst
der Lauf ein Artefakt statt der Regel. Mit A/B startet Platt nahe am
Optimum und konvergiert; die Information aendert die affine Abbildung
nicht, Platt passt sie ohnehin neu an. Kein Zufall, keine Labels.
"""
import numpy as np

BESCHREIBUNG = ("Werbe-Nachhall: nach sicherem Werbeblock (5-s-Mittel z > 4) "
                "bekommen nicht sendungssichere Sekunden (z > -4) 45 s lang "
                "einen abklingenden Werbezuschlag (3 * (1 - dt/45)) vor der Kalibrierung")

STARK = 4.0      # Kopf-Logit, ab dem ein 5-s-Mittel als sichere Werbung gilt
UNTEN = -4.0     # darunter gilt eine Sekunde als sendungssicher, kein Zuschlag
W = 45.0         # Nachhall in Sekunden
C = 3.0          # Zuschlag direkt hinter dem Anker (Logit)
A, B = 0.63, -1.10   # Vorkonditionierung fuer Platt (K-Loesung, Trainingsteil)
GRENZE = 13.8


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _mittel(z, k):
    pad = np.r_[np.full(k, z[0]), z, np.full(k, z[-1])]
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")


def _nachhall(z, schritt):
    n = len(z)
    if n == 0:
        return z
    k = max(1, int(round(2 / schritt)))          # +-2 s
    anker = _mittel(z, k) > STARK
    idx = np.where(anker, np.arange(n), -1)
    letzter = np.maximum.accumulate(idx)          # letzter Anker <= t
    dt = (np.arange(n) - letzter) * float(schritt)
    zusch = np.where((letzter >= 0) & ~anker & (dt <= W) & (z > UNTEN),
                     C * (1.0 - dt / W), 0.0)
    return z + zusch


def nachbearbeiten(p, rec, ktx):
    z = np.clip(_logit(p), -GRENZE, GRENZE)
    schritt = int(ktx.get("schritt") or 1)
    aus = z.copy()
    for i in np.unique(rec):
        m = np.flatnonzero(rec == i)
        aus[m] = _nachhall(z[m], schritt)
    aus = np.clip(A * np.clip(aus, -GRENZE, GRENZE) + B, -GRENZE, GRENZE)
    return 1.0 / (1.0 + np.exp(-aus))
