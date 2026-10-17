"""Sendeplan-Prior je Serie: Werbebloecke liegen bei derselben Sendung
(Titel + Sender) Folge fuer Folge zur selben Uhrzeit.

Neue Informationsquelle: die Zeitstruktur des Programms. Der Kopf sieht nur
das Bild der einzelnen Sekunde; dass bei "Galileo" auf ProSieben um 19:23
fast immer Werbung laeuft, weiss er nicht. Der alte Minute-Prior (je Sender,
Minute der Stunde) war inert, weil Werbung an der Sendung haengt und nicht an
der Uhr -- dieser Prior haengt an der Sendung UND der Uhr.

Ablauf (keine Labels ausserhalb des Trainingsteils, kein Zufall):
* spalten("training"): je (Titel, Sender) eine Tagestabelle in 10-s-Faechern
  (Ortszeit) aus den Trainingslabels: Werbe-Sekunden und Sekunden gesamt,
  umgerechnet in "Folgen". X bleibt unveraendert.
* nachbearbeiten (Kalibrierung und Eval gleich): je Aufnahme Prior =
  (Werbung + a*Grundrate) / (Folgen + a), +-60 s geglaettet, a = 2; dann
  label-frei verschoben (Versatz +-600 s, der die Korrelation mit dem
  Kopf-p maximiert: Sendung beginnt verspaetet / frueher); Fusion
  logit' = logit(p) + 0.5 * (logit(prior) - logit(Grundrate)).
  Ohne Folgen im Trainingsteil ist der Prior die Grundrate -> keine Wirkung.
"""
import time
import collections
import numpy as np

BESCHREIBUNG = ("Sendeplan-Prior je (Titel, Sender) und Uhrzeit aus den Trainingsfolgen, "
                "per Versatz an das Kopf-p angeglichen, als Logit-Zusatz 0.5 vor Platt")

FACH = 10                 # Sekunden je Fach
N_FACH = 86400 // FACH
GLATT = 6                 # +-6 Faecher = +-60 s
ALPHA = 2.0               # Schrumpfung in Folgen
BETA = 0.5                # Gewicht des Priors im Logit
VERSATZ = 600             # Sekunden, Suche +-
VERSATZ_SCHRITT = 10

_z = {"tab": None, "basis": None}


def _schluessel(m):
    return ((m.get("title") or "").strip().lower(), m.get("slug") or "")


def _faecher(start_ts, sek):
    off = time.localtime(int(start_ts)).tm_gmtoff
    return (((int(start_ts) + off + np.asarray(sek, np.int64)) % 86400) // FACH).astype(np.int64)


def _glatt(v):
    k = np.ones(2 * GLATT + 1)
    return np.convolve(np.concatenate([v[-GLATT:], v, v[:GLATT]]), k, "valid") / (2 * GLATT + 1)


def _lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def spalten(teil, X, ktx):
    if teil == "training":
        rec, y, sek, schritt = ktx["rec"], np.asarray(ktx["y"]), ktx["sek"], ktx["schritt"]
        roh = collections.defaultdict(lambda: [np.zeros(N_FACH), np.zeros(N_FACH)])
        for i, u in enumerate(ktx["uuids"]):
            m = ktx["meta"].get(u) or {}
            if not m.get("start_ts"):
                continue
            mk = rec == i
            b = _faecher(m["start_ts"], sek[mk])
            g = roh[_schluessel(m)]
            f = schritt / FACH          # eine Zeile steht fuer `schritt` Sekunden
            np.add.at(g[0], b, y[mk] * f)
            np.add.at(g[1], b, f)
        _z["tab"] = {k: (_glatt(a), _glatt(t)) for k, (a, t) in roh.items()}
        _z["basis"] = float(np.clip(y.mean(), 0.01, 0.99))
    return X


def _verschiebe(v, L):
    if L == 0:
        return v
    q = np.roll(v, L)
    if L > 0:
        q[:L] = v[0]
    else:
        q[L:] = v[-1]
    return q


def nachbearbeiten(p, rec, ktx):
    p = np.asarray(p, np.float64)
    if not _z["tab"]:
        return p
    basis = _z["basis"]
    z = _lg(p)
    aus = z.copy()
    schritt = int(ktx.get("schritt") or 1)
    for i, u in enumerate(ktx["uuids"]):
        m = ktx["meta"].get(u) or {}
        g = _z["tab"].get(_schluessel(m))
        if g is None or not m.get("start_ts"):
            continue
        mk = np.flatnonzero(rec == i)
        if len(mk) < 10:
            continue
        b = _faecher(m["start_ts"], ktx["sek"][mk])
        a, t = g
        pr = (a[b] + ALPHA * basis) / (t[b] + ALPHA)
        if pr.std() < 1e-6:
            continue
        pk = p[mk]
        best_c, best_L = -2.0, 0
        weit = VERSATZ // schritt
        for L in range(-weit, weit + 1, max(1, VERSATZ_SCHRITT // schritt)):
            ps = _verschiebe(pr, L)
            if ps.std() < 1e-9:
                continue
            c = float(np.corrcoef(ps, pk)[0, 1])
            if c > best_c + 1e-9:
                best_c, best_L = c, L
        pr = _verschiebe(pr, best_L)
        aus[mk] = z[mk] + BETA * (_lg(pr) - _lg(basis))
    return 1.0 / (1.0 + np.exp(-aus))
