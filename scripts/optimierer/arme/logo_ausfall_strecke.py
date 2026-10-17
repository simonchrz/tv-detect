"""Logo-Ausfall ueber zusammenhaengende Strecken: Laufzeit-Spalte + Strecken-Augmentation.

Fehlerposten: NN erfindet lange Phantombloecke, wenn die Logo-Spur einer
gestoerten Quelle ueber viele Minuten ausfaellt (dvr-super-rtl-1781112000 im
Schiedsrichter-Satz: Logo 0.00 ueber 30 min, K erfindet in jedem Seed einen
15-min-Block ueber einer Toggo-Serie; Befund 039: keine Decode-Spur im
Backbone, sondern Logo-Ausfall). Ein Werbeblock ohne Logo dauert 3-7 min;
eine logo-lose Strecke von >= 10 min ist im Trainingsteil fast immer Sendung
(10 Strecken in 7 von 524 Aufnahmen, Werbeanteil Median 0.16, keine > 0.8).
Diese Strecken sind dort selten (0.5 % der Sekunden), in Entwicklung (1.5 %,
3 Aufnahmen) und Bestaetigung (1.8 %, 1 Aufnahme) haeufiger.

K sieht jede Sekunde einzeln und kann "Logo fehlt seit 2 min" nicht von
"Logo fehlt seit 25 min" unterscheiden. Anders als 025 (Logo zeilenweise
zufaellig auf 0, keine Zeitstruktur), 004 (Logo-Mittel ueber 30 s) und
009/027 (Aufnahme-Konstanten / Sender-Skala):

* Spalte logo_aus_lauf: Logo 31 s median-geglaettet, "aus" = < 0.15; je
  Sekunde die Laenge (Minuten, gekappt bei 20, geteilt durch 20) der
  logo-losen Strecke, in der sie liegt; 0 bei sichtbarem Logo. Fenster und
  Laenge in Sekunden, also mit ktx["schritt"] umgerechnet (Training jede 4. s).
* Augmentation nur im Training: in 25 % der Trainingsaufnahmen wird ueber ein
  zufaelliges zusammenhaengendes Fenster von 10-30 min die Logo-Spalte auf
  0 gesetzt (simulierter Spur-Ausfall), Ziel unveraendert. Die Laufzeit-Spalte
  wird danach aus dem gestoerten Logo berechnet. Der Kopf lernt: lange
  logo-lose Strecke = Logo traegt nicht, am Bild entscheiden.

Zufall: spalten() wird je Lauf einmal aufgerufen; die Augmentation zieht aus
einem festen Generator (Seed 17), alle Wiederholungen sehen dieselben
gestoerten Fenster. Kein Label, kein Meta.
"""
import numpy as np

BESCHREIBUNG = ("Logo-Ausfall-Strecken: Spalte 'Laenge der logo-losen Strecke' + im Training "
                "in 25 % der Aufnahmen 10-30 min Logo auf 0 (Ziel unveraendert)")

AUS = 0.15
GLATT_S = 31
KAPPE_MIN = 20.0
ANTEIL = 0.25
FENSTER_S = (600.0, 1800.0)
SEED = 17


def _median_glatt(x, k):
    if k <= 1 or len(x) < 2:
        return x.copy()
    h = k // 2
    p = np.pad(x, h, mode="edge")
    w = np.lib.stride_tricks.sliding_window_view(p, k)
    return np.median(w, axis=1)[:len(x)]


def _lauf_spalte(lg, schritt):
    k = max(3, int(GLATT_S // schritt) | 1)
    aus = _median_glatt(lg, k) < AUS
    c = np.zeros(len(lg), np.float32)
    if not aus.any():
        return c
    d = np.diff(np.concatenate([[0], aus.astype(np.int8), [0]]))
    anf = np.where(d == 1)[0]
    end = np.where(d == -1)[0]
    for a, b in zip(anf, end):
        c[a:b] = min((b - a) * schritt / 60.0, KAPPE_MIN) / KAPPE_MIN
    return c


def spalten(teil, X, ktx):
    X = np.array(X, np.float32, copy=True)
    i = ktx["spalten_namen"].index("logo")
    rec = np.asarray(ktx["rec"])
    schritt = int(ktx["schritt"])
    if teil == "training":
        rng = np.random.default_rng(SEED)
        for r in range(len(ktx["uuids"])):
            m = np.where(rec == r)[0]
            zieh = rng.random()
            ln_s = rng.uniform(*FENSTER_S)
            pos = rng.random()
            if zieh >= ANTEIL or len(m) * schritt < FENSTER_S[0] + 300:
                continue
            ln = min(int(ln_s / schritt), len(m) - 1)
            a = int(pos * (len(m) - ln + 1))
            X[m[a:a + ln], i] = 0.0
    c = np.zeros(len(X), np.float32)
    for r in range(len(ktx["uuids"])):
        m = np.where(rec == r)[0]
        if len(m):
            c[m] = _lauf_spalte(X[m, i], schritt)
    return np.concatenate([X, c[:, None]], 1)
