"""Klammer-Kette: wie weit reicht die Szenen-Kette, zu der diese Sekunde
gehoert?

018 (klammer_wiederkehr) mass die DIREKTE Wiederkehr: groesster Kosinus zu
einer Sekunde 20..240 s davor/danach. Das war der beste bisherige
Informationstraeger (Entwicklung +0.0034, 4/4 erfundene UND verpasste
Sekunden runter), bestaetigte aber nur +0.0013. Ein Maximum ist ein
einzelner Treffer: eine neue Szene ohne direkte Wiederkehr sieht aus wie
Werbung, ein zufaellig aehnlicher Spot wie Sendung.

Neuer Mechanismus (kein Parameter von 018): ein Aehnlichkeits-GRAPH je
Aufnahme. Kante zwischen zwei Sekunden des Vorrats (jede 4. Sekunde), wenn
Kosinus >= TAU und Abstand <= 240 s. Die Zusammenhangskomponenten sind
Szenen-Ketten: Sendung verkettet ueber Schuss/Gegenschuss und
wiederkehrende Kulissen ueber viele Minuten (auch eine neue Szene haengt
ueber ihre eigenen Einstellungen an), ein Spot verkettet nur sich selbst
(20-30 s), weil Spots einander fremd sind. Weil Kanten hoechstens 240 s
ueberbruecken, verbinden Trenner und Spots, die in jedem Block
wiederkehren, die Bloecke nicht (Bloecke sind laenger). Die Richtung traegt
Kanteninformation: kurz vor einem Werbeblock reicht die Kette weit zurueck
und kaum nach vorn.

Spalten (018 unveraendert 4, dazu 3 neu), standardisiert mit Mittel/
Streuung aus dem Trainingsteil:
  018: vor, nach, min, max (groesster Kosinus in [s-240,s-20] / [s+20,s+240])
  neu: log1p(Reichweite zurueck), log1p(Reichweite vor), log1p(Spanne)
       der Vereinigung aller Ketten, mit denen die Sekunde eine Kante hat.
Fuer Vorrats-Sekunden ist das genau ihre eigene Komponente; jede andere
Sekunde (Kalibrierung/Eval) haengt sich ueber ihre Kanten zum Vorrat an.
Damit ist die Definition in allen Teilen gleich (Vorrat immer jede
4. Sekunde, Sekunden aus ktx["sek"]). Kein Zufall in der Merkmalsbildung.
"""
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BESCHREIBUNG = ("Klammer-Kette: 018-Wiederkehr plus Reichweite der Szenen-Kette "
                "(Zusammenhangskomponente im Kosinus-Graph, Kanten <= 240 s)")

NAH_S = 20
FERN_S = 240
VORRAT_S = 4
TAU = 0.42  # auf dem Trainingsteil gewaehlt (0.35/0.42/0.5/0.6/0.7/0.8), AUC der Spanne am hoechsten
_STAT = {}


def _merkmale(E, sek, vorrat):
    e = torch.from_numpy(np.ascontiguousarray(E))
    e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ts = torch.from_numpy(sek.astype(np.int64))
    aus = np.full((len(E), 7), -1.0, np.float32)
    aus[:, 4:] = 0.0
    if len(vorrat) == 0:
        return aus
    pv = e[vorrat]
    tv = ts[vorrat]
    # Graph auf dem Vorrat
    S = pv @ pv.T
    d = (tv[:, None] - tv[None, :]).abs()
    kante = (S >= TAU) & (d <= FERN_S) & (d > 0)
    i, j = np.nonzero(kante.numpy())
    nv = len(vorrat)
    g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(nv, nv))
    _, komp = connected_components(g, directed=False)
    komp = torch.from_numpy(komp.astype(np.int64))
    nk = int(komp.max()) + 1
    tvf = tv.double()
    kmin = torch.full((nk,), float("inf"), dtype=torch.float64).scatter_reduce(
        0, komp, tvf, "amin")
    kmax = torch.full((nk,), float("-inf"), dtype=torch.float64).scatter_reduce(
        0, komp, tvf, "amax")
    vmin = kmin[komp]
    vmax = kmax[komp]
    for a in range(0, len(E), 4096):
        s = e[a:a + 4096] @ pv.T
        t = ts[a:a + 4096]
        dd = t[:, None] - tv[None, :]
        vor = (dd >= NAH_S) & (dd <= FERN_S)
        nach = (-dd >= NAH_S) & (-dd <= FERN_S)
        v = s.masked_fill(~vor, -1.0).max(1).values
        n = s.masked_fill(~nach, -1.0).max(1).values
        verb = (s >= TAU) & (dd.abs() <= FERN_S) & (dd != 0)
        tf = t.double()
        lo = torch.where(verb, vmin[None, :], torch.tensor(float("inf"), dtype=torch.float64)).min(1).values
        hi = torch.where(verb, vmax[None, :], torch.tensor(float("-inf"), dtype=torch.float64)).max(1).values
        lo = torch.minimum(lo, tf)
        hi = torch.maximum(hi, tf)
        rueck = (tf - lo).float()
        vorw = (hi - tf).float()
        aus[a:a + 4096] = torch.stack(
            [v, n, torch.minimum(v, n), torch.maximum(v, n),
             torch.log1p(rueck), torch.log1p(vorw), torch.log1p(rueck + vorw)], 1).numpy()
    return aus


def roh(X, ktx):
    """Ungestandardisierte Zusatzspalten (auch fuer Diagnose)."""
    B = np.asarray(X[:, :1280], np.float32)
    if ktx["teil"] == "training" or "mu" not in _STAT:
        _STAT["mu"] = B.mean(0)
    mu = _STAT["mu"]
    schritt = int(ktx["schritt"])
    rec = np.asarray(ktx["rec"])
    sek_alle = ktx.get("sek")
    zus = np.zeros((len(X), 7), np.float32)
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        if sek_alle is not None:
            sek = np.asarray(sek_alle)[m].astype(np.int64)
        else:
            sek = np.arange(len(m)) * schritt
        vorrat = np.flatnonzero(sek % VORRAT_S == 0)
        zus[m] = _merkmale(B[m] - mu, sek, vorrat)
    return zus


def spalten(teil, X, ktx):
    ktx = dict(ktx, teil=teil)
    zus = roh(X, ktx)
    if teil == "training" or "sd" not in _STAT:
        _STAT["m"] = zus.mean(0)
        _STAT["sd"] = zus.std(0) + 1e-6
    zus = (zus - _STAT["m"]) / _STAT["sd"]
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
