"""Klammer-Wiederkehr: kehrt das Bild dieser Sekunde in den Minuten davor
bzw. danach wieder?

Sendung, auch ohne Logo (Spielfilm, Kindersender, ausgewaschenes Logo),
wiederholt sich LOKAL: Schuss/Gegenschuss, dieselben Personen und Kulissen
innerhalb einer Szene, also 20 s bis 4 min vor und nach der Sekunde. Ein
Werbeblock ist eine Folge kurzer, einander fremder Spots: im selben Fenster
findet eine Werbesekunde kaum ein aehnliches Bild. Der Kopf sieht jede
Sekunde fuer sich und weiss das nicht; ohne Logo erfindet er deshalb in
Filmen Werbung (Pegel-Problem aus 016).

Anders als 011 (Selbstaehnlichkeit ueber die GANZE Aufnahme, Nachbarn unter
90 s ausgeschlossen, Median der Aufnahmen sank): hier nur ein lokales
Fenster 20..240 s, getrennt nach vorher und nachher. Gerade die naechsten
Minuten tragen die Szenen-Wiederkehr; die globale Suche findet dagegen
auch in Werbung Treffer (derselbe Spot im naechsten Block) und mischte das
in dieselbe Spalte. Anders als 015 (Bildwechsel-Dichte, Abstand zu t+-4 s)
keine Schnittfrequenz, sondern Wiederkehr; Teleshopping bleibt ueber OCR
und Logo unterscheidbar. Anders als 004/O16 kein Fenster ueber Logo/OCR.

Spalten (4), Kosinus auf dem zentrierten Backbone (Mittel aus dem
Trainingsteil), standardisiert mit Mittel/Streuung aus dem Trainingsteil:
  vor   groesste Aehnlichkeit zu Sekunden in [s-240, s-20]
  nach  groesste Aehnlichkeit zu Sekunden in [s+20, s+240]
  min(vor, nach), max(vor, nach)
Leeres Fenster (Aufnahme-Rand) = -1.

Schritt: der Vergleichsvorrat ist IMMER jede 4. Sekunde der Aufnahme (im
Training alle geladenen Zeilen, in Kalibrierung/Eval die Zeilen mit
Sekunde % 4 == 0), damit das Maximum in allen Teilen aus gleich dichten
Vorraeten kommt. Sekunden aus ktx["sek"] (sonst Zeilenindex * schritt).
Kein Zufall in der Merkmalsbildung.
"""
import numpy as np
import torch

BESCHREIBUNG = ("Klammer-Wiederkehr: max. Kosinus des zentrierten Backbones zu "
                "[s-240,s-20] und [s+20,s+240] derselben Aufnahme (vor/nach/min/max)")

NAH_S = 20
FERN_S = 240
VORRAT_S = 4
_STAT = {}


def _merkmale(E, sek, vorrat):
    e = torch.from_numpy(np.ascontiguousarray(E))
    e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    pv = e[vorrat]
    ts = torch.from_numpy(sek.astype(np.int64))
    tv = ts[vorrat]
    aus = np.full((len(E), 4), -1.0, np.float32)
    if len(vorrat) == 0:
        return aus
    for a in range(0, len(E), 4096):
        s = e[a:a + 4096] @ pv.T
        d = ts[a:a + 4096, None] - tv[None, :]
        vor = (d >= NAH_S) & (d <= FERN_S)
        nach = (-d >= NAH_S) & (-d <= FERN_S)
        v = s.masked_fill(~vor, -1.0).max(1).values
        n = s.masked_fill(~nach, -1.0).max(1).values
        aus[a:a + 4096] = torch.stack(
            [v, n, torch.minimum(v, n), torch.maximum(v, n)], 1).numpy()
    return aus


def spalten(teil, X, ktx):
    B = np.asarray(X[:, :1280], np.float32)
    if teil == "training" or "mu" not in _STAT:
        _STAT["mu"] = B.mean(0)
    mu = _STAT["mu"]
    schritt = int(ktx["schritt"])
    rec = np.asarray(ktx["rec"])
    sek_alle = ktx.get("sek")
    zus = np.zeros((len(X), 4), np.float32)
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        if sek_alle is not None:
            sek = np.asarray(sek_alle)[m].astype(np.int64)
        else:
            sek = np.arange(len(m)) * schritt
        vorrat = np.flatnonzero(sek % VORRAT_S == 0)
        zus[m] = _merkmale(B[m] - mu, sek, vorrat)
    if teil == "training" or "sd" not in _STAT:
        _STAT["m"] = zus.mean(0)
        _STAT["sd"] = zus.std(0) + 1e-6
    zus = (zus - _STAT["m"]) / _STAT["sd"]
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
