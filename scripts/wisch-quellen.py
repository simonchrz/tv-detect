#!/usr/bin/env python3
"""Pruefkarten fuer die Wischqueue: Labels zeigen, die vermutlich falsch sind
oder die den Massstab stellen, ohne dass ein Mensch sie je gesehen hat.

WARUM
-----
Fehlerbudget 06.10.: Dekoder-Deckel 0.001, NN-Deckel 0.017 -- und daneben
2002 s, in denen das LABEL falsch aussieht (1401 s davon "Wiederholungs-Anker
sagen Werbung, Label sagt Sendung"). Dazu 25 maschinelle Labels in golden/test
(massstab-audit): Golden 0.964 mit ihnen, 0.985 auf den 22 menschlichen. Ein
besseres Modell kann gegen eine falsche Wahrheit nicht gewinnen; jede Nacht
endet deshalb "gleich gut". Die Wischqueue ist der billigste Weg, das zu
pruefen: ein Wisch je Sekunde statt einer ganzen Aufnahme.

ZWEI QUELLEN (Spalte source)
----------------------------
  anker     Abschnitte >= --min-s, in denen Wiederholungs-Anker (96.8 %
            Praezision, scripts/wiederholung.py) Werbung sagen und das Label
            Sendung. Karte in der Mitte, lange Abschnitte bis zu drei.
            Laengste zuerst.
  massstab  Maschinelle Labels (auto-confirm, Agent) im golden-Satz und im
            test-Eimer, deren Aufnahme noch lebt: je Block die Mitte und, wenn
            der Block lang genug ist, 20 s innen/aussen an jeder Kante, dazu
            alle 5 min eine Stichprobe in der Sendung. Sind alle beantwortet
            und keine widerspricht, hebt der tv-recorder das Label (Heben per
            Wisch, wisch_anheben.go). Vorrang vor den Anker-Karten.

⚠️ Ein Wisch macht ein maschinelles Label NICHT menschlich (Punkt-Marke, keine
Review der Aufnahme; agent_review_schutzkette). Er korrigiert die Sekunde, und
die Abweichungen zeigen, welche Aufnahme "Gepruefte" Review lohnt.

Ausgabe: TSV wie head.uncertain.txt (uuid, time_s, probability, title, source).
Der tv-recorder filtert Raender (60 s), schon Gewischtes und fehlende
Aufnahmen selbst; dieses Skript liest nur und schreibt nur die Ausgabedatei.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ARCHIV = Path.home() / ".cache/tvd-train-archive"
# Labels aus dem Trainings-Snapshot derselben Nacht (die Sicherung laeuft
# erst danach und waere einen Tag alt).
SNAPSHOT = Path("/tmp/tv-train-snapshot")
BILD = Path.home() / ".cache/tvd-wiederholung/korpus"
_HIER = Path(__file__).resolve().parent
KANTE_S = 20
SENDUNG_S = 300


def _lade(name, datei):
    spec = importlib.util.spec_from_file_location(name, _HIER / datei)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def labels(pfad):
    try:
        roh = json.loads(pfad.read_text())
    except Exception:
        return None, None
    bl = sorted((float(a), float(b)) for a, b in (roh.get("ads") or []) if float(b) > float(a))
    return roh, bl


def abschnitte(maske):
    """[(start, ende)] zusammenhaengender True-Laeufe, ende exklusiv."""
    d = np.diff(np.concatenate([[0], maske.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def anker_karten(uuid, bl, min_s):
    p = BILD / f"{uuid}.json"
    if not p.is_file():
        return []
    try:
        anker = [(float(a["window_start_s"]), float(a["end_s"]))
                 for a in json.loads(p.read_text())["anchored"]]
    except Exception:
        return []
    if not anker:
        return []
    n = int(max([e for _, e in anker] + [e for _, e in bl] + [0])) + 1
    A = np.zeros(n, bool)
    T = np.zeros(n, bool)
    for s, e in anker:
        A[int(s):int(e)] = True
    for s, e in bl:
        T[int(s):int(e)] = True
    aus = []
    for s, e in abschnitte(A & ~T):
        laenge = int(e - s)
        if laenge < min_s:
            continue
        punkte = [s + laenge / 2] if laenge < 90 else [s + laenge / 4, s + laenge / 2, s + 3 * laenge / 4]
        aus += [(laenge, uuid, round(float(t)), "anker") for t in punkte]
    return aus


def dauer(rec_dir):
    """Laenge aus index.m3u8 (Summe EXTINF), 0 wenn unbekannt."""
    try:
        return sum(float(z.split(":", 1)[1].split(",")[0])
                   for z in (rec_dir / "index.m3u8").read_text().splitlines()
                   if z.startswith("#EXTINF:"))
    except (OSError, ValueError, IndexError):
        return 0.0


def massstab_karten(uuid, bl, laenge=0.0):
    """Blockmitte, ±KANTE_S an jeder Kante und Stichproben in der Sendung
    (alle SENDUNG_S): ohne diese wuerde ein Block, der im Label ganz fehlt,
    nie gefragt — und "Heben per Wisch" (tv-recorder wisch_anheben.go) haette
    nur die vorhandenen Bloecke geprueft."""
    aus = []

    def in_block(t):
        return any(s <= t <= e for s, e in bl)

    for s, e in bl:
        aus.append((uuid, round((s + e) / 2)))
        if e - s >= 3 * KANTE_S:
            aus += [(uuid, round(s + KANTE_S)), (uuid, round(e - KANTE_S))]
        for t in (s - KANTE_S, e + KANTE_S):
            if t > 0 and not in_block(t):
                aus.append((uuid, round(t)))
    # Sendungsabschnitte: Luecken zwischen den Bloecken (und bis zum Ende,
    # wenn die Laenge bekannt ist), mit 60 s Abstand zu jeder Kante.
    grenzen = [0.0] + [x for b in bl for x in b] + ([laenge] if laenge > 0 else [])
    for a, b in zip(grenzen[::2], grenzen[1::2]):
        lo, hi = a + 60, b - 60
        if hi - lo < 0:
            continue
        n = max(1, round((hi - lo) / SENDUNG_S))
        aus += [(uuid, round(lo + (hi - lo) * (i + 0.5) / n)) for i in range(n)]
    return aus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aus", default=str(Path.home() / ".cache/tv-train-head-out/wisch-quellen.txt"))
    ap.add_argument("--labels", default=str(SNAPSHOT),
                    help="Verzeichnis mit _rec_<uuid>/ads_user.json")
    ap.add_argument("--min-s", type=int, default=10, dest="min_s",
                    help="kuerzester Anker-Widerspruch, der eine Karte bekommt")
    a = ap.parse_args()

    lh = _lade("lh", "label_herkunft.py")
    aus_mod = _lade("aus", "test_ausschluss.py")
    ledger = json.loads((ARCHIV / "split-ledger.json").read_text())
    golden = set(json.loads((ARCHIV / "golden-eval-set.json").read_text()).get("uuids") or [])
    raus = aus_mod.ausgeschlossen(ARCHIV)
    massstab = {u for u in golden} | {u for u, b in ledger.items() if b == "test" and u not in raus}

    anker, mass = [], []
    n_masch = 0
    for d in sorted(Path(a.labels).glob("_rec_*")):
        uuid = d.name[5:]
        roh, bl = labels(d / "ads_user.json")
        if roh is None:
            continue
        anker += anker_karten(uuid, bl, a.min_s)
        if uuid in massstab and lh.mensch_aus_markern(roh) is not True and bl:
            n_masch += 1
            mass += massstab_karten(uuid, bl, dauer(d))

    anker.sort(key=lambda z: -z[0])
    # Massstab zuerst, aelteste Aufnahme zuerst: diese Labels verschwinden mit
    # der Aufnahme (14-Tage-Fenster), und nur vollstaendig beantwortete
    # koennen gehoben werden. Der tv-recorder nimmt je Runde hoechstens eine
    # Karte je Aufnahme, die Reihenfolge innerhalb einer Aufnahme bleibt.
    def alter(u):
        try:
            return int(u.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return 0
    mass.sort(key=lambda z: alter(z[0]))
    zeilen = [f"{u}\t{t}.0\t0\t\tmassstab" for u, t in mass]
    zeilen += [f"{u}\t{t}.0\t0\t\t{q}" for _, u, t, q in anker]

    Path(a.aus).write_text("# uuid\ttime_s\tprobability\ttitle\tsource\n" + "\n".join(zeilen) + "\n")
    n_anker_rec = len({z[1] for z in anker})
    print(f"wisch-quellen: {len(anker)} Anker-Karten aus {n_anker_rec} Aufnahmen, "
          f"{len(mass)} Massstab-Karten aus {n_masch} maschinellen Labels → {a.aus}")


if __name__ == "__main__":
    main()
