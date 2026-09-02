#!/usr/bin/env python3
"""Feature-Cache aufräumen: pro Aufnahme nur den neuesten Stand behalten.

Der Cache-Name lautet <uuid>-<source_mtime>-<key>.npy. Jede Invalidierung
(Re-Filter, Recovery, Chase-Play) ändert die Source-mtime und legt einen
neuen Stand daneben — die alten liest niemand mehr, sie liegen nur herum
(09-2026: 27 GB in 375 Aufnahmen). Ausnahme: das Train-Archiv verweist per
`feature_npy` auf einen konkreten Pfad; fehlt der, fällt die Aufnahme
still aus dem Training (train-head.py, Archiv-Injektion). Diese Pfade
bleiben immer stehen, auch wenn ein neuerer Stand existiert.

Ohne --loeschen nur Trockenlauf.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FEATURES = Path.home() / ".cache/tvd-features"
ARCHIV = Path.home() / ".cache/tvd-train-archive"
# uuid = 32 Hex (alt) oder dvr-<slug>-<start> (neu); danach der mtime-Stempel.
NAME = re.compile(r"^([0-9a-f]{32}|dvr-[a-z0-9-]+?-\d{9,10})-(\d{9,10})-(.+)\.npy$")


def zu_loeschen(dateien, referenzen):
    """dateien: Namen im Cache; referenzen: vom Archiv referenzierte Namen.
    Liefert die Namen, die weg können: pro (uuid, key) alles außer dem
    Stand mit dem höchsten mtime-Stempel — nie ein referenzierter."""
    staende = defaultdict(list)
    for name in dateien:
        m = NAME.match(name)
        if not m:
            continue  # fremde Datei, nicht anfassen
        uuid, stempel, key = m.groups()
        staende[(uuid, key)].append((int(stempel), name))
    weg = []
    for gruppe in staende.values():
        gruppe.sort()
        for _, name in gruppe[:-1]:
            if name not in referenzen:
                weg.append(name)
    return sorted(weg)


def archiv_referenzen(archiv):
    import numpy as np
    refs = set()
    for npz in archiv.glob("*.npz"):
        try:
            meta = json.loads(str(np.load(npz, allow_pickle=False)["meta"]))
        except Exception:
            continue
        p = meta.get("feature_npy")
        if p:
            refs.add(Path(p).name)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loeschen", action="store_true", help="wirklich löschen")
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--archiv", type=Path, default=ARCHIV)
    args = ap.parse_args()

    refs = archiv_referenzen(args.archiv)
    dateien = [p.name for p in args.features.glob("*.npy")]
    weg = zu_loeschen(dateien, refs)
    groesse = sum((args.features / n).stat().st_size for n in weg)
    print("Stände: %d Dateien, %d Aufnahmen, %d Archiv-Referenzen" % (
        len(dateien), len({NAME.match(n).group(1) for n in dateien if NAME.match(n)}),
        len(refs)))
    print("alt: %d Dateien, %.1f GB%s" % (
        len(weg), groesse / 1e9, "" if args.loeschen else " (Trockenlauf)"))
    if not args.loeschen:
        return 0
    n = 0
    for name in weg:
        try:
            (args.features / name).unlink()
            n += 1
        except FileNotFoundError:
            pass
    print("gelöscht: %d" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
