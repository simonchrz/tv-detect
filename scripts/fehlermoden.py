#!/usr/bin/env python3
"""Fehlermoden-Zerlegung: WO wohnt der restliche Fehler?

Vergleicht je Aufnahme den Auto-Vorschlag (ads.json) mit dem menschlichen
Label (ads_user.json) und klassifiziert jede Abweichung:

  GRENZE     — Blockpaar deckt sich, aber Start/Ende verschoben.
               Asymmetrie wird getrennt gezählt: Sekunden SENDUNG
               weggeschnitten (schlimm) vs Sekunden WERBUNG stehen
               gelassen (egal).
  PHANTOM    — Auto-Block ohne menschliche Entsprechung (erfundene Werbung).
  VERPASST   — Mensch-Block ohne Auto-Entsprechung (übersehene Werbung).

    fehlermoden.py [--uuids u1,u2,…] [--snapshot PFAD]

Ohne --uuids: alle Golden-Aufnahmen. Die Zerlegung wählt die nächsten
Fragen — sie ist Beleg-Lieferant, kein Urteil (das fällt weiter das Audit).

⚠️ ads.json ist der Vorschlag des LAUFENDEN Produktionskopfs (inkl.
Config-Fingerprint-Redetects), nicht des Kandidaten. Für die Frage „welche
FEHLERKLASSE dominiert" ist das das richtige Objekt — der Zuschauer sieht
genau diese Vorschläge.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SNAPSHOT = Path("/tmp/tv-train-snapshot")
ARCHIV = Path.home() / ".cache/tvd-train-archive"


def bloecke(pfad):
    if not pfad.exists():
        return None
    j = json.loads(pfad.read_text())
    if isinstance(j, list):
        roh = j
    else:
        roh = j.get("ads") or []
    return [(float(a), float(b)) for a, b in roh if float(b) > float(a)]


def ueberlapp(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def zerlege(auto, user):
    """Klassifiziert die Abweichungen einer Aufnahme."""
    aus = {"grenze": [], "phantom": [], "verpasst": [],
           "sendung_geschnitten_s": 0.0, "werbung_gelassen_s": 0.0}
    benutzt = set()
    for ub in user:
        # bester Auto-Partner nach Überlappung
        best, best_ov = None, 0.0
        for i, ab in enumerate(auto):
            if i in benutzt:
                continue
            ov = ueberlapp(ab, ub)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is None or best_ov <= 0:
            aus["verpasst"].append(ub)
            # Alles, was der Mensch als Werbung markiert hat und der
            # Vorschlag nicht: Werbung stehen gelassen.
            aus["werbung_gelassen_s"] += ub[1] - ub[0]
            continue
        benutzt.add(best)
        ab = auto[best]
        d_start, d_ende = ab[0] - ub[0], ab[1] - ub[1]
        if abs(d_start) > 0.5 or abs(d_ende) > 0.5:
            aus["grenze"].append((ub, ab, round(d_start, 1), round(d_ende, 1)))
        # Asymmetrie: Auto-Block ragt ÜBER den menschlichen hinaus =
        # Sendung weggeschnitten; bleibt er dahinter zurück = Werbung
        # stehen gelassen.
        # d_start<0: Auto beginnt VOR dem Label → Sendung geschnitten.
        # d_ende>0:  Auto endet NACH dem Label → Sendung geschnitten.
        aus["sendung_geschnitten_s"] += max(0.0, -d_start) + max(0.0, d_ende)
        aus["werbung_gelassen_s"] += max(0.0, d_start) + max(0.0, -d_ende)
    for i, ab in enumerate(auto):
        if i not in benutzt:
            aus["phantom"].append(ab)
            aus["sendung_geschnitten_s"] += ab[1] - ab[0]
    return aus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuids", default=None)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = ap.parse_args()

    if args.uuids:
        uuids = [u.strip() for u in args.uuids.split(",") if u.strip()]
    else:
        uuids = json.loads(
            (ARCHIV / "golden-eval-set.json").read_text())["uuids"]

    summe = Counter()
    zeilen = []
    fehlend = 0
    for u in uuids:
        d = args.snapshot / f"_rec_{u}"
        user = bloecke(d / "ads_user.json")
        auto = bloecke(d / "ads.json")
        if user is None or auto is None:
            fehlend += 1
            continue
        z = zerlege(auto, user)
        summe["grenze"] += len(z["grenze"])
        summe["phantom"] += len(z["phantom"])
        summe["verpasst"] += len(z["verpasst"])
        summe["sendung_s"] += z["sendung_geschnitten_s"]
        summe["werbung_s"] += z["werbung_gelassen_s"]
        if z["grenze"] or z["phantom"] or z["verpasst"]:
            zeilen.append((u, z))

    print(f"Fehlermoden über {len(uuids) - fehlend} Aufnahmen "
          f"({fehlend} ohne beide Dateien):\n")
    for u, z in sorted(zeilen, key=lambda p: -(
            p[1]["sendung_geschnitten_s"] + p[1]["werbung_gelassen_s"])):
        teile = []
        if z["grenze"]:
            teile.append(f"{len(z['grenze'])}×GRENZE "
                         + " ".join(f"[{ds:+g}/{de:+g}s]"
                                    for _, _, ds, de in z["grenze"][:3]))
        if z["phantom"]:
            teile.append(f"{len(z['phantom'])}×PHANTOM "
                         + " ".join(f"({a:.0f}-{b:.0f})"
                                    for a, b in z["phantom"][:2]))
        if z["verpasst"]:
            teile.append(f"{len(z['verpasst'])}×VERPASST "
                         + " ".join(f"({a:.0f}-{b:.0f})"
                                    for a, b in z["verpasst"][:2]))
        print(f"  {u}")
        print(f"      Sendung geschnitten {z['sendung_geschnitten_s']:.0f}s, "
              f"Werbung gelassen {z['werbung_gelassen_s']:.0f}s — "
              + "; ".join(teile))

    print(f"\nSUMME: {summe['grenze']}×Grenze, {summe['phantom']}×Phantom, "
          f"{summe['verpasst']}×Verpasst")
    print(f"ASYMMETRIE: {summe['sendung_s']:.0f}s Sendung weggeschnitten "
          f"vs {summe['werbung_s']:.0f}s Werbung stehen gelassen")
    print("\nLesart: Grenzen-dominiert → Decoder-Kosten/Snaps; "
          "Verpasst-dominiert → Erkennung/Prior; Phantom-dominiert → "
          "Schwelle/Prior. Die Zahlen WÄHLEN die nächste Frage, sie "
          "entscheiden sie nicht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
