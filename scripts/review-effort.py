#!/usr/bin/env python3
"""Wie viel muss der Mensch am Vorschlag der Maschine noch korrigieren?

Die einzige Zahl in diesem Stack, die NICHT aus der Schleife selbst stammt.
Block-IoU auf dem Golden-Satz ist ein Stellvertreter: er kann steigen, während
sich für den Menschen nichts ändert. Diese Zahl misst das Gegenteilige — sie
kommt von der Person, die am Ende schneidet.

Gemessen wird je Aufnahme die symmetrische Differenz zwischen `auto` und
`user` aus `/recording/<uuid>/ads`, in Sekunden je Stunde Aufnahme. 0 = die
Maschine lag genau richtig.

    review-effort.py [--host …] [--limit N] [--csv datei]

⚠️ Drei Dinge, die diese Zahl NICHT ist — sie stehen hier, damit sie nicht
später als Entdeckung verkauft werden:

1. `auto` ist der AKTUELLE Vorschlag, nicht der, den der Mensch beim Review
   gesehen hat: Cutlists werden bei Konfigurationsänderungen neu erkannt
   (siehe Memory detect_config_fingerprint_invalidation). Die Zahl misst also
   "wie weit liegt das heutige Modell von der menschlichen Wahrheit", nicht
   "wie viel Arbeit war es damals".
2. Auf Aufnahmen, die im TRAINING waren, ist die Übereinstimmung geschönt —
   das Modell hat genau diese Labels gesehen. Deshalb wird getrennt
   ausgewiesen; belastbar ist nur die Spalte "nicht im Training".
3. `edited=true` heißt NICHT "ein Mensch war dran". Auto-Confirm schreibt
   ein synthetisches `ads_user.json` mit `auto_confirmed_at` — dort ist
   `user == auto` per Konstruktion, der Korrekturaufwand also zwangsläufig 0.
   Die erste Fassung dieses Skripts (2026-08-09) hat genau darauf
   hereingefallen und für August "100 % exakt" gemeldet; von 250 Dateien
   waren 101 auto-bestätigt. Gezählt wird deshalb nur, was KEIN
   `auto_confirmed_at` trägt.

Die Herkunft der Daten ist geteilt, weil keine Quelle beides hat: `auto` und
`user` kommen aus dem Endpunkt (nie die Caches catten, s. Memory
never_cat_gateway_caches), die Auto-Confirm-Markierung aus dem lokalen
Label-Backup-Spiegel (~/tv-labels-backup, täglich 04:30 vom Pi).
"""
import argparse
import json
import statistics
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path.home() / ".cache/tvd-train-archive/split-ledger.json"
SPIEGEL = Path.home() / "tv-labels-backup"


def auto_bestaetigte(spiegel):
    """uuids, deren ads_user.json von Auto-Confirm stammt, nicht vom Menschen."""
    out = set()
    for p in spiegel.glob("_rec_*/ads_user.json"):
        try:
            if json.loads(p.read_text()).get("auto_confirmed_at"):
                out.add(p.parent.name[len("_rec_"):])
        except Exception:
            continue
    return out


def hole(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def sekunden(bloecke):
    """Vereinigte Länge einer Blockliste (überlappende Blöcke einmal)."""
    if not bloecke:
        return 0.0
    s = sorted((float(a), float(b)) for a, b in bloecke if b > a)
    if not s:
        return 0.0
    ges, (ca, cb) = 0.0, s[0]
    for a, b in s[1:]:
        if a > cb:
            ges += cb - ca
            ca, cb = a, b
        else:
            cb = max(cb, b)
    return ges + (cb - ca)


def symmetrische_differenz(x, y):
    """Sekunden, die in genau einer der beiden Listen liegen.

    |A Δ B| = |A| + |B| − 2·|A ∩ B|, und |A ∩ B| = |A| + |B| − |A ∪ B|.
    """
    a, b = sekunden(x), sekunden(y)
    vereinigt = sekunden(list(x or []) + list(y or []))
    schnitt = a + b - vereinigt
    return a + b - 2 * schnitt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://raspberrypi5lan:9984")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    try:
        recs = hole(f"{args.host}/api/recordings?limit={args.limit}")["recordings"]
    except Exception as e:
        print(f"tv-recorder nicht erreichbar: {e}", file=sys.stderr)
        return 2
    split = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    if not SPIEGEL.exists():
        print(f"Label-Spiegel {SPIEGEL} fehlt — ohne ihn ist Auto-Confirm "
              f"nicht von menschlichem Review zu unterscheiden, und die Zahl "
              f"waere geschoent statt unvollstaendig. Abbruch.", file=sys.stderr)
        return 2
    auto_uuids = auto_bestaetigte(SPIEGEL)

    def einer(r):
        try:
            a = hole(f"{args.host}/recording/{r['uuid']}/ads")
        except Exception:
            return None
        if not a.get("edited"):
            return None          # kein Review = keine menschliche Wahrheit
        if r["uuid"] in auto_uuids:
            return None          # Auto-Confirm: user == auto per Konstruktion
        dauer = float(a.get("duration_s") or r.get("duration") or 0)
        if dauer < 60:
            return None
        diff = symmetrische_differenz(a.get("auto"), a.get("user"))
        return {
            "uuid": r["uuid"], "titel": r.get("title", ""),
            "start": r.get("start") or 0,
            "kanal": r.get("channel", ""),
            "dauer_h": dauer / 3600.0,
            "korrektur_s_pro_h": diff / (dauer / 3600.0),
            "im_training": split.get(r["uuid"]) == "train",
            "bekannt": r["uuid"] in split,
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        zeilen = [z for z in ex.map(einer, recs) if z]

    if not zeilen:
        print("Keine reviewten Aufnahmen gefunden.")
        return 0

    print("=" * 68)
    print("KORREKTURAUFWAND — Sekunden je Stunde, die der Mensch verschoben hat")
    print("=" * 68)
    print(f"  {len(zeilen)} von Menschen reviewte Aufnahmen "
          f"({len(recs)} gesamt, {len(auto_uuids)} auto-bestaetigt und "
          f"deshalb ausgeschlossen)\n")

    def block(name, teil):
        if not teil:
            print(f"  {name:24s}  —")
            return
        w = sorted(z["korrektur_s_pro_h"] for z in teil)
        null = sum(1 for v in w if v < 1.0)
        print(f"  {name:24s}  n={len(w):>3}  Median {statistics.median(w):>7.1f}  "
              f"Mittel {statistics.mean(w):>7.1f}  exakt {null:>3} "
              f"({100*null/len(w):.0f}%)")

    block("alle", zeilen)
    block("nicht im Training", [z for z in zeilen if not z["im_training"]])
    block("im Training (geschönt)", [z for z in zeilen if z["im_training"]])

    # Der Verlauf ist die eigentliche Frage: wird es besser?
    print("\n  Nach Monat der Ausstrahlung, nur NICHT im Training:")
    nach_monat = defaultdict(list)
    for z in zeilen:
        if z["im_training"] or not z["start"]:
            continue
        m = datetime.fromtimestamp(z["start"], timezone.utc).strftime("%Y-%m")
        nach_monat[m].append(z["korrektur_s_pro_h"])
    for m in sorted(nach_monat):
        w = sorted(nach_monat[m])
        null = sum(1 for v in w if v < 1.0)
        print(f"    {m}  n={len(w):>3}  Median {statistics.median(w):>7.1f}  "
              f"exakt {100*null/len(w):>3.0f}%")
    if len(nach_monat) < 3:
        print("    (zu wenige Monate für einen Verlauf — beobachten)")

    print("\n  Schlechteste 5 (nicht im Training):")
    for z in sorted((z for z in zeilen if not z["im_training"]),
                    key=lambda z: -z["korrektur_s_pro_h"])[:5]:
        print(f"    {z['korrektur_s_pro_h']:>7.1f} s/h  {z['kanal']:<14s} "
              f"{z['titel'][:38]}  ({z['uuid']})")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
            w.writeheader()
            w.writerows(zeilen)
        print(f"\n  → {args.csv}")

    print("\n  Lesart: Block-IoU kann steigen, während diese Zahl steht. Dann "
          "verbessert die Schleife etwas, das niemanden erreicht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
