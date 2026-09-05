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
never_cat_gateway_caches), die Auto-Confirm-Markierung direkt aus den
`ads_user.json` auf dem Pi (per ssh, Quelle der Wahrheit) — und nur wenn der
Pi nicht antwortet aus dem lokalen Label-Backup-Spiegel (~/tv-labels-backup,
täglich 04:32 vom Pi).

4. Der Spiegel allein reicht NICHT (Befund 2026-09-05): die Fingerprint-
   Bestätigung schreibt um 08:00, der Spiegel wurde um 04:32 gezogen, der
   Tagesdurchgang liest um 08:07. Jede Bestätigung des Morgens galt so einen
   Tag lang als Mensch mit 0 s/h — "2026-09 n=1, 100 % exakt" war genau das.
   Deshalb wird die Markierung jetzt live gelesen; der Spiegel ist Rückfall
   und wird dazuvereinigt (er kann nur Dateien kennen, die es gab).
"""
import argparse
import json
import statistics
import subprocess
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path.home() / ".cache/tvd-train-archive/split-ledger.json"
SPIEGEL = Path.home() / "tv-labels-backup"


def auto_bestaetigte(spiegel):
    """uuids, deren ads_user.json von Auto-Confirm stammt, nicht vom Menschen.

    Zwei Maschinen-Schreiber, zwei Markierungen (gleiche Bauart wie in
    kanten-schatten.label_quelle, Befund 2026-09-03):
      * `auto_confirmed_at`  — Auto-Confirm des Recorders.
      * `auto_confirmed_via_fingerprint` — Fingerprint-Bestaetigung
        (tv-receiver learning.go): schreibt die Auto-Bloecke unveraendert
        plus `reviewed_at`, OHNE `auto_confirmed_at`. Sah hier wie ein
        Mensch mit 0 s/h aus und hat fuer 2026-09 "n=2, 100 % exakt"
        gemeldet — beide Aufnahmen waren Fingerprint-Bestaetigungen.
    """
    out = set()
    for p in spiegel.glob("_rec_*/ads_user.json"):
        try:
            d = json.loads(p.read_text())
            if not isinstance(d, dict):
                continue
            if d.get("auto_confirmed_at") or d.get("auto_confirmed_via_fingerprint"):
                out.add(p.parent.name[len("_rec_"):])
        except Exception:
            continue
    return out


PI_HOST = "raspberrypi5lan"
PI_HLS = "/mnt/tv/hls"
MARKIERUNGEN = ("auto_confirmed_at", "auto_confirmed_via_fingerprint")


def auto_bestaetigte_live(host=PI_HOST, hls=PI_HLS, timeout=20):
    """uuids mit Auto-Confirm-Markierung, direkt von den Dateien auf dem Pi.

    Liefert None, wenn der Pi nicht antwortet — der Aufrufer faellt dann auf
    den Spiegel zurueck und sagt das. Ein leeres Set ist ein Ergebnis
    (keine Markierung), None ist keins.
    """
    muster = "|".join(MARKIERUNGEN)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
           f"grep -lE '{muster}' {hls}/_rec_*/ads_user.json; true"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    out = set()
    for zeile in r.stdout.splitlines():
        name = Path(zeile.strip()).parent.name
        if name.startswith("_rec_"):
            out.add(name[len("_rec_"):])
    return out


def auto_bestaetigte_vereinigt(live, spiegel_uuids):
    """Live-Ergebnis mit dem Spiegel vereinigen; ohne Live nur der Spiegel.

    Rueckgabe: (uuids, quelle) mit quelle in {"pi+spiegel", "spiegel"}.
    """
    if live is None:
        return set(spiegel_uuids), "spiegel"
    return set(live) | set(spiegel_uuids), "pi+spiegel"


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
    ap.add_argument("--trotz-detect", action="store_true",
                    help="auch messen, waehrend eine Detect-Welle laeuft "
                         "(die Zahlen sind dann unbrauchbar, s.u.)")
    args = ap.parse_args()

    # Waehrend einer Detect-Welle wird die Cutlist neu geschrieben: die .txt
    # ist zwischenzeitlich leer, und `auto` kommt dann aus einer Ersatzquelle
    # (Anker-Fenster), die plausibel aussieht und es nicht ist. Gemessen am
    # 2026-08-09 mitten in einer Welle: derselbe Titel sprang von 169 auf
    # 800 s/h. Nicht auffaellig genug, um von selbst aufzufallen — deshalb
    # der Riegel und nicht bloss eine Warnung.
    try:
        iz = hole(f"{args.host}/api/integrity", timeout=8)
        laufend = int(iz.get("detect_running") or 0)
        wartend = int((iz.get("queues") or {}).get("detect_pending") or 0)
        if (laufend or wartend) and not args.trotz_detect:
            print(f"Detect laeuft ({laufend} aktiv, {wartend} wartend) — "
                  f"Cutlists werden gerade neu geschrieben, die Messung waere "
                  f"unbrauchbar. Spaeter erneut, oder --trotz-detect.",
                  file=sys.stderr)
            return 3
    except Exception as e:
        print(f"warn: Detect-Zustand nicht abfragbar ({e}) — Zahlen mit "
              f"Vorsicht lesen", file=sys.stderr)

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
    live = auto_bestaetigte_live()
    auto_uuids, quelle = auto_bestaetigte_vereinigt(live, auto_bestaetigte(SPIEGEL))
    if quelle == "spiegel":
        print("warn: Pi nicht erreichbar — Auto-Confirm-Markierung nur aus dem "
              "Spiegel von 04:32. Bestaetigungen von heute frueh zaehlen dann "
              "faelschlich als Mensch mit 0 s/h.", file=sys.stderr)

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
          f"deshalb ausgeschlossen; Markierung aus: {quelle})\n")

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
