#!/usr/bin/env python3
"""Weg 3: ein kleines Budget echter Blicke — nur für den Maßstab, nie für den Korpus.

Die Lage, aus der das kommt (Ledger 2026-09-06): der Maßstab lebt von
menschlichen Labeln und bekommt keine mehr. Zwei Instrumente ohne Menschen
wurden getestet (Agenten-Referenz, Einzelgänger-Zähler) — beide zeigen
Rückschritte, keine Verbesserung, weil sie das Regime nicht abtasten, in
dem das Modell noch irrt: die Kanten. Und eine Aufnahme ist nur ~14 Tage
lang überhaupt ansehbar (Serien-Retention).

WAS DIESES SKRIPT TUT
---------------------
`--vorschlagen`  sucht EINE frische Aufnahme aus, die den Blick lohnt:
                 noch ansehbar, noch nicht menschlich gelabelt, nicht im
                 test-Eimer, bevorzugt aus einer Serie, in der das Modell
                 schwach ist, und mit wenigen Kanten (2–4 Blöcke).
                 Simon sieht sich in der App die Blockkanten an —
                 bestätigen oder schieben, ~2 Minuten.
`--versiegeln U` prüft danach, dass jetzt ein MENSCHLICHES Label da ist,
                 und legt die Aufnahme in den versiegelten Satz. Der
                 Daemon-Pin (geschuetzte_uuids) greift von selbst.

WARUM SO UND NICHT ANDERS
-------------------------
* **Kanten, nicht Vollsichtung.** Von 173 Label-Blöcken war genau EINER
  ganz verpasst (Ledger §3af); der Fehler sitzt in der Kantenlage. Zwei
  Minuten an den Kanten sind mehr wert als zwanzig über die ganze Folge.
* **Frisch, nicht alt.** Nach 14 Tagen ist die Aufnahme weg
  (Memory review_fenster_14_tage). Alles, was älter als ~10 Tage ist,
  wird gar nicht erst vorgeschlagen — sonst ist sie weg, bevor der Blick
  stattfindet.
* **Versiegeln, nicht trainieren.** Das Label geht in den versiegelten
  Satz, den kein Training je sieht. Ein menschliches Label im Korpus wäre
  eines von 744; im versiegelten Satz ist es eines von null. Dort zählt es.
* **Nie aus dem test-Eimer.** Der ist sticky gehasht; wer ihn umschichtet,
  bricht das Head-to-Head-Gate (Memory test_set_stickiness). train →
  versiegelt ist dagegen unbedenklich: der Kopf wird jede Nacht neu
  gefittet, morgen sieht er die Aufnahme nicht mehr.

⚠️ SCHREIBT IN DEN SPLIT-LEDGER — die einzige Stelle, an der das
außerhalb von train-head.py passiert. Deshalb: atomar (tmp + rename),
Sicherung daneben, Weigerung wenn gerade ein Training läuft, und nur der
Übergang train/unbekannt → versiegelt. Alles andere wird abgelehnt.
"""
import argparse
import collections
import importlib.util
import json
import os
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_HIER = Path(__file__).resolve().parent


def _lade(name):
    sp = importlib.util.spec_from_file_location(name.replace("-", "_").replace(".py", ""),
                                                _HIER / name)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


_lh = _lade("label_herkunft.py")
ARCHIV = Path.home() / ".cache/tvd-train-archive"
LEDGER = ARCHIV / "split-ledger.json"
QUELLE = Path.home() / ".cache/tv-detect-daemon/source"
MAX_ALTER_TAGE = 10       # danach ist die 14-Tage-Frist zu knapp
MAX_BLOECKE = 4           # mehr Kanten = mehr Arbeit, weniger lohnend
TRENNER = (" - ", " – ", " — ", ": ")


def hole(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def norm_titel(t):
    for s in TRENNER:
        i = t.find(s)
        if i >= 0:
            return t[:i].strip()
    return t.strip()


def schwache_serien():
    """Serie → Median-IoU aus dem letzten nächtlichen Lauf. Fehlt der, {}."""
    try:
        import numpy as np
        letzte = None
        for z in (ARCHIV / "per-rec-iou.jsonl").read_text().splitlines():
            if z.strip():
                letzte = json.loads(z)
        pr = (letzte or {}).get("champion") or {}
        titel = {}
        for f in ARCHIV.glob("*.npz"):
            try:
                m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
                titel[m["uuid"]] = norm_titel(m.get("title", ""))
            except Exception:
                continue
        je = collections.defaultdict(list)
        for u, v in pr.items():
            if u in titel:
                je[titel[u]].append(v)
        return {t: st.median(v) for t, v in je.items() if len(v) >= 2}
    except Exception:
        return {}


def mmss(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def vorschlagen(args):
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    recs = hole(f"{args.pi}/api/recordings")["recordings"]
    schwach = schwache_serien()
    jetzt = time.time()
    kandidaten = []
    for r in recs:
        u = r["uuid"]
        alter = (jetzt - r.get("start", 0)) / 86400
        if alter > MAX_ALTER_TAGE or r.get("state") != "completed":
            continue
        if led.get(u) in ("test", "versiegelt"):
            continue
        try:
            a = hole(f"{args.pi}/recording/{u}/ads")
        except Exception:
            continue
        user_raw = None
        try:
            user_raw = hole(f"{args.pi}/recording/{u}/ads").get("user")
        except Exception:
            pass
        # Das rohe ads_user.json braucht es fuer die Marker; der Endpunkt
        # liefert nur die Bloecke. Marker holen wir ueber das Backup.
        bk = Path.home() / "tv-labels-backup" / f"_rec_{u}" / "ads_user.json"
        if bk.is_file():
            try:
                if _lh.mensch_aus_markern(json.loads(bk.read_text())) is True:
                    continue          # schon menschlich — nichts zu holen
            except Exception:
                pass
        auto = a.get("auto") or []
        if not (1 <= len(auto) <= MAX_BLOECKE):
            continue
        serie = norm_titel(r.get("title", ""))
        kandidaten.append({
            "uuid": u, "titel": r.get("title", ""), "serie": serie,
            "kanal": r.get("channel", ""), "start": r.get("start", 0),
            "alter": alter, "bloecke": auto, "eimer": led.get(u, "—"),
            "serien_iou": schwach.get(serie),
            "quelle_lokal": (QUELLE / f"{u}.ts").is_file(),
        })
    if not kandidaten:
        print("Kein Kandidat: nichts Frisches ohne menschliches Label mit 1–4 Blöcken.")
        return 1
    # Reihung: schwache Serie zuerst (unbekannt = mittig), dann wenige
    # Kanten, dann juenger.
    kandidaten.sort(key=lambda k: (k["serien_iou"] if k["serien_iou"] is not None else 0.9,
                                   len(k["bloecke"]), k["alter"]))
    if args.alle:
        print(f"{'Serie':<28} {'IoU':>5} {'Bl':>3} {'Alter':>6} {'Eimer':<6} uuid")
        for k in kandidaten[:15]:
            iou = f"{k['serien_iou']:.2f}" if k["serien_iou"] is not None else "  —"
            print(f"{k['serie'][:28]:<28} {iou:>5} {len(k['bloecke']):>3} "
                  f"{k['alter']:>5.1f}d {k['eimer']:<6} {k['uuid']}")
        return 0
    k = kandidaten[0]
    datum = time.strftime("%a %d.%m. %H:%M", time.localtime(k["start"]))
    print(f"Vorschlag: {k['titel']}")
    print(f"  {k['kanal']}, {datum}, {k['alter']:.1f} Tage alt, "
          f"noch ~{max(0, 14 - k['alter']):.0f} Tage ansehbar")
    if k["serien_iou"] is not None:
        print(f"  Serie im letzten Nachtlauf: Median-IoU {k['serien_iou']:.2f}")
    print(f"  {len(k['bloecke'])} Block/Blöcke laut Modell — diese Kanten ansehen:")
    for a, b in k["bloecke"]:
        print(f"    {mmss(a)} – {mmss(b)}")
    if not k["quelle_lokal"]:
        print("  ⚠️ Quelle noch nicht im Mac-Cache — der Pin greift erst, wenn sie da ist")
    print(f"\n  In der App unter Aufnahmen öffnen, Kanten bestätigen oder schieben.")
    print(f"  Danach:  kanten-blick.py --versiegeln {k['uuid']}")
    return 0


def versiegeln(args):
    u = args.versiegeln
    if subprocess.run(["pgrep", "-f", "train-head.py"], capture_output=True).returncode == 0:
        sys.exit("✗ Ein Training läuft — der Ledger wird gerade gelesen. Später.")
    bk = Path.home() / "tv-labels-backup" / f"_rec_{u}" / "ads_user.json"
    try:
        raw = json.loads(bk.read_text())
    except Exception:
        # Backup kann hinterherhinken -- direkt vom Pi holen.
        try:
            raw = json.loads(subprocess.run(
                ["ssh", "-n", "raspberrypi5lan", f"cat /mnt/tv/hls/_rec_{u}/ads_user.json"],
                capture_output=True, text=True, timeout=20).stdout)
        except Exception:
            sys.exit(f"✗ {u}: kein ads_user.json lesbar — wurde die Aufnahme angesehen?")
    mensch = _lh.mensch_aus_markern(raw)
    if mensch is not True:
        sys.exit(f"✗ {u}: Label ist nicht menschlich (mensch_aus_markern={mensch}). "
                 f"Marker: auto_confirmed_at={raw.get('auto_confirmed_at')!r}, "
                 f"reviewed_by={raw.get('reviewed_by')!r}. Nicht versiegelt.")
    if not raw.get("reviewed_at"):
        sys.exit(f"✗ {u}: kein reviewed_at — die Kanten wurden nicht bestätigt.")
    led = json.loads(LEDGER.read_text())
    alt = led.get(u)
    if alt == "versiegelt":
        print(f"  {u} ist bereits versiegelt.")
        return 0
    if alt == "test":
        sys.exit(f"✗ {u} liegt im test-Eimer — den rührt dieses Skript nicht an "
                 f"(test_set_stickiness).")
    if args.trocken:
        print(f"[Probe] {u}: {alt or 'nicht im Ledger'} → versiegelt")
        return 0
    sicherung = LEDGER.with_suffix(f".json.bak.{time.strftime('%Y%m%dT%H%M%S')}")
    sicherung.write_text(LEDGER.read_text())
    led[u] = "versiegelt"
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(led, indent=1, sort_keys=True))
    os.replace(tmp, LEDGER)
    n = sum(1 for v in led.values() if v == "versiegelt")
    print(f"✓ {u}: {alt or 'nicht im Ledger'} → versiegelt "
          f"(Satz jetzt {n}; Sicherung {sicherung.name})")
    if not (QUELLE / f"{u}.ts").is_file():
        print("  ⚠️ Quelle nicht im Mac-Cache — Pin schützt erst, wenn der Daemon sie holt")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vorschlagen", action="store_true")
    ap.add_argument("--alle", action="store_true", help="mit --vorschlagen: Top-15 statt einer")
    ap.add_argument("--versiegeln", metavar="UUID")
    ap.add_argument("--trocken", action="store_true")
    ap.add_argument("--pi", default="http://raspberrypi5lan:9984")
    args = ap.parse_args()
    if args.vorschlagen:
        return vorschlagen(args)
    if args.versiegeln:
        return versiegeln(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
