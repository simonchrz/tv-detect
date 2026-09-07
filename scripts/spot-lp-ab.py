#!/usr/bin/env python3
"""A/B für --spot-lp-w: Spot-Anker als Übergangs-Evidenz im HSMM.

Registriert in `docs/o19-spot-lp-preregistration.md`, geschrieben BEVOR
ein Lauf gerechnet wurde. Die Entscheidungsregel steht dort, nicht hier.

Das Modell ist eingefroren: die NN-Ausgaben stehen fest in den
Signal-Dumps, `--replay-signals` formt daraus nur die Blöcke neu.
Variiert wird ausschliesslich der Dekoder-Parameter. Deshalb ist die
train/test-Grenze fuer diese Frage nicht bindend — es fliesst kein Label
in Gewichte — und deshalb ist es auch keine Nachtserie.

Gemessen wird gegen das MENSCHENLABEL, nie gegen `auto`. Ein
maschinelles Label ist die frühere Ausgabe desselben Dekoders; dagegen zu
messen liesse jede Aenderung wie einen Rueckschritt aussehen.

  --stimmsatz   frei suchen (Quelle x Gewicht)
  --pruefsatz   EINE Konfiguration, einmal
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DUMPS = Path.home() / ".cache/tv-detect-daemon/emit-signals"
BILD = Path.home() / ".cache/tvd-wiederholung/korpus"
AUDIO = Path("/tmp/anchors")
ADS = Path("/tmp/ads")
BIN = Path.home() / ".local/bin/tv-detect"


def block_iou(pred, gt):
    """Definition aus eval_production_cutlists.py — dieselbe wie in
    train-head.py: je Wahrheitsblock die beste Ueberlappung, gemittelt."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tot = 0.0
    for gs, ge in gt:
        best = 0.0
        for ps, pe in pred:
            inter = max(0.0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            if union > 0:
                best = max(best, inter / union)
        tot += best
    return tot / len(gt)


def _bl(x):
    return [(float(a), float(b)) for a, b in (x or []) if float(b) > float(a)]


def menschlabel(uuid):
    p = ADS / f"{uuid}.json"
    if not p.is_file():
        return None
    return _bl(json.loads(p.read_text()).get("user"))


def anker_datei(uuid, quelle, tmp):
    """Fuer 'beide' werden die Intervalle zusammengelegt. family_id bleibt
    erhalten; SpotBoundaryLP liest es ohnehin nicht, aber eine erfundene
    id waere eine stille Luege in der Datei."""
    if quelle == "audio":
        p = AUDIO / f"{uuid}.json"
        return p if p.is_file() else None
    if quelle == "bild":
        p = BILD / f"{uuid}.json"
        return p if p.is_file() else None
    a = AUDIO / f"{uuid}.json"
    b = BILD / f"{uuid}.json"
    ank = []
    for p in (a, b):
        if p.is_file():
            ank += json.loads(p.read_text()).get("anchored") or []
    if not ank:
        return None
    ziel = tmp / f"{uuid}-beide.json"
    ziel.write_text(json.dumps({"uuid": uuid, "anchored": ank}))
    return ziel


def lauf(uuid, quelle, w, tmp):
    cmd = [str(BIN), "--replay-signals", str(DUMPS / f"{uuid}.json"),
           "--decoder", "hsmm", "--hsmm-dur-w", "15"]
    if w > 0:
        p = anker_datei(uuid, quelle, tmp)
        if p is None:
            return None
        cmd += ["--spot-anchors", str(p), "--spot-lp-w", str(w)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Fehlerausgabe vom ENDE kuerzen: die Ursache steht dort, nicht
        # in den ersten 300 Zeichen (Memory fehlerausgabe_vom_ende_kuerzen).
        print(f"    {uuid}: rc={r.returncode} {r.stderr[-300:]}", file=sys.stderr)
        return None
    try:
        return _bl(json.loads(r.stdout).get("blocks"))
    except Exception as e:
        print(f"    {uuid}: Ausgabe unlesbar ({e})", file=sys.stderr)
        return None


def kanten_wanderung(basis, neu):
    """Zaehlt, wie viele Kanten nach AUSSEN und wie viele nach INNEN
    wandern. Nebenbeobachtung aus der Registrierung: spot_lp.go behauptet,
    der Block werde nur ausgedehnt."""
    aus = ein = 0
    for (bs, be) in basis:
        kand = [(ns, ne) for ns, ne in neu if ne > bs and ns < be]
        if len(kand) != 1:
            continue
        ns, ne = kand[0]
        for alt, jetzt, vorzeichen in ((bs, ns, -1), (be, ne, +1)):
            d = (jetzt - alt) * vorzeichen
            if d > 0.5:
                aus += 1
            elif d < -0.5:
                ein += 1
    return aus, ein


def satz(uuids, halbe):
    return [u for u in uuids
            if (int(hashlib.sha1(u.encode()).hexdigest(), 16) % 2) == halbe]


def messe(uuids, quelle, w, tmp, basis_cache):
    zeilen = []
    aus = ein = 0
    for u in uuids:
        gt = menschlabel(u)
        if not gt:
            continue
        if u not in basis_cache:
            basis_cache[u] = lauf(u, quelle, 0, tmp)
        b0 = basis_cache[u]
        b1 = lauf(u, quelle, w, tmp)
        if b0 is None or b1 is None:
            continue
        zeilen.append((u, block_iou(b0, gt), block_iou(b1, gt)))
        a_, e_ = kanten_wanderung(b0, b1)
        aus += a_
        ein += e_
    if not zeilen:
        return None
    d = np.array([z[2] - z[1] for z in zeilen])
    return {"n": len(zeilen), "median": float(np.median(d)), "mittel": float(d.mean()),
            "besser": int((d > 0.001).sum()), "schlechter": int((d < -0.001).sum()),
            "groesster_verlust": float(-d.min()) if len(d) else 0.0,
            "kanten_aus": aus, "kanten_ein": ein,
            "zeilen": [(z[0], round(z[1], 4), round(z[2], 4)) for z in zeilen]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kandidaten", default="/tmp/ab_kandidaten.json")
    ap.add_argument("--stimmsatz", action="store_true")
    ap.add_argument("--pruefsatz", action="store_true")
    ap.add_argument("--quelle", choices=["audio", "bild", "beide"])
    ap.add_argument("--gewicht", type=float)
    ap.add_argument("--json", help="Ergebnis hierhin schreiben")
    a = ap.parse_args()

    uuids = json.loads(Path(a.kandidaten).read_text())
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "spot-lp-ab"
    tmp.mkdir(parents=True, exist_ok=True)

    if a.stimmsatz:
        us = satz(uuids, 0)
        print(f"=== Stimmsatz: {len(us)} Aufnahmen (frei durchsuchbar) ===")
        erg = {}
        for quelle in ("audio", "bild", "beide"):
            cache = {}
            for w in (0.5, 1.0, 2.0, 4.0, 8.0):
                r = messe(us, quelle, w, tmp, cache)
                if not r:
                    continue
                erg[f"{quelle}-{w}"] = r
                print(f"  {quelle:<6} w={w:<4} n={r['n']:<3} Median {r['median']:+.4f}  "
                      f"besser {r['besser']:<3} schlechter {r['schlechter']:<3} "
                      f"groesster Verlust {r['groesster_verlust']:.3f}  "
                      f"Kanten aussen/innen {r['kanten_aus']}/{r['kanten_ein']}", flush=True)
        if erg:
            best = max(erg.items(), key=lambda kv: (kv[1]["median"], -float(kv[0].split("-")[1])))
            print(f"\n  bester Stimmsatz-Arm: {best[0]}  Median {best[1]['median']:+.4f}")
        if a.json:
            Path(a.json).write_text(json.dumps(erg, indent=1))
        return 0

    if a.pruefsatz:
        if not (a.quelle and a.gewicht):
            print("Pruefsatz braucht --quelle und --gewicht (genau EINE Konfiguration)")
            return 1
        us = satz(uuids, 1)
        print(f"=== Pruefsatz: {len(us)} Aufnahmen, EINE Konfiguration: "
              f"{a.quelle} w={a.gewicht} ===")
        r = messe(us, a.quelle, a.gewicht, tmp, {})
        if not r:
            print("keine messbaren Aufnahmen"); return 1
        print(f"  n={r['n']}  Median {r['median']:+.4f}  Mittel {r['mittel']:+.4f}")
        print(f"  besser {r['besser']}, schlechter {r['schlechter']}, "
              f"groesster Einzelverlust {r['groesster_verlust']:.3f}")
        print(f"  Kanten nach aussen {r['kanten_aus']}, nach innen {r['kanten_ein']}")
        b1 = r["median"] >= 0.005
        b2 = r["besser"] >= 2 * max(r["schlechter"], 1) if r["schlechter"] else r["besser"] > 0
        b3 = r["groesster_verlust"] <= 0.10
        print(f"\n  Median >= +0.005            {'JA' if b1 else 'NEIN'}")
        print(f"  besser >= 2x schlechter     {'JA' if b2 else 'NEIN'}")
        print(f"  kein Verlust > 0.10         {'JA' if b3 else 'NEIN'}")
        print(f"\n  ==> O19 {'ERFUELLT' if (b1 and b2 and b3) else 'NICHT ERFUELLT'}")
        if a.json:
            Path(a.json).write_text(json.dumps(r, indent=1))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
