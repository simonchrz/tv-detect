#!/usr/bin/env python3
"""Alle Menschen-Labels VOR einer Konventions-Grenze auf zu spaete
Blockanfaenge vorbereiten.

Hintergrund: die Trailer-Konvention (§3y, 13.08.) wurde nur auf Aufnahmen
nachgezogen, fuer die gespeicherte BILD-Urteile vorlagen. Menschliche Labels
von davor folgen weiter der alten Regel — der Korpus misst damit gegen einen
gemischten Massstab (Ledger-Eintrag 2026-09-03).

Schneidet je Blockanfang Bilder von -24 s bis +4 s in 2-s-Schritten. Nicht
geblendet: die Frage hat nichts mit der OCR-Regel zu tun. ⚠️ Deshalb duerfen
die Ergebnisse NIE als O13-Referenz dienen.
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SNAPSHOT = Path("/tmp/tv-train-snapshot")
QUELLEN = Path.home() / ".cache/tv-detect-daemon/source"
BLIND = Path(__file__).with_name("blindkanten.py")
PY = "/Users/simon/ml/tv-classifier/.venv/bin/python"


def label_zeit(j):
    """Wann wurde entschieden? reviewed_at ODER auto_at_review_at — je
    nachdem, welches Feld die Review-Runde geschrieben hat. Nur eines zu
    lesen liess 2 von 5 Kandidaten ohne Datum dastehen."""
    werte = [v for v in (j.get("reviewed_at"), j.get("auto_at_review_at"))
             if isinstance(v, (int, float)) and v > 0]
    return max(werte) if werte else 0


def kandidaten(schnitt_ts, min_start=30.0):
    aus = []
    for d in sorted(SNAPSHOT.glob("_rec_*")):
        p = d / "ads_user.json"
        if not p.is_file():
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        # nur ECHTE Menschen-Labels
        if (j.get("auto_confirmed_at") or j.get("reviewed_by")
                or j.get("auto_confirmed_via_fingerprint")):
            continue
        ts = label_zeit(j)
        if not ts or ts >= schnitt_ts:
            continue
        u = d.name[5:]
        if not (QUELLEN / f"{u}.ts").exists():
            continue
        for a, _b in (j.get("ads") or []):
            if a > min_start:
                aus.append({"uuid": u, "kanal": u.rsplit("-", 1)[0][4:],
                            "start": float(a),
                            "label_datum": datetime.date.fromtimestamp(ts).isoformat()})
    return aus


def schneiden(a, ziel_wurzel):
    ziel = ziel_wurzel / ("e%04d" % a["nr"])
    if ziel.exists():
        shutil.rmtree(ziel)
    r = subprocess.run([PY, str(BLIND), a["uuid"], str(a["start"]),
                        "--ohne-blendung", "--vorlauf", "24", "--nachlauf", "4",
                        "--schritt", "2", "--ziel", str(ziel)],
                       capture_output=True, text=True)
    bilder = sorted(ziel.glob("*.png"))
    karte = {}
    for i, p in enumerate(bilder, 1):
        karte["bild%02d" % i] = float(p.stem[:-1])
        p.rename(ziel / ("bild%02d.png" % i))
    (ziel / "karte.json").write_text(json.dumps(karte, indent=1))
    return a["nr"], len(bilder), r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schnitt", default="2026-08-13")
    ap.add_argument("--ziel", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    ts = datetime.datetime.fromisoformat(args.schnitt).timestamp()
    kand = kandidaten(ts)
    for i, a in enumerate(kand, 1):
        a["nr"] = i
    args.ziel.mkdir(parents=True, exist_ok=True)
    (args.ziel / "kanten.json").write_text(json.dumps(kand, indent=1, ensure_ascii=False))
    print(f"{len(kand)} Blockanfaenge aus "
          f"{len({a['uuid'] for a in kand})} Aufnahmen, Schnitt {args.schnitt}",
          flush=True)

    fehl = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for nr, n, rc in ex.map(lambda a: schneiden(a, args.ziel), kand):
            if n != 15:
                fehl += 1
                print(f"  e{nr:04d}: nur {n} Bilder (rc={rc})", flush=True)
            if nr % 20 == 0:
                print(f"  … {nr}/{len(kand)}", flush=True)
    print(f"fertig. unvollstaendig: {fehl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
