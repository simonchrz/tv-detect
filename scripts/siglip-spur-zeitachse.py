#!/usr/bin/env python3
"""Zeitachsen-Pruefung der SigLIP-Spur (Entwurf siglip-spur-design.md, Schritt 1).

Referenz = Einzelbild per `ffmpeg -ss t` (die absolute Verankerung, die auch
die OCR-Spur nutzt). Verglichen wird Zeile t von
  am_stueck  dem O28-Cache (~/.cache/tvd-siglip2, fps=1 ueber die ganze Datei)
  gekachelt  dem neuen Erzeuger (scripts/siglip-spur.py)
je als Kosinus zur Referenz, dazu die beste Verschiebung in -3..3 s. Liegt
am_stueck spuerbar unter gekachelt oder verschoben, war der O28/O29-Cache
schief und muss neu gerechnet werden.

Laeuft in ~/ml/siglip-exp/.venv. Schreibt nur nach --tmp.
"""
import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
C = Path.home() / ".cache"
spec = importlib.util.spec_from_file_location("siglip_spur", HIER / "siglip-spur.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def einzelbild(quelle, t, hoehe):
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(quelle), "-map", "0:v:0",
                        "-vf", sp.VF.replace("fps=1,", ""), "-frames:v", "1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(hoehe, sp.BREITE, 3) if len(b) == hoehe * sp.BREITE * 3 else None


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="Aufnahmen")
    ap.add_argument("--punkte", type=int, default=25, help="Stichproben je Aufnahme")
    ap.add_argument("--tmp", default="/tmp/siglip-zeitachse")
    ap.add_argument("--json")
    a = ap.parse_args()
    random.seed(3)
    alt = sorted(p.stem for p in (C / "tvd-siglip2").glob("*.npy"))
    quellen = [u for u in alt if (C / "tv-detect-daemon/source" / f"{u}.ts").is_file()]
    # lange Aufnahmen bevorzugt: dort waere Drift am groessten
    quellen.sort(key=lambda u: -np.load(C / "tvd-siglip2" / f"{u}.npy", mmap_mode="r").shape[0])
    wahl = quellen[:a.n // 2] + random.sample(quellen[a.n // 2:], a.n - a.n // 2)
    enc = sp.Encoder()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    erg = []
    for u in wahl:
        q = C / "tv-detect-daemon/source" / f"{u}.ts"
        A = np.load(C / "tvd-siglip2" / f"{u}.npy").astype(np.float32)
        K, dauer, _ = sp.spur(q, enc)
        hoehe = sp._hoehe(q, 0.0)
        ts = sorted(random.sample(range(5, min(len(A), len(K)) - 5), min(a.punkte, min(len(A), len(K)) - 10)))
        R = []
        for t in ts:
            b = einzelbild(q, t, hoehe)
            R.append(enc([b])[0] if b is not None else None)
        zeile = {"uuid": u, "dauer": round(dauer), "zeilen_alt": len(A), "zeilen_neu": len(K)}
        for name, M in (("am_stueck", A), ("gekachelt", K)):
            best = {}
            for s in range(-3, 4):
                v = [cos(M[t + s], r) for t, r in zip(ts, R) if r is not None and 0 <= t + s < len(M)]
                best[s] = float(np.mean(v))
            zeile[name] = {"cos0": round(best[0], 4), "beste_versch": max(best, key=best.get),
                           "cos_beste": round(max(best.values()), 4)}
        # Drift: Verschiebung erste vs letzte Viertel
        erg.append(zeile)
        print(json.dumps(zeile), flush=True)
    for name in ("am_stueck", "gekachelt"):
        c0 = [z[name]["cos0"] for z in erg]
        vs = [z[name]["beste_versch"] for z in erg]
        print(f"{name}: Kosinus zur Referenz Median {np.median(c0):.4f} (min {min(c0):.4f}); "
              f"beste Verschiebung != 0 in {sum(v != 0 for v in vs)}/{len(vs)}")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
