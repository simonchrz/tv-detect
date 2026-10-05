#!/usr/bin/env python3
"""siglip-spur — SigLIP-2-Bildmerkmale je Sekunde ueber die GANZE Aufnahme.

Entwurf docs/siglip-spur-design.md, Schritt 1. EIN Erzeuger fuer Training
und Detect: Train/Serve-Paritaet durch Bauart (Lehre O27).

Laeuft in der eigenen Umgebung ~/ml/siglip-exp/.venv (torch, transformers),
nicht in der Trainings-venv.

⚠️ ZEILEN = ZEILEN DES KOPFS, NICHT ABSOLUTE SEKUNDEN. Ein `fps=1`-Durchlauf
ueber die ganze Datei, genau wie die Backbone-Merkmale des Kopfs entstehen.
Gemessen 2026-09-26 (scripts/siglip-spur-zeilenbezug.py, 20 Aufnahmen,
Szenenschnitt-Korrelation gegen die Kopf-Zeilen): am Stueck Lag 0 in 20/20
(r 0.59–0.86, vorne wie hinten); gekachelt mit eigenem -ss je 180-s-Kachel
(wie tv-ocr-spur) Lag +1 in 15/20 und bei einem PTS-Sprung (14056 Kopf-
Zeilen fuer 12570 s) voellig daneben (r ~ 0). Nicht wieder kacheln.

Ausgabe: <aus>.npy (float16, n x 768) + <aus>.json (Frische-Beilage:
quelle_bytes/mtime, Modell, Laufzeit). Beide werden atomar geschrieben, die
JSON ZULETZT: ohne JSON gilt eine Spur als nicht vorhanden.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "daemon"))
from ss_versatz import versatz  # noqa: E402  (-ss ab Video-, nicht Container-Beginn)

import numpy as np

MODELL = "google/siglip2-base-patch16-naflex"
MAX_PATCHES = 576
BREITE = 640
STAPEL = 64
VF = f"fps=1,scale=iw*sar:ih,scale={BREITE}:-2"


def kacheln(dauer, halb=90.0):
    """[(von, bis)] wie tv-ocr-spur kacheln()."""
    out, von = [], 0.0
    while von < dauer:
        out.append((von, min(von + 2 * halb, dauer)))
        von += 2 * halb
    return out


def dauer_von(quelle):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(quelle)], capture_output=True, text=True)
    return float(r.stdout.strip())


def _hoehe(quelle, von):
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{von + versatz(quelle):.3f}", "-i", str(quelle),
                        "-map", "0:v:0", "-vf", VF, "-frames:v", "1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return len(b) // (BREITE * 3)


def bilder_kachel(quelle, von, bis, hoehe):
    """Bilder bei von, von+1, … (< bis). Genau ceil(bis-von) Stueck oder weniger."""
    fenster = [] if bis == float("inf") else ["-ss", f"{von + versatz(quelle):.3f}", "-t", f"{bis - von:.3f}"]
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                          *fenster, "-i", str(quelle),
                          "-map", "0:v:0", "-vf", VF, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = BREITE * hoehe * 3
    soll = math.inf if bis == float("inf") else math.ceil(bis - von - 1e-6)
    k = 0
    while k < soll:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        k += 1
        yield np.frombuffer(b, np.uint8).reshape(hoehe, BREITE, 3)
    p.stdout.close()
    p.wait()


class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.gerät = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(MODELL)
        self.model = AutoModel.from_pretrained(MODELL, torch_dtype=torch.float16).to(self.gerät).eval()

    def __call__(self, bilder):
        from PIL import Image
        t = self.torch
        b = self.proc(images=[Image.fromarray(x) for x in bilder], return_tensors="pt",
                      max_num_patches=MAX_PATCHES)
        b = {k: (v.to(self.gerät, dtype=t.float16) if v.dtype.is_floating_point else v.to(self.gerät))
             for k, v in b.items()}
        with t.no_grad():
            e = self.model.get_image_features(**b)
        e = getattr(e, "pooler_output", e)
        return e.float().cpu().numpy()


def spur(quelle, enc, halb=None):
    """(n, 768) float32 — ein fps=1-Durchlauf ueber die ganze Datei, Zeile i =
    i-tes Ausgabebild (= Zeile i des Kopfs). `halb` bleibt nur fuer die alte
    Aufrufform; die Kachelung ist gemessen falsch (s. Modul-Doku)."""
    dauer = dauer_von(quelle)
    hoehe = _hoehe(quelle, 0.0)
    zeilen, stapel = [], []
    for bild in bilder_kachel(quelle, 0.0, float("inf"), hoehe):
        stapel.append(bild)
        if len(stapel) == STAPEL:
            zeilen.append(enc(stapel)); stapel = []
    if stapel:
        zeilen.append(enc(stapel))
    E = np.concatenate(zeilen) if zeilen else np.zeros((0, 768), np.float32)
    return E, dauer, ([] if len(E) else [[0.0, dauer]])


def _atomar(pfad, schreiben):
    tmp = pfad.with_name(pfad.name + ".tmp")
    schreiben(tmp)
    os.replace(tmp, pfad)


def schreibe_spur(quelle, aus, enc, halb=90.0):
    """Rechnet und schreibt <aus>.npy + <aus>.json (JSON zuletzt). Gibt die
    Beilage zurueck. Von main() und der Kampagne genutzt."""
    quelle, aus = Path(quelle), Path(aus)
    st = quelle.stat()
    t0 = time.time()
    E, dauer, leer = spur(quelle, enc, halb)
    aus.parent.mkdir(parents=True, exist_ok=True)
    def npy(p):
        with open(p, "wb") as f:
            np.save(f, E.astype(np.float16))
    _atomar(aus.with_suffix(".npy"), npy)
    meta = {"quelle": str(quelle), "quelle_bytes": st.st_size, "quelle_mtime": int(st.st_mtime),
            "dauer_s": dauer, "zeilen": len(E), "modell": MODELL, "max_patches": MAX_PATCHES,
            "breite": BREITE, "leer": leer, "gekachelt": False,
            "erstellt": time.strftime("%Y-%m-%dT%H:%M:%S"), "laufzeit_s": round(time.time() - t0, 1)}
    _atomar(aus.with_suffix(".json"), lambda p: Path(p).write_text(json.dumps(meta)))
    print(f"  {aus.name}: {len(E)} Zeilen in {meta['laufzeit_s']:.0f}s"
          + (f", {len(leer)} leere Kacheln" if leer else ""), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quelle", required=True)
    ap.add_argument("--aus", required=True, help="Ziel ohne Endung (…/<uuid>)")
    ap.add_argument("--halb", type=float, default=90.0)
    a = ap.parse_args()
    schreibe_spur(a.quelle, a.aus, Encoder(), a.halb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
