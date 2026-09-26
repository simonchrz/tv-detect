#!/usr/bin/env python3
"""siglip-spur — SigLIP-2-Bildmerkmale je Sekunde ueber die GANZE Aufnahme.

Entwurf docs/siglip-spur-design.md, Schritt 1. EIN Erzeuger fuer Training
und Detect: Train/Serve-Paritaet durch Bauart (Lehre O27).

Laeuft in der eigenen Umgebung ~/ml/siglip-exp/.venv (torch, transformers),
nicht in der Trainings-venv.

Zeitachse wie tv-ocr-spur: gekachelt in 2*halb Sekunden (Vorgabe 180 s), jede
Kachel mit eigenem -ss. Zeile i ist das Bild bei Sekunde i ab Dateianfang.
Am Stueck gerechnet driftet `fps=1` bei .ts (Memory
frames_tragen_erwartete_zeit); je Kachel neu verankert bleibt der Fehler so
klein wie in der OCR-Spur.

Ausgabe: <aus>.npy (float16, n x 768) + <aus>.json (Frische-Beilage:
quelle_bytes/mtime, Modell, Kachelung, Laufzeit). Beide werden atomar
geschrieben, die JSON ZULETZT: ohne JSON gilt eine Spur als nicht vorhanden.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

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
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{von:.3f}", "-i", str(quelle),
                        "-map", "0:v:0", "-vf", VF, "-frames:v", "1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return len(b) // (BREITE * 3)


def bilder_kachel(quelle, von, bis, hoehe):
    """Bilder bei von, von+1, … (< bis). Genau ceil(bis-von) Stueck oder weniger."""
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                          "-ss", f"{von:.3f}", "-t", f"{bis - von:.3f}", "-i", str(quelle),
                          "-map", "0:v:0", "-vf", VF, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = BREITE * hoehe * 3
    soll = math.ceil(bis - von - 1e-6)
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


def spur(quelle, enc, halb=90.0):
    """(n, 768) float32, Zeile i = Sekunde i. Fehlende Bilder am Kachelende
    werden mit dem letzten Bild der Kachel aufgefuellt; ganz leere Kacheln
    bleiben 0 und stehen in `leer`."""
    dauer = dauer_von(quelle)
    n = math.ceil(dauer - 1e-6)
    E = np.zeros((n, 768), np.float32)
    leer = []
    hoehe = _hoehe(quelle, 0.0)
    for von, bis in kacheln(dauer, halb):
        i0 = int(round(von))
        zeilen, stapel = [], []
        for bild in bilder_kachel(quelle, von, bis, hoehe):
            stapel.append(bild)
            if len(stapel) == STAPEL:
                zeilen.append(enc(stapel)); stapel = []
        if stapel:
            zeilen.append(enc(stapel))
        if not zeilen:
            leer.append([von, bis]); continue
        Z = np.concatenate(zeilen)
        soll = min(math.ceil(bis - von - 1e-6), n - i0)
        if len(Z) < soll:
            Z = np.concatenate([Z, np.repeat(Z[-1:], soll - len(Z), 0)])
        E[i0:i0 + soll] = Z[:soll]
    return E, dauer, leer


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
            "breite": BREITE, "halb_s": halb, "leer": leer, "gekachelt": True,
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
