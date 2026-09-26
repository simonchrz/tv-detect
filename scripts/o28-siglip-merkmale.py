#!/usr/bin/env python3
"""O28-Vorarbeit — SigLIP-2-Bildmerkmale je Sekunde fuer Aufnahmen mit Quelle.

Laeuft in der EIGENEN Umgebung ~/ml/siglip-exp/.venv (transformers), nicht in
der Trainings-venv: neue Abhaengigkeiten sollen den Nightly nicht verschieben.

Zeilengenau wie das Training: ffmpeg `fps=1` auf der Quelle (dieselbe
Filterkette wie train-head.py extract_frames_via_ffmpeg), danach nur die
Pixel-Seitenverhaeltnis-Korrektur (SD-Aufnahmen sind anamorph; NaFlex
arbeitet im echten Seitenverhaeltnis — genau dafuer wurde es gewaehlt: es
liest Text). Ergebnis: ~/.cache/tvd-siglip2/<uuid>.npy, float16 (n, 768).
Zeilenzahl wird an die Archiv-Merkmale angeglichen (±2 toleriert, sonst
verworfen und gemeldet).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODELL = "google/siglip2-base-patch16-naflex"
C = Path.home() / ".cache"
ZIEL = C / "tvd-siglip2"
QUELLEN = C / "tv-detect-daemon/source"
ARCH = C / "tvd-train-archive"


def archiv_zeilen(uuid):
    f = ARCH / f"{uuid}.npz"
    if not f.is_file():
        return None
    try:
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        fp = m.get("feature_npy", "")
        return np.load(fp, mmap_mode="r").shape[0] if fp and Path(fp).exists() else None
    except Exception:
        return None


def bilder(src, breite):
    """Je Sekunde ein Bild, Seitenverhaeltnis korrigiert, feste Breite."""
    vf = f"fps=1,scale=iw*sar:ih,scale={breite}:-2"
    probe = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:v:0", "-vf", vf, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stderr=subprocess.DEVNULL)
    hoehe = len(probe) // (breite * 3)
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src),
                          "-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = breite * hoehe * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(hoehe, breite, 3)
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uuids", help="Datei mit uuids (je Zeile); sonst alle mit Quelle im Ledger")
    ap.add_argument("--max-patches", type=int, default=576)
    ap.add_argument("--breite", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODELL)
    model = AutoModel.from_pretrained(MODELL, torch_dtype=torch.float16).to("mps").eval()
    ZIEL.mkdir(parents=True, exist_ok=True)

    if a.uuids:
        uuids = [l.strip() for l in Path(a.uuids).read_text().splitlines() if l.strip()]
    else:
        led = json.loads((ARCH / "split-ledger.json").read_text())
        led = led.get("eimer", led)
        uuids = sorted(u for u, v in led.items() if v in ("train", "test"))
    uuids = [u for u in uuids if (QUELLEN / f"{u}.ts").is_file() and not (ZIEL / f"{u}.npy").is_file()]
    if a.limit:
        uuids = uuids[:a.limit]
    print(f"{len(uuids)} Aufnahmen zu rechnen", flush=True)

    for k, u in enumerate(uuids):
        soll = archiv_zeilen(u)
        if soll is None:
            print(f"  {u}: keine Archiv-Merkmale, uebersprungen", flush=True)
            continue
        t0 = time.time()
        embs, stapel = [], []

        def flush():
            b = proc(images=[Image.fromarray(x) for x in stapel], return_tensors="pt",
                     max_num_patches=a.max_patches)
            b = {kk: (v.to("mps", dtype=torch.float16) if v.dtype.is_floating_point else v.to("mps"))
                 for kk, v in b.items()}
            with torch.no_grad():
                e = model.get_image_features(**b)
            e = getattr(e, "pooler_output", e)
            embs.append(e.float().cpu().numpy())
            stapel.clear()

        for bild in bilder(QUELLEN / f"{u}.ts", a.breite):
            stapel.append(bild)
            if len(stapel) == 64:
                flush()
        if stapel:
            flush()
        E = np.concatenate(embs) if embs else np.zeros((0, 768), np.float32)
        if abs(len(E) - soll) > 2:
            print(f"  {u}: {len(E)} Bilder gegen {soll} Archiv-Zeilen — verworfen", flush=True)
            continue
        if len(E) < soll:
            E = np.concatenate([E, np.repeat(E[-1:], soll - len(E), 0)])
        E = E[:soll]
        np.save(ZIEL / f"{u}.npy", E.astype(np.float16))
        print(f"  {k+1}/{len(uuids)} {u}: {soll} s in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
