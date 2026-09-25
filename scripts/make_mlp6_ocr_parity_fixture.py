#!/usr/bin/env python3
"""Paritaets-Fixture fuer den MLP6-Kopf mit OCR-Spalten (O26, L5).

Wie make_mlp1_bare_parity_fixture.py: die .bin entsteht ueber den ECHTEN
`write_mlp_head_v6` aus train-head.py, die OCR-Spalten ueber die ECHTE
scripts/ocr_spalten.py, die Erwartung aus der Python-Vorwaertsrechnung.
Der Go-Test laedt ueber den Produktions-Loader, setzt die Spur ueber
SetOCRSpalten und muss die Werte reproduzieren.

Nackter Kopf wie in Produktion (backbone + logo + audio) + 3 OCR-Spalten.
Die OCR-Gewichte sind absichtlich GROSS, damit ein vertauschter oder
fehlender OCR-Slot die Ausgabe sichtbar verschiebt — sonst bestuende der
Test auch mit Nullen.
Deterministisch (Seed 43).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TESTDATA = REPO / "internal/signals/testdata"


def lade(name, datei):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / datei)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class KopfAttrappe:
    def __init__(self, W1, b1, W2, b2):
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, b2]


SPUR = {
    "dauer_s": 30.0,
    "abgetastet": [{"von": 0, "bis": 30.0}],
    "funde": [{"time_s": 3, "hinweis": True, "werbemarker": False},
              {"time_s": 18, "hinweis": False, "werbemarker": True}],
}


def main():
    th = lade("train_head", "train-head.py")
    oc = lade("ocr_spalten", "ocr_spalten.py")
    rng = np.random.RandomState(43)
    backbone, hidden = 1280, 32
    in_dim = backbone + 1 + 1 + 3
    W1 = rng.randn(in_dim, hidden).astype(np.float32) * 0.05
    W1[-3:, :] = rng.randn(3, hidden).astype(np.float32) * 2.0  # OCR sichtbar
    b1 = rng.randn(hidden).astype(np.float32) * 0.05
    W2 = rng.randn(hidden, 1).astype(np.float32) * 0.05
    b2 = rng.randn(1).astype(np.float32) * 0.05

    pfad = TESTDATA / "mlp6-ocr.bin"
    th.write_mlp_head_v6(pfad, KopfAttrappe(W1, b1, W2, b2),
                         input_dim=in_dim, hidden_dim=hidden,
                         backbone_dim=backbone, n_logo=1, n_audio=1, n_ocr=3)

    # Sekunden 0..39 bei 1 fps: hinweis bei 0..13, werbung 8..28, spur_da
    # 0..29, dahinter (30..39) alles 0 — jede Spalte mal 0, mal 1.
    n = 40
    embeds = rng.randn(n, backbone).astype(np.float32) * 0.5
    logo = rng.rand(n).astype(np.float32)
    rms = rng.rand(n).astype(np.float32)
    ocr = oc.aus_spur(SPUR, n)
    X = np.concatenate([embeds, logo[:, None], rms[:, None], ocr], axis=1)
    h = np.maximum(X.astype(np.float32) @ W1 + b1, 0).astype(np.float32)
    logit = (h @ W2 + b2).astype(np.float32).ravel()
    p = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
    # Gegenprobe im Generator: OHNE OCR muss etwas anderes herauskommen.
    X0 = X.copy(); X0[:, -3:] = 0
    h0 = np.maximum(X0 @ W1 + b1, 0); p0 = 1 / (1 + np.exp(-((h0 @ W2 + b2).ravel())))
    assert np.abs(p - p0).max() > 1e-3, "OCR-Gewichte zu klein, Test waere zahnlos"

    (TESTDATA / "mlp6-ocr-parity.json").write_text(json.dumps({
        "kommentar": "Erzeugt von make_mlp6_ocr_parity_fixture.py (Seed 43) "
                     "— NICHT von Hand editieren.",
        "n": n, "backbone": backbone, "spur": SPUR,
        "embeds": [float(v) for v in embeds.ravel()],
        "logo": [float(v) for v in logo], "rms": [float(v) for v in rms],
        "erwartet": [round(float(v), 9) for v in p],
        "erwartet_ohne_ocr": [round(float(v), 9) for v in p0],
    }) + "\n")
    print(f"{pfad} ({pfad.stat().st_size} B), max |p - p_ohne_ocr| = "
          f"{np.abs(p - p0).max():.4f}")


if __name__ == "__main__":
    main()
