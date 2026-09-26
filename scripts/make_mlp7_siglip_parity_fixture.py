#!/usr/bin/env python3
"""Paritaets-Fixture fuer den MLP7-Kopf mit SigLIP-Spalten.

Muster make_mlp6_ocr_parity_fixture.py: .bin ueber den ECHTEN
write_mlp_head_v7, Spalten ueber die ECHTE siglip_spalten.py, Spur als
float16-.npy wie siglip-spur.py sie schreibt, Erwartung aus der
Python-Vorwaertsrechnung. Der Go-Test laedt ueber den Produktions-Loader und
LadeSigLIPSpur.

Die Spur ist KUERZER als die Aufnahme (30 von 40 Sekunden): dahinter muessen
alle 65 Spalten 0 sein. SigLIP-Gewichte absichtlich gross (zahnloser Test
sonst). Deterministisch (Seed 47).
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


def main():
    th = lade("train_head", "train-head.py")
    sl = lade("siglip_spalten", "siglip_spalten.py")
    rng = np.random.RandomState(47)
    backbone, hidden, n, n_spur = 1280, 32, 40, 30
    in_dim = backbone + 1 + 1 + 3 + sl.N_SPALTEN
    W1 = rng.randn(in_dim, hidden).astype(np.float32) * 0.05
    W1[-sl.N_SPALTEN:, :] = rng.randn(sl.N_SPALTEN, hidden).astype(np.float32) * 0.5
    b1 = rng.randn(hidden).astype(np.float32) * 0.05
    W2 = rng.randn(hidden, 1).astype(np.float32) * 0.2
    b2 = rng.randn(1).astype(np.float32) * 0.05

    # Projektion aus einer Zufalls-"train"-Menge, wie im Nightly angepasst
    mu, V = sl.projektion_anpassen(rng.randn(500, sl.DIM).astype(np.float32) * 0.3 + 0.1)
    pfad = TESTDATA / "mlp7-siglip.bin"
    th.write_mlp_head_v7(pfad, KopfAttrappe(W1, b1, W2, b2), input_dim=in_dim,
                         hidden_dim=hidden, backbone_dim=backbone, n_logo=1, n_audio=1,
                         n_ocr=3, n_siglip=sl.N_SPALTEN, siglip_mu=mu, siglip_V=V)

    E16 = (rng.randn(n_spur, sl.DIM) * 0.3 + 0.1).astype(np.float16)
    np.save(TESTDATA / "mlp7-siglip-spur.npy", E16)
    embeds = rng.randn(n, backbone).astype(np.float32) * 0.5
    logo = rng.rand(n).astype(np.float32)
    rms = rng.rand(n).astype(np.float32)
    ocr = np.zeros((n, 3), np.float32)  # ohne OCR-Spur, wie ein Detect ohne --ocr-spur
    sig = sl.spalten(E16, n, mu, V)

    def vorwaerts(S):
        X = np.concatenate([embeds, logo[:, None], rms[:, None], ocr, S], axis=1)
        h = np.maximum(X.astype(np.float32) @ W1 + b1, 0).astype(np.float32)
        return 1.0 / (1.0 + np.exp(-(h @ W2 + b2).astype(np.float64).ravel()))

    p, p0 = vorwaerts(sig), vorwaerts(np.zeros_like(sig))
    assert np.abs(p - p0)[:n_spur].max() > 1e-2, "SigLIP-Gewichte zu klein, Test waere zahnlos"
    assert np.abs(p - p0)[n_spur:].max() == 0, "hinter der Spur muss alles 0 sein"
    (TESTDATA / "mlp7-siglip-parity.json").write_text(json.dumps({
        "kommentar": "Erzeugt von make_mlp7_siglip_parity_fixture.py (Seed 47) "
                     "— NICHT von Hand editieren.",
        "n": n, "n_spur": n_spur, "backbone": backbone,
        "embeds": [float(v) for v in embeds.ravel()],
        "logo": [float(v) for v in logo], "rms": [float(v) for v in rms],
        "erwartet": [round(float(v), 9) for v in p],
        "erwartet_ohne_spur": [round(float(v), 9) for v in p0],
        "spalten_t5": [float(v) for v in sig[5]],
    }) + "\n")
    print(f"{pfad} ({pfad.stat().st_size} B), max |p - p_ohne| = {np.abs(p - p0).max():.4f}")


if __name__ == "__main__":
    main()
