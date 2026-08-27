#!/usr/bin/env python3
"""Paritäts-Fixture für den ENSEMBLE-Kopf (Seed-Ensemble, 2026-08-27).

Der Nightly liefert seit 2026-08-27 nicht mehr einen ausgewählten Seed aus,
sondern k Seeds zusammengelegt zu EINEM Kopf, der ihren Logit-Mittelwert
rechnet (`merge_mlp_ensemble`). Das ist bewusst KEIN neues Dateiformat: es
ist ein ganz normaler MLP1-v1-Kopf mit hidden_dim = k*32.

Genau das prüft diese Fixture — und zwar an der einzigen Stelle, an der es
schiefgehen kann, ohne dass jemand es merkt:

  1. rechnet `merge_mlp_ensemble` wirklich den Mittelwert der Einzel-Logits?
     Die Erwartungswerte hier werden aus den DREI EINZELKÖPFEN gerechnet,
     nicht aus dem zusammengelegten — sonst prüfte die Fixture ihre eigene
     Abschrift.
  2. akzeptiert `write_mlp_head_v1` die breitere Schicht?
  3. liest die Go-Seite hidden_dim wirklich aus dem Header, statt 32
     anzunehmen? Der Produktions-Loader im Go-Test beantwortet das.

Deterministisch (Seed 4242), damit die eingecheckte Fixture stabil ist.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TESTDATA = REPO / "internal/signals/testdata"
SEEDS = 3


def lade_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_head", REPO / "scripts/train-head.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class KopfAttrappe:
    """Nur coefs_/intercepts_ — mehr liest Writer wie Merge nicht."""
    def __init__(self, W1, b1, W2, b2):
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, b2]


def main():
    th = lade_trainer()
    rng = np.random.RandomState(4242)
    backbone, hidden = 1280, 32
    in_dim = backbone + 1 + 1  # + logo + audio, KEIN Kanal-Block

    koepfe = []
    for _ in range(SEEDS):
        koepfe.append(KopfAttrappe(
            rng.randn(in_dim, hidden).astype(np.float32) * 0.05,
            rng.randn(hidden).astype(np.float32) * 0.05,
            rng.randn(hidden, 1).astype(np.float32) * 0.05,
            rng.randn(1).astype(np.float32) * 0.05))

    merged = th.merge_mlp_ensemble(koepfe)
    hd = int(np.asarray(merged.coefs_[0]).shape[1])
    assert hd == SEEDS * hidden, f"hidden {hd} != {SEEDS}*{hidden}"

    TESTDATA.mkdir(parents=True, exist_ok=True)
    pfad = TESTDATA / "mlp1-ensemble.bin"
    th.write_mlp_head_v1(pfad, merged,
                         input_dim=in_dim, hidden_dim=hd,
                         backbone_dim=backbone,
                         n_logo=1, n_audio=1, n_channel=0)

    # Testeingaben: 4 Frames, darunter die Randfälle logo/rms 0 und 1.
    n = 4
    embeds = rng.randn(n, backbone).astype(np.float32) * 0.5
    logo = np.array([0.0, 1.0, 0.5, 0.83], dtype=np.float32)
    rms = np.array([0.5, 0.0, 1.0, 0.27], dtype=np.float32)
    X = np.concatenate([embeds, logo[:, None], rms[:, None]],
                       axis=1).astype(np.float32)

    # ⚠️ Erwartung aus den EINZELKÖPFEN — Mittelwert der Logits, DANN
    # sigmoid. Nicht aus `merged`, sonst prüft der Test die Zusammenlegung
    # gegen sich selbst.
    logits = []
    for k in koepfe:
        W1, W2 = k.coefs_
        b1, b2 = k.intercepts_
        h = np.maximum(X @ W1 + b1, 0).astype(np.float32)
        logits.append((h @ W2 + b2).astype(np.float32).ravel())
    logit = np.mean(np.asarray(logits, dtype=np.float64), axis=0)
    p = 1.0 / (1.0 + np.exp(-logit))

    aus = {
        "kommentar": "Erzeugt von make_mlp1_ensemble_parity_fixture.py "
                     "(Seed 4242) — NICHT von Hand editieren. Erwartung "
                     "stammt aus den Einzelköpfen, nicht aus dem Merge.",
        "n": n, "backbone": backbone, "seeds": SEEDS, "hidden": hd,
        "embeds": [float(v) for v in X[:, :backbone].ravel()],
        "logo": logo.tolist(), "rms": rms.tolist(),
        "erwartet": [round(float(v), 9) for v in p],
    }
    (TESTDATA / "mlp1-ensemble-parity.json").write_text(json.dumps(aus))
    print(f"Fixture: {pfad} ({pfad.stat().st_size} B, hidden {hd})")
    print(f"Erwartete p: {[round(float(v), 6) for v in p]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
