#!/usr/bin/env python3
"""Paritäts-Fixture für den nackten MLP1-Kopf (Architekturwechsel, L5).

Schreibt internal/signals/testdata/mlp1-bare.bin über den ECHTEN
`write_mlp_head_v1` aus train-head.py (nicht über eine Kopie des Formats —
sonst prüft die Fixture ihre eigene Abschrift) plus eine JSON mit
Eingaben und Python-seitig gerechneten Erwartungswerten. Der Go-Test lädt
die .bin über den Produktions-Loader und muss die Werte reproduzieren.

Deterministisch (Seed 42), damit die eingecheckte Fixture stabil ist.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TESTDATA = REPO / "internal/signals/testdata"


def lade_writer():
    spec = importlib.util.spec_from_file_location(
        "train_head", REPO / "scripts/train-head.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_mlp_head_v1


class KopfAttrappe:
    """Nur coefs_/intercepts_ — mehr liest der Writer nicht."""
    def __init__(self, W1, b1, W2, b2):
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, b2]


def main():
    rng = np.random.RandomState(42)
    backbone, hidden = 1280, 32
    in_dim = backbone + 1 + 1  # + logo + audio, KEIN Kanal-Block

    W1 = rng.randn(in_dim, hidden).astype(np.float32) * 0.05
    b1 = rng.randn(hidden).astype(np.float32) * 0.05
    W2 = rng.randn(hidden, 1).astype(np.float32) * 0.05
    b2 = rng.randn(1).astype(np.float32) * 0.05

    TESTDATA.mkdir(parents=True, exist_ok=True)
    write = lade_writer()
    pfad = TESTDATA / "mlp1-bare.bin"
    write(pfad, KopfAttrappe(W1, b1, W2, b2),
          input_dim=in_dim, hidden_dim=hidden, backbone_dim=backbone,
          n_logo=1, n_audio=1, n_channel=0)

    # Testeingaben: 4 Frames, darunter die Randfälle logo/rms 0 und 1.
    n = 4
    embeds = rng.randn(n, backbone).astype(np.float32) * 0.5
    logo = np.array([0.0, 1.0, 0.5, 0.83], dtype=np.float32)
    rms = np.array([0.5, 0.0, 1.0, 0.27], dtype=np.float32)

    # Erwartung: exakt die Rechnung aus dem Format-Kommentar —
    # x = [backbone, logo, audio], h = relu(xW1+b1), p = sigmoid(hW2+b2).
    # float32 durchgehend, wie der Go-Pfad rechnet.
    X = np.concatenate([embeds, logo[:, None], rms[:, None]], axis=1)
    h = np.maximum(X.astype(np.float32) @ W1 + b1, 0).astype(np.float32)
    logit = (h @ W2 + b2).astype(np.float32).ravel()
    p = (1.0 / (1.0 + np.exp(-logit.astype(np.float64))))

    aus = {
        "kommentar": "Erzeugt von make_mlp1_bare_parity_fixture.py (Seed 42) "
                     "— NICHT von Hand editieren.",
        "n": n, "backbone": backbone,
        # ⚠️ VOLLE Praezision: float(np.float32) ist der exakte Wert, und
        # json→float64→float32 ist damit verlustfrei. Mit %.6g gerundete
        # Eingaben liessen die Paritaet an der SERIALISIERUNG scheitern,
        # nicht am Loader — der Test prüfte dann das Falsche.
        "embeds": [float(v) for v in X[:, :backbone].ravel()],
        "logo": logo.tolist(), "rms": rms.tolist(),
        "erwartet": [round(float(v), 9) for v in p],
    }
    (TESTDATA / "mlp1-bare-parity.json").write_text(json.dumps(aus))
    print(f"Fixture: {pfad} ({pfad.stat().st_size} B)")
    print(f"Erwartete p: {[round(float(v), 6) for v in p]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
