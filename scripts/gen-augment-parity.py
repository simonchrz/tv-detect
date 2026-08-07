#!/usr/bin/env python3
"""Goldwert-Vektoren fuer den Go-Paritaetstest des Temporal-Blocks.

Der Temporal-Block (dp, dn, Unruhe) entsteht zweimal: in train-head.py
`zusatzspalten` fuer das Training, in internal/signals/nn.go
`temporalSpalten` fuer den Betrieb. Zusammenlegen geht nicht — die eine
Seite rechnet in numpy auf Sekundenzeilen, die andere in Go auf Frames —
also wird die Naht wie beim hsmm-Decoder mit Goldwerten festgenagelt.

⚠️ Warum das noetig ist. Laufen die beiden auseinander, sieht der Kopf im
Betrieb eine andere Eingabe als im Training. Das erzeugt KEINEN Fehler,
sondern ein leicht schlechteres Modell — und das faellt gegen die
Nacht-zu-Nacht-Streuung der IoU-Zahlen nicht auf. Genau diese Sorte Bruch
hat hier schon mehrfach Tage gekostet.

Zwei Faelle werden erzeugt:

  step=1   Go rechnet je Sekunde, wie Python. Direkter Vergleich.
  step>1   Go rechnet je FRAME (Produktion: 25 fps), Python kennt das
           nicht. Trotzdem exakt vergleichbar, wenn alle Frames einer
           Sekunde denselben Vektor tragen: dann muss Go genau die
           per-Sekunde-Werte liefern, jeden step-mal wiederholt. Das
           prueft die Schrittweite im Fenster (j = i + k*step) — die
           Stelle, an der ein Port am ehesten danebengreift.

Ausfuehren: python3 scripts/gen-augment-parity.py
Schreibt:   internal/signals/testdata/augment-parity.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("th", str(HIER / "train-head.py"))
th = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["train-head"]
try:
    _spec.loader.exec_module(th)
except SystemExit:
    pass
finally:
    sys.argv = _argv

# Whisper/Minute-Prior spielen hier keine Rolle — geprueft wird der
# Temporal-Block, der ausschliesslich aus X entsteht.
th._load_whisper_per_sec = lambda uuid, n: np.zeros(n, np.float32)
th._whisper_present = lambda uuid: False


def spalten_python(X):
    """dp, dn, Unruhe aus train-head.py — die Trainingsseite."""
    z = th.zusatzspalten(X, "u", "kein-kanal", {}, 0,
                         kanal=False, temporal=True, churn=True)
    return z[:, 0], z[:, 1], z[:, 2]


def fall(name, X, step, beschreibung):
    dp, dn, ch = spalten_python(X)
    T, dim = X.shape
    if step == 1:
        base = X
    else:
        # jeden Sekundenvektor step-mal wiederholen = "Frames innerhalb
        # einer Sekunde sind identisch"
        base = np.repeat(X, step, axis=0)
        dp = np.repeat(dp, step)
        dn = np.repeat(dn, step)
        ch = np.repeat(ch, step)
    return {
        "name": name,
        "beschreibung": beschreibung,
        "n": int(base.shape[0]),
        "baseDim": int(dim),
        "step": int(step),
        "nTemporal": 3,
        "base": [float(v) for v in base.reshape(-1)],
        "dp": [float(v) for v in dp],
        "dn": [float(v) for v in dn],
        "churn": [float(v) for v in ch],
    }


def main():
    rng = np.random.default_rng(20260807)
    faelle = []

    # --- entartete Laengen: hier ist die erste Fassung schon gestorben ----
    faelle.append(fall("eine-sekunde", np.ones((1, 4), np.float32), 1,
                       "kein Vorgaenger, kein Nachfolger — alles 0"))
    faelle.append(fall("zwei-sekunden",
                       np.array([[0, 0], [3, 4]], np.float32), 1,
                       "genau ein Abstand (3-4-5-Dreieck: dp[1]=5)"))
    faelle.append(fall("kuerzer-als-fenster",
                       np.cumsum(np.ones((7, 3), np.float32), axis=0), 1,
                       "7 Sekunden bei 61er Fenster — der Fall, an dem "
                       "np.convolve(mode='same') 61 Werte lieferte"))

    # --- Fenstergrenzen ---------------------------------------------------
    for T in (60, 61, 62):
        faelle.append(fall(f"fensterbreite-{T}",
                           rng.standard_normal((T, 5)).astype(np.float32), 1,
                           "rund um die Fensterbreite — Randnormierung"))

    # --- der gewoehnliche Fall -------------------------------------------
    faelle.append(fall("rauschen-200",
                       rng.standard_normal((200, 8)).astype(np.float32), 1,
                       "200 Sekunden Rauschen, der Normalfall"))

    # --- konstante Eingabe: alle Abstaende exakt 0 ------------------------
    faelle.append(fall("konstant", np.full((80, 6), 2.5, np.float32), 1,
                       "keine Bewegung — dp/dn/Unruhe muessen exakt 0 sein, "
                       "nicht knapp daneben"))

    # --- Sprung: eine einzelne Szenenwechsel-Spitze ------------------------
    X = np.zeros((120, 4), np.float32)
    X[60:] = 10.0
    faelle.append(fall("einzelner-sprung", X, 1,
                       "ein Sprung bei Sekunde 60 — die Unruhe muss ihn "
                       "ueber das Fenster verschmieren, nicht verschieben"))

    # --- Frame-Rate: der eigentliche Grund fuer diesen Test ---------------
    for step in (2, 25):
        faelle.append(fall(f"framerate-{step}",
                           rng.standard_normal((40, 4)).astype(np.float32),
                           step,
                           f"{step} Frames je Sekunde, innerhalb einer "
                           f"Sekunde identisch — prueft j = i + k*step"))

    ziel = HIER.parent / "internal/signals/testdata/augment-parity.json"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(
        {"churnWindowS": int(
            __import__("inspect").signature(th._churn_col)
            .parameters["fenster"].default),
         "faelle": faelle}, indent=1))
    werte = sum(len(f["dp"]) for f in faelle)
    print(f"{len(faelle)} Faelle, {werte} Werte je Spalte → {ziel}")


if __name__ == "__main__":
    main()
