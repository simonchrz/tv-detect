#!/usr/bin/env python3
"""Goldwerte fuer die OCR-Spalten-Paritaet Python ↔ Go.

Schreibt internal/signals/testdata/ocr_paritaet.json: eine absichtlich
gemeine Spur (Treffer am Rand, ueberlappende Fenster, Bruchteil-Sekunden,
beide Marker zugleich, Luecke zwischen abgetasteten Kacheln) und die
Spalten, die scripts/ocr_spalten.py daraus macht. Der Go-Test
(ocrspalten_test.go) muss sie bitgleich treffen.

    python3 scripts/gen-ocr-paritaet.py
"""
import importlib.util
import json
from pathlib import Path

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ocr_spalten", HIER / "ocr_spalten.py")
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)

SPUR = {
    "dauer_s": 247.6,
    "abgetastet": [{"von": 0, "bis": 180}, {"von": 190.5, "bis": 247.6}],
    "fehlgeschlagen": [{"von": 180, "bis": 190.5}],
    "funde": [
        {"time_s": 0, "hinweis": True, "werbemarker": False},
        {"time_s": 4.9, "hinweis": False, "werbemarker": True},
        {"time_s": 100, "hinweis": True, "werbemarker": True},
        {"time_s": 112, "hinweis": True, "werbemarker": False},
        {"time_s": 185.2, "hinweis": False, "werbemarker": True},
        {"time_s": 246, "hinweis": True, "werbemarker": False},
        {"time_s": 400, "hinweis": True, "werbemarker": True},
    ],
}
N_SEK = 250

spalten = oc.aus_spur(SPUR, N_SEK)
ziel = HIER.parent / "internal/signals/testdata/ocr_paritaet.json"
ziel.write_text(json.dumps({
    "fenster": oc.FENSTER, "n_sek": N_SEK, "spur": SPUR,
    "erwartet": spalten.tolist()}, indent=None) + "\n")
print(f"{ziel}: {N_SEK} Sekunden, hinweis {int(spalten[:, 0].sum())}, "
      f"werbung {int(spalten[:, 1].sum())}, spur_da {int(spalten[:, 2].sum())}")
