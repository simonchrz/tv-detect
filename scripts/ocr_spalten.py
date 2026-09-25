"""OCR-Spalten — die EINE Python-Definition (O26, ERFUELLT 2026-09-25).

Drei Spalten je Sekunde aus der flaechendeckenden OCR-Spur (tv-ocr-spur):

    hinweis_nah  1, wenn ein Programmhinweis in +-FENSTER s liegt
    werbung_nah  1, wenn eine Werbe-Kennzeichnung in +-FENSTER s liegt
    spur_da      1, wenn die Sekunde in einer abgetasteten Kachel liegt

Keine Spur → alle drei 0. Genutzt von train-head.py (zusatzspalten, ocr=True)
und scripts/o26-ocr-spalte.py. Die Go-Seite (internal/signals/ocrspalten.go)
baut dieselben Spalten beim Detect nach; scripts/gen-ocr-paritaet.py bindet
beide ueber Goldwerte aneinander — eine Abweichung waere ein stiller
Train/Serve-Bruch, kein Fehler.
"""
import json
import math
from pathlib import Path

import numpy as np

SPUR = Path.home() / ".cache/tvd-ocr-spur"
FENSTER = 10  # +- Sekunden (O26-Registrierung)
BREITE = 3


def aus_spur(s, n_sek):
    """(n_sek, 3) float32 aus einem bereits geladenen Spur-Dict."""
    aus = np.zeros((n_sek, BREITE), np.float32)
    for sp in s.get("abgetastet") or []:
        a = max(0, int(sp["von"]))
        b = min(n_sek, int(math.ceil(sp["bis"])))
        if b > a:
            aus[a:b, 2] = 1
    for f in s.get("funde") or []:
        t = int(f["time_s"])
        lo, hi = max(0, t - FENSTER), min(n_sek, t + FENSTER + 1)
        if hi <= lo:
            continue
        if f.get("hinweis"):
            aus[lo:hi, 0] = 1
        if f.get("werbemarker"):
            aus[lo:hi, 1] = 1
    return aus


QUELLEN = Path.home() / ".cache/tv-detect-daemon/source"


def spur_passt(s, quelle):
    """Passt die Spur zur Quelle? Liegt keine Quelle im Cache, ist das nicht
    pruefbar — dann gilt die Spur (der Daemon raeumt sie bei jedem Re-Filter
    mit ab, _invalidate_derived)."""
    try:
        st = Path(quelle).stat()
    except OSError:
        return True
    return (s.get("quelle_bytes") == st.st_size
            and int(s.get("quelle_mtime", -1)) == int(st.st_mtime))


def ocr_spalten(uuid, n_sek, spur_dir=None, quellen_dir=None):
    """(n_sek, 3) float32 fuer eine Aufnahme; keine oder veraltete Spur →
    Nullen.

    ⚠️ Veraltet = gegen eine andere Quelle gerechnet (Groesse/mtime). Bis
    2026-09-25 las das Training jede Spur ungeprueft, der Daemon dagegen
    nur frische (_ocr_spur_frisch): nach einem Re-Filter lernte der Kopf
    OCR-Spalten auf der alten Zeitachse, im Betrieb bekam dieselbe Aufnahme
    Nullen — ein stiller Train/Serve-Bruch (Sweep).
    """
    p = Path(spur_dir or SPUR) / f"{uuid}.json"
    if not p.is_file():
        return np.zeros((n_sek, BREITE), np.float32)
    s = json.loads(p.read_text())
    if not spur_passt(s, Path(quellen_dir or QUELLEN) / f"{uuid}.ts"):
        return np.zeros((n_sek, BREITE), np.float32)
    return aus_spur(s, n_sek)
