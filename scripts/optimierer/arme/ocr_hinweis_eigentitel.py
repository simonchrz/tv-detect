"""OCR-Programmhinweis nach Titel trennen: eigene Sendung gegen fremde.

Die Spalte ocr_hinweis ist 1, wenn in +-10 s eine Programmhinweis-Karte
("Montag 20:15", "Mo-Fr 19:05 joyn") gelesen wurde. Sie wirft zwei Formen
zusammen, die im Trainingsteil entgegengesetzt gelabelt sind:
* Hinweis auf eine FREMDE Sendung = Trailer im Werbeblock (93 % Werbung)
* Hinweis auf die EIGENE Sendung (Titelwort der Aufnahme steht im Text) =
  Bauchbinde/Eigenwerbung in der laufenden Sendung (21 % Werbung)
Der Arm ersetzt ocr_hinweis durch den Fremd-Anteil und haengt eine Spalte
hinweis_eigen an (gleiche +-10-s-Fenster wie ocr_spalten.py). Grundlage ist
nur der OCR-Text und der Titel der Aufnahme (Titel ist erlaubtes Merkmal,
auch im Eval vorhanden), kein Label. Die Spalten entstehen je Sekunde aus
der Spur und werden danach mit ktx["schritt"] ausgeduennt, haben also in
allen Teilen dieselbe Bedeutung. Kein Zufall.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

BESCHREIBUNG = ("OCR-Hinweis nach Titel getrennt: ocr_hinweis nur fuer fremde "
                "Titel, neue Spalte hinweis_eigen fuer Hinweise auf die eigene Sendung")

FENSTER = 10
STOP = {"DER", "DIE", "DAS", "UND", "MIT", "SHOW", "FOLGE", "STAFFEL", "NEUE",
        "SPECIAL", "LIVE", "JOYN", "HEUTE", "MORGEN", "DIESE", "WOCHE"}


def _oc():
    m = sys.modules.get("ocr_spalten_opt")
    if m is not None:
        return m
    pfad = Path(__file__).resolve().parents[2] / "ocr_spalten.py"
    spec = importlib.util.spec_from_file_location("ocr_spalten_opt", pfad)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ocr_spalten_opt"] = m
    spec.loader.exec_module(m)
    return m


def _worte(t):
    return {w for w in re.findall(r"[A-ZÄÖÜ0-9]{4,}", (t or "").upper()) if w not in STOP}


def _spur(u, n):
    """Gueltige OCR-Spur der Aufnahme oder None (wie ocr_spalten.py)."""
    oc = _oc()
    p = oc.SPUR / f"{u}.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text())
    if not oc.spur_passt(s, oc.QUELLEN / f"{u}.ts"):
        return None
    return s


def spalten(teil, X, ktx):
    X = np.array(X, np.float32, copy=True)
    namen = ktx["spalten_namen"]
    ih = namen.index("ocr_hinweis")
    rec, schritt = ktx["rec"], int(ktx["schritt"])
    eig_sp = np.zeros(len(X), np.float32)
    for i, u in enumerate(ktx["uuids"]):
        m = rec == i
        k = int(m.sum())
        if k == 0:
            continue
        n = k * schritt
        s = _spur(u, n)
        if s is None:
            continue
        titel = _worte((ktx["meta"].get(u) or {}).get("title"))
        e = np.zeros(n, np.float32)
        f = np.zeros(n, np.float32)
        for fu in s.get("funde") or []:
            if not fu.get("hinweis"):
                continue
            t = int(fu["time_s"])
            lo, hi = max(0, t - FENSTER), min(n, t + FENSTER + 1)
            if hi <= lo:
                continue
            if titel and (titel & _worte(fu.get("text"))):
                e[lo:hi] = 1
            else:
                f[lo:hi] = 1
        alt = X[m, ih]
        X[m, ih] = np.minimum(alt, f[::schritt][:k])
        eig_sp[m] = np.minimum(e[::schritt][:k], (alt > 0).astype(np.float32))
    return np.concatenate([X, eig_sp[:, None]], 1)
