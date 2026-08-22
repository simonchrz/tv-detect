"""Der Gate-Lader muss JEDES Format lesen, das die Produktion schreibt.

⚠️ Der Anlass (2026-08-22): `load_deployed_mlp` kannte MLP2..MLP5, aber
NICHT MLP1 — und MLP1 ist kein Altbestand, sondern das laufende Format für
Köpfe ohne Whisper-Spalte ("MLP2 v2" if wants_whisper else "MLP1 v1").
Produktion fährt seit dem Rückbau auf den nackten Kopf genau das.

Folge: der Lader gab None zurück, `deployed_test_metrics` blieb None, und
der PAARWEISE Gate-Vergleich fiel jede Nacht aus. Der Nightly meldete
stattdessen „test-set composition changed — comparison invalidated,
deploying"; es schützte allein der Golden-Boden. Drei Nächte lang, und
aufgefallen ist es nur, weil die Invalidierungs-Zahl (101→102) sich nicht
mehr änderte.

Dieser Test prüft die Eigenschaft, die davor schützt: für jedes
Magic, das die Schreibseite kennt, muss der Lader einen Kopf zurückgeben —
und zwar mit den Maßen, die im Header stehen.
"""
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _modul():
    spec = importlib.util.spec_from_file_location("th", REPO / "scripts/train-head.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["th"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def _kopf(magic, version, kopf_felder, D=8, H=4, O=1):
    """Baut einen minimalen, aber formal gültigen Kopf."""
    hdr = struct.pack(f"<{kopf_felder}I", magic, version, D, H, O,
                      *([0] * (kopf_felder - 5)))
    body = (np.zeros(D * H, np.float32).tobytes()
            + np.zeros(H, np.float32).tobytes()
            + np.zeros(H * O, np.float32).tobytes()
            + np.zeros(O, np.float32).tobytes())
    return hdr + body, D, H


# magic, version, Anzahl uint32 im Header
FORMATE = [
    (0x31504C4D, 1, 9),   # MLP1 — 36 B, der Fall, der gefehlt hat
    (0x32504C4D, 2, 10),  # MLP2 — 40 B
    (0x33504C4D, 3, 11),  # MLP3 — 44 B
    (0x34504C4D, 4, 12),  # MLP4 — 48 B
    (0x35504C4D, 5, 13),  # MLP5 — 52 B
]


def test_alle_geschriebenen_formate_sind_lesbar(tmp_path=None):
    m = _modul()
    tmp = Path(tmp_path or "/tmp") / "gate-lader-test"
    tmp.mkdir(parents=True, exist_ok=True)
    for magic, version, felder in FORMATE:
        roh, D, H = _kopf(magic, version, felder)
        p = tmp / f"head-v{version}.bin"
        p.write_bytes(roh)
        d = m.load_deployed_mlp(p)
        name = struct.pack("<I", magic).decode("latin1")
        assert d is not None, (
            f"{name} nicht lesbar — der paarweise Gate-Vergleich faellt fuer "
            f"dieses Format STILL aus (genau der Fehler vom 2026-08-22)")
        assert d.input_dim == D, f"{name}: input_dim {d.input_dim} != {D}"


def test_unbekanntes_magic_bleibt_none():
    """Die Gegenrichtung: ein echter Fremdkopf darf NICHT durchrutschen."""
    m = _modul()
    p = Path("/tmp") / "gate-lader-fremd.bin"
    p.write_bytes(struct.pack("<9I", 0x11223344,
                              1, 8, 4, 1, 0, 0, 0, 0)
                  + b"\x00" * 200)
    assert m.load_deployed_mlp(p) is None


if __name__ == "__main__":
    test_alle_geschriebenen_formate_sind_lesbar()
    test_unbekanntes_magic_bleibt_none()
    print("beide Tests gruen")
