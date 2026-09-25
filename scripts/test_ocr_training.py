#!/usr/bin/env python3
"""O26 im Training: OCR-Spalten + MLP6-Kopf (L5-Vertrag).

Was still brechen koennte:
  * die OCR-Spalten stehen nicht GANZ HINTEN (Praefix-Vertrag) oder sind
    andere als die, die scripts/ocr_spalten.py und damit Go bauen;
  * load_deployed_mlp kennt n_ocr nicht — dann rechnet der Hygiene-Lehrer
    die Breite falsch: nackt+OCR (+3) sieht aus wie ein v3-Block
    (whisper+dp+dn) und bekaeme eine voellig andere Eingabe;
  * write_mlp_head_v6 → load_deployed_mlp verliert Gewichte.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TD = REPO / "internal/signals/testdata"


def _lade(name, datei):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / datei)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


TH = _lade("th_ocr", "train-head.py")

SPUR = {"dauer_s": 30.0, "abgetastet": [{"von": 0, "bis": 30.0}],
        "funde": [{"time_s": 3, "hinweis": True, "werbemarker": False},
                  {"time_s": 18, "hinweis": False, "werbemarker": True}]}


def _mit_spur(uuid):
    d = Path(tempfile.mkdtemp())
    (d / f"{uuid}.json").write_text(json.dumps(SPUR))
    oc = TH._ocr_modul()
    alt = oc.SPUR
    oc.SPUR = d
    return oc, alt


def test_ocr_spalten_ganz_hinten():
    oc, alt = _mit_spur("dvr-x-1")
    try:
        rng = np.random.RandomState(0)
        X = rng.randn(40, 1282).astype(np.float32)
        kw = dict(kanal=True, whisper=False, temporal=True, churn=True)
        ohne = TH.mit_zusatz(X, "dvr-x-1", "x", {"x": 0, "y": 1}, **kw)
        mit = TH.mit_zusatz(X, "dvr-x-1", "x", {"x": 0, "y": 1}, ocr=True, **kw)
        assert mit.shape[1] == ohne.shape[1] + 3
        assert np.array_equal(mit[:, :-3], ohne), "Praefix verschoben"
        assert np.array_equal(mit[:, -3:], oc.aus_spur(SPUR, 40)), \
            "andere Spalten als ocr_spalten.py (= als Go)"
        assert mit[:, -3:].sum() > 0, "Test zahnlos: Spur lieferte nur Nullen"
    finally:
        oc.SPUR = alt


def test_ohne_spur_nullen():
    oc, alt = _mit_spur("dvr-x-1")
    try:
        X = np.zeros((10, 1282), np.float32)
        z = TH.zusatzspalten(X, "dvr-gibt-es-nicht-1", "x", {}, kanal=False,
                             ocr=True)
        assert z.shape == (10, 3) and not z.any()
    finally:
        oc.SPUR = alt


class _Attrappe:
    def __init__(self, W1, b1, W2, b2):
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, b2]


def test_v6_rundreise_und_n_ocr():
    rng = np.random.RandomState(1)
    D, H = 1285, 8
    k = _Attrappe(rng.randn(D, H).astype(np.float32), rng.randn(H).astype(np.float32),
                  rng.randn(H, 1).astype(np.float32), rng.randn(1).astype(np.float32))
    p = Path(tempfile.mkdtemp()) / "head.bin"
    TH.write_mlp_head_v6(p, k, input_dim=D, hidden_dim=H, n_logo=1, n_audio=1,
                         n_ocr=3)
    d = TH.load_deployed_mlp(p)
    assert d is not None and d.input_dim == D and d.n_ocr == 3
    assert np.allclose(d.W1, k.coefs_[0]) and np.allclose(d.b2, k.intercepts_[1])


def test_echte_fixture_und_lehrer_breite():
    """Die Go-Paritaets-Fixture: nackt (1282) + 3 OCR. Die Lehrer-Rechnung
    muss 0 Zusatzspalten ergeben — NICHT 3 (= v3-Block)."""
    d = TH.load_deployed_mlp(TD / "mlp6-ocr.bin")
    assert d is not None and d.n_ocr == 3
    feat_dim, n_chan = 1282, 0
    assert d.input_dim - (feat_dim + n_chan) - d.n_ocr == 0
    assert TH.load_deployed_mlp(TD / "mlp1-bare.bin").n_ocr == 0


def test_v6_verweigert_falsche_ocr_breite():
    k = _Attrappe(np.zeros((1284, 2), np.float32), np.zeros(2, np.float32),
                  np.zeros((2, 1), np.float32), np.zeros(1, np.float32))
    try:
        TH.write_mlp_head_v6(Path(tempfile.mkdtemp()) / "h.bin", k,
                             input_dim=1284, hidden_dim=2, n_logo=1, n_audio=1,
                             n_ocr=2)
    except ValueError:
        return
    raise AssertionError("n_ocr=2 wurde geschrieben")




def test_veraltete_spur_gibt_nullen():
    """Spur gegen eine andere Quelle gerechnet → Nullen, wie im Betrieb."""
    oc = TH._ocr_modul()
    d = Path(tempfile.mkdtemp())
    q = d / "quellen"; q.mkdir()
    ts = q / "dvr-x-7.ts"; ts.write_bytes(b"x" * 1000)
    st = ts.stat()
    spur = dict(SPUR, quelle_bytes=1000, quelle_mtime=int(st.st_mtime))
    (d / "dvr-x-7.json").write_text(json.dumps(spur))
    frisch = oc.ocr_spalten("dvr-x-7", 40, spur_dir=d, quellen_dir=q)
    assert frisch.sum() > 0, "frische Spur verworfen"
    spur["quelle_bytes"] = 999
    (d / "dvr-x-7.json").write_text(json.dumps(spur))
    alt = oc.ocr_spalten("dvr-x-7", 40, spur_dir=d, quellen_dir=q)
    assert not alt.any(), "veraltete Spur in den Spalten — Train/Serve-Bruch"
    # Ohne Quelle im Cache nicht prüfbar → Spur gilt.
    assert oc.ocr_spalten("dvr-x-7", 40, spur_dir=d, quellen_dir=d / "leer").sum() > 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok ", n)
