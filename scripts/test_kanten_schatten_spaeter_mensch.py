#!/usr/bin/env python3
"""kanten-schatten: ein spaeter menschlich reviewter Stand verdraengt den
Agenten-Stand derselben Aufnahme.

Bis 2026-09-25 zaehlte je Aufnahme nur der erste Fund; wurde ein
Agentenlabel spaeter von Simon korrigiert, blieb die Aufnahme als "agent"
im Ledger und fehlte O13.
"""
import contextlib, importlib.util, io, json, sys, tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ks", Path(__file__).with_name("kanten-schatten.py"))
ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)

U = "dvr-rtl-1787000000"


def test_je_aufnahme_bevorzugt_spaeteren_menschen():
    e = [{"uuid": U, "label_quelle": "agent", "n": 1},
         {"uuid": "x", "label_quelle": "agent", "n": 2},
         {"uuid": U, "label_quelle": "mensch", "n": 3}]
    aus = {z["uuid"]: z["n"] for z in ks.je_aufnahme(e)}
    assert aus == {U: 3, "x": 2}, aus


def test_mensch_bleibt_auch_wenn_spaeter_agent():
    e = [{"uuid": U, "label_quelle": "mensch", "n": 1},
         {"uuid": U, "label_quelle": "agent", "n": 2}]
    assert [z["n"] for z in ks.je_aufnahme(e)] == [1]


def test_sammle_schreibt_den_menschlichen_nachtrag():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        snap, dumps = t / "snap", t / "dumps"
        rec = snap / f"_rec_{U}"
        rec.mkdir(parents=True); dumps.mkdir()
        (rec / "ads.json").write_text(json.dumps([[100, 200]]))
        (dumps / f"{U}.json").write_text(json.dumps(
            {"fps": 1, "nn_confs": [0.1] * 400, "logo_confs": [0.9] * 400}))
        ts = ks.AKTIVIERUNG + 1000
        alt = (ks.SNAPSHOT, ks.DUMPS, ks.LEDGER)
        ks.SNAPSHOT, ks.DUMPS, ks.LEDGER = snap, dumps, t / "ledger.jsonl"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # 1. Agent
                (rec / "ads_user.json").write_text(json.dumps(
                    {"ads": [[105, 195]], "reviewed_by": "agent-review.py",
                     "auto_at_review_at": ts}))
                ks.sammle()
                ks.sammle()          # nichts Neues: bleibt eine Zeile
                n1 = len(ks.LEDGER.read_text().splitlines())
                # 2. Mensch speichert, die App schickt kein reviewed_by
                (rec / "ads_user.json").write_text(json.dumps(
                    {"ads": [[110, 190]], "auto_at_review_at": ts + 500}))
                ks.sammle()
                ks.sammle()          # und danach nicht noch einmal
                zeilen = [json.loads(z) for z in ks.LEDGER.read_text().splitlines()]
        finally:
            ks.SNAPSHOT, ks.DUMPS, ks.LEDGER = alt
    assert n1 == 1, n1
    assert [z["label_quelle"] for z in zeilen] == ["agent", "mensch"], zeilen
    gewertet = ks.je_aufnahme(zeilen)
    assert len(gewertet) == 1 and gewertet[0]["label_quelle"] == "mensch"


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
