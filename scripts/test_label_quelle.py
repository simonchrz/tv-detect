#!/usr/bin/env python3
"""Wer hat das Label gesetzt? Die Frage entscheidet ueber O13s Gueltigkeit."""
import importlib.util, json, sys, tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ks", Path(__file__).with_name("kanten-schatten.py"))
ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)


def quelle(**felder):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"ads": [[10, 20]], **felder}, f)
    return ks.label_quelle(Path(f.name))


def test_blanko_ist_mensch():
    assert quelle(reviewed_at=1788328805) == "mensch"

def test_auto_confirmed_at():
    assert quelle(auto_confirmed_at=1788328805) == "auto"

def test_review_agent():
    assert quelle(reviewed_by="agent-3", reviewed_at=1788328805) == "agent"

def test_fingerprint_ist_maschine():
    # Der Fall vom 2026-09-03: frisches reviewed_at, sonst nichts.
    assert quelle(auto_confirmed_via_fingerprint=True,
                  fingerprint_show="Achtung Abzocke",
                  reviewed_at=1788328805) == "auto"

def test_fingerprint_false_bleibt_mensch():
    # Kein Autoconfirm stattgefunden -> das Feld darf nicht blind zaehlen.
    assert quelle(auto_confirmed_via_fingerprint=False,
                  reviewed_at=1788328805) == "mensch"

def test_agent_schlaegt_nicht_fingerprint():
    # Beides gesetzt: Maschine gewinnt, nie "mensch".
    assert quelle(auto_confirmed_via_fingerprint=True,
                  reviewed_by="agent-3") == "auto"

def test_unlesbar_ist_none():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{kaputt")
    assert ks.label_quelle(Path(f.name)) is None


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError: fails += 1; print("FAIL", n)
    sys.exit(1 if fails else 0)
