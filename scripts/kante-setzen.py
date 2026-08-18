#!/usr/bin/env python3
"""Eine einzelne Golden-Kante auf einen von Simon abgelesenen Wert setzen.

Gegenstueck zu `golden-korrigieren.py`, das die Funde EINER Audit-Runde in
einem Rutsch schrieb und sich danach selbst sperrt. Hier geht es um den
Einzelfall: Simon sieht sich einen Bildstreifen an, nennt eine Sekunde,
und die wird gesetzt — nachvollziehbar und mit Sicherung.

⚠️ **Eingriff in den Massstab (L2).** Jede Aenderung braucht Simons
ausdrueckliche Zahl. Ein Agenten-Urteil ist hier KEINE Deckung; es darf
hoechstens der Anlass sein, ihn hinsehen zu lassen.

⚠️ Geschrieben wird gegen die MENSCHLICHEN Labels (`user`), nicht gegen die
zusammengefuehrte `ads`-Sicht des Gateways. Die enthaelt zusaetzlich den
overrun-Block (Schwanz hinter dem geplanten Ende), und der wuerde beim
Zurueckschreiben still zu einem Menschen-Label werden.

⚠️ Folge, erwartet: `label_hash` schneidet die Label-Epoche, die Latte setzt
neu auf. Solange in der laufenden Epoche noch kein Wert zweimal erreicht
wurde, kostet das nichts — spaeter schon.

    scripts/kante-setzen.py <uuid> start|ende <alt> <neu>          # zeigt
    scripts/kante-setzen.py <uuid> start|ende <alt> <neu> --schreiben
"""
import argparse
import importlib.util
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "agent_review", Path(__file__).with_name("agent-review.py"))
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
ARCHIV = Path.home() / ".cache/tvd-train-archive"


def hole(u):
    req = urllib.request.Request(f"{_ar.GATEWAY}/recording/{u}/ads")
    with urllib.request.urlopen(req, context=CTX, timeout=20) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("uuid")
    ap.add_argument("seite", choices=["start", "ende"])
    ap.add_argument("alt", type=float, help="bisheriger Wert (Sicherung gegen Drift)")
    ap.add_argument("neu", type=float, help="von Simon abgelesener Wert")
    ap.add_argument("--schreiben", action="store_true")
    a = ap.parse_args()

    d = hole(a.uuid)
    user = [[float(x[0]), float(x[1])] for x in (d.get("user") or [])]
    if not user:
        print(f"⚠️ {a.uuid}: keine menschlichen Labels — hier ist nichts zu "
              f"korrigieren, was ein Mensch gesetzt haette.")
        return 1
    j = 0 if a.seite == "start" else 1
    treffer = [i for i, b in enumerate(user) if abs(b[j] - a.alt) < 0.5]
    if len(treffer) != 1:
        print(f"⚠️ {a.uuid}: {len(treffer)} Bloecke mit {a.seite} = {a.alt}. "
              f"Erwartet genau einen.\n   user = {user}")
        return 1
    i = treffer[0]
    vorher = list(user[i])
    user[i][j] = a.neu
    if user[i][0] >= user[i][1]:
        print(f"⚠️ Ergebnis waere ein leerer oder verdrehter Block: {user[i]}")
        return 1

    print(f"{a.uuid}")
    print(f"  Block {i}: {vorher[0]:.1f}-{vorher[1]:.1f}  ->  "
          f"{user[i][0]:.1f}-{user[i][1]:.1f}   ({a.neu - a.alt:+.1f}s)")
    auto = [(round(float(x[0]), 1), round(float(x[1]), 1))
            for x in (d.get("auto") or [])]
    nah = min(auto, key=lambda x: abs(x[j] - a.neu), default=None)
    if nah:
        print(f"  Modell an dieser Kante: {nah[j]:.1f}  "
              f"(Abstand zum neuen Wert {abs(nah[j] - a.neu):.1f}s, "
              f"zum alten {abs(nah[j] - a.alt):.1f}s)")
    if not a.schreiben:
        print("\n(Probelauf — nichts geschrieben)")
        return 0

    ARCHIV.mkdir(parents=True, exist_ok=True)
    p_sich = ARCHIV / f"kante-setzen-{time.strftime('%Y-%m-%d')}.jsonl"
    with p_sich.open("a") as f:
        f.write(json.dumps({"zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "uuid": a.uuid, "seite": a.seite, "block": i,
                            "vorher": vorher, "nachher": user[i],
                            "user_vorher": [list(x) for x in
                                            (d.get("user") or [])]}) + "\n")
    body = json.dumps({"ads": user, "reviewed_by": "simon-bildstreifen"}).encode()
    req = urllib.request.Request(
        f"{_ar.GATEWAY}/api/recording/{a.uuid}/ads/edit", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=CTX, timeout=20) as r:
        print(f"\n  HTTP {r.status}   Sicherung: {p_sich.name}")
    nach = hole(a.uuid).get("user") or []
    ist = round(float(nach[i][j]), 1) if i < len(nach) else None
    print(f"  nachgelesen: {a.seite} = {ist}  "
          f"{'✓' if ist == round(a.neu, 1) else '✗ NICHT uebernommen'}")
    return 0 if ist == round(a.neu, 1) else 1


if __name__ == "__main__":
    sys.exit(main())
