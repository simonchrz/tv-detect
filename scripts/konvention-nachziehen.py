#!/usr/bin/env python3
"""Agenten-Labels neu ableiten, wenn sich KONVENTION geändert hat.

Der Grund, warum das ohne einen einzigen neuen Agentenlauf geht: die Agenten
urteilen über **einzelne Bilder** in neutralen Kategorien, und die Zuordnung
„was davon zählt als Werbung" steht an genau einer Stelle
(`agent-review.py:KONVENTION`). Ändert sich die Konvention, sind die
gespeicherten Bild-Urteile weiter gültig — nur die Ableitung ist neu.

⚠️ **Der Anlass, 2026-08-17.** Vom 13.08. bis 17.08. galt „Gewinnspiel-Insert
= Sendung". Unter dieser Lesart haben die Review-Agenten 12 kabel-eins-Blöcke
um ~30 s nach hinten geschoben — und nur kabel-eins, weil nur dort solche
Inserts am Blockanfang stehen. Simon hat die Konvention am 17.08.
zurückgenommen. Ohne dieses Werkzeug müsste jede solche Rücknahme entweder
alle Agentenläufe wiederholen oder die verschobenen Labels stehen lassen —
und ein Korpus mit zwei Konventionen ist schlimmer als einer mit der
falschen.

⚠️ **Der Golden-Satz wird nie angefasst** (dieselbe Sperre wie in
`agent-review.py --anwenden`). Was dort steht, ist der Maßstab.

    scripts/konvention-nachziehen.py            # zeigt nur
    scripts/konvention-nachziehen.py --schreiben
"""
import argparse
import importlib.util
import json
import ssl
import sys
import urllib.request
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ar", Path(__file__).resolve().parent / "agent-review.py")
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def urteile(d):
    """Alle Bild-Urteile einer Aufnahme, über alle Runden hinweg."""
    je = {}
    for p in [d / "urteil.json"] + sorted(d.glob("urteil-r*.json")):
        if not p.is_file():
            continue
        try:
            bilder = json.loads(p.read_text()).get("bilder") or []
        except Exception:
            continue
        for b in bilder:
            try:
                je.setdefault(b["verzeichnis"], []).append(
                    (float(b["zeit"]), str(b["kategorie"])))
            except (KeyError, TypeError, ValueError):
                continue
    return je


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schreiben", action="store_true",
                    help="ohne dies wird nur gezeigt, nichts geändert")
    a = ap.parse_args()

    gesperrt = _ar.golden_uuids()
    if gesperrt is None:
        return 1
    n_gleich = n_neu = n_offen = 0
    for d in sorted(_ar.ARBEIT.glob("*/")):
        ap_ = d / "auftrag.json"
        if not ap_.is_file() or not (d / "angewandt").is_file():
            continue
        auftrag = json.loads(ap_.read_text())
        u = auftrag["uuid"]
        if u in gesperrt:
            print(f"  {u}: GOLDEN — nicht angefasst")
            continue
        je = urteile(d)
        neu = [list(x) for x in auftrag["bloecke"]]
        offen = []
        for k in auftrag["kanten"]:
            kante, grund = _ar.kante_aus_folge(je.get(k["verzeichnis"], []),
                                               k["seite"])
            if kante is None:
                offen.append((k["block"], k["seite"], grund))
                continue
            neu[k["block"]][0 if k["seite"] == "start" else 1] = round(kante, 2)
        # Was steht heute in der Aufnahme?
        p = _ar.SNAPSHOT / f"_rec_{u}" / "ads_user.json"
        jetzt = _ar.bloecke(p)
        neu = [b for b in neu if b[1] - b[0] >= 30]
        if offen:
            # ⚠️ Eine unbestimmte Kante hiess schon beim ersten Anwenden:
            # ganze Aufnahme auslassen. Sonst stuende der Modellwert als
            # Wahrheit im Label. Nach einem Konventionswechsel ist das
            # HAEUFIGER, weil die Bilder um die alte Kante gezogen wurden und
            # die neue Grenze am Fensterrand liegen kann.
            n_offen += 1
            print(f"  {u}: unbestimmt unter der neuen Konvention "
                  f"({offen[0][2]}) — bleibt wie es ist, braucht neue Bilder")
            continue
        if jetzt and all(abs(x[0] - y[0]) < 0.5 and abs(x[1] - y[1]) < 0.5
                         for x, y in zip(jetzt, neu)) and len(jetzt) == len(neu):
            n_gleich += 1
            continue
        n_neu += 1
        print(f"  {u}: {jetzt} -> {neu}")
        if not a.schreiben:
            continue
        body = json.dumps({"ads": neu,
                           "reviewed_by": "agent-review.py"}).encode()
        req = urllib.request.Request(
            f"{_ar.GATEWAY}/api/recording/{u}/ads/edit", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=CTX, timeout=20) as r:
            print(f"      HTTP {r.status}")
    print(f"\n{n_gleich} unveraendert, {n_neu} neu abgeleitet, "
          f"{n_offen} jetzt unbestimmt"
          f"{'' if a.schreiben else '  (Probelauf — nichts geschrieben)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
