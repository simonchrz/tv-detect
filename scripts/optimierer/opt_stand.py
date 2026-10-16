#!/usr/bin/env python3
"""Kurzer Stand fuer den Planer: die letzten Laeufe in einer Zeile je Lauf,
dazu die Befragungszahlen. Ersetzt das Lesen aller vorschlag/urteil-Dateien,
damit der Planer-Kontext nicht mit der Zahl der Laeufe waechst."""
import argparse
import json
from pathlib import Path

LAEUFE = Path(__file__).resolve().parent.parent.parent / "docs/optimierer/laeufe"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letzte", type=int, default=15)
    a = ap.parse_args()
    zeilen, n_entw, n_best = [], 0, 0
    for d in sorted(LAEUFE.glob("[0-9][0-9][0-9]-*")):
        v = json.loads((d / "vorschlag.json").read_text())
        u = json.loads((d / "urteil.json").read_text()) if (d / "urteil.json").is_file() else {}
        n_entw += "entwicklung" in u
        n_best += "bestaetigung" in u
        e = u.get("entwicklung", {})
        zeilen.append(f"{d.name} [{v['regel']}] {u.get('status', 'offen')}"
                      + (f" Δ{e['median_delta']:+.4f}" if e else "")
                      + f" — {v['hypothese'][:160]}")
    print(f"{len(zeilen)} Laeufe, Entwicklung {n_entw}x befragt, Bestaetigung {n_best}x.")
    print("\n".join(zeilen[-a.letzte:]) or "(noch keine)")


if __name__ == "__main__":
    main()
