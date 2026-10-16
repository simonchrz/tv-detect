#!/usr/bin/env python3
"""Entscheidungsblatt fuer einen bestaetigten Arm (Status `kandidat`).

Schreibt docs/optimierer/laeufe/NNN-<arm>/kandidat.md: Hypothese, Zahlen auf
beiden Saetzen, groesste Gewinne und Verluste je Aufnahme (mit Titel), wie oft
die Saetze bis dahin befragt wurden, und was die Uebertragung in die
Produktion kostet — abgeleitet aus den Schnittstellen, die der Arm benutzt.
Simon antwortet mit „uebertragen“ oder „nein“. Aendert nichts sonst.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
REPO = HIER.parent.parent
LAEUFE = REPO / "docs/optimierer/laeufe"
ARCH = Path.home() / ".cache/tvd-train-archive"

KOSTEN = {
    "HYPER": "nur Trainings-Parameter: ein Flag bzw. eine Konstante in train-head.py, "
             "kein Go, kein Header-Wechsel.",
    "gewichte": "nur Training: Gewichtung in train-head.py nachbauen, kein Go.",
    "ziel": "nur Training: Zielbildung in train-head.py nachbauen, kein Go.",
    "spalten": "NEUE EINGABESPALTEN: Training UND Go-Inferenz (nn.go), Header-Bump, "
               "Paritaets-Fixture — L5, braucht dein ausdrueckliches OK.",
    "modell": "ANDERE ARCHITEKTUR: neues head.bin-Format, Go-Lader, Paritaets-Fixture — L5.",
    "nachbearbeiten": "NACHBEARBEITUNG DER NN-AUSGABE: Go-Inferenz-Pfad aendern, "
                      "Paritaets-Fixture — L5.",
}


def titel(u):
    try:
        m = json.loads(str(np.load(ARCH / f"{u}.npz", allow_pickle=True)["meta"]))
        return f"{m.get('title', '?')} ({m.get('slug', '?')})"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lauf", required=True)
    a = ap.parse_args()
    ld = LAEUFE / a.lauf
    v = json.loads((ld / "vorschlag.json").read_text())
    u = json.loads((ld / "urteil.json").read_text())
    if u.get("status") != "kandidat":
        sys.exit(f"Status {u.get('status')} — kein Kandidat")
    armdatei = HIER / "arme" / f"{v['arm']}.py"
    code = armdatei.read_text()
    benutzt = [k for k in KOSTEN if (f"def {k}(" in code or f"{k} =" in code)]
    n_entw = sum(1 for d in LAEUFE.glob("*/urteil.json") if "entwicklung" in json.loads(d.read_text()))
    n_best = sum(1 for d in LAEUFE.glob("*/urteil.json") if "bestaetigung" in json.loads(d.read_text()))
    e, b = u["entwicklung"], u["bestaetigung"]

    def je(lst):
        return "\n".join(f"| {x[0]} | {titel(x[0])} | {x[1]:+.3f} |" for x in lst) or "| – | – | – |"

    md = f"""# Kandidat {a.lauf}: {v['arm']}

**Entscheidung:** „übertragen“ oder „nein“. Ein Kandidat ist NICHT
ausgeliefert. Übertragen heißt: in `train-head.py` nachbauen, dann entscheidet
das Nachtgate wie immer (L3).

## Hypothese

{v['hypothese']}

**Begründung:** {v['begruendung']}

**Beschreibung im Arm:** {v.get('beschreibung', '')}

## Zahlen (Block-IoU über den Produktions-HSMM, gepaart gegen die Kontrolle)

| Satz | Aufnahmen | Kontrolle | Median Δ | Δ je Wiederholung |
|---|---|---|---|---|
| Entwicklung (Messsatz ∩ train) | 76 | {e['k_mittel']:.4f} | {e['median_delta']:+.4f} | {e['deltas']} |
| Bestätigung (Messsatz ∩ test) | 22 | {b['k_mittel']:.4f} | {b['median_delta']:+.4f} | {b['deltas']} |

Rauschen zum Vergleich: eine Wiederholung schwankt um ±0.001. Abbild-Lücke
der Kontrolle zur Produktion: −0.009 (16.10.).

**Größte Gewinne (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
{je(e.get('groesste_gewinne', []))}

**Verluste über 0.10 (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
{je(e.get('grosse_verluste', []))}

## Wie belastbar

Bis zu diesem Kandidaten wurde der Entwicklungssatz {n_entw}-mal und der
Bestätigungssatz {n_best}-mal befragt. Je höher die erste Zahl, desto eher
ist ein knapper Entwicklungsgewinn Glück; die Bestätigung auf ungesehenen
Aufnahmen ist deshalb das eigentliche Kriterium.

## Was die Übertragung kostet

""" + "\n".join(f"* **{k}:** {KOSTEN[k]}" for k in benutzt) + f"""

## Code des Arms (`scripts/optimierer/arme/{v['arm']}.py`, Abdruck {v['code']})

```python
{code}```
"""
    (ld / "kandidat.md").write_text(md)
    print(ld / "kandidat.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
