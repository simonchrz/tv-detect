#!/usr/bin/env python3
"""Registriert einen Versuchsarm VOR dem ersten Messpunkt.

Schreibt docs/optimierer/laeufe/NNN-<arm>/vorschlag.json mit Hypothese,
Regeltyp und dem Code-Abdruck (opt_lauf.py + Arm-Datei) und committet Arm
und Registrierung auf main. opt_lauf.py --lauf verweigert jede Messung,
deren Code vom registrierten Abdruck abweicht; opt_urteil.py rechnet nur
die hier festgelegte Regel. Die Schwellen stehen in opt_urteil.py, nicht
hier — ein Arm kann sie nicht lockern.
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
REPO = HIER.parent.parent
LAEUFE = REPO / "docs/optimierer/laeufe"

# Was ein Arm nie anfassen darf. Grob, aber billig; der Pruefer-Agent liest
# den Code zusaetzlich.
VERBOTEN = [
    (r"ads_user|cutlist|ads\.json", "Labels/Cutlists lesen oder schreiben (L2)"),
    (r"messsatz|bestaetigung|entwicklung|golden|versiegelt|split-ledger",
     "Saetze oder Ledger selbst waehlen"),
    (r"tvd-train-archive|tv-train-head-out|\.tvd-models|head\.bin",
     "Archiv, Kopf oder Produktion beruehren (L3/L4)"),
    (r"emit-signals|replay|subprocess|os\.system|requests|urllib|socket",
     "Prozesse, Netz oder die Dekoder-Eingaben"),
    (r"open\([^)]*['\"][wa]", "Dateien schreiben"),
    (r"\bunlink\b|\brmtree\b|os\.remove", "loeschen (L6)"),
]
ERLAUBT = {"BESCHREIBUNG", "HYPER", "spalten", "gewichte", "ziel", "modell", "nachbearbeiten"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="Dateiname unter arme/ ohne .py")
    ap.add_argument("--hypothese", required=True)
    ap.add_argument("--begruendung", required=True,
                    help="Welcher Fehlerposten, welcher Beleg, warum nicht im Friedhof")
    ap.add_argument("--regel", required=True, choices=["gewinn", "nicht_schlechter"])
    ap.add_argument("--klasse", required=True,
                    choices=["merkmale", "training", "nachbearbeitung", "daten", "labels"],
                    help="Ideenklasse fuer die Banditen-Bilanz (opt_bandit.py)")
    a = ap.parse_args()

    if not re.fullmatch(r"[a-z0-9_]{3,40}", a.arm):
        sys.exit("Armname: 3-40 Zeichen a-z0-9_")
    datei = HIER / "arme" / f"{a.arm}.py"
    if not datei.is_file():
        sys.exit(f"{datei} fehlt")
    if any((d / "vorschlag.json").is_file()
           and json.loads((d / "vorschlag.json").read_text())["arm"] == a.arm
           for d in LAEUFE.glob("*")):
        sys.exit(f"Arm {a.arm} ist schon registriert — eine Variante braucht einen neuen Namen.")
    code = datei.read_text()
    for muster, warum in VERBOTEN:
        if re.search(muster, code, re.I):
            sys.exit(f"verboten im Arm ({warum}): Muster {muster!r}")
    spec = importlib.util.spec_from_file_location("arm_pruef", datei)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    fremd = {k for k in vars(m) if not k.startswith("_") and callable(getattr(m, k))
             and getattr(getattr(m, k), "__module__", None) == "arm_pruef"} - ERLAUBT
    if fremd:
        print(f"Hinweis: Hilfsfunktionen {sorted(fremd)} (erlaubt, werden nicht aufgerufen)")
    if not (set(vars(m)) & ERLAUBT - {"BESCHREIBUNG"}):
        sys.exit("Der Arm aendert nichts (keine der Schnittstellen definiert).")

    pflicht = {"merkmale": ["spalten"], "nachbearbeitung": ["nachbearbeiten"],
               "labels": ["ziel"], "daten": ["gewichte", "ziel"],
               "training": ["HYPER", "gewichte", "modell", "ziel"]}[a.klasse]
    if not any(hasattr(m, k) for k in pflicht):
        sys.exit(f"Klasse {a.klasse} verlangt mindestens eins von {pflicht}")
    sys.path.insert(0, str(HIER))
    lauf = importlib.util.spec_from_file_location("opt_lauf_reg", HIER / "opt_lauf.py")
    ol = importlib.util.module_from_spec(lauf)
    lauf.loader.exec_module(ol)
    abdruck = ol.code_abdruck(m)

    nr = 1 + max([int(d.name[:3]) for d in LAEUFE.glob("[0-9][0-9][0-9]-*")] or [0])
    ld = LAEUFE / f"{nr:03d}-{a.arm}"
    ld.mkdir(parents=True)
    v = {"nr": nr, "arm": a.arm, "hypothese": a.hypothese, "begruendung": a.begruendung,
         "regel": a.regel, "klasse": a.klasse, "code": abdruck,
         "beschreibung": getattr(m, "BESCHREIBUNG", "")}
    (ld / "vorschlag.json").write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n")
    neu = [str(datei), str(ld / "vorschlag.json")]
    befund = HIER / "arme" / f"{a.arm}.befund.md"
    if befund.is_file():
        (ld / "befund.md").write_text(befund.read_text())
        neu.append(str(ld / "befund.md"))
        befund.unlink()
    subprocess.run(["git", "-C", str(REPO), "add"] + neu, check=True)
    subprocess.run(["git", "-C", str(REPO), "commit", "-q", "-m",
                    f"Optimierer {nr:03d} registriert: {a.arm} ({a.regel})\n\n{a.hypothese}\n\n"
                    "Claude-Session: https://claude.ai/code/session_011Hzo2LXp3b9pNNwwFazY4p"],
                   check=True)
    print(json.dumps({"lauf": ld.name, "code": abdruck}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
