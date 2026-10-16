#!/usr/bin/env python3
"""Urteil zu einem registrierten Lauf — mechanisch, Schwellen fest im Code.

Rauschen (A/A, 16.10.): Kontrolle gegen Kontrolle mit anderen Seeds,
je 5 Wiederholungen zu 3 Koepfen auf 76 Entwicklungs-Aufnahmen: einzelne
Wiederholung sd ~0.001 (mittlere Block-IoU), Mediane dreier Saetze
0.9294 / 0.9304 / 0.9306. Daraus:

ENTWICKLUNG, Regel "gewinn":
  Delta_r = IoU-Mittel(Arm, Wiederholung r) - IoU-Mittel(K, r), gepaart
  [1] Median Delta_r >= +0.003
  [2] mindestens 4 von 5 Delta_r > 0
  [3] hoechstens 1 Aufnahme mit mittlerem Delta < -0.10 (kein Tausch
      grosser Einzelverluste gegen viele kleine Gewinne)
ENTWICKLUNG, Regel "nicht_schlechter" (Vereinfachungen):
  [1] Median Delta_r >= -0.001   [2] hoechstens 1 Delta_r < -0.003   [3] wie oben
BESTAETIGUNG (22 test-Aufnahmen, einmal):
  [1] Median Delta_r > 0 (bei nicht_schlechter: >= -0.002)   [2] >= 4/5 positiv
      (bei nicht_schlechter: hoechstens 1 unter -0.005)

Status: verworfen | bestaetigung_faellig | kandidat | nicht_bestaetigt.
"kandidat" heisst NICHT ausliefern: die Uebertragung in train-head.py und
der Nachweis dort sind Simons Entscheidung (L3/L5).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
REPO = HIER.parent.parent
LAEUFE = REPO / "docs/optimierer/laeufe"
JOURNAL = REPO / "docs/optimierer/journal.md"


def deltas(erg, arm):
    K = {r["seed"]: r for r in erg["laeufe"]["kontrolle"]}
    A = {r["seed"]: r for r in erg["laeufe"][arm]}
    seeds = sorted(set(K) & set(A))
    d = np.array([A[s]["iou_mittel"] - K[s]["iou_mittel"] for s in seeds])
    je = {}
    for u in K[seeds[0]]["je_aufnahme"]:
        v = [A[s]["je_aufnahme"].get(u, np.nan) - K[s]["je_aufnahme"].get(u, np.nan) for s in seeds]
        je[u] = float(np.nanmean(v))
    return d, je, float(np.mean([K[s]["iou_mittel"] for s in seeds]))


def pruefe(d, je, regel, satz):
    gross = sorted((v, u) for u, v in je.items() if v < -0.10)
    med = float(np.median(d))
    if satz == "entwicklung":
        if regel == "gewinn":
            r = [med >= 0.003, int((d > 0).sum()) >= 4, len(gross) <= 1]
        else:
            r = [med >= -0.001, int((d < -0.003).sum()) <= 1, len(gross) <= 1]
    else:
        if regel == "gewinn":
            r = [med > 0, int((d > 0).sum()) >= 4]
        else:
            r = [med >= -0.002, int((d < -0.005).sum()) <= 1]
    return all(r), r, med, gross


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lauf", required=True)
    a = ap.parse_args()
    ld = LAEUFE / a.lauf
    v = json.loads((ld / "vorschlag.json").read_text())
    u_pfad = ld / "urteil.json"
    urteil = json.loads(u_pfad.read_text()) if u_pfad.is_file() else {}

    zeilen = []
    for satz in ("entwicklung", "bestaetigung"):
        e = ld / f"ergebnis-{satz}.json"
        if not e.is_file() or satz in urteil:
            continue
        erg = json.loads(e.read_text())
        if erg["code"] != v["code"]:
            sys.exit(f"{satz}: Code-Abdruck {erg['code']} != registriert {v['code']} — ungueltig")
        d, je, k_mittel = deltas(erg, v["arm"])
        ok, teile, med, gross = pruefe(d, je, v["regel"], satz)
        beste = sorted(je.items(), key=lambda x: -x[1])[:3]
        urteil[satz] = {"erfuellt": ok, "teile": teile, "median_delta": round(med, 4),
                        "deltas": [round(float(x), 4) for x in d], "k_mittel": round(k_mittel, 4),
                        "grosse_verluste": [(u, round(x, 3)) for x, u in gross],
                        "groesste_gewinne": [(u, round(x, 3)) for u, x in beste],
                        "sekunden": erg.get("sekunden")}
        zeilen.append(f"{satz}: {'ERFUELLT' if ok else 'NICHT ERFUELLT'} "
                      f"(Median Delta {med:+.4f}, {[round(float(x), 4) for x in d]}, Teile {teile})")
    if "bestaetigung" in urteil:
        urteil["status"] = "kandidat" if urteil["bestaetigung"]["erfuellt"] else "nicht_bestaetigt"
    elif "entwicklung" in urteil:
        urteil["status"] = "bestaetigung_faellig" if urteil["entwicklung"]["erfuellt"] else "verworfen"
    else:
        sys.exit("noch kein Ergebnis")
    u_pfad.write_text(json.dumps(urteil, indent=1, ensure_ascii=False) + "\n")
    if urteil["status"] == "kandidat" and not (ld / "kandidat.md").exists():
        subprocess.run([sys.executable, str(HIER / "opt_kandidat.py"), "--lauf", a.lauf], check=True)

    if zeilen:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        if not JOURNAL.exists():
            JOURNAL.write_text("# Optimierer-Journal\n\nEine Zeile je Urteil, neueste unten. "
                               "Zaehlt mit, wie oft der Entwicklungs- und der Bestaetigungssatz "
                               "befragt wurden.\n\n")
        n_entw = sum(1 for d in LAEUFE.glob("*/urteil.json")
                     if "entwicklung" in json.loads(d.read_text()))
        n_best = sum(1 for d in LAEUFE.glob("*/urteil.json")
                     if "bestaetigung" in json.loads(d.read_text()))
        with JOURNAL.open("a") as fh:
            for z in zeilen:
                fh.write(f"- **{a.lauf}** ({v['regel']}) {z} → Status `{urteil['status']}` "
                         f"[Entwicklung befragt: {n_entw}, Bestaetigung: {n_best}]\n")
        subprocess.run(["git", "-C", str(REPO), "add", str(ld), str(JOURNAL)], check=True)
        subprocess.run(["git", "-C", str(REPO), "commit", "-q", "-m",
                        f"Optimierer {a.lauf}: {urteil['status']}\n\n" + "\n".join(zeilen)
                        + "\n\nClaude-Session: https://claude.ai/code/session_011Hzo2LXp3b9pNNwwFazY4p"],
                       check=True)
    print(json.dumps({"lauf": a.lauf, **urteil}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
