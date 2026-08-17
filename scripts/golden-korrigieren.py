#!/usr/bin/env python3
"""Die im Audit gefundenen Golden-Kanten korrigieren — einmalig, mit Sicherung.

Gegenstück zu `golden-audit.py`, das nur zeigt. Dieses Skript schreibt, und
zwar bewusst eng:

* **Nur einzelne Kanten, keine ganzen Blöcke.** Was innerhalb der feinen
  Auflösung (5 s) auf dem Bildmaterial sitzt, bleibt unverändert stehen —
  dort hat ein Mensch entschieden, und eine Agenten-Ableitung ist kein
  Grund, das um zwei Sekunden zu verschieben. Angefasst wird nur, was
  **über** der Schwelle liegt.
* **Nur bestimmte Kanten.** Wo `kante_aus_folge` NEIN sagt, passiert nichts.
* **Nur feine Urteile.** Grobe Funde sind Kandidaten, keine Werte (§3al).

⚠️ **Das ist ein Eingriff in den Maßstab (L2).** Er ist gedeckt durch Simons
Entscheidung vom 2026-08-17, nachdem 3 von 3 Stichproben die Funde bestätigt
haben: nick 585 statt 563, rtlzwei-Ende 176 statt 121, rtlzwei-Start 1705
statt 1663. Ohne diese Deckung darf hier nichts geschrieben werden.

⚠️ **Folge, die erwartet ist und kein Fehler:** `label_hash` schneidet danach
die Label-Epoche, die Latte setzt neu auf, und der gemessene Golden-Wert
**sinkt zunächst** — das Modell trifft heute die falschen Kanten, denn es
wurde gegen sie gemessen. Ein Absacken im nächsten Nachtlauf ist also der
Beweis, dass die Korrektur greift, nicht ihr Widerlegung.

Sicherung: der Stand VOR dem Eingriff geht nach
`~/.cache/tvd-train-archive/golden-labels-vor-audit-<datum>.json` — also in
das Verzeichnis, das `tv-backup-labels.sh` nächtlich in das private Repo
committet.

    scripts/golden-korrigieren.py            # zeigt nur
    scripts/golden-korrigieren.py --schreiben
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

ARCHIV = Path.home() / ".cache/tvd-train-archive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schreiben", action="store_true")
    ap.add_argument("--datum", default="2026-08-17",
                    help="Namensteil der Sicherungsdatei (kein Zeitstempel im "
                         "Code, damit ein zweiter Lauf dieselbe Datei meint)")
    a = ap.parse_args()

    golden = _ar.golden_uuids()
    if golden is None:
        return 1
    sicherung, plan = {}, []
    for u in sorted(golden):
        d = _ar.ARBEIT / u
        ap_ = d / "auftrag.json"
        p_lab = _ar.SNAPSHOT / f"_rec_{u}" / "ads_user.json"
        if not ap_.is_file() or not p_lab.is_file():
            continue
        auftrag = json.loads(ap_.read_text())
        if auftrag.get("art") != "fein-nach-grob":
            continue          # grobe Funde sind keine Werte
        lab = _ar.bloecke(p_lab)
        if not lab:
            continue
        je = _ar.urteile_von(d)
        neu = [list(x) for x in lab]
        geaendert = []
        for k in auftrag["kanten"]:
            kante, _ = _ar.kante_aus_folge(je.get(k["verzeichnis"], []),
                                           k["seite"])
            if kante is None:
                continue
            # Den Label-Block finden, der zu diesem gefundenen Block gehoert —
            # ueber die Naehe, nicht ueber den Index: der grobe Durchgang kann
            # anders viele Bloecke gefunden haben als das Label hat.
            g = auftrag["bloecke"][k["block"]]
            i = min(range(len(lab)), key=lambda x: abs(lab[x][0] - g[0]))
            if min(abs(lab[i][0] - g[0]), abs(lab[i][1] - g[1])) > 180:
                continue      # kein plausibler Partner
            j = 0 if k["seite"] == "start" else 1
            if abs(kante - lab[i][j]) <= _ar.SCHRITT_S:
                continue      # sitzt — bleibt, wie der Mensch es gesetzt hat
            neu[i][j] = round(kante, 2)
            geaendert.append((i, k["seite"], lab[i][j], kante))
        if not geaendert:
            continue
        if any(b[1] - b[0] < 30 for b in neu):
            print(f"  {u}: ergaebe einen Block unter 30 s — uebersprungen")
            continue
        sicherung[u] = json.loads(p_lab.read_text())
        plan.append((u, neu, geaendert))

    n = sum(len(g) for _, _, g in plan)
    print(f"{len(plan)} Aufnahme(n), {n} Kante(n) zu korrigieren:\n")
    for u, neu, geaendert in plan:
        for i, seite, alt, jetzt in geaendert:
            print(f"  {u:36} Block{i} {seite:5} {alt:7.1f} -> {jetzt:7.1f}"
                  f"  ({jetzt-alt:+6.1f}s)")
    if not a.schreiben:
        print("\n(Probelauf — nichts geschrieben)")
        return 0

    p_sich = ARCHIV / f"golden-labels-vor-audit-{a.datum}.json"
    if p_sich.exists():
        print(f"\n⚠️ {p_sich.name} existiert bereits — nicht ueberschrieben. "
              f"Ein zweiter Lauf wuerde die Sicherung des ERSTEN zerstoeren.")
        return 1
    p_sich.write_text(json.dumps(sicherung, indent=1))
    print(f"\nSicherung: {p_sich}")
    for u, neu, _ in plan:
        body = json.dumps({"ads": neu, "reviewed_by": "golden-audit"}).encode()
        req = urllib.request.Request(
            f"{_ar.GATEWAY}/api/recording/{u}/ads/edit", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=CTX, timeout=20) as r:
            print(f"  {u}: HTTP {r.status}")
    print(f"\n{n} Kanten geschrieben. ⚠️ Der naechste Nachtlauf zeigt einen "
          f"NEUEN label_hash und\n   eine neu aufsetzende Latte — und der "
          f"Golden-Wert sinkt zunaechst. Beides ist\n   erwartet: das Modell "
          f"wurde gegen die alten Kanten gemessen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
