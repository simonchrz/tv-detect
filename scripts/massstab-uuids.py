#!/usr/bin/env python3
"""Massstab-Liste fuer die Serien-Retention des tv-receivers.

WARUM
-----
golden, test und versiegelt sind die Ground Truth des Modell-Gates. Ein
Label dort laesst sich nur pruefen, solange die Aufnahme lebt — und die
Serien-Retention (5 juengste je Serie, danach 14 Tage) loescht Massstab-
Mitglieder wie alles andere. Stand 2026-10-11: von 199 maschinell gelabelten
Massstab-Mitgliedern waren noch 34 pruefbar, alle 16 im Golden-Satz tot. Der
belegte Kern (22 menschliche Golden-Labels) konnte so nie wachsen, egal wie
viel gewischt wurde.

WAS
---
Eine UUID je Zeile: Golden-Satz + Ledger-Eimer test und versiegelt, ohne die
Ausschlussliste/Quarantaene (die darf nie Messziel sein, also auch nicht
aufgehoben werden). tv-train-head.sh laedt die Datei jede Nacht per
POST /api/internal/model-datei/massstab-uuids.txt zum tv-recorder; der
tv-receiver liest sie unter <shared-hls-dir>/.tvd-models und verschont die
Aufnahmen (dvr_retention.go, retentionKeepFile). Fehlt die Datei, gilt die
Retention unveraendert.

Nur Bericht + Datei; schreibt nichts ins Archiv.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
EIMER = ("test", "versiegelt")


def _lade(name, datei):
    sp = importlib.util.spec_from_file_location(name, Path(__file__).with_name(datei))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def massstab_uuids(archiv=ARCHIV):
    ledger = json.loads((archiv / "split-ledger.json").read_text())
    golden = set(json.loads((archiv / "golden-eval-set.json").read_text()).get("uuids") or [])
    raus = _lade("aus", "test_ausschluss.py").ausgeschlossen(archiv)
    eimer = {u for u, b in ledger.items() if b in EIMER}
    return sorted((golden | eimer) - raus), len(golden), len(eimer), len((golden | eimer) & raus)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aus", default=str(Path.home() / ".cache/tv-train-head-out/massstab-uuids.txt"))
    a = ap.parse_args()
    uuids, n_golden, n_eimer, n_raus = massstab_uuids()
    kopf = (f"# Massstab-Mitglieder (golden + {'/'.join(EIMER)}), von der Serien-Retention "
            f"verschont. Erzeugt von scripts/massstab-uuids.py — nicht von Hand pflegen.\n")
    Path(a.aus).parent.mkdir(parents=True, exist_ok=True)
    Path(a.aus).write_text(kopf + "\n".join(uuids) + "\n")
    print(f"massstab-uuids: {len(uuids)} UUIDs (golden {n_golden}, {'/'.join(EIMER)} {n_eimer}, "
          f"{n_raus} ausgeschlossen) → {a.aus}")


if __name__ == "__main__":
    main()
