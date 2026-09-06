#!/usr/bin/env python3
"""Stehen im Massstab Labels, die die Maschine selbst geschrieben hat? — NUR Bericht.

Der Golden-Satz und der test-Eimer sind die Ground Truth des Gates. Ein
Mitglied, dessen Label die MODELLAUSGABE ist, misst das Modell an sich
selbst: der Fehler ist dort per Konstruktion 0, und jede Alternative kann
nur verlieren. Dasselbe Muster wie in `fingerprint_bestaetigung_ist_kein_mensch`,
nur ueber den zweiten Weg — auto-confirm und cluster-anchored.

⚠️ ZWEI ZAHLEN, DIE MAN NICHT VERWECHSELN DARF (gemessen 2026-09-06):

  * Archiv-Label gegen das ausgelieferte `ads.json`: Herkunft
    `auto`/`auto-confirm` liegt zu 24 von 25 bei IoU **exakt 1.000**. Das
    ist Definition, kein Ergebnis — auto-confirm schreibt die
    Detektorausgabe unveraendert als Label. Diese Zahl sagt: das Label
    traegt keine eigene Information.
  * Naechtliche `per_rec_iou` (frisch trainierter Kopf, HSMM-Dekoder)
    gegen dasselbe Label: Median 0.9720 fuer maschinelle gegen 0.9554 fuer
    menschliche Labels, **keine einzige** bei 1.000. Der Kopf reproduziert
    die alte Entscheidung also nicht punktgenau.

Der Kurzschluss ist deshalb subtiler als "Gratispunkte": die Metrik belohnt
Uebereinstimmung mit dem VORHERIGEN Modell statt mit der Wirklichkeit. Der
gemessene Aufschlag ist mit +0.017 auf den Median klein — der Anteil
maschineller Labels waechst aber, seit niemand mehr reviewt.

WARUM DIESES SKRIPT NICHTS ENTFERNT
-----------------------------------
`golden_boden()` in train-head.py sagt es selbst: Kompositions-Konstanz ist
die ganze Geschaeftsgrundlage der Golden-Zahl. Faellt ein gepinntes Mitglied
weg, ist der Median mit keiner frueheren Nacht mehr vergleichbar — und der
Golden-Boden ist der Gate-Boden (Leitplanke L1). Ein Mitglied zu entfernen
ist deshalb eine Entscheidung mit Kosten, keine Aufraeumarbeit, und sie
gehoert nicht in ein Skript. Hier steht die Zahl, auf der sie getroffen
werden kann: der Golden-Median ueber ALLE Gepinnten neben dem ueber die
menschlich gelabelte Teilmenge.

WOHER DAS URTEIL "MENSCH" KOMMT
-------------------------------
Fuer lebende Aufnahmen aus `ads_user.json` im Label-Backup (dieselbe Regel
und dieselbe NICHT_MENSCH-Liste wie `golden_v3_vorschlag.py`). Fuer tote
Aufnahmen ist das Archiv die einzige Zeugin: `which` in ("user","merged")
heisst, dass zum Zeitpunkt des Archivlaufs ein `ads_user.json` ohne
Auto-Marker vorlag; `auto`/`auto-confirm` heisst Maschine. `merged` ist
dabei die schwaechere Aussage — es ist die Verschmelzung aus Auto- und
Nutzer-Labeln, und laut `label_merge_resurrects_rejected_blocks` koennen
darin verworfene Bloecke wieder auftauchen. Sie wird hier trotzdem als
"Mensch beteiligt" gezaehlt, weil train-head.py sie genau so zaehlt
(`has_user = which in ("user","merged")`) — wer die Konvention aendert,
aendert sie an beiden Stellen.

Exit 1, wenn im Golden-Satz oder im test-Eimer maschinelle Labels stehen —
damit der Bericht in einen Tagesdurchgang gehaengt werden kann.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
BACKUP = Path.home() / "tv-labels-backup"

# Deckungsgleich mit golden_v3_vorschlag.py. Wer dort einen Schreiber
# ergaenzt, ergaenzt ihn auch hier.
NICHT_MENSCH = {"agent-review.py", "claude-code", "zurueckgenommen",
                "folgen-vergleich.py"}

# which-Werte, die ein rein maschinelles Label bezeichnen.
MASCHINE_WHICH = {"auto", "auto-confirm"}


def archiv_meta():
    """uuid -> meta-dict aus den .npz-Sidecars (ohne numpy: nur der Header
    interessiert uns nicht, wir brauchen das JSON — deshalb doch numpy)."""
    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy fehlt — mit /Users/simon/ml/tv-classifier/.venv/bin/python starten")
    out = {}
    for f in ARCHIV.glob("*.npz"):
        try:
            out[f.stem] = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        except Exception:
            continue
    return out


def mensch_lebend(uuid):
    """None = keine Quelle mehr da (dann entscheidet das Archiv)."""
    f = BACKUP / f"_rec_{uuid}" / "ads_user.json"
    if not f.is_file():
        return None
    try:
        j = json.loads(f.read_text())
    except Exception:
        return None
    if j.get("auto_confirmed_at") or j.get("auto_confirmed_via_fingerprint"):
        return False
    if j.get("reviewed_by") in NICHT_MENSCH:
        return False
    return True


def herkunft(uuid, meta):
    """('mensch'|'maschine'|'unbekannt', begruendung)"""
    lebend = mensch_lebend(uuid)
    if lebend is True:
        return "mensch", "ads_user.json ohne Auto-Marker"
    if lebend is False:
        return "maschine", "ads_user.json mit Auto-/Agenten-Marker"
    w = (meta.get(uuid) or {}).get("which")
    if w == "user":
        return "mensch", "Archiv which=user"
    if w == "merged":
        # ⚠️ KORRIGIERT 2026-09-06. `which` entsteht in train-head.py aus der
        # blossen EXISTENZ von ads_user.json (Zeile ~3071) -- und auto-confirm
        # legt genau so eine Datei an. Gemessen an 234 lebenden Aufnahmen mit
        # nicht-leerem ads_user.json: 78 maschinell und 17 agentengeschrieben,
        # zusammen 41 %, alle mit which="merged". Fuer eine tote Aufnahme ist
        # daher NICHT entscheidbar, ob ein Mensch daran war.
        return "unbekannt", "Archiv which=merged (deckt Mensch UND auto-confirm)"
    if w in MASCHINE_WHICH:
        return "maschine", f"Archiv which={w}"
    return "unbekannt", f"keine Quelle, Archiv which={w!r}"


def letzter_iou_lauf():
    """Der juengste Eintrag aus per-rec-iou.jsonl (Champion-Spalte)."""
    p = ARCHIV / "per-rec-iou.jsonl"
    if not p.is_file():
        return None
    letzte = None
    for zeile in p.read_text().splitlines():
        if zeile.strip():
            try:
                letzte = json.loads(zeile)
            except Exception:
                pass
    return letzte


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="maschinenlesbar")
    ap.add_argument("--nur-zaehlen", action="store_true",
                    help="keine Einzelzeilen, nur die Tabelle")
    args = ap.parse_args()

    ledger = json.loads((ARCHIV / "split-ledger.json").read_text())
    golden = json.loads((ARCHIV / "golden-eval-set.json").read_text())
    gold_uuids = list(golden.get("uuids") or [])
    meta = archiv_meta()

    def eimer(u):
        return "golden" if u in set(gold_uuids) else ledger.get(u, "—")

    # Golden ist auf test gepinnt; wir berichten es als eigenen Eimer, damit
    # die Zahlen sich nicht doppelt zaehlen.
    gruppen = {"golden": list(gold_uuids),
               "test": [u for u, b in ledger.items()
                        if b == "test" and u not in set(gold_uuids)],
               "versiegelt": [u for u, b in ledger.items() if b == "versiegelt"]}

    bericht = {}
    for name, uus in gruppen.items():
        eintraege = []
        for u in sorted(uus):
            k, grund = herkunft(u, meta)
            eintraege.append({"uuid": u, "herkunft": k, "grund": grund,
                              "titel": (meta.get(u) or {}).get("title", ""),
                              "n_bloecke": len((meta.get(u) or {}).get("ads") or [])})
        bericht[name] = eintraege

    if args.json:
        print(json.dumps(bericht, ensure_ascii=False, indent=2))
    else:
        print(f"{'Eimer':<12} {'n':>4} {'Mensch':>7} {'Maschine':>9} {'unbekannt':>10}")
        for name, eintraege in bericht.items():
            z = {k: sum(1 for e in eintraege if e["herkunft"] == k)
                 for k in ("mensch", "maschine", "unbekannt")}
            print(f"{name:<12} {len(eintraege):>4} {z['mensch']:>7} "
                  f"{z['maschine']:>9} {z['unbekannt']:>10}")

        if not args.nur_zaehlen:
            for name, eintraege in bericht.items():
                schlecht = [e for e in eintraege if e["herkunft"] != "mensch"]
                if not schlecht:
                    continue
                print(f"\n--- {name}: kein menschliches Label ---")
                for e in schlecht:
                    print(f"  {e['herkunft']:<9} {e['uuid']:<30} "
                          f"{e['n_bloecke']:>2} Bl.  {e['titel'][:34]:<34} ({e['grund']})")

    # Was der Massstab kostet: Golden-Median mit und ohne die Maschinen.
    lauf = letzter_iou_lauf()
    if lauf:
        pr = lauf.get("champion") or {}
        alle = [pr[u] for u in gold_uuids if u in pr]
        menschlich = [pr[e["uuid"]] for e in bericht["golden"]
                      if e["herkunft"] == "mensch" and e["uuid"] in pr]
        if alle and menschlich and len(alle) != len(menschlich):
            print(f"\nGolden-Median (Lauf {lauf.get('ts')}):")
            print(f"  alle {len(alle):>3} Gepinnten     {st.median(alle):.4f}")
            print(f"  nur  {len(menschlich):>3} menschlichen  {st.median(menschlich):.4f}")
            print(f"  Aufschlag durch maschinelle Labels: "
                  f"{st.median(alle) - st.median(menschlich):+.4f}")
            print("\n  ⚠️ Das ist KEINE Empfehlung, den Satz zu aendern. Ein Mitglied"
                  "\n     zu entfernen bricht die Vergleichbarkeit mit jeder frueheren"
                  "\n     Nacht (golden_boden(): Kompositions-Konstanz) und senkt den"
                  "\n     Gate-Boden — Leitplanke L1. Die Zahl steht hier, damit die"
                  "\n     Entscheidung auf ihr getroffen werden kann, nicht ohne sie.")
        elif alle and not (set(gold_uuids) <= set(pr)):
            fehlen = len(set(gold_uuids) - set(pr))
            print(f"\nGolden-Median nicht berechnet: {fehlen} Gepinnte fehlen im "
                  f"letzten per-rec-iou-Lauf.")

    schlimm = sum(1 for name in ("golden", "test")
                  for e in bericht[name] if e["herkunft"] == "maschine")
    if schlimm:
        print(f"\n✗ {schlimm} maschinelle Label(s) im Massstab (golden/test).")
        return 1
    print("\n✓ Massstab frei von maschinellen Labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
