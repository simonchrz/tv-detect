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
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
BACKUP = Path.home() / "tv-labels-backup"

# Herkunfts-Regel: EINE Definition fuer alle drei Leser (s. label_herkunft.py).
_lh_spec = importlib.util.spec_from_file_location(
    "label_herkunft", Path(__file__).resolve().parent / "label_herkunft.py")
_lh = importlib.util.module_from_spec(_lh_spec)
_lh_spec.loader.exec_module(_lh)
_aus_spec = importlib.util.spec_from_file_location(
    "test_ausschluss", Path(__file__).resolve().parent / "test_ausschluss.py")
_aus = importlib.util.module_from_spec(_aus_spec)
_aus_spec.loader.exec_module(_aus)
NICHT_MENSCH = _lh.NICHT_MENSCH

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
    """Der juengste Eintrag aus per-rec-iou.jsonl (beide Spalten, s. produktionskopf)."""
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


def produktionskopf(lauf):
    """Die Spalte des Kopfes, der NACH diesem Lauf in Produktion ist.

    ⚠️ Bis 2026-09-15 stand hier fest `champion`. Der Champion ist aber der
    Kopf VOR dem Lauf -- jede Trendzeile trug damit den Kopf der Vornacht
    unter dem Zeitstempel dieser Nacht. Aufgefallen, weil die Zeile vom
    15.09. alle=0.9639 meldete, waehrend der in derselben Nacht deployte
    Kopf 0.957 hatte; 0.9639 war der Kopf vom 14.09.

    Deployt der Lauf, ist `candidate` der neue Produktionskopf. Lehnt das
    Gate ab, bleibt `champion` in Produktion. Eine abgelehnte Nacht misst
    damit denselben Kopf wie die Nacht davor -- das ist richtig so: die
    Frage lautet, was der Massstab ueber das Modell sagt, das tatsaechlich
    laeuft.
    """
    return "candidate" if lauf.get("deploy") else "champion"


def trend_schreiben(pfad, ts, bericht, alle, menschlich):
    """Eine Zeile je Lauf: der Massstab neben sich selbst.

    ⚠️ Die interessante Zahl ist nicht der Abstand an EINEM Abend, sondern
    ob er waechst. Waechst er, entfernt sich der Gate-Boden von dem, was
    ein Mensch je bestaetigt hat -- und das faellt ohne Spur niemandem auf.
    """
    pfad = Path(pfad)
    if pfad.exists():
        for zeile in reversed(pfad.read_text().splitlines()):
            if not zeile.strip():
                continue
            try:
                if json.loads(zeile).get("ts") == ts:
                    print(f"\nTrend: {ts} steht schon drin — nicht doppelt gezaehlt.")
                    return
            except Exception:
                pass
            break
    z = {k: {n: sum(1 for e in bericht[k] if e["herkunft"] == n)
             for n in ("mensch", "maschine", "unbekannt")}
         for k in bericht}
    eintrag = {
        "ts": ts,
        "golden_median_alle": round(st.median(alle), 4),
        "golden_median_belegt": round(st.median(menschlich), 4),
        "abstand": round(st.median(alle) - st.median(menschlich), 4),
        "n_alle": len(alle),
        "n_belegt": len(menschlich),
        "eimer": z,
    }
    with pfad.open("a") as f:
        f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
    print(f"\nTrend fortgeschrieben: {pfad.name} "
          f"(alle {eintrag['golden_median_alle']:.4f} / belegt "
          f"{eintrag['golden_median_belegt']:.4f}, Abstand {eintrag['abstand']:+.4f})")


N_SOLL = 10          # Naechte, docs/o23-massstab-belegt-preregistration.md
SCHWELLE = 0.010     # Aenderung des Abstands zwischen erster und zweiter Haelfte


def auswerten(pfad):
    """O23: driftet der Abstand, oder ist er ein konstanter Versatz?

    ⚠️ Beurteilt werden die ERSTEN 10 Naechte, nicht die letzten. Ein
    Urteil, das sich mit jeder weiteren Nacht verschiebt, ist keins --
    dieselbe Begruendung wie der n_soll-Stopp in audit-preregistration.py.

    ⚠️ Vor der zehnten Nacht gibt es KEINEN Zwischenstand. Wer den Abstand
    nach drei Naechten anschaut, hat die Regel schon gelesen und die
    Zahlen dazu -- ab da ist nicht mehr trennbar, was Regel und was
    Wunsch war.
    """
    if not pfad.exists():
        print("O23: noch keine Zeile im Massstab-Trend.")
        return 0
    zeilen = []
    for z in pfad.read_text().splitlines():
        if not z.strip():
            continue
        try:
            zeilen.append(json.loads(z))
        except Exception:
            pass
    if len(zeilen) < N_SOLL:
        print(f"O23 (Massstab belegt gegen alle): {len(zeilen)}/{N_SOLL} "
              f"Naechte — kein Zwischenstand, so registriert.")
        return 0

    erste, zweite = zeilen[:5], zeilen[5:N_SOLL]
    a1 = st.median([e.get("abstand", 0.0) for e in erste])
    a2 = st.median([e.get("abstand", 0.0) for e in zweite])
    d = a2 - a1
    print("\n" + "=" * 68)
    print("O23 — Misst der belegbare Massstab etwas anderes als der ganze?")
    print("  Registrierung: o23-massstab-belegt-preregistration.md")
    print("=" * 68)
    print(f"  Abstand, Naechte 1-5   Median {a1:+.4f}  ({erste[0].get('ts')} …)")
    print(f"  Abstand, Naechte 6-10  Median {a2:+.4f}  (… {zweite[-1].get('ts')})")
    print(f"  Aenderung              {d:+.4f}   Schwelle {SCHWELLE:.3f}")
    if abs(d) >= SCHWELLE:
        print("\n  → REGEL ERFUELLT: der Abstand DRIFTET. Der Gate-Boden "
              "entfernt sich von dem,\n     was ein Mensch bestaetigt hat. "
              "Konsequenz laut Registrierung: Umstellung\n     des Massstabs "
              "auf die belegte Teilmenge — als eigener, registrierter\n"
              "     Schritt mit Dual-Zeile im Trend.")
        return 1
    print("\n  → REGEL NICHT ERFUELLT: konstanter Versatz. Der volle Satz "
          "bleibt\n     Gate-Grundlage; der Abstand wird weiter berichtet, "
          "begruendet aber\n     keine Umstellung.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="maschinenlesbar")
    ap.add_argument("--nur-zaehlen", action="store_true",
                    help="keine Einzelzeilen, nur die Tabelle")
    ap.add_argument("--trend", nargs="?", const=str(ARCHIV / "massstab-trend.jsonl"),
                    default=None, metavar="PFAD",
                    help="eine Zeile je Lauf anhaengen: beide Golden-Mediane "
                         "nebeneinander. Doppelte ts werden uebersprungen, das "
                         "Skript darf also mehrfach laufen.")
    ap.add_argument("--auswerten", action="store_true",
                    help="O23 nach der registrierten Regel beurteilen, sobald "
                         "10 Naechte vorliegen. Vorher KEIN Zwischenstand.")
    args = ap.parse_args()

    if args.auswerten:
        return auswerten(Path(args.trend or (ARCHIV / "massstab-trend.jsonl")))

    ledger = json.loads((ARCHIV / "split-ledger.json").read_text())
    golden = json.loads((ARCHIV / "golden-eval-set.json").read_text())
    gold_uuids = list(golden.get("uuids") or [])
    meta = archiv_meta()

    def eimer(u):
        return "golden" if u in set(gold_uuids) else ledger.get(u, "—")

    # Golden ist auf test gepinnt; wir berichten es als eigenen Eimer, damit
    # die Zahlen sich nicht doppelt zaehlen.
    # Ausschlussliste + Quarantaene: nie Messziel, also auch hier nicht zaehlen.
    _raus = _aus.ausgeschlossen(ARCHIV)
    _n_raus = sum(1 for u, b in ledger.items() if b == "test" and u in _raus)
    if _n_raus:
        print(f"  ({_n_raus} Ledger-Test-Aufnahmen ausgeschlossen/quarantaeniert — nicht gezaehlt)")
    gruppen = {"golden": list(gold_uuids),
               "test": [u for u, b in ledger.items()
                        if b == "test" and u not in set(gold_uuids)
                        and u not in _raus],
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
        _spalte = produktionskopf(lauf)
        pr = lauf.get(_spalte) or {}
        alle = [pr[u] for u in gold_uuids if u in pr]
        menschlich = [pr[e["uuid"]] for e in bericht["golden"]
                      if e["herkunft"] == "mensch" and e["uuid"] in pr]
        if alle and menschlich and len(alle) != len(menschlich):
            print(f"\nGolden-Median (Lauf {lauf.get('ts')}, Kopf: {_spalte}"
                  f"{' = neu deployt' if _spalte == 'candidate' else ' = Gate lehnte ab'}):")
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

        if args.trend and alle and menschlich:
            trend_schreiben(args.trend, lauf.get("ts"), bericht, alle, menschlich)
    elif args.trend:
        print("\nTrend NICHT fortgeschrieben: kein per-rec-iou-Lauf gefunden.")

    schlimm = sum(1 for name in ("golden", "test")
                  for e in bericht[name] if e["herkunft"] == "maschine")
    if schlimm:
        print(f"\n✗ {schlimm} maschinelle Label(s) im Massstab (golden/test).")
        return 1
    print("\n✓ Massstab frei von maschinellen Labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
