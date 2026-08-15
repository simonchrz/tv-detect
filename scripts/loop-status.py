#!/usr/bin/env python3
"""Zustand der Verbesserungs-Schleife in einem Aufruf.

Der Sensor, nicht der Fahrer: liest zusammen, was für die tägliche
Entscheidung nötig ist — was letzte Nacht passiert ist, wie die Serien
stehen, welche Frage gerade offen ist. Das Urteil bleibt draußen.

Warum als Skript und nicht als Blick ins Log: das Nightly-Protokoll ist
inzwischen über 4 MB Prosa, und wer daraus jede Nacht neu die Serie
zusammensucht, liest zwangsläufig mal die falsche Zelle. Am 2026-08-09 sind
so zwei zufällig gleiche Zahlen als Reproduzierbarkeit durchgegangen.

    loop-status.py [--naechte N] [--archiv PFAD] [--log PFAD]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
LOG = Path.home() / "Library/Logs/tv-train-head.log"
LEDGER = Path(__file__).resolve().parent.parent / "docs/experiment-ledger.md"


def zeilen(pfad, n=None):
    """jsonl lesen, defekte Zeilen überspringen statt den Aufruf zu killen."""
    if not pfad.exists():
        return []
    out = []
    for ln in pfad.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out[-n:] if n else out


def letzter_lauf(log_pfad):
    """Ausgang der letzten Nacht: DEPLOYED oder REJECTED, plus Begründung.

    Der Grund steht mehrzeilig hinter 'reason:' — die Fortsetzungszeilen
    sind eingerückt, das ist das Abbruchkriterium.
    """
    if not log_pfad.exists():
        return None
    text = log_pfad.read_text(errors="replace")
    # Läufe sind durch '=== <ts> ===' getrennt; nur den letzten ansehen.
    teile = re.split(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===$",
                     text, flags=re.M)
    if len(teile) < 3:
        return None
    ts, block = teile[-2], teile[-1]
    m = re.search(r"^(DEPLOYED|REJECTED)\b.*$", block, flags=re.M)
    if not m:
        return {"ts": ts, "ausgang": "unklar (Lauf abgebrochen?)",
                "grund": block.strip().splitlines()[-1:] or [""]}
    grund = []
    for ln in block[m.end():].splitlines():
        if ln.startswith("  reason:") or grund:
            if grund and not ln.startswith("    "):
                break
            grund.append(ln.strip())
    boden = re.search(r"^\s*(Golden-Boden:.*|⚠ Golden-Boden blockt.*)$",
                      block, flags=re.M)
    return {"ts": ts, "ausgang": m.group(1),
            "grund": " ".join(grund).replace("reason: ", ""),
            "boden": boden.group(1).strip() if boden else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naechte", type=int, default=7)
    ap.add_argument("--archiv", type=Path, default=ARCHIV)
    ap.add_argument("--log", type=Path, default=LOG)
    args = ap.parse_args()

    print("=" * 68)
    print("SCHLEIFEN-STATUS")
    print("=" * 68)

    lauf = letzter_lauf(args.log)
    if lauf:
        print(f"\nLetzte Nacht ({lauf['ts']}): {lauf['ausgang']}")
        if lauf.get("boden"):
            print(f"  {lauf['boden']}")
        if lauf.get("grund"):
            print(f"  {lauf['grund'][:400]}")
    else:
        print("\nLetzte Nacht: kein auswertbarer Lauf im Log")

    # ── Golden-Verlauf des jeweils gebauten Kandidaten ───────────────
    gt = zeilen(args.archiv / "golden-trend.jsonl", args.naechte)
    if gt:
        haeufig = defaultdict(int)
        for e in gt:
            haeufig[(e.get("set_hash"), e.get("decoder"))] += 1
        print(f"\nGolden-Verlauf (letzte {len(gt)}):")
        for e in gt:
            mark = "deployed" if e.get("deployed") else "REJECTED"
            # ⚠️ Ab 20260810 liefert der Nightly den MITTLEREN von drei Seeds
            # aus, davor immer Seed 0. Die Zahl misst seither etwas anderes.
            # seed_golden["0"] ist der Wert nach der alten Regel — nur so ist
            # die Reihe ueber den Bruch hinweg lesbar.
            alt_regel = (e.get("seed_golden") or {}).get("0")
            if alt_regel is not None and e.get("seed") not in (0, None):
                mark += f"  [alte Regel (Seed 0): {alt_regel}]"
            warn = ""
            if len(haeufig) > 1:
                warn = f"  [set={e.get('set_hash')} dec={e.get('decoder')}]"
            print(f"  {e['ts'][:8]}  median {e.get('golden_median')}  "
                  f"mean {e.get('golden_mean')}  n={e.get('n')}  "
                  f"{mark}{warn}")
        if len(haeufig) > 1:
            print("  ⚠ Nicht alle Zeilen teilen Satz UND Decoder — die "
                  "Mediane sind NICHT direkt vergleichbar.")
        deployt = [e for e in zeilen(args.archiv / "golden-trend.jsonl")
                   if e.get("deployed") and e.get("golden_median")]
        if deployt:
            hash_jetzt = gt[-1].get("set_hash")
            dec_jetzt = gt[-1].get("decoder")
            # label_hash gehoert in denselben Filter: set_hash sichert nur
            # die Zusammensetzung, nicht die Labels. Wer die Labels eines
            # Mitglieds korrigiert, misst danach etwas anderes.
            lab_jetzt = gt[-1].get("label_hash")
            # Die eigene Zeile zaehlt nicht: der Trend wird VOR dem Gate
            # geschrieben, sonst ist der Boden bei einer frischen Epoche der
            # heutige Wert selbst ("zweitbester von 1 Tagen").
            ts_jetzt = gt[-1].get("ts")
            passend = [e for e in deployt
                       if e.get("set_hash") == hash_jetzt
                       and e.get("decoder") == dec_jetzt
                       and e.get("label_hash") == lab_jetzt
                       and e.get("ts") != ts_jetzt]
            if passend:
                # MUSS dieselbe Rechnung sein wie golden_bestwert() in
                # train-head.py: hoechster Wert, der mindestens zweimal
                # erreicht wurde, hoechstens ein Wert je Kalendertag. Ein
                # Sensor, der anders rechnet als das Gate, meldet Sperren, die
                # es nicht gibt — oder verschweigt welche, die es gibt.
                je_tag = {}
                for e in passend:
                    je_tag[e["ts"][:8]] = e
                sortiert = sorted(je_tag.values(),
                                  key=lambda e: e["golden_median"],
                                  reverse=True)
                best = sortiert[1] if len(sortiert) >= 3 else sortiert[0]
                champ = passend[-1]
                print(f"  Boden {best['golden_median']} ({best['ts'][:8]}, "
                      f"zweitbester von {len(sortiert)} Tagen), "
                      f"Champion {champ['golden_median']} "
                      f"({champ['ts'][:8]})")
                if champ["golden_median"] < best["golden_median"]:
                    print("  ⚠ Der Boden liegt ÜBER dem Champion — es kommt "
                          "nur noch durch, wer den Champion schlägt (O3).")

    # ── Schatten-Serie je Variante ───────────────────────────────────
    st = zeilen(args.archiv / "shadow-trend.jsonl")
    if not st:
        print("\nSchatten-Serie: noch keine Zeilen "
              "(shadow-trend.jsonl entsteht im nächsten Nightly).")
    else:
        nach_ts = defaultdict(dict)
        for e in st:
            nach_ts[e["ts"]][e["arch"]] = e
        stempel = sorted(nach_ts)[-args.naechte:]
        archs = []
        for t in stempel:
            for a in nach_ts[t]:
                if a not in archs:
                    archs.append(a)
        print(f"\nSchatten-Serie, Golden-Median ({len(stempel)} Läufe):")
        kopf = "  ".join(t[4:8] for t in stempel)
        print(f"  {'arch':32s}  {kopf}")
        for a in archs:
            werte = []
            for t in stempel:
                e = nach_ts[t].get(a)
                g = e.get("golden_median") if e else None
                werte.append(f"{g:.3f}" if g is not None else "  —  ")
            print(f"  {a:32s}  {'  '.join(werte)}")
        # Die eine Zahl, um die es bei O1 geht.
        paare = [(nach_ts[t].get("mlp32-cwt-mp"), nach_ts[t].get("mlp32-ct-mp"))
                 for t in stempel]
        deltas = [round(m["golden_median"] - o["golden_median"], 4)
                  for m, o in paare
                  if m and o and m.get("golden_median") is not None
                  and o.get("golden_median") is not None]
        if deltas:
            neg = sum(1 for d in deltas if d < 0)
            print(f"\n  O1 Whisper-Beitrag (mit − ohne), Golden: "
                  f"{', '.join(f'{d:+.3f}' for d in deltas)}")
            print(f"     {neg}/{len(deltas)} Läufe negativ, "
                  f"Median {sorted(deltas)[len(deltas)//2]:+.3f}")

    # ── Versiegelter Satz ────────────────────────────────────────────
    # Nur die Groesse, nie ein Ergebnis: ihn taeglich auszuwerten waere
    # genau die Nutzung, die ihn wertlos macht.
    sl = args.archiv / "split-ledger.json"
    if sl.exists():
        try:
            eimer = json.loads(sl.read_text())
            n_v = sum(1 for v in eimer.values() if v == "versiegelt")
            print(f"\nVersiegelter Satz: {n_v} Aufnahmen "
                  f"(von {len(eimer)} im Ledger)")
            if n_v < 30:
                print("  wächst noch — vor ~30 Aufnahmen nicht öffnen, "
                      "ein Median über weniger sagt nichts.")
        except Exception as e:
            print(f"\nVersiegelter Satz: Ledger nicht lesbar ({e})")

    # ── Offene Fragen aus dem Ledger ─────────────────────────────────
    if LEDGER.exists():
        # Der Status darf umbrechen — ohne DOTALL fiel genau die Frage
        # durchs Raster, die gerade läuft.
        offen = re.findall(r"^### (O\d+ — .+?)$\s*\*Status: (.+?)\*",
                           LEDGER.read_text(), flags=re.M | re.S)
        if offen:
            print("\nOffene Fragen (docs/experiment-ledger.md):")
            for titel, status in offen:
                status = " ".join(status.split())
                print(f"  {titel}\n      Status: {status}")
    print("\nVor jedem Vorschlag: Friedhof in docs/experiment-ledger.md §4.")


if __name__ == "__main__":
    main()
