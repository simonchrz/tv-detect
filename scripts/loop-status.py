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
import subprocess
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


# Produktions-Standard, wenn die Sendung keine eigene Blocklaenge hat
# (detect-config min_block_s). Kuerzere Label-Bloecke kann der Decoder nie
# bilden — sie koennen nur verfehlt werden.
MIN_BLOCK_S = 60
SCHWANZ_N = 5


def golden_schwanz(eintraege, golden, meta_von, naechte=None,
                   min_block_s=MIN_BLOCK_S, n=SCHWANZ_N):
    """Die schlechtesten Golden-Aufnahmen der letzten Nacht und warum.

    Warum das ueberhaupt: Golden-Median 0.96 bei Mittelwert 0.92 heisst,
    ein paar Aufnahmen tragen den ganzen Verlust — und bis 2026-09-02 war
    nicht zu sehen, welche. Die Nightly schreibt je Aufnahme nur den
    Testsatz-Wert (per-rec-iou.jsonl); Golden ist aber eine Teilmenge des
    Testsatzes, also steht alles schon da.

    Was die Zahl misst (block_iou in train-head.py): Mittel ueber die
    LABEL-Bloecke, jeder mit seinem bestueberlappenden Vorhersageblock.
    Zwei Folgen, die man kennen muss, bevor man an den Kopf denkt:
      * ein Label-Block unter min_block_s ist UNERREICHBAR — der Decoder
        bildet ihn nie, er zaehlt trotzdem 1/n der Note (kabel eins
        Mein Lokal: 32 s am Aufnahmeende = 0.64 statt 0.96);
      * Vorhersageblocke ohne Label-Gegenstueck kosten NICHTS.
    Und ein Mitglied mit which=auto misst den Abstand zum Kopf von damals,
    nicht zur Wahrheit.

    eintraege: per-rec-iou.jsonl-Zeilen (aelteste zuerst). Gezaehlt wird
    der Wert, den die PRODUKTION traegt: candidate bei deploy, sonst
    champion. meta_von(uuid) -> dict(title, which, ads) oder None.
    Liefert dict(nacht, schlechteste, beharrlich, n_naechte) oder None."""
    def prod(e):
        quelle = e.get("candidate") if e.get("deploy") else e.get("champion")
        return {u: v for u, v in (quelle or {}).items() if u in golden}

    reihe = [prod(e) for e in eintraege]
    reihe = [r for r in reihe if len(r) >= max(1, len(golden) // 2)]
    if naechte:
        reihe = reihe[-naechte:]
    if not reihe:
        return None
    heute = reihe[-1]
    im_schwanz = defaultdict(int)
    for r in reihe:
        for u, _ in sorted(r.items(), key=lambda kv: kv[1])[:n]:
            im_schwanz[u] += 1

    schlechteste = []
    for u, v in sorted(heute.items(), key=lambda kv: kv[1])[:n]:
        m = meta_von(u) or {}
        ads = m.get("ads") or []
        kurz = [(a, b) for a, b in ads if (b - a) < min_block_s]
        # Was die Note hoechstens erreichen kann, wenn jeder erreichbare
        # Block perfekt sitzt: die kurzen zaehlen als 0.
        decke = (len(ads) - len(kurz)) / len(ads) if ads else 1.0
        schlechteste.append({
            "uuid": u, "iou": v, "title": m.get("title") or "",
            "echo": m.get("which") == "auto",
            "unerreichbar": kurz, "n_bloecke": len(ads), "decke": decke,
            "im_schwanz": im_schwanz.get(u, 0)})
    beharrlich = sorted(((u, k) for u, k in im_schwanz.items()
                         if k >= max(2, len(reihe) // 2)),
                        key=lambda t: -t[1])
    return {"nacht": eintraege[-1].get("ts", "")[:8] if eintraege else "",
            "schlechteste": schlechteste, "beharrlich": beharrlich,
            "n_naechte": len(reihe)}


def archiv_meta(archiv):
    """meta_von() ueber das Trainingsarchiv; ohne numpy leer, nicht tot."""
    try:
        import numpy as np
    except ImportError:
        return lambda u: None

    def lies(u):
        p = archiv / f"{u}.npz"
        if not p.exists():
            return None
        try:
            return json.loads(str(np.load(p, allow_pickle=True)["meta"]))
        except Exception:
            return None
    return lies


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
        # ⚠️ "kein Ergebnis im Log" heisst NICHT "abgebrochen" — derselbe
        # Fehlalarm wie im Tagesbericht (behoben 2026-08-16): ein langer
        # Lauf steht um 08:07 noch mitten in Extraktion oder Shadow-Eval.
        ausgang = ("läuft noch" if _laeuft_noch()
                   else "unklar (Lauf abgebrochen?)")
        return {"ts": ts, "ausgang": ausgang,
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


def _laeuft_noch():
    """Läuft gerade ein train-head? Der Sensor läuft auf demselben Mac."""
    try:
        return subprocess.run(["pgrep", "-f", "train-head.py"],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


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
            # select_rule gehoert ebenfalls in den Filter (Seed-Ensemble seit
            # 2026-08-27): golden_bestwert() im Nightly filtert darauf, und
            # ohne den Filter meldete dieser Sensor am 2026-08-28 einen Boden
            # (0.9557, alte Regel) samt O3-Sperre, waehrend das Gate "noch
            # kein Bestwert" sagte.
            regel_jetzt = gt[-1].get("select_rule") or "median-seed"
            passend = [e for e in deployt
                       if e.get("set_hash") == hash_jetzt
                       and e.get("decoder") == dec_jetzt
                       and e.get("label_hash") == lab_jetzt
                       and (e.get("select_rule") or "median-seed") == regel_jetzt
                       and e.get("ts") != ts_jetzt]
            if not passend:
                print(f"  Boden: noch kein Bestwert fuer diesen Satz/Decoder "
                      f"unter Auswahl {regel_jetzt} — baut sich neu auf, "
                      f"bis dahin schuetzt nur das Head-to-Head.")
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
                      f"zweitbester von {len(sortiert)} Tagen, "
                      f"Auswahl {regel_jetzt}), "
                      f"Champion {champ['golden_median']} "
                      f"({champ['ts'][:8]})")
                if champ["golden_median"] < best["golden_median"]:
                    print("  ⚠ Der Boden liegt ÜBER dem Champion — es kommt "
                          "nur noch durch, wer den Champion schlägt (O3).")

    # ── Golden-Schwanz: wer traegt den Verlust, und war er erreichbar? ──
    gpfad = args.archiv / "golden-eval-set.json"
    if gpfad.exists():
        golden = set(json.loads(gpfad.read_text()).get("uuids", []))
        sch = golden_schwanz(zeilen(args.archiv / "per-rec-iou.jsonl"),
                             golden, archiv_meta(args.archiv),
                             naechte=max(args.naechte, 14))
        if sch:
            print(f"\nGolden-Schwanz ({sch['nacht']}, die {SCHWANZ_N} "
                  f"schlechtesten von {len(golden)}):")
            for r in sch["schlechteste"]:
                grund = []
                if r["echo"]:
                    grund.append("which=auto: misst Echo, nicht Wahrheit")
                if r["unerreichbar"]:
                    kurz = ", ".join(f"{b - a:.0f}s@{a:.0f}"
                                     for a, b in r["unerreichbar"])
                    grund.append(f"Decke {r['decke']:.2f}: "
                                 f"{len(r['unerreichbar'])} Label-Block "
                                 f"unter {MIN_BLOCK_S}s ({kurz})")
                print(f"  {r['iou']:.3f}  {r['uuid']:32} "
                      f"{r['title'][:26]:26}  "
                      f"{r['im_schwanz']:2d}/{sch['n_naechte']} Naechte"
                      + (f"\n         {'; '.join(grund)}" if grund else ""))
            if sch["beharrlich"]:
                print(f"  beharrlich (>= halbe Serie im Schwanz): "
                      + ", ".join(f"{u} ({k})" for u, k in sch["beharrlich"]))
                print("  → dieselben Aufnahmen jede Nacht = Sendungs- oder "
                      "Label-Sache, nicht der Kopf.")

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
