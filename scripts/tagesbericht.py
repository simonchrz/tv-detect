#!/usr/bin/env python3
"""Der tägliche Statusbericht der Verbesserungs-Schleife.

Nicht dasselbe wie `loop-status.py`. Der Sensor sammelt ALLES, was für eine
Entscheidung nötig ist — dieser Bericht sagt in fünf Zeilen, ob du hinsehen
musst. Er ist die Antwort auf „ich will nicht jeden Morgen fragen, wie das
Training lief".

    tagesbericht.py [--kurz]      # --kurz = die Fassung für die Push-Meldung

⚠️ Zwei Dinge macht dieser Bericht bewusst NICHT:

* Er deutet keine Zwischenstände. Eine laufende Serie bekommt „Nacht 3/5"
  und ihren Median, aber kein Urteil — die Regel entscheidet am Ende, und
  zwar im Audit, nicht hier. Ein Bericht, der täglich einen Trend erzählt,
  erzieht den Leser dazu, Rauschen für Fortschritt zu halten.
* Er meldet keinen Erfolg, den er nicht belegen kann. „DEPLOYED" heißt, dass
  ein Kopf durchs Gate kam — nicht, dass er besser ist. Liegt der Wert
  innerhalb der Seed-Streuung, steht das dabei.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
LOG = Path.home() / "Library/Logs/tv-train-head.log"
REPO = Path(__file__).resolve().parent.parent

# Aus dem Seed-Sweep vom 2026-08-11 (12 Fits, identische Daten): Std 0.0059.
# Alles, was näher als das am Vortag liegt, ist keine Nachricht.
SEED_STD = 0.006


def zeilen(pfad):
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
    return out


def letzter_lauf():
    """Ausgang der letzten Nacht aus dem Nightly-Protokoll."""
    if not LOG.exists():
        return None
    text = LOG.read_text(errors="replace")
    teile = re.split(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===$",
                     text, flags=re.M)
    if len(teile) < 3:
        return None
    ts, block = teile[-2], teile[-1]
    m = re.search(r"^(DEPLOYED|REJECTED)\b", block, flags=re.M)
    boden = re.search(r"^\s*(Golden-Boden:.*|⚠ Golden-Boden blockt.*)$", block, flags=re.M)
    return {
        "ts": ts,
        "ausgang": m.group(1) if m else "unklar (Lauf abgebrochen?)",
        "boden": boden.group(1).strip() if boden else "",
    }


def boden_und_champion(gt):
    """Muss dieselbe Rechnung sein wie golden_bestwert() im Trainer."""
    deployt = [e for e in gt if e.get("deployed") and e.get("golden_median")]
    if not deployt:
        return None, None
    hash_jetzt, dec_jetzt = gt[-1].get("set_hash"), gt[-1].get("decoder")
    passend = [e for e in deployt
               if e.get("set_hash") == hash_jetzt and e.get("decoder") == dec_jetzt]
    if not passend:
        return None, None
    je_tag = {}
    for e in passend:
        je_tag[e["ts"][:8]] = e
    sortiert = sorted(je_tag.values(), key=lambda e: e["golden_median"], reverse=True)
    best = sortiert[1] if len(sortiert) >= 3 else sortiert[0]
    return best, passend[-1]


def audit():
    """Das Audit ist die Urteilsinstanz — hier wird es nur zitiert."""
    try:
        p = subprocess.run([sys.executable, str(REPO / "scripts/audit-preregistration.py")],
                           capture_output=True, text=True, timeout=120)
        return p.stdout, p.returncode
    except Exception as e:
        return f"(Audit nicht lauffähig: {e})", 1


def serien(audit_text):
    """Die Serien-Stände aus der Audit-Ausgabe ziehen.

    ⚠️ NICHT an den Trennlinien zerlegen: Titel und Urteil stehen dann in
    verschiedenen Stücken, und der Bericht verschweigt genau die Zeile, für
    die er gebaut wurde. Stattdessen je Titel bis zum nächsten Titel lesen.
    """
    treffer = list(re.finditer(r"^(O\d+ — .+)$", audit_text, flags=re.M))
    aus = []
    for i, t in enumerate(treffer):
        ende = treffer[i + 1].start() if i + 1 < len(treffer) else len(audit_text)
        block = audit_text[t.end():ende]
        stand = re.search(r"→ (.+)$", block, flags=re.M)
        med = re.search(r"Median\s+([-+]?[\d.]+)\s+negativ\s+(\d+/\d+)", block)
        if not stand:
            continue
        aus.append({
            "frage": t.group(1).strip(),
            "stand": " ".join(stand.group(1).split()),
            "median": med.group(1) if med else None,
            "negativ": med.group(2) if med else None,
        })
    return aus


def kurzTitel(frage, breite=32):
    """O1 — lange Frage…  →  O1 (an der Wortgrenze gekuerzt)."""
    if len(frage) <= breite:
        return frage
    schnitt = frage[:breite].rsplit(" ", 1)[0]
    return schnitt + "…"


def baue():
    gt = zeilen(ARCHIV / "golden-trend.jsonl")
    lauf = letzter_lauf()
    a_text, a_rc = audit()
    s = serien(a_text)

    kopf = []
    warnungen = []

    # ── Letzte Nacht ────────────────────────────────────────────────
    if not lauf:
        kopf.append("Nightly: KEIN auswertbarer Lauf im Protokoll.")
        warnungen.append("kein Lauf")
    else:
        zeile = f"Nightly {lauf['ts'][:10]}: {lauf['ausgang']}"
        if gt:
            e = gt[-1]
            med = e.get("golden_median")
            zeile += f", Golden {med}"
            best, champ = boden_und_champion(gt)
            if best and med is not None:
                latte = round(best["golden_median"] - 0.010, 4)
                abstand = round(med - latte, 4)
                zeile += f" (Latte {latte}, {abstand:+.4f})"
                # ⚠️ Der ehrliche Zusatz: liegt der Abstand innerhalb der
                # Seed-Streuung, ist das Durchkommen keine Aussage über das
                # Modell. Ohne diesen Satz liest sich jedes DEPLOYED wie ein
                # Fortschritt.
                if lauf["ausgang"] == "DEPLOYED" and abs(abstand) < SEED_STD:
                    zeile += " — innerhalb der Seed-Streuung, kein Beleg für Verbesserung"
            if champ and best and champ["golden_median"] < best["golden_median"]:
                warnungen.append("Boden über Champion (O3)")
        kopf.append(zeile)
        if lauf["ausgang"] == "unklar (Lauf abgebrochen?)":
            warnungen.append("Lauf abgebrochen")

    # ── Serien ──────────────────────────────────────────────────────
    for e in s:
        if "NOCH OFFEN" in e["stand"]:
            n = re.search(r"(\d+/\d+) gültige", e["stand"])
            kopf.append(f"{kurzTitel(e['frage'])}: Nacht {n.group(1) if n else '?'}"
                        + (f", Median {e['median']} ({e['negativ']} negativ)"
                           if e["median"] else "")
                        + " — läuft")
        else:
            kopf.append(f"{kurzTitel(e['frage'])}: {e['stand'][:90]}")
            warnungen.append("Serie entschieden — Urteil steht an")

    if a_rc != 0:
        warnungen.append("Audit meldet Exit 1")

    # ── Versiegelter Satz ───────────────────────────────────────────
    sl = ARCHIV / "split-ledger.json"
    if sl.exists():
        try:
            eimer = json.loads(sl.read_text())
            n_v = sum(1 for v in eimer.values() if v == "versiegelt")
            kopf.append(f"Versiegelt: {n_v} Aufnahmen (ab ~30 auswertbar)")
        except Exception:
            pass

    return kopf, warnungen, a_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kurz", action="store_true",
                    help="Fassung für die Push-Meldung")
    args = ap.parse_args()

    kopf, warnungen, a_text = baue()

    if args.kurz:
        # Die Push-Meldung führt mit dem, was zu tun ist — nicht mit Zahlen.
        if warnungen:
            print("⚠ " + "; ".join(dict.fromkeys(warnungen)))
        else:
            print("nichts zu tun")
        for z in kopf[:3]:
            print(z)
        return 0

    print("=" * 68)
    print("TAGESBERICHT — Verbesserungs-Schleife tv-detect")
    print("=" * 68)
    for z in kopf:
        print("  " + z)
    print()
    if warnungen:
        print("  ⚠ " + "; ".join(dict.fromkeys(warnungen)))
    else:
        print("  Nichts zu entscheiden. Das ist der Normalfall, kein Stillstand.")
    print()
    print(a_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
