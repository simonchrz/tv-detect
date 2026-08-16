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
    # ⚠️ "kein Ergebnis im Log" heisst NICHT "abgebrochen". Am 2026-08-16
    # meldete der Bericht um 07:30 einen abgebrochenen Lauf, waehrend
    # train-head seit 03:40 durchgehend Merkmale extrahierte (53 neue
    # Aufnahmen, Mac zu 95 % ausgelastet). Ein Bericht, der bei jedem langen
    # Lauf Alarm schlaegt, wird nach zwei Tagen weggeklickt — und dann faellt
    # der ECHTE Abbruch auch nicht mehr auf.
    laeuft = _laeuft_noch()
    if m:
        ausgang = m.group(1)
    elif laeuft:
        ausgang = "läuft noch"
    else:
        ausgang = "unklar (Lauf abgebrochen?)"
    return {
        "ts": ts,
        "ausgang": ausgang,
        "laeuft": laeuft,
        "boden": boden.group(1).strip() if boden else "",
    }


def _laeuft_noch():
    """Laeuft gerade ein train-head? Der Bericht laeuft auf demselben Mac."""
    try:
        return subprocess.run(["pgrep", "-f", "train-head.py"],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def boden_und_champion(gt):
    """Muss dieselbe Rechnung sein wie golden_bestwert() im Trainer."""
    deployt = [e for e in gt if e.get("deployed") and e.get("golden_median")]
    if not deployt:
        return None, None
    hash_jetzt, dec_jetzt = gt[-1].get("set_hash"), gt[-1].get("decoder")
    # ⚠️ label_hash gehört in denselben Filter wie set_hash und decoder.
    # `set_hash` sichert nur die ZUSAMMENSETZUNG; werden die LABELS eines
    # Mitglieds korrigiert, misst der Satz danach etwas anderes, ohne dass
    # eine Kennzahl es anzeigt. Am 2026-08-13 sprang Golden so 0.906 → 0.937
    # (87 Kanten-Korrekturen) und die Latte stieg mit — als hätte das Modell
    # zugelegt. Ab 2026-08-14 schneidet der Boden an jeder Label-Änderung.
    # Alte Zeilen tragen kein Feld (None) und fallen damit korrekt heraus.
    lab_jetzt = gt[-1].get("label_hash")
    # ⚠️ Die eigene Zeile zaehlt nicht — sonst ist die Latte bei einer
    # frischen Epoche der heutige Wert minus Toleranz, und jeder Lauf meldet
    # +0.0100 gegen sich selbst (beobachtet 2026-08-15).
    ts_jetzt = gt[-1].get("ts")
    passend = [e for e in deployt
               if e.get("set_hash") == hash_jetzt and e.get("decoder") == dec_jetzt
               and e.get("label_hash") == lab_jetzt and e.get("ts") != ts_jetzt]
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
        # ⚠️ Auch die noch nicht begonnenen Serien einsammeln. Sonst
        # verschweigt der Bericht genau die Fragen, die in der
        # Warteschlange stehen — und die Schleife sieht leerer aus als sie
        # ist.
        stand = re.search(r"→ (.+)$", block, flags=re.M) \
            or re.search(r"^\s*(Stand: .+)$", block, flags=re.M)
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
    warnungen = []   # Handlungsbedarf — gehört in die Push-Meldung
    chronisch = []   # bekannte Dauerzustände — nur in die Langfassung

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
            if best is None and gt[-1].get("label_hash"):
                # Sichtbar machen, statt die Latte stillschweigend wegfallen
                # zu lassen: nach einer Label-Korrektur gibt es schlicht noch
                # keine zweite vergleichbare Messung.
                zeile += " (keine Latte — Labels geändert, Reihe beginnt neu)"
                chronisch.append("Golden-Reihe nach Label-Änderung neu")
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
                # ⚠️ CHRONISCH, nicht Handlungsbedarf. Der Zustand steht seit
                # 08-09 und ist als O3 offen — täglich zu pushen wäre genau
                # die Meldung, die man nach drei Tagen wegwischt und dann
                # auch die echte übersieht.
                chronisch.append("Boden über Champion (O3)")
        kopf.append(zeile)
        if lauf["ausgang"] == "unklar (Lauf abgebrochen?)":
            warnungen.append("Lauf abgebrochen")
        elif lauf.get("laeuft"):
            # Kein Warnfall: ein laufender Nightly ist der Normalzustand,
            # nur eben noch ohne Ergebnis.
            chronisch.append("Nightly läuft noch")

    # ── Serien ──────────────────────────────────────────────────────
    for e in s:
        if "NOCH OFFEN" in e["stand"]:
            n = re.search(r"(\d+/\d+) gültige", e["stand"])
            kopf.append(f"{kurzTitel(e['frage'])}: Nacht {n.group(1) if n else '?'}"
                        + (f", Median {e['median']} ({e['negativ']} negativ)"
                           if e["median"] else "")
                        + " — läuft")
        elif "abgeschlossen am" in e["stand"]:
            # Entschieden UND verbucht — Information, kein Handlungsbedarf.
            kopf.append(f"{kurzTitel(e['frage'])}: {e['stand'][:70]}")
        elif "noch nicht begonnen" in e["stand"]:
            ab = re.search(r"(\d{8})", e["stand"])
            kopf.append(f"{kurzTitel(e['frage'])}: eingereiht"
                        + (f", ab {ab.group(1)}" if ab else ""))
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

    return kopf, warnungen, chronisch, a_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kurz", action="store_true",
                    help="Fassung für die Push-Meldung")
    args = ap.parse_args()

    kopf, warnungen, chronisch, a_text = baue()

    if args.kurz:
        # ⚠️ Eine Benachrichtigung ist KEIN Bericht — und iOS zeigt
        # zusammengeklappt nur ZWEI Zeilen. Der erste Entwurf schob den
        # ganzen Kopf hinein und wurde mitten im Satz gekappt, der zweite
        # hatte eine zu lange zweite Zeile. Deshalb: Zeile 1 sagt, ob jemand
        # hinsehen muss, Zeile 2 fasst den Rest in Stichworten. Alles
        # Weitere steht in der Langfassung.
        print("Handlung nötig: " + "; ".join(dict.fromkeys(warnungen))
              if warnungen else "Nichts zu tun.")

        teile = []
        for z in kopf:
            if z.startswith("Nightly"):
                # ⚠️ Nicht "alles was nicht DEPLOYED ist, ist REJECTED".
                # Genau das meldete die Kurzfassung am 2026-08-16 als
                # "REJECTED 0.935", waehrend der Lauf noch lief und die
                # Langfassung korrekt "läuft noch" sagte. Die Kurzfassung
                # geht als Push aufs Telefon — sie darf nicht das Gegenteil
                # der Langfassung behaupten.
                if "läuft noch" in z:
                    t = "Nightly läuft"
                elif "DEPLOYED" in z:
                    t = "Nightly DEPLOYED"
                elif "REJECTED" in z:
                    t = "Nightly REJECTED"
                else:
                    t = "Nightly unklar"
                g = re.search(r"Golden ([\d.]+)", z)
                if g:
                    t += f" {float(g.group(1)):.3f}"
                if "innerhalb der Seed-Streuung" in z:
                    t += " (Rauschen)"
                teile.append(t)
            elif " — läuft" in z:
                n = re.search(r"Nacht (\d+/\d+)", z)
                kuerzel = z.split(" ")[0]
                teile.append(f"{kuerzel} {n.group(1)}" if n else kuerzel)
            elif ": eingereiht" in z:
                teile.append(z.split(" ")[0] + " wartet")
        if teile:
            print(" · ".join(teile))
        return 0

    print("=" * 68)
    print("TAGESBERICHT — Verbesserungs-Schleife tv-detect")
    print("=" * 68)
    for z in kopf:
        print("  " + z)
    print()
    if warnungen:
        print("  ⚠ Handlung nötig: " + "; ".join(dict.fromkeys(warnungen)))
    else:
        print("  Nichts zu entscheiden. Das ist der Normalfall, kein Stillstand.")
    for c in dict.fromkeys(chronisch):
        print(f"  · bekannt und offen: {c}")
    print()
    print(a_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
