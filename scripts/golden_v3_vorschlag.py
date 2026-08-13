#!/usr/bin/env python3
"""Kandidaten für einen größeren Golden-Satz (v3) — VORSCHLAG, keine Aktivierung.

Warum: §2 im Ledger — ein Gutteil des Golden-Rauschens ist
Kleine-Stichprobe-Rauschen (38 Aufnahmen; der Testsatz mit ~98 streut halb
so breit). Ein größerer Satz macht jede künftige Serie schärfer oder kürzer.

⚠️ Was dieses Skript bewusst NICHT tut:
  * golden-eval-set.json anfassen. O1s Regel pinnt set_hash c8727e8266a8 —
    eine Aktivierung vor dem O1-Urteil (14.08.) würde dessen letzte Nacht
    ungültig machen. Aktivierung ist ein eigener, expliziter Schritt.
  * Aufnahmen nach IoU auswählen. Wer den Satz aus "gut erkannten"
    Aufnahmen baut, baut sich einen Schmeichel-Satz — die Auswahl ist
    blind gegenüber jeder Modellleistung: nur Review-Status, Split-Eimer
    und Kanal-Verteilung.

Auswahlkriterien, in dieser Reihenfolge:
  1. Im TEST-Eimer des Split-Ledgers (Golden ⊆ Test hält die
     Leakage-Freiheit; der Eimer ist sticky).
  2. Von einem MENSCHEN reviewt — ads_user.json OHNE auto_confirmed_at.
     Auto-Confirm ist die Maschine, die sich selbst bestätigt.
  3. Nicht schon im v2-Satz, nicht versiegelt.
  4. Kanal-stratifiziert: aufgefüllt wird proportional zum Test-Eimer,
     damit der Satz nicht die Kanäle überrepräsentiert, die zufällig
     viele Reviews haben.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
BACKUP = Path.home() / "tv-labels-backup"


def main():
    ledger = json.loads((ARCHIV / "split-ledger.json").read_text())
    golden = json.loads((ARCHIV / "golden-eval-set.json").read_text())
    v2 = set(golden.get("uuids") or [])

    test = {u for u, b in ledger.items() if b == "test"}
    versiegelt = {u for u, b in ledger.items() if b == "versiegelt"}

    # Menschlich reviewt: ads_user.json ohne auto_confirmed_at, aus dem
    # Label-Backup (die Quelle, die auch review-effort.py benutzt).
    mensch = set()
    slugs = {}
    # Layout: ~/tv-labels-backup/_rec_<uuid>/ads_user.json — die dirs
    # tragen das _rec_-Praefix, der Ledger die nackte uuid.
    for d in BACKUP.iterdir() if BACKUP.exists() else []:
        f = d / "ads_user.json"
        if not f.is_file() or not d.name.startswith("_rec_"):
            continue
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        if j.get("auto_confirmed_at"):
            continue
        uuid = d.name[len("_rec_"):]
        mensch.add(uuid)

    def slug_von(uuid):
        # dvr-<slug>-<unix> — der Slug steht in der uuid selbst.
        teile = uuid.split("-")
        return "-".join(teile[1:-1]) if len(teile) >= 3 else "?"

    kandidaten = sorted((test & mensch) - v2 - versiegelt)
    print(f"v2-Satz: {len(v2)}  |  Test-Eimer: {len(test)}  |  "
          f"menschlich reviewt: {len(mensch)}")
    print(f"Kandidaten (Test ∩ Mensch − v2 − versiegelt): {len(kandidaten)}")

    je_kanal = defaultdict(list)
    for u in kandidaten:
        je_kanal[slug_von(u)].append(u)
    print("\nJe Kanal:")
    for k in sorted(je_kanal, key=lambda k: -len(je_kanal[k])):
        print(f"  {k:16s} {len(je_kanal[k])}")

    ziel = 38 + len(kandidaten)
    print(f"\nMöglicher v3-Umfang: bis zu {ziel} Aufnahmen "
          f"(v2 bleibt vollständig enthalten — der Satz WÄCHST nur, "
          f"sonst bricht die Vergleichbarkeit härter als nötig).")
    print("Erwartete Rauschbreite des Medians ~ 1/√n: "
          f"38 → {ziel}: Faktor {(38 / ziel) ** 0.5:.2f}")

    # ⚠️ Der eigentliche Befund (2026-08-13): der Engpass ist NICHT die
    # Auswahl, sondern die Review-Abdeckung des Test-Eimers. Fast alle
    # menschlichen Reviews liegen im Train-Eimer — und Train-Aufnahmen
    # duerfen nie Golden werden (sie haben das Modell trainiert). Der Weg
    # zu einem groesseren Satz fuehrt ueber gezielte Reviews GENAU DIESER
    # Liste:
    offen = sorted((test - mensch - v2 - versiegelt))
    je_kanal_offen = defaultdict(list)
    for u in offen:
        je_kanal_offen[slug_von(u)].append(u)
    print(f"\nREVIEW-HEBEL: {len(offen)} Test-Eimer-Aufnahmen ohne "
          f"menschliches Review — jede davon wird nach dem Review ein "
          f"Golden-Kandidat:")
    for k in sorted(je_kanal_offen, key=lambda k: -len(je_kanal_offen[k])):
        print(f"  {k:16s} {len(je_kanal_offen[k])}")

    aus = {"basis": "v2 + Kandidaten", "v2_n": len(v2),
           "review_hebel": offen,
           "kandidaten": kandidaten,
           "je_kanal": {k: len(v) for k, v in je_kanal.items()}}
    ziel_pfad = ARCHIV / "golden-eval-set-v3-vorschlag.json"
    ziel_pfad.write_text(json.dumps(aus, indent=2))
    print(f"\nVorschlag geschrieben: {ziel_pfad}")
    print("Aktivierung: EIGENER Schritt nach dem O1-Urteil — neue Datei, "
          "neuer set_hash, Dual-Zeile im Trend bis alle Serien umgestellt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
