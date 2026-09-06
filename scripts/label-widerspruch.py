#!/usr/bin/env python3
"""Wo widersprechen Agenten-Bildurteile dem Label? — NUR Bericht, schreibt nichts.

Nebenprodukt von `agent-review.py --anwenden` sichtbar gemacht. Dort zaehlt
nur, ob eine Kante ABLEITBAR ist; die Ablehnungen laufen als Rauschen mit.
Genau darin steckt aber die interessantere Auskunft:

  Sieht der Agent ueber das GANZE Fenster (±40 s) nur EINE Seite, waehrend
  das Label dort eine Werbeblock-Kante behauptet, dann ist da kein
  Uebergang — der Block liegt falsch, nicht die Kante.

Das kann `agent-review.py` per Konstruktion nicht heilen: es verschiebt
Kanten, es loescht und legt keine Bloecke an. Deshalb ein eigener Bericht
statt eines Umbaus.

⚠️ Ein Widerspruch ist ein HINWEIS, kein Beweis. Der Agent kann sich irren,
und ein Fenster voller "unklar" ist KEIN Widerspruch, sondern Schweigen —
beides wird hier getrennt ausgewiesen.

⚠️ Golden- und Test-Aufnahmen werden mitberichtet, weil der Hinweis dort am
wertvollsten waere — aber Leitplanke L2 gilt: Labels des Golden-Satzes sind
Eingabe, nicht Stellschraube. Der Eimer steht deshalb in jeder Zeile.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "agent_review", Path(__file__).with_name("agent-review.py"))
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

LEDGER = Path.home() / ".cache/tvd-train-archive/split-ledger.json"


def eimer_map():
    try:
        return json.loads(LEDGER.read_text())
    except Exception:
        return {}


def urteile(d):
    """Alle Bildurteile einer Aufnahme, spaetere Runden zuerst gewinnend."""
    bilder = []
    up = d / "urteil.json"
    if up.is_file():
        try:
            bilder += json.loads(up.read_text()).get("bilder") or []
        except Exception:
            return None
    else:
        return None
    for frueher in sorted(d.glob("urteil-r*.json")):
        try:
            bilder += json.loads(frueher.read_text()).get("bilder") or []
        except Exception:
            pass
    return bilder


def seiten(punkte):
    """Welche Seiten (sendung/werbung) kommen im Fenster vor, und wie oft."""
    zaehl = {}
    for _t, k in punkte:
        s = _ar.KONVENTION.get(k)
        zaehl[s] = zaehl.get(s, 0) + 1
    return zaehl


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PFAD",
                    help="Widersprueche zusaetzlich maschinenlesbar ablegen")
    ap.add_argument("--nur-eimer", metavar="EIMER",
                    help="nur diesen Ledger-Eimer zeigen (train|test|versiegelt)")
    args = ap.parse_args()

    eimer = eimer_map()
    befunde = []
    veraltete = []
    ohne_urteil = 0

    for d in sorted(_ar.ARBEIT.glob("*/")):
        ap_datei = d / "auftrag.json"
        if not ap_datei.is_file():
            continue
        try:
            auftrag = json.loads(ap_datei.read_text())
        except Exception:
            continue
        bilder = urteile(d)
        if bilder is None:
            ohne_urteil += 1
            continue
        u = auftrag["uuid"]
        # ⚠️ Ist die Quelle NEUER als der Auftrag, stammen die Frames aus
        # einem anderen Schnitt und JEDER Befund darauf ist wertlos. Am
        # 2026-09-06 betraf das 6 von 9 Widerspruechen, darunter den
        # groessten (dvr-rtl-1780078500, 11/16) — dort lagen die Frames
        # 111 s daneben, die Labels waren in Wahrheit korrekt.
        if _ar.veraltet(u, ap_datei):
            veraltete.append(u)
            continue
        if args.nur_eimer and eimer.get(u) != args.nur_eimer:
            continue
        je_verz = {}
        for b in bilder:
            try:
                je_verz.setdefault(b["verzeichnis"], []).append(
                    (float(b["zeit"]), str(b["kategorie"])))
            except (KeyError, TypeError, ValueError):
                continue

        kanten = []
        for k in auftrag["kanten"]:
            punkte = je_verz.get(k["verzeichnis"], [])
            kante, grund = _ar.kante_aus_folge(punkte, k["seite"])
            z = seiten(punkte)
            n_unklar = z.get(None, 0)
            eindeutig = {s: n for s, n in z.items() if s is not None}
            zeiten = [t for t, _ in punkte]
            einseitig = bool(zeiten) and (k["ist"] <= min(zeiten) or k["ist"] >= max(zeiten))
            if kante is not None:
                art, detail = "ableitbar", f"{kante - k['ist']:+.0f}s"
            elif einseitig:
                # ⚠️ Kante am Rand der Aufnahme (t=0 oder Ende): das Fenster
                # liegt komplett auf EINER Seite, ein Uebergang kann dort
                # gar nicht sichtbar sein. Am 2026-09-06 meldete der Bericht
                # so dvr-nick-1778516100 (SpongeBob, Block ab 0:00) als
                # Widerspruch — der Agent sagte korrekt "Werbung", es gab
                # nur kein Davor. Kein Widerspruch, sondern Randlage.
                art, detail = "stumm", "Randlage (Fenster einseitig, kein Davor/Danach)"
            elif grund and "kein Wechsel" in grund:
                # DAS ist der Widerspruch: eine Seite ueber das ganze Fenster,
                # obwohl das Label hier eine Kante behauptet.
                if len(eindeutig) == 1 and n_unklar <= len(punkte) // 3:
                    seite_ist = next(iter(eindeutig))
                    art = "WIDERSPRUCH"
                    detail = (f"durchgehend {seite_ist} "
                              f"({eindeutig[seite_ist]}/{len(punkte)} Bilder)")
                else:
                    art, detail = "stumm", "kein Wechsel, aber auch nicht eindeutig"
            elif grund and "unklar" in grund:
                art, detail = "stumm", "unklar am Uebergang"
            else:
                art, detail = "stumm", (grund or "?")
            kanten.append({"block": k["block"], "seite": k["seite"],
                           "ist": k["ist"], "art": art, "detail": detail})

        n_w = sum(1 for x in kanten if x["art"] == "WIDERSPRUCH")
        if kanten:
            befunde.append({"uuid": u, "eimer": eimer.get(u, "?"),
                            "kanten": kanten, "n_widerspruch": n_w,
                            "n_kanten": len(kanten)})

    befunde.sort(key=lambda b: (-b["n_widerspruch"],
                                -b["n_widerspruch"] / max(1, b["n_kanten"])))
    ges_k = sum(b["n_kanten"] for b in befunde)
    ges_w = sum(b["n_widerspruch"] for b in befunde)
    ges_a = sum(1 for b in befunde for x in b["kanten"] if x["art"] == "ableitbar")

    print("=" * 72)
    print("LABEL-WIDERSPRUCH — wo der Agent keinen Uebergang sieht, "
          "obwohl das Label einen behauptet")
    print("=" * 72)
    print(f"\n{len(befunde)} Aufnahmen mit Urteil, {ges_k} Kanten gepruft"
          f"{f' ({ohne_urteil} Auftraege noch ohne Urteil)' if ohne_urteil else ''}")
    if veraltete:
        print(f"  ⚠ {len(veraltete)} Aufnahmen uebersprungen: QUELLE NEUER ALS "
              f"AUFTRAG — die Frames stammen aus einem anderen Schnitt,")
        print(f"    jeder Befund darauf waere wertlos. Neu vorbereiten, um sie "
              f"zu pruefen:")
        for u in veraltete:
            print(f"      {u}")
    print(f"  ableitbar   {ges_a:4d}  (Kante berechenbar — das nutzt agent-review)")
    print(f"  WIDERSPRUCH {ges_w:4d}  (eine Seite ueber das ganze Fenster)")
    print(f"  stumm       {ges_k - ges_a - ges_w:4d}  (unklar / Hin und Her / Fensterrand)")

    mit = [b for b in befunde if b["n_widerspruch"]]
    if not mit:
        print("\nKein Widerspruch gefunden.")
    else:
        print(f"\nAufnahmen mit Widerspruch ({len(mit)}), staerkste zuerst:\n")
        for b in mit:
            warn = "  ⚠️ L2: Labels nicht anfassen" if b["eimer"] in ("test", "versiegelt") or b["eimer"] == "?" else ""
            print(f"  {b['uuid']:34s} [{b['eimer']:10s}] "
                  f"{b['n_widerspruch']}/{b['n_kanten']} Kanten{warn}")
            for x in b["kanten"]:
                if x["art"] == "WIDERSPRUCH":
                    print(f"      Block{x['block']} {x['seite']:5s} @{x['ist']:8.1f}s "
                          f"— {x['detail']}")
    print("\nLesart: WIDERSPRUCH heisst, dort liegt vermutlich der BLOCK falsch, "
          "nicht die Kante.\n        agent-review.py kann das nicht heilen — es "
          "verschiebt nur Kanten.\n        Ein Hinweis, kein Beweis: der Agent "
          "kann irren, 'stumm' ist kein Widerspruch.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [b for b in befunde if b["n_widerspruch"]], indent=2, ensure_ascii=False))
        print(f"\nMaschinenlesbar: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
