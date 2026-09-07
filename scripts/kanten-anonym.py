#!/usr/bin/env python3
"""Anonymisiert die Bilder eines agent-review-Auftrags und packt das Urteil zurück.

WARUM
-----
`agent-review.py --vorbereiten` zieht je Kante ein Fenster ±40 s um die
MODELLKANTE und legt die Bilder als `<block>_<seite>/t<sekunde>.jpg` ab;
`auftrag.json` nennt dazu `bloecke` und je Kante `ist`. Ein Agent, der
das liest, weiß, wo das Modell die Kante gesetzt hat — und das mittlere
Bild des Fensters IST die Modellkante. Genau dieser Anker erklärt die
45-%-Null-Fehler-Wolke aus `kanten_massstab_ist_agent_echo`: Agenten
bestätigten, was sie sahen, statt zu urteilen.

Hier werden alle Bilder einer Aufnahme unter anonymen Namen in ZUFÄLLIGER
Reihenfolge in ein Unterverzeichnis gelegt, ohne Zeit, ohne Block, ohne
Seite. Der Agent klassifiziert jedes Bild für sich. Danach wird das Urteil
über die Zuordnung zurück in das Format gebracht, das `urteile_von` und
`kante_aus_folge` erwarten — die Kantenableitung bekommt also dieselben
Daten wie bisher, nur ohne dass der Agent den Anker je gesehen hat.

`kante_aus_folge` profitiert davon sogar: es braucht unabhängige Urteile je
Bild, keine Geschichte über die Sequenz.

  --verpacken    für jeden offenen Auftrag: anonym/ anlegen
  --entpacken    für jeden mit anonym/urteil.json: urteil.json zurückschreiben
"""
import argparse
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

_HIER = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("agent_review", _HIER / "agent-review.py")
_ar = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_ar)

ARBEIT = _ar.ARBEIT
HINWEISE = [
    "sendungsinhalt: laufende Sendung, gleich welche.",
    "produktwerbung: Werbespot fuer ein Produkt oder eine Marke.",
    "programmvorschau: Trailer, Sendertrenner, Programmhinweis — AUCH wenn er eine andere Sendung bewirbt.",
    "mitmachtafel: kostenpflichtige Gewinnspiel-Einblendung.",
    "folgesendung: die naechste Sendung LAEUFT bereits (Abspann-Squeeze, Vorspann, laufendes Programm). NICHT fuer Trailer oder Werbung auf eine andere Sendung — das ist programmvorschau.",
    "unklar: wenn du es nicht entscheiden kannst. Nutze das.",
]


def verpacken(args):
    n = 0
    for d in sorted(ARBEIT.iterdir()):
        if not (d / "auftrag.json").is_file():
            continue
        if (d / "urteil.json").is_file() or (d / "angewandt").is_file():
            continue
        anon = d / "anonym"
        if (anon / "auftrag.json").is_file() and not args.neu:
            continue
        bilder = []
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and p.name != "anonym"):
            for f in sorted(sub.glob("t*.jpg")):
                try:
                    zeit = float(f.stem[1:])
                except ValueError:
                    continue
                bilder.append((sub.name, zeit, f))
        if not bilder:
            continue
        rng = random.Random(d.name)
        rng.shuffle(bilder)
        if anon.exists():
            shutil.rmtree(anon)
        anon.mkdir()
        zuordnung = {}
        for i, (verz, zeit, f) in enumerate(bilder, 1):
            name = f"bild_{i:03d}.jpg"
            try:
                (anon / name).hardlink_to(f)
            except OSError:
                shutil.copy2(f, anon / name)
            zuordnung[name] = {"verzeichnis": verz, "zeit": zeit}
        (d / "_anonym.json").write_text(json.dumps(zuordnung, indent=1))
        (anon / "auftrag.json").write_text(json.dumps({
            "aufgabe": ("Ordne JEDES Bild in diesem Verzeichnis genau einer "
                        "Kategorie zu. Sieh dir jedes Bild an. Die Bilder "
                        "stehen in keiner zeitlichen Reihenfolge und "
                        "gehoeren nicht zusammen."),
            "kategorien": sorted(_ar.KONVENTION.keys()),
            "hinweise": HINWEISE,
            "bilder": sorted(zuordnung),
            "ausgabe": ('Schreibe urteil.json in DIESES Verzeichnis: '
                        '{"bild_001.jpg": "<kategorie>", ...} — ein Eintrag je Bild.'),
        }, ensure_ascii=False, indent=2))
        print(f"  {d.name:<36} {len(bilder):>4} Bilder → {anon}")
        n += 1
    print(f"\n{n} Auftrag/Aufträge verpackt.")
    return 0


def entpacken(args):
    n = 0
    for d in sorted(ARBEIT.iterdir()):
        zp, up = d / "_anonym.json", d / "anonym" / "urteil.json"
        if not (zp.is_file() and up.is_file()):
            continue
        if (d / "urteil.json").is_file() and not args.neu:
            continue
        zuordnung = json.loads(zp.read_text())
        try:
            urteil = json.loads(up.read_text())
        except Exception as e:
            print(f"  ⚠️ {d.name}: anonym/urteil.json unlesbar ({e})")
            continue
        bilder = []
        fehlend = 0
        for name, info in zuordnung.items():
            kat = urteil.get(name)
            if kat is None:
                fehlend += 1
                continue
            bilder.append({"verzeichnis": info["verzeichnis"],
                           "zeit": info["zeit"], "kategorie": str(kat)})
        (d / "urteil.json").write_text(json.dumps({"bilder": bilder}, indent=1))
        print(f"  {d.name:<36} {len(bilder):>4} Urteile"
              + (f", {fehlend} Bilder ohne Urteil" if fehlend else ""))
        n += 1
    print(f"\n{n} Auftrag/Aufträge entpackt.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verpacken", action="store_true")
    ap.add_argument("--entpacken", action="store_true")
    ap.add_argument("--neu", action="store_true", help="vorhandenes überschreiben")
    args = ap.parse_args()
    if args.verpacken:
        return verpacken(args)
    if args.entpacken:
        return entpacken(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
