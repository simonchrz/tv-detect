#!/usr/bin/env python3
"""Frames um eine Kante holen und die Einblendungen SCHWAERZEN.

Wozu: ein Review-Agent soll sagen, was auf einem Bild zu sehen ist
(Werbung oder Sendung) — das kann er zuverlaessig. Was er NICHT sehen darf,
ist der Programmhinweis ("Donnerstag 20:15"), denn genau dorthin zieht die
OCR-Regel ihre Kante. Ungeschwaerzt bestaetigt der Agent die Regel mit
ihrer eigenen Evidenz, und O13 besteht trivial.

Die Rahmen kommen von Vision selbst (tv-ocr TVOCR_BOXEN=1), nicht aus einem
festen Zuschnitt — die Marker sitzen in ZWEI Bildregionen (gemessen: y=0.82
und y=0.14 im selben Frame).

⚠️ Die Grenze wird NIE erfragt, sondern aus den Einzelurteilen abgeleitet:
Agenten klassifizieren Bilder zuverlaessig (4/4), Grenzen nicht (2/3
falsch). Siehe Memory agenten_review_frage_entscheidet.
"""
import argparse
import subprocess
import sys
from pathlib import Path

QUELLEN = Path.home() / ".cache/tv-detect-daemon/source"
OCR = Path.home() / ".local/bin/tv-ocr"


class RahmenFehlt(RuntimeError):
    """Der Helfer liefert keine dritte Spalte — er kann TVOCR_BOXEN nicht.

    ⚠️ MUSS knallen. Ohne Rahmen schwaerzt das Skript nichts, die Frames
    gehen ungeschwaerzt an den Agenten, und O13 misst wieder die eigene
    Evidenz — ohne dass irgendwo etwas rot wird. Genau so ist es beim ersten
    Lauf passiert: `~/.local/bin/tv-ocr` war ein aelterer Build, jede Zeile
    hatte zwei Spalten, und das Skript notierte brav "geschwaerzt=0"."""



def rahmen_lesen(bilder, ocr=OCR):
    """{Pfad: [(x,y,w,h) normiert, Ursprung UNTEN links]} — leer wenn kein Text."""
    if not bilder:
        return {}
    p = subprocess.run([str(ocr)] + [str(b) for b in bilder],
                       capture_output=True, text=True,
                       env={"TVOCR_BOXEN": "1", "PATH": "/usr/bin:/bin"})
    aus = {}
    for zeile in p.stdout.splitlines():
        teile = zeile.split("\t")
        if len(teile) < 3:
            raise RahmenFehlt(
                f"{ocr} liefert {len(teile)} Spalten statt 3 — zu alt fuer "
                f"TVOCR_BOXEN. Neu bauen: swiftc -O tools/ocr/ocr.swift "
                f"-o build/tv-ocr")
        if not teile[1]:
            aus[teile[0]] = []   # kein Text im Bild: nichts zu schwaerzen
            continue
        kaesten = []
        for k in teile[2].split(";"):
            if not k:
                continue
            try:
                kaesten.append(tuple(float(v) for v in k.split(",")))
            except ValueError:
                pass
        aus[teile[0]] = kaesten
    return aus


def in_pixel(kasten, breite, hoehe, rand=0.02):
    """Normiert (Ursprung unten) → Pixel (Ursprung oben), mit Sicherheitsrand.

    Der Rand ist nicht Kosmetik: Vision umschliesst die Glyphen knapp, und
    ein stehengebliebener Buchstabenrest ist fuer einen Agenten weiterhin
    lesbar."""
    x, y, w, h = kasten
    x0 = max(0.0, x - rand); x1 = min(1.0, x + w + rand)
    y0 = max(0.0, y - rand); y1 = min(1.0, y + h + rand)
    return (int(x0 * breite), int((1.0 - y1) * hoehe),
            int((x1 - x0) * breite), int((y1 - y0) * hoehe))


def ueberlappt(a, b):
    """Zwei normierte Rahmen (x,y,w,h) — beruehren sie sich?"""
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def lecks(vorher, nachher):
    """Welche Resttexte liegen DORT, wo geschwaerzt wurde?

    ⚠️ Die Unterscheidung entscheidet ueber Brauchbarkeit statt Strenge.
    Nach dem Schwaerzen findet Vision regelmaessig Text, den es vorher
    uebersehen hat — das Senderlogo ("E RTLZWEI"), einen Produktnamen
    ("Galaxy | S26"). Das ist kein Durchsickern: es ist nicht die Evidenz,
    an der die OCR-Regel haengt, und ein Mensch saehe es auch. Ein Rest IM
    geschwaerzten Bereich ist dagegen genau das (gesehen: "ERBUN" von
    "WERBUNG"). Wer beides gleich behandelt, verwirft brauchbare Fenster —
    oder, schlimmer, gewoehnt sich an die Warnung."""
    return [r for r in nachher if any(ueberlappt(r, v) for v in vorher)]


def hauptteil(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("uuid")
    ap.add_argument("kante", type=float, help="Sekunde der Kante")
    ap.add_argument("--fenster", type=float, default=6.0)
    ap.add_argument("--schritt", type=float, default=1.0)
    ap.add_argument("--ziel", type=Path, required=True)
    ap.add_argument("--breite", type=int, default=640)
    ap.add_argument("--rand", type=float, default=0.03,
                    help="Sicherheitsrand um jeden Textrahmen (normiert)")
    ap.add_argument("--ohne-blendung", action="store_true",
                    help="NICHT schwaerzen. Nur erlaubt, wenn die Frage nichts "
                         "mit der OCR-Regel zu tun hat — etwa: liegt vor einem "
                         "gelabelten Blockanfang schon Werbung? Ein Mensch "
                         "laese die Einblendung dort auch. ⚠️ So gewonnene "
                         "Urteile duerfen NIE als O13-Referenz dienen, sonst "
                         "ist die Zirkularitaet durch die Hintertuer zurueck.")
    ap.add_argument("--vorlauf", type=float, default=None,
                    help="statt eines symmetrischen Fensters: von kante-VORLAUF "
                         "bis kante+nachlauf")
    ap.add_argument("--nachlauf", type=float, default=4.0)
    args = ap.parse_args(argv)

    quelle = QUELLEN / f"{args.uuid}.ts"
    if not quelle.exists():
        print(f"keine Quelle im Cache: {quelle}", file=sys.stderr)
        return 1
    args.ziel.mkdir(parents=True, exist_ok=True)

    if args.vorlauf is not None:
        start = args.kante - args.vorlauf
        n = int((args.vorlauf + args.nachlauf) / args.schritt) + 1
    else:
        start = args.kante - args.fenster
        n = int(2 * args.fenster / args.schritt) + 1
    roh = args.ziel / "roh"
    roh.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", str(quelle),
         "-vf", f"fps=1/{args.schritt},scale={args.breite}:-2",
         "-frames:v", str(n), "-y", str(roh / "f%02d.png")],
        check=True)

    bilder = sorted(roh.glob("f*.png"))
    kaesten = {} if args.ohne_blendung else rahmen_lesen(bilder)
    from PIL import Image, ImageDraw
    zeilen = []
    for i, b in enumerate(bilder):
        sek = start + i * args.schritt
        im = Image.open(b).convert("RGB")
        d = ImageDraw.Draw(im)
        n_k = 0
        for k in kaesten.get(str(b), []):
            x, y, w, h = in_pixel(k, im.width, im.height, args.rand)
            d.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
            n_k += 1
        ziel = args.ziel / f"{sek:07.1f}s.png"
        im.save(ziel)
        zeilen.append((sek, ziel.name, n_k, str(b)))
    # ⚠️ GEGENPROBE, nicht optional: dieselben Bilder noch einmal durch
    # Vision. Was jetzt noch lesbar ist, hat das Schwaerzen uebersehen — ein
    # halb stehengebliebener Programmhinweis reicht dem Agenten. "Es wurden
    # N Kaesten gemalt" ist KEIN Beleg dafuer, dass nichts mehr dasteht.
    if args.ohne_blendung:
        for sek, name, n_k, _ in zeilen:
            print(f"{sek:8.1f}  {name}")
        print(f"\nUNGEBLENDET ({len(zeilen)} Bilder) — nur fuer Fragen "
              f"ausserhalb der OCR-Regel zulaessig.")
        return 0
    nach = rahmen_lesen([args.ziel / n for _, n, _, _ in zeilen])
    n_leck = n_neu = 0
    for sek, name, n_k, roh_name in zeilen:
        rest = nach.get(str(args.ziel / name), [])
        durch = lecks(kaesten.get(roh_name, []), rest)
        n_neu += len(rest) - len(durch)
        marke = ""
        if durch:
            n_leck += 1
            marke = f"  ⚠ DURCHGESICKERT: {len(durch)} Rest(e) im geschwaerzten Bereich"
        print(f"{sek:8.1f}  {name}  geschwaerzt={n_k}{marke}")
    if n_leck:
        print(f"\n⚠️ {n_leck} von {len(zeilen)} Bildern zeigen noch Text DORT, wo "
              f"geschwaerzt wurde — nicht an einen Agenten geben "
              f"(--rand erhoehen, aktuell {args.rand}).", file=sys.stderr)
        return 2
    print(f"\ngegengeprueft: {len(zeilen)} Bilder, nichts im geschwaerzten "
          f"Bereich lesbar ({n_neu} Fundstelle(n) ausserhalb — Logo, "
          f"Produktname und dergleichen; nicht die Evidenz der Regel).")
    return 0


if __name__ == "__main__":
    sys.exit(hauptteil())
