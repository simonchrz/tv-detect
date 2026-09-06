#!/usr/bin/env python3
"""Stimmen die Kantenfunde aus `folgen-vergleich.py`? — Agentenprobe, schreibt keine Labels.

Der Test ist in beiden Richtungen derselbe und angenehm binär:

  * `kante-ende`  — das Label lässt den Block bei `von` enden, die anderen
    Folgen sagen, er läuft bis `bis`.
  * `kante-start` — das Label lässt den Block bei `bis` beginnen, die
    anderen Folgen sagen, er beginnt schon bei `von`.

In BEIDEN Fällen gilt: **die strittige Spanne [von, bis) ist Werbung, genau
dann wenn der Folgen-Vergleich recht hat.** Ist sie Sendung, hat das Label
recht. Es braucht also keine Kantenbestimmung, nur eine Klassifikation —
und das ist das, was Agenten nachweislich gut können
(Memory `agent_als_kantenmassstab`).

BAUART, UND WARUM GENAU DIESE
-----------------------------
Drei Vorkehrungen, jede gegen einen dokumentierten Fehlschlag:

1. **Anonyme, gemischte Bilder.** `agent-review.py` legt die Zeit in den
   Dateinamen (`t001234.jpg`) — für eine Kantenableitung nötig, hier
   schädlich: aus der Reihenfolge liesse sich erraten, welche Bilder die
   strittigen sind. Hier heissen sie `bild_01.jpg` … in zufälliger
   Reihenfolge; die Zuordnung steht in `_loesung.json`, die der Agent nicht
   liest. Anlass: `agenten_review_frage_entscheidet` — ein Auftrag, der die
   erwartete Antwort nahelegt, erzeugt sie.

2. **Zwei Kontrollbilder je Fund**, deren Antwort feststeht: eines aus der
   Mitte eines unstrittigen Werbeblocks, eines aus der Mitte der längsten
   werbefreien Strecke. Wer die Kontrollen nicht trifft, dessen Urteil über
   die strittige Spanne zählt nicht. Anlass: `agent_liefert_ohne_zu_lesen` —
   erfundene Urteile sind oft richtig, und ohne eine Frage mit bekannter
   Antwort fällt das nicht auf.

3. **Keine Blocklage im Auftrag.** Der Agent erfährt weder, wo Blöcke
   liegen, noch dass es um eine Kante geht.

Die Kategorien und ihre Auslegung kommen aus `agent-review.py`
(`KONVENTION`) — keine zweite Wahrheit.

⚠️ Schreibt **nichts** ins Label. Das Ergebnis ist eine Trefferquote, auf
deren Grundlage entschieden werden kann, ob der Folgen-Vergleich Kanten
selbst setzen darf. Leitplanke L2.
"""
import argparse
import importlib.util
import json
import random
import statistics as st
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "agent_review", Path(__file__).with_name("agent-review.py"))
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

ARBEIT = Path.home() / ".cache/tvd-folgen-probe"
N_STREIT = 5          # Abtastpunkte in der strittigen Spanne
RAND_S = 6            # Abstand zu beiden Enden der Spanne
KONTROLL_ABSTAND = 45  # Mindestabstand einer Kontrolle zur nächsten Blockkante


def funde_holen(args):
    ruf = [sys.executable, str(Path(__file__).with_name("folgen-vergleich.py")),
           "--nur", "kante", "--min-andere", str(args.min_andere), "--json"]
    r = subprocess.run(ruf, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"folgen-vergleich.py fehlgeschlagen:\n{r.stderr[-300:]}")
    return json.loads(r.stdout)


def kontrollpunkte(bloecke, dauer, streit):
    """(werbung_t, sendung_t) — Zeiten, deren Antwort feststeht, oder None."""
    def stoert(t):
        return streit[0] - KONTROLL_ABSTAND <= t <= streit[1] + KONTROLL_ABSTAND

    werbung = None
    for a, b in sorted(bloecke, key=lambda p: -(p[1] - p[0])):
        if b - a < 2 * KONTROLL_ABSTAND:
            continue
        t = (a + b) / 2
        if not stoert(t):
            werbung = t
            break

    luecken = []
    rand = 0.0
    for a, b in sorted(bloecke):
        if a - rand > 0:
            luecken.append((rand, a))
        rand = b
    if dauer - rand > 0:
        luecken.append((rand, dauer))
    sendung = None
    for a, b in sorted(luecken, key=lambda p: -(p[1] - p[0])):
        if b - a < 2 * KONTROLL_ABSTAND:
            continue
        t = (a + b) / 2
        if not stoert(t):
            sendung = t
            break
    return werbung, sendung


def vorbereiten(args):
    funde = funde_holen(args)
    funde.sort(key=lambda f: -f["gewicht"])
    rng = random.Random(args.seed)
    gebaut = 0

    for f in funde:
        if gebaut >= args.anzahl:
            break
        uuid = f["uuid"]
        try:
            a = _ar_hole(f"{args.pi}/recording/{uuid}/ads")
        except Exception as e:
            print(f"  ⚠️ {uuid}: {e}", file=sys.stderr)
            continue
        bloecke = [[float(x), float(y)] for x, y in (a.get("ads") or [])]
        dauer = float(a.get("duration_s") or 0)
        streit = (f["von"], f["bis"])
        if streit[1] - streit[0] < 2 * RAND_S + 4:
            continue

        k_werb, k_send = kontrollpunkte(bloecke, dauer, streit)
        if k_werb is None or k_send is None:
            print(f"  ⚠️ {uuid} {streit}: keine sauberen Kontrollpunkte — übersprungen")
            continue

        lo, hi = streit[0] + RAND_S, streit[1] - RAND_S
        streitzeiten = [lo + (hi - lo) * i / (N_STREIT - 1) for i in range(N_STREIT)]
        punkte = ([(t, "streit") for t in streitzeiten]
                  + [(k_werb, "kontrolle_werbung"), (k_send, "kontrolle_sendung")])
        rng.shuffle(punkte)

        ziel = ARBEIT / f"{uuid}_{int(streit[0])}"
        roh = ziel / "_roh"
        gezogen = _ar.frames_ziehen(uuid, [t for t, _ in punkte], roh)
        if len(gezogen) < len(punkte):
            print(f"  ⚠️ {uuid} {streit}: nur {len(gezogen)}/{len(punkte)} Bilder — übersprungen")
            continue

        # Anonymisieren: bild_NN.jpg in gemischter Reihenfolge.
        loesung = {}
        for i, (t, rolle) in enumerate(punkte, 1):
            name = f"bild_{i:02d}.jpg"
            (roh / gezogen[f"{t:.0f}"]).rename(ziel / name)
            loesung[name] = {"t": round(t, 1), "rolle": rolle}
        roh.rmdir()

        (ziel / "_loesung.json").write_text(json.dumps(
            {"uuid": uuid, "fund": f, "punkte": loesung}, ensure_ascii=False, indent=2))
        (ziel / "auftrag.json").write_text(json.dumps({
            "aufgabe": ("Ordne JEDES Bild in diesem Verzeichnis genau einer "
                        "Kategorie zu. Sieh dir jedes Bild an. Die Bilder "
                        "stehen in keiner zeitlichen Reihenfolge und "
                        "gehoeren nicht zusammen."),
            "kategorien": sorted(_ar.KONVENTION.keys()),
            "hinweise": [
                "sendungsinhalt: laufende Sendung, gleich welche.",
                "produktwerbung: Werbespot fuer ein Produkt oder eine Marke.",
                "programmvorschau: Trailer, Sendertrenner, Programmhinweis.",
                "mitmachtafel: kostenpflichtige Gewinnspiel-Einblendung.",
                "folgesendung: eine ANDERE Sendung als die aufgezeichnete.",
                "unklar: wenn du es nicht entscheiden kannst. Nutze das.",
            ],
            "bilder": sorted(loesung),
            "ausgabe": ("Schreibe urteil.json in dieses Verzeichnis: "
                        '{"bild_01.jpg": "<kategorie>", ...} — ein Eintrag je Bild.'),
        }, ensure_ascii=False, indent=2))
        print(f"  ✓ {ziel.name}  {f['art']} {f['versatz_s']}s  {f['serie'][:24]}")
        gebaut += 1

    print(f"\n{gebaut} Auftrag/Auftraege unter {ARBEIT}")
    return 0


def _ar_hole(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def auswerten(args):
    zeilen = []
    for d in sorted(ARBEIT.iterdir()) if ARBEIT.exists() else []:
        lp, up = d / "_loesung.json", d / "urteil.json"
        if not (lp.is_file() and up.is_file()):
            continue
        loes = json.loads(lp.read_text())
        try:
            urteil = json.loads(up.read_text())
        except Exception as e:
            print(f"  ⚠️ {d.name}: urteil.json unlesbar ({e})")
            continue

        def deutung(bild):
            return _ar.KONVENTION.get(str(urteil.get(bild, "")).strip().lower())

        kontrollen = {}
        streit = []
        for bild, info in loes["punkte"].items():
            d_ = deutung(bild)
            if info["rolle"] == "streit":
                streit.append(d_)
            else:
                erwartet = "werbung" if info["rolle"].endswith("werbung") else "sendung"
                kontrollen[info["rolle"]] = (d_ == erwartet, d_, erwartet)

        # ⚠️ `unklar` auf einer Kontrolle ist KEIN Fehlurteil, sondern
        # Zurueckhaltung -- und Zurueckhaltung ist laut
        # agent_review_schutzkette ein Guetezeichen. Nur eine FALSCHE
        # Kontrolle disqualifiziert. Am 2026-09-06 waeren sonst zwei Laeufe
        # als "gescheitert" gezaehlt worden, die inhaltlich dasselbe sagten
        # wie die uebrigen sechs.
        falsch = [k for k, v in kontrollen.items() if v[1] is not None and not v[0]]
        kontrolle_ok = not falsch and len(kontrollen) == 2
        bestimmt = [x for x in streit if x is not None]
        anteil_werbung = (sum(1 for x in bestimmt if x == "werbung") / len(bestimmt)
                          if bestimmt else None)
        f = loes["fund"]
        if not kontrolle_ok:
            urt = "KONTROLLE FALSCH"
        elif anteil_werbung is None:
            urt = "nur unklar"
        elif anteil_werbung >= args.schwelle:
            urt = "bestaetigt"
        elif anteil_werbung <= 1 - args.schwelle:
            urt = "widerlegt"
        else:
            urt = "unentschieden"
        zeilen.append({"verzeichnis": d.name, "urteil": urt,
                       "anteil_werbung": anteil_werbung,
                       "n_bestimmt": len(bestimmt), "n_streit": len(streit),
                       "kontrollen": {k: v[1] for k, v in kontrollen.items()},
                       "art": f["art"], "versatz_s": f["versatz_s"],
                       "eimer": f["eimer"], "serie": f["serie"], "uuid": f["uuid"]})

    if args.json:
        print(json.dumps(zeilen, ensure_ascii=False, indent=2))
        return 0
    if not zeilen:
        print(f"Keine ausgewerteten Auftraege unter {ARBEIT}.")
        return 0
    print(f"{'Urteil':<19} {'Werb':>5} {'Art':<12} {'Fehler':>7} {'Eimer':<11} Serie")
    for z in zeilen:
        aw = f"{z['anteil_werbung']:.0%}" if z["anteil_werbung"] is not None else "—"
        print(f"{z['urteil']:<19} {aw:>5} {z['art']:<12} {z['versatz_s']:>6}s "
              f"{z['eimer']:<11} {z['serie'][:26]}")
    gut = [z for z in zeilen if z["urteil"] == "bestaetigt"]
    schlecht = [z for z in zeilen if z["urteil"] == "widerlegt"]
    verfehlt = [z for z in zeilen if z["urteil"] == "KONTROLLE FALSCH"]
    print(f"\n{len(gut)} bestaetigt, {len(schlecht)} widerlegt, "
          f"{len(zeilen)-len(gut)-len(schlecht)} ohne Urteil "
          f"(davon {len(verfehlt)} mit falscher Kontrolle).")
    if gut:
        print(f"Bestaetigte Kantenfehler: Median {st.median([z['versatz_s'] for z in gut])}s")
    print("\n⚠️ Eine Trefferquote aus dieser Stichprobe rechtfertigt noch kein "
          "automatisches\n   Setzen von Kanten — sie sagt, ob es sich lohnt, "
          "das zu bauen.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vorbereiten", action="store_true")
    ap.add_argument("--auswerten", action="store_true")
    ap.add_argument("--anzahl", type=int, default=8)
    ap.add_argument("--min-andere", type=int, default=8)
    ap.add_argument("--schwelle", type=float, default=0.6,
                    help="ab diesem Werbeanteil gilt der Fund als bestaetigt")
    ap.add_argument("--pi", default="http://raspberrypi5lan:9984")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.vorbereiten:
        return vorbereiten(args)
    if args.auswerten:
        return auswerten(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
