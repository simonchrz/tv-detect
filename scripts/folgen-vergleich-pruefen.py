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

`luecke` behauptet dasselbe (dort fehlt ein Block), **`einzelgaenger` das
Gegenteil**: dort steht ein Block, den die anderen Folgen nicht kennen —
die Spanne ist also **Sendung**, genau dann wenn der Vergleich recht hat.
Die erwartete Polarität steht in `POLARITAET` und darf nicht geraten
werden; wer sie vertauscht, liest jedes Ergebnis spiegelverkehrt.

⚠️ ERGEBNIS FÜR `kante-*` (2026-09-06): **0 von 8 bestätigt.** Reproduzierbar
sind Block-Anfänge, nicht Block-Längen — eine echt verschobene Werbepause
erzeugt dasselbe Signal wie ein falsches Label. Siehe Ledger gleichen Datums.

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

# Was in der strittigen Spanne stehen MUESSTE, wenn der Folgen-Vergleich
# recht hat. `einzelgaenger` ist die Gegenrichtung der beiden anderen.
POLARITAET = {
    "kante-start": "werbung",
    "kante-ende": "werbung",
    "luecke": "werbung",
    "einzelgaenger": "sendung",
}
N_STREIT = 5          # Abtastpunkte in der strittigen Spanne
RAND_S = 6            # Abstand zu beiden Enden der Spanne
KONTROLL_ABSTAND = 45  # Mindestabstand einer Kontrolle zur nächsten Blockkante


def funde_holen(args):
    ruf = [sys.executable, str(Path(__file__).with_name("folgen-vergleich.py")),
           "--nur", args.art, "--min-andere", str(args.min_andere), "--json"]
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

        # quelle_bytes: Groesse der Quelle ZUM ZEITPUNKT der Bildentnahme.
        # `veraltet()` in agent-review.py vergleicht dagegen -- ohne den
        # Vermerk faellt es auf die mtime zurueck und verwirft gueltige
        # Urteile, sobald eine Quelle bloss neu geholt wurde.
        try:
            qb = (_ar.QUELLE / f"{uuid}.ts").stat().st_size
        except OSError:
            qb = None
        (ziel / "_loesung.json").write_text(json.dumps(
            {"uuid": uuid, "fund": f, "punkte": loesung,
             "quelle_bytes": qb, "bloecke_vorher": bloecke},
            ensure_ascii=False, indent=2))
        (ziel / "auftrag.json").write_text(json.dumps({
            "aufgabe": ("Ordne JEDES Bild in diesem Verzeichnis genau einer "
                        "Kategorie zu. Sieh dir jedes Bild an. Die Bilder "
                        "stehen in keiner zeitlichen Reihenfolge und "
                        "gehoeren nicht zusammen."),
            "kategorien": sorted(_ar.KONVENTION.keys()),
            "hinweise": [
                "sendungsinhalt: laufende Sendung, gleich welche.",
                "produktwerbung: Werbespot fuer ein Produkt oder eine Marke.",
                "programmvorschau: Trailer, Sendertrenner, Programmhinweis — AUCH wenn er eine andere Sendung bewirbt.",
                "mitmachtafel: kostenpflichtige Gewinnspiel-Einblendung.",
                "folgesendung: die naechste Sendung LAEUFT bereits (Abspann-Squeeze, Vorspann, laufendes Programm). NICHT fuer Trailer oder Werbung auf eine andere Sendung — das ist programmvorschau.",
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
        got = _urteil_fuer(d)
        if got is None:
            continue
        loes, anteil_passend, kontrolle_falsch = got
        f = loes["fund"]
        erwartet_spanne = loes["_erwartet"]
        bestimmt = loes["_bestimmt"]
        streit = loes["_streit"]
        kontrollen = loes["_kontrollen"]
        anteil_werbung = (sum(1 for x in bestimmt if x == "werbung") / len(bestimmt)
                          if bestimmt else None)
        kontrolle_ok = not kontrolle_falsch and loes["_vollzaehlig"]
        if not kontrolle_ok:
            urt = "KONTROLLE FALSCH"
        elif anteil_passend is None:
            urt = "nur unklar"
        elif anteil_passend >= args.schwelle:
            urt = "bestaetigt"
        elif anteil_passend <= 1 - args.schwelle:
            urt = "widerlegt"
        else:
            urt = "unentschieden"
        zeilen.append({"verzeichnis": d.name, "urteil": urt,
                       "anteil_werbung": anteil_werbung,
                       "anteil_passend": anteil_passend,
                       "erwartet": erwartet_spanne,
                       "n_bestimmt": len(bestimmt), "n_streit": len(streit),
                       "kontrollen": kontrollen,
                       "art": f["art"], "versatz_s": f["versatz_s"],
                       "eimer": f["eimer"], "serie": f["serie"], "uuid": f["uuid"]})

    if args.json:
        print(json.dumps(zeilen, ensure_ascii=False, indent=2))
        return 0
    if not zeilen:
        print(f"Keine ausgewerteten Auftraege unter {ARBEIT}.")
        return 0
    print(f"{'Urteil':<19} {'passt':>6} {'erwartet':>9} {'Art':<14} {'Eimer':<11} Serie")
    for z in zeilen:
        ap_ = f"{z['anteil_passend']:.0%}" if z["anteil_passend"] is not None else "—"
        print(f"{z['urteil']:<19} {ap_:>6} {z['erwartet']:>9} {z['art']:<14} "
              f"{z['eimer']:<11} {z['serie'][:26]}")
    gut = [z for z in zeilen if z["urteil"] == "bestaetigt"]
    schlecht = [z for z in zeilen if z["urteil"] == "widerlegt"]
    verfehlt = [z for z in zeilen if z["urteil"] == "KONTROLLE FALSCH"]
    print(f"\n{len(gut)} bestaetigt, {len(schlecht)} widerlegt, "
          f"{len(zeilen)-len(gut)-len(schlecht)} ohne Urteil "
          f"(davon {len(verfehlt)} mit falscher Kontrolle).")
    if gut and all(z["versatz_s"] for z in gut):
        print(f"Bestaetigt, Median-Versatz {st.median([z['versatz_s'] for z in gut])}s")
    print("\n⚠️ Eine Trefferquote aus dieser Stichprobe rechtfertigt noch kein "
          "automatisches\n   Setzen von Kanten — sie sagt, ob es sich lohnt, "
          "das zu bauen.")
    return 0


SCHREIBER = "folgen-vergleich.py"


def _urteil_fuer(d):
    """Ein Probe-Verzeichnis auswerten. `None`, wenn es keins ist.

    ⚠️ DIE EINE DEFINITION. Bis 2026-09-06 stand dieselbe Logik ein
    zweites Mal in `auswerten()` -- eine Mutationsprobe zeigte, dass eine
    Aenderung dort von keinem Test bemerkt wurde, weil die Tests nur diese
    Funktion trafen. Zwei Kopien einer Regel sind zwei Regeln, sobald eine
    davon angefasst wird (Memory zusatzspalten_eine_definition).

    Liefert (loesung, anteil_passend, kontrolle_falsch); `anteil_passend`
    ist der Anteil der BESTIMMTEN Streitbilder, die zur Behauptung des
    Fundes passen -- bei `einzelgaenger` also der Sendungs-, sonst der
    Werbeanteil (POLARITAET).
    """
    lp, up = d / "_loesung.json", d / "urteil.json"
    if not (lp.is_file() and up.is_file()):
        return None
    try:
        loes = json.loads(lp.read_text())
        urteil = json.loads(up.read_text())
    except Exception:
        return None
    erwartet = POLARITAET[loes["fund"]["art"]]
    streit, kontrollen = [], {}
    for bild, info in loes["punkte"].items():
        gedeutet = _ar.KONVENTION.get(str(urteil.get(bild, "")).strip().lower())
        if info["rolle"] == "streit":
            streit.append(gedeutet)
        else:
            soll = "werbung" if info["rolle"].endswith("werbung") else "sendung"
            kontrollen[info["rolle"]] = (gedeutet, soll)
    # ⚠️ `unklar` auf einer Kontrolle ist KEIN Fehlurteil, sondern
    # Zurueckhaltung -- und Zurueckhaltung ist laut
    # agent_review_schutzkette ein Guetezeichen. Nur eine FALSCHE Kontrolle
    # disqualifiziert. Am 2026-09-06 waeren sonst zwei Laeufe als
    # "gescheitert" gezaehlt worden, die inhaltlich dasselbe sagten wie die
    # uebrigen sechs.
    kontrolle_falsch = any(g is not None and g != soll
                           for g, soll in kontrollen.values())
    bestimmt = [x for x in streit if x is not None]
    anteil = None
    if bestimmt:
        anteil_w = sum(1 for x in bestimmt if x == "werbung") / len(bestimmt)
        anteil = anteil_w if erwartet == "werbung" else 1.0 - anteil_w
    loes["_streit"] = streit
    loes["_bestimmt"] = bestimmt
    loes["_kontrollen"] = {k: v[0] for k, v in kontrollen.items()}
    loes["_erwartet"] = erwartet
    loes["_vollzaehlig"] = len(kontrollen) == 2
    return loes, anteil, kontrolle_falsch


def anwenden(args):
    """Bestaetigte Einzelgaenger loeschen. NUR train, NUR nach der Probe.

    Sieben Schranken, jede aus einem bezahlten Fehler:
      1. nur `einzelgaenger` -- `kante-*` ist am 2026-09-06 mit 0 von 8
         durchgefallen und darf nie geschrieben werden.
      2. `erlaubte_uuids()` fail-closed auf train (agent_review_schutzkette).
      3. Kontrolle darf nicht falsch sein.
      4. `veraltet()` -- Quelle darf sich seit der Bildentnahme nicht
         geaendert haben.
      5. Der Block muss im AKTUELLEN Label noch stehen; zwischenzeitliche
         Aenderungen brechen ab statt zu ueberschreiben.
      6. `reviewed_by` = folgen-vergleich.py, damit NICHT_MENSCH das Label
         als maschinell erkennt und es nie in den Massstab rutscht
         (fingerprint_bestaetigung_ist_kein_mensch).
      7. Probelauf ist die Vorgabe; Schreiben braucht --schreiben.
    """
    erlaubt = _ar.erlaubte_uuids()
    if erlaubt is None:
        return 1
    ges = 0
    for d in sorted(ARBEIT.iterdir()) if ARBEIT.exists() else []:
        got = _urteil_fuer(d)
        if not got:
            continue
        loes, anteil, kontrolle_falsch = got
        f = loes["fund"]
        uuid = f["uuid"]
        if f["art"] != "einzelgaenger":
            continue
        if (d / "angewandt").exists():
            continue
        if anteil is None or anteil < args.schwelle:
            continue
        if kontrolle_falsch:
            print(f"  {d.name}: UEBERSPRUNGEN — Kontrolle falsch")
            continue
        # ⚠️ Auch FEHLENDE Kontrollen disqualifizieren. Der Bericht verlangt
        # beide (`_vollzaehlig`), der Schreibpfad tat es bis 2026-09-06
        # nicht -- gefunden von genau dem Test, der das festhaelt. Eine
        # Probe ohne bekannte Antwort ist keine Probe, und der schreibende
        # Pfad muss mindestens so streng sein wie der berichtende.
        if not loes["_vollzaehlig"]:
            print(f"  {d.name}: UEBERSPRUNGEN — Kontrollbilder unvollstaendig")
            continue
        if uuid not in erlaubt:
            print(f"  {d.name}: UEBERSPRUNGEN — Eimer {f['eimer']!r}, nicht train")
            continue
        if _ar.veraltet(uuid, d / "_loesung.json"):
            print(f"  {d.name}: UEBERSPRUNGEN — Quelle seit der Bildentnahme geaendert")
            continue
        try:
            jetzt = _ar_hole(f"{args.pi}/recording/{uuid}/ads")
        except Exception as e:
            print(f"  {d.name}: UEBERSPRUNGEN — {e}")
            continue
        aktuell = [[float(x), float(y)] for x, y in (jetzt.get("ads") or [])]
        treffer = [b for b in aktuell
                   if abs(b[0] - f["von"]) <= 3 and abs(b[1] - f["bis"]) <= 3]
        if len(treffer) != 1:
            print(f"  {d.name}: UEBERSPRUNGEN — Block steht so nicht mehr im Label "
                  f"({len(treffer)} Treffer)")
            continue
        neu = [b for b in aktuell if b is not treffer[0]]
        z = f"{f['von']//60}:{f['von']%60:02d}–{f['bis']//60}:{f['bis']%60:02d}"
        if not args.schreiben:
            print(f"  [Probe] {uuid} {z} loeschen "
                  f"({len(aktuell)} → {len(neu)} Bloecke, {anteil:.0%} Sendung)")
            ges += 1
            continue
        body = json.dumps({"ads": neu, "reviewed_by": SCHREIBER}).encode()
        req = _urllib().Request(f"{args.pi}/api/recording/{uuid}/ads/edit",
                                data=body, headers={"Content-Type": "application/json"})
        with _urllib().urlopen(req, timeout=20) as r:
            print(f"  {uuid} {z} geloescht ({len(aktuell)} → {len(neu)}), HTTP {r.status}")
        (d / "angewandt").write_text(json.dumps(
            {"wann": __import__("time").time(), "vorher": aktuell, "nachher": neu,
             "schreiber": SCHREIBER}, ensure_ascii=False))
        ges += 1
    print(f"\n{ges} Block/Bloecke"
          + (" (Probelauf — mit --schreiben wirklich aendern)"
             if not args.schreiben else " geloescht") + ".")
    return 0


def _urllib():
    import urllib.request
    return urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vorbereiten", action="store_true")
    ap.add_argument("--auswerten", action="store_true")
    ap.add_argument("--anwenden", action="store_true",
                    help="bestaetigte einzelgaenger im train-Eimer loeschen")
    ap.add_argument("--schreiben", action="store_true",
                    help="mit --anwenden: wirklich schreiben statt Probelauf")
    ap.add_argument("--art", choices=["kante", "luecke", "einzelgaenger"],
                    default="kante", help="welche Fundart geprueft wird")
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
    if args.anwenden:
        return anwenden(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
