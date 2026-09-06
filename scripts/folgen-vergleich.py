#!/usr/bin/env python3
"""Wo weicht eine Folge vom Muster ihrer Serie ab? — NUR Bericht, schreibt nichts.

Tagesformate senden ihre Werbeblöcke Tag für Tag an fast derselben Stelle.
Gemessen 2026-09-06 an "Das perfekte Dinner" (13 Folgen): Block 1 beginnt in
JEDER Folge zwischen Minute 43 und 45, Block 2 zwischen 65 und 67. Bei
"Galileo" (14 Folgen) liegen drei Blöcke ähnlich fest. Damit ist eine Folge,
die aus dem Muster fällt, ohne jeden Menschen als verdächtig erkennbar:

  * EINZELGÄNGER — dort, wo diese Folge einen Block hat, zeigen die anderen
    Folgen Sendung. Wahrscheinlich ein Fehlalarm.
  * LÜCKE — dort, wo die anderen Folgen einen Block haben, hat diese keinen.
    Wahrscheinlich ein übersehener Block.

Das ist der einzige starke Korrektur-Hinweis in diesem Stapel, der ohne
Decode, ohne Whisper und ohne menschliches Urteil auskommt — und er hat
Blockauflösung, wo `autoconfirm.go` mit 60-s-Whisper-Fenstern nur zustimmen
oder ablehnen kann.

⚠️⚠️ DIE KANTEN-FUNDE SIND WIDERLEGT (2026-09-06, am selben Tag)
----------------------------------------------------------------
`folgen-vergleich-pruefen.py` hat die acht staerksten `kante-*`-Funde
gegen das Bildmaterial gehalten: **0 bestaetigt, 8 widerlegt.** In der
strittigen Spanne lief Sendung, nicht Werbung — das Label hatte recht.

Der Denkfehler war meiner, und er ist lehrreich: reproduzierbar sind die
Block-ANFAENGE (Das perfekte Dinner: Minute 43–45 in jeder Folge), nicht
die Block-LAENGEN und nicht die genaue Lage. Eine Folge, deren Werbeblock
echt 85 s spaeter beginnt oder 150 s frueher endet, erzeugt **exakt
dasselbe Konsens-Signal** wie ein falsches Label. Der Vergleich kann die
beiden nicht trennen — und die Wirklichkeit war in acht von acht Faellen
die harmlose Erklaerung.

`kante-start`/`kante-ende` sind deshalb **kein Korrekturhinweis**. Die
Spalte bleibt im Bericht, weil sie die Streuung sichtbar macht, aber wer
daraus Kanten setzt, verschlechtert Labels. Ungeprueft sind noch
`einzelgaenger` und `luecke` — das sind andere Behauptungen (ein ganzer
Block existiert nicht bzw. fehlt), und nur fuer die spricht die
Ausgangsmessung.

⚠️ WAS DIESER BERICHT AUCH SONST NICHT KANN
-------------------------------------------
Der Vergleichsmaßstab sind die Labels der anderen Folgen — und die sind
inzwischen mehrheitlich maschinell (siehe `massstab-audit.py`). Gefunden
wird deshalb UNEINIGKEIT, nicht Wahrheit. Ein Fehler, den der Detektor in
JEDER Folge derselben Serie gleich macht, ist hier unsichtbar. Dieselbe
Klasse wie `grobes_raster_misst_sich_selbst`: ein Maßstab, der aus dem
Gemessenen gebaut ist, sieht nur die Streuung, nicht die Verschiebung.

⚠️ Sendeplatz-Wechsel verschieben das Muster echt. "Mein Lokal" lief erst
59–61 min, später 71–72 min, und die Blöcke wanderten mit (25→28 min). Der
Bericht gleicht das über eine Verschiebungssuche aus und weist die gefundene
Verschiebung je Folge aus — eine große Verschiebung ist ein Grund, die Zeile
zu misstrauen, kein Fehler des Verfahrens.

⚠️ Leitplanke L2: geschrieben wird hier nichts. Der Eimer steht in jeder
Zeile, weil ausschließlich `train` je beschrieben werden dürfte.
"""
import argparse
import collections
import json
import statistics as st
import sys
import urllib.request
from pathlib import Path

LEDGER = Path.home() / ".cache/tvd-train-archive/split-ledger.json"
GOLDEN = Path.home() / ".cache/tvd-train-archive/golden-eval-set.json"

# Deckungsgleich mit tv-recorder normalizeTitle() (cmd/tv-recorder/recordings.go).
# En- und Em-Dash stehen seit 2026-09-06 mit drin: RTLZWEI hat die Schreibweise
# von "Die Geissens" mitten in der Serie gewechselt und damit eine Serie in
# zwei Eimer zerlegt.
TRENNER = (" - ", " – ", " — ", ": ")


def norm_titel(t):
    for s in TRENNER:
        i = t.find(s)
        if i >= 0:
            return t[:i].strip()
    return t.strip()


def hole(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def eimer_map():
    try:
        led = json.loads(LEDGER.read_text())
    except Exception:
        led = {}
    try:
        gold = set(json.loads(GOLDEN.read_text()).get("uuids") or [])
    except Exception:
        gold = set()
    return led, gold


def indikator(bloecke, n):
    """Sekundenraster: 1 = Werbung."""
    v = bytearray(n)
    for a, b in bloecke:
        a, b = max(0, int(a)), min(n, int(b))
        for i in range(a, b):
            v[i] = 1
    return v


def beste_verschiebung(ind, mittel, spanne, schritt, marge):
    """Verschiebung s (Sekunden), die die Überlappung mit dem Mittel maximiert.

    Positives s heißt: diese Folge liegt SPÄTER als die anderen.
    Ohne eigene Blöcke ist die Suche sinnlos — dann 0.

    ⚠️ Die Suche wird bewusst zu 0 hin gebremst. Ohne Bremse findet sie bei
    einer bereits ausgerichteten Serie noch Verschiebungen von ±35–55 s,
    weil irgendein Versatz die Überlappung um ein Promille verbessert — und
    schiebt damit gerade die Folgen auseinander, die man vergleichen will.
    Eine Verschiebung gilt deshalb nur, wenn sie die Überlappung um `marge`
    schlägt; bei Gleichstand gewinnt der kleinere Betrag.
    """
    if not any(ind):
        return 0, 0.0
    n = len(ind)

    def guete(s):
        summe = 0.0
        for i in range(0, n, 2):          # 2-s-Abtastung reicht und halbiert die Kosten
            if ind[i]:
                j = i - s
                if 0 <= j < len(mittel):
                    summe += mittel[j]
        return summe

    null = guete(0)
    best, best_s = null, 0
    for s in range(-spanne, spanne + 1, schritt):
        if s == 0:
            continue
        g = guete(s)
        if g > best * (1.0 + marge) or (g > best and abs(s) < abs(best_s)):
            best, best_s = g, s
    return best_s, best


def serie_pruefen(folgen, args):
    """folgen: [(uuid, start_ts, dauer, bloecke)] -> Liste von Funden."""
    n = max(int(f[2]) for f in folgen) + args.spanne + 1
    inds = {f[0]: indikator(f[3], n) for f in folgen}

    # Runde 1: Mittel über alle, ohne Ausrichtung — grobe Referenz.
    roh = [sum(inds[u][i] for u in inds) / len(inds) for i in range(n)]
    verschiebung = {}
    for u, _s, _d, _b in folgen:
        verschiebung[u] = beste_verschiebung(inds[u], roh, args.spanne,
                                             args.schritt, args.vs_marge)[0]

    # ⚠️ Reissleine: liegt eine Verschiebung an der Suchgrenze, hat die
    # Ausrichtung NICHT konvergiert -- sie ist ans Ende gerutscht, weil
    # nirgends ein echtes Optimum lag. Das passiert bei Erzaehlserien
    # (Charmed, Futurama, Call Me Kat): dort ist jede Folge anderer Inhalt,
    # und ein serienweites Blockmuster gibt es nicht. Solche Folgen werden
    # aus dem Vergleich genommen, statt Funde zu erzeugen, die nur
    # Ausrichtungsrauschen sind.
    rail = int(args.spanne * 0.8)
    unruhig = [u for u in verschiebung if abs(verschiebung[u]) >= rail]
    if len(unruhig) > len(folgen) * args.max_unruhig:
        return [], {"grund": f"{len(unruhig)}/{len(folgen)} Folgen ohne "
                             f"konvergente Ausrichtung", "uebersprungen": True}
    folgen = [f for f in folgen if abs(verschiebung[f[0]]) < rail]
    if len(folgen) < args.min_folgen:
        return [], {"grund": "nach der Ausrichtung zu wenige Folgen",
                    "uebersprungen": True}
    inds = {f[0]: inds[f[0]] for f in folgen}

    # Runde 2: Konsens je Folge aus den AUSGERICHTETEN anderen (leave-one-out).
    def konsens_fuer(ziel):
        # ⚠️ Nur Folgen aehnlicher LAENGE vergleichen. Die Ausrichtung kennt
        # nur einen konstanten Versatz; eine 66-min-Folge gegen eine
        # 72-min-Gruppe laesst sich damit nicht zur Deckung bringen, und die
        # spaeten Bloecke erzeugen dann Phantom-Kantenfehler. Genau so ist
        # dvr-prosieben-1783401000 (Galileo, 66.3 min gegen 72 min) als
        # "Kante 149 s zu frueh" gemeldet worden -- es war die Programmlaenge,
        # nicht die Kante.
        dz = next(int(f[2]) for f in folgen if f[0] == ziel)
        andere = [u for u in inds if u != ziel
                  and abs(next(int(f[2]) for f in folgen if f[0] == u) - dz)
                  <= args.dauer_toleranz * dz]
        if not andere:
            return None, None, 0
        sz = verschiebung[ziel]
        acc = [0.0] * n
        deckung = [0] * n
        for u in andere:
            versatz = sz - verschiebung[u]       # u in die Zeitachse von ziel schieben
            du = next(int(f[2]) for f in folgen if f[0] == u)
            for i in range(n):
                j = i - versatz
                if 0 <= j < du:
                    deckung[i] += 1
                    if inds[u][j]:
                        acc[i] += 1.0
        kons = [(acc[i] / deckung[i]) if deckung[i] else None for i in range(n)]
        return kons, deckung, len(andere)

    def andere_mit_block(ziel, a, b):
        dz = next(int(f[2]) for f in folgen if f[0] == ziel)
        """Wie viele andere Folgen haben im Bereich [a,b) überhaupt Werbung?

        Trennt den Einzelfall (1 von 13) vom Minderheitsmuster (3 von 13).
        Ein Muster, das drei Folgen teilen, ist kein Rauschen — es ist
        entweder ein dreimal wiederholter Fehlalarm oder ein zehnmal
        wiederholtes Übersehen, und die Richtung entscheidet der Bericht
        nicht.
        """
        sz = verschiebung[ziel]
        n_treffer = 0
        for u in inds:
            if u == ziel:
                continue
            du = next(int(f[2]) for f in folgen if f[0] == u)
            if abs(du - dz) > args.dauer_toleranz * dz:
                continue
            versatz = sz - verschiebung[u]
            for i in range(a, b):
                j = i - versatz
                if 0 <= j < du and inds[u][j]:
                    n_treffer += 1
                    break
        return n_treffer

    def art_der_luecke(bloecke, a, b, dauer):
        """Grenzt die Fehlstelle an einen eigenen Block, ist es eine KANTE.

        Der wertvollste Fall dieses Berichts, und ich habe ihn beim Bauen
        zuerst uebersehen: "Das perfekte Dinner" meldete eine Luecke
        43:42-45:07 bei einer Folge, deren Block bei 45:07 BEGINNT. Es fehlt
        dort kein Block, er faengt 85 s zu spaet an. Genau solche Kanten
        bestimmen block_iou -- ein fehlender Block ist selten, eine
        verrutschte Kante ist der Normalfall.
        """
        for x, y in bloecke:
            x, y = int(x), min(int(y), dauer)
            if abs(x - b) <= args.kanten_naehe:
                return "kante-start", x - a      # Block faengt zu spaet an
            if abs(y - a) <= args.kanten_naehe:
                return "kante-ende", b - y       # Block hoert zu frueh auf
        return "luecke", 0

    funde = []
    for uuid, start, dauer, bloecke in folgen:
        kons, deckung, n_andere = konsens_fuer(uuid)
        if kons is None or n_andere < args.min_andere:
            continue
        dauer = int(dauer)

        # (a) Einzelgänger: eigener Block, den die anderen nicht kennen.
        for a, b in bloecke:
            a, b = int(a), min(int(b), dauer)
            werte = [kons[i] for i in range(a, b) if kons[i] is not None]
            deck = [deckung[i] for i in range(a, b)]
            if len(werte) < args.min_luecke // 2:
                continue                       # zu kurz für ein Urteil
            # ⚠️ Am Aufnahmeende laufen die Vergleichsfolgen aus. Ein Block
            # in den letzten Sekunden hätte sonst einen Konsens aus zwei
            # Folgen — formal 0.00, inhaltlich nichts.
            if st.median(deck) < args.min_andere:
                continue
            m = st.median(werte)
            if m <= args.fp_schwelle:
                funde.append({"art": "einzelgaenger", "versatz_s": 0,
                              "uuid": uuid, "von": a, "bis": b,
                              "konsens": round(m, 2), "n_andere": n_andere,
                              "n_mit_block": andere_mit_block(uuid, a, b),
                              "verschiebung": verschiebung[uuid],
                              "gewicht": (b - a) * (1.0 - m) * min(1.0, n_andere / 8.0)})

        # (b) Lücke: die anderen sind sich einig, diese Folge hat nichts.
        eigen = inds[uuid]
        lauf_von = None
        for i in range(dauer):
            treffer = (kons[i] is not None and kons[i] >= args.miss_schwelle
                       and deckung[i] >= args.min_andere and not eigen[i])
            if treffer and lauf_von is None:
                lauf_von = i
            elif not treffer and lauf_von is not None:
                if i - lauf_von >= args.min_luecke:
                    werte = [kons[k] for k in range(lauf_von, i)]
                    _art, _vers = art_der_luecke(bloecke, lauf_von, i, dauer)
                    funde.append({"art": _art, "versatz_s": _vers,
                                  "uuid": uuid, "von": lauf_von, "bis": i,
                                  "konsens": round(st.median(werte), 2),
                                  "n_andere": n_andere,
                                  "n_mit_block": andere_mit_block(uuid, lauf_von, i),
                                  "verschiebung": verschiebung[uuid],
                                  "gewicht": (i - lauf_von) * st.median(werte) * min(1.0, n_andere / 8.0)})
                lauf_von = None
        if lauf_von is not None and dauer - lauf_von >= args.min_luecke:
            werte = [kons[k] for k in range(lauf_von, dauer)]
            _art, _vers = art_der_luecke(bloecke, lauf_von, dauer, dauer)
            funde.append({"art": _art, "versatz_s": _vers,
                          "uuid": uuid, "von": lauf_von, "bis": dauer,
                          "konsens": round(st.median(werte), 2), "n_andere": n_andere,
                          "n_mit_block": andere_mit_block(uuid, lauf_von, dauer),
                          "verschiebung": verschiebung[uuid],
                          "gewicht": (dauer - lauf_von) * st.median(werte) * min(1.0, n_andere / 8.0)})
    vs = [abs(verschiebung[f[0]]) for f in folgen]
    return funde, {"n_folgen": len(folgen), "vs_median": int(st.median(vs)) if vs else 0,
                   "uebersprungen": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi", default="http://raspberrypi5lan:9984")
    ap.add_argument("--min-folgen", type=int, default=5,
                    help="Serien mit weniger Folgen werden übersprungen (Vorgabe 5)")
    ap.add_argument("--min-andere", type=int, default=4,
                    help="so viele Vergleichsfolgen müssen mindestens da sein")
    ap.add_argument("--fp-schwelle", type=float, default=0.25,
                    help="Einzelgänger ab Konsens ≤ diesem Wert (Vorgabe 0.25)")
    ap.add_argument("--miss-schwelle", type=float, default=0.70,
                    help="Lücke ab Konsens ≥ diesem Wert (Vorgabe 0.70)")
    ap.add_argument("--min-luecke", type=int, default=60,
                    help="kürzere Abweichungen werden nicht gemeldet (Sekunden)")
    ap.add_argument("--spanne", type=int, default=300,
                    help="Suchbereich der Verschiebung in Sekunden (Vorgabe ±300)")
    ap.add_argument("--dauer-toleranz", type=float, default=0.10,
                    help="nur Folgen vergleichen, deren Laenge um hoechstens "
                         "diesen Anteil abweicht (Vorgabe 0.10)")
    ap.add_argument("--max-unruhig", type=float, default=0.34,
                    help="Serie wird uebersprungen, wenn mehr als dieser Anteil "
                         "der Folgen sich nicht ausrichten laesst (Vorgabe 0.34)")
    ap.add_argument("--kanten-naehe", type=int, default=10,
                    help="Fehlstelle gilt als Kantenfehler, wenn sie so nah "
                         "an einem eigenen Block liegt (Sekunden, Vorgabe 10)")
    ap.add_argument("--vs-marge", type=float, default=0.05,
                    help="eine Verschiebung != 0 muss die Ueberlappung um "
                         "diesen Anteil schlagen (Vorgabe 0.05)")
    ap.add_argument("--schritt", type=int, default=5,
                    help="Schrittweite der Verschiebungssuche (Sekunden)")
    ap.add_argument("--serie", help="nur diese Serie (Teilstring, ohne Beachtung der Groß-/Kleinschreibung)")
    ap.add_argument("--nur", choices=["kante", "luecke", "einzelgaenger"],
                    help="nur diese Fundart zeigen")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    led, gold = eimer_map()

    def eimer(u):
        return "golden" if u in gold else led.get(u, "—")

    recs = hole(f"{args.pi}/api/recordings")["recordings"]
    gruppen = collections.defaultdict(list)
    for r in recs:
        gruppen[(norm_titel(r.get("title", "")), r.get("channel", ""))].append(r)

    alle_funde = []
    uebersprungen = []
    serien_info = []
    geprueft = 0
    for (titel, kanal), rs in sorted(gruppen.items()):
        if len(rs) < args.min_folgen:
            continue
        if args.serie and args.serie.lower() not in titel.lower():
            continue
        folgen = []
        for r in rs:
            try:
                a = hole(f"{args.pi}/recording/{r['uuid']}/ads")
            except Exception as e:
                print(f"  ⚠️ {r['uuid']}: {e}", file=sys.stderr)
                continue
            dauer = a.get("duration_s") or r.get("duration") or 0
            if not dauer:
                continue
            folgen.append((r["uuid"], r.get("start", 0), dauer, a.get("ads") or []))
        if len(folgen) < args.min_folgen:
            continue
        funde, info = serie_pruefen(folgen, args)
        if info.get("uebersprungen"):
            uebersprungen.append((titel, kanal, len(folgen), info["grund"]))
            continue
        geprueft += 1
        serien_info.append((titel, kanal, info["n_folgen"], info["vs_median"], len(funde)))
        for f in funde:
            f["serie"], f["kanal"] = titel, kanal
            f["eimer"] = eimer(f["uuid"])
            alle_funde.append(f)

    if args.nur:
        alle_funde = [f for f in alle_funde if f["art"].startswith(args.nur)]
    alle_funde.sort(key=lambda f: -f["gewicht"])

    if args.json:
        print(json.dumps(alle_funde, ensure_ascii=False, indent=2))
        return 0

    print(f"{geprueft} Serien mit ≥{args.min_folgen} Folgen geprüft, "
          f"{len(alle_funde)} Abweichung(en).")
    if uebersprungen:
        print(f"{len(uebersprungen)} Serie(n) übersprungen — kein gemeinsames "
              f"Blockmuster (typisch für Erzählserien):")
        for t, k, n, grund in uebersprungen:
            print(f"    {t[:34]:<34} {k[:12]:<12} {n:>3} Folgen — {grund}")
    print()
    if not alle_funde:
        return 0
    print(f"{'Art':<13} {'Eimer':<11} {'Zeit':>14} {'Kons':>5} {'Andere':>7} "
          f"{'Fehler':>7} {'Vs':>5}  {'uuid':<30} Serie")
    for f in alle_funde:
        zeit = f"{f['von']//60}:{f['von']%60:02d}–{f['bis']//60}:{f['bis']%60:02d}"
        andere = f"{f['n_mit_block']}/{f['n_andere']}"
        fehler = f"{f['versatz_s']}s" if f.get("versatz_s") else ""
        print(f"{f['art']:<13} {f['eimer']:<11} {zeit:>14} {f['konsens']:>5.2f} "
              f"{andere:>7} {fehler:>7} {f['verschiebung']:>+5d}  "
              f"{f['uuid']:<30} {f['serie'][:28]}")
    print("\nKons   = Anteil der anderen Folgen, die dort Werbung haben (leave-one-out)."
          "\nAndere = wie viele andere Folgen dort ueberhaupt einen Block haben."
          "\n         1/13 ist ein Einzelfall, 3/13 ein Minderheitsmuster — das ist"
          "\n         entweder ein dreimal wiederholter Fehlalarm oder ein zehnmal"
          "\n         wiederholtes Uebersehen. Die Richtung entscheidet dieser"
          "\n         Bericht NICHT."
          "\nFehler = um so viele Sekunden liegt die Kante daneben."
          "\nVs     = Verschiebung dieser Folge gegen die Serie in Sekunden."
          "\n"
          "\nkante-start  = der Block faengt zu spaet an, kante-ende = er hoert zu"
          "\n               frueh auf. Das ist der haeufigste und teuerste Fall:"
          "\n               block_iou wird von Kanten bestimmt, nicht von fehlenden"
          "\n               Bloecken."
          "\nluecke       = eine Fehlstelle OHNE angrenzenden eigenen Block, also"
          "\n               tatsaechlich ein uebersehener Block."
          "\neinzelgaenger = eigener Block, den die anderen nicht kennen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
