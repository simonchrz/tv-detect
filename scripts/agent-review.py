#!/usr/bin/env python3
"""Kanten-Review durch Agenten statt von Hand.

Zwei Schritte, dazwischen sichten die Agenten die Bilder:

  1. `--vorbereiten [--anzahl N]`  → wählt Aufnahmen ohne menschliches
     Label, zieht um jede Blockkante Frames und schreibt je Aufnahme einen
     Auftrag nach `~/.cache/tvd-agent-review/<uuid>/`.
  2. (Agent liest `auftrag.json`, sichtet die Frames, schreibt
     `urteil.json`)
  3. `--anwenden` → liest die Urteile und setzt die Kanten über
     `POST /api/recording/<uuid>/ads/edit`.

⚠️ **Zurücknehmen kennzeichnen.** Ein Rollback setzt die Blöcke aus
`auftrag.json` zurück, aber er kann die ABWESENHEIT eines Labels nicht
wiederherstellen — die API kennt nur „setze diese Blöcke", und der Server
setzt dabei `reviewed_at`. Ohne `reviewed_by: "zurueckgenommen"` steht
danach ein MODELLWERT mit frischem Zeitstempel in der Datei und ist von
einem menschlichen Urteil nicht zu unterscheiden. Am 2026-08-16 ist genau
so einer in den Schattenlauf gerutscht.

⚠️ **Kein eigenes Backup.** Der Docstring behauptete das eine Zeit lang und
es stimmte nie. Der Stand VOR dem Review steht in `auftrag.json` unter
"bloecke" — daraus lässt sich zurücknehmen, solange das Arbeitsverzeichnis
existiert. Wer sich auf mehr verlässt, verlässt sich auf nichts.

**Warum nur die KANTEN und keine Vollsichtung:** gemessen wird von 173
Label-Blöcken genau EINER ganz verpasst (Ledger §3af). Blöcke finden ist
gelöst, die Kantenlage ist es nicht — eine Vollsichtung würde also das
Zehnfache an Frames für das Problem ausgeben, das der Detektor bereits
kann. Der Auftrag deckt deshalb ±16 s um jede Kante ab.

⚠️ **Bauart, und warum genau diese.** Die erste Fassung fragte den Agenten
„bei welcher Sekunde ist der Übergang" und nannte im Auftrag die erwarteten
Kategorien samt Markennamen. Menschliche Stichprobe: **3 von 3 Urteilen
falsch** — er antwortete, wo die Grenze außerhalb des Fensters lag,
bestätigte eine Kante einen Abtastpunkt zu spät, und lehnte vier von vier
Blockstarts mit einer erfundenen Begründung ab, die wörtlich aus meinem
eigenen Auftrag stammte.

Deshalb jetzt drei Trennungen:

  1. Der Agent klassifiziert **einzelne Bilder** in neutralen Kategorien
     (`sendungsinhalt`, `produktwerbung`, `programmvorschau`,
     `mitmachtafel`, `unklar`). Keine Markennamen im Auftrag — ein Auftrag,
     der die erwarteten Antworten aufzählt, erzeugt sie.
  2. Die **Konvention** (was davon zählt als Werbung) steht in `KONVENTION`,
     an einer Stelle, sichtbar und änderbar ohne Agenten-Neuinstruktion.
  3. Die **Kante** leitet `kante_aus_folge` ab und kann NEIN sagen: kein
     Wechsel im Fenster, mehrere Wechsel, falsche Richtung, `unklar` am
     Übergang oder Wechsel am Fensterrand → keine Kante. Genau das konnte
     die erfragte Antwort nicht.

Die Bauart „Spannen klassifizieren" hat dieselbe Stichprobe mit **4 von 4**
bestanden; der Unterschied ist die Frage, nicht der Agent.

⚠️ **Was diese Labels NICHT beantworten können:** O13 prüft, ob eine Regel
die Kante an einen Programmhinweis ziehen soll. Ein Agent, der denselben
Frame sieht, liest „Montag 20:15" und setzt die Kante genau dort — die
Regel erzeugte ihre eigene Evidenz. Agent-Labels werden deshalb als
`reviewed_by` markiert, und `kanten-schatten.py` wertet O13 nur auf
menschlichen Labels aus. O14 (Auswahl zwischen NN- und Logo-Flanke) ist
unbetroffen: diese Signale sieht kein Agent.
"""
import argparse
import json
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ARBEIT = Path.home() / ".cache/tvd-agent-review"
SNAPSHOT = Path("/tmp/tv-train-snapshot")
QUELLE = Path.home() / ".cache/tv-detect-daemon/source"
GATEWAY = "https://raspberrypi5lan:8443"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# ±16 s in 4-s-Schritten = 9 Bilder je Kante. Enger als die typische
# Kantenabweichung (Median 4.4 s, p75 14.8 s) wäre nutzlos, weiter kostet
# Bilder ohne Ertrag: jenseits von 16 s liegt nur noch jede zehnte Kante.
FENSTER_S = 16
SCHRITT_S = 4

# Zweiter Durchgang fuer Kanten, die im engen Fenster nicht entscheidbar
# waren. ⚠️ Der Bedarf ist nicht theoretisch: bei kabel-eins setzt das
# Modell den Blockstart regelmaessig auf ein GewinnArena-Insert, das nach
# Konvention SENDUNG ist — der echte Werbebeginn liegt danach und damit
# jenseits von 16 s. Ohne diesen Durchgang ist der Kanal gar nicht
# reviewbar (4 von 4 Aufnahmen im ersten Stapel betroffen).
WEIT_FENSTER_S = 72
WEIT_SCHRITT_S = 6


# Die Konvention wird HIER angewandt, nicht vom Agenten.
#
# ⚠️ Warum: am 2026-08-16 nannte der Auftrag die Konvention samt Markennamen
# ("winario, GewinnArena, 01379") — der Agent meldete daraufhin genau das,
# mit erfundenen Details, konsistent ueber vier Aufnahmen. Ein Auftrag, der
# die erwarteten Antworten aufzaehlt, erzeugt sie.
#
# Der Agent sagt jetzt nur noch NEUTRAL, was im Bild ist. Die Zuordnung zu
# Werbung/Sendung steht an einer Stelle, ist sichtbar und aenderbar, ohne
# dass ein Agent neu instruiert werden muss.
KONVENTION = {
    "sendungsinhalt":  "sendung",
    "mitmachtafel":    "sendung",   # kostenpflichtige Gewinnspiel-Einblendung
    "produktwerbung":  "werbung",
    "programmvorschau": "werbung",  # Trailer und Sendertrenner am Blockrand
    "unklar":          None,
}


def kante_aus_folge(punkte, seite):
    """Aus Einzelbild-Urteilen die Kante ableiten — oder None.

    ⚠️ Der Agent wird NICHT nach der Kante gefragt. Genau diese Frage
    ("bei welcher Sekunde ist der Uebergang") hat am 2026-08-16 drei von
    drei Fehlurteilen erzeugt: sie hat immer eine Antwort, auch wenn die
    Grenze gar nicht im Bild ist. Hier muss die Folge die Grenze zeigen,
    sonst gibt es keine.

    Bedingungen, alle maschinell pruefbar:
      * kein "unklar" am Uebergang,
      * mindestens ein Bild jeder Seite,
      * genau EIN Wechsel (kein Hin und Her),
      * der Wechsel liegt nicht am Fensterrand (sonst ist die echte Grenze
        vermutlich ausserhalb).
    """
    folge = [(t, KONVENTION.get(k)) for t, k in punkte]
    folge.sort()
    if len(folge) < 3:
        return None, "zu wenige Bilder"
    erwartet_vor = "sendung" if seite == "start" else "werbung"
    erwartet_nach = "werbung" if seite == "start" else "sendung"
    wechsel = []
    for i in range(len(folge) - 1):
        a, b = folge[i][1], folge[i + 1][1]
        if a is None or b is None:
            continue
        if a != b:
            wechsel.append(i)
    if not wechsel:
        # ⚠️ „kein Wechsel" und „unklar sitzt genau auf dem Wechsel" brauchen
        # ENTGEGENGESETZTE Abhilfen: das erste ein weiteres Fenster, das
        # zweite eine feinere Abtastung derselben Stelle. Ohne die
        # Unterscheidung schickt das Nachfassen den zweiten Fall ins weite
        # Fenster und findet dort wieder nichts. Genau so passiert am
        # 2026-08-16 bei comedy-central und disney.
        ohne_unklar = [(t, k) for t, k in folge if k is not None]
        for i in range(len(ohne_unklar) - 1):
            if ohne_unklar[i][1] == erwartet_vor and \
                    ohne_unklar[i + 1][1] == erwartet_nach:
                return None, "unklar am Uebergang"
        return None, "kein Wechsel im Fenster"
    if len(wechsel) > 1:
        return None, f"{len(wechsel)} Wechsel (Hin und Her)"
    i = wechsel[0]
    if folge[i][1] != erwartet_vor or folge[i + 1][1] != erwartet_nach:
        return None, "Wechsel in der falschen Richtung"
    if i == 0 or i + 1 == len(folge) - 1:
        return None, "Wechsel am Fensterrand — echte Grenze vermutlich ausserhalb"
    # Kante = erstes Bild der neuen Seite.
    return float(folge[i + 1][0]), None


def bloecke(pfad):
    """ads.json ist eine nackte Liste, ads_user.json ein Objekt mit "ads"."""
    try:
        d = json.loads(Path(pfad).read_text())
    except Exception:
        return None
    b = d.get("ads") if isinstance(d, dict) else d
    if not b:
        return None
    try:
        return sorted((float(x[0]), float(x[1])) for x in b)
    except (TypeError, IndexError, ValueError):
        return None


def braucht_review(d):
    """Hat die Aufnahme Auto-Blöcke, aber kein menschliches/Agenten-Label?"""
    up = d / "ads_user.json"
    if up.is_file():
        try:
            j = json.loads(up.read_text())
        except Exception:
            return False
        # Nur rein maschinell bestätigte lohnen ein Review.
        if not j.get("auto_confirmed_at"):
            return False
    return bool(bloecke(d / "ads.json"))


def frames_ziehen(uuid, zeiten, ziel):
    """Ein ffmpeg-Aufruf je Kante, aus der lokal gecachten Quelle."""
    src = QUELLE / f"{uuid}.ts"
    if not src.is_file():
        return {}
    ziel.mkdir(parents=True, exist_ok=True)
    raus = {}
    for i, t in enumerate(zeiten):
        p = ziel / f"t{int(t):06d}.jpg"
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{max(0,t):.2f}",
             "-i", str(src), "-frames:v", "1", "-vf", "scale=960:-1",
             "-q:v", "4", "-y", str(p)],
            capture_output=True, timeout=120)
        if p.is_file() and p.stat().st_size > 2000:
            raus[f"{t:.0f}"] = p.name
    return raus


DUMPS = Path.home() / ".cache/tv-detect-daemon/emit-signals"


def vorbereiten(anzahl, dump_anfordern=False):
    kandidaten = [d for d in sorted(SNAPSHOT.glob("_rec_*")) if braucht_review(d)]
    kandidaten = [d for d in kandidaten if (QUELLE / f"{d.name[5:]}.ts").is_file()]
    # ⚠️ Aufnahmen mit vorhandenem Signal-Dump zuerst. Der Schattenlauf
    # (kanten-schatten.py) braucht je Aufnahme einen Dump; ohne ihn ist das
    # Review zwar richtig, aber fuer die offenen Fragen unsichtbar.
    #
    # Und NICHT hier einen Redetect ausloesen: der schreibt ads.json neu,
    # waehrend der Agent noch gegen die alten Bloecke urteilt. Das Urteil
    # laege dann auf Kanten, die es nicht mehr gibt. Wer Dumps braucht,
    # stoesst den Redetect VOR dem Review an und wartet ihn ab.
    mit = [d for d in kandidaten if (DUMPS / f"{d.name[5:]}.json").is_file()]
    ohne = [d for d in kandidaten if d not in mit]
    print(f"{len(kandidaten)} Aufnahmen ohne Review mit lokaler Quelle "
          f"({len(mit)} davon mit Signal-Dump — die zuerst)")
    gewaehlt = _reihum(mit) + _reihum(ohne)
    gewaehlt = gewaehlt[:anzahl]
    if dump_anfordern:
        gewaehlt = _dumps_besorgen([d for d in gewaehlt
                                    if not (DUMPS / f"{d.name[5:]}.json").is_file()],
                                   gewaehlt)
    for d in gewaehlt:
        u = d.name[5:]
        auto = bloecke(d / "ads.json")
        ziel = ARBEIT / u
        kanten = []
        for i, (s, e) in enumerate(auto):
            for seite, t in (("start", s), ("ende", e)):
                zeiten = [t + k for k in range(-FENSTER_S, FENSTER_S + 1, SCHRITT_S)]
                zeiten = [z for z in zeiten if z >= 0]
                frames = frames_ziehen(u, zeiten, ziel / f"{i}_{seite}")
                kanten.append({"block": i, "seite": seite, "ist": round(t, 1),
                               "verzeichnis": f"{i}_{seite}", "frames": frames})
        auftrag = {"uuid": u, "bloecke": [[round(a, 1), round(b, 1)] for a, b in auto],
                   "kanten": kanten}
        ziel.mkdir(parents=True, exist_ok=True)
        (ziel / "auftrag.json").write_text(json.dumps(auftrag, indent=1))
        n = sum(len(k["frames"]) for k in kanten)
        print(f"  {u:36} {len(auto)} Bloecke, {len(kanten)} Kanten, {n} Frames "
              f"-> {ziel}")
    return 0


def _reihum(dirs):
    """Kanaele abwechselnd, nicht alphabetisch.

    ⚠️ Die erste Fassung nahm schlicht die ersten N der sortierten Liste —
    und lieferte 7 von 8 Aufnahmen desselben Senders (kabel-eins). Ein
    Stapel, der einen Kanal abbildet, misst diesen Kanal, nicht den Korpus;
    fuer O14 waere das Ergebnis wertlos gewesen, ohne dass die Zahl es
    verraten haette.
    """
    je = {}
    for d in dirs:
        u = d.name[5:]
        je.setdefault(u[4:u.rfind("-")], []).append(d)
    raus, i = [], 0
    while any(je.values()):
        for k in sorted(je):
            if i < len(je[k]):
                raus.append(je[k][i])
        i += 1
        if i > max((len(v) for v in je.values()), default=0):
            break
    return raus


def _dumps_besorgen(fehlen, alle, wartesekunden=1800):
    """Redetect anstossen und abwarten, BEVOR die Frames gezogen werden.

    ⚠️ Die Reihenfolge ist der Punkt. Ein Redetect schreibt ads.json neu —
    laeuft er waehrend oder nach dem Review, urteilt der Agent gegen Bloecke,
    die es danach nicht mehr gibt, und das Urteil landet auf Kanten, die
    verschoben sind. Deshalb erst der Dump, dann die Frames.

    Der `.want`-Marker ist das, was den Signal-Dump ueberhaupt entstehen
    laesst (tv-thumbs-daemon schreibt ihn nur auf Anforderung — sonst waeren
    es ~2.8 MB je Detect).
    """
    if not fehlen:
        return alle
    import time
    E = DUMPS
    E.mkdir(parents=True, exist_ok=True)
    for d in fehlen:
        u = d.name[5:]
        (E / f"{u}.want").touch()
        try:
            req = urllib.request.Request(
                f"{GATEWAY}/api/recording/{u}/redetect", data=b"", method="POST")
            urllib.request.urlopen(req, context=CTX, timeout=20).read()
        except Exception as e:
            print(f"  redetect {u} fehlgeschlagen: {e}")
    print(f"  {len(fehlen)} Redetect(s) angestossen, warte auf die Dumps …")
    ende = time.time() + wartesekunden
    while time.time() < ende:
        da = [d for d in fehlen if (E / f"{d.name[5:]}.json").is_file()]
        if len(da) == len(fehlen):
            break
        time.sleep(20)
    fertig = [d for d in alle if (E / f"{d.name[5:]}.json").is_file()]
    if len(fertig) < len(alle):
        print(f"  ⚠ {len(alle)-len(fertig)} ohne Dump — werden ausgelassen, "
              f"sonst waere das Review fuer den Schattenlauf unsichtbar")
    return fertig


def nachfassen():
    """Unbestimmte Kanten neu vorbereiten — mit dem passenden Mittel.

    ⚠️ Die Ablehnung sagt, WAS fehlt, und danach richtet sich die Abhilfe.
    Ohne diese Unterscheidung kostet jede Ablehnung eine ganze Aufnahme,
    obwohl im haeufigsten Fall nur vier Sekunden strittig sind:

      * „kein Wechsel im Fenster" / „am Fensterrand" → die Grenze liegt
        AUSSERHALB. Hilft nur ein weiteres Fenster (±72 s in 6-s-Schritten).
        Beobachtet bei kabel-eins, wo vor jedem Block eine Mitmachtafel
        steht und der echte Werbebeginn weit hinter der Modellkante liegt.
      * alles andere (ein `unklar` genau am Uebergang, Hin und Her) → die
        Grenze IST im Fenster, nur zu grob abgetastet. Hilft feiner
        abtasten (Sekundentakt) zwischen den beiden strittigen Punkten.
        Beobachtet bei comedy-central und disney: dort war jeweils genau
        ein Bild unklar, und daran ist die ganze Aufnahme gescheitert.
    """
    n_auf = 0
    for d in sorted(ARBEIT.glob("*/")):
        ap, up = d / "auftrag.json", d / "urteil.json"
        if not (ap.is_file() and up.is_file()):
            continue
        auftrag = json.loads(ap.read_text())
        try:
            bilder = json.loads(up.read_text()).get("bilder") or []
        except Exception:
            continue
        je_verz = {}
        for b in bilder:
            try:
                je_verz.setdefault(b["verzeichnis"], []).append(
                    (float(b["zeit"]), str(b["kategorie"])))
            except (KeyError, TypeError, ValueError):
                continue
        u = auftrag["uuid"]
        offen = []
        for k in auftrag["kanten"]:
            kante, grund = kante_aus_folge(je_verz.get(k["verzeichnis"], []),
                                           k["seite"])
            if kante is None:
                offen.append((k, grund))
        if not offen:
            continue
        runde = int(auftrag.get("runde", 0)) + 1
        if runde > 2:
            print(f"  {u}: bereits zweimal nachgefasst, bleibt offen")
            continue
        for k, grund in offen:
            weit = ("kein Wechsel" in grund) or ("Fensterrand" in grund)
            if weit:
                zeiten = [k["ist"] + x for x in
                          range(-WEIT_FENSTER_S, WEIT_FENSTER_S + 1, WEIT_SCHRITT_S)]
            else:
                # Feiner zwischen den bisherigen Punkten — Sekundentakt ueber
                # die Spanne, in der der Wechsel liegen muss.
                punkte = sorted(t for t, _ in je_verz.get(k["verzeichnis"], []))
                if len(punkte) < 2:
                    continue
                zeiten = [float(x) for x in range(int(punkte[0]), int(punkte[-1]) + 1)]
            verz = f"{k['block']}_{k['seite']}_r{runde}"
            k["frames"] = frames_ziehen(u, [z for z in zeiten if z >= 0], d / verz)
            k["verzeichnis"] = verz
            print(f"  {u} Block{k['block']} {k['seite']}: "
                  f"{'weiter' if weit else 'feiner'} ({len(k['frames'])} Frames) "
                  f"— war: {grund}")
        auftrag["runde"] = runde
        auftrag["kanten"] = [k for k, _ in offen] + [
            k for k in auftrag["kanten"]
            if all(k is not o for o, _ in offen)]
        ap.write_text(json.dumps(auftrag, indent=1))
        up.rename(d / f"urteil-r{runde-1}.json")
        n_auf += 1
    print(f"{n_auf} Aufnahme(n) fuer die naechste Runde bereit.")
    return 0


def anwenden(trocken):
    """Urteile lesen, Kanten ABLEITEN, schreiben.

    Erwartetes Urteil-Format (eine Zeile je BILD, nicht je Kante):

        {"bilder": [{"verzeichnis": "0_start", "zeit": 1564,
                     "kategorie": "produktwerbung"}, ...]}

    Kategorien: sendungsinhalt | produktwerbung | programmvorschau |
    mitmachtafel | unklar. Die Zuordnung zu Werbung/Sendung passiert in
    KONVENTION, die Kante in kante_aus_folge — beides hier, nicht im Agenten.
    """
    ges = 0
    for d in sorted(ARBEIT.glob("*/")):
        up, ap = d / "urteil.json", d / "auftrag.json"
        if not (up.is_file() and ap.is_file()):
            continue
        auftrag = json.loads(ap.read_text())
        try:
            bilder = json.loads(up.read_text()).get("bilder") or []
        except Exception as e:
            print(f"  {auftrag['uuid']}: urteil.json unlesbar ({e})")
            continue
        # Urteile frueherer Runden mitnehmen: die dort entschiedenen Kanten
        # sollen nicht noch einmal beurteilt werden muessen.
        for frueher in sorted(d.glob("urteil-r*.json")):
            try:
                bilder += json.loads(frueher.read_text()).get("bilder") or []
            except Exception:
                pass
        u = auftrag["uuid"]
        je_verz = {}
        for b in bilder:
            try:
                je_verz.setdefault(b["verzeichnis"], []).append(
                    (float(b["zeit"]), str(b["kategorie"])))
            except (KeyError, TypeError, ValueError):
                continue
        neu_bl = [list(x) for x in auftrag["bloecke"]]
        n, offen = 0, []
        for k in auftrag["kanten"]:
            i, seite = k["block"], k["seite"]
            kante, grund = kante_aus_folge(je_verz.get(k["verzeichnis"], []), seite)
            if kante is None:
                offen.append((i, seite, grund))
                continue
            j = 0 if seite == "start" else 1
            if abs(kante - neu_bl[i][j]) >= 0.5:
                neu_bl[i][j] = round(kante, 2)
                n += 1
        # ⚠️ Eine unbestimmte Kante macht die ganze Aufnahme unbrauchbar.
        # Wuerde sie geschrieben, stuende der MODELLWERT als menschliche
        # Wahrheit im Label und die Aufnahme gaelte als reviewt — das Modell
        # haette seinen eigenen Fehler als Referenz bestaetigt bekommen.
        if offen:
            for i, seite, grund in offen:
                print(f"    {u} Block{i} {seite}: unbestimmt ({grund})")
            print(f"  {u}: UEBERSPRUNGEN — {len(offen)} Kante(n) unbestimmt")
            continue
        neu_bl = [b for b in neu_bl if b[1] - b[0] >= 30]
        if not neu_bl:
            print(f"  {u}: keine plausiblen Bloecke uebrig")
            continue
        if trocken:
            print(f"  {u}: {n} Kante(n) — {neu_bl}")
            ges += n
            continue
        body = json.dumps({"ads": neu_bl,
                           "reviewed_by": "agent-review.py"}).encode()
        req = urllib.request.Request(f"{GATEWAY}/api/recording/{u}/ads/edit",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=CTX, timeout=20) as r:
            print(f"  {u}: {n} Kante(n) geaendert, HTTP {r.status}")
            ges += n
        (d / "angewandt").write_text("1")
    print(f"Gesamt {ges} Kanten{' (Probelauf)' if trocken else ''}.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vorbereiten", action="store_true")
    ap.add_argument("--anwenden", action="store_true")
    ap.add_argument("--nachfassen", action="store_true",
                    help="unentschiedene Kanten mit weiterem Fenster neu vorbereiten")
    ap.add_argument("--trocken", action="store_true",
                    help="mit --anwenden: nur zeigen, nichts schreiben")
    ap.add_argument("--anzahl", type=int, default=5)
    ap.add_argument("--dump-anfordern", action="store_true",
                    help="fehlende Signal-Dumps per redetect anfordern und "
                         "abwarten, BEVOR die Frames gezogen werden (sonst "
                         "ist das Review fuer den Schattenlauf unsichtbar)")
    a = ap.parse_args()
    ARBEIT.mkdir(parents=True, exist_ok=True)
    if a.vorbereiten:
        return vorbereiten(a.anzahl, a.dump_anfordern)
    if a.nachfassen:
        return nachfassen()
    if a.anwenden:
        return anwenden(a.trocken)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
