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

⚠️ **Kein eigenes Backup.** Der Docstring behauptete das eine Zeit lang und
es stimmte nie. Der Stand VOR dem Review steht in `auftrag.json` unter
"bloecke" — daraus lässt sich zurücknehmen, solange das Arbeitsverzeichnis
existiert. Wer sich auf mehr verlässt, verlässt sich auf nichts.

**Warum nur die KANTEN und keine Vollsichtung:** gemessen wird von 173
Label-Blöcken genau EINER ganz verpasst (Ledger §3af). Blöcke finden ist
gelöst, die Kantenlage ist es nicht — eine Vollsichtung würde also das
Zehnfache an Frames für das Problem ausgeben, das der Detektor bereits
kann. Der Auftrag deckt deshalb ±16 s um jede Kante ab.

⚠️⚠️ **STAND 2026-08-16: NICHT PRODUKTIONSREIF.** Die erste menschliche
Stichprobe über drei Urteile ergab **drei Fehler**:

  * Ein Blockende wurde auf „Sendung läuft wieder" gesetzt, obwohl über das
    ganze Fenster Trailer lief — der Agent belegte es sogar mit einem
    Vorspann, den es dort nicht gibt.
  * Eine Kante wurde unverändert bestätigt, obwohl der Übergang einen
    Abtastpunkt früher lag.
  * Vier von vier kabel-eins-Blockstarts wurden als „GewinnArena-Insert,
    01379-Nummer, 4×25.000 €" abgelehnt — tatsächlich ist dort ein
    Werbeübergang.

Der dritte Fall ist der lehrreichste: **der Auftrag nannte die Marken
selbst** („winario, GewinnArena, 01379"), und der Agent hat sie auf
mehrdeutige Bilder gemustert — konsistent über vier Aufnahmen, was wie ein
systematischer Befund aussieht statt wie eine Halluzination. Ein Auftrag,
der die erwarteten Antworten aufzählt, erzeugt sie.

Die bis dahin geschriebenen Labels sind zurückgenommen. Bevor das hier
wieder läuft, braucht es (a) einen Auftrag ohne Antwortvorgaben und (b) eine
Konstruktion, die „ich sehe die Grenze nicht" maschinell prüfbar macht,
statt dem Selbsturteil des Agenten zu vertrauen.

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
    """Unentschiedene Kanten mit weiterem Fenster neu vorbereiten.

    Nur die betroffenen Kanten, nicht die ganze Aufnahme: die bereits
    entschiedenen bleiben stehen und werden beim Anwenden mit den neuen
    zusammengefuehrt.
    """
    n_auf = 0
    for d in sorted(ARBEIT.glob("*/")):
        ap, up = d / "auftrag.json", d / "urteil.json"
        if not (ap.is_file() and up.is_file()):
            continue
        auftrag = json.loads(ap.read_text())
        urteil = json.loads(up.read_text())
        beurteilt = set()
        for k in urteil.get("kanten", []):
            eintrag = next((x for x in auftrag["kanten"]
                            if x["block"] == k.get("block")
                            and x["seite"] == k.get("seite")), None)
            if not eintrag or not eintrag["frames"]:
                continue
            punkte = sorted(float(x) for x in eintrag["frames"])
            t = float(k.get("zeit", -1))
            # Randwert = Fenstergrenze, gilt NICHT als entschieden.
            if abs(t - punkte[0]) < 0.5 or abs(t - punkte[-1]) < 0.5:
                continue
            beurteilt.add((k["block"], k["seite"]))
        offen = [k for k in auftrag["kanten"]
                 if (k["block"], k["seite"]) not in beurteilt]
        if not offen or auftrag.get("weit"):
            continue
        u = auftrag["uuid"]
        for k in offen:
            t = k["ist"]
            zeiten = [t + x for x in range(-WEIT_FENSTER_S, WEIT_FENSTER_S + 1,
                                           WEIT_SCHRITT_S)]
            zeiten = [z for z in zeiten if z >= 0]
            verz = f"{k['block']}_{k['seite']}_weit"
            k["frames"] = frames_ziehen(u, zeiten, d / verz)
            k["verzeichnis"] = verz
        auftrag["weit"] = True
        auftrag["kanten"] = [k for k in auftrag["kanten"]
                             if (k["block"], k["seite"]) not in beurteilt]
        ap.write_text(json.dumps(auftrag, indent=1))
        (d / "urteil-eng.json").write_text(json.dumps(urteil, indent=1))
        up.unlink()
        n_auf += 1
        print(f"  {u}: {len(offen)} Kante(n) mit ±{WEIT_FENSTER_S}s neu "
              f"vorbereitet ({sum(len(k['frames']) for k in offen)} Frames)")
    print(f"{n_auf} Aufnahme(n) fuer den zweiten Durchgang bereit.")
    return 0


def anwenden(trocken):
    ges = 0
    for d in sorted(ARBEIT.glob("*/")):
        up = d / "urteil.json"
        ap = d / "auftrag.json"
        if not up.is_file() or not ap.is_file():
            continue
        auftrag = json.loads(ap.read_text())
        urteil = json.loads(up.read_text())
        # Beim zweiten Durchgang stehen die frueher entschiedenen Kanten in
        # urteil-eng.json; ohne sie waere die Aufnahme wieder unvollstaendig.
        eng = d / "urteil-eng.json"
        if eng.is_file():
            alt_urteil = json.loads(eng.read_text())
            urteil["kanten"] = (alt_urteil.get("kanten", [])
                                + urteil.get("kanten", []))
            auftrag["kanten"] = (json.loads((d / "auftrag-eng.json").read_text())["kanten"]
                                 if (d / "auftrag-eng.json").is_file()
                                 else auftrag["kanten"])
        u = auftrag["uuid"]
        neu = [list(b) for b in auftrag["bloecke"]]
        n = 0
        verworfen = set()
        for k in urteil.get("kanten", []):
            i, seite = k.get("block"), k.get("seite")
            t = k.get("zeit")
            if i is None or seite not in ("start", "ende") or t is None:
                continue
            if not (0 <= i < len(neu)):
                continue
            # ⚠️ Randwerte verwerfen. Liegt der Uebergang ausserhalb des
            # abgetasteten Fensters, sieht der Agent nur Werbung (oder nur
            # Sendung) und kann bestenfalls den aeussersten Abtastpunkt
            # nennen — das ist keine Kante, sondern die Fenstergrenze. Ein
            # Agent hat genau das am 2026-08-15 sauber dazugeschrieben; ohne
            # diesen Filter waere die Fenstergrenze als Label gelandet.
            ist = auftrag["kanten"][0]["ist"] if auftrag.get("kanten") else None
            rand = next((k2 for k2 in auftrag["kanten"]
                         if k2["block"] == i and k2["seite"] == seite), None)
            if rand and rand["frames"]:
                punkte = sorted(float(x) for x in rand["frames"])
                if abs(float(t) - punkte[0]) < 0.5 or abs(float(t) - punkte[-1]) < 0.5:
                    print(f"    {u} Block {i} {seite}: Uebergang ausserhalb "
                          f"des Fensters, verworfen")
                    verworfen.add((i, seite))
                    continue
            j = 0 if seite == "start" else 1
            if abs(float(t) - neu[i][j]) < 0.5:
                continue
            neu[i][j] = round(float(t), 2)
            n += 1
        # ⚠️⚠️ UNENTSCHIEDENE KANTE = GAR KEIN LABEL fuer diese Aufnahme.
        #
        # Der gefaehrlichste Fall der ganzen Kette, gefunden 2026-08-16 an
        # dvr-kabel-eins-1783954500: der Agent erkannte beide Blockstarts
        # korrekt als Gewinnspiel (= Sendung nach Konvention), konnte den
        # ECHTEN Start aber nicht sehen — er liegt ausserhalb des Fensters —
        # und liess die Kante weg. Wuerde die Aufnahme trotzdem geschrieben,
        # stuende der MODELLWERT als menschliche Wahrheit in ads_user.json,
        # und die Aufnahme gaelte als reviewt. Das Modell haette seinen
        # eigenen Fehler als Referenz bestaetigt bekommen — schlimmer als
        # gar kein Label, weil es unsichtbar ist.
        #
        # Also: jede Kante des Auftrags braucht ein Urteil. Fehlt eines,
        # wird die Aufnahme uebersprungen und gehoert in einen Durchgang mit
        # weiterem Fenster.
        erwartet = {(k["block"], k["seite"]) for k in auftrag["kanten"]}
        beurteilt = {(k.get("block"), k.get("seite"))
                     for k in urteil.get("kanten", [])} & erwartet
        beurteilt -= verworfen
        if beurteilt != erwartet:
            fehlt = sorted(erwartet - beurteilt)
            print(f"  {u}: UEBERSPRUNGEN — {len(fehlt)} Kante(n) unentschieden "
                  f"({', '.join(f'{b}/{se}' for b, se in fehlt)}). "
                  f"Ein Modellwert als Label waere schlimmer als keines.")
            continue
        # Nur plausible Bloecke schreiben. Ein Agent, der Start und Ende
        # vertauscht, darf keine kaputte Cutlist erzeugen.
        neu = [b for b in neu if b[1] - b[0] >= 30]
        if not neu:
            print(f"  {u}: keine plausiblen Bloecke uebrig")
            continue
        if not n:
            print(f"  {u}: alle Kanten bestaetigt, keine Aenderung — "
                  f"wird trotzdem als Label geschrieben")
        if trocken:
            print(f"  {u}: {n} Kante(n) — {neu}")
            ges += n
            continue
        # ⚠️ reviewed_by, NICHT irgendein selbstgewaehlter Schluessel:
        # handleAdsEdit baut die Nutzlast neu auf und verwirft jedes Feld,
        # das es nicht kennt. Ein Marker, den der Server wegwirft, ist kein
        # Marker — das ist am 2026-08-15 einen halben Tag lang unbemerkt so
        # gelaufen.
        body = json.dumps({"ads": neu,
                           "reviewed_by": "agent-review.py"}).encode()
        req = urllib.request.Request(f"{GATEWAY}/api/recording/{u}/ads/edit",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=CTX, timeout=20) as r:
            print(f"  {u}: {n} Kante(n), HTTP {r.status}")
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
