#!/usr/bin/env python3
"""OCR-Spur ueber die GANZE Aufnahme fuer alle Aufnahmen mit Quelle nachrechnen.

WOZU
----
Vorarbeit fuer die Frage „OCR als Eingabespalte des Kopfs“. Der Backbone
sieht 224x224 und kann eingeblendeten Text nicht lesen; OCR holt genau
diese verworfene Information zurueck (10 von 11 Trailer-Spannen, 0
Fehlalarme). Stand 2026-09-24: 0 von 38 Golden- und 124 von 712
train-Aufnahmen hatten OCR-Werte, weil der Detect erst seit Anfang
September und nur um die Blockkanten erhebt.

WARUM FLAECHENDECKEND UND NICHT UM DIE KANTEN
--------------------------------------------
Um die vorhergesagten Kanten haengt das Fenster am Kopf — der wechselt jede
Nacht, eine mehrstuendige Nachrechnung waere nie fertig und bestuende aus
Fenstern verschiedener Modelle. Um die Label-Kanten waere ein Leck: schon
DASS an einer Stelle OCR-Werte existieren, verriete die Menschenkante.
Flaechendeckend haengt die Spur an nichts ausser der Quelle.

Verifiziert 2026-09-24 gegen die Produktions-OCR auf zwei Sendern
(nick, disney-channel): auf Einblendungs-Ebene (±3 s) nichts verloren,
nichts dazuerfunden; einzelne Abtastpunkte unterscheiden sich nur durch
die Phase des 2-s-Rasters. Kosten ~1 s je Minute Video.

WAS DIESES SKRIPT NICHT TUT
---------------------------
Keine Labels, keine Merkmals-Caches, kein Training. Es schreibt nur
<ziel>/<uuid>.json. Ob die Spur als Spalte etwas bringt, entscheidet eine
eigene Registrierung — nicht dieser Lauf.

⚠️ QUELLE GEWECHSELT = SPUR VERALTET. Eine neu geholte oder getrimmte
Quelle hat eine andere Zeitachse (dieselbe Klasse wie die veralteten
Fingerprints von CSI und South Park, 2026-09-22). Jede Spur traegt Groesse
und mtime ihrer Quelle; weicht eines ab, wird neu gerechnet.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ⚠️ launchd startet mit einem PATH ohne /opt/homebrew/bin. Der erste echte
# Start (2026-09-24) scheiterte deshalb an ALLEN 322 Aufnahmen mit
# "ffprobe: executable file not found" — die Probe aus der Shell lief, weil
# die Shell ihn hat. Derselbe PATH wie in tv-tagesserie.sh; /opt/homebrew/bin
# zeigt auf den gepinnten Mac-ffmpeg (~/ffmpeg-h3-mac/out), wie im Detect.
PFAD = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
# Gleicher Fehler so oft in Folge = systematisch, nicht die Aufnahme.
ABBRUCH_NACH = 3

HIER = Path(__file__).resolve().parent
QUELLEN = Path.home() / ".cache/tv-detect-daemon/source"
ZIEL = Path.home() / ".cache/tvd-ocr-spur"
BINARY = Path.home() / ".local/bin/tv-ocr-spur"
ARCHIV = Path.home() / ".cache/tvd-train-archive"
MESSSATZ = ARCHIV / "messsatz-2026-09-07.json"
GOLDEN = ARCHIV / "golden-eval-set.json"


def _dumps_erneuern():
    """Sperre und Warte-Logik NICHT nachbauen — dieselben Funktionen wie
    die Messsatz-Kampagne, die ihre Fallen schon hinter sich hat."""
    spec = importlib.util.spec_from_file_location(
        "dumps_erneuern", HIER / "dumps-erneuern.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _uuids(pfad, feld="uuids"):
    try:
        d = json.loads(Path(pfad).read_text())
    except Exception:
        return []
    if isinstance(d, dict):
        roh = d.get(feld) or d.get("recs") or []
    else:
        roh = d
    return [r if isinstance(r, str) else r.get("uuid") for r in roh]


def reihenfolge(vorhanden):
    """Messsatz und Golden zuerst — dort wird zuerst gemessen."""
    vorne = []
    for u in _uuids(MESSSATZ) + _uuids(GOLDEN):
        if u in vorhanden and u not in vorne:
            vorne.append(u)
    rest = sorted(u for u in vorhanden if u not in set(vorne))
    return vorne + rest


def veraltet_oder_fehlt(ziel, u, quelle):
    """True, wenn gerechnet werden muss. Grund als zweiter Wert."""
    p = ziel / f"{u}.json"
    if not p.is_file():
        return True, "fehlt"
    try:
        s = json.loads(p.read_text())
    except Exception:
        return True, "unlesbar"
    st = quelle.stat()
    if s.get("quelle_bytes") != st.st_size or s.get("quelle_mtime") != int(st.st_mtime):
        return True, "Quelle gewechselt"
    if s.get("fehlgeschlagen"):
        return True, "Luecken"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ziel", default=str(ZIEL))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trocken", action="store_true")
    ap.add_argument("--ruecksichtslos", action="store_true",
                    help="nicht auf Detects/Ausbildung warten")
    a = ap.parse_args()

    os.environ["PATH"] = PFAD + ":" + os.environ.get("PATH", "")
    fehlt = [w for w in ("ffmpeg", "ffprobe") if not shutil.which(w)]
    if fehlt:
        print(f"{', '.join(fehlt)} nicht im PATH ({os.environ['PATH']}) — "
              f"ABBRUCH vor der ersten Aufnahme.")
        return 1
    print(f"ffmpeg {shutil.which('ffmpeg')}", flush=True)

    ziel = Path(a.ziel)
    ziel.mkdir(parents=True, exist_ok=True)
    if not BINARY.is_file():
        print(f"{BINARY} fehlt — erst bauen (make build) und kopieren.")
        return 1
    de = _dumps_erneuern()

    quellen = {p.stem: p for p in QUELLEN.glob("*.ts")}
    offen = []
    gruende = {}
    for u in reihenfolge(set(quellen)):
        noetig, grund = veraltet_oder_fehlt(ziel, u, quellen[u])
        if noetig:
            offen.append(u)
            gruende[grund] = gruende.get(grund, 0) + 1
    print(f"Quellen {len(quellen)}, offen {len(offen)} "
          f"({', '.join(f'{k} {v}' for k, v in gruende.items()) or '-'}), "
          f"fertig {len(quellen) - len(offen)}", flush=True)
    if a.limit:
        offen = offen[:a.limit]
    if a.trocken:
        for u in offen[:10]:
            print("  wuerde:", u)
        return 0
    if not offen:
        print("nichts zu tun.")
        return 0

    sperre = de.einzelstueck(ziel)
    if sperre is None:
        return 0
    ok = fehl = 0
    letzter_fehler, in_folge = None, 0
    abgebrochen = False
    t0 = time.time()
    try:
        for i, u in enumerate(offen, 1):
            if not a.ruecksichtslos:
                gewartet = 0
                while True:
                    was = de.andere_arbeit_laeuft()
                    if not was:
                        break
                    if gewartet == 0:
                        print(f"  warte, {was} laeuft…", flush=True)
                    time.sleep(30)
                    gewartet += 30
            q = quellen[u]
            if not q.is_file():      # zwischendurch vom LRU geraeumt
                print(f"[{i}/{len(offen)}] {u}: Quelle weg, uebersprungen", flush=True)
                continue
            rest = ""
            if ok:
                je = (time.time() - t0) / ok
                rest = f", Rest ~{je * (len(offen) - i + 1) / 3600:.1f} h"
            print(f"[{i}/{len(offen)}] {u}{rest}", flush=True)
            try:
                r = subprocess.run(
                    [str(BINARY), "--quelle", str(q), "--aus", str(ziel / f"{u}.json")],
                    capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired:
                print("  Zeitueberschreitung (1 h) — uebersprungen", flush=True)
                fehl += 1
                continue
            if r.returncode == 0:
                ok += 1
                letzter_fehler, in_folge = None, 0
                print("  " + r.stdout.strip(), flush=True)
            else:
                fehl += 1
                # vom ENDE kuerzen: dort steht der eigentliche Fehler
                meldung = r.stderr.strip()[-300:]
                print(f"  FEHLER rc={r.returncode}: {meldung}", flush=True)
                if abbrechen(letzter_fehler, in_folge, meldung):
                    print(f"\n⚠️ ABBRUCH: {ABBRUCH_NACH}x derselbe Fehler in "
                          f"Folge — systematisch, nicht die Aufnahme.",
                          flush=True)
                    abgebrochen = True
                    break
                in_folge = in_folge + 1 if meldung == letzter_fehler else 1
                letzter_fehler = meldung
    finally:
        try:
            sperre.unlink()
        except Exception:
            pass
    print(f"\nfertig: {ok} Spuren, {fehl} Fehlschlaege, "
          f"{(time.time() - t0) / 3600:.1f} h", flush=True)
    if abgebrochen:
        return 2
    return 0 if fehl == 0 else 1


def abbrechen(letzter, in_folge, meldung):
    """True, wenn diese Meldung die ABBRUCH_NACH-te gleiche in Folge ist."""
    return meldung == letzter and in_folge + 1 >= ABBRUCH_NACH


if __name__ == "__main__":
    sys.exit(main())
