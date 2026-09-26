#!/usr/bin/env python3
"""SigLIP-Spur fuer alle Aufnahmen mit Quelle nachrechnen (Kampagne).

Entwurf docs/siglip-spur-design.md, Schritt 2. Muster und Fallen wie
scripts/ocr-spur-nachrechnen.py: Messsatz/Golden zuerst, fortsetzbar, wartet
auf Detects und Ausbildung (dieselben Funktionen aus dumps-erneuern.py),
Sperre gegen Doppelstart, Abbruch nach 3x demselben Fehler.

Laeuft in ~/ml/siglip-exp/.venv und laedt das Modell EINMAL fuer alle
Aufnahmen (siglip-spur.py als Modul, nicht als Kind je Aufnahme).

⚠️ QUELLE GEWECHSELT = SPUR VERALTET: jede Spur traegt Groesse und mtime
ihrer Quelle in <uuid>.json; weicht eines ab, wird neu gerechnet. Eine .npy
ohne .json (die O28-Merkmale, am Stueck gerechnet) gilt als fehlend und wird
gekachelt neu gerechnet.

Schreibt nur <ziel>/<uuid>.npy + .json. Keine Labels, kein Training.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

PFAD = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ABBRUCH_NACH = 3
HIER = Path(__file__).resolve().parent
QUELLEN = Path.home() / ".cache/tv-detect-daemon/source"
ZIEL = Path.home() / ".cache/tvd-siglip2"


def _modul(name, datei):
    spec = importlib.util.spec_from_file_location(name, HIER / datei)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def veraltet_oder_fehlt(ziel, u, quelle):
    p = ziel / f"{u}.json"
    if not p.is_file() or not (ziel / f"{u}.npy").is_file():
        return True, "fehlt"
    try:
        s = json.loads(p.read_text())
    except Exception:
        return True, "unlesbar"
    st = quelle.stat()
    if s.get("quelle_bytes") != st.st_size or s.get("quelle_mtime") != int(st.st_mtime):
        return True, "Quelle gewechselt"
    if s.get("leer"):
        return True, "leere Kacheln"
    return False, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ziel", default=str(ZIEL))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trocken", action="store_true")
    ap.add_argument("--ruecksichtslos", action="store_true",
                    help="nicht auf Detects/Ausbildung warten")
    a = ap.parse_args()

    os.environ["PATH"] = PFAD + ":" + os.environ.get("PATH", "")
    fehlt = [w for w in ("ffmpeg", "ffprobe") if not shutil.which(w)]
    if fehlt:
        print(f"{', '.join(fehlt)} nicht im PATH — ABBRUCH vor der ersten Aufnahme.")
        return 1
    ziel = Path(a.ziel)
    ziel.mkdir(parents=True, exist_ok=True)
    de = _modul("dumps_erneuern", "dumps-erneuern.py")
    ocr = _modul("ocr_nachrechnen", "ocr-spur-nachrechnen.py")

    quellen = {p.stem: p for p in QUELLEN.glob("*.ts")}
    offen, gruende = [], {}
    for u in ocr.reihenfolge(set(quellen)):
        noetig, grund = veraltet_oder_fehlt(ziel, u, quellen[u])
        if noetig:
            offen.append(u)
            gruende[grund] = gruende.get(grund, 0) + 1
    print(f"Quellen {len(quellen)}, offen {len(offen)} "
          f"({', '.join(f'{k} {v}' for k, v in gruende.items()) or '-'})", flush=True)
    if a.limit:
        offen = offen[:a.limit]
    if a.trocken or not offen:
        for u in offen[:10]:
            print("  wuerde:", u)
        return 0

    sperre = de.einzelstueck(ziel)
    if sperre is None:
        return 0
    sp = _modul("siglip_spur", "siglip-spur.py")
    enc = sp.Encoder()
    ok = fehl = 0
    letzter, in_folge, abgebrochen = None, 0, False
    t0 = time.time()
    try:
        for i, u in enumerate(offen, 1):
            if not a.ruecksichtslos:
                gewartet = 0
                while (was := de.andere_arbeit_laeuft()):
                    if gewartet == 0:
                        print(f"  warte, {was} laeuft…", flush=True)
                    time.sleep(30)
                    gewartet += 30
            q = quellen[u]
            if not q.is_file():
                print(f"[{i}/{len(offen)}] {u}: Quelle weg, uebersprungen", flush=True)
                continue
            rest = f", Rest ~{(time.time() - t0) / ok * (len(offen) - i + 1) / 3600:.1f} h" if ok else ""
            print(f"[{i}/{len(offen)}] {u}{rest}", flush=True)
            try:
                sp.schreibe_spur(q, ziel / u, enc)
                ok += 1
                letzter, in_folge = None, 0
            except Exception as e:
                fehl += 1
                meldung = (traceback.format_exc().strip()[-300:])
                print(f"  FEHLER: {e}", flush=True)
                if meldung == letzter and in_folge + 1 >= ABBRUCH_NACH:
                    print(f"\n⚠️ ABBRUCH: {ABBRUCH_NACH}x derselbe Fehler in Folge.", flush=True)
                    abgebrochen = True
                    break
                in_folge = in_folge + 1 if meldung == letzter else 1
                letzter = meldung
    finally:
        try:
            sperre.unlink()
        except Exception:
            pass
    print(f"\nfertig: {ok} Spuren, {fehl} Fehlschlaege, {(time.time() - t0) / 3600:.1f} h", flush=True)
    return 2 if abgebrochen else (0 if fehl == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
