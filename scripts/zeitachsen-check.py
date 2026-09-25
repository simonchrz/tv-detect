#!/usr/bin/env python3
"""Läuft die Label-Zeitachse mit der Merkmals-Zeitachse synchron?

⚠️ Die Labels entstehen auf der **VOD**-Zeitachse — die App zeigt das HLS,
und dort setzt der Mensch die Blockgrenzen. Die Merkmale entstehen auf der
**Quell-.ts**. Normalerweise sind beide gleich lang. Sind sie es nicht,
liegen Label und Signal um genau diese Differenz gegeneinander verschoben,
und zwar **still**: nichts an der Aufnahme sieht kaputt aus, die Blöcke
haben die richtige LÄNGE und sitzen nur an der falschen Stelle.

Gefunden 2026-08-15 an `dvr-rtl-1780078500` (Let's Dance): VOD 16090 s
gegen Quelle 16202 s. Die 112 s sahen aus wie ein hartnäckiger Modellfehler
— acht Blöcke, alle „zu spät", alle mit richtiger Länge — und trugen allein
**ein Drittel der gesamten Kantenfehler-Masse** (4511 s → 3089 s ohne die
Aufnahme). Schlimmer: der Eintrag liegt im TRAIN-Split, die verschobenen
Labels haben den Kopf also mittrainiert.

Von 107 geprüften Aufnahmen waren 105 auf 2 s deckungsgleich. Das ist also
ein Einzelfall-Detektor, kein Dauerproblem — aber ein Einzelfall reichte,
um einen Tag lang die falsche Ursache zu jagen.

Schreibt `~/.cache/tvd-train-archive/zeitachsen-versatz.json`; train-head.py
liest die Liste und lässt die betroffenen Aufnahmen aus dem Korpus.

Aufruf: python3 scripts/zeitachsen-check.py [--schwelle 15]
"""
import argparse
import concurrent.futures as cf
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
SNAPSHOT = Path("/tmp/tv-train-snapshot")
GATEWAY = "https://raspberrypi5lan:8443"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def quelldauer(rec_dir):
    """Aus der Cutlist-Kopfzeile: 'FILE PROCESSING COMPLETE 405048 FRAMES AT 2500'.

    Die Kopfzeile ist die einzige Quell-Dauer, die lokal im Snapshot liegt —
    index.m3u8 wird dort nur als 0-Byte-Marker gefuehrt.
    """
    for t in sorted(rec_dir.glob("*.txt")):
        try:
            kopf = t.read_text(errors="replace").split("\n", 1)[0]
        except Exception:
            continue
        if "FRAMES AT" not in kopf:
            continue
        teile = kopf.split()
        try:
            i = teile.index("FRAMES")
            frames = float(teile[i - 1])
            fps = float(teile[i + 2]) / 100.0
        except (ValueError, IndexError):
            continue
        if fps > 0 and frames > 0:
            return frames / fps
    return None


class AbrufFehler(Exception):
    """VOD-Dauer unbekannt (Timeout, 5xx, Netz) — NICHT dasselbe wie "kein VOD"."""


def voddauer(uuid):
    # ⚠️ Bis 2026-09-25 lieferte JEDER Fehler None, und pruefe() liess die
    # Aufnahme dann einfach weg. Ein Timeout beim Abruf nahm eine
    # quarantaenierte Aufnahme so aus der Liste, die Datei wurde ohne sie
    # neu geschrieben, und train-head trainierte die versetzten Labels in
    # der naechsten Nacht wieder mit. Jetzt: 404 = kein VOD (Dauer 0, siehe
    # main), alles andere = AbrufFehler, und main behaelt den alten Stand.
    try:
        with urllib.request.urlopen(f"{GATEWAY}/recording/{uuid}/index.m3u8",
                                    context=CTX, timeout=25) as r:
            txt = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 0.0
        raise AbrufFehler(f"HTTP {e.code}") from e
    except Exception as e:
        raise AbrufFehler(str(e) or type(e).__name__) from e
    s = 0.0
    for z in txt.splitlines():
        if z.startswith("#EXTINF:"):
            try:
                s += float(z[8:].strip().rstrip(","))
            except ValueError:
                pass
    return s


def pruefe(rec_dir):
    uuid = rec_dir.name[5:]
    q = quelldauer(rec_dir)
    if q is None:
        return None
    try:
        v = voddauer(uuid)
    except AbrufFehler as e:
        return uuid, None, None, str(e)
    return uuid, q, v, q - v


def versatz_liste(res, schwelle, alt):
    """Quarantaene aus den Messungen, Abruffehler behalten den alten Stand.

    res: Tupel aus pruefe(); q None = Abruf fehlgeschlagen.
    alt: das bisherige "versetzt" aus der Datei ({} wenn keine).
    """
    # ⚠️ VOD-Dauer 0 heisst "kein VOD", nicht "Versatz gleich Laufzeit".
    # Beides gehoert in die Quarantaene, aber aus verschiedenen Gruenden —
    # ohne VOD konnte niemand Labels darauf setzen, die vorhandenen stammen
    # dann aus einer anderen Quelle und sind erst recht nicht vertrauenswuerdig.
    versetzt = {u: round(d, 1) for u, q, v, d in res
                if q is not None and (v <= 0 or abs(d) > schwelle)}
    # Nicht gemessen heisst nicht entlastet: wer vorher drin war, bleibt.
    for u, q, v, d in res:
        if q is None and u in alt:
            versetzt[u] = alt[u]
    return versetzt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # 15 s statt eines kleineren Werts, weil die Kennzahl nicht sagt, WO die
    # Differenz sitzt. Fehlt am ENDE ein Segment, verschiebt das keine
    # einzige Label-Grenze; fehlt am ANFANG etwas, verschiebt es alle. Die
    # Segmente sind 6 s lang, und im Korpus liegen drei Aufnahmen bei exakt
    # +6.0 s — das ist Endsegment-Rundung, kein Versatz. Ueber zwei
    # Segmentlaengen hinaus ist es keine Rundung mehr.
    ap.add_argument("--schwelle", type=float, default=15.0,
                    help="ab wieviel Sekunden Versatz eine Aufnahme "
                         "quarantaeniert wird (Vorgabe 15 = mehr als zwei "
                         "HLS-Segmente)")
    ap.add_argument("--hls-root", default=str(SNAPSHOT))
    a = ap.parse_args()

    dirs = [d for d in sorted(Path(a.hls_root).glob("_rec_*"))
            if (d / "ads_user.json").is_file()]
    with cf.ThreadPoolExecutor(8) as ex:
        alle = [r for r in ex.map(pruefe, dirs) if r]
    fehler = [r for r in alle if r[1] is None]
    res = [r for r in alle if r[1] is not None]

    ARCHIV.mkdir(parents=True, exist_ok=True)
    ziel = ARCHIV / "zeitachsen-versatz.json"
    try:
        alt = json.loads(ziel.read_text()).get("versetzt", {})
    except FileNotFoundError:
        alt = {}
    except Exception as e:
        # Unlesbar = der alte Stand ist unbekannt. Dann nicht mit einer
        # Liste ueberschreiben, in der die Abruffehler fehlen.
        if fehler:
            print(f"FEHLER: {ziel} unlesbar ({e}) und {len(fehler)} Abrufe "
                  f"fehlgeschlagen — Datei bleibt unveraendert",
                  file=sys.stderr)
            return 1
        alt = {}
    versetzt = versatz_liste(alle, a.schwelle, alt)
    tmp = ziel.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "geprueft": len(res), "schwelle_s": a.schwelle,
        "abruf_fehler": sorted(r[0] for r in fehler),
        "versetzt": versetzt}, indent=1))
    tmp.replace(ziel)

    print(f"{len(res)} Aufnahmen mit Labels geprueft, "
          f"{len(versetzt)} quarantaeniert (> {a.schwelle:.0f}s)")
    if fehler:
        behalten = [r[0] for r in fehler if r[0] in alt]
        print(f"  ⚠️ {len(fehler)} VOD-Abrufe fehlgeschlagen, "
              f"{len(behalten)} davon bleiben aus dem Vorlauf quarantaeniert:")
        for u, _, _, grund in fehler:
            print(f"    {u:36} {grund}"
                  + ("   (alter Eintrag behalten)" if u in alt else ""))
    # Auch die knapp darunter zeigen — wer die Schwelle spaeter anzweifelt,
    # soll sehen, was sie gerade durchlaesst.
    knapp = [(u, q, v, d) for u, q, v, d in res
             if u not in versetzt and abs(d) > 2.0]
    if knapp:
        print(f"  ({len(knapp)} weitere zwischen 2s und der Schwelle — "
              f"vermutlich Endsegment-Rundung, nicht quarantaeniert)")
    for u, q, v, d in sorted(res, key=lambda r: -abs(r[3])):
        if u in versetzt:
            print(f"  {u:36} Quelle {q:8.0f}s  VOD {v:8.0f}s  {d:+8.1f}s"
                  + ("   (kein VOD)" if v <= 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
