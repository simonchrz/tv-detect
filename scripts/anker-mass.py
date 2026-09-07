#!/usr/bin/env python3
"""Misst die Bloecke des Modells gegen Anker — ohne ein einziges Label.

WARUM ES DAS BRAUCHT
--------------------
Der Golden-Satz pendelt seit Wochen zwischen 0.957 und 0.964. Ob das
Rauschen um ein Plateau ist oder die Decke der METRIK, laesst sich mit
Labels nicht entscheiden: `golden_schwanz_metrik_blinde_flecken` und
`golden_massstab` halten fest, dass ein Viertel der Golden-Kanten selbst
falsch war. Ein Maßstab, der keine Labels benutzt, umgeht das.

Anker sind dafuer der richtige Stoff. Ein Werbespot, der nachweislich
mehrfach lief, IST Werbung — darueber muss niemand urteilen. Zwei
Quellen liefern sie:

  Audio    tv-recorder, Chromaprint + dHash (spot_extract.go)
  Bild     scripts/wiederholung.py, wiederholte Folgen im Merkmals-Cache

DIE ZIRKULARITAETSFALLE — UND WARUM ES ZWEI QUELLEN SIND
--------------------------------------------------------
Die Audio-Extraktion faehrt ffmpeg NUR ueber die bereits bestaetigten
Werbebloecke. Ein Audio-Anker kann deshalb per Konstruktion nie
ausserhalb eines Blocks liegen. Wer damit "Deckung" misst, misst das
Raster an sich selbst (`grobes_raster_misst_sich_selbst`). Gemessen am
2026-09-07: 3380 von 3421 senderuebergreifenden Audio-Ankern liegen
ganz in einem auto-Block, und KEIN Block schneidet in einen hinein.
Als Trend ist das gesaettigt und damit wertlos.

Als WAECHTER ist es genau deshalb gut: faellt der Wert, hat das Modell
angefangen, in bekannte Werbung zu schneiden. Das ist ein Alarm, kein
Fortschrittsbalken, und dieses Skript nennt es auch so.

Die Bild-Anker haben die Beschraenkung nicht. Sie entstehen ueber die
GANZE Aufnahme, nicht nur in den Bloecken, und brauchen weder Audio noch
die Quelle auf dem Pi. Wo sie ausserhalb eines Blocks liegen, ist das ein
echter Befund: entweder verpasste Werbung oder ein Fehler des Verfahrens.
Beides will man sehen.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np


def lade_anker(pfad):
    aus = {}
    for f in Path(pfad).glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("anchored"):
            aus[d["uuid"]] = [(float(a["window_start_s"]), float(a["end_s"]),
                               int(a.get("family_id", -1)), int(a.get("family_size", 0)))
                              for a in d["anchored"] if a["end_s"] > a["window_start_s"]]
    return aus


def lade_bloecke(pfad, art):
    aus = {}
    for f in Path(pfad).glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        b = d.get(art) or []
        aus[f.stem] = sorted((float(x[0]), float(x[1])) for x in b if len(x) >= 2)
    return aus


def senderweit(anker):
    """family_id -> Menge der Sender. Eigener Verbund, nicht vom Endpoint:
    so gilt dieselbe Regel fuer Audio- und Bild-Anker."""
    f2c = collections.defaultdict(set)
    for u, ans in anker.items():
        slug = u.rsplit("-", 1)[0][4:] if u.startswith("dvr-") else "?"
        for (s, e, fid, fs) in ans:
            if fid >= 0:
                f2c[fid].add(slug)
    return f2c


def bericht(anker, bloecke, name, nur_hart):
    f2c = senderweit(anker)
    drin = teil = draussen = 0
    schnitt = 0
    aussen_uuids = collections.Counter()
    aussen_s = 0
    for u, ans in anker.items():
        bl = bloecke.get(u)
        if not bl:
            continue
        for (s, e, fid, fs) in ans:
            if nur_hart and fid >= 0 and len(f2c[fid]) < 2:
                continue
            ov = max((min(e, be) - max(s, bs) for bs, be in bl), default=0.0)
            if ov <= 0:
                draussen += 1
                aussen_uuids[u] += 1
                aussen_s += e - s
            elif ov >= (e - s) - 0.01:
                drin += 1
            else:
                teil += 1
                schnitt += 1
                aussen_uuids[u] += 1
                aussen_s += (e - s) - ov
    tot = drin + teil + draussen
    if not tot:
        print(f"  {name}: keine Anker")
        return None
    print(f"  {name}: {tot} Anker in {len({u for u in anker if u in bloecke})} Aufnahmen")
    print(f"    ganz in einem Block   {drin:>6}  ({100*drin/tot:5.1f} %)")
    print(f"    nur teilweise         {teil:>6}  ({100*teil/tot:5.1f} %)")
    print(f"    ganz ausserhalb       {draussen:>6}  ({100*draussen/tot:5.1f} %)")
    print(f"    ungedeckte Ankerzeit  {aussen_s/60:>6.0f} min in {len(aussen_uuids)} Aufnahmen")
    return {"anker": tot, "drin": drin, "teil": teil, "draussen": draussen,
            "gedeckt_pct": round(100 * drin / tot, 2),
            "ungedeckt_s": round(aussen_s, 1),
            "schlimmste": aussen_uuids.most_common(10)}


def versaetze(anker, bloecke):
    """Wie weit reicht ein Block ueber seinen aeussersten Anker hinaus?

    Negativ waere ein Block, der in einen bekannten Spot hineinschneidet.
    Das darf nicht vorkommen: ein Anker IST Werbung, und ueber das, was
    davor liegt, sagt er nichts (dieselbe Begruendung wie in
    internal/blocks/spot_lp.go). Positive Werte sind kein Fehler --
    dort steht der Ident oder die Programmvorschau.
    """
    S, E, neg = [], [], []
    for u, ans in anker.items():
        for (bs, be) in bloecke.get(u, []):
            inn = [(s, e) for (s, e, _, _) in ans if e > bs and s < be]
            if not inn:
                continue
            ds = min(s for s, _ in inn) - bs
            de = be - max(e for _, e in inn)
            S.append(ds)
            E.append(de)
            if ds < -0.5 or de < -0.5:
                neg.append((u, round(ds, 1), round(de, 1)))
    if not S:
        return None
    def q(v, p):
        return float(np.percentile(v, p))
    print(f"    Startversatz  Median {np.median(S):6.1f}s   10% {q(S,10):6.1f}   90% {q(S,90):6.1f}")
    print(f"    Endversatz    Median {np.median(E):6.1f}s   10% {q(E,10):6.1f}   90% {q(E,90):6.1f}")
    print(f"    Bloecke, die in einen Anker schneiden: {len(neg)}  <-- muss 0 sein")
    for x in neg[:5]:
        print(f"      {x[0]}  Start {x[1]}s  Ende {x[2]}s")
    return {"start_median": round(float(np.median(S)), 2),
            "ende_median": round(float(np.median(E)), 2),
            "schnitte": len(neg), "schnitt_liste": neg[:20]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="/tmp/anchors",
                    help="Verzeichnis mit cluster-anchored-Antworten")
    ap.add_argument("--bild", default=str(Path.home() / ".cache/tvd-wiederholung/korpus"),
                    help="Verzeichnis mit Ankern aus wiederholung.py --anker")
    ap.add_argument("--bloecke", default="/tmp/ads",
                    help="Verzeichnis mit /recording/<uuid>/ads-Antworten")
    ap.add_argument("--art", default="auto", choices=["auto", "user", "ads"],
                    help="welche Bloecke gemessen werden (auto = das Modell)")
    ap.add_argument("--json", help="Ergebnis zusaetzlich als JSON hierhin")
    a = ap.parse_args()

    bloecke = lade_bloecke(a.bloecke, a.art)
    print(f"{len(bloecke)} Aufnahmen mit {a.art}-Bloecken\n")
    erg = {"art": a.art}

    audio = lade_anker(a.audio)
    if audio:
        print("=== WAECHTER: Audio-Anker (gesaettigt, misst nur Rueckschritt) ===")
        erg["audio"] = bericht(audio, bloecke, "senderuebergreifend", nur_hart=True)
        erg["audio_versatz"] = versaetze(audio, bloecke)

    bild = lade_anker(a.bild) if Path(a.bild).is_dir() else {}
    if bild:
        print("\n=== BEFUND: Bild-Anker (unabhaengig von den Bloecken entstanden) ===")
        erg["bild"] = bericht(bild, bloecke, "wiederholte Folgen", nur_hart=False)
        erg["bild_versatz"] = versaetze(bild, bloecke)
    elif not audio:
        print("weder Audio- noch Bild-Anker gefunden"); return 1

    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
