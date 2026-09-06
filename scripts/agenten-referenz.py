#!/usr/bin/env python3
"""Eine vom Modell unabhängige Messreihe — ohne menschliches Review.

Das Problem, das hier gelöst wird: der Maßstab lebt von menschlichen
Labeln, und er bekommt keine mehr. Neue Labels sind Modellausgabe
(auto-confirm), und ein Maßstab aus Modellausgabe misst nur noch, ob das
Modell mit sich selbst übereinstimmt. Der versiegelte Satz — die
Gegenprobe — hat 0 von 37 menschliche Labels und hat am 2026-09-06 genau
so ein wertloses Gutachten ausgestellt (0.9867 gegen golden 0.9665).

DIE IDEE
--------
Agenten klassifizieren einzelne, zufällig gezogene Bilder aus den
versiegelten Aufnahmen: Werbung oder Sendung. Das können sie nachweislich
(2026-09-06: 112 Bilder, keine einzige falsche Kontrolle). Das Urteil wird
**eingefroren** — die Bilder ändern sich nie, also muss niemand zweimal
hinsehen. Danach wird wöchentlich nur noch gefragt: *stimmt die aktuelle
Modellausgabe an diesen Stellen mit dem eingefrorenen Urteil überein?*
Daraus entsteht eine Reihe, die man lesen kann.

WAS ES MISST, UND WAS NICHT
---------------------------
Bild-Genauigkeit an festen Stichpunkten. NICHT die Block-Überlappung des
Golden-Satzes: Agenten sind blind sehr gut im Klassifizieren, aber
schlecht im Übergang-Finden (Memory agent_als_kantenmassstab). Die Zahlen
sind also mit `golden_median` nicht vergleichbar — sie sind eine eigene
Reihe mit eigener Frage: *wird es über Wochen besser oder schlechter?*

⚠️ DREI LEITPLANKEN, JEDE AUS EINEM BEZAHLTEN FEHLER
----------------------------------------------------
1. **Das Urteil ist KEIN Label.** Es liegt unter ~/.cache/tvd-agenten-
   referenz/, nie in einem _rec_-Verzeichnis, und wird nie an die
   ads-API geschickt. Ein Agentenurteil, das ins Training läuft, ist der
   Echo-Kreis aus kanten_massstab_ist_agent_echo. Hier misst es nur.
2. **Kontrollbilder je Aufnahme.** Zwei Bilder mit bekannter Antwort
   (Mitte eines langen Blocks, Mitte der längsten werbefreien Strecke).
   Wer sie falsch beantwortet, dessen Urteil zählt nicht — sonst ist ein
   erfundenes Urteil von einem gesehenen nicht zu unterscheiden
   (agent_liefert_ohne_zu_lesen). Die Kontrollen stammen aus dem
   aktuellen Label; das ist zulässig, weil sie aus der MITTE langer
   Abschnitte gezogen werden, wo auch ein maschinelles Label stimmt.
3. **Anonyme, gemischte Bilder, feste Stichpunkte.** Dateinamen tragen
   keine Zeit; die Zuordnung liegt in _loesung.json, die der Agent nicht
   liest. Die Stichpunkte sind je uuid deterministisch (Seed aus der
   uuid), damit die Reihe über Wochen dieselben Stellen misst — dieselbe
   Kompositions-Konstanz, die den Golden-Satz überhaupt lesbar macht.

VERGLEICHSGRÖSSE
----------------
`auto` aus /recording/<uuid>/ads — die aktuelle Ausgabe des ausgelieferten
Detektors. NICHT `ads`: das ist die zusammengeführte Ansicht mit dem
DVR-Nachlauf-Block, der kein Label ist (Ledger 2026-09-06). Wo ein
`user`-Label existiert, wird trotzdem `auto` verglichen — gemessen werden
soll der Detektor, nicht das Label.

Ausführen:
  --vorbereiten            Aufträge für die versiegelten Aufnahmen bauen
  (Agenten sichten die Bilder, schreiben urteil.json)
  --einfrieren             Urteile prüfen und in referenz.json festhalten
  --messen                 aktuelle Modellausgabe gegen die Referenz halten,
                           eine Zeile an agenten-referenz-trend.jsonl anhängen
"""
import argparse
import hashlib
import importlib.util
import json
import random
import ssl
import sys
import time
import urllib.request
from pathlib import Path

_HIER = Path(__file__).resolve().parent


def _lade(name):
    sp = importlib.util.spec_from_file_location(name.replace("-", "_").replace(".py", ""),
                                                _HIER / name)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


_ar = _lade("agent-review.py")            # frames_ziehen, KONVENTION, QUELLE, GATEWAY, CTX
_fp = _lade("folgen-vergleich-pruefen.py")  # kontrollpunkte

ARBEIT = Path.home() / ".cache/tvd-agenten-referenz"
ARCHIV = Path.home() / ".cache/tvd-train-archive"
REFERENZ = ARBEIT / "referenz.json"
TREND = ARCHIV / "agenten-referenz-trend.jsonl"
N_PUNKTE = 6        # Stichpunkte je Aufnahme
RAND_S = 60         # nicht in den ersten/letzten 60 s ziehen


def hole(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def versiegelte_mit_quelle():
    led = json.loads((ARCHIV / "split-ledger.json").read_text())
    ver = sorted(u for u, e in led.items() if e == "versiegelt")
    return [u for u in ver if (_ar.QUELLE / f"{u}.ts").is_file()]


def stichpunkte(uuid, dauer):
    """Deterministisch je uuid — die Reihe misst jede Woche dieselben Stellen."""
    rng = random.Random(int(hashlib.sha256(uuid.encode()).hexdigest()[:12], 16))
    lo, hi = RAND_S, max(RAND_S + 1, int(dauer) - RAND_S)
    return sorted(rng.sample(range(lo, hi), min(N_PUNKTE, hi - lo)))


def in_block(t, bloecke):
    return any(a <= t < b for a, b in bloecke)


def vorbereiten(args):
    uuids = versiegelte_mit_quelle()
    print(f"{len(uuids)} versiegelte Aufnahme(n) mit lokaler Quelle")
    gebaut = 0
    for uuid in uuids:
        ziel = ARBEIT / uuid
        if (ziel / "auftrag.json").exists() and not args.neu:
            continue
        try:
            a = hole(f"{args.pi}/recording/{uuid}/ads")
        except Exception as e:
            print(f"  ⚠️ {uuid}: {e}", file=sys.stderr)
            continue
        dauer = float(a.get("duration_s") or 0)
        label = a.get("user")
        if label is None:
            label = a.get("auto") or []
        bloecke = [[float(x), float(y)] for x, y in label]
        if dauer < 4 * RAND_S:
            continue
        pts = stichpunkte(uuid, dauer)
        k_werb, k_send = _fp.kontrollpunkte(bloecke, dauer, (-1, -1))
        punkte = [(float(t), "streit") for t in pts]
        if k_werb is not None:
            punkte.append((k_werb, "kontrolle_werbung"))
        if k_send is not None:
            punkte.append((k_send, "kontrolle_sendung"))
        if len(punkte) < N_PUNKTE + 2:
            print(f"  ⚠️ {uuid}: keine sauberen Kontrollen — übersprungen")
            continue
        rng = random.Random(uuid)
        rng.shuffle(punkte)
        roh = ziel / "_roh"
        gezogen = _ar.frames_ziehen(uuid, [t for t, _ in punkte], roh)
        if len(gezogen) < len(punkte):
            print(f"  ⚠️ {uuid}: nur {len(gezogen)}/{len(punkte)} Bilder")
            continue
        loesung = {}
        for i, (t, rolle) in enumerate(punkte, 1):
            name = f"bild_{i:02d}.jpg"
            (roh / gezogen[f"{t:.0f}"]).rename(ziel / name)
            loesung[name] = {"t": round(t, 1), "rolle": rolle}
        roh.rmdir()
        (ziel / "_loesung.json").write_text(json.dumps({
            "uuid": uuid, "punkte": loesung, "dauer": dauer,
            "quelle_bytes": (_ar.QUELLE / f"{uuid}.ts").stat().st_size,
        }, ensure_ascii=False, indent=2))
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
            "ausgabe": ('Schreibe urteil.json in dieses Verzeichnis: '
                        '{"bild_01.jpg": "<kategorie>", ...} — ein Eintrag je Bild.'),
        }, ensure_ascii=False, indent=2))
        print(f"  ✓ {uuid}  {len(punkte)} Bilder")
        gebaut += 1
    print(f"\n{gebaut} Auftrag/Aufträge unter {ARBEIT}")
    return 0


def einfrieren(args):
    """Urteile prüfen, Kontrollen anwenden, Referenz schreiben."""
    ref = json.loads(REFERENZ.read_text()) if REFERENZ.exists() else {}
    neu = verworfen = 0
    for d in sorted(ARBEIT.iterdir()):
        lp, up = d / "_loesung.json", d / "urteil.json"
        if not (lp.is_file() and up.is_file()):
            continue
        loes = json.loads(lp.read_text())
        uuid = loes["uuid"]
        if uuid in ref and not args.neu:
            continue
        try:
            urteil = json.loads(up.read_text())
        except Exception:
            continue
        deut = {b: _ar.KONVENTION.get(str(urteil.get(b, "")).strip().lower())
                for b in loes["punkte"]}
        kontrolle_falsch = False
        for b, info in loes["punkte"].items():
            if info["rolle"].startswith("kontrolle") and deut[b] is not None:
                soll = "werbung" if info["rolle"].endswith("werbung") else "sendung"
                if deut[b] != soll:
                    kontrolle_falsch = True
        if kontrolle_falsch:
            print(f"  ✗ {uuid}: Kontrolle falsch — Urteil verworfen")
            verworfen += 1
            continue
        punkte = [{"t": info["t"], "urteil": deut[b]}
                  for b, info in loes["punkte"].items()
                  if info["rolle"] == "streit" and deut[b] is not None]
        if len(punkte) < 3:
            print(f"  ✗ {uuid}: nur {len(punkte)} bestimmte Stichpunkte — verworfen")
            verworfen += 1
            continue
        ref[uuid] = {"eingefroren": time.strftime("%Y%m%dT%H%M%S"),
                     "quelle_bytes": loes.get("quelle_bytes"),
                     "punkte": sorted(punkte, key=lambda p: p["t"])}
        neu += 1
    REFERENZ.parent.mkdir(parents=True, exist_ok=True)
    REFERENZ.write_text(json.dumps(ref, ensure_ascii=False, indent=1))
    n_pkt = sum(len(v["punkte"]) for v in ref.values())
    print(f"\nReferenz: {len(ref)} Aufnahme(n), {n_pkt} Stichpunkte "
          f"({neu} neu, {verworfen} verworfen) → {REFERENZ}")
    return 0


def messen(args):
    if not REFERENZ.exists():
        sys.exit("keine Referenz — erst --vorbereiten, Agenten, --einfrieren")
    ref = json.loads(REFERENZ.read_text())
    kopf = "?"
    try:
        req = urllib.request.Request(f"{_ar.GATEWAY}/api/internal/detect-models/head.bin")
        with urllib.request.urlopen(req, context=_ar.CTX, timeout=30) as r:
            kopf = hashlib.sha256(r.read()).hexdigest()[:12]
    except Exception as e:
        print(f"  ⚠️ head.bin nicht lesbar ({e}) — Kopf-Kennung bleibt '?'")
    n = treffer = 0
    je_aufnahme = {}
    fehlend = []
    for uuid, eintrag in sorted(ref.items()):
        try:
            a = hole(f"{args.pi}/recording/{uuid}/ads")
        except Exception:
            fehlend.append(uuid)
            continue
        auto = [[float(x), float(y)] for x, y in (a.get("auto") or [])]
        t_ok = 0
        for p in eintrag["punkte"]:
            modell = "werbung" if in_block(p["t"], auto) else "sendung"
            n += 1
            if modell == p["urteil"]:
                treffer += 1
                t_ok += 1
        je_aufnahme[uuid] = round(t_ok / len(eintrag["punkte"]), 3)
    quote = treffer / n if n else None
    zeile = {"ts": time.strftime("%Y%m%dT%H%M%S"), "kopf": kopf,
             "n_aufnahmen": len(je_aufnahme), "n_punkte": n,
             "treffer": treffer, "quote": round(quote, 4) if quote is not None else None,
             "fehlend": fehlend, "je_aufnahme": je_aufnahme}
    if not args.trocken:
        TREND.parent.mkdir(parents=True, exist_ok=True)
        with open(TREND, "a") as f:
            f.write(json.dumps(zeile, ensure_ascii=False) + "\n")
    print(f"Agenten-Referenz {zeile['ts']}  Kopf {kopf}")
    print(f"  {len(je_aufnahme)} Aufnahmen, {n} Stichpunkte, "
          f"Übereinstimmung Modell↔Referenz: "
          f"{quote:.1%}" if quote is not None else "  keine Daten")
    schwach = sorted(je_aufnahme.items(), key=lambda kv: kv[1])[:5]
    if schwach:
        print("  schwächste:", ", ".join(f"{u[:26]} {q:.0%}" for u, q in schwach))
    if fehlend:
        print(f"  ⚠️ {len(fehlend)} Aufnahme(n) ohne Antwort vom Dienst")
    if args.trocken:
        print("  (Probelauf — nichts angehängt)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vorbereiten", action="store_true")
    ap.add_argument("--einfrieren", action="store_true")
    ap.add_argument("--messen", action="store_true")
    ap.add_argument("--neu", action="store_true",
                    help="vorhandene Aufträge/Referenzen überschreiben")
    ap.add_argument("--trocken", action="store_true",
                    help="--messen ohne an die Reihe anzuhängen")
    ap.add_argument("--pi", default="http://raspberrypi5lan:9984")
    args = ap.parse_args()
    if args.vorbereiten:
        return vorbereiten(args)
    if args.einfrieren:
        return einfrieren(args)
    if args.messen:
        return messen(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
