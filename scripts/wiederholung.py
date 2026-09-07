#!/usr/bin/env python3
"""Findet wiederholte Bildfolgen im Merkmals-Cache — Werbung ohne ein Label.

WARUM
-----
Werbung WIEDERHOLT sich. Der Stapel nutzt das seit Monaten (tv-recorder
`spot.go`, Audio-Fingerprints), aber nur INNERHALB bereits gesetzter
Bloecke: `spot_extract.go` faehrt ffmpeg ueber die bestaetigten Werbe-
bloecke einer Aufnahme. Damit kann ein Anker per Konstruktion nie
ausserhalb eines Blocks liegen — als Kanten-Maßstab misst sich das Raster
selbst (`grobes_raster_misst_sich_selbst`). Ausserdem braucht die
Extraktion die Segmente auf dem Pi: 252 von 1028 Aufnahmen haben Anker,
der Rest ist laengst geloescht.

Der Merkmals-Cache hat beides nicht. Er liegt fuer 1028 Aufnahmen auf
Platte, eine Backbone-Einbettung je Sekunde, und deckt die GANZE Aufnahme
ab — nicht nur die Bloecke. Wer dort nach wiederholten Folgen sucht,
findet Werbespots an Stellen, an denen das Modell keinen Block hat. Das
ist der nicht-zirkulaere Teil, den die Audio-Kette strukturell nicht
liefern kann.

Gegenprobe in die andere Richtung: 96.3 % der unabhaengig entstandenen
Audio-Anker liegen in einer hier gefundenen Wiederholung. Die beiden
Verfahren sehen also dasselbe, nur reicht dieses weiter.

WAS HIER GEMESSEN WURDE (2026-09-07, Probe an Familie 3068, 10 Airings)
-----------------------------------------------------------------------
Die Vektoren sind nicht-negativ (ReLU), roher Cosinus hat deshalb einen
Sockel von ~0.68 und trennt kaum: 7 von 9 Airings getroffen, Abstand zum
besten Fehltreffer im Median +0.03, zwei Fehlgriffe. Nach Abzug des
globalen Mittelwerts: 9 von 9, Abstand +0.35. Das Zentrieren ist kein
Feinschliff, es ist die Bedingung dafuer, dass das Verfahren traegt.

WAS EINE WIEDERHOLUNG BEDEUTET — UND WAS NICHT
-----------------------------------------------
Eine wiederholte Folge ist NICHT automatisch Werbung. Vorspann, Abspann
und Rubriken-Bumper einer Serie wiederholen sich genauso, ueber alle
Folgen. Der erste Unterscheider war deshalb: kommt die Folge in FREMDEN
Titeln vor?

Der traegt fuer sich genommen nicht. Gemessen gegen die Labels von 143
menschlich reviewten Aufnahmen lag die Praezision ueber die Titelzahl
allein zwischen 47 % und 76 %, egal welche Schwelle. Auch die Erwartung,
SENDERUEBERGREIFENDE Wiederholung sei die haerteste Evidenz, hielt der
Messung nicht stand: senderuebergreifend kam auf hoechstens 71 %,
senderintern auf 73 % — beide zu schwach.

Der Grund steht in den Fehlpaaren: die haeufigsten sind
Simpsons/Futurama, Simpsons/SpongeBob, South Park/Futurama. Formal fremde
Titel und fremde Sender, aber flache 2D-Animation sieht bei 224x224
gleich aus. Dasselbe bei "Ab ins Beet!" gegen "Die Beet-Brueder" —
gleiche Produktion, gleicher Garten.

Was traegt, ist der SCORE des Laufs zusammen mit der Titelzahl. Die
Zahlen dazu stehen bei anker(). Kurz: Score allein 74 % bei 0.88, Score
0.85 mit drei fremden Titeln 94.5 %.

WARUM KEINE FAMILIEN GEBILDET WERDEN
------------------------------------
Der erste Entwurf schloss die Laeufe transitiv zu Familien (Union-Find),
wie es die Audio-Kette tut. Das kippt: 374k Kanten auf 80k Knoten liegen
weit ueber der Perkolationsschwelle, und eine einzige Riesenkomponente
verschluckte 78619 von 79926 Vorkommen. Ein Werbeblock teilt eben viel
Material mit dem naechsten.

Die Familienzugehoerigkeit wird auch gar nicht gebraucht. Gefragt ist
nur, WOMIT sich ein Intervall wiederholt, und das steht in seinen
DIREKTEN Partnern. Die Einordnung ist damit lokal und kann nicht
durchschlagen.

STUFEN
------
  --index      Deskriptoren aus dem Merkmals-Cache bauen (einmal teuer)
  --suchen     Paare auf der GPU, zu Laeufen verketten
  --anker      je Sekunde fremde Titel zaehlen, Anker schreiben
  --pruefen    gegen die Audio-Anker halten (misst DIESES Verfahren)

Voller Durchgang ueber 1028 Aufnahmen, Stand 2026-09-07:

    python3 scripts/wiederholung.py --index          #  30 s
    python3 scripts/wiederholung.py --suchen         # ~11 min, 336 Mio Paare
    python3 scripts/wiederholung.py --anker          # ~ 8 min
    python3 scripts/wiederholung.py --pruefen
    python3 scripts/anker-mass.py --bild ~/.cache/tvd-wiederholung/korpus

Die Anker sind NIRGENDS angeschlossen — nicht am Daemon, nicht am
naechtlichen Durchgang, nicht am Training. Der Go-Dekoder liest sie ueber
--spot-anchors unveraendert (Test TestLadeSpotAnkerBildform), aber
wirksam werden sie erst, wenn `spot_lp_w` in der Detect-Config ueber null
steht. Das ist eine Entscheidung, keine Voreinstellung.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "tvd-wiederholung"
FEATS = Path.home() / ".cache" / "tvd-features"
ARCHIV = Path.home() / ".cache" / "tvd-train-archive"
SNAPSHOT = Path("/tmp/tv-train-snapshot")

# Fensterbreite und Schrittweite der Deskriptoren. W=8 mittelt acht
# Sekunden — das hebt den Abstand zum Fehltreffer von +0.35 auf +0.42 und
# macht die Suche unempfindlich gegen ein bis drei Sekunden Versatz, was
# bei Schrittweite 4 noetig ist. Ein Spot von 15 s traegt so drei bis vier
# Deskriptoren; einer reicht als Saatkorn.
W = 8
SCHRITT = 4
DIM = 64


def _dateiliste():
    """uuid -> juengste Merkmalsdatei (mtime steckt im Namen)."""
    aus = {}
    for p in sorted(FEATS.glob("*.npy")):
        u = p.name.rsplit("-", 4)[0]
        aus[u] = p
    return aus


def _roh(pfad):
    a = np.load(pfad, mmap_mode="r")
    return np.array(a[:, :1280], dtype=np.float32, copy=True)


def kopf(text):
    print(f"\n=== {text} ===", flush=True)


# --------------------------------------------------------------- Stufe 1
def index(args):
    kopf("Deskriptoren bauen")
    dateien = _dateiliste()
    uuids = sorted(dateien)
    if args.kanal:
        uuids = [u for u in uuids if u.startswith(f"dvr-{args.kanal}-")]
    print(f"{len(uuids)} Aufnahmen")

    # Globaler Mittelwert und PCA-Basis aus einer Stichprobe. Beides wird
    # mitgeschrieben, damit --suchen und spaetere Laeufe dieselbe Achse
    # benutzen; eine neu gezogene Basis waere ein stiller Maßstabswechsel.
    rng = np.random.default_rng(0)
    stich = list(rng.choice(uuids, size=min(40, len(uuids)), replace=False))
    acc = np.zeros(1280, np.float64)
    n = 0
    teile = []
    for u in stich:
        X = _roh(dateien[u])
        acc += X.sum(0)
        n += X.shape[0]
        teile.append(X[::19])
    mu = (acc / max(n, 1)).astype(np.float32)
    S = np.vstack(teile) - mu
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-6
    _, sv, Vt = np.linalg.svd(S - S.mean(0), full_matrices=False)
    P = Vt[:DIM].astype(np.float32).T
    erkl = float((sv[:DIM] ** 2).sum() / (sv ** 2).sum())
    print(f"Achse aus {len(stich)} Aufnahmen, {n} Sekunden; d={DIM} haelt {erkl:.1%} der Varianz")

    desk, quelle, zeit = [], [], []
    t0 = time.time()
    for i, u in enumerate(uuids, 1):
        X = _roh(dateien[u]) - mu
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
        if X.shape[0] < W + SCHRITT:
            continue
        c = np.cumsum(np.vstack([np.zeros((1, 1280), np.float32), X]), 0)
        F = (c[W:] - c[:-W]) / W
        F = F[::SCHRITT]
        F = F @ P
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6
        desk.append(F.astype(np.float32))
        quelle.append(np.full(F.shape[0], i - 1, np.int32))
        zeit.append((np.arange(F.shape[0]) * SCHRITT).astype(np.int32))
        if i % 100 == 0:
            print(f"  {i}/{len(uuids)}  {time.time()-t0:.0f}s", flush=True)
    D = np.vstack(desk)
    CACHE.mkdir(parents=True, exist_ok=True)
    ziel = CACHE / (args.name + ".npz")
    np.savez(ziel, D=D, quelle=np.concatenate(quelle), zeit=np.concatenate(zeit),
             uuids=np.array(uuids, object), mu=mu, P=P, W=W, schritt=SCHRITT)
    print(f"{D.shape[0]} Deskriptoren ({D.nbytes/1e6:.0f} MB) -> {ziel}")
    return 0


# --------------------------------------------------------------- Stufe 2
def suchen(args):
    """Alle Paare ueber der Schwelle, dann zu Laeufen verketten.

    Speicher ist hier der Engpass, nicht Rechenzeit. Ein Block von 2048
    Zeilen gegen 938286 Spalten sind 7.7 GB je Aehnlichkeitsmatrix, und
    die Maske "gleiche Aufnahme" verdoppelt das noch einmal — der erste
    Versuch kam damit auf 35 s je Block statt 0.2 s. Deshalb wird auch
    ueber die SPALTEN gekachelt und die eigene Aufnahme nicht ueber eine
    Boolean-Maske ausgeblendet, sondern ueber ihren zusammenhaengenden
    Spaltenbereich: die Deskriptoren liegen nach Aufnahme sortiert.
    """
    import collections
    import torch
    kopf("Paare suchen")
    z = np.load(CACHE / (args.name + ".npz"), allow_pickle=True)
    D, quelle, zeit, uuids = z["D"], z["quelle"], z["zeit"], list(z["uuids"])
    schritt = int(z["schritt"])
    N = D.shape[0]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{N} Deskriptoren, {len(uuids)} Aufnahmen, Geraet {dev}, Schwelle {args.schwelle}")

    # Spaltenbereich je Aufnahme (quelle ist aufsteigend sortiert).
    grenzen = np.searchsorted(quelle, np.arange(len(uuids) + 1))

    T = torch.from_numpy(D).to(dev)
    RZ, CZ = args.block, args.spalten
    treffer = []
    t0 = time.time()
    for a in range(0, N, RZ):
        b = min(a + RZ, N)
        recs = np.unique(quelle[a:b])
        for c in range(0, N, CZ):
            d = min(c + CZ, N)
            S = T[a:b] @ T[c:d].T
            for r in recs:                       # eigene Aufnahme wegschneiden
                lo, hi = max(int(grenzen[r]), c), min(int(grenzen[r + 1]), d)
                if lo < hi:
                    zeilen = torch.from_numpy(
                        np.nonzero(quelle[a:b] == r)[0].astype(np.int64)).to(dev)
                    S[zeilen.unsqueeze(1),
                      torch.arange(lo - c, hi - c, device=dev).unsqueeze(0)] = -1.0
            # i<j SOFORT, nicht erst nach dem Sammeln: die Matrix ist
            # symmetrisch, und bei ~800 Mio Paaren ist die zweite Haelfte
            # nicht Ballast, sondern der Unterschied zwischen 6 und 24 GB.
            # int32 aus demselben Grund -- 938286 Indizes passen bequem.
            m = S >= args.schwelle
            if bool(m.any()):
                zi = torch.nonzero(m, as_tuple=False)
                gi = (zi[:, 0] + a).to(torch.int32)
                gj = (zi[:, 1] + c).to(torch.int32)
                behalt = gi < gj
                if bool(behalt.any()):
                    treffer.append(np.stack([
                        gi[behalt].cpu().numpy(),
                        gj[behalt].cpu().numpy(),
                        (S[zi[behalt, 0], zi[behalt, 1]] * 1000).to(torch.int32).cpu().numpy()], 1))
            del S, m
        if (a // RZ) % 50 == 0:
            n = sum(len(t) for t in treffer)
            print(f"  {b}/{N}  {time.time()-t0:.0f}s  {n} Paare", flush=True)
    P = np.vstack(treffer) if treffer else np.zeros((0, 3), np.int32)
    del treffer
    print(f"{len(P)} Paare ueber der Schwelle ({time.time()-t0:.0f}s)")

    # Verketten: ein echter Wiederholungs-Lauf zeigt sich als Folge von
    # Paaren mit KONSTANTEM Versatz zwischen zwei Aufnahmen. Ein einzelnes
    # Paar kann ein aehnliches Studiobild sein; drei in Reihe mit gleichem
    # Versatz sind dieselbe Sendeminute.
    # Vektorisiert, weil eine Python-Schleife ueber 400 Mio Paare nicht
    # traegt: nach (Aufnahme A, Aufnahme B, Versatz, Zeit) sortieren, dann
    # bricht ein Lauf genau dort, wo einer der drei Schluessel wechselt
    # oder die Zeitluecke groesser ist als zwei Schritte.
    qi, zi_ = quelle[P[:, 0]].astype(np.int64), zeit[P[:, 0]].astype(np.int64)
    qj, zj_ = quelle[P[:, 1]].astype(np.int64), zeit[P[:, 1]].astype(np.int64)
    off = zj_ - zi_
    sc = P[:, 2].astype(np.float32) / 1000.0
    del P
    ordn = np.lexsort((zi_, off, qj, qi))
    qi, qj, off, zi_, sc = qi[ordn], qj[ordn], off[ordn], zi_[ordn], sc[ordn]
    bruch = np.empty(len(qi), bool)
    bruch[0] = True
    bruch[1:] = ((qi[1:] != qi[:-1]) | (qj[1:] != qj[:-1]) | (off[1:] != off[:-1])
                 | ((zi_[1:] - zi_[:-1]) > schritt * 2))
    start = np.flatnonzero(bruch)
    ende = np.append(start[1:], len(qi))
    laenge = ende - start
    behalt = laenge >= args.min_lauf
    start, ende, laenge = start[behalt], ende[behalt], laenge[behalt]
    print(f"{len(start)} Laeufe nach Verkettung")
    summe = np.add.reduceat(sc, np.append(start, len(sc))[:-1]) if len(start) else np.zeros(0)
    # reduceat summiert bis zum naechsten Startindex -- fuer verworfene
    # Laeufe dazwischen waere das falsch, deshalb einzeln kumulieren.
    cum = np.concatenate([[0.0], np.cumsum(sc, dtype=np.float64)])
    summe = cum[ende] - cum[start]
    # Ablage als Array, nicht als JSON: 23.2 Mio Laeufe sind als JSON
    # 3.2 GB und beim Einlesen ein Vielfaches davon im Speicher. Als npz
    # sind es 650 MB und der Leser ist sofort da.
    a_s = zi_[start].astype(np.int32)
    a_e = (zi_[ende - 1] + W).astype(np.int32)
    ra = qi[start].astype(np.int32)
    rb = qj[start].astype(np.int32)
    ov = off[start].astype(np.int32)
    ziel = CACHE / (args.name + "-laeufe.npz")
    np.savez(ziel, ra=ra, rb=rb, a_s=a_s, a_e=a_e, off=ov,
             n=laenge.astype(np.int32),
             score=(summe / laenge).astype(np.float32),
             uuids=np.array(uuids, object))
    print(f"{len(start)} Laeufe (>= {args.min_lauf} Deskriptoren), "
          f"Median-Dauer {int(np.median(a_e - a_s))}s -> {ziel}")
    return 0


# --------------------------------------------------------------- Metadaten
GATEWAY = "https://raspberrypi5lan:8443"


def _grid_titel():
    """Titel der noch lebenden Aufnahmen vom tv-recorder, einmal gecacht.

    Archiv und Snapshot decken 816 der 1028 indizierten Aufnahmen ab. Das
    Grid holt weitere nach. Wer keinen Titel hat, wird als PARTNER
    uebersprungen -- er kann einen Beleg dadurch nur schwaechen, nie
    erfinden, und das ist die richtige Richtung fuer einen Zeugen.
    """
    cache = CACHE / "grid-titel.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    import ssl
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    aus = {}
    try:
        req = urllib.request.Request(f"{GATEWAY}/api/dvr/entry/grid_finished?limit=5000")
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            d = json.load(r)
        for e in (d.get("entries") or d.get("data") or []):
            t = e.get("title") or e.get("disp_title") or ""
            if e.get("uuid") and t:
                aus[e["uuid"]] = t
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(aus))
    except Exception as e:
        print(f"  DVR-Grid nicht erreichbar ({e}) -- nur Archiv und Snapshot")
    return aus


def _titeltext(t):
    """Titel als Zeichenkette — egal in welcher Form er ankommt.

    Der Snapshot liefert ihn sprachcodiert (`{"ger": "Micky Maus
    Wunderhaus+"}`, aus XMLTV), Archiv und Grid als nackte Zeichenkette.
    143 der 959 Eintraege sind Woerterbuecher. Ohne diese Normalisierung
    landet ein dict als Schluessel in der Titeltabelle — und das ist der
    seltene freundliche Fall, weil Python dabei laut abbricht statt still
    jeden dieser Titel als eigenen zu zaehlen.
    """
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        for k in ("ger", "de", "deu"):
            if isinstance(t.get(k), str):
                return t[k]
        for v in t.values():
            if isinstance(v, str):
                return v
    return ""


def metadaten():
    """uuid -> (titel, slug). Archiv, Snapshot, DVR-Grid."""
    aus = {}
    for p in ARCHIV.glob("*.npz"):
        try:
            m = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
            aus[m["uuid"]] = (_titeltext(m.get("title")), m.get("slug") or "")
        except Exception:
            pass
    if SNAPSHOT.is_dir():
        for d in SNAPSHOT.glob("_rec_*"):
            u = d.name[5:]
            f = d / "meta.json"
            if f.is_file():
                try:
                    m = json.loads(f.read_text())
                    aus[u] = (_titeltext(m.get("title")) or aus.get(u, ("", ""))[0],
                              m.get("slug") or aus.get(u, ("", ""))[1])
                except Exception:
                    pass
    for u, t in _grid_titel().items():
        if not aus.get(u, ("", ""))[0]:
            aus[u] = (_titeltext(t), aus.get(u, ("", ""))[1])
    return aus


def _slug(u, meta):
    s = meta.get(u, ("", ""))[1]
    if s:
        return s
    return u.rsplit("-", 1)[0][4:] if u.startswith("dvr-") else "?"



def lade_laeufe(name, min_score, min_dauer):
    """Laeufe als Spalten-Arrays lesen und gleich filtern.

    Gibt (uuids, ra, rb, a_s, a_e, off) zurueck -- alles int32, alles
    schon nach Score und Dauer gesiebt. Die alte JSON-Form bleibt
    lesbar, damit ein alter Index nicht neu gerechnet werden muss.
    """
    p_npz = CACHE / (name + "-laeufe.npz")
    if p_npz.is_file():
        z = np.load(p_npz, allow_pickle=True)
        m = (z["score"] >= min_score) & ((z["a_e"] - z["a_s"]) >= min_dauer)
        return (list(z["uuids"]), z["ra"][m], z["rb"][m],
                z["a_s"][m], z["a_e"][m], z["off"][m])
    L = json.loads((CACHE / (name + "-laeufe.json")).read_text())
    L = [x for x in L if x["score"] >= min_score and (x["a_e"] - x["a_s"]) >= min_dauer]
    uu = sorted({x["a"] for x in L} | {x["b"] for x in L})
    idx = {u: i for i, u in enumerate(uu)}
    A = np.array([[idx[x["a"]], idx[x["b"]], x["a_s"], x["a_e"], x["b_s"] - x["a_s"]]
                  for x in L], np.int32) if L else np.zeros((0, 5), np.int32)
    return uu, A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]



# --------------------------------------------------------------- Stufe 4
def anker(args):
    """Je Sekunde zaehlen, in wie vielen FREMDEN Titeln sie ebenfalls lief,
    und daraus Anker im Format der Audio-Anker schreiben.

    ZWEI SCHWELLEN, UND DIE ERSTE IST DIE WICHTIGE
    -----------------------------------------------
    Der Score eines Laufs traegt mehr als die Titelzahl. Geeicht mit
    --eichen gegen 137 Aufnahmen, die `label_herkunft.py` als MENSCHLICH
    ausweist (nicht gegen die auto-Bloecke -- sonst misst man das
    Verfahren am Modell statt an der Sendewirklichkeit):

        Score  Dauer  Titel   Praezision   Deckung
         0.85     16      3       92.3 %    72.0 %
         0.85     16      5       96.2 %    62.6 %
         0.88     16      2     **96.8 %    73.8 %**
         0.88     16      3       97.6 %    66.1 %
         0.90     16      2       97.9 %    67.7 %

    Voreinstellung ist deshalb Score 0.88, Dauer 16 s, zwei fremde Titel:
    das schlaegt die vorige Einstellung (0.85 / 3) auf BEIDEN Achsen.

    DIE DAUER IST DER FALSCHE HEBEL — UND ZWAR ANDERSHERUM ALS ERWARTET
    -------------------------------------------------------------------
    Die naheliegende Vermutung war, dass ein echter Spot-Wiederholer kurz
    und bildgenau ist und eine Stilaehnlichkeit lose, also durch eine
    hoehere Mindestdauer wegfaellt. Das Gegenteil stimmt. Bei Score 0.85:

        Dauer  Titel   Praezision   Deckung
           16      3       92.3 %    72.0 %
           24      3       94.2 %    51.6 %
           32      3       90.4 %    18.7 %
           40      3       79.7 %     3.8 %

    Ab 32 s FAELLT die Praezision. Die Stilverwechslungen sind die LANGEN
    Laeufe: eine ganze Szene aehnlich aussehender Actionfilm-Bilder haelt
    minutenlang durch, ein Werbespot ist nach 30 s vorbei. Wer die
    Mindestdauer hochdreht, waehlt den Fehler aus.

    Die beiden Haelften der Eichmenge stimmen ueberein (96.4 % / 73.1 %
    gegen 97.2 % / 74.4 %), die Zahlen sind also kein Anpassungsartefakt.
    Die WAHL der Einstellung fiel allerdings mit Blick auf alle 137
    Aufnahmen; sie ist Eichung, kein blinder Test.

    WORAN DAS VERFAHREN SCHEITERT
    ------------------------------
    Bei niedrigem Score verwechselt das Backbone Zeichentrick-Stile ueber
    Serien hinweg: die haeufigsten Fehlpaare sind Simpsons/Futurama,
    Simpsons/SpongeBob, South Park/Futurama -- verschiedene Titel, also
    formal "fremd", aber flache 2D-Animation sieht bei 224x224 eben
    gleich aus ([[backbone_liest_keinen_text]] ist derselbe Verlust an
    Aufloesung). Dieselbe Klasse: "Ab ins Beet!" gegen "Die
    Beet-Brueder", gleiche Produktion, gleicher Garten. Diese Fehltreffer
    liegen im Median 400 s vom naechsten Blockrand entfernt, also tief im
    Programm und nicht an der Kante. Der Score trennt sie, die Titelzahl
    allein nicht.

    Ausgabe traegt dieselben Feldnamen wie
    GET /api/internal/spot-fp/cluster-anchored/{uuid}, damit der bereits
    gebaute Weg (--spot-anchors im Go-Dekoder, internal/signals.SpotAnchor)
    sie ohne eine Zeile Anpassung liest.
    """
    import collections
    kopf("Anker schreiben")
    meta = metadaten()
    uuids, ra, rb, a_s, a_e, off = lade_laeufe(args.name, args.min_score, args.min_dauer)
    print(f"{len(ra)} Laeufe nach Score>={args.min_score} und Dauer>={args.min_dauer}s")

    def titel(u):
        return meta.get(u, ("", ""))[0]

    # 188 der 1028 indizierten Aufnahmen haben keinen Titel mehr: sie
    # stehen weder im Trainings-Archiv noch im DVR-Grid noch im
    # Label-Backup, und der Whisper-Index loescht seine Zeile zusammen
    # mit der Aufnahme (whisper.go:268). Ihre Merkmalsdatei ist alles,
    # was von ihnen uebrig ist. Als PARTNER fallen sie damit aus.
    #
    # --ersatztitel gibt jeder von ihnen einen eigenen Ersatztitel, aber
    # nur gegenueber Aufnahmen auf einem ANDEREN Sender. Zwei Aufnahmen
    # desselben Senders koennen zwei Folgen derselben Serie sein, und
    # dann waere der Vorspann ploetzlich "fremder Titel" -- genau der
    # Fehler, den der Unterscheider verhindern soll. Gemessen gegen 137
    # menschlich gelabelte Aufnahmen:
    #
    #   uebersprungen (Voreinstellung)   92.3 % Praezision, 72.0 % Deckung
    #   Ersatztitel immer                88.6 %              78.0 %
    #   Ersatztitel nur fremder Sender   91.5 %              73.8 %
    #
    # Die Option ist aus, weil 1.8 Punkte Deckung 0.8 Punkte Praezision
    # nicht wert sind, solange Schwestersender (ProSieben/Sixx/kabel eins)
    # Programm teilen und ein Ersatztitel dort doch dieselbe Serie
    # treffen kann.
    tid = {}
    for u in uuids:
        t = titel(u)
        if t:
            tid.setdefault(t, len(tid))
    tvon = np.array([tid.get(titel(u), -1) for u in uuids], np.int32)
    ohne_titel = np.array([not titel(u) for u in uuids])
    if args.ersatztitel:
        ersatz = len(tid)
        for i in np.flatnonzero(ohne_titel):
            tvon[i] = ersatz + int(i)
        print(f"  --ersatztitel: {int(ohne_titel.sum())} Aufnahmen ohne Titel bekommen "
              f"einen Ersatz, aber nur gegenueber fremden Sendern")
    slugvon = np.array([hash(_slug(u, meta)) for u in uuids], np.int64)

    # Je Aufnahme ein Satz je Sekunde -- als dict von Mengen, aber die
    # Laeufe kommen als Arrays herein, damit die 23 Mio Zeilen nicht als
    # Python-Objekte materialisiert werden muessen.
    sek = collections.defaultdict(lambda: collections.defaultdict(set))
    ohne = 0
    for seite in (0, 1):
        eig, part = (ra, rb) if seite == 0 else (rb, ra)
        st = a_s if seite == 0 else (a_s + off)
        en = a_e if seite == 0 else (a_e + off)
        tp = tvon[part]
        gut = tp >= 0
        if args.ersatztitel:
            # Ersatztitel zaehlen nur ueber Sendergrenzen hinweg.
            gut = gut & (~ohne_titel[part] | (slugvon[eig] != slugvon[part]))
        ohne += int((~gut).sum())
        for k in np.flatnonzero(gut):
            d = sek[uuids[eig[k]]]
            t_ = int(tp[k])
            for t in range(int(st[k]), int(en[k])):
                d[t].add(t_)
    if ohne:
        print(f"  {ohne} Laufenden ohne Titel im Partner uebersprungen "
              f"(Metadaten fehlen) — sie koennen nur schwaechen, nie erfinden")

    ZIEL = CACHE / args.name
    ZIEL.mkdir(parents=True, exist_ok=True)
    n_rec = n_ank = 0
    dauern = []
    verworfen = [0]
    for u, tm in sek.items():
        eig = tid.get(titel(u), -1)
        gut = sorted(t for t, ts in tm.items() if len(ts - {eig}) >= args.min_titel)
        if not gut:
            continue
        ank = []
        s = p = gut[0]
        for t in gut[1:]:
            if t - p <= 1:
                p = t
                continue
            if p + 1 - s >= args.min_anker:
                ank.append((s, p + 1))
            s = p = t
        if p + 1 - s >= args.min_anker:
            ank.append((s, p + 1))
        # Zu lange Anker wegwerfen, nicht kuerzen. Ein Werbeblock ist
        # selten laenger als fuenf Minuten; was darueber liegt, ist keine
        # Spotfolge mehr, sondern ein durchgehend aehnlicher Abschnitt --
        # meist eine Serie, die sich mit einer stilverwandten deckt. Ein
        # Kuerzen wuerde die Grenze irgendwohin setzen; SpotBoundaryLP
        # nimmt aber genau die Endpunkte als Uebergangs-Evidenz, und eine
        # geratene Grenze ist dort schaedlicher als gar keine.
        zu_lang = [x for x in ank if x[1] - x[0] > args.max_anker]
        ank = [x for x in ank if x[1] - x[0] <= args.max_anker]
        verworfen[0] += len(zu_lang)
        if not ank:
            continue
        # family_id/family_size fuellen wir mit der beobachteten Breite:
        # so viele fremde Titel, wie die schwaechste Sekunde des Ankers
        # noch traegt. Der Go-Dekoder liest FamilySize heute nicht aus
        # (das Gewicht ist je Anker konstant) -- das Feld bleibt trotzdem
        # ehrlich befuellt, damit ein Handlauf es lesen kann.
        aus = [{"window_start_s": float(a), "end_s": float(b),
                "family_id": -1,
                "family_size": int(min(len(tm[t] - {eig}) for t in range(a, b)))}
               for a, b in ank]
        (ZIEL / f"{u}.json").write_text(json.dumps({"uuid": u, "anchored": aus}))
        n_rec += 1
        n_ank += len(aus)
        dauern += [b - a for a, b in ank]
    print(f"{n_ank} Anker in {n_rec} Aufnahmen, Median-Dauer "
          f"{int(np.median(dauern)) if dauern else 0}s, "
          f"laengster {int(max(dauern)) if dauern else 0}s -> {ZIEL}/")
    if verworfen[0]:
        print(f"  {verworfen[0]} zu lange Belege verworfen (> {args.max_anker}s) "
              f"-- kein Spot, sondern ein stilverwandter Abschnitt")
    return 0


# --------------------------------------------------------------- Stufe 5
def pruefen(args):
    """Gegen die Audio-Anker halten — misst DIESES Verfahren, nicht das Modell.

    Die Audio-Anker sind der einzige Maßstab hier, der unabhaengig
    entstanden ist: Chromaprint + dHash auf dem Pi, ohne je eine
    Einbettung gesehen zu haben. Was sie NICHT koennen, ist ausserhalb
    eines Blocks liegen -- deshalb ist die Deckungsrichtung
    "Audio-Anker in Wiederholung" aussagekraeftig, die Gegenrichtung
    nicht.
    """
    import collections
    kopf("Gegen die Audio-Anker")
    ZIEL = CACHE / args.name
    if not ZIEL.is_dir():
        print("erst --anker laufen lassen"); return 1
    A = {}
    for f in Path(args.audio_anker).glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("anchored"):
            A[d["uuid"]] = d["anchored"]
    z = np.load(CACHE / (args.name + ".npz"), allow_pickle=True)
    imIndex = set(map(str, z["uuids"]))
    A = {u: v for u, v in A.items() if u in imIndex}
    print(f"{len(A)} Aufnahmen mit Audio-Ankern UND im Index "
          f"(der Index haelt {len(imIndex)} Aufnahmen)")
    tot = hit = 0
    fehlend = []
    for u, ans in A.items():
        p = ZIEL / f"{u}.json"
        eig = json.loads(p.read_text())["anchored"] if p.is_file() else []
        iv = [(x["window_start_s"], x["end_s"]) for x in eig]
        for a in ans:
            m = (a["window_start_s"] + a["end_s"]) / 2
            tot += 1
            if any(s <= m < e for s, e in iv):
                hit += 1
            else:
                fehlend.append(u)
    print(f"Audio-Anker (Mitte) von einer Wiederholung gedeckt: "
          f"{hit}/{tot} = {100*hit/max(tot,1):.1f}%")
    c = collections.Counter(fehlend)
    if c:
        print("Aufnahmen mit den meisten ungedeckten Audio-Ankern:")
        for u, n in c.most_common(6):
            print(f"  {u:<34} {n}")
    return 0


# --------------------------------------------------------------- Stufe 6
def menschlich_gelabelt(snapshot=SNAPSHOT):
    """uuids, deren ads_user.json nachweislich von einem Menschen stammt.

    Die EINE Definition steht in label_herkunft.py und wird hier geladen,
    nicht nachgebaut: eine Regel in zwei Kopien ist zwei Regeln, sobald
    eine angefasst wird.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lh", Path(__file__).resolve().parent / "label_herkunft.py")
    lh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lh)
    aus = {}
    for d in sorted(Path(snapshot).glob("_rec_*")) if Path(snapshot).is_dir() else []:
        f = d / "ads_user.json"
        if not f.is_file():
            continue
        try:
            roh = json.loads(f.read_text())
        except Exception:
            continue
        if lh.mensch_aus_markern(roh) is not True:
            continue
        bl = [(float(a), float(b)) for a, b in (roh.get("ads") or [])
              if float(b) > float(a)]
        if bl:
            aus[d.name[5:]] = bl
    return aus


def eichen(args):
    """Praezision und Deckung gegen MENSCHENLABEL, ueber ein Gitter.

    Warum gegen Menschenlabel und nicht gegen die auto-Bloecke: ein
    maschinelles Label ist die frueherere Ausgabe desselben Dekoders.
    Wer dagegen eicht, eicht das Verfahren auf das Modell, nicht auf die
    Sendewirklichkeit -- und 113 der 256 reviewten Aufnahmen im Snapshot
    sind maschinell (which_merged_ist_kein_mensch).

    Praezision = Anteil der belegten Sekunden, die im Werbeblock liegen.
    Deckung    = Anteil aller Werbesekunden, die belegt sind.
    """
    import collections
    kopf("Eichen gegen Menschenlabel")
    ub = menschlich_gelabelt()
    meta = metadaten()
    # Geeicht wird auf einer Haelfte, berichtet auf der anderen. Ohne das
    # waeren die Zahlen in-sample: dieselben 137 Aufnahmen haben die
    # Schwelle gewaehlt, die dann an ihnen gut aussieht.
    if args.haelfte is not None:
        import hashlib
        ub = {u: v for u, v in ub.items()
              if int(hashlib.sha1(u.encode()).hexdigest(), 16) % 2 == args.haelfte}
        print(f"Haelfte {args.haelfte}: {len(ub)} von den menschlich gelabelten Aufnahmen")
    else:
        print(f"{len(ub)} menschlich gelabelte Aufnahmen im Snapshot (ALLE, in-sample)")

    def titel(u):
        return meta.get(u, ("", ""))[0]

    schritte = sorted({args.min_dauer, 16, 20, 24, 32, 40})
    print(f"{'Score':>6}{'Dauer':>7}{'Titel':>7}{'Sekunden':>10}"
          f"{'Praezision':>12}{'Deckung':>10}")
    beste = None
    for md in schritte:
        uuids, ra, rb, a_s, a_e, off = lade_laeufe(args.name, args.min_score, md)
        tid = {}
        for u in uuids:
            t = titel(u)
            if t:
                tid.setdefault(t, len(tid))
        tvon = np.array([tid.get(titel(u), -1) for u in uuids], np.int32)
        imsatz = np.array([u in ub for u in uuids])
        tot = sum(sum(int(b) - int(a) for a, b in ub[u])
                  for u in ub if u in set(uuids))
        sek = collections.defaultdict(lambda: collections.defaultdict(set))
        for seite in (0, 1):
            eig, part = (ra, rb) if seite == 0 else (rb, ra)
            st = a_s if seite == 0 else a_s + off
            en = a_e if seite == 0 else a_e + off
            tp = tvon[part]
            gut = (tp >= 0) & imsatz[eig]
            for k in np.flatnonzero(gut):
                d = sek[uuids[eig[k]]]
                t_ = int(tp[k])
                for t in range(int(st[k]), int(en[k])):
                    d[t].add(t_)
        for mt in sorted({args.min_titel, 2, 3, 5}):
            tp_ = fp_ = 0
            for u, tm in sek.items():
                e_ = tid.get(titel(u), -1)
                bl = ub[u]
                for t, ts in tm.items():
                    if len(ts - {e_}) >= mt:
                        if any(a <= t < b for a, b in bl):
                            tp_ += 1
                        else:
                            fp_ += 1
            n = tp_ + fp_
            pr = 100 * tp_ / max(n, 1)
            dk = 100 * tp_ / max(tot, 1)
            # Nur Konfigurationen mit brauchbarer Deckung kommen fuer die
            # Empfehlung infrage. Ohne diese Schranke gewinnt immer die
            # schaerfste Einstellung mit 1 % Deckung -- formal die hoechste
            # Praezision, praktisch kein Anker.
            mark = ""
            if n and dk >= args.deckung_mindestens and (beste is None or pr > beste[0]):
                beste = (pr, dk, args.min_score, md, mt)
                mark = "  <-"
            print(f"{args.min_score:>6.2f}{md:>7}{mt:>7}{n:>10}"
                  f"{pr:>11.1f}%{dk:>9.1f}%{mark}", flush=True)
    if beste:
        print(f"\n  beste Praezision bei mindestens {args.deckung_mindestens:.0f}% Deckung: "
              f"Score {beste[2]}, Dauer {beste[3]}s, Titel {beste[4]} "
              f"-> {beste[0]:.1f}% bei {beste[1]:.1f}% Deckung")
    else:
        print(f"\n  keine Konfiguration erreicht {args.deckung_mindestens:.0f}% Deckung")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="korpus", help="Name des Index")
    ap.add_argument("--kanal", help="nur dieser Sender-Slug (beim Indizieren)")
    sub = ap.add_argument_group()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--suchen", action="store_true")
    ap.add_argument("--anker", action="store_true")
    ap.add_argument("--pruefen", action="store_true")
    ap.add_argument("--eichen", action="store_true")
    ap.add_argument("--haelfte", type=int, choices=[0, 1],
                    help="nur diese Haelfte der Menschenlabel (sha1(uuid) mod 2)")
    ap.add_argument("--deckung-mindestens", type=float, default=50.0,
                    dest="deckung_mindestens")
    ap.add_argument("--min-titel", type=int, default=2, dest="min_titel")
    ap.add_argument("--min-anker", type=int, default=8, dest="min_anker")
    ap.add_argument("--max-anker", type=int, default=300, dest="max_anker")
    ap.add_argument("--ersatztitel", action="store_true",
                    help="Aufnahmen ohne Titel als eigener Titel zaehlen, "
                         "aber nur gegenueber fremden Sendern")
    ap.add_argument("--audio-anker", default="/tmp/anchors", dest="audio_anker")
    ap.add_argument("--min-score", type=float, default=0.88, dest="min_score")
    ap.add_argument("--min-dauer", type=int, default=16, dest="min_dauer")
    ap.add_argument("--schwelle", type=float, default=0.72)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--spalten", type=int, default=131072)
    ap.add_argument("--min-lauf", type=int, default=3, dest="min_lauf")
    a = ap.parse_args()
    if a.index:
        return index(a)
    if a.suchen:
        return suchen(a)
    if a.anker:
        return anker(a)
    if a.pruefen:
        return pruefen(a)
    if a.eichen:
        return eichen(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
