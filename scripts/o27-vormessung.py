#!/usr/bin/env python3
"""O27-Vormessung — kostet der Aufloesungs-Bruch der Logo-Spalte etwas?

DIE LAGE (Sweep 2026-09-25)
---------------------------
Das Training extrahiert die Logo-Spalte (Index 1280) mit dem Template bei
VOLLER Aufloesung (train-head.py extract_logo_per_second). Die Produktion
dekodiert halb (DETECT_DECODE_SCALE=0.5) mit skaliertem Template. Am Clip
gemessen: Template-Wert MAD 0.05 (prosieben) bzw. 0.10 (sat-1), Produktion
hoeher. Der Kopf lernt also eine andere Verteilung, als er im Betrieb sieht.

WAS HIER GEMESSEN WIRD (ohne neues Training, keine Behandlung)
--------------------------------------------------------------
Derselbe Kontrollarm wie O26 (train aus dem Archiv, 5 Seeds). Ausgewertet
wird jede test-Aufnahme mit Quelle ZWEIMAL, mit demselben Modell:
  voll  — Logo-Spalte wie im Training (Archiv)
  halb  — Logo-Spalte frisch, wie die Produktion sie baut
Delta F1 (halb - voll) = was der Bruch heute kostet. Nahe 0 → nichts tun.
Deutlich negativ → eine registrierte Frage (O27) lohnt, und diese Zahl
ist ihr Rauschmass-Anker.

GEGENPROBE: fuer einige Aufnahmen wird zusaetzlich VOLL frisch extrahiert;
das muss die Archiv-Spalte treffen, sonst misst der Vergleich etwas anderes.
"""
import argparse
import importlib.util
import json
import os
import stat
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
REPO = HIER.parent
LOGOS = Path.home() / ".cache/tv-detect-daemon/logos"
QUELLEN = Path.home() / ".cache/tv-detect-daemon/source"
TV_DETECT = str(Path.home() / ".local/bin/tv-detect")
SKALA = 0.5
# Eigener Cache, NICHT im Train-Archiv: die Spalten bei halber Aufloesung
# werden spaeter (O27) fuer die Behandlung gebraucht.
HALB_CACHE = Path.home() / ".cache/tvd-o27-logo-halb"
SNAPSHOT = Path("/tmp/tv-train-snapshot")


def _lade(name, pfad):
    spec = importlib.util.spec_from_file_location(name, pfad)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def slug_aus(uuid):
    # dvr-<slug>-<start>
    return uuid[4:].rsplit("-", 1)[0] if uuid.startswith("dvr-") else ""


def halb_wrapper(tmp, w, h):
    """tv-detect mit festen --decode-width/height, als eigenes Binary, damit
    extract_logo_per_second unveraendert benutzt werden kann."""
    p = Path(tmp) / f"tv-detect-{w}x{h}"
    p.write_text(f'#!/bin/sh\nexec "{TV_DETECT}" --decode-width {w} --decode-height {h} "$@"\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--gegenprobe", type=int, default=4,
                    help="so viele Aufnahmen zusaetzlich VOLL frisch extrahieren")
    ap.add_argument("--nur-mensch", action="store_true",
                    help="nur test-Aufnahmen mit menschlichem Label auswerten "
                         "(label_herkunft.mensch_aus_markern) — gegen die "
                         "Zirkularitaet: maschinelle Labels stammen vom Detektor, "
                         "der selbst halb aufgeloest dekodiert")
    ap.add_argument("--json")
    a = ap.parse_args()
    lh = _lade("lh_o27", HIER / "label_herkunft.py")

    def mensch(u):
        f = SNAPSHOT / f"_rec_{u}" / "ads_user.json"
        try:
            return lh.mensch_aus_markern(json.loads(f.read_text())) is True
        except Exception:
            return False

    th = _lade("th_o27", HIER / "train-head.py")
    dm = _lade("dm_o27", REPO / "daemon/tv-thumbs-daemon.py")
    o = _lade("o20_o27", HIER / "o20-klassen-split.py")

    print("Lade train …", flush=True)
    Xtr, ytr, rec_tr, u_tr = o.lade("train", None, a.schritt, mit_uuids=True)
    print("Lade test …", flush=True)
    Xte, yte, rec_te, u_te = o.lade("test", None, 1, mit_uuids=True)

    tmp = tempfile.mkdtemp(prefix="o27-")
    kandidaten = []
    for i, u in enumerate(u_te):
        slug = slug_aus(u)
        q = QUELLEN / f"{u}.ts"
        tpl = LOGOS / f"{slug}.logo.txt"
        if q.is_file() and tpl.is_file() and (not a.nur_mensch or mensch(u)):
            kandidaten.append((i, u, slug, q, tpl))
    print(f"test {len(u_te)} Aufnahmen, davon {len(kandidaten)} mit Quelle + Template",
          flush=True)

    HALB_CACHE.mkdir(parents=True, exist_ok=True)

    def extrahiere(k, voll=False):
        i, u, slug, q, tpl = k
        n = int((rec_te == i).sum())
        c = HALB_CACHE / f"{u}.npy"
        if not voll and c.is_file():
            v = np.load(c)
            if len(v) == n:
                return v
        y_off = th.detect_letterbox_offset(str(q))
        if voll:
            return th.extract_logo_per_second(str(q), str(tpl), n, TV_DETECT, y_offset=y_off)
        dst = Path(tmp) / f"{slug}.logo.scaled.txt"
        pfad, w, h = dm.rescale_logo_template(tpl, dst, SKALA)
        y = round(y_off * SKALA) if y_off > 0 else 0
        v = th.extract_logo_per_second(str(q), str(pfad), n, halb_wrapper(tmp, w, h), y_offset=y)
        np.save(c, v)
        return v

    with ThreadPoolExecutor(a.parallel) as ex:
        halb = list(ex.map(extrahiere, kandidaten))
    # Gegenprobe: voll frisch == Archiv?
    gp = kandidaten[:a.gegenprobe]
    with ThreadPoolExecutor(a.parallel) as ex:
        voll_frisch = list(ex.map(lambda k: extrahiere(k, voll=True), gp))
    for k, v in zip(gp, voll_frisch):
        arch = Xte[rec_te == k[0], 1280]
        ok = ~np.isnan(v)
        print(f"  Gegenprobe {k[1]}: MAD voll-frisch vs Archiv "
              f"{np.mean(np.abs(v[ok] - arch[ok])):.4f}", flush=True)

    # Dritte Variante: VOLL frisch mit dem AKTUELLEN Template. Trennt den
    # Aufloesungs-Effekt vom Template-Stand (Templates werden nachtrainiert;
    # das Archiv traegt den Stand zur Extraktionszeit).
    Xte_vf = Xte.copy()
    vf_ok = set()
    if a.gegenprobe >= len(kandidaten):
        for k, v in zip(gp, voll_frisch):
            m = rec_te == k[0]
            if len(v) == int(m.sum()):
                Xte_vf[m, 1280] = np.where(np.isnan(v), 0.5, v).astype(np.float32)
                vf_ok.add(k[0])
    Xte_halb = Xte.copy()
    auswahl = np.zeros(len(u_te), bool)
    mads = []
    for k, h in zip(kandidaten, halb):
        m = rec_te == k[0]
        h = np.where(np.isnan(h), 0.5, h).astype(np.float32)
        if len(h) != int(m.sum()):
            continue
        mads.append(float(np.mean(np.abs(h - Xte[m, 1280]))))
        Xte_halb[m, 1280] = h
        auswahl[k[0]] = True
    zeilen = auswahl[rec_te]
    print(f"ausgewertet: {int(auswahl.sum())} Aufnahmen, MAD halb vs voll "
          f"Median {np.median(mads):.3f} (min {min(mads):.3f}, max {max(mads):.3f})",
          flush=True)

    Xtr_s, Xte_s = o.standardisieren(Xtr, Xte)
    _, Xte_hs = o.standardisieren(Xtr, Xte_halb)
    _, Xte_vfs = o.standardisieren(Xtr, Xte_vf)
    wahr = (yte > 0).astype(np.int64)

    def f1_auf(p):
        pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
        return o.f1(pred[zeilen], wahr[zeilen])

    # EIN Fit je Seed, beide test-Varianten untereinander: dasselbe Modell
    # sieht einmal die Trainings-, einmal die Produktionsspalte. Zwei Fits
    # mit demselben Seed waeren auf MPS nicht garantiert identisch, und das
    # Fit-Rauschen gehoerte dann ins Delta.
    n_te = len(yte)
    dritte = bool(vf_ok)
    X2 = np.concatenate([Xte_s, Xte_hs] + ([Xte_vfs] if dritte else []))
    y2 = np.concatenate([yte, yte] + ([yte] if dritte else []))
    off = rec_te.max() + 1
    rec2 = np.concatenate([rec_te, rec_te + off] + ([rec_te + 2 * off] if dritte else []))
    erg = []
    for seed in range(a.seeds):
        p2 = o.fit_und_werte(Xtr_s, ytr, X2, y2, rec2, 2, seed, a.epochen, a.hidden)
        p_voll, p_halb = p2[:n_te], p2[n_te:2 * n_te]
        if dritte:
            fvf = f1_auf(p2[2 * n_te:])
            print(f"  Seed {seed}: F1 voll-frisch (aktuelles Template) {fvf:.4f}", flush=True)
        fv, fh = f1_auf(p_voll), f1_auf(p_halb)
        erg.append((fv, fh))
        print(f"  Seed {seed}: F1 voll {fv:.4f}  halb {fh:.4f}  Delta {fh - fv:+.4f}", flush=True)
    d = np.array([h - v for v, h in erg])
    print(f"\nDelta F1 (halb - voll): Median {np.median(d):+.4f}  "
          f"negativ in {int((d < 0).sum())} von {len(d)} Seeds")
    if a.json:
        Path(a.json).write_text(json.dumps({"erg": erg, "mad": mads,
                                            "n": int(auswahl.sum())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
