#!/usr/bin/env python3
"""O16: Kantenfehler zweier Koepfe gegen den korrigierten Golden-Satz.

Misst fuer je einen Kandidaten-Kopf (--kopf) und einen Basis-Kopf (--basis)
den Blockkanten-Fehler gegen die Labels der bildgepruéften Golden-Aufnahmen
(= die 22 mit Agenten-Review; deren Gateway-Labels sind seit dem Audit vom
2026-08-17 auf <=5 s gegen das Bildmaterial bestaetigt, golden-audit.py).

Beide Koepfe laufen ueber DENSELBEN Pfad wie die naechtliche Bewertung:
Archiv-Features (.npy) -> Kopf-eigene Zusatzspalten (Header-Flags + eigene
Sidecars) -> predict_proba -> tv-detect --replay-signals mit dem
Produktions-Decoder (EVAL_DECODER aus train-head.py). Gepaart: gewertet
werden nur Aufnahmen, bei denen BEIDE Koepfe Bloecke liefern.

Kantenzuordnung (fixiert, fuer beide Koepfe identisch): je Label-Block der
Modell-Block mit dem naechsten Start; liegt min(|dStart|,|dEnde|) > 180 s,
zaehlt der Block als "ohne Partner" und faellt aus der Kantenwertung
(dieselbe Partner-Regel wie golden-audit.py).

⚠️ Das Werkzeug von §3ao (Baseline "Median 2.0 s / >10 s: 23 %") wurde nicht
persistiert — deshalb misst dieses Skript IMMER beide Koepfe selbst und
entscheidet auf dem gepaarten Delta, nicht gegen die alte Zahl. Die
Basis-Messung dient zugleich als Paritaets-Probe gegen die registrierten
Baseline-Werte.

Schreibt NICHTS (Archive, Labels, Gate bleiben unberuehrt).

Aufruf:
  ~/ml/tv-classifier/.venv/bin/python3 scripts/o16-kantenmass.py \
      --kopf ~/.cache/tv-train-head-out/archive/head.20260818T033011.bin \
      --basis ~/.cache/tv-train-head-out/archive/head.20260817T033010.bin
"""
import argparse
import importlib.util
import json
import ssl
import statistics as st
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent


def _lade(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), HIER / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


th = _lade("train-head.py")
ar = _lade("agent-review.py")

ARCHIV = Path.home() / ".cache/tvd-train-archive"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def label_vom_gateway(u):
    try:
        req = urllib.request.Request(f"{ar.GATEWAY}/recording/{u}/ads")
        with urllib.request.urlopen(req, context=_CTX, timeout=20) as r:
            d = json.loads(r.read())
        a = d.get("ads") if isinstance(d, dict) else d
        return [(float(x[0]), float(x[1])) for x in a] if a else None
    except Exception:
        return None


def kopf_laden(pfad):
    """Kopf + seine Zusatzspalten-Flags aus dem Header, Sidecars aus dem
    gleichen Verzeichnis/Stamm (head.<ts>.bin -> head.<ts>.channel-map.json
    bzw. unversioniert head.channel-map.json daneben)."""
    mlp = th.load_deployed_mlp(pfad)
    raw = Path(pfad).read_bytes()
    magic = struct.unpack("<I", raw[:4])[0]
    flags = dict(kanal=0, whisper=0, temporal=0, churn=0, mp=0, maske=0)
    if mlp is None:
        # MLP1 (v1, nackt oder mit Kanal) laedt load_deployed_mlp nicht —
        # nachbauen aus dem v1-Header (36 B: 9 uint32).
        if magic != 0x31504C4D:
            raise SystemExit(f"{pfad}: weder MLP1 noch MLP2+ — nicht lesbar")
        hdr = struct.unpack("<9I", raw[:36])
        input_dim, hidden = hdr[2], hdr[3]
        off = 36
        def take(n):
            nonlocal off
            a = np.frombuffer(raw, dtype=np.float32, count=n,
                              offset=off).astype(np.float64)
            off += n * 4
            return a
        W1 = take(input_dim * hidden).reshape(input_dim, hidden)
        b1 = take(hidden)
        W2 = take(hidden * hdr[4]).reshape(hidden, hdr[4])
        b2 = take(hdr[4])
        mlp = th._DeployedMLP(W1, b1, W2, b2, input_dim)
        flags["kanal"] = hdr[8]
    else:
        hdr = struct.unpack("<13I", raw[:52]) if magic == 0x35504C4D else None
        if hdr is None:
            raise SystemExit(f"{pfad}: nur MLP1/MLP5 verdrahtet — Header pruefen")
        (_, _, _, _, _, _, _, _, n_chan,
         n_whisper, n_temporal, n_mp, n_maske) = hdr
        flags.update(kanal=n_chan, whisper=n_whisper,
                     temporal=1 if n_temporal >= 2 else 0,
                     churn=1 if n_temporal == 3 else 0,
                     mp=n_mp, maske=n_maske)
    p = Path(pfad)
    stamm = p.name[:-len(".bin")] if p.name.endswith(".bin") else p.name
    chan_idx = {}
    if flags["kanal"]:
        for kand in (p.with_name(stamm + ".channel-map.json"),
                     p.parent / "head.channel-map.json",
                     p.parent.parent / "head.channel-map.json"):
            if kand.is_file():
                slugs = json.loads(kand.read_text()).get("slugs", [])
                if len(slugs) == flags["kanal"]:
                    chan_idx = {s: i for i, s in enumerate(slugs)}
                    break
        if not chan_idx:
            raise SystemExit(f"{pfad}: n_channel={flags['kanal']}, aber keine "
                             f"passende channel-map gefunden")
    mp_col = None
    if flags["mp"]:
        priors, neutral = {}, 0.25
        for kand in (p.with_name(stamm + ".minute-prior.json"),
                     p.parent / "head.minute-prior.json",
                     p.parent.parent / "head.minute-prior.json"):
            if kand.is_file():
                side = json.loads(kand.read_text())
                priors = {k: np.array(v, dtype=np.float32)
                          for k, v in (side.get("priors") or {}).items()}
                neutral = float(side.get("neutral", 0.25))
                break

        def mp_col(uuid, T, _p=priors, _n=neutral):
            m = META.get(uuid, {})
            slug, start = m.get("slug", ""), m.get("start_ts", 0)
            if start and slug in _p:
                minutes = ((start + np.arange(T)) // 60 % 60).astype(int)
                return _p[slug][minutes].reshape(-1, 1)
            return np.full((T, 1), _n, dtype=np.float32)

    return mlp, flags, chan_idx, mp_col


META = {}  # uuid -> npz-meta (slug, start_ts, feature_npy)


def bloecke(mlp, flags, chan_idx, mp_col, uuid):
    X = np.load(META[uuid]["feature_npy"])
    # Wie der Nightly: schmalere Alt-Features neutral auf 1282 auffuellen
    # (z. B. fehlende Logo-Spalte -> 0.5), s. train-head.py "Padding with
    # neutral 0.5".
    if X.shape[1] < 1282:
        X = np.concatenate(
            [X, np.full((X.shape[0], 1282 - X.shape[1]), 0.5,
                        dtype=X.dtype)], axis=1)
    Xa = th.mit_zusatz(X, uuid, META[uuid]["slug"], chan_idx,
                       n_chan=flags["kanal"] or None,
                       kanal=bool(flags["kanal"]),
                       whisper=bool(flags["whisper"]),
                       temporal=bool(flags["temporal"]),
                       churn=bool(flags["churn"]),
                       mp_col=mp_col, maske=bool(flags["maske"]))
    if Xa.shape[1] != mlp.input_dim:
        raise SystemExit(f"{uuid[:8]}: Featurebreite {Xa.shape[1]} != "
                         f"Kopf {mlp.input_dim}")
    proba = mlp.predict_proba(Xa)[:, 1]
    cache = th._signals_cache_path(uuid)
    if cache is None:
        return None
    return th._replay_blocks(cache, list(proba), 1.0, uuid)


def kanten(bloecke_modell, lab):
    fehler, ohne = [], 0
    for l in lab:
        g = min(bloecke_modell, key=lambda b: abs(b[0] - l[0]))
        if min(abs(g[0] - l[0]), abs(g[1] - l[1])) > 180:
            ohne += 1
            continue
        fehler.append(("start", abs(g[0] - l[0])))
        fehler.append(("ende", abs(g[1] - l[1])))
    return fehler, ohne


def zeile(name, fehler, ohne):
    w = [f for _, f in fehler]
    q = st.quantiles(w, n=20) if len(w) >= 20 else None
    starts = [f for s, f in fehler if s == "start"]
    enden = [f for s, f in fehler if s == "ende"]
    print(f"{name:8} n={len(w):3}  Median {st.median(w):4.1f} s  "
          f"p75 {st.quantiles(w, n=4)[2]:4.1f}  "
          f"p90 {(q[17] if q else max(w)):4.1f}  "
          f">10s {100 * sum(1 for f in w if f > 10) / len(w):3.0f} %  "
          f"ohne Partner {ohne}  "
          f"[Starts {st.median(starts):.1f} | Enden {st.median(enden):.1f}]")
    return 100 * sum(1 for f in w if f > 10) / len(w), st.median(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", required=True, help="Kandidat (z. B. voller Kopf)")
    ap.add_argument("--basis", required=True, help="Vergleichskopf (nackt)")
    args = ap.parse_args()

    golden = json.loads((ARCHIV / "golden-eval-set.json").read_text())["uuids"]
    arbeit = Path.home() / ".cache/tvd-agent-review"
    geprueft = [u for u in sorted(golden) if (arbeit / u / "auftrag.json").is_file()]

    for u in geprueft:
        z = np.load(ARCHIV / f"{u}.npz", allow_pickle=False)
        META[u] = json.loads(str(z["meta"]))

    k = kopf_laden(args.kopf)
    b = kopf_laden(args.basis)
    print(f"Kandidat {args.kopf} (dim {k[0].input_dim}), "
          f"Basis {args.basis} (dim {b[0].input_dim}), "
          f"{len(geprueft)} bildgepruefte Golden-Aufnahmen")

    fk, fb, ok, ob, n_rec = [], [], 0, 0, 0
    for u in geprueft:
        lab = label_vom_gateway(u)
        if not lab:
            print(f"  {u[:12]}: kein Label vom Gateway — ausgelassen")
            continue
        bk = bloecke(*k, u)
        bb = bloecke(*b, u)
        if not bk or not bb:
            print(f"  {u[:12]}: Replay fehlgeschlagen "
                  f"(Kandidat {'ok' if bk else 'LEER'}, "
                  f"Basis {'ok' if bb else 'LEER'}) — ausgelassen")
            continue
        n_rec += 1
        f1, o1 = kanten(bk, lab)
        f2, o2 = kanten(bb, lab)
        fk += f1; ok += o1
        fb += f2; ob += o2
    print(f"\n{n_rec} Aufnahmen gepaart gewertet")
    pk, mk = zeile("Kandidat", fk, ok)
    pb, mb = zeile("Basis", fb, ob)
    print(f"\nDelta (Kandidat - Basis): >10s {pk - pb:+.1f} pp, "
          f"Median {mk - mb:+.1f} s")
    print("O16-Regel: ERFUELLT wenn >10s um >=5 pp sinkt UND Median um "
          "<=0.5 s steigt (Registrierung o16-voller-kopf-preregistration.md)")


if __name__ == "__main__":
    sys.exit(main())
