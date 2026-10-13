#!/usr/bin/env python3
"""O34: Trenner-Ende mit Logo-Wächter, gemessen auf dem Messsatz.

Registrierung: docs/o34-trenner-ende-preregistration.md. Die Regel T und die
Entscheidungsregel stehen dort und sind hier nur umgesetzt — wer an den
Konstanten dreht, misst eine andere Frage.

Nur lesend: Dumps, Messsatz, menschliche Labels. Schreibt höchstens --json.
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(_HIER))

SCHWELLE, FENSTER, LUECKE_S = 0.95, (2.0, 45.0), 2.0
LOGO_MAX, LOGO_MIN_WERTE, RAND_S, GEGEN_S = 0.35, 25, 60.0, 120.0


def regel_t(bloecke, dump):
    """Liefert (neue Blöcke, [(alt, neu, ocr_marker)] der angefassten Enden)."""
    fps = float(dump.get("fps") or 25)
    bc = np.asarray(dump["bumper_confs"], np.float32)
    lg = np.asarray(dump["logo_confs"], np.float32)
    dauer = len(bc) / fps
    ocr = [float(o["time_s"]) for o in (dump.get("ocr_funde") or []) if o.get("werbemarker")]
    neu, angefasst = [], []
    for i, (a, e) in enumerate(bloecke):
        neu.append((a, e))
        if e > dauer - RAND_S:
            continue
        lo, hi = int((e + FENSTER[0]) * fps) + 1, min(len(bc) - 1, int((e + FENSTER[1]) * fps))
        treffer = np.nonzero(bc[lo:hi + 1] >= SCHWELLE)[0] + lo
        if len(treffer) == 0:
            continue
        f0 = p = int(treffer[0])
        for t in treffer[1:]:
            if t - p > LUECKE_S * fps:
                break
            p = int(t)
        stueck = lg[int(e * fps):f0]
        stueck = stueck[~np.isnan(stueck)]
        if len(stueck) < LOGO_MIN_WERTE or float(stueck.mean()) >= LOGO_MAX:
            continue
        e2 = (p + 1) / fps
        if i + 1 < len(bloecke) and e2 >= bloecke[i + 1][0]:
            continue
        neu[-1] = (a, e2)
        angefasst.append((e, e2, any(e <= t <= e2 for t in ocr)))
    return neu, angefasst


def naechstes(t, enden):
    return min((abs(t - x) for x in enden), default=float("inf"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json")
    a = ap.parse_args()
    import fehlerbudget as fb
    ms = json.loads(fb.MESSSATZ.read_text())
    ub = fb._lade("bs", "backbone-sonde.py").menschlabels()
    led = json.loads((Path.home() / ".cache/tvd-train-archive/split-ledger.json").read_text())
    zeilen, fehlend = [], collections.Counter()
    iou_k, iou_t = [], []
    for u in ms["uuids"]:
        truth, dp = ub.get(u), fb.DUMPS / f"{u}.json"
        if not truth:
            fehlend["kein Menschenlabel"] += 1
            continue
        if not dp.is_file():
            fehlend["kein Dump"] += 1
            continue
        dump = json.loads(dp.read_text())
        if not dump.get("bumper_confs") or not dump.get("logo_confs"):
            fehlend["ohne Trenner-/Logo-Spur"] += 1
            continue
        k = fb.replay(dp)
        if k is None:
            fehlend["Replay fehlgeschlagen"] += 1
            continue
        truth = fb._bl(truth)
        t, angefasst = regel_t(k, dump)
        iou_k.append(fb.block_iou(k, truth))
        iou_t.append(fb.block_iou(t, truth))
        enden = [e for _, e in truth]
        for alt, neu, marker in angefasst:
            fv, fn = naechstes(alt, enden), naechstes(neu, enden)
            zeilen.append({"uuid": u, "kanal": u[4:].rsplit("-", 1)[0], "eimer": led.get(u, "?"),
                           "alt": round(alt, 1), "neu": round(neu, 1), "fehler_k": round(fv, 1),
                           "fehler_t": round(fn, 1), "gegenkante": min(fv, fn) <= GEGEN_S,
                           "ocr_marker": marker})
    n = len(iou_k)
    print(f"Messsatz {ms['name']}: {n} gemessen, nicht gemessen: {dict(fehlend) or 'keine'}")
    mit = [z for z in zeilen if z["gegenkante"]]
    ohne = [z for z in zeilen if not z["gegenkante"]]
    aufn = len({z["uuid"] for z in mit})
    print(f"angefasste Enden: {len(zeilen)} ({len(mit)} mit Gegenkante aus {aufn} Aufnahmen, {len(ohne)} ohne)")
    for z in sorted(zeilen, key=lambda z: (z["kanal"], z["uuid"], z["alt"])):
        print(f"  {z['uuid']:34s} {z['eimer']:10s} {z['alt']:8.1f} -> {z['neu']:8.1f}  "
              f"Fehler {z['fehler_k']:6.1f} -> {z['fehler_t']:6.1f}"
              f"{'  OCR' if z['ocr_marker'] else ''}{'' if z['gegenkante'] else '  (ohne Gegenkante)'}")
    d_iou = (sum(iou_t) - sum(iou_k)) / n if n else 0.0
    print(f"block_iou Mittel: K {sum(iou_k)/n:.4f}  T {sum(iou_t)/n:.4f}  Δ {d_iou:+.4f}")
    erg = {"n": n, "angefasst": len(zeilen), "mit_gegenkante": len(mit), "aufnahmen": aufn,
           "iou_k": sum(iou_k) / n, "iou_t": sum(iou_t) / n, "zeilen": zeilen}
    if len(mit) < 15 or aufn < 8:
        print("==> O34 NICHT AUSWERTBAR (verlangt >= 15 Enden mit Gegenkante aus >= 8 Aufnahmen)")
        erg["urteil"] = "NICHT AUSWERTBAR"
    else:
        delta = [z["fehler_k"] - z["fehler_t"] for z in mit]
        med = statistics.median(delta)
        besser = sum(d > 0 for d in delta) / len(delta)
        schlechter = sum(d < -5 for d in delta) / len(delta)
        r1, r2, r3 = med >= 5 and besser >= 0.70, schlechter <= 0.10, d_iou >= 0
        print(f"  R1 Median Δ {med:+.1f} s, {besser:.0%} besser      {'hält' if r1 else 'VERFEHLT'} (>= +5 s, >= 70 %)")
        print(f"  R2 {schlechter:.0%} um > 5 s schlechter           {'hält' if r2 else 'VERFEHLT'} (<= 10 %)")
        print(f"  R3 Δ block_iou {d_iou:+.4f}                {'hält' if r3 else 'VERFEHLT'} (>= 0)")
        erg.update(median_delta=med, besser=besser, schlechter=schlechter,
                   urteil="ERFÜLLT" if r1 and r2 and r3 else "NICHT ERFÜLLT")
        print(f"==> O34 {erg['urteil']}")
        kan = collections.defaultdict(list)
        for z in mit:
            kan[z["kanal"]].append(z["fehler_k"] - z["fehler_t"])
        for kk, v in sorted(kan.items()):
            print(f"     {kk:16s} n={len(v):2d}  Median Δ {statistics.median(v):+.1f} s")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))


if __name__ == "__main__":
    main()
