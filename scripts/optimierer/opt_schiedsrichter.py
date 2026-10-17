#!/usr/bin/env python3
"""Schiedsrichter: Qwen urteilt nur dort, wo Kontrolle und Arm sich uneinig sind.

WARUM
-----
Fuer den Vergleich zweier Koepfe zaehlt nur, wer in den Sekunden recht hat,
in denen ihre Bloecke voneinander abweichen -- ueberall sonst ist das Delta
null, egal was die Wahrheit ist. Diese Sekunden kann Qwen3.8-27B beurteilen
(binaer 91 % gegen Simons Wischantworten, bei Sicherheit >= 0.95 98 %). Damit
braucht ein Vergleich keine Menschenlabels mehr, und jede Aufnahme mit
Signal-Dump kann Messsatz werden. Bevor dem jemand traut, wird er hier gegen
die Menschenlabels geeicht (Unterbefehl `eichung`).

Qwen-Urteile sind NUR Messung, nie Trainingslabel (L2). Golden und Nachtgate
bleiben menschlich (L1, L3).

ABLAUF
------
  bloecke  <arm> --out B.json   (venv tv-classifier, torch)
      trainiert K und Arm genau wie opt_lauf.py (dessen main, unveraendert)
      und haelt zusaetzlich die dekodierten Bloecke je Seed und Aufnahme fest,
      dazu die Menschenwahrheit -- nur fuer die Eichung.
  urteil   B.json [B2.json ...] --out U.json   (venv vlm-exp, mlx)
      Uneinigkeits-Sekunden (Raster --abstand, hoechstens --max je Aufnahme)
      -> Bilder vom Pi (zieh.py, wie die Wischkarten) -> Qwen in zwei
      Durchgaengen (480 px und 320 px), sicher nur bei gleicher Antwort und
      beide >= --schwelle -> je Seed: Arm-richtig minus K-richtig, in
      Sekunden hochgerechnet. Dieselbe Rechnung mit der Menschenwahrheit auf
      DENSELBEN Sekunden steht daneben.
      Qwen-Antworten werden je (Aufnahme, Sekunde) zwischengespeichert, so
      teilen sich Arme die Kosten.

  lauf     NNN-name   (venv tv-classifier; ruft beide Teile als Prozesse)
      Schiedsrichter-Bestaetigung eines registrierten Laufs auf dem
      SCHIEDSRICHTER-SATZ (Ledger test, nicht im Messsatz, ohne Menschenlabel,
      eigene Signal-Dumps). Schreibt laeufe/NNN/schiedsrichter.json, genau
      einmal. Regel siehe urteil_schiri().

Beide Teile halten die Optimierer-Sperre (opt_lauf.schutz) -- Qwen und
Kopf-Training liegen nie gleichzeitig im Speicher.

Konvention (Simon 17.10.): Split-Screen-Werbung (Spot gross, Sendung im
kleinen Fenster) ist Werbung, ausser bei Let's Dance (dort Teil der Sendung).
Steht in der Qwen-Frage (vlm_wisch.FRAGE); der
Zwischenspeicher gilt nur fuer dieselbe Fassung der Frage.
"""
import hashlib
import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import opt_lauf as L  # noqa: E402

CACHE_Q = L.CACHE / "schiedsrichter-qwen.jsonl"
SCHIRI_SATZ = L.ARCH / "schiri-satz-2026-10-17.json"
SCHIRI_DUMPS = L.C / "tv-detect-daemon/emit-signals-schiri"
PY_TORCH = "/Users/simon/ml/tv-classifier/.venv/bin/python"
PY_MLX = str(Path.home() / "ml/vlm-exp/.venv/bin/python")
# Vorlaeufige Regel aus der Eichung 17.10. (README): Qwen ist nur an ~40 %
# der Uneinigkeitsstellen sicher und zaehlt deshalb weniger Sekunden als der
# Mensch; 018 (Entwicklung +0.0034) las +0.8 s je Aufnahme.
GEWINN_S_JE_AUFNAHME = 0.5
GEWINN_SEEDS = 4
BEST_MENSCH_MIN = -0.002  # menschliche Bestaetigung darf nicht klar schlechter sein
OFFS = (-6, -3, 0, 3, 6)
RAND = 6  # nur so weit vom Rand, dass alle 5 Bilder (t-6..t+6) existieren


# ---------------------------------------------------------------- Bloecke
def bloecke(a):
    """opt_lauf.main mit abgegriffenen Bloecken. bewerte() wird je Seed einmal
    aufgerufen, erst fuer K, dann fuer den Arm; replay() liefert die Bloecke."""
    aufrufe = []
    orig_bewerte, orig_replay = L.bewerte, L.replay
    schiri = a.satz == "schiedsrichter"

    def bewerte(p_eval, ev, truth, tmp, ursachen_aus=None):
        aufrufe.append({})
        if not schiri:
            return orig_bewerte(p_eval, ev, truth, tmp, ursachen_aus)
        # Ohne Menschenlabel gibt es keine IoU; nur die Bloecke zaehlen.
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(8) as ex:
            for f in [ex.submit(replay, u, p_eval[ev["rec"] == i], tmp)
                      for i, u in enumerate(ev["uuids"])]:
                f.result()
        return {}

    def replay(u, p, tmp):
        bl = orig_replay(u, p, tmp)
        if p is not None:
            aufrufe[-1][u] = bl
        return bl

    L.bewerte, L.replay = bewerte, replay
    satz = a.satz
    if schiri:
        uu = json.loads(SCHIRI_SATZ.read_text())["uuids"]
        orig_saetze = L.saetze

        def saetze():
            S = orig_saetze()
            assert not set(uu) & (set(S["training"]) | set(S["kalibrierung"]))
            S["entwicklung"] = uu
            return S
        L.saetze, L.DUMPS, L.wahrheit = saetze, SCHIRI_DUMPS, lambda x: {}
        satz = "entwicklung"
    roh = Path(a.out).with_suffix(".lauf.json")
    sys.argv = ["opt_lauf.py", "--arm", a.arm, "--satz", satz, "--seeds", str(a.seeds),
                "--json", str(roh)]
    L.main()
    erg = json.loads(roh.read_text())
    erg["satz"] = a.satz
    n = len(erg["seeds"])
    namen = list(erg["laeufe"])
    erg["bloecke"] = {nm: aufrufe[i * n:(i + 1) * n] for i, nm in enumerate(namen)}
    uu = sorted(erg["bloecke"][namen[0]][0])
    erg["wahrheit"] = L.wahrheit(uu)
    erg["dauer"] = {u: int(L.replay._fc[u] / L.replay._fps[u]) for u in uu}
    Path(a.out).write_text(json.dumps(erg))
    roh.unlink()
    print(f"Bloecke -> {a.out}")


# ---------------------------------------------------------------- Urteil
def maske(bl, n):
    m = np.zeros(n, bool)
    for s, e in bl or []:
        m[max(0, int(s)):min(n, int(e))] = True
    return m


def punkte(erg, abstand, hoechstens):
    """Rasterpunkte in der Vereinigung der Uneinigkeit ueber alle Seeds."""
    k, arm = [x for x in erg["bloecke"]]
    aus = {}
    for u, n in erg["dauer"].items():
        uneinig = np.zeros(n, bool)
        for bk, ba in zip(erg["bloecke"][k], erg["bloecke"][arm]):
            if u in bk and u in ba:
                uneinig |= maske(bk[u], n) != maske(ba[u], n)
        t = np.flatnonzero(uneinig)
        t = t[(t >= RAND) & (t < n - RAND) & (t % abstand == 0)]
        if len(t) > hoechstens:
            t = t[np.linspace(0, len(t) - 1, hoechstens).round().astype(int)]
        if len(t):
            aus[u] = [int(x) for x in t]
    return aus


def lade_cache(fassung):
    c = {}
    if CACHE_Q.is_file():
        for z in CACHE_Q.read_text().splitlines():
            r = json.loads(z)
            if r.get("frage") == fassung:
                c[(r["uuid"], r["t"])] = r
    return c


def titel_vom_pi(uuids):
    cmd = "for u in " + " ".join(uuids) + "; do f=$(ls /mnt/tv/hls/_rec_$u/*.txt 2>/dev/null | head -1); echo \"$u|$(basename \"$f\" .txt)\"; done"
    r = subprocess.run(["ssh", "raspberrypi5lan", cmd], capture_output=True, timeout=300)
    aus = {}
    for z in r.stdout.decode("utf-8", "replace").splitlines():
        u, _, t = z.partition("|")
        aus[u] = t.split(" $")[0].replace("_", ":") or None
    return aus


def ziehe(bedarf, tmp):
    """Alle Bilder in einem Zug vom Pi: je Aufnahme zusammenhaengende Laeufe
    im 3-s-Raster als zieh.py-Spezifikation."""
    specs = []
    for u, ts in bedarf.items():
        ts = sorted(ts)
        a = b = ts[0]
        for t in ts[1:] + [None]:
            if t is not None and t == b + 3:
                b = t
                continue
            specs.append(f"{u}:{a}:{b}:3")
            if t is not None:
                a = b = t
    cmd = ("rm -rf /tmp/schiri && mkdir -p /tmp/schiri && cd /home/simon/tv-tools && "
           "xargs -n 40 env ZIEH_OUT=/tmp/schiri ZIEH_BREITE=480 PYTHONIOENCODING=utf-8 python3 zieh.py "
           "&& cd /tmp/schiri && tar cf - .")
    r = subprocess.run(["ssh", "raspberrypi5lan", cmd], input="\n".join(specs).encode(),
                       capture_output=True, timeout=4 * 3600)
    if r.returncode != 0:
        sys.exit("Bilder vom Pi fehlgeschlagen: " + r.stderr.decode()[-300:])
    tarfile.open(fileobj=io.BytesIO(r.stdout)).extractall(tmp)


def urteil(a):
    sperre = L.schutz()  # noqa: F841 -- haelt die Sperre bis Prozessende
    sys.path.insert(0, str(L.SKRIPTE))
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_wisch", L.SKRIPTE / "qwen-wisch.py")
    qw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qw)
    from PIL import Image

    fassung = hashlib.sha1(qw.FRAGE.encode()).hexdigest()[:8]
    ergs = [json.loads(Path(p).read_text()) for p in a.bloecke]
    alle = [punkte(e, a.abstand, a.max) for e in ergs]
    cache = lade_cache(fassung)
    offen = {}
    for pk in alle:
        for u, ts in pk.items():
            for t in ts:
                if (u, t) not in cache:
                    offen.setdefault(u, set()).add(t)
    n_offen = sum(len(v) for v in offen.values())
    print(f"{sum(len(t) for pk in alle for t in pk.values())} Punkte, davon {n_offen} neu fuer Qwen", flush=True)
    if a.zeit and n_offen > a.zeit // 6:
        # Nur ein Teil in diesem Aufruf (Bilder ~1 s + Qwen ~4.5 s je Punkt);
        # Rueckgabe 3 heisst: erneut aufrufen, der Zwischenspeicher haelt den Rest.
        rest, offen = a.zeit // 6, {}
        for pk in alle:
            for u, ts in pk.items():
                for t in ts:
                    if rest and (u, t) not in cache and t not in offen.get(u, set()):
                        offen.setdefault(u, set()).add(t)
                        rest -= 1
        print(f"  davon jetzt {sum(len(v) for v in offen.values())} (--zeit {a.zeit})", flush=True)
        a.teil = True
    if n_offen:
        tmp = Path(tempfile.mkdtemp(prefix="schiri-"))
        bedarf = {u: {t + o for t in ts for o in OFFS} for u, ts in offen.items()}
        t0 = time.time()
        ziehe(bedarf, tmp)
        print(f"Bilder in {time.time() - t0:.0f} s", flush=True)
        titel = titel_vom_pi(sorted(offen))
        m = qw.Modell()
        t0, i = time.time(), 0
        with open(CACHE_Q, "a") as fh:
            for u, ts in offen.items():
                sender = u[4:].rsplit("-", 1)[0]
                for t in sorted(ts):
                    A = [tmp / f"{u}_{t + o:05d}.jpg" for o in OFFS]
                    if not all(p.is_file() for p in A):
                        r = {"uuid": u, "t": t, "fehlt": True, "frage": fassung}
                    else:
                        B = []
                        for p in A:
                            q = tmp / f"B_{p.name}"
                            if not q.is_file():
                                im = Image.open(p)
                                im.resize((320, round(im.height * 320 / im.width))).save(q, quality=80)
                            B.append(q)
                        wa, ca = qw.antwort(m.frage(A, sender, titel.get(u)))
                        wb, cb = qw.antwort(m.frage(B, sender, titel.get(u)))
                        r = {"uuid": u, "t": t, "a": wa, "ca": round(ca, 4), "b": wb, "cb": round(cb, 4),
                             "frage": fassung}
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    cache[(u, t)] = r
                    i += 1
                    if i % 50 == 0:
                        print(f"  {i}/{n_offen}  {(time.time() - t0) / i:.1f} s/Punkt", flush=True)

    if getattr(a, "teil", False):
        sys.exit(3)
    bericht = []
    for pfad, e, pk in zip(a.bloecke, ergs, alle):
        bericht.append(auswerten(e, pk, cache, a))
        bericht[-1]["datei"] = pfad
    Path(a.out).write_text(json.dumps(bericht, indent=1))
    for b in bericht:
        print(json.dumps({k: v for k, v in b.items() if k != "je_seed"}, ensure_ascii=False))


def qwen_binaer(r, schwelle):
    if r.get("fehlt") or r["a"] != r["b"] or min(r["ca"], r["cb"]) < schwelle:
        return None
    return "sendung" if r["a"] == "sendung" else "werbung"


def auswerten(e, pk, cache, a):
    k, arm = list(e["bloecke"])
    wahr = e["wahrheit"]
    je_seed = []
    stat = {"punkte": 0, "sicher": 0, "mit_mensch": 0, "qwen_gleich_mensch": 0}
    for u, ts in pk.items():
        for t in ts:
            stat["punkte"] += 1
            q = qwen_binaer(cache[(u, t)], a.schwelle)
            if q is None:
                continue
            stat["sicher"] += 1
            if u in wahr:
                stat["mit_mensch"] += 1
                mw = "werbung" if maske(wahr[u], t + 1)[t] else "sendung"
                stat["qwen_gleich_mensch"] += int(q == mw)
    for i, (bk, ba) in enumerate(zip(e["bloecke"][k], e["bloecke"][arm])):
        dq = dm = 0.0
        for u, ts in pk.items():
            if u not in bk or u not in ba:
                continue
            n = e["dauer"][u]
            mk, ma = maske(bk[u], n), maske(ba[u], n)
            mm = maske(wahr.get(u), n) if u in wahr else None
            for t in ts:
                if mk[t] == ma[t]:
                    continue
                q = qwen_binaer(cache[(u, t)], a.schwelle)
                if q is not None:
                    w = q == "werbung"
                    dq += (int(ma[t] == w) - int(mk[t] == w)) * a.abstand
                if mm is not None:
                    dm += (int(ma[t] == mm[t]) - int(mk[t] == mm[t])) * a.abstand
        je_seed.append({"seed": e["seeds"][i], "delta_qwen_s": dq, "delta_mensch_s": dm})
    # Menschliche Gesamtrechnung ueber ALLE Sekunden aus dem Lauf selbst (Fehlersekunden).
    voll = []
    for rk, ra in zip(e["laeufe"][k], e["laeufe"][arm]):
        voll.append(sum(rk["ursachen"].values()) - sum(ra["ursachen"].values()))
    n_auf = len(e["dauer"])
    return {"arm": arm, "satz": e.get("satz"), "aufnahmen": n_auf, **stat,
            "qwen_s_je_aufnahme": [round(s["delta_qwen_s"] / n_auf, 2) for s in je_seed],
            "qwen_trefferquote": round(stat["qwen_gleich_mensch"] / max(1, stat["mit_mensch"]), 3),
            "delta_qwen_s": [s["delta_qwen_s"] for s in je_seed],
            "delta_mensch_punkte_s": [s["delta_mensch_s"] for s in je_seed],
            "delta_mensch_voll_s": voll,
            "iou_delta": [round(ra["iou_mittel"] - rk["iou_mittel"], 4)
                          for rk, ra in zip(e["laeufe"][k], e["laeufe"][arm])],
            "je_seed": je_seed}


def urteil_schiri(b, best_mensch):
    """Gewinn: in >= GEWINN_SEEDS von 5 Seeds hat der Arm an den sicheren
    Qwen-Stellen mehr recht als K, und der Median liegt bei mindestens
    GEWINN_S_JE_AUFNAHME. Kandidat, wenn dazu die menschliche Bestaetigung
    (falls gemessen) nicht klar schlechter ist."""
    je = b["qwen_s_je_aufnahme"]
    gewinn = (sum(x > 0 for x in je) >= GEWINN_SEEDS
              and float(np.median(je)) >= GEWINN_S_JE_AUFNAHME)
    mensch_ok = best_mensch is None or best_mensch >= BEST_MENSCH_MIN
    return {"gewinn": bool(gewinn), "median_s_je_aufnahme": float(np.median(je)),
            "positive_seeds": int(sum(x > 0 for x in je)),
            "bestaetigung_mensch_median": best_mensch,
            "status": "kandidat" if gewinn and mensch_ok else "nicht_bestaetigt"}


def lauf(a):
    ld = L.REPO / "docs/optimierer/laeufe" / a.lauf
    ziel = ld / "schiedsrichter.json"
    if ziel.exists():
        sys.exit(f"{ziel} existiert schon -- jeder Lauf wird genau einmal geschiedsrichtert.")
    v = json.loads((ld / "vorschlag.json").read_text())
    u = json.loads((ld / "urteil.json").read_text())
    if not u.get("entwicklung", {}).get("erfuellt"):
        sys.exit("Schiedsrichter nur nach erfuelltem Entwicklungs-Urteil.")
    if L.code_abdruck(L.arm_laden(v["arm"])) != v["code"]:
        sys.exit("Code geaendert seit der Registrierung -- neu registrieren.")
    B = L.CACHE / "schiri" / f"lauf-{a.lauf}.json"
    U = B.with_suffix(".urteil.json")
    B.parent.mkdir(parents=True, exist_ok=True)
    # In Schritten unter 10 min: erst die Bloecke, dann Qwen in Teilen.
    # Rueckgabe 3 = noch nicht fertig, denselben Befehl erneut aufrufen.
    if not B.is_file():
        r = subprocess.run([PY_TORCH, __file__, "bloecke", v["arm"], "--satz", "schiedsrichter",
                            "--out", str(B)])
        if r.returncode != 0:
            sys.exit(f"bloecke gescheitert, Rueckgabe {r.returncode}")
        print("WEITER: Bloecke fertig, denselben Befehl erneut aufrufen")
        sys.exit(3)
    r = subprocess.run([PY_MLX, __file__, "urteil", str(B), "--out", str(U), "--zeit", "420"])
    if r.returncode == 3:
        print("WEITER: Qwen noch nicht fertig, denselben Befehl erneut aufrufen")
        sys.exit(3)
    if r.returncode != 0:
        sys.exit(f"urteil gescheitert, Rueckgabe {r.returncode}")
    b = json.loads(U.read_text())[0]
    best = u.get("bestaetigung", {}).get("median_delta")
    erg = {"lauf": a.lauf, "satz": json.loads(SCHIRI_SATZ.read_text())["hash"],
           **{k: b[k] for k in ("aufnahmen", "punkte", "sicher", "delta_qwen_s", "qwen_s_je_aufnahme")},
           **urteil_schiri(b, best)}
    ziel.write_text(json.dumps(erg, indent=1))
    print(json.dumps(erg, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bloecke")
    b.add_argument("arm")
    b.add_argument("--out", required=True)
    b.add_argument("--satz", default="entwicklung",
                   choices=["entwicklung", "bestaetigung", "schiedsrichter"])
    b.add_argument("--seeds", type=int, default=5)
    u = sp.add_parser("urteil")
    u.add_argument("bloecke", nargs="+")
    u.add_argument("--out", required=True)
    u.add_argument("--abstand", type=int, default=12, help="Raster der Punkte in s (Vielfaches von 3)")
    u.add_argument("--max", type=int, default=40, help="hoechstens so viele Punkte je Aufnahme")
    u.add_argument("--schwelle", type=float, default=0.95)
    u.add_argument("--zeit", type=int, default=0,
                   help="nur so viele Sekunden Qwen-Arbeit, dann Rueckgabe 3 (fortsetzbar)")
    w = sp.add_parser("lauf")
    w.add_argument("lauf", help="registrierter Lauf NNN-name")
    a = ap.parse_args()
    {"bloecke": bloecke, "urteil": urteil, "lauf": lauf}[a.cmd](a)


if __name__ == "__main__":
    main()
