#!/usr/bin/env python3
"""Optimierer — ein Arm gegen die Kontrolle, offline, ohne Nacht.

Trainiert den Kontrollkopf K (Produktionszustand MLP7 wie O30: Backbone
1280 + Logo + Audio + 3 OCR-Spalten + SigLIP-PCA-64 + Indikator) und einen
Versuchsarm mit DENSELBEN Seeds und bewertet beide wie die Produktion:
Wahrscheinlichkeit je Sekunde -> Platt-Kalibrierung -> per-Frame-CSV ->
`tv-detect --replay-signals --replay-nn-csv` mit dem Produktions-HSMM ->
Block-IoU gegen das Menschenlabel.

SAETZE (fest, nicht vom Arm waehlbar)
-------------------------------------
  entwicklung   Messsatz-Aufnahmen im train-Eimer (76). Hier wird gesucht.
  bestaetigung  Messsatz-Aufnahmen im test-Eimer (22). EINMAL je Lauf, nur
                nach erfuelltem Entwicklungs-Urteil (opt_urteil.py).
  kalibrierung  40 train-Aufnahmen, fest per Hash gewaehlt: nur Platt.
  training      train-Eimer minus die drei oben, ohne die 90 O25-Aufnahmen
                (docs/archiv-ausschluss-o25.json, wie die Produktion).
Golden und versiegelter Satz kommen nicht vor. Der Arm sieht von
Entwicklung/Bestaetigung nur Merkmale, nie Labels.

WAS EIN ARM DARF (scripts/optimierer/arme/<name>.py, alles optional)
--------------------------------------------------------------------
  BESCHREIBUNG = "…"
  HYPER = {"hidden": 96, "epochen": 12, "lr": 1e-3, "batch": 8192,
           "dropout": 0.0, "weight_decay": 0.0, "ensemble": 3}
  def spalten(teil, X, ktx) -> X'        teil in training|kalibrierung|eval
  def gewichte(ktx) -> w (je train-Zeile, >= 0)
  def ziel(ktx) -> y' (je train-Zeile, 0/1)
  def modell(n_in, hyper) -> torch.nn.Module (Ausgabe 2 Logits)
  def nachbearbeiten(p, rec, ktx) -> p'  (je Sekunde, vor der Kalibrierung)
ktx: {"teil", "uuids", "rec", "meta" (uuid -> Archiv-Meta), "schritt",
      "y" (NUR bei training), "spalten_namen", "sek" (absolute Sekunde je Zeile)}.
DATEN-/LABEL-Arme: ziel() und gewichte() veraendern NUR die Trainingskopie
im Speicher (Labels umschreiben, Aufnahmen herausnehmen). Gemessen wird
weiter gegen die echten Menschenlabels — so laesst sich eine Labelkorrektur
beziffern, bevor jemand das echte Archiv anfasst (L2/L4 bleiben gewahrt).

Schutz: laeuft nicht zwischen 02:15 und 05:45 (Nachttraining), nicht neben
train-head.py, nicht bei weniger als 24 GB freiem Speicher, nie zweimal
gleichzeitig. Schreibt nichts ausser seiner Ergebnis-Datei und dem Cache
~/.cache/tvd-optimierer/.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
SKRIPTE = HIER.parent
REPO = SKRIPTE.parent
C = Path.home() / ".cache"
ARCH = C / "tvd-train-archive"
LEDGER = ARCH / "split-ledger.json"
MESSSATZ = ARCH / "messsatz-2026-09-07.json"
AUSSCHLUSS = REPO / "docs/archiv-ausschluss-o25.json"
DUMPS = C / "tv-detect-daemon/emit-signals-neu"
CACHE = C / "tvd-optimierer"
BIN = Path.home() / ".local/bin/tv-detect"
DEKODER = ["--decoder", "hsmm", "--hsmm-dur-w", "15", "--hsmm-inner-show-min", "90"]
N_KALIB = 40
SCHRITT = 4


def _lade(name, pfad):
    spec = importlib.util.spec_from_file_location(name, pfad)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- Schutz
def schutz():
    jetzt = dt.datetime.now().time()
    if dt.time(2, 15) <= jetzt < dt.time(5, 45):
        sys.exit("Nachtfenster 02:15-05:45: das Nachttraining braucht den Speicher.")
    r = subprocess.run(["pgrep", "-f", "train-head.py"], capture_output=True, text=True)
    if r.stdout.strip():
        sys.exit("train-head.py laeuft gerade — nicht daneben.")
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    seiten = {}
    for z in vm.splitlines()[1:]:
        if ":" in z:
            k, v = z.split(":", 1)
            try:
                seiten[k.strip()] = int(v.strip().rstrip("."))
            except ValueError:
                pass
    frei = (seiten.get("Pages free", 0) + seiten.get("Pages inactive", 0)
            + seiten.get("Pages purgeable", 0)) * 16384 / 2**30
    if frei < 24:
        sys.exit(f"nur {frei:.0f} GB frei, brauche 24 (Gipfel eines Laufs ~16 GB).")
    CACHE.mkdir(parents=True, exist_ok=True)
    sperre = open(CACHE / "lauf.lock", "w")
    try:
        fcntl.flock(sperre, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("ein anderer Optimierer-Lauf haelt die Sperre.")
    return sperre


# ---------------------------------------------------------------- Saetze
def saetze():
    led = json.loads(LEDGER.read_text())
    ms = json.loads(MESSSATZ.read_text())["uuids"]
    entw = sorted(u for u in ms if led.get(u) == "train")
    best = sorted(u for u in ms if led.get(u) == "test")
    rest = sorted((u for u, v in led.items() if v == "train" and u not in set(ms)),
                  key=lambda u: hashlib.sha1(u.encode()).hexdigest())
    kalib = sorted(rest[:N_KALIB])
    # Wie die Produktion seit 24.09. (--archiv-ausschluss): die 90 O25-Aufnahmen
    # nicht im Fit. Lauf 014 hat das als Arm gemessen (Kandidat, nicht
    # schlechter) -- seit 17.10. gehoert es zu K. Die Kalibrierung bleibt
    # unberuehrt, damit K genau dem gemessenen Arm entspricht.
    raus = set(json.loads(AUSSCHLUSS.read_text())["uuids"])
    train = sorted(u for u in rest[N_KALIB:] if u not in raus)
    return {"training": train, "kalibrierung": kalib,
            "entwicklung": entw, "bestaetigung": best}


def meta_von(u):
    f = ARCH / f"{u}.npz"
    if not f.is_file():
        return None
    try:
        return json.loads(str(np.load(f, allow_pickle=True)["meta"]))
    except Exception:
        return None


def lade(uuids, schritt, mit_label):
    """Wie o20.lade, aber fuer eine feste uuid-Liste. Labels nur auf Wunsch."""
    Xs, ys, recs, geladen, metas = [], [], [], [], {}
    for u in uuids:
        m = meta_von(u)
        if not m:
            continue
        fp = m.get("feature_npy", "")
        if not fp or not os.path.exists(fp):
            continue
        if mit_label and not m.get("ads"):
            continue
        F = np.asarray(np.load(fp, mmap_mode="r"), dtype=np.float32)
        n = F.shape[0]
        if F.shape[1] == 1281:
            F = np.concatenate([F[:, :1280], np.full((n, 1), 0.5, np.float32), F[:, 1280:]], 1)
        if F.shape[1] != 1282:
            continue
        if np.isnan(F).any():
            F = np.nan_to_num(F, nan=0.5)
        y = np.zeros(n, np.int64)
        if mit_label:
            for a, b in m["ads"]:
                y[max(0, int(a)):min(n, int(b))] = 1
        if schritt > 1:
            F, y = F[::schritt], y[::schritt]
        Xs.append(F)
        ys.append(y)
        recs.append(np.full(F.shape[0], len(geladen), np.int32))
        geladen.append(u)
        # Ausserhalb des Trainings sieht der Arm nur, was auch die Produktion
        # bei der Erkennung weiss: Titel, Sender, Zeit. Nichts vom Label.
        metas[u] = m if mit_label == "training" else {
            k: m.get(k) for k in ("uuid", "title", "slug", "start_ts", "cohort")}
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(recs), geladen, metas


def baue_k(teile):
    """Kontroll-Merkmale fuer alle Teile, mit den PRODUKTIONS-Definitionen:
    OCR aus ocr_spalten.py, SigLIP aus siglip_spalten.py (Projektion wie
    train-head.siglip_projektion, aber nur aus dem Trainingsteil)."""
    oc = _lade("ocr_spalten_opt", SKRIPTE / "ocr_spalten.py")
    sl = _lade("siglip_spalten_opt", SKRIPTE / "siglip_spalten.py")
    spur = C / "tvd-siglip2"
    zeilen = []
    for u in teile["training"]["uuids"]:
        E = sl.lade_spur(u, spur)
        if E is not None and len(E):
            zeilen.append(np.asarray(E[::10], np.float32))
    mu, V = sl.projektion_anpassen(np.concatenate(zeilen))
    for t, d in teile.items():
        zus = np.zeros((len(d["rec"]), 3 + sl.N_SPALTEN), np.float32)
        for i, u in enumerate(d["uuids"]):
            m = d["rec"] == i
            k = int(m.sum())
            voll = k * d["schritt"]
            o = oc.ocr_spalten(u, voll, spur_dir=C / "tvd-ocr-spur")
            s = sl.spalten(sl.lade_spur(u, spur), voll, mu, V)
            zus[m] = np.concatenate([o, s], 1)[::d["schritt"]][:k]
        d["X"] = np.concatenate([d["X"], zus], 1)
    namen = ([f"backbone{i}" for i in range(1280)] + ["logo", "audio",
             "ocr_hinweis", "ocr_werbung", "ocr_da"]
             + [f"siglip{i}" for i in range(64)] + ["siglip_da"])
    return namen


# ---------------------------------------------------------------- Modell
HYPER_K = {"hidden": 96, "epochen": 12, "lr": 1e-3, "batch": 8192,
           "dropout": 0.0, "weight_decay": 0.0, "ensemble": 3}


def trainiere(Xtr, ytr, w, seed, hyper, modell_fn):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr.astype(np.int64)).to(dev)
    if modell_fn:
        net = modell_fn(Xtr.shape[1], hyper).to(dev)
    else:
        schichten = [nn.Linear(Xtr.shape[1], hyper["hidden"]), nn.ReLU()]
        if hyper["dropout"]:
            schichten.append(nn.Dropout(hyper["dropout"]))
        schichten.append(nn.Linear(hyper["hidden"], 2))
        net = nn.Sequential(*schichten).to(dev)
    zaehl = torch.bincount(yt, minlength=2).float()
    kw = (len(yt) / (2 * zaehl.clamp(min=1))).to(dev)
    wt = torch.from_numpy(w.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])
    lf = nn.CrossEntropyLoss(weight=kw, reduction="none")
    idx = np.arange(len(yt))
    for _ in range(hyper["epochen"]):
        net.train()
        np.random.shuffle(idx)
        for s in range(0, len(idx), hyper["batch"]):
            j = torch.from_numpy(idx[s:s + hyper["batch"]]).to(dev)
            opt.zero_grad()
            l = (lf(net(Xt[j]), yt[j]) * wt[j]).sum() / wt[j].sum().clamp(min=1e-6)
            l.backward()
            opt.step()
    net.eval()

    def vorhersage(X):
        aus = []
        with torch.no_grad():
            for s in range(0, len(X), 65536):
                p = torch.softmax(net(torch.from_numpy(X[s:s + 65536]).to(dev)), 1)
                aus.append(p[:, 1].cpu().numpy())
        return np.concatenate(aus)
    return vorhersage


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(p, y):
    """Logistische Kalibrierung z = a*logit(p) + b, Newton, 2 Parameter."""
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    a, b = 1.0, 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([((q - y) * x).sum(), (q - y).sum()])
        h = q * (1 - q)
        H = np.array([[(h * x * x).sum(), (h * x).sum()], [(h * x).sum(), h.sum()]]) + 1e-6 * np.eye(2)
        d = np.linalg.solve(H, g)
        a, b = a - d[0], b - d[1]
        if np.abs(d).max() < 1e-8:
            break
    return lambda p2: 1 / (1 + np.exp(-(a * np.log(np.clip(p2, 1e-6, 1 - 1e-6)
                                               / np.clip(1 - p2, 1e-6, 1)) + b)))


# ---------------------------------------------------------------- Bewertung
def block_iou(pred, gt):
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tot = 0.0
    for gs, ge in gt:
        best = 0.0
        for ps, pe in pred:
            inter = max(0.0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            if union > 0:
                best = max(best, inter / union)
        tot += best
    return tot / len(gt)


def replay(u, p_sek, tmp):
    dp = DUMPS / f"{u}.json"
    args = [str(BIN), "--replay-signals", str(dp)] + DEKODER
    if p_sek is not None:
        fc, fps = replay._fc[u], replay._fps[u]
        csv = tmp / f"{u}.csv"
        i = np.arange(fc)
        s = np.minimum((i / fps).astype(np.int64), len(p_sek) - 1)
        with open(csv, "w") as fh:
            fh.write("idx,time_s,nn_confidence\n")
            fh.writelines(f"{k},{k / fps:.3f},{v:.5f}\n" for k, v in zip(i, p_sek[s]))
        args += ["--replay-nn-csv", str(csv)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return [(float(a), float(b)) for a, b in (json.loads(r.stdout).get("blocks") or [])
                if float(b) > float(a)]
    except Exception:
        return None


def wahrheit(uuids):
    bs = _lade("bs_opt", SKRIPTE / "backbone-sonde.py")
    ub = bs.menschlabels()
    return {u: ub[u] for u in uuids if u in ub}


def maske(bl, n):
    m = np.zeros(n, bool)
    for a, b in bl:
        m[max(0, int(a)):min(n, int(b))] = True
    return m


def ursachen(pred, wahr, p_sek, rand=15):
    """Je Fehlersekunde genau eine Ursache, Regel wie fehlerbudget.ursachen:
    nahe einer wahren Kante (<= 15 s) = Kante; sonst je nach Richtung und
    danach, ob das NN (hier: kalibrierter Arm) selbst schon falsch lag."""
    n = len(p_sek)
    P, T = maske(pred, n), maske(wahr, n)
    kanten = np.array([x for b in wahr for x in b]) if wahr else np.zeros(0)
    falsch = np.flatnonzero(P != T)
    c = {"kante": 0, "nn_erfindet": 0, "dekoder_erfindet": 0, "nn_verpasst": 0, "dekoder_verpasst": 0}
    for t in falsch:
        if len(kanten) and np.min(np.abs(kanten - t)) <= rand:
            c["kante"] += 1
        elif P[t]:
            c["nn_erfindet" if p_sek[t] > 0.5 else "dekoder_erfindet"] += 1
        else:
            c["dekoder_verpasst" if p_sek[t] > 0.5 else "nn_verpasst"] += 1
    return c


def bewerte(p_eval, ev, truth, tmp, ursachen_aus=None):
    """Block-IoU je Aufnahme ueber den Produktions-Dekoder; mit ursachen_aus
    (dict) zusaetzlich die Fehlersekunden je Ursache, summiert."""
    jobs = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for i, u in enumerate(ev["uuids"]):
            if u not in truth:
                continue
            jobs[u] = ex.submit(replay, u, p_eval[ev["rec"] == i], tmp)
    aus = {}
    for u, f in jobs.items():
        bl = f.result()
        if bl is not None:
            aus[u] = block_iou(bl, truth[u])
            if ursachen_aus is not None:
                i = ev["uuids"].index(u)
                for k, v in ursachen(bl, truth[u], p_eval[ev["rec"] == i]).items():
                    ursachen_aus[k] = ursachen_aus.get(k, 0) + v
    return aus


# ---------------------------------------------------------------- Lauf
def arm_laden(name):
    if name == "kontrolle":
        return None
    p = HIER / "arme" / f"{name}.py"
    if not p.is_file():
        sys.exit(f"Arm {p} fehlt")
    return _lade(f"arm_{name}", p)


def code_abdruck(arm):
    h = hashlib.sha1()
    for p in [Path(__file__)] + ([Path(arm.__file__)] if arm else []):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="Name unter arme/ oder 'kontrolle'")
    ap.add_argument("--satz", default="entwicklung", choices=["entwicklung", "bestaetigung"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-basis", type=int, default=0)
    ap.add_argument("--json", help="Ausgabe (ohne --lauf Pflicht)")
    ap.add_argument("--lauf", help="registrierter Lauf NNN-name unter docs/optimierer/laeufe/; "
                    "setzt --arm und --json und prueft die Registrierung")
    ap.add_argument("--abbild", action="store_true",
                    help="zusaetzlich die Produktions-NN aus den Dumps bewerten (Abbild-Luecke)")
    a = ap.parse_args()
    if a.lauf:
        ld = REPO / "docs/optimierer/laeufe" / a.lauf
        v = json.loads((ld / "vorschlag.json").read_text())
        if a.arm != v["arm"]:
            sys.exit(f"--arm {a.arm} passt nicht zur Registrierung ({v['arm']})")
        a.json = str(ld / f"ergebnis-{a.satz}.json")
        if Path(a.json).exists():
            sys.exit(f"{a.json} existiert schon — ein registrierter Lauf misst jeden Satz genau einmal.")
        if a.satz == "bestaetigung":
            u = ld / "urteil.json"
            if not u.is_file() or json.loads(u.read_text()).get("status") != "bestaetigung_faellig":
                sys.exit("Bestaetigung nur nach erfuelltem Entwicklungs-Urteil (opt_urteil.py).")
    elif a.satz == "bestaetigung" and a.arm != "kontrolle":
        sys.exit("Bestaetigung fuer einen Versuchsarm nur ueber --lauf.")
    if not a.json:
        sys.exit("--json oder --lauf angeben")
    sperre = schutz()
    t0 = time.time()

    arm = arm_laden(a.arm)
    if a.lauf and code_abdruck(arm) != v["code"]:
        sys.exit(f"Code geaendert seit der Registrierung ({code_abdruck(arm)} != {v['code']}) — neu registrieren.")
    S = saetze()
    teile = {}
    for t, schritt, lab in (("training", SCHRITT, True), ("kalibrierung", 1, True),
                            ("eval", 1, False)):
        uu = S[a.satz] if t == "eval" else S[t]
        if t == "eval":
            uu = [u for u in uu if (DUMPS / f"{u}.json").is_file()]
        X, y, rec, geladen, metas = lade(uu, schritt, "training" if t == "training" else lab)
        teile[t] = {"X": X, "y": y, "rec": rec, "uuids": geladen, "meta": metas,
                    "schritt": schritt,
                    # absolute Sekunde je Zeile (Zeile j einer Aufnahme = Sekunde j*schritt)
                    "sek": np.concatenate([np.arange(int((rec == i).sum())) * schritt
                                           for i in range(len(geladen))]).astype(np.int64)}
        print(f"{t}: {len(geladen)} Aufnahmen, {len(X)} Zeilen", flush=True)
    namen = baue_k(teile)
    ev = teile["eval"]
    # Eval-Labels gibt es hier nicht: der Arm bekommt sie nie zu sehen.
    truth = wahrheit(ev["uuids"])
    replay._fps, replay._fc = {}, {}
    for u in ev["uuids"]:
        d = json.loads((DUMPS / f"{u}.json").read_text())
        replay._fps[u] = float(d.get("fps") or 25)
        replay._fc[u] = int(d.get("frame_count") or len(d["nn_confs"]))

    def ktx(t):
        d = teile[t]
        k = {"teil": t, "uuids": d["uuids"], "rec": d["rec"], "meta": d["meta"],
             "schritt": d["schritt"], "spalten_namen": namen, "sek": d["sek"]}
        if t == "training":
            k["y"] = d["y"]
        return k

    tmp = Path(tempfile.mkdtemp(prefix="optimierer-"))
    ergebnis = {"arm": a.arm, "satz": a.satz, "seeds": list(range(a.seed_basis, a.seed_basis + a.seeds)),
                "code": code_abdruck(arm), "n_eval": len(truth),
                "beschreibung": getattr(arm, "BESCHREIBUNG", "Kontrolle K (MLP7-Abbild)"),
                "laeufe": {}}

    if a.abbild:
        prod = bewerte_prod(ev, truth, tmp)
        ergebnis["abbild_produktion"] = {"iou_mittel": float(np.mean(list(prod.values()))),
                                         "je_aufnahme": prod}

    for name, modul in (("kontrolle", None), (a.arm, arm)):
        hyper = dict(HYPER_K)
        if modul is not None:
            hyper.update(getattr(modul, "HYPER", {}))
        Xs = {}
        for t in teile:
            X = teile[t]["X"]
            if modul is not None and hasattr(modul, "spalten"):
                X = np.asarray(modul.spalten(t if t != "eval" else "eval", X, ktx(t)), np.float32)
            Xs[t] = X
        mu = Xs["training"].mean(0, keepdims=True)
        sd = Xs["training"].std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        for t in Xs:
            Xs[t] = (Xs[t] - mu) / sd
        ytr = teile["training"]["y"]
        w = np.ones(len(ytr), np.float32)
        if modul is not None and hasattr(modul, "ziel"):
            ytr = np.asarray(modul.ziel(ktx("training")), np.int64)
        if modul is not None and hasattr(modul, "gewichte"):
            w = np.asarray(modul.gewichte(ktx("training")), np.float32)
            assert len(w) == len(ytr) and (w >= 0).all()
        runs = []
        for seed in ergebnis["seeds"]:
            # Wie die Produktion: Logit-Mittel aus `ensemble` Koepfen je Wiederholung.
            lk, le = 0.0, 0.0
            for j in range(hyper["ensemble"]):
                vh = trainiere(Xs["training"], ytr, w, seed * 100 + j, hyper,
                               getattr(modul, "modell", None) if modul is not None else None)
                lk = lk + logit(vh(Xs["kalibrierung"]))
                le = le + logit(vh(Xs["eval"]))
            pk = 1 / (1 + np.exp(-lk / hyper["ensemble"]))
            pe = 1 / (1 + np.exp(-le / hyper["ensemble"]))
            if modul is not None and hasattr(modul, "nachbearbeiten"):
                pk = modul.nachbearbeiten(pk, teile["kalibrierung"]["rec"], ktx("kalibrierung"))
                pe = modul.nachbearbeiten(pe, ev["rec"], ktx("eval"))
            kal = platt(pk, teile["kalibrierung"]["y"].astype(np.float64))
            pe = kal(pe)
            urs = {}
            iou = bewerte(pe, ev, truth, tmp, urs)
            runs.append({"seed": seed, "iou_mittel": float(np.mean(list(iou.values()))),
                         "iou_median": float(np.median(list(iou.values()))), "je_aufnahme": iou,
                         "ursachen": urs})
            print(f"  {name:<12} Seed {seed}  IoU mittel {runs[-1]['iou_mittel']:.4f}  "
                  f"median {runs[-1]['iou_median']:.4f}  ({len(iou)} Aufnahmen)", flush=True)
        ergebnis["laeufe"][name] = runs
        if a.arm == "kontrolle":
            break

    ergebnis["sekunden"] = round(time.time() - t0)
    ergebnis["speicher_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 1)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(ergebnis, indent=1))
    print(f"fertig in {ergebnis['sekunden']} s, Gipfel {ergebnis['speicher_gb']} GB -> {a.json}")
    sperre.close()
    return 0


def bewerte_prod(ev, truth, tmp):
    aus = {}
    with cf.ThreadPoolExecutor(8) as ex:
        jobs = {u: ex.submit(replay, u, None, tmp) for u in ev["uuids"] if u in truth}
    for u, f in jobs.items():
        bl = f.result()
        if bl is not None:
            aus[u] = block_iou(bl, truth[u])
    return aus


if __name__ == "__main__":
    sys.exit(main())
