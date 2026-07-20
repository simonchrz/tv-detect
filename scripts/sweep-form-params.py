#!/usr/bin/env python3
"""Per-channel Form() parameter sweep over the decode-signals cache.

Uses the ~/.cache/tvd-eval-signals cache (built by train-head.py's
realistic-eval pre-pass) + `tv-detect --replay-signals` to re-run ONLY
block formation with different parameters — no video decode, so a full
grid over every cached test recording costs minutes, not days. This is
the first time sweeping Form() params (scene-cut-snap, nn-gate,
nn-weight, min-absent-open, max-ad-gap) is affordable at all; the
motivating case is --scene-cut-snap, which defaults to 0 (off) because
it regressed "on some shows" in a global test — a per-channel verdict
was never measured.

For each cached recording:
  1. load its cached 1fps features (.npy), score with the DEPLOYED
     head (fetched from the gateway) → per-second ad-probability,
     written once as a replay CSV (upsampled by tv-detect to frame rate)
  2. fetch its production detect-config (per-show/channel overrides)
     as the BASELINE parameter set
  3. replay the baseline + every grid combo, compute block-IoU vs the
     recording's ground-truth blocks (train-archive frozen `ads`,
     fallback: gateway /recording/<uuid>/ads)

Output: per-channel table (baseline IoU vs best combo + which params
moved), plus a JSON dump for later analysis. SUGGESTIONS ONLY — nothing
is written to .channel-config.json automatically; a param that helps a
channel's 4 cached recs may still hurt shows not in the cache.

Usage:
    sweep-form-params.py                  # full grid, all cached recs
    sweep-form-params.py --channel rtl    # one channel only
    sweep-form-params.py --workers 8      # replay parallelism
"""

import argparse
import concurrent.futures as cf
import importlib.util
import itertools
import json
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TVD_BIN = HERE.parent / "build" / "tv-detect"
SIGNALS_CACHE = Path.home() / ".cache" / "tvd-eval-signals"
FEATURES_DIR = Path.home() / ".cache" / "tvd-features"
ARCHIVE_DIR = Path.home() / ".cache" / "tvd-train-archive"
SPEAKER_CSV_DIR = Path.home() / ".cache" / "tv-detect-daemon" / "speaker-csv"
GATEWAY = "https://raspberrypi5lan:8443"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# Import train-head.py as a module for its helpers (block_iou,
# load_deployed_mlp, _augment_teacher_feats, _load_whisper_per_sec,
# smooth_mean) — single source of truth for scoring semantics.
_spec = importlib.util.spec_from_file_location("th", HERE / "train-head.py")
th = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(th)


def http_json(url):
    with urllib.request.urlopen(url, timeout=15, context=CTX) as r:
        return json.loads(r.read())


def http_bytes(url):
    with urllib.request.urlopen(url, timeout=30, context=CTX) as r:
        return r.read()


def slug_of(uuid):
    # dvr-<slug>-<startts>
    parts = uuid.split("-")
    return "-".join(parts[1:-1]) if uuid.startswith("dvr-") else ""


def gt_blocks_for(uuid):
    """Frozen archive ads first (same GT eval_split scores against),
    gateway /ads fallback (user > merged)."""
    p = ARCHIVE_DIR / f"{uuid}.npz"
    if p.is_file():
        try:
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            ads = meta.get("ads") or []
            return [(float(a[0]), float(a[1])) for a in ads]
        except Exception:
            pass
    try:
        d = http_json(f"{GATEWAY}/recording/{uuid}/ads")
        ads = d.get("user") or d.get("ads") or []
        return [(float(a[0]), float(a[1])) for a in ads]
    except Exception:
        return None


def features_for(uuid):
    hits = sorted(FEATURES_DIR.glob(f"{uuid}-*-fps100-l2-a1.npy"))
    return hits[-1] if hits else None


def build_nn_csv(uuid, proba, fps, frame_count, out_path):
    with open(out_path, "w") as f:
        f.write("idx,time_s,nn_confidence\n")
        for i in range(frame_count):
            src_i = min(len(proba) - 1, int(i / fps))
            f.write(f"{i},{i / fps:.3f},{proba[src_i]:.4f}\n")


def replay(cache_path, nn_csv, params):
    """One blocks.Form() replay run. params mirrors the daemon's flag
    construction (tv-thumbs-daemon.py process_detect)."""
    cmd = [str(TVD_BIN), "--quiet", "--replay-signals", str(cache_path),
           "--replay-nn-csv", nn_csv, "--output", "summary",
           "--min-block-sec", str(params["min_block_s"]),
           "--nn-weight", str(params["nn_weight"]),
           "--logo-smooth", str(params["logo_smooth_s"]),
           "--bumper-snap", "90",
           "--bumper-threshold", str(params["bumper_threshold"]),
           "--nn-gate", str(params["nn_gate"]),
           "--scene-cut-snap", str(params["scene_cut_snap"]),
           "--min-absent-open", str(params["min_absent_open"]),
           "--max-ad-gap", str(params["max_ad_gap"]),
           "--nn-smooth", str(params.get("nn_smooth", 10))]
    if params.get("max_block_s"):
        cmd += ["--max-block-sec", str(params["max_block_s"])]
    if params.get("start_extend_s"):
        cmd += ["--start-extend", str(params["start_extend_s"])]
    if params.get("end_extend_s"):
        cmd += ["--end-extend", str(params["end_extend_s"])]
    if params.get("speaker_csv") and params.get("speaker_weight", 0) > 0:
        cmd += ["--speaker-csv", str(params["speaker_csv"]),
                "--speaker-weight", str(params["speaker_weight"])]
    cmd.append("dummy")
    r = subprocess.run(cmd, check=True, capture_output=True,
                       text=True, timeout=120)
    out = json.loads(r.stdout)
    return [(float(b[0]), float(b[1])) for b in out.get("blocks", [])]


def baseline_params(cfg):
    """detect-config response → the params production would use.
    Mirrors tv-thumbs-daemon.py + train-head.py _replay_blocks."""
    nn_weight = cfg.get("nn_weight", -1)
    nn_weight = nn_weight if nn_weight is not None and nn_weight >= 0 else 0.3
    nn_gate = cfg.get("nn_gate", -1)
    nn_gate = nn_gate if nn_gate is not None and nn_gate >= 0 else 0.3
    nn_smooth = cfg.get("nn_smooth", -1)
    nn_smooth = nn_smooth if nn_smooth is not None and nn_smooth >= 0 else 10
    p = {
        "nn_smooth": nn_smooth,
        "nn_weight": nn_weight,
        "nn_gate": nn_gate,
        "logo_smooth_s": cfg.get("logo_smooth_s") or 0,
        "bumper_threshold": cfg.get("bumper_threshold", 0.75),
        "scene_cut_snap": 0,      # production default (CLI default is 0 too)
        "min_absent_open": 5,     # tv-detect CLI default, daemon doesn't set it
        "max_ad_gap": 30,         # tv-detect CLI default, daemon doesn't set it
        "min_block_s": 60,
        "max_block_s": None,
        "start_extend_s": cfg.get("start_extend_s", 0),
        "end_extend_s": cfg.get("end_extend_s", 0),
    }
    if cfg.get("min_block_s") and cfg.get("max_block_s"):
        p["min_block_s"] = cfg["min_block_s"]
        p["max_block_s"] = cfg["max_block_s"]
    return p


# The sweep grid: per-dimension candidate values. Kept deliberately
# small — combinatorial product is evaluated per recording. Baseline
# values are merged in per-recording (so e.g. Let's Dance sweeps around
# ITS production override, not the global default).
GRID = {
    "scene_cut_snap": [0, 1.5],
    "nn_gate": [0.3, 0.15, 0.0],
    "nn_weight": [0.3, 0.5, 1.0],
    "min_absent_open": [5, 3],
    "max_ad_gap": [30, 60],
    # NNSmoothS was globally 10s since the smoothing landed, never
    # evaluated per channel: fast formats (short ad islands, kids'
    # channels) may lose boundary precision to 10s smearing, calm
    # formats may benefit from more.
    "nn_smooth": [5, 10, 20],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", help="restrict to one channel slug")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel replay processes")
    ap.add_argument("--out", default="/tmp/sweep-form-params.json")
    ap.add_argument("--smooth-s", type=float, default=0.0,
                    help="pre-smooth proba before replay (default 0: "
                         "Form applies its own --nn-smooth 10)")
    ap.add_argument("--sweep-speaker", action="store_true",
                    help="sweep ONLY --speaker-weight [0, 0.3, 0.5] "
                         "(everything else at production baseline), "
                         "restricted to recordings that have a "
                         "speaker CSV — answers 'does the live "
                         "SPEAKER_WEIGHT=0.3 help, and would another "
                         "weight be better?' per show/channel")
    ap.add_argument("--uuids", default="",
                    help="comma-separated uuid list — sweep only these "
                         "(e.g. all cached recordings of ONE SHOW, for "
                         "a per-show override decision like the Let's "
                         "Dance nn_gate case)")
    args = ap.parse_args()

    global GRID
    if args.sweep_speaker:
        GRID = {"speaker_weight": [0.0, 0.3, 0.5]}

    if not TVD_BIN.is_file():
        sys.exit(f"tv-detect binary missing at {TVD_BIN} — run make build")

    # Deployed champion head + channel map from the gateway.
    tmpdir = Path(tempfile.mkdtemp(prefix="sweep-form-"))
    head_path = tmpdir / "head.bin"
    head_path.write_bytes(http_bytes(
        f"{GATEWAY}/api/internal/detect-models/head.bin"))
    chan_map = http_json(
        f"{GATEWAY}/api/internal/detect-models/head.channel-map.json")
    chan_idx = {s: i for i, s in enumerate(chan_map["slugs"])}
    mlp = th.load_deployed_mlp(head_path)
    if mlp is None:
        sys.exit("deployed head.bin is not an MLP2/MLP3 head")
    print(f"champion head: input_dim={mlp.input_dim}, "
          f"{len(chan_idx)} channels")

    # Collect sweepable recordings.
    only_uuids = ({u.strip() for u in args.uuids.split(",") if u.strip()}
                  if args.uuids else None)
    recs = []
    for p in sorted(SIGNALS_CACHE.glob("*.json")):
        uuid = p.stem
        slug = slug_of(uuid)
        if only_uuids is not None and uuid not in only_uuids:
            continue
        if args.channel and slug != args.channel:
            continue
        fnpy = features_for(uuid)
        if fnpy is None:
            print(f"  skip {uuid}: no cached features")
            continue
        gt = gt_blocks_for(uuid)
        if gt is None:
            print(f"  skip {uuid}: no ground truth")
            continue
        if args.sweep_speaker and not (
                SPEAKER_CSV_DIR / f"{uuid}.speaker.csv").is_file():
            continue  # speaker sweep needs the CSV
        recs.append((uuid, slug, p, fnpy, gt))
    print(f"{len(recs)} sweepable recording(s)")
    if not recs:
        return

    # Per-recording: score once, write CSV once, then replay the grid.
    results = []  # (uuid, slug, params_key, iou)
    grid_keys = list(GRID.keys())

    def sweep_one(rec):
        uuid, slug, cache_path, fnpy, gt = rec
        try:
            with open(cache_path) as f:
                hdr = json.load(f)
            fps, frame_count = hdr["fps"], hdr["frame_count"]
            X = np.load(fnpy)
            Xa = th._augment_teacher_feats(
                X, slug, chan_idx, uuid,
                wants_whisper=True, wants_temporal=True)
            if Xa.shape[1] != mlp.input_dim:
                return (uuid, slug, None,
                        f"feature dim {Xa.shape[1]} != head {mlp.input_dim}")
            proba = mlp.predict_proba(Xa)[:, 1]
            if args.smooth_s > 0:
                proba = th.smooth_mean(proba, int(args.smooth_s / 2))
            cfg = None
            try:
                cfg = http_json(
                    f"{GATEWAY}/api/internal/detect-config/{uuid}")
            except Exception:
                cfg = {}
            base = baseline_params(cfg or {})
            # Speaker-fingerprint stream: production runs with
            # SPEAKER_WEIGHT=0.3 whenever a per-recording CSV was
            # computable (needs a show centroid from ≥2 edited
            # episodes) — mirror that in the baseline, and let the
            # grid sweep the weight where the CSV exists.
            spk_csv = SPEAKER_CSV_DIR / f"{uuid}.speaker.csv"
            if spk_csv.is_file():
                base["speaker_csv"] = str(spk_csv)
                base["speaker_weight"] = 0.3

            nn_csv = str(tmpdir / f"{uuid}.nn.csv")
            build_nn_csv(uuid, proba, fps, frame_count, nn_csv)

            out = []
            # Baseline first.
            blocks = replay(cache_path, nn_csv, base)
            out.append(("BASELINE", th.block_iou(blocks, gt)))
            # Grid: full product, each combo merged over the baseline.
            for combo in itertools.product(*(GRID[k] for k in grid_keys)):
                params = dict(base)
                params.update(dict(zip(grid_keys, combo)))
                key = ",".join(f"{k}={v}" for k, v in
                               zip(grid_keys, combo))
                blocks = replay(cache_path, nn_csv, params)
                out.append((key, th.block_iou(blocks, gt)))
            return (uuid, slug, out, None)
        except Exception as e:
            return (uuid, slug, None, str(e))

    n_combos = 1
    for k in grid_keys:
        n_combos *= len(GRID[k])
    print(f"grid: {n_combos} combos + baseline per recording "
          f"({len(recs)} recs × {n_combos + 1} replays)")

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(sweep_one, r) for r in recs]
        done = 0
        for fut in cf.as_completed(futs):
            uuid, slug, out, err = fut.result()
            done += 1
            if err:
                print(f"  [{done}/{len(recs)}] {uuid}: FAILED {err}")
                continue
            base_iou = out[0][1]
            best_key, best_iou = max(out[1:], key=lambda x: x[1])
            print(f"  [{done}/{len(recs)}] {uuid} ({slug}): "
                  f"baseline {base_iou:.3f} → best {best_iou:.3f} "
                  f"({best_key})")
            for key, iou in out:
                results.append(
                    {"uuid": uuid, "slug": slug, "params": key,
                     "iou": iou})

    # ── Per-channel aggregation ─────────────────────────────────────
    by_chan = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_chan[r["slug"]][r["params"]].append(r["iou"])

    print("\n=== per-channel summary (mean IoU over cached recs) ===")
    print(f"{'channel':16s} {'n':>3} {'baseline':>9} {'best':>7}  best-combo")
    suggestions = {}
    for slug in sorted(by_chan):
        combos = by_chan[slug]
        n = len(combos.get("BASELINE", []))
        base = (sum(combos["BASELINE"]) / n) if n else 0.0
        best_key, best_mean = "BASELINE", base
        for key, ious in combos.items():
            if key == "BASELINE" or len(ious) != n:
                continue
            m = sum(ious) / len(ious)
            if m > best_mean + 1e-9:
                best_key, best_mean = key, m
        marker = ""
        if best_key != "BASELINE" and best_mean - base >= 0.02:
            marker = "  ← suggest"
            suggestions[slug] = {
                "params": best_key, "baseline_iou": round(base, 3),
                "best_iou": round(best_mean, 3), "n_recs": n}
        print(f"{slug:16s} {n:>3} {base:>9.3f} {best_mean:>7.3f}  "
              f"{best_key}{marker}")

    Path(args.out).write_text(json.dumps(
        {"results": results, "suggestions": suggestions}, indent=2))
    print(f"\nfull results → {args.out}")
    if suggestions:
        print("suggested .channel-config.json changes (VERIFY on more "
              "recordings before applying — cached test recs may not "
              "represent all the channel's shows):")
        for slug, s in suggestions.items():
            print(f"  {slug}: {s['params']} "
                  f"({s['baseline_iou']} → {s['best_iou']}, "
                  f"n={s['n_recs']})")


if __name__ == "__main__":
    main()
