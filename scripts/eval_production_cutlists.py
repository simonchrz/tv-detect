#!/usr/bin/env python3
"""Confound-clean production-cutlist eval harness.

Measures how close the production auto-detection (ads.json) is to the user
ground-truth (ads_user.json), per show + per channel, as Block-IoU + frame-IoU.
Built after a 2026-06 session where the naive production-vs-label sweep was
unreliable - every "detection bug" turned out to be a measurement confound.
This harness handles all of them explicitly:

  - confirmed_show-only labels   (ads=[] but confirmed_show set ≠ "no ads")
  - overrun blocks               (synthetic skip past schedule end, not detection)
  - truncated VODs               (user ad ends past playlist → ads clamped away)
  - recDir/empty plumbing        (auto=[] while a real cutlist exists on disk)
  - trivial auto-confirm GT       (label == detection → IoU~1 tells you nothing)

Reports the HONEST detector quality on manually-reviewed recordings, keeping
auto-confirmed ones in a separate bucket. Pure stdlib; runs ON THE PI (reads
/mnt/tv/hls/_rec_*/{ads.json,ads_user.json,index.m3u8,*.txt} + the dvr grid).

Usage (on Pi):  python3 eval_production_cutlists.py [--channel <slug>] [--all-gt]
  --channel  restrict to one channel slug
  --all-gt   also include auto-confirmed GT in the headline (default: manual only)
"""
import glob, json, os, re, sys, urllib.request
from collections import defaultdict

HLS = "/mnt/tv/hls"
GRID = "http://127.0.0.1:9983/api/dvr/entry/grid?limit=3000"
HDR = re.compile(r"(\d+)\s+FRAMES AT\s+(\d+)")


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def playlist_dur(d):
    s = 0.0
    try:
        for ln in open(os.path.join(d, "index.m3u8")):
            if ln.startswith("#EXTINF"):
                try:
                    s += float(ln.split(":")[1].split(",")[0])
                except Exception:
                    pass
    except Exception:
        pass
    return s


def cutlist_dur(d):
    for t in glob.glob(os.path.join(d, "*.txt")):
        if os.path.basename(t).startswith("."):
            continue
        try:
            m = HDR.search(open(t, errors="ignore").read())
        except Exception:
            continue
        if m and int(m.group(2)) > 0:
            return int(m.group(1)) / (int(m.group(2)) / 100.0)
    return 0.0


def _txt_has_blocks(d):
    # True if the raw cutlist .txt has at least one "<startframe> <endframe>" line
    for t in glob.glob(os.path.join(d, "*.txt")):
        if os.path.basename(t).startswith("."):
            continue
        try:
            for ln in open(t, errors="ignore"):
                p = ln.split()
                if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
                    return True
        except Exception:
            pass
    return False


def blocks(x):
    out = []
    if isinstance(x, dict):
        x = x.get("ads", [])
    for b in x or []:
        try:
            s, e = float(b[0]), float(b[1])
            if e > s:
                out.append((s, e))
        except Exception:
            pass
    return out


def block_iou(pred, gt):
    # per-GT-block best-match IoU averaged - matches train-head.py's metric
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tot = 0.0
    for gs, ge in gt:
        best = 0.0
        for ps, pe in pred:
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            if union > 0:
                best = max(best, inter / union)
        tot += best
    return tot / len(gt)


def frame_iou(pred, gt, res=1.0):
    # second-resolution set IoU - penalises both FN and FP ad-time
    def ex(bl):
        s = set()
        for a, e in bl:
            for t in range(int(a // res), int(e // res) + 1):
                s.add(t)
        return s
    P, G = ex(pred), ex(gt)
    if not (P | G):
        return 1.0
    return len(P & G) / len(P | G)


def block_confusion(pred, gt):
    # matched/missed GT blocks, extra pred blocks (overlap-based)
    missed = sum(1 for gs, ge in gt if not any(ps < ge and pe > gs for ps, pe in pred))
    extra = sum(1 for ps, pe in pred if not any(gs < pe and ge > ps for gs, ge in gt))
    return len(gt) - missed, missed, extra


def main():
    only = None
    all_gt = "--all-gt" in sys.argv
    if "--channel" in sys.argv:
        only = sys.argv[sys.argv.index("--channel") + 1]

    grid = {}
    try:
        d = json.load(urllib.request.urlopen(GRID, timeout=10))
        for e in (d.get("entries", d) if isinstance(d, dict) else d):
            if e.get("uuid"):
                grid[e["uuid"]] = e
    except Exception as ex:
        print(f"grid fetch failed: {ex}", file=sys.stderr)

    excl = defaultdict(int)
    rows = []  # (slug, title, gt_src, n_auto, n_user, biou, fiou, matched, missed, extra)
    for dpath in sorted(glob.glob(f"{HLS}/_rec_*")):
        uuid = os.path.basename(dpath)[5:]
        ent = grid.get(uuid, {})
        slug = ent.get("uuid", uuid).split("dvr-")[-1]
        # derive slug from grid channelname → fallback uuid prefix
        slug = (ent.get("channel_slug") or
                re.sub(r"-\d+$", "", re.sub(r"^dvr-", "", uuid)))
        if only and slug != only:
            continue
        if ent and ent.get("sched_status") != "completed":
            excl["not-completed"] += 1
            continue
        au = jload(os.path.join(dpath, "ads_user.json"))
        if not isinstance(au, dict):
            excl["no-user-label"] += 1
            continue
        user = blocks(au.get("ads", []))
        confirmed_show = au.get("confirmed_show") or []
        is_auto = any(k in au for k in
                      ("auto_confirmed_at", "auto_confirm_score",
                       "auto_confirmed_via_fingerprint"))
        # --- confound exclusions ---
        if not user and confirmed_show:
            excl["confirmed_show_only"] += 1
            continue
        if not user and not confirmed_show:
            excl["unreviewed/empty"] += 1
            continue
        pdur = playlist_dur(dpath)
        cdur = cutlist_dur(dpath)
        last_user = max((e for _, e in user), default=0)
        if pdur > 0 and last_user > pdur + 60:
            excl["truncated_vod"] += 1
            continue
        auto = blocks(jload(os.path.join(dpath, "ads.json")) or [])
        # Stale-cache / plumbing confound: ads.json empty but the raw cutlist
        # .txt actually has detected block lines → the served auto is stale (or
        # the /ads recDir resolution returned empty), NOT a clean detector miss.
        if not auto and _txt_has_blocks(dpath):
            excl["stale_auto_cache"] += 1
            continue
        gt_src = "auto-confirm" if is_auto else "manual"
        biou = block_iou(auto, user)
        fiou = frame_iou(auto, user)
        mt, ms, ex2 = block_confusion(auto, user)
        rows.append((slug, ent.get("disp_title", uuid)[:30], gt_src,
                     len(auto), len(user), biou, fiou, mt, ms, ex2))

    # headline bucket
    bucket = [r for r in rows if all_gt or r[2] == "manual"]
    label = "manual+auto-confirm" if all_gt else "manual-review only"

    def agg(key_idx, name):
        groups = defaultdict(list)
        for r in bucket:
            groups[r[key_idx]].append(r)
        print(f"\n=== per-{name} (GT: {label}) ===")
        print(f"{name:22s} {'recs':>4} {'medBIoU':>7} {'medFIoU':>7} "
              f"{'matched':>7} {'missed':>6} {'extra':>5}")
        for k in sorted(groups, key=lambda k: _median([r[5] for r in groups[k]])):
            g = groups[k]
            mb = _median([r[5] for r in g])
            mf = _median([r[6] for r in g])
            print(f"{str(k)[:22]:22s} {len(g):>4} {mb:>7.2f} {mf:>7.2f} "
                  f"{sum(r[7] for r in g):>7} {sum(r[8] for r in g):>6} "
                  f"{sum(r[9] for r in g):>5}")

    print(f"\n{'='*64}\nProduction cutlist eval - {len(bucket)} recs in headline "
          f"({label})")
    print("excluded confounds: " + ", ".join(f"{k}={v}" for k, v in
                                              sorted(excl.items())))
    n_ac = sum(1 for r in rows if r[2] == "auto-confirm")
    print(f"auto-confirm recs (trivial GT, separate bucket): {n_ac}")
    agg(0, "channel")
    agg(1, "show")
    if bucket:
        all_b = sorted(r[5] for r in bucket)
        all_f = sorted(r[6] for r in bucket)
        print(f"\nOVERALL ({label}): median Block-IoU {_median(all_b):.2f}, "
              f"median frame-IoU {_median(all_f):.2f}, "
              f"missed-blocks {sum(r[8] for r in bucket)}, "
              f"extra-blocks {sum(r[9] for r in bucket)}")


def _median(v):
    v = sorted(v)
    n = len(v)
    if not n:
        return 0.0
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


if __name__ == "__main__":
    main()
