#!/usr/bin/env python3
"""Extract per-channel logo-classification training data from
user-reviewed recordings.

For each recording with valid ads_user.json:
  - "show" frames: sampled from segments OUTSIDE user-marked ad-blocks
  - "ad" frames:   sampled from segments INSIDE  user-marked ad-blocks

Output: ~/.cache/logo-cnn-data/<slug>/{show,ad}/<uuid>-<sec>.png

Crop region per channel: read .logo.txt bbox + LOGO_MARGIN_PX padding
on each side, clamped to frame bounds. Output PNG keeps the cropped
size (variable per channel; not resized — training pipeline handles
resize to fixed input).

Source: T7 cache at ~/.cache/tv-detect-daemon/source/<uuid>.ts.
Recordings without local source are skipped (= would need re-fetch
from Pi via gateway, expensive at scale).

Run on Mac. Reads tvh DVR grid via gateway for channel mapping.
"""
import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

GATEWAY = os.environ.get("GATEWAY", "http://raspberrypi5lan:8080")
TVH_BASE = "http://raspberrypi5lan:9981"
SOURCE_CACHE = Path.home() / ".cache" / "tv-detect-daemon" / "source"
LOGO_DIR = Path.home() / ".cache" / "tv-detect-daemon" / "logos"
OUT_ROOT = Path.home() / ".cache" / "logo-cnn-data"

LOGO_MARGIN_PX = 50      # pad bbox by this much on each side
MAX_FRAMES_PER_REC = 30  # cap per (recording, class) so one rec doesn't dominate
MIN_BLOCK_DUR = 30       # skip ad-blocks shorter than this (boundary noise)
MIN_SHOW_DUR = 60        # skip show-segments shorter than this
EDGE_MARGIN = 5          # skip first/last N seconds of each segment (transitions)


def parse_logo_bbox(path):
    """Parse comskip-format .logo.txt and return (minX, minY, maxX, maxY).
    File contains binary mask data after the header so read as bytes +
    decode header lines individually with errors='ignore'."""
    bbox = {}
    raw = path.read_bytes()
    for line in raw.splitlines():
        try:
            line = line.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if line.startswith("logoMinX="):
            bbox["minx"] = int(line.split("=", 1)[1])
        elif line.startswith("logoMaxX="):
            bbox["maxx"] = int(line.split("=", 1)[1])
        elif line.startswith("logoMinY="):
            bbox["miny"] = int(line.split("=", 1)[1])
        elif line.startswith("logoMaxY="):
            bbox["maxy"] = int(line.split("=", 1)[1])
    if len(bbox) != 4:
        return None
    return bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]


def crop_region(bbox, frame_w=720, frame_h=576):
    """Pad bbox by margin, clamp to frame bounds. Return (x, y, w, h)."""
    x = max(0, bbox[0] - LOGO_MARGIN_PX)
    y = max(0, bbox[1] - LOGO_MARGIN_PX)
    x2 = min(frame_w, bbox[2] + LOGO_MARGIN_PX)
    y2 = min(frame_h, bbox[3] + LOGO_MARGIN_PX)
    return x, y, x2 - x, y2 - y


def extract_frames(src, timestamps, crop, out_paths):
    """Extract one frame per timestamp + crop. timestamps + out_paths same length."""
    x, y, w, h = crop
    n_ok = 0
    for t, out in zip(timestamps, out_paths):
        if out.exists():
            n_ok += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # -ss before -i = fast keyframe seek (slightly imprecise but fine
        # since we want diverse samples not exact-frame).
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{t:.2f}", "-i", str(src),
               "-frames:v", "1",
               "-vf", f"crop={w}:{h}:{x}:{y}",
               str(out)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            n_ok += 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return n_ok


def sample_timestamps(start, end, n_max):
    """Evenly-space n_max timestamps in [start+EDGE_MARGIN, end-EDGE_MARGIN]."""
    s = start + EDGE_MARGIN
    e = end - EDGE_MARGIN
    if e - s < EDGE_MARGIN:
        return []
    if n_max == 1:
        return [(s + e) / 2]
    step = (e - s) / (n_max - 1)
    return [s + i * step for i in range(n_max)]


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # uuid → channel slug
    chans = json.loads(urllib.request.urlopen(
        f"{GATEWAY}/api/channels", timeout=10).read())
    name_to_slug = {c["name"]: c["slug"] for c in chans.get("channels", [])
                    if c.get("name") and c.get("slug")}
    dvr = json.loads(urllib.request.urlopen(
        f"{TVH_BASE}/api/dvr/entry/grid?limit=2000", timeout=15).read())
    uuid_slug = {}
    for e in dvr.get("entries", []):
        u = e.get("uuid")
        cn = e.get("channelname", "")
        if u and cn in name_to_slug:
            uuid_slug[u] = name_to_slug[cn]

    # Per-channel bbox lookup (skip channels without a logo template).
    slug_bbox = {}
    for slug in set(uuid_slug.values()):
        p = LOGO_DIR / f"{slug}.logo.txt"
        if not p.is_file():
            continue
        bbox = parse_logo_bbox(p)
        if bbox:
            slug_bbox[slug] = bbox
    print(f"channels with logo: {len(slug_bbox)}", flush=True)

    # Per-channel counts
    stats = defaultdict(lambda: {"recs": 0, "show": 0, "ad": 0,
                                   "skipped_no_src": 0, "skipped_no_user": 0})

    # Walk all uuids that have local T7 source
    cached_uuids = {p.stem for p in SOURCE_CACHE.glob("*.ts")
                    if p.stat().st_size > 100 * 1024 * 1024}  # >=100 MB
    print(f"T7-cached recordings (>=100 MB): {len(cached_uuids)}", flush=True)

    for uuid, slug in sorted(uuid_slug.items()):
        if slug not in slug_bbox:
            continue
        st = stats[slug]
        if uuid not in cached_uuids:
            st["skipped_no_src"] += 1
            continue
        # Pi-side ads_user.json fetch via gateway (= source of truth).
        try:
            with urllib.request.urlopen(
                    f"{GATEWAY}/recording/{uuid}/ads", timeout=8) as r:
                d = json.loads(r.read())
            user_blocks = []
            for b in d.get("user", []):
                try:
                    s, e = float(b[0]), float(b[1])
                    if e - s >= MIN_BLOCK_DUR:
                        user_blocks.append((s, e))
                except (TypeError, ValueError, IndexError):
                    pass
            duration_s = d.get("duration_s") or 0
        except Exception:
            st["skipped_no_user"] += 1
            continue
        if not user_blocks or duration_s <= 0:
            st["skipped_no_user"] += 1
            continue

        src = SOURCE_CACHE / f"{uuid}.ts"
        crop = crop_region(slug_bbox[slug])
        out_show = OUT_ROOT / slug / "show"
        out_ad   = OUT_ROOT / slug / "ad"

        # Show segments = inverse of user-blocks within [0, duration]
        show_segs = []
        cur = 0.0
        for s, e in sorted(user_blocks):
            if s - cur >= MIN_SHOW_DUR:
                show_segs.append((cur, s))
            cur = e
        if duration_s - cur >= MIN_SHOW_DUR:
            show_segs.append((cur, duration_s))

        # Per-rec sampling caps: split MAX_FRAMES_PER_REC across segments
        # proportionally to length, up to the cap. Spacing is class-specific:
        # ad-blocks are typically short (82-290s on CC South Park) so 30s
        # spacing yielded 2-9 ad frames vs ~30 show frames per recording,
        # producing 4:1 imbalance. 10s spacing for ad balances the dataset
        # without requiring more recordings — BCE pos_weight alone didn't
        # rescue calibration (CC show-conf stuck at 0.4-0.7).
        def _sample_class(segs, kind, out_dir):
            if not segs:
                return 0
            spacing = 10 if kind == "ad" else 30
            total_len = sum(e - s for s, e in segs)
            n_total = min(MAX_FRAMES_PER_REC, int(total_len // spacing))
            if n_total <= 0:
                return 0
            timestamps = []
            out_paths = []
            for s, e in segs:
                share = (e - s) / total_len
                n = max(1, int(round(n_total * share)))
                ts = sample_timestamps(s, e, n)
                timestamps.extend(ts)
                out_paths.extend(
                    out_dir / f"{uuid[:8]}-{int(t)}.png" for t in ts)
            return extract_frames(src, timestamps, crop, out_paths)

        n_show = _sample_class(show_segs, "show", out_show)
        n_ad   = _sample_class(user_blocks, "ad",   out_ad)
        st["recs"] += 1
        st["show"] += n_show
        st["ad"]   += n_ad
        if (st["recs"] % 10) == 0 or n_show + n_ad > 0:
            print(f"  {slug:14s} rec#{st['recs']:3d}  +{n_show:2d} show "
                  f"+{n_ad:2d} ad  (uuid={uuid[:8]})", flush=True)

    print()
    print(f"{'channel':14s} {'recs':>5s} {'show':>6s} {'ad':>6s} "
          f"{'skip-src':>9s} {'skip-user':>10s}")
    print("-" * 60)
    for slug in sorted(stats):
        st = stats[slug]
        print(f"{slug:14s} {st['recs']:>5d} {st['show']:>6d} {st['ad']:>6d} "
              f"{st['skipped_no_src']:>9d} {st['skipped_no_user']:>10d}")


if __name__ == "__main__":
    main()
