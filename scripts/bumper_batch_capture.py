#!/usr/bin/env python3
"""Batch-capture bumper templates from user-reviewed recordings on a
target channel by walking ads_user.json boundaries and POSTing to
/api/recording/<uuid>/bumper-capture for each block edge.

Pattern (proven 2026-05-11 on kabel-eins/sat-1/prosieben → 0.04→0.826
IoU, 111 zero-block recordings recovered):
  - For each user-confirmed ad block [s, e]:
      end-bumper   = window [s-3, s]   (last 3s of show before ad)
      start-bumper = window [e,   e+3] (first 3s of show after ad)

Run on Pi (gateway-local; needs source .ts via /source endpoint).

Usage: python3 bumper_batch_capture.py <channel_slug> [--dry-run]
"""
import sys, urllib.request, urllib.error, json, os, time
from pathlib import Path

GATEWAY = os.environ.get("GATEWAY", "http://localhost:8080")
TVH_BASE = "http://localhost:9981"
HLS_DIR = Path("/mnt/tv/hls")


def http_post_json(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: bumper_batch_capture.py <channel_slug> [--dry-run]")
    slug = sys.argv[1].lower()
    dry = "--dry-run" in sys.argv

    # Map channel-slug → channel-name (= what tvh stores per entry).
    chans = json.loads(urllib.request.urlopen(
        f"{GATEWAY}/api/channels", timeout=10).read())
    name = next((c["name"] for c in chans.get("channels", [])
                 if c.get("slug") == slug), None)
    if not name:
        sys.exit(f"unknown channel slug: {slug}")
    print(f"target: {slug} (= '{name}')")

    # Find user-reviewed recordings on this channel with non-empty ads.
    dvr = json.loads(urllib.request.urlopen(
        f"{TVH_BASE}/api/dvr/entry/grid?limit=2000", timeout=10).read())
    candidates = []
    for e in dvr.get("entries", []):
        if e.get("channelname") != name: continue
        if e.get("sched_status") not in ("completed", "completedError"): continue
        uuid = e.get("uuid")
        if not uuid: continue
        rec = HLS_DIR / f"_rec_{uuid}"
        if not rec.is_dir(): continue
        user_p = rec / "ads_user.json"
        if not user_p.is_file(): continue
        try:
            d = json.loads(user_p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict): continue
        ads = d.get("ads") or []
        if not ads: continue
        # Only TRUE user blocks (= not auto-confirmed empty).
        candidates.append((uuid, e.get("disp_title", "?"), ads))

    print(f"candidates with user-confirmed ad blocks: {len(candidates)}")
    if not candidates:
        return

    n_captured_end = n_captured_start = n_skipped = n_failed = 0
    for uuid, title, ads in candidates:
        for blk in ads:
            try:
                s, e = float(blk[0]), float(blk[1])
            except (TypeError, ValueError, IndexError):
                continue
            if e - s < 5:
                n_skipped += 1
                continue
            for kind, win in (("end",   (max(0, s - 3), s)),
                              ("start", (e, e + 3))):
                if win[1] - win[0] < 1.5:
                    n_skipped += 1
                    continue
                if dry:
                    print(f"  DRY {uuid[:8]} {kind} {win[0]:.0f}-{win[1]:.0f}s")
                    continue
                try:
                    r = http_post_json(
                        f"{GATEWAY}/api/recording/{uuid}/bumper-capture",
                        {"start_s": win[0], "end_s": win[1], "kind": kind})
                    if r.get("ok"):
                        if kind == "end": n_captured_end += 1
                        else:             n_captured_start += 1
                    else:
                        n_failed += 1
                        print(f"  FAIL {uuid[:8]} {kind} {win[0]:.0f}-{win[1]:.0f}s: "
                              f"{r.get('error','?')[:100]}")
                except urllib.error.HTTPError as ex:
                    n_failed += 1
                    print(f"  HTTP {ex.code} {uuid[:8]} {kind}: {ex.reason}")
                except Exception as ex:
                    n_failed += 1
                    print(f"  ERR  {uuid[:8]} {kind}: {ex}")

    print(f"\nResult:")
    print(f"  end-bumper captures:    {n_captured_end}")
    print(f"  start-bumper captures:  {n_captured_start}")
    print(f"  skipped (too-short):    {n_skipped}")
    print(f"  failed:                 {n_failed}")
    print(f"\nNext: scripts/dedup_bumpers.py + cap_bumpers.py to "
          f"hash-prune the new templates.")


if __name__ == "__main__":
    main()
