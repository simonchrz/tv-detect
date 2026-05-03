#!/usr/bin/env python3
"""Fetch training-snapshot from gateway, materialise as a local
directory mirror so tv-train-head.py can run with --hls-root pointing
at the local copy instead of the SMB mount.

Layout written:
  <out>/_rec_<uuid>/{ads_user.json, ads.json, pseudo_labels.json,
                      <base>.txt, index.m3u8}
  <out>/.minute_prior_by_channel.json

Files are only written when their source content is non-empty (so
ads.json absence is preserved as absence, not as an empty file).
index.m3u8 is materialised as an empty marker file — train-head.py
only checks .exists() on it.

Idempotent: snapshot is fetched fresh each run. Mirror is wiped
+ rebuilt to avoid stale entries from deleted recordings."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway-url",
                    default="http://raspberrypi5lan:8080",
                    help="Gateway base URL (= where /api/internal/"
                         "training-snapshot lives)")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/tv-train-snapshot"),
                    help="Output mirror directory; will be wiped + "
                         "rebuilt")
    args = ap.parse_args()

    url = f"{args.gateway_url.rstrip('/')}/api/internal/training-snapshot"
    print(f"fetching {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"fetch err: {e}", file=sys.stderr)
        sys.exit(1)

    recs = data.get("recordings", [])
    minute_prior = data.get("minute_prior_by_channel", {})
    print(f"got {len(recs)} recordings, "
          f"{len(minute_prior)} minute-prior channels", flush=True)

    # Wipe + recreate to drop stale per-rec dirs (e.g. recordings the
    # user deleted between runs).
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    n_user = n_auto = n_cutlist = n_pseudo = 0
    for r in recs:
        uuid = r["uuid"]
        d = args.out / f"_rec_{uuid}"
        d.mkdir()
        if r.get("ads_user"):
            (d / "ads_user.json").write_text(json.dumps(r["ads_user"]))
            n_user += 1
        if r.get("ads_auto"):
            (d / "ads.json").write_text(json.dumps(r["ads_auto"]))
            n_auto += 1
        if r.get("pseudo_labels"):
            (d / "pseudo_labels.json").write_text(
                json.dumps(r["pseudo_labels"]))
            n_pseudo += 1
        # train-head only needs the cutlist .txt to EXIST with the
        # right basename (= "Show $YYYY-MM-DD-HHMM.txt") — its content
        # is parsed as a fallback only when ads_user/ads.json is
        # absent. After a head deploy V2 truncates ads.txt to 0 B
        # until the daemon re-detects, so a content-only check would
        # silently drop the entire post-deploy corpus from training.
        # Write it either way; train-head reads ads from ads_user.
        if r.get("base"):
            (d / f"{r['base']}.txt").write_text(r.get("cutlist_text") or "")
            n_cutlist += 1
        # Cluster-anchored ad spots: silence-aligned spots from this
        # recording whose audio+visual fingerprint matches a known
        # cluster (≥3 family members across corpus). High-confidence
        # ad anchors that train-head treats as bonus-weight ad labels
        # (= 1.5× user-base) — useful both as a sanity-check on
        # reviewed recordings and as pseudo-label anchors on
        # unreviewed ones.
        if r.get("cluster_anchored"):
            (d / "cluster_anchored.json").write_text(
                json.dumps(r["cluster_anchored"]))
        if r.get("has_index_m3u8"):
            (d / "index.m3u8").write_text("")  # marker; only .exists()
                                                # is checked downstream

    if minute_prior:
        (args.out / ".minute_prior_by_channel.json").write_text(
            json.dumps(minute_prior))

    print(f"wrote {len(recs)} recs to {args.out}: "
          f"{n_user} user, {n_auto} auto, {n_cutlist} cutlist, "
          f"{n_pseudo} pseudo", flush=True)


if __name__ == "__main__":
    main()
