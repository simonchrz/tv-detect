#!/usr/bin/env python3
"""Rebuild a per-show speaker centroid from all available cached episodes.

Discovers episodes by querying the gateway for recordings of the same
show_title, filters to those that (a) have a pre-extracted embeddings
.npz cached locally and (b) have user-edited ad blocks (= trustworthy
truth for the show/ad split), then invokes build-show-centroid.py.

Designed to be called by the Mac daemon before each detect run — cheap
when nothing changed (re-builds in <1s once embeddings exist).

Usage:
    update-show-centroid.py <show-slug> <output-centroid.npz> \\
        --embeddings-dir <dir>         # where <uuid>.npz files live
        --gateway <url>                # http://raspberrypi5lan:8080
        [--show-title "Display Name"]  # for metadata only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import ssl
import urllib.request
from pathlib import Path


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# Das interne Gateway (Caddy) traegt ein selbstsigniertes Zertifikat. Ohne
# eigenen Kontext scheitert JEDER Aufruf sofort an CERTIFICATE_VERIFY_FAILED
# — und zwar an der allerersten Zeile von main(), also bevor irgendetwas
# passiert.
#
# ⚠️ Genau daran ist die Sprecher-Kennung ueber einen Monat lang still
# ausgefallen: der Aufrufer (tv-thumbs-daemon) uebergibt seit der
# Caddy-Umstellung `--gateway https://raspberrypi5lan:8443`, dieses Skript
# hatte aber nur den alten http-Vorgabewert im Kopf und keinen Kontext. Der
# Daemon meldete `centroid build failed (rc=1): {stderr[:300]}` — und in den
# ersten 300 Zeichen steht die FORTSCHRITTS-Zeile, nicht der Traceback. Der
# Fehler stand da, jede Nacht, 2166 Mal, und sah aus wie ein Slug-Problem.
# Dieselbe Kopf-statt-Ende-Truncation war fuer den Embedding-Zweig im Daemon
# schon einmal repariert worden, fuer diesen Zweig nicht.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def http_json(url: str):
    # context nur fuer https noetig; bei http ignoriert urlopen ihn ohnehin
    # nicht — deshalb nur dort mitgeben.
    if url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=15, context=_CTX) as r:
            return json.loads(r.read())
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("show_slug",
                    help="normalised show identifier (e.g. 'taff', "
                         "'gute-zeiten-schlechte-zeiten')")
    ap.add_argument("output")
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--ads-cache-dir",
                    help="optional dir of <uuid>.ads.json files written "
                         "alongside detect runs; if absent, fetched from gateway")
    ap.add_argument("--gateway", default="http://raspberrypi5lan:8080")
    ap.add_argument("--show-title", default="")
    ap.add_argument("--builder",
                    default=str(Path(__file__).parent / "build-show-centroid.py"))
    args = ap.parse_args()

    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        print(f"embeddings dir missing: {emb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"discovering episodes of show='{args.show_slug}'", file=sys.stderr)
    all_uuids = http_json(f"{args.gateway}/api/internal/recording-uuids")["uuids"]

    # Per-recording: fetch config + ads, filter by show_slug match + edited=true.
    episodes = []
    skipped_no_emb = 0
    skipped_not_edited = 0
    skipped_wrong_show = 0
    for u in all_uuids:
        try:
            cfg = http_json(f"{args.gateway}/api/internal/detect-config/{u}")
            if slugify(cfg.get("show_title", "")) != args.show_slug:
                skipped_wrong_show += 1
                continue
            ads = http_json(f"{args.gateway}/recording/{u}/ads")
            if not ads.get("edited"):
                skipped_not_edited += 1
                continue
            emb_path = emb_dir / f"{u}.npz"
            if not emb_path.exists():
                skipped_no_emb += 1
                continue
            # Persist the ads to a small file the builder can read.
            ads_path = emb_dir / f"{u}.ads.json"
            with open(ads_path, "w") as f:
                json.dump(ads, f)
            episodes.append((u, str(emb_path), str(ads_path)))
        except Exception as e:
            print(f"  err {u[:8]}: {e}", file=sys.stderr)

    print(f"  match show: {len(episodes)+skipped_no_emb+skipped_not_edited} of "
          f"{len(all_uuids)}", file=sys.stderr)
    print(f"  usable episodes: {len(episodes)} "
          f"(skipped: {skipped_not_edited} not-edited, "
          f"{skipped_no_emb} no-embeddings)", file=sys.stderr)

    if len(episodes) < 1:
        print(f"insufficient data for centroid", file=sys.stderr)
        sys.exit(2)

    # Invoke build-show-centroid.py with one --episode per usable recording.
    cmd = [sys.executable, args.builder, args.show_slug, args.output]
    if args.show_title:
        cmd += ["--show-title", args.show_title]
    for uuid, emb_path, ads_path in episodes:
        cmd += ["--episode", f"{uuid}:{emb_path}:{ads_path}"]
    print(f"\ninvoking builder with {len(episodes)} episode(s)...",
          file=sys.stderr)
    proc = subprocess.run(cmd, check=False)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
