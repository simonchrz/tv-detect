#!/usr/bin/env python3
"""Capture a bumper-template frame from an existing recording so the
daemon's bumper-snap boundary refinement can use it. Pi serves new
templates immediately via the existing /api/internal/detect-bumpers
endpoint — no daemon restart needed.

Usage:
  capture-bumper.py <uuid_or_prefix> <timestamp_s> [name]

Workflow:
  1. Open a recording in the player, scrub to a bumper frame at the
     end of an ad block (the channel station-id card).
  2. Note the displayed timestamp in seconds.
  3. Run this script with the recording uuid (8-char prefix is fine)
     and the timestamp.
  4. The PNG lands at /mnt/nvme/tv/hls/.tvd-bumpers/<slug>/<name>.png
     on the Pi. Multiple per channel is encouraged — different color
     variants of the same animation each act as an independent
     template, and the matcher takes the max IoU.

Default name is `<ts>-auto.png` if not given. Filenames just need to
end in .png; the slug subdir is auto-derived from the recording's
channel."""
import json
import ssl
import subprocess
import sys
import urllib.request

GATEWAY = "http://raspberrypi5lan:8080"
PI_HOST = "raspberrypi5lan"
PI_BUMPER_DIR = "/mnt/nvme/tv/hls/.tvd-bumpers"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def http_json(url):
    with urllib.request.urlopen(url, timeout=10, context=CTX) as r:
        return json.loads(r.read())


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    uuid_in = sys.argv[1]
    try:
        ts = float(sys.argv[2])
    except ValueError:
        print(f"timestamp must be a number, got {sys.argv[2]!r}", file=sys.stderr)
        sys.exit(2)
    name = sys.argv[3] if len(sys.argv) > 3 else f"{int(ts)}-auto.png"
    if not name.endswith(".png"):
        name += ".png"

    # Resolve uuid (allow prefix)
    all_uuids = http_json(f"{GATEWAY}/api/internal/recording-uuids").get("uuids", [])
    matches = [u for u in all_uuids if u.startswith(uuid_in)]
    if len(matches) == 0:
        print(f"no recording matches uuid={uuid_in!r}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"ambiguous uuid prefix — {len(matches)} matches:",
              file=sys.stderr)
        for m in matches[:5]:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)
    uuid = matches[0]

    cfg = http_json(f"{GATEWAY}/api/internal/detect-config/{uuid}")
    slug = cfg.get("channel_slug")
    if not slug:
        print(f"recording {uuid} has no channel slug", file=sys.stderr)
        sys.exit(1)
    title = cfg.get("show_title", "?")
    print(f"→ {uuid[:8]} {slug:<12} {title[:40]}  @ t={ts}s")

    pi_dest = f"{PI_BUMPER_DIR}/{slug}/{name}"
    # Resolve the .ts on the Pi via the same trick the daemon uses,
    # then ffmpeg-extract one frame at the requested timestamp.
    pi_cmd = f"""set -e
shopt -s nullglob
rec_dir=/mnt/nvme/tv/hls/_rec_{uuid}
txt=""
for t in "$rec_dir"/*.txt; do
  case "$t" in *.logo.txt|*.cskp.txt|*.tvd.txt|*.trained.logo.txt) continue ;; esac
  txt="$t"; break
done
[ -z "$txt" ] && {{ echo "no cutlist for {uuid[:8]}" >&2; exit 1; }}
base=$(basename "$txt" .txt)
src=""
for c in /mnt/nvme/tv/*/"$base.ts"; do [ -f "$c" ] && {{ src="$c"; break; }}; done
[ -z "$src" ] && {{ echo "no .ts source for {uuid[:8]}" >&2; exit 1; }}
mkdir -p "{PI_BUMPER_DIR}/{slug}"
/usr/bin/ffmpeg -y -hide_banner -loglevel error -ss {ts} -i "$src" \\
    -frames:v 1 -vf scale=720:576 "{pi_dest}"
stat -c '%s bytes' "{pi_dest}"
"""
    r = subprocess.run(["ssh", PI_HOST, pi_cmd],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    print(r.stdout.strip())
    print(f"✓ saved to {slug}/{name}")
    # Sanity-check the new template list
    blist = http_json(
        f"{GATEWAY}/api/internal/detect-bumpers/{slug}").get("templates", [])
    print(f"  {slug} now has {len(blist)} bumper template(s); daemon "
          f"picks them up on the next detect for any {slug} recording.")


if __name__ == "__main__":
    main()
