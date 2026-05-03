#!/bin/bash
# Capture N minutes of a live HLS stream from the gateway and run
# tv-detect-train-logo on the result. Validates the bbox is sensible
# (= small, in a corner) before installing the template Mac- + Pi-side.
#
# Usage:
#   tv-train-logo-from-live.sh <slug> [<minutes>]
#
# Defaults to 10 min capture. Channels without an active live stream
# fail at the curl/ffmpeg stage — start the channel in the player
# first to warm the buffer.

SLUG="${1:?slug required (= channel slug as on /epg)}"
MINUTES="${2:-10}"
GATEWAY="${GATEWAY:-https://raspberrypi5lan:8443}"
LOGO_DIR="$HOME/.cache/tv-detect-daemon/logos"
PI_LOGO_DIR="/mnt/tv/hls/.tvd-logos"
TMP_TS="/tmp/tv-logo-train-${SLUG}.ts"

mkdir -p "$LOGO_DIR"

if [ -f "$LOGO_DIR/$SLUG.logo.txt" ]; then
    echo "logo already present: $LOGO_DIR/$SLUG.logo.txt"
    echo "  delete it first if you want to re-train"
    exit 1
fi

echo "=== capturing ${MINUTES} min of live $SLUG → $TMP_TS ==="
ffmpeg -loglevel error -y \
    -i "$GATEWAY/hls/$SLUG/dvr.m3u8" \
    -t "$((MINUTES*60))" \
    -c copy \
    "$TMP_TS"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "  ffmpeg capture failed (rc=$rc) — is the channel live + warm?"
    exit 2
fi
sz=$(du -m "$TMP_TS" | cut -f1)
echo "  captured: ${sz} MB"

OUT="/tmp/$SLUG.logo.txt"
echo ""
echo "=== training logo (start=120s, persist=0.88) ==="
tv-detect-train-logo \
    -minutes "$MINUTES" -start 120 -persistence 0.88 \
    -output "$OUT" -quiet \
    "$TMP_TS"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "  train failed (rc=$rc)"
    exit 3
fi

# Sanity-check bbox
read W H AREA < <(awk -F= '
    /^logoMinX=/{minx=$2}/^logoMaxX=/{maxx=$2}
    /^logoMinY=/{miny=$2}/^logoMaxY=/{maxy=$2}
    END{w=maxx-minx; h=maxy-miny; print w, h, w*h}' "$OUT")

echo "  bbox: ${W}x${H} = ${AREA} px²"
if [ "$AREA" -gt 8000 ] || [ "$AREA" -lt 100 ]; then
    echo "  REJECTED — bbox too big/small (sane: 200-5000 px²)."
    echo "  template kept at $OUT for inspection. Try other --start /"
    echo "  --persistence values + manual install."
    exit 4
fi

echo ""
echo "=== install Mac + Pi ==="
cp "$OUT" "$LOGO_DIR/$SLUG.logo.txt"
scp "$OUT" "raspberrypi5lan:$PI_LOGO_DIR/$SLUG.logo.txt"
echo ""
echo "DONE. Future train-head runs will include logo features for"
echo "$SLUG → no more 1281-dim mixed-shape padding."
echo ""
echo "Cleanup: rm $TMP_TS"
