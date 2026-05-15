#!/bin/bash
# Capture N minutes of a live HLS stream from the gateway and run
# tv-detect-train-logo on the result. Validates the bbox is sensible
# (= small, in a corner) before installing the template Mac- + Pi-side.
#
# Usage:
#   tv-train-logo-from-live.sh [--hd] <slug> [<minutes>]
#
# --hd: train HD logo (= source must be 1280+ wide). Output goes to
#       <slug>.logo.hd.txt, larger bbox area threshold tolerated.
#       Without --hd: SD path (= legacy default), <slug>.logo.txt.
# Defaults to 10 min capture. Channels without an active live stream
# fail at the curl/ffmpeg stage — start the channel in the player
# first to warm the buffer.

HD=0
if [ "$1" = "--hd" ]; then HD=1; shift; fi
SLUG="${1:?slug required (= channel slug as on /epg)}"
MINUTES="${2:-10}"
GATEWAY="${GATEWAY:-https://raspberrypi5lan:8443}"
LOGO_DIR="$HOME/.cache/tv-detect-daemon/logos"
PI_LOGO_DIR="/mnt/tv/hls/.tvd-logos"
TMP_TS="/tmp/tv-logo-train-${SLUG}.ts"

if [ "$HD" = "1" ]; then
  SUFFIX=".hd"
  MIN_AREA=800
  MAX_AREA=30000
else
  SUFFIX=""
  MIN_AREA=100
  MAX_AREA=8000
fi

mkdir -p "$LOGO_DIR"

if [ -f "$LOGO_DIR/$SLUG.logo${SUFFIX}.txt" ]; then
    echo "logo already present: $LOGO_DIR/$SLUG.logo${SUFFIX}.txt"
    echo "  delete it first if you want to re-train"
    exit 1
fi

echo "=== warmup HLS subscription for $SLUG (= tvh subscribes the mux on demand; cold ffmpeg connect times out before the tuner locks) ==="
curl -fsSk --max-time 15 "$GATEWAY/hls/$SLUG/dvr.m3u8" -o /dev/null
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "  warmup curl failed (rc=$rc) — channel not available; aborting"
    exit 2
fi
echo "  warmed."

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

# Verify resolution matches expected mode (= prevent SD-train from running
# when --hd was passed but stream is actually SD, or vice versa).
WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width \
        -of default=nokey=1:noprint_wrappers=1 "$TMP_TS" | head -1)
echo "  source width: ${WIDTH}px"
if [ "$HD" = "1" ] && [ "${WIDTH:-0}" -lt 1280 ]; then
    echo "  ABORT: --hd requested but source width ${WIDTH} < 1280 (= SD stream)."
    echo "  Either drop --hd or fix HD-routing for $SLUG."
    exit 5
fi
if [ "$HD" = "0" ] && [ "${WIDTH:-0}" -ge 1280 ]; then
    echo "  WARN: SD-mode but source width ${WIDTH} >= 1280 (= HD stream)."
    echo "  bbox will be HD-coords in <slug>.logo.txt — broken on future"
    echo "  SD-routed recordings. Re-run with --hd."
    exit 5
fi

OUT="/tmp/$SLUG.logo${SUFFIX}.txt"
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
if [ "$AREA" -gt "$MAX_AREA" ] || [ "$AREA" -lt "$MIN_AREA" ]; then
    echo "  REJECTED — bbox area ${AREA} outside sane range "
    echo "  ${MIN_AREA}-${MAX_AREA} px² (HD=$HD)."
    echo "  template kept at $OUT for inspection. Try other --start /"
    echo "  --persistence values + manual install."
    exit 4
fi

echo ""
echo "=== install Mac + Pi ==="
cp "$OUT" "$LOGO_DIR/$SLUG.logo${SUFFIX}.txt"
scp "$OUT" "raspberrypi5lan:$PI_LOGO_DIR/$SLUG.logo${SUFFIX}.txt"
echo ""
echo "DONE. Future train-head runs will include logo features for"
echo "$SLUG → no more 1281-dim mixed-shape padding."
echo ""
echo "Cleanup: rm $TMP_TS"
