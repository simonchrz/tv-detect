#!/bin/bash
# Nightly head-retrain. Picks up any new ads_user.json corrections
# the user made during the day, writes a fresh head.bin + the
# uncertain-frames file. Started by ~/Library/LaunchAgents/com.user.tv-train-head.plist.
#
# Safe to run by hand at any time; logs append to LOG.

LOG="$HOME/Library/Logs/tv-train-head.log"
MOUNT="$HOME/mnt/pi-tv"
VENV_PY="$HOME/ml/tv-classifier/.venv/bin/python"
SCRIPT="$HOME/src/tv-detect/scripts/train-head.py"
LOCAL_CACHE="$HOME/local-tv-cache"

exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') ==="

# launchd starts agents with an empty PATH — train-head.py spawns
# ffprobe/ffmpeg subprocesses which would FileNotFoundError without
# this. Includes Homebrew (Apple Silicon default) and ~/.local/bin
# (where tv-detect lives if invoked via the Python path).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

# SMB mount: tv-comskip.sh handles this every minute, so it's almost
# always already mounted. Sanity-check anyway — the train-head.py
# script reads recordings from $MOUNT/hls/_rec_*.
if ! mount | grep -q "on $MOUNT ("; then
  echo "SMB not mounted at $MOUNT — skipping (tv-comskip.sh will mount soon)"
  exit 0
fi

if [ ! -x "$VENV_PY" ]; then
  echo "venv python missing at $VENV_PY"
  exit 1
fi

# Auto re-detect after binary upgrade: when the tv-detect binary's
# mtime changes (= new build deployed), all existing .txt cutlists
# are stale relative to whatever the new binary would emit. Run a
# batch re-detect BEFORE training so the labels reflect the current
# pipeline. Skipped if mtime unchanged (= no new binary). The marker
# file lives in cache so it doesn't pollute the SMB share.
# DETECT_OFFLOAD=mac on the Pi gateway means re-detection happens via
# the HTTP daemon (~/bin/tv-thumbs-daemon.py). Pi's prewarm-loop
# notices missing/stale cutlist files and writes .detect-requested
# markers; daemon picks them up. So nightly head changes propagate
# automatically as soon as the next prewarm cycle runs.
#
# Old tv-detect-rebatch.sh path (commented out) iterated all
# recordings in one Mac-shell pass — disabled now that the daemon
# handles incremental re-detection. Keep the script around for
# manual one-off use if needed.
#TVD_BIN="$HOME/.local/bin/tv-detect"
#TVD_MARKER="$HOME/.cache/tvd-features/.last-binary-mtime"
#mkdir -p "$(dirname "$TVD_MARKER")"
#cur_mt=$(stat -f %m "$TVD_BIN" 2>/dev/null || echo 0)
#last_mt=$(cat "$TVD_MARKER" 2>/dev/null || echo 0)
#if [ "$cur_mt" != "$last_mt" ] && [ -x "$HOME/bin/tv-detect-rebatch.sh" ]; then
#  echo "tv-detect mtime changed ($last_mt → $cur_mt) — running batch re-detect"
#  "$HOME/bin/tv-detect-rebatch.sh" || echo "rebatch failed; continuing with stale labels"
#  echo "$cur_mt" > "$TVD_MARKER"
#fi

# --source-root prefers local SSD copy of recordings (3-4× faster
# feature extraction than SMB streaming). Recordings only on SMB
# are still picked up via the script's fallback to $MOUNT/hls/...
"$VENV_PY" "$SCRIPT" \
    --workers 4 \
    --source-root "$LOCAL_CACHE" \
    --surface-uncertain 6 \
    --with-logo \
    --with-minute-prior \
    --with-self-training \
    --write-pseudo-labels
rc=$?

# Cache hygiene: every recording experiment (different --with-* flags)
# leaves orphan .npy files in tvd-features/ — bumped cache key, old
# files unused but still on disk. Each is ~1-3 MB; over months of
# experiments the directory grows several GB. Prune anything that
# hasn't been read in 60 days. The current production flag set
# (--with-logo) is touched every nightly run so its caches stay
# alive; only stale experiment caches get reaped.
deleted=$(find "$HOME/.cache/tvd-features" -name "*.npy" -atime +60 -print -delete 2>/dev/null | wc -l)
if [ "$deleted" -gt 0 ]; then
  echo "cache cleanup: pruned $deleted stale .npy file(s)"
fi
# Per-show IoU snapshot. The new head.bin invalidates all auto-cutlists
# so the post-train daemon bulk-redetect must drain BEFORE the snapshot
# is meaningful (otherwise most ads.json files are stale → IoU=0). Wait
# up to 3 hours for queue-empty, then trigger. Run in background so
# train-head exits quickly; the snapshot trigger doesn't gate retraining.
if [ "$rc" -eq 0 ]; then
  ( ts="$(date '+%Y%m%dT%H%M%S')"
    deadline=$((SECONDS + 10800))
    while [ $SECONDS -lt $deadline ]; do
      pending=$(curl -s -m 5 "http://raspberrypi5lan:8080/api/internal/detect-pending" 2>/dev/null \
        | python3 -c "import json,sys; print(len(json.load(sys.stdin)['pending']))" 2>/dev/null)
      if [ "$pending" = "0" ]; then
        curl -s -m 30 -X POST "http://raspberrypi5lan:8080/api/internal/snapshot-per-show-iou?ts=$ts" \
          | head -c 200 >> "$LOG"; echo >> "$LOG"
        echo "snapshot taken (ts=$ts)" >> "$LOG"
        exit 0
      fi
      sleep 60
    done
    echo "snapshot timeout — queue never drained" >> "$LOG"
  ) &
fi

echo "train-head exit=$rc"
exit $rc
