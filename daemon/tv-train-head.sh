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

# SMB mount: macOS sleep/wake or any brief network blip can leave the
# mount-point alive but pointing to a dead CIFS connection — so we
# don't trust "is the directory there", we actively re-mount when
# needed. mount-pi-tv.sh is idempotent (no-op if already up) and pulls
# the password from login-keychain, so this works under launchd's
# limited environment too.
if ! mount | grep -q "on $MOUNT ("; then
  echo "SMB not mounted at $MOUNT — running mount-pi-tv.sh"
  if [ -x "$HOME/bin/mount-pi-tv.sh" ]; then
    "$HOME/bin/mount-pi-tv.sh" || {
      echo "mount-pi-tv.sh failed — bailing"
      exit 1
    }
  else
    echo "mount-pi-tv.sh missing — bailing"
    exit 1
  fi
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
# is meaningful (stale ads.json → IoU=0). Earlier this script tried to
# background a wait-loop here, but launchd reaps subshells when the
# parent train-head.sh exits. Instead we drop a marker file with the
# desired snapshot timestamp; the long-running tv-thumbs-daemon picks
# it up on its next poll cycle when detect queue == 0, fires the
# snapshot endpoint, deletes the marker. Stateless + survives daemon
# restarts.
if [ "$rc" -eq 0 ]; then
  marker="$HOME/.cache/tv-detect-daemon/snapshot-requested"
  mkdir -p "$(dirname "$marker")"
  date '+%Y%m%dT%H%M%S' > "$marker"
  echo "snapshot marker written → $marker" >> "$LOG"
fi

echo "train-head exit=$rc"
exit $rc
