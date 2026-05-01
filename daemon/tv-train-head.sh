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
# (test-set) + .detect-requested-low (rest) markers; daemon picks
# them up V2-style — high prio first, background backfill once idle.
# Nightly head changes propagate automatically.

# --source-root prefers local SSD copy of recordings (3-4× faster
# feature extraction than SMB streaming). Recordings only on SMB
# are still picked up via the script's fallback to $MOUNT/hls/...

# Training-active marker for the /learning page banner. Posted via
# HTTP (no SMB needed) so the gateway can show "Training läuft seit
# N min" — gateway writes the file on its own filesystem; trap fires
# the DELETE on exit (success OR crash) so a hung script doesn't
# leave the banner stuck on. The mtime tells the banner WHEN
# training started.
GATEWAY="http://raspberrypi5lan:8080"
curl -fsS -X POST "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true
trap 'curl -fsS -X DELETE "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true' EXIT
TRAIN_START_TS=$(date +%s)

"$VENV_PY" "$SCRIPT" \
    --workers 4 \
    --source-root "$LOCAL_CACHE" \
    --surface-uncertain 6 \
    --with-logo \
    --with-audio \
    --with-minute-prior \
    --with-self-training \
    --write-pseudo-labels
rc=$?

# Persist the wall-time of this run to the history file so the
# /learning banner can derive an ETA for future training runs as
# median(last-N-durations) - elapsed. JSON-lines format = append-
# only, gateway parses cheaply line-by-line. POSTed via HTTP =
# gateway appends server-side, no SMB write.
TRAIN_DUR=$(( $(date +%s) - TRAIN_START_TS ))
TRAIN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
curl -fsS -X POST -H "Content-Type: application/json" \
    -d "{\"ts\":\"$TRAIN_TS\",\"dur_s\":$TRAIN_DUR,\"rc\":$rc}" \
    "$GATEWAY/api/internal/training-duration" >/dev/null 2>&1 || true

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
