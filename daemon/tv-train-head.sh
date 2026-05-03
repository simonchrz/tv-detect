#!/bin/bash
# Nightly head-retrain. Picks up any new ads_user.json corrections
# the user made during the day, writes a fresh head.bin + the
# uncertain-frames file. Started by ~/Library/LaunchAgents/com.user.tv-train-head.plist.
#
# Safe to run by hand at any time; logs append to LOG.

LOG="$HOME/Library/Logs/tv-train-head.log"
VENV_PY="$HOME/ml/tv-classifier/.venv/bin/python"
SCRIPT="$HOME/src/tv-detect/scripts/train-head.py"
SNAPSHOT_FETCH="$HOME/bin/tv-train-snapshot-fetch.py"
SNAPSHOT_DIR="/tmp/tv-train-snapshot"
GATEWAY="http://raspberrypi5lan:8080"

exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') ==="

# launchd starts agents with an empty PATH — train-head.py spawns
# ffprobe/ffmpeg subprocesses which would FileNotFoundError without
# this. Includes Homebrew (Apple Silicon default) and ~/.local/bin
# (where tv-detect lives if invoked via the Python path).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

# Metadata snapshot: fetch all ads_user.json + cutlist + minute-prior
# from the gateway via HTTP into a local mirror at SNAPSHOT_DIR.
# train-head.py then runs with --hls-root pointing at the mirror,
# never touching SMB. Replaces the previous mount-pi-tv.sh dance.
if [ ! -x "$VENV_PY" ]; then
  echo "venv python missing at $VENV_PY"
  exit 1
fi
if [ ! -f "$SNAPSHOT_FETCH" ]; then
  echo "snapshot-fetch helper missing at $SNAPSHOT_FETCH"
  exit 1
fi
"$VENV_PY" "$SNAPSHOT_FETCH" --gateway-url "$GATEWAY" --out "$SNAPSHOT_DIR" || {
  echo "snapshot fetch failed — bailing"
  exit 1
}

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

# Source-cache hit-rate: train-head defaults --daemon-cache to
# ~/.cache/tv-detect-daemon/source/, populated by tv-thumbs-daemon
# (detect-driven + 30-min prefetch loop). With SOURCE_CACHE_MAX_GB=300
# the full Pi corpus fits locally — daemon-cache is the sole source
# for .ts files since the SMB-fallback was removed (2026-05-02).

# Training-active marker for the /learning page banner. Posted via
# HTTP (no SMB needed) so the gateway can show "Training läuft seit
# N min" — gateway writes the file on its own filesystem; trap fires
# the DELETE on exit (success OR crash) so a hung script doesn't
# leave the banner stuck on. The mtime tells the banner WHEN
# training started.
curl -fsS -X POST "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true
trap 'curl -fsS -X DELETE "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true' EXIT
TRAIN_START_TS=$(date +%s)

# Local output dir — head.bin + sidecars + archive/ all land here.
# Old SMB-mounted path was ~/mnt/pi-tv/hls/.tvd-models/ which broke
# when SMB unmounted; we now write locally and POST to Pi at end.
TRAIN_OUT="/tmp/tv-train-head-out"
mkdir -p "$TRAIN_OUT"
LOCAL_BACKBONE="$HOME/.cache/tv-detect-daemon/backbone.onnx"

# Prefetch head.history.json from Pi so train-head appends to the
# existing trail rather than writing a single-entry file. Same for
# head.test-set.json so the test-composition-changed reason fires
# correctly (compare against actual previous test-set, not "first
# run"). 404 just means none-yet-on-Pi → start fresh, fine.
for f in head.history.json head.test-set.json head.calibration.json; do
  curl -fsS -o "$TRAIN_OUT/$f" \
      "$GATEWAY/api/internal/detect-models/$f" || rm -f "$TRAIN_OUT/$f"
done

"$VENV_PY" "$SCRIPT" \
    --workers 4 \
    --backbone "$LOCAL_BACKBONE" \
    --logo-dir "$HOME/.cache/tv-detect-daemon/logos" \
    --output "$TRAIN_OUT/head.bin" \
    --hls-root "$SNAPSHOT_DIR" \
    --surface-uncertain 6 \
    --with-logo \
    --with-audio \
    --with-minute-prior \
    --with-self-training \
    --write-pseudo-labels
rc=$?

# Bundle head.bin + sidecars + archive/ into a tar.gz and POST to
# the gateway. Pi extracts atomically into /mnt/tv/hls/.tvd-models/
# (head.bin written LAST so daemons never see a half-deploy).
# Replaces the rsync over ssh — keeps everything on the unified
# Mac→HTTP→Pi path the rest of the stack uses.
if [ "$rc" -eq 0 ]; then
  echo "bundling + uploading head to gateway…"
  BUNDLE="/tmp/tv-train-head-bundle.tar.gz"
  ( cd "$TRAIN_OUT" && tar czf "$BUNDLE" \
      head.bin head.*.json head.*.txt archive 2>/dev/null )
  size_mb=$(du -m "$BUNDLE" | cut -f1)
  echo "  bundle: ${size_mb} MB"
  resp=$(curl -fsS -X POST --data-binary "@$BUNDLE" \
      -H "Content-Type: application/gzip" \
      "$GATEWAY/api/internal/head-bundle")
  echo "  upload response: $resp"
  if [ "$?" -ne 0 ]; then
      echo "  WARN: upload failed — head NOT deployed"
      rc=2
  fi
  rm -f "$BUNDLE"
fi

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
