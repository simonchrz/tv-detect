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
# Gateway via Caddy on :8443 since the go-front bypass (self-signed local CA →
# curl -k). The "~40% Caddy CPU on TLS" rationale for the old :8080 path was
# disproven (Caddy 0.05% CPU on a TLS bulk pull; Pi5 ARM HW-AES = TLS ~free).
GATEWAY="https://raspberrypi5lan:8443"
CURL="curl -fsS -k"

exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') ==="

# launchd starts agents with an empty PATH — train-head.py spawns
# ffprobe/ffmpeg subprocesses which would FileNotFoundError without
# this. Includes Homebrew (Apple Silicon default) and ~/.local/bin
# (where tv-detect lives if invoked via the Python path).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

# launchd jobs can't reach the login keychain, so `gh` (used by the
# off-site model-anchor at deploy time) can't read its keyring-stored
# token → auto-anchor silently failed every night since 2026-06-02.
# Export GH_TOKEN from secrets.env (chmod 600); gh prefers the env var
# over the keychain. Fail-soft: if the file/key is absent the anchor
# stays a no-op, exactly as before.
SECRETS_ENV="$HOME/.config/tv-stack/secrets.env"
if [ -f "$SECRETS_ENV" ]; then
  GH_TOKEN="$(grep -E '^GH_TOKEN=' "$SECRETS_ENV" | head -1 | cut -d= -f2-)"
  [ -n "$GH_TOKEN" ] && export GH_TOKEN
fi

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

# Conditional skip: only retrain when there are >=NEW_REVIEWS_THR
# fresh user reviews since the last deploy. Otherwise we'd burn 12+h
# of detect-drain post-deploy for no IoU gain. Threshold tunable via
# TVH_TRAIN_NEW_REVIEWS_MIN env (default 10; set 1 to always run).
NEW_REVIEWS_THR="${TVH_TRAIN_NEW_REVIEWS_MIN:-10}"
LAST_DEPLOY_TS=$(grep -oE '"ts": "[0-9T]+"' "$SNAPSHOT_DIR/.tvd-models/head.history.json" 2>/dev/null \
                  | tail -1 | grep -oE '[0-9T]+' \
                  | python3 -c "import sys, datetime
ts = sys.stdin.read().strip()
if ts:
    print(int(datetime.datetime.strptime(ts, '%Y%m%dT%H%M%S').timestamp()))
" 2>/dev/null)
if [ -n "$LAST_DEPLOY_TS" ]; then
  # Count ads_user.json files modified since last deploy
  N_NEW=$(find "$SNAPSHOT_DIR" -maxdepth 2 -name "ads_user.json" \
                -newermt "@$LAST_DEPLOY_TS" 2>/dev/null | wc -l | tr -d ' ')
  echo "  conditional check: $N_NEW reviews since last deploy "
  echo "  (threshold $NEW_REVIEWS_THR; override via TVH_TRAIN_NEW_REVIEWS_MIN=1)"
  if [ "$N_NEW" -lt "$NEW_REVIEWS_THR" ]; then
    echo "  → SKIP training (= not enough new data, save ~12h drain cycle)"
    exit 0
  fi
fi

# Training-active marker for the /learning page banner. Posted via
# HTTP (no SMB needed) so the gateway can show "Training läuft seit
# N min" — gateway writes the file on its own filesystem; trap fires
# the DELETE on exit (success OR crash) so a hung script doesn't
# leave the banner stuck on. The mtime tells the banner WHEN
# training started.
$CURL -X POST "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true
trap '$CURL -X DELETE "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true' EXIT

# Pre-train drain check: after a head deploy, Smart-V2 invalidates
# ~250+ cutlists which the daemon re-detects via low-prio queue.
# When cron fires mid-drain, train-head sees a shrunk corpus and
# the gate rejects (= 2026-05-17 incident: 173 < 214 floor).
# Wait up to 30 min for the low-prio queue to drain below 30 before
# starting — daemon makes ~50/h so 30min is enough headroom for
# 25-30 pending markers. Cap at 30min so cron eventually runs even
# if drain stalls (= per-channel detect-spawn loop tends to recover
# but no guarantees).
WAIT_DEADLINE=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "$WAIT_DEADLINE" ]; do
  N=$($CURL "$GATEWAY/api/internal/detect-pending-low" 2>/dev/null \
       | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("pending",[])))' \
       2>/dev/null || echo 999)
  if [ "$N" -lt 30 ]; then
    echo "pre-train drain check: low-prio queue=$N (<30 threshold) — proceeding"
    break
  fi
  echo "pre-train drain check: low-prio queue=$N, waiting up to $((WAIT_DEADLINE - $(date +%s)))s..."
  sleep 60
done

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
  $CURL -o "$TRAIN_OUT/$f" \
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
    --write-pseudo-labels \
    --train-archive "$HOME/.cache/tvd-train-archive" \
    --head-arch mlp32-channel-whisper \
    ${TVH_TRAIN_EXTRA_ARGS:-}
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
  resp=$($CURL -X POST --data-binary "@$BUNDLE" \
      -H "Content-Type: application/gzip" \
      "$GATEWAY/api/internal/head-bundle")
  echo "  upload response: $resp"
  if [ "$?" -ne 0 ]; then
      echo "  WARN: upload failed — head NOT deployed"
      rc=2
  fi
  rm -f "$BUNDLE"
  # Auto-anchor the just-deployed champion off-site so rollback never needs a
  # manual step. Fail-soft (gh/keychain may be unreachable under launchd) — the
  # local full-bundle archive + rollback-head.sh is the reliable path; this is
  # the off-site DR bonus. Only on a real deploy (rc==0, head changed).
  if [ "$rc" -eq 0 ]; then
    # Diagnostic markers: auto-anchor produced NO output at all on the
    # 06-10/11/12 nightly runs (neither ✓ nor failure) though it works run by
    # hand and the deploy was rc==0 — the launchd context skips/swallows it for
    # reasons invisible in the log. These two lines prove whether the block is
    # reached and whether GH_TOKEN is present there. Remove once root-caused.
    echo "auto-anchor: invoking (rc=$rc, GH_TOKEN ${GH_TOKEN:+set}${GH_TOKEN:-UNSET})"
    TVD_MODELS_DIR="$TRAIN_OUT" \
      "$HOME/src/tv-detect/scripts/model-anchor.sh" auto 2>&1 | sed 's/^/  /'
    echo "auto-anchor: model-anchor.sh auto done (pipe-status=${PIPESTATUS[0]})"
  fi
elif [ "$rc" -eq 3 ]; then
  # Python rejected the candidate (regression vs last-deployed). Skip
  # bundle+upload — Pi keeps its current head. Without this gate, the
  # local /tmp/tv-train-head-out/head.bin (which can drift from Pi
  # after a manual rollback) would be bundled + pushed, silently
  # overwriting Pi with whatever stale candidate happened to be in
  # the local dir (14.05 incident: rejected 0.87 candidate but
  # uploaded the morning's 0.83 regression that was left in /tmp).
  echo "REJECTED by python — skipping bundle+upload (Pi keeps current head)"
  rc=0  # not a script-level failure, just a no-op deploy
fi

# Persist the wall-time of this run to the history file so the
# /learning banner can derive an ETA for future training runs as
# median(last-N-durations) - elapsed. JSON-lines format = append-
# only, gateway parses cheaply line-by-line. POSTed via HTTP =
# gateway appends server-side, no SMB write.
TRAIN_DUR=$(( $(date +%s) - TRAIN_START_TS ))
TRAIN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
$CURL -X POST -H "Content-Type: application/json" \
    -d "{\"ts\":\"$TRAIN_TS\",\"dur_s\":$TRAIN_DUR,\"rc\":$rc}" \
    "$GATEWAY/api/internal/training-duration" >/dev/null 2>&1 || true

# Push the LAST history entry — deploy OR reject — to the Pi as a standalone
# attempt record. The head-bundle upload above is gated to deploys only (it
# must never overwrite the Pi head with a rejected candidate), so a rejected
# run never reaches the Pi's head.history.json and would be invisible to
# /api/learning/last-training + HA. .last-attempt.json closes that gap without
# touching the deployed head. Python writes this run's entry as the last array
# element of the local head.history.json in BOTH the deploy and reject paths.
if [ -f "$TRAIN_OUT/head.history.json" ]; then
  "$VENV_PY" -c "import json,sys; h=json.load(open('$TRAIN_OUT/head.history.json')); sys.stdout.write(json.dumps(h[-1]) if h else '')" 2>/dev/null \
    | $CURL -X POST -H "Content-Type: application/json" --data-binary @- \
        "$GATEWAY/api/internal/training-attempt" >/dev/null 2>&1 || true
fi

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
