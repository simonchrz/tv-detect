#!/bin/bash
# Daily backup of irreplaceable user-labelled training data + small
# regenerable model artifacts that are slow to recompute. Runs from
# launchd at 04:30 (after nightly training, captures latest reviews).
#
# Source of truth = Pi at /mnt/tv/hls. Mirror = local git repo at
# REPO with a private GitHub remote as off-site copy.
#
# What's backed up (ranked by replaceability):
#   Tier 1 (irreplaceable user work, ~150 KB total):
#     - ads_user.json        (per-recording user-confirmed ad blocks)
#   Tier 1.5 (irreplaceable tv-receiver state, ~0.5 MB, in tv-receiver-state/):
#     - dvr.json        (DVR schedules + uuid->recording registry)
#     - autorec.json    (autorec rules)
#     - channels.json   (channel slug->freq/pids map)
#   Tier 2 (regenerable but slow, ~10 MB):
#     - .tvd-models/head.bin            (~5 KB, 30-60 min to retrain)
#     - .tvd-models/head.calibration.json
#     - .tvd-models/head.test-set.json
#     - .channel-config.json
#     - .detection_learning.json
#     - .block_length_prior.json
#     - .block_length_prior_by_show.json
#
# Restore: `git clone <remote> ~/tv-labels-backup`
#   labels:           rsync -av ~/tv-labels-backup/_rec_*/ pi:/mnt/tv/hls/
#   tv-receiver state: rsync -av ~/tv-labels-backup/tv-receiver-state/ pi:/home/simon/bin/
#                      (then: docker restart tv-receiver)
LOG="$HOME/Library/Logs/tv-backup-labels.log"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') ==="

REPO="$HOME/tv-labels-backup"
HOST=raspberrypi5lan
SRC=/mnt/tv/hls

if [ ! -d "$REPO/.git" ]; then
  echo "$REPO is not a git repo — bailing"
  exit 1
fi

# Pull just the files we want. rsync's --include / --exclude lets us
# enumerate the small set without copying the gigabyte HLS segments.
# --delete so deletions on Pi are mirrored (= recordings purged from
# DVR also drop out of backup; old git commits keep the history).
rsync -av --delete \
    --include='_rec_*/' --include='_rec_*/ads_user.json' \
    --include='.tvd-models/' --include='.tvd-models/head.bin' \
    --include='.tvd-models/head.calibration.json' \
    --include='.tvd-models/head.test-set.json' \
    --include='.tvd-models/head.channel-map.json' \
    --include='.channel-config.json' \
    --include='.detection_learning.json' \
    --include='.block_length_prior.json' \
    --include='.block_length_prior_by_show.json' \
    --exclude='*' \
    "$HOST:$SRC/" "$REPO/" || {
  echo "rsync failed — bailing"
  exit 1
}

# tv-receiver runtime state (lives ONLY on the Pi SSD, irreplaceable user
# config): DVR schedules + the uuid->recording registry (dvr.json), autorec
# rules (autorec.json), channel slug->freq/pids map (channels.json). epg.json
# is regenerable from the XMLTV feed -> skipped (3.9 MB churn). No --delete:
# these three always exist; nothing to prune.
mkdir -p "$REPO/tv-receiver-state"
rsync -av \
    --include='channels.json' --include='dvr.json' --include='autorec.json' \
    --exclude='*' \
    "$HOST:/home/simon/bin/" "$REPO/tv-receiver-state/" || {
  echo "tv-receiver-state rsync failed — bailing"
  exit 1
}

cd "$REPO" || exit 1

# Skip the commit dance if rsync produced no changes.
if [ -z "$(git status --porcelain)" ]; then
  echo "no changes"
  exit 0
fi

# Stats for the commit message.
n_user=$(find . -name 'ads_user.json' | wc -l | tr -d ' ')
n_changed=$(git status --porcelain | wc -l | tr -d ' ')

git add -A
git commit -q -m "snapshot $(date -u +%Y%m%dT%H%M%SZ): $n_user reviews, $n_changed changed"

if git remote get-url origin >/dev/null 2>&1; then
  git push -q origin main 2>&1 || echo "push failed (non-fatal)"
fi

echo "committed: $n_changed changed, $n_user total reviews"
