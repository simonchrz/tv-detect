#!/usr/bin/env bash
# Backfill whisper.json transcripts for the /api/recordings/search FTS5 index
# from the Mac source cache. Search-transcript coverage stalled 2026-06-01 when
# the whisper ad-refiner was superseded by the speaker signal (WHISPER_ENABLE
# off → no whisper.json). The daemon's WHISPER_TRANSCRIBE=1 fix handles NEW
# recordings; this script re-fills the ~334 already-recorded ones.
#
# Idempotent + resumable: skips recordings already transcribed (re-pushes their
# json so the index is complete), only classifies the missing ones. Serial (one
# classify at a time) to stay gentle on the detect daemon — run when it's idle.
#
#   ./tv-whisper-backfill.sh            # full backlog
#   FILTER=rtlzwei ./tv-whisper-backfill.sh   # only uuids matching FILTER
set -u

VENV=/Users/simon/ml/tv-classifier/.venv/bin/python
CLASSIFY=/Users/simon/src/tv-detect/daemon/tv-whisper-classify.py
SRCDIR=/Users/simon/.cache/tv-detect-daemon/source
WCACHE="$HOME/.cache/tv-whisper"
GATEWAY="${GATEWAY:-https://raspberrypi5lan:8443}"
FILTER="${FILTER:-}"
mkdir -p "$WCACHE"

push() { # $1=uuid $2=jsonpath
  curl -sk -X POST "$GATEWAY/api/internal/whisper-reindex?uuid=$1" \
    --data-binary @"$2" -H 'Content-Type: application/json' >/dev/null 2>&1
}

shopt -s nullglob
ts_files=("$SRCDIR"/dvr-*.ts)
total=${#ts_files[@]}
n=0; done=0; skip=0; fail=0
for src in "${ts_files[@]}"; do
  uuid=$(basename "$src" .ts)
  [ -n "$FILTER" ] && [[ "$uuid" != *"$FILTER"* ]] && continue
  n=$((n+1))
  out="$WCACHE/$uuid.whisper.json"
  if [ -f "$out" ] && [ "$out" -nt "$src" ]; then
    push "$uuid" "$out"; skip=$((skip+1)); continue
  fi
  printf '[%d/%d] classify %s\n' "$n" "$total" "$uuid"
  if "$VENV" "$CLASSIFY" --src "$src" --output "$out" >/dev/null 2>&1 && [ -f "$out" ]; then
    push "$uuid" "$out"; done=$((done+1))
  else
    echo "  FAIL $uuid"; fail=$((fail+1))
  fi
done
echo "backfill complete: scanned=$n transcribed=$done already-cached=$skip failed=$fail"
