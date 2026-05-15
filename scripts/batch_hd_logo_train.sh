#!/bin/bash
# Sequential HD-logo train for all channels currently routed via
# DVB-C HD svc (= the 31+2 channels bound to HD by bind_hd_routing.py
# + earlier one-offs). Each train takes ~5 min capture + ~10 s
# processing on Mac. Sequential to avoid tuner contention with live
# DVR recordings (= one tuner busy at a time, channel-pinning won't
# steal from a scheduled recording).
#
# Skips channels that already have an HD logo trained.
# Skips channels with an active DVR subscription on their mux.
#
# Usage:
#   batch_hd_logo_train.sh [<minutes>]
# Default: 5 min capture per channel.

MINUTES="${1:-5}"
TRAIN="$HOME/src/tvheadend/mac-daemon/tv-train-logo-from-live.sh"
LOGO_DIR="$HOME/.cache/tv-detect-daemon/logos"
LOG="$HOME/Library/Logs/batch-hd-logo-train.log"

# Channels to train: all that have an HD svc bound. Pull dynamically
# rather than hard-coding so future bind_hd_routing.py runs flow in.
CHANNELS=$(curl -s "http://raspberrypi5lan:9981/api/channel/grid?limit=500" |
    python3 -c "
import json, sys
d = json.load(sys.stdin)
svcs = json.loads(__import__('urllib.request', fromlist=['x']).urlopen(
    'http://raspberrypi5lan:9981/api/mpegts/service/grid?limit=2000').read())
by_uuid = {s['uuid']: s for s in svcs.get('entries', [])}
for c in d.get('entries', []):
    n = c.get('name','')
    if 'HD' in n: continue
    has_hd = any(by_uuid.get(u, {}).get('svcname','').endswith(' HD')
                 for u in c.get('services', []))
    if not has_hd: continue
    # convert name to slug like the gateway does
    slug = (n.lower().replace(' ', '-').replace('.', '-')
              .replace('+','plus').replace('!','').replace(',',''))
    print(slug)
" | sort -u)

echo "=== batch HD-logo train ($(date)) ===" | tee -a "$LOG"
echo "channels to consider: $(echo "$CHANNELS" | wc -l)" | tee -a "$LOG"

n_skip_existing=0
n_train=0
n_fail=0
for SLUG in $CHANNELS; do
  if [ -f "$LOGO_DIR/$SLUG.logo.hd.txt" ]; then
    n_skip_existing=$((n_skip_existing+1))
    continue
  fi
  echo "" | tee -a "$LOG"
  echo "--- $SLUG ($(date +%H:%M:%S)) ---" | tee -a "$LOG"
  bash "$TRAIN" --hd "$SLUG" "$MINUTES" 2>&1 | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  if [ "$rc" = "0" ]; then
    n_train=$((n_train+1))
  else
    n_fail=$((n_fail+1))
    echo "  FAILED rc=$rc — continuing with next channel" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
echo "  trained:        $n_train" | tee -a "$LOG"
echo "  skipped (exists): $n_skip_existing" | tee -a "$LOG"
echo "  failed:         $n_fail" | tee -a "$LOG"
