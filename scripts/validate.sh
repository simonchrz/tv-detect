#!/bin/bash
# tv-detect validation: train per-channel logos, run detection on all
# recordings, diff cutlists against comskip's .txt ground truth.
#
# macOS /bin/bash is 3.2 and has no associative arrays; we use a
# plain table of "slug|relative-path" instead.

set -u
TVD=/Users/simon/src/tv-detect/build/tv-detect
TRAIN=/Users/simon/src/tv-detect/build/tv-detect-train-logo
LOGO_DIR=/tmp/tvd-logos
HLS_ROOT=/Users/simon/mnt/pi-tv/hls
TVH_ROOT=/Users/simon/mnt/pi-tv

mkdir -p "$LOGO_DIR"

TRAIN_TABLE='
vox|CSI_ Den Tätern auf der Spur/CSI_ Den Tätern auf der Spur $2026-04-24-1125.ts
rtl|Ulrich Wetzel - Das Strafgericht/Ulrich Wetzel - Das Strafgericht $2026-04-24-1000.ts
prosieben|The Middle/The Middle $2026-04-24-0918.ts
kabel-eins|Mein Lokal, Dein Lokal - Der Profi kommt/Mein Lokal, Dein Lokal - Der Profi kommt $2026-04-24-1756.ts
'

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed -e 's/ä/ae/g;s/ö/oe/g;s/ü/ue/g;s/ß/ss/g' -e 's/[^a-z0-9]/-/g' -e 's/--*/-/g;s/^-//;s/-$//'; }

# Step 1: train one logo per channel (skip if exists).
echo "=== TRAINING ==="
echo "$TRAIN_TABLE" | while IFS='|' read -r ch rel; do
  [ -z "$ch" ] && continue
  out="$LOGO_DIR/$ch.logo.txt"
  if [ -s "$out" ]; then
    echo "[$ch] cached $out"
    continue
  fi
  src="$TVH_ROOT/$rel"
  if [ ! -f "$src" ]; then
    echo "[$ch] MISSING training source: $src"
    continue
  fi
  echo "[$ch] training from $(basename "$src") ..."
  "$TRAIN" --quiet --edge-threshold 40 --output "$out" "$src" 2>&1 | tail -3
done

# Step 2: for each recording, run tv-detect + diff against comskip .txt.
echo
echo "=== VALIDATION ==="
printf '%-45s %-12s %-25s %-25s %s\n' RECORDING CHANNEL TVD_BLOCKS COMSKIP_BLOCKS DIFF
echo "----------------------------------------------------------------------------------------------------------------------------------"

curl -s "http://raspberrypi5lan:9981/api/dvr/entry/grid_finished?limit=500" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for e in d.get('entries', []):
    if e.get('sched_status') != 'completed': continue
    fn = e.get('filename','').replace('/recordings/','/Users/simon/mnt/pi-tv/',1)
    print(f\"{e.get('uuid','')}\t{e.get('channelname','')}\t{fn}\")
" | while IFS=$'\t' read -r uuid chname src; do
  ch=$(slug "$chname")
  logo="$LOGO_DIR/$ch.logo.txt"
  [ ! -s "$logo" ] && continue   # skip channels we didn't train
  [ ! -f "$src" ] && continue    # source file missing on disk

  # tv-detect blocks
  tvd_json=$("$TVD" --quiet --workers 4 --logo "$logo" --output summary "$src" 2>/dev/null)
  tvd_blocks=$(echo "$tvd_json" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(','.join(f'{int(s)}-{int(e)}' for s,e in d.get('blocks',[])) or '-')
except: print('?')")

  # comskip blocks (parse .txt from HLS dir; same logic as Python parse_comskip)
  cs_dir="$HLS_ROOT/_rec_$uuid"
  cs_blocks=$(python3 - <<PY
import re, os, glob
txts = [p for p in glob.glob(os.path.join('''$cs_dir''', '*.txt')) if not p.endswith('.logo.txt')]
if not txts: print('-'); raise SystemExit
fps = 25.0; raw = []
LINE = re.compile(r'(\d+)\s+(\d+)\s*\$')
try:
    for line in open(txts[0], errors='ignore'):
        line = line.strip().replace('\x00','')
        if not line or line.startswith('-'): continue
        if 'FRAMES AT' in line:
            try:
                f = float(line.split()[-1])/100.0
                if f > 0: fps = f
            except: pass
            continue
        m = LINE.search(line)
        if not m: continue
        a = float(m.group(1))/fps; b = float(m.group(2))/fps
        if b-a >= 60: raw.append((a,b))
    raw.sort()
    merged = []
    for a,b in raw:
        if merged and a-merged[-1][1] <= 45.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1],b))
        else: merged.append((a,b))
    print(','.join(f'{int(a)}-{int(b)}' for a,b in merged) or '-')
except Exception as ex: print(f'ERR: {ex}')
PY
)

  # diff: count match if same number of blocks AND boundaries within 30s each
  diff=$(python3 - <<PY
def parse(s):
    if s in ('-','?',''): return []
    out=[]
    for chunk in s.split(','):
        a,b=chunk.split('-')
        out.append((int(a),int(b)))
    return out
a, b = parse('$tvd_blocks'), parse('$cs_blocks')
if not a and not b: print('both-empty')
elif len(a) != len(b): print(f'count-mismatch {len(a)} vs {len(b)}')
else:
    ds=[]
    for (sa,ea),(sb,eb) in zip(a,b):
        ds.append(f'Δ{sa-sb:+d}/{ea-eb:+d}')
    print('  '.join(ds))
PY
)

  rec_short=$(basename "$src" | sed 's/\$2026.*//' | cut -c1-44)
  printf '%-45s %-12s %-25s %-25s %s\n' "$rec_short" "$ch" "${tvd_blocks:0:24}" "${cs_blocks:0:24}" "$diff"
done
