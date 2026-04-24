# Phase 6: Python integration (shadow mode)

`tv-detect` runs ALONGSIDE `comskip` on every completed recording the
Mac scanner processes. Comskip's cutlist is still authoritative; the
shadow output is captured for diff comparison so we can decide when
(and per-channel whether) to swap defaults.

## Wiring

`~/bin/tv-comskip.sh` — Mac launchd agent — does two passes per run:

1. **Live-fold-in** — when a fresh comskip run just succeeded for a
   recording, immediately also call `tv-detect` with the matching
   per-channel logo and append a row to the shadow log.
2. **Back-fill** — at end of each run, walk every completed recording
   that has a comskip `.txt` but no `.tvd.txt` yet, and shadow up to
   `TVD_SHADOW_MAX_PER_RUN = 3` of them. Caps the per-cycle CPU.

Both passes only fire for channels that have a trained logo at
`/mnt/tv/hls/.tvd-logos/<slug>.logo.txt`. Untrained channels are
skipped silently.

## File layout

| Path | Purpose |
|---|---|
| `~/.local/bin/tv-detect` | symlink to dev build |
| `~/.local/bin/tv-detect-train-logo` | symlink to dev build |
| `/mnt/tv/hls/.tvd-logos/<slug>.logo.txt` | per-channel trained logo (SMB-shared, also visible on Pi) |
| `<rec_dir>/<basename>.tvd.txt` | tv-detect cutlist sibling to comskip's `.txt` |
| `~/Library/Logs/tv-detect-shadow.csv` | per-recording diff row, append-only |
| `~/Library/Logs/tv-comskip.log` | inline progress (existing log) |

## Diff format

Each CSV row: `ts, uuid, channel, comskip_n, tvd_n, tvd_secs, diff`.

`diff` is one of:
- `'+1/-2  +5/+0'` — per-block (start_delta, end_delta) in seconds, joined by `  `
- `'count 1 vs 2'` — block count mismatch
- `'both-empty'`  — both said no blocks

## Reading the stats

```bash
~/bin/tv-detect-shadow-stats.py
```

Buckets:
- **frame-perfect** — every block boundary within ±10 s of comskip
- **close** — count match, every boundary within ±60 s
- **loose** — count match but a boundary off by >60 s
- **count-mismatch** — block counts disagree (both directions; sometimes
  comskip's cutlist is itself nonsensical, e.g. `0-75 s`)
- **both-empty** — both agreed: no blocks

`acceptable = frame-perfect + close + both-empty`.

## When to swap

Heuristic: per channel, when ≥80% of recordings land in acceptable
AND zero land in count-mismatch over a 7-day window, that channel is
ready to swap from comskip to tv-detect by default. Easier channels
(VOX, RTL Unter uns/GZSZ) likely meet this faster than channels with
animated/transparent logos (VOX news, RTL Wetter, ProSieben event
specials).

## Path to live cutover

Once a channel passes the threshold, in `comskip()` of tv-comskip.sh
flip the order: try tv-detect first, fall back to comskip on error
or low-confidence detection. Keep shadow logging in reverse so we
keep collecting comparison data after the swap.

Pi-side `hls-gateway/service.py` shadow integration is the next
logical step (Phase 6b) so we get coverage of recordings the Mac
scanner doesn't reach (lock-contention edge cases, Mac off-LAN, etc.).
