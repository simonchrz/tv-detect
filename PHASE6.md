# Phase 6: Python integration (full swap)

The Mac scanner (`~/bin/tv-comskip.sh`, despite the name) now calls
`tv-detect` instead of `comskip` on every completed recording.
Output filename is unchanged (`<basename>.txt`), so existing parsers
(`hls-gateway/_rec_parse_comskip`, `tv-live-comskip/parse_comskip`)
consume it as-is. Comskip is no longer invoked.

Channels without a trained tv-detect logo (e.g. public broadcasters,
or channels we haven't recorded yet) get an empty cutlist marker
(header only) so the scanner stops re-trying them forever. For
public broadcasters this is the correct behaviour — they don't run
ads, so an empty block list is the truth.

The earlier shadow phase (running both side-by-side, logging to
`~/Library/Logs/tv-detect-shadow.csv`) is removed: its data
informed the swap decision and is no longer needed.

## Wiring

`~/bin/tv-comskip.sh` — Mac launchd agent — for each recording that
has no cutlist yet:

1. Looks up the channel's slug.
2. If a logo exists at `/mnt/tv/hls/.tvd-logos/<slug>.logo.txt`,
   spawns `tv-detect --workers 4 --logo $logo --output cutlist $src`,
   redirecting stdout into `<basename>.txt`.
3. If no logo exists, writes an empty cutlist marker so the scanner
   doesn't loop on it.

Cooperative `.scanning` lock-file behaviour is unchanged — Mac and
Pi-side handlers still arbitrate so they don't race on the same
recording.

## File layout

| Path | Purpose |
|---|---|
| `~/.local/bin/tv-detect` | symlink to dev build |
| `~/.local/bin/tv-detect-train-logo` | symlink to dev build |
| `/mnt/tv/hls/.tvd-logos/<slug>.logo.txt` | per-channel trained logo (SMB-shared, also visible on Pi) |
| `<rec_dir>/<basename>.txt` | cutlist (was comskip's, now tv-detect's — same filename, same format) |
| `~/Library/Logs/tv-comskip.log` | inline progress (existing log, name kept for launchd compatibility) |

## Re-scanning existing recordings

Old recordings keep their comskip-produced cutlists by default. To
force tv-detect re-scan, delete the cutlist:

```bash
rm /Users/simon/mnt/pi-tv/hls/_rec_<uuid>/<basename>.txt
```

Next launchd cycle (within 60 s) will pick it up and write a fresh
tv-detect cutlist.

## Pi-side swap (Phase 6b)

`hls-gateway/service.py` has its own `_rec_cskip_spawn` that runs
comskip on completed recordings. This is gated on
`_mac_comskip_alive()`, so when the Mac scanner is up the Pi never
runs detection at all — Mac handles everything.

The Pi-side path matters when:
- Mac is off-LAN or its launchd agent is down
- The user opens a recording in the player before Mac has scanned it

Phase 6b ships `tv-detect-linux-arm64` into the hls-gateway container
and replaces the comskip subprocess call with tv-detect. Same logo
files are already SMB-shared, so both sides read identical templates.
