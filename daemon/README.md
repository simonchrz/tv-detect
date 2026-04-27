# mac-daemon

Mac-side scripts that take heavy CPU work off the Pi by polling the
hls-gateway HTTP API for pending jobs (HLS-remux, thumbnails, ad
detection), running ffmpeg / tv-detect locally, and POSTing results
back. See [`../docs/mac-handlers.md`](../docs/mac-handlers.md) for
the architecture and HTTP-vs-SMB rationale.

## Files

- **`tv-thumbs-daemon.py`** — long-running launchd agent. Single
  poll loop services three job types: combined HLS-remux + thumbs
  pass, plus ad-detection via tv-detect. Caches the source `.ts`
  on local NVMe (LRU at `SOURCE_CACHE_MAX_GB`, hourly orphan GC
  against the gateway's `/recording-uuids` endpoint). Runs ffmpeg
  cropdetect per-recording for letterbox compensation, fetches
  per-channel bumper templates for boundary-snap.
- **`tv-train-head.sh`** — nightly retrain wrapper (launchd 03:30).
  Mounts SMB sanity-check, exports a launchd-friendly PATH, runs
  `train-head.py` from the tv-detect repo with the active flag set
  (logo + minute-prior + self-training + pseudo-labels +
  active-learning surface).
- **`mount-pi-tv.sh`** — pulls the SMB password from login-keychain
  and mounts the Pi `tv` share at `~/mnt/pi-tv`. Required because
  Finder's auto-mount lands at `/Volumes/tv` which other scripts
  don't recognise; needed after every Pi reboot.
- **`capture-bumper.py`** — captures one bumper-template frame from
  an existing recording for the daemon's bumper-snap refinement.
  Usage: `capture-bumper.py <uuid_prefix> <timestamp_s> [name]`.
  Workflow: pause the player at a clean station-id frame, note the
  displayed timestamp, run the script — channel slug is auto-derived,
  PNG lands in `/mnt/nvme/tv/hls/.tvd-bumpers/<slug>/`, daemon picks
  it up on the next detect job. Multiple per channel cover color
  variants of the same animation (max IoU wins).

## Install

These files live here for version control; the runtime paths
(`~/bin/...`) are symlinks. Verify after a fresh checkout:

```sh
ln -sf $PWD/tv-thumbs-daemon.py ~/bin/tv-thumbs-daemon.py
ln -sf $PWD/tv-train-head.sh    ~/bin/tv-train-head.sh
ln -sf $PWD/mount-pi-tv.sh      ~/bin/mount-pi-tv.sh
ln -sf $PWD/capture-bumper.py   ~/bin/capture-bumper.py
```

The `~/Library/LaunchAgents/com.user.tv-thumbs-daemon.plist` agent
references `~/bin/tv-thumbs-daemon.py` — launchd dereferences the
symlink at exec time, so the canonical edit target stays the repo
copy.

## Editing

Edit the file in this directory directly; the `~/bin` symlink picks
it up automatically. After editing the daemon, restart with:

```sh
launchctl kickstart -k gui/$(id -u)/com.user.tv-thumbs-daemon
```
