#!/usr/bin/env bash
# refresh-ffmpeg-master.sh — keep the Mac's ffmpeg/ffprobe on FFmpeg master.
#
# tv-detect runs ONLY on the Mac (launchd agent com.user.tv-thumbs-daemon),
# shelling out to ffmpeg/ffprobe on $PATH — which is Homebrew's
# /opt/homebrew/bin/ffmpeg. The Pi containers (tv-receiver/tv-recorder) build
# ffmpeg from master in their image; this is the Mac analogue.
#
# We use Homebrew's --HEAD build (not a from-source clone) because:
#   • it pulls master and compiles it with all the brew-managed deps (x264,
#     etc.) and videotoolbox decode enabled by default — what the daemon wants,
#   • it installs to /opt/homebrew/bin, so the daemons need NO path change,
#   • `brew upgrade --fetch-HEAD` rebuilds ONLY when master actually moved —
#     same "refresh only on movement" behaviour as the containers' build-arg
#     cache key.
# Chromaprint: the brew ffmpeg formula omits the chromaprint muxer, but the Mac
# daemon uses the separate `fpcalc` CLI anyway (tv-spot-extract.py), so that's
# fine — unchanged from today.
set -euo pipefail

command -v brew >/dev/null 2>&1 || { echo "✗ Homebrew nicht gefunden"; exit 1; }

if brew list --versions ffmpeg 2>/dev/null | grep -q HEAD; then
  echo "→ ffmpeg ist HEAD — prüfe master + rebuild falls bewegt (brew upgrade --fetch-HEAD)"
  brew upgrade --fetch-HEAD ffmpeg || true
else
  echo "→ ffmpeg install von master (brew --HEAD)"
  # A linked stable keg blocks `install --HEAD` ("ffmpeg X is already installed,
  # first run brew unlink"). Unlink it first, then build + link HEAD. The
  # unlink is a harmless no-op when nothing is installed/linked.
  brew unlink ffmpeg 2>/dev/null || true
  brew install --HEAD ffmpeg
fi

echo "✓ ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
