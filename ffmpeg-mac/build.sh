#!/usr/bin/env bash
# Canonical Mac ffmpeg build: pinned upstream master + local patches/*.patch.
# Replaces brew --HEAD for tv-detect so Mac == Pi (one state).
# Idempotent: re-running rebuilds ffmpeg cleanly.
#
# The libcurl/HTTP-3 URLProtocol was removed 2026-06-17: the only consumer was
# the iOS app, which dropped its ffmpeg stack entirely; nothing on Pi/Mac ever
# opened libcurl:// URLs. gnutls stays as the native TLS backend for plain
# https:// fetches (the daemon's source-fallback uses ffmpeg -i https://…).
set -euo pipefail

ROOT="$HOME/ffmpeg-h3-mac"
SRC="$ROOT/src"
OUT="$ROOT/out"            # ffmpeg install prefix
# Default policy: track LATEST upstream master (resolve HEAD → concrete SHA so
# the checkout below is reproducible). Override with the FFMPEG_COMMIT env to
# pin a specific commit/tag.
COMMIT="${FFMPEG_COMMIT:-$(git ls-remote https://github.com/FFmpeg/FFmpeg.git refs/heads/master | cut -f1)}"
[ -n "$COMMIT" ] || { echo "✗ konnte FFmpeg master HEAD nicht auflösen"; exit 1; }

BREW="$(brew --prefix)"
export PKG_CONFIG_PATH="$BREW/opt/gnutls/lib/pkgconfig:$BREW/lib/pkgconfig"
JOBS="$(sysctl -n hw.ncpu)"

mkdir -p "$SRC" "$OUT"
echo "=== build started $(date) | commit=$COMMIT | jobs=$JOBS ==="

# --- ffmpeg: pinned master, plain ---
echo "=== ffmpeg fetch @ $COMMIT ==="
if [ ! -d "$SRC/ffmpeg/.git" ]; then
  mkdir -p "$SRC/ffmpeg" && cd "$SRC/ffmpeg" && git init -q && git remote add origin https://github.com/FFmpeg/FFmpeg.git
fi
cd "$SRC/ffmpeg"
git fetch --depth 1 origin "$COMMIT"
git checkout -qf FETCH_HEAD
git clean -qfdx

# --- local patches ---
# Not cosmetic: patches/0001-ffmpeg_sched-fix-decoder-starvation.patch fixes an
# upstream regression (03dfac56) that makes ffmpeg drop ~58% of video frames,
# silently and nondeterministically, whenever a second output stream is mapped.
# Building master WITHOUT it produces VODs holding ~40% of their recording and
# exits 0 while doing so. Fail loudly rather than build an unpatched binary —
# if a patch stops applying it has most likely landed upstream, in which case
# delete it here after verifying with a frame count.
shopt -s nullglob
for p in "$ROOT"/patches/*.patch; do
  echo "=== patch: $(basename "$p") ==="
  git apply --check "$p" 2>/dev/null || {
    echo "✗ $p laesst sich nicht anwenden — vermutlich upstream gefixt."
    echo "  Pruefen, dann hier entfernen. KEIN ungepatchter Build."
    exit 1
  }
  git apply "$p"
done
shopt -u nullglob

echo "=== configure ==="
./configure --prefix="$OUT" \
  --enable-gpl --enable-version3 \
  --enable-libx264 --enable-chromaprint \
  --enable-gnutls \
  --disable-doc
echo "=== make ==="
make -j"$JOBS"
make install
echo "=== DONE $(date) ==="
"$OUT/bin/ffmpeg" -version | head -1
