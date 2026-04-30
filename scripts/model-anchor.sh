#!/bin/bash
# Capture and restore semantic snapshots of the trained NN model.
#
# The nightly retrain leaves a rolling timestamped archive at
#   $TVD_MODELS_DIR/archive/head.<unix-ts>.bin
# but those are opaque (timestamps only) and only on the Pi NVMe —
# if the SSD dies, every snapshot dies with it. Anchors are off-Pi
# (GitHub Releases), semantically named, and bundle backbone +
# head + history together so a restore is one command.
#
# Usage:
#   ./scripts/model-anchor.sh create <tag> [--notes "free text"]
#       Bundle current head.bin + backbone.onnx + history snippet,
#       create a git tag at HEAD, push it, attach the bundle as a
#       GitHub release. Notes auto-include latest train metrics
#       (Acc, IoU, n recs) — your free text is appended.
#
#   ./scripts/model-anchor.sh install <tag>
#       Download the named release from GitHub, back up existing
#       local files (rename to .bak.<ts>), copy the anchor into
#       $TVD_MODELS_DIR. The deployed tv-detect picks up head.bin
#       automatically via mtime watch.
#
#   ./scripts/model-anchor.sh list
#       Show all anchors (gh release list, prefix-filtered to the
#       anchor naming scheme).
#
# Requirements: gh CLI authenticated for this repo.
#
# Models dir auto-detect: Mac SMB mount → Pi NVMe → env override.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANCHOR_PREFIX="model-anchor-"

PI_HOST="${TVD_PI_HOST:-raspberrypi5lan}"
PI_REMOTE_DIR="/mnt/nvme/tv/hls/.tvd-models"

# Resolve a local directory containing the model files. If none of
# the known mountpoints exist, scp them from the Pi into a tmp dir
# (works after the Mac SMB mount has died, which it does after every
# Pi reboot). install does the reverse: writes locally if the dir
# is real, scp's to the Pi if the dir was scp'd in.
if [ -n "${TVD_MODELS_DIR:-}" ]; then
  MODELS_DIR="$TVD_MODELS_DIR"; MODELS_REMOTE=0
elif [ -d "$HOME/mnt/pi-tv/hls/.tvd-models" ]; then
  MODELS_DIR="$HOME/mnt/pi-tv/hls/.tvd-models"; MODELS_REMOTE=0
elif [ -d "$PI_REMOTE_DIR" ]; then
  MODELS_DIR="$PI_REMOTE_DIR"; MODELS_REMOTE=0
elif [ -d "/mnt/tv/hls/.tvd-models" ]; then
  MODELS_DIR="/mnt/tv/hls/.tvd-models"; MODELS_REMOTE=0
elif ssh -o ConnectTimeout=3 -o BatchMode=yes "$PI_HOST" "test -d $PI_REMOTE_DIR" 2>/dev/null; then
  MODELS_DIR="$(mktemp -d)"; MODELS_REMOTE=1
  trap 'rm -rf "$MODELS_DIR"' EXIT
  echo "→ SMB not mounted; pulling models from $PI_HOST:$PI_REMOTE_DIR ..." >&2
  scp -q "$PI_HOST:$PI_REMOTE_DIR/{head.bin,backbone.onnx,head.history.json}" "$MODELS_DIR/" 2>/dev/null || \
    for f in head.bin backbone.onnx head.history.json; do
      scp -q "$PI_HOST:$PI_REMOTE_DIR/$f" "$MODELS_DIR/$f" 2>/dev/null || true
    done
else
  echo "error: no .tvd-models dir found and Pi unreachable; set TVD_MODELS_DIR" >&2
  exit 1
fi

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \?//' >&2
  exit 1
}

# Pull latest entry from head.history.json so the release notes
# carry the metrics that this snapshot was made at.
fmt_metrics() {
  local hist="$MODELS_DIR/head.history.json"
  [ -f "$hist" ] || { echo "(no history.json — metrics unknown)"; return; }
  python3 - "$hist" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
if not h:
    print("(empty history)")
    raise SystemExit
e = h[-1]
def f(k, fmt="{}"):
    v = e.get(k)
    return fmt.format(v) if v is not None else "?"
# IoU/Acc are stored as fractions [0,1]
iou = e.get("test_iou"); acc = e.get("test_acc"); ta = e.get("train_acc")
print(f"- ts:           {f('ts')}")
print(f"- Block-IoU:    {iou*100:.1f}%" if iou is not None else "- Block-IoU:    ?")
print(f"- Test Acc:     {acc*100:.1f}%" if acc is not None else "- Test Acc:     ?")
print(f"- Train Acc:    {ta*100:.1f}%"  if ta  is not None else "- Train Acc:    ?")
print(f"- n test/train: {f('n_test_recs')}/{f('n_train_recs')}")
print(f"- deployed:     {f('deployed')}")
print(f"- reason:       {f('reason')}")
PY
}

cmd_create() {
  local raw_tag="${1:-}"; shift || true
  local notes=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --notes) notes="$2"; shift 2 ;;
      *) echo "unknown arg: $1" >&2; usage ;;
    esac
  done
  [ -n "$raw_tag" ] || { echo "error: tag required" >&2; usage; }
  local tag="${ANCHOR_PREFIX}${raw_tag}"

  for f in head.bin backbone.onnx; do
    [ -f "$MODELS_DIR/$f" ] || { echo "error: missing $MODELS_DIR/$f" >&2; exit 1; }
  done

  local stage; stage="$(mktemp -d)"
  trap "rm -rf '$stage'" EXIT
  cp "$MODELS_DIR/head.bin"      "$stage/head.bin"
  cp "$MODELS_DIR/backbone.onnx" "$stage/backbone.onnx"
  [ -f "$MODELS_DIR/head.history.json" ] && \
    cp "$MODELS_DIR/head.history.json" "$stage/head.history.json"
  # Platt-calibration sidecar (head.calibration.json) — written by
  # train-head.py since 2026-04-30. Optional: legacy heads have no
  # sidecar and detection still works (calibrated_proba falls back
  # to raw clf.predict_proba). Bundle when present so the anchor is
  # bit-for-bit reproducible including the calibration constants.
  [ -f "$MODELS_DIR/head.calibration.json" ] && \
    cp "$MODELS_DIR/head.calibration.json" "$stage/head.calibration.json"

  local body; body=$(mktemp)
  {
    echo "## Metrics at snapshot time"
    echo
    fmt_metrics
    if [ -n "$notes" ]; then
      echo
      echo "## Notes"
      echo
      echo "$notes"
    fi
    echo
    echo "## Restore"
    echo
    echo '```sh'
    echo "./scripts/model-anchor.sh install $raw_tag"
    echo '```'
  } > "$body"

  cd "$REPO_ROOT"
  echo "→ creating annotated git tag $tag at HEAD..."
  git tag -a "$tag" -F "$body"
  git push origin "$tag"

  echo "→ creating GitHub release $tag with bundled artefacts..."
  # Build the asset list dynamically so optional sidecars don't
  # break the upload when missing on legacy installs.
  local assets=("$stage/head.bin" "$stage/backbone.onnx")
  [ -f "$stage/head.history.json" ]     && assets+=("$stage/head.history.json")
  [ -f "$stage/head.calibration.json" ] && assets+=("$stage/head.calibration.json")
  gh release create "$tag" \
    --title "Model anchor: $raw_tag" \
    --notes-file "$body" \
    "${assets[@]}"

  rm -f "$body"
  echo "✓ anchor created: $tag"
  echo "  install elsewhere with:  ./scripts/model-anchor.sh install $raw_tag"
}

cmd_install() {
  local raw_tag="${1:-}"
  [ -n "$raw_tag" ] || { echo "error: tag required" >&2; usage; }
  local tag="${ANCHOR_PREFIX}${raw_tag}"
  local stage; stage="$(mktemp -d)"
  trap "rm -rf '$stage'" EXIT

  echo "→ downloading $tag from GitHub..."
  gh release download "$tag" --dir "$stage"
  for f in head.bin backbone.onnx; do
    [ -f "$stage/$f" ] || { echo "error: $f missing from release" >&2; exit 1; }
  done

  local ts; ts=$(date +%s)
  mkdir -p "$MODELS_DIR"
  for f in head.bin backbone.onnx head.history.json head.calibration.json; do
    [ -f "$MODELS_DIR/$f" ] && cp "$MODELS_DIR/$f" "$MODELS_DIR/$f.bak.$ts"
    [ -f "$stage/$f" ] && cp "$stage/$f" "$MODELS_DIR/$f"
  done

  echo "✓ installed $tag → $MODELS_DIR"
  echo "  previous files backed up as *.bak.$ts (delete after verifying)"
  echo "  tv-detect picks up head.bin automatically via mtime watch"
}

cmd_list() {
  gh release list --limit 50 | grep -F "$ANCHOR_PREFIX" || \
    echo "(no anchors yet — create one with: model-anchor.sh create <tag>)"
}

[ $# -ge 1 ] || usage
case "$1" in
  create)  shift; cmd_create  "$@" ;;
  install) shift; cmd_install "$@" ;;
  list)    shift; cmd_list    "$@" ;;
  -h|--help) usage ;;
  *) echo "unknown command: $1" >&2; usage ;;
esac
