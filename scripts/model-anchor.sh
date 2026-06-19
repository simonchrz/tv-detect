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
# Pin the release repo explicitly. Without this, `gh release …` infers the repo
# from the cwd's git remote — which silently breaks the nightly auto-anchor:
# launchd runs the trainer with cwd=/ (no WorkingDirectory set), so gh dies with
# "not a git repository" and, under `set -e`, takes the whole script with it
# before any error is logged. Anchoring stalled 2026-06-12 this way.
ANCHOR_REPO="${TVD_ANCHOR_REPO:-simonchrz/tv-detect}"

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
  # Pull the head, backbone AND the sidecars: a channel-aware head is
  # un-restorable without head.channel-map.json, and cmd_auto refuses to
  # anchor when it's missing. Omitting them here (pre-2026-06-08) made a
  # manual `auto` from the Mac (SMB unmounted) skip with "no channel-map".
  for f in head.bin backbone.onnx head.history.json \
           head.calibration.json head.channel-map.json head.test-set.json; do
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
  # Channel-map + test-set sidecars (since 2026-05-30): a channel-aware head
  # restored WITHOUT its channel-map misaligns the one-hot columns = broken
  # inference. Bundle them so an anchor is a SELF-CONTAINED, restorable unit.
  [ -f "$MODELS_DIR/head.channel-map.json" ] && \
    cp "$MODELS_DIR/head.channel-map.json" "$stage/head.channel-map.json"
  [ -f "$MODELS_DIR/head.test-set.json" ] && \
    cp "$MODELS_DIR/head.test-set.json" "$stage/head.test-set.json"

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
  [ -f "$stage/head.channel-map.json" ] && assets+=("$stage/head.channel-map.json")
  [ -f "$stage/head.test-set.json" ]    && assets+=("$stage/head.test-set.json")
  gh release create "$tag" --repo "$ANCHOR_REPO" \
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
  gh release download "$tag" --repo "$ANCHOR_REPO" --dir "$stage"
  [ -f "$stage/head.bin" ] || { echo "error: head.bin missing from release" >&2; exit 1; }
  # backbone.onnx is optional: lightweight auto-anchors omit it (it changes
  # rarely) — install then keeps whatever backbone is already deployed.
  [ -f "$stage/backbone.onnx" ] || echo "  note: no backbone in release — keeping current backbone.onnx"

  local ts; ts=$(date +%s)
  mkdir -p "$MODELS_DIR"
  # head.bin LAST so a daemon watching it never sees a new head with the
  # old (mismatched) channel-map.
  for f in backbone.onnx head.history.json head.calibration.json \
           head.channel-map.json head.test-set.json head.bin; do
    [ -f "$MODELS_DIR/$f" ] && cp "$MODELS_DIR/$f" "$MODELS_DIR/$f.bak.$ts"
    [ -f "$stage/$f" ] && cp "$stage/$f" "$MODELS_DIR/$f"
  done

  # When models live on the Pi but SMB isn't mounted, MODELS_DIR is a
  # local tmp-staging dir — push the installed files back to the Pi
  # so tv-detect actually picks them up. Without this, install
  # silently succeeds while leaving the Pi running the old (or worse)
  # head; rolling back a regression looks effective until you query
  # head.history on the Pi and find nothing changed.
  if [ "${MODELS_REMOTE:-0}" = "1" ]; then
    echo "→ pushing models back to $PI_HOST:$PI_REMOTE_DIR ..."
    ssh "$PI_HOST" "mkdir -p '$PI_REMOTE_DIR/rollback-bak-$ts' && \
      for f in head.bin backbone.onnx head.history.json head.calibration.json head.channel-map.json head.test-set.json; do \
        [ -f '$PI_REMOTE_DIR'/\$f ] && cp '$PI_REMOTE_DIR'/\$f '$PI_REMOTE_DIR/rollback-bak-$ts/'; \
      done"
    # head.bin LAST (daemon mtime-watch consistency, see above).
    for f in backbone.onnx head.history.json head.calibration.json \
             head.channel-map.json head.test-set.json head.bin; do
      [ -f "$MODELS_DIR/$f" ] && scp -q "$MODELS_DIR/$f" \
        "$PI_HOST:$PI_REMOTE_DIR/$f"
    done
    echo "  → Pi backups under rollback-bak-$ts/"
  fi

  echo "✓ installed $tag → $MODELS_DIR"
  echo "  previous files backed up as *.bak.$ts (delete after verifying)"
  echo "  tv-detect picks up head.bin automatically via mtime watch"
}

cmd_list() {
  gh release list --repo "$ANCHOR_REPO" --limit 50 | grep -F "$ANCHOR_PREFIX" || \
    echo "(no anchors yet — create one with: model-anchor.sh create <tag>)"
}

# cmd_auto — automatic off-site anchor of the just-deployed champion, called
# by the nightly trainer after each deploy so rollback never needs a manual
# `create`. Release-only (no git tag = no tag spam), keeps the last
# MODEL_ANCHOR_AUTO_KEEP (default 14). FAIL-SOFT: any gh/auth/keychain hiccup
# (e.g. launchd can't reach the login keychain) just warns and returns 0 —
# it must never break the trainer. The reliable rollback path is the local
# full-bundle archive (rollback-head.sh); this is the off-site DR bonus.
cmd_auto() {
  command -v gh >/dev/null 2>&1 || { echo "auto-anchor: gh missing, skip" >&2; return 0; }
  [ -f "$MODELS_DIR/head.bin" ] || { echo "auto-anchor: no head.bin, skip" >&2; return 0; }
  # A channel-aware head is only restorable WITH its channel-map.
  [ -f "$MODELS_DIR/head.channel-map.json" ] || {
    echo "auto-anchor: no channel-map sidecar — skip (would be un-restorable)" >&2; return 0; }
  local ts
  ts=$(grep -oE '"ts":[ ]*"[0-9T]+"' "$MODELS_DIR/head.channel-map.json" 2>/dev/null \
        | grep -oE '[0-9T]+' | head -1)
  [ -n "$ts" ] || ts=$(date +%Y%m%dT%H%M%S)
  local tag="${ANCHOR_PREFIX}auto-$ts"
  local stage; stage="$(mktemp -d)"; trap "rm -rf '$stage'" RETURN
  local assets=()
  for f in head.bin head.channel-map.json head.calibration.json \
           head.test-set.json head.history.json backbone.onnx; do
    [ -f "$MODELS_DIR/$f" ] && { cp "$MODELS_DIR/$f" "$stage/$f"; assets+=("$stage/$f"); }
  done
  # Capture gh's stderr so a failure is debuggable (the old >/dev/null 2>&1 hid
  # WHY it failed — auth vs network vs tag-exists all looked identical in the
  # nightly log). GH_TOKEN (from secrets.env, exported by tv-train-head.sh) is
  # the auth path under launchd, which can't reach the login keychain.
  # Keep the assignment INSIDE the `if` condition: a bare `gherr="$(…)"` under
  # `set -e` aborts the whole script the instant gh exits non-zero — before the
  # error branch below can ever run (the old silent-exit-1 nightly failure).
  local gherr
  if gherr="$(gh release create "$tag" --repo "$ANCHOR_REPO" --title "Auto champion $ts" \
       --notes "Nightly auto-anchor of the deployed head. Restore: model-anchor.sh install auto-$ts" \
       "${assets[@]}" 2>&1 >/dev/null)"; then
    echo "✓ auto-anchored $tag"
  else
    echo "auto-anchor: gh release create failed — skip. gh said: ${gherr:-<no stderr>}" >&2
    return 0
  fi
  # Prune to the last N auto-anchors (release + tag).
  local keep="${MODEL_ANCHOR_AUTO_KEEP:-14}"
  gh release list --repo "$ANCHOR_REPO" --limit 100 2>/dev/null \
    | grep -oE "${ANCHOR_PREFIX}auto-[0-9T]+" | sort -ru | tail -n +$((keep + 1)) \
    | while read -r old; do gh release delete "$old" --repo "$ANCHOR_REPO" --yes --cleanup-tag >/dev/null 2>&1 || true; done
}

[ $# -ge 1 ] || usage
case "$1" in
  create)  shift; cmd_create  "$@" ;;
  auto)    shift; cmd_auto    "$@" ;;
  install) shift; cmd_install "$@" ;;
  list)    shift; cmd_list    "$@" ;;
  -h|--help) usage ;;
  *) echo "unknown command: $1" >&2; usage ;;
esac
