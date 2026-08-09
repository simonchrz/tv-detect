#!/bin/bash
# Nightly head-retrain. Picks up any new ads_user.json corrections
# the user made during the day, writes a fresh head.bin + the
# uncertain-frames file. Started by ~/Library/LaunchAgents/com.user.tv-train-head.plist.
#
# Safe to run by hand at any time; logs append to LOG.

LOG="$HOME/Library/Logs/tv-train-head.log"
VENV_PY="$HOME/ml/tv-classifier/.venv/bin/python"
SCRIPT="$HOME/src/tv-detect/scripts/train-head.py"
SNAPSHOT_FETCH="$HOME/bin/tv-train-snapshot-fetch.py"
SNAPSHOT_DIR="/tmp/tv-train-snapshot"
# Gateway via Caddy on :8443 since the go-front bypass (self-signed local CA →
# curl -k). The "~40% Caddy CPU on TLS" rationale for the old :8080 path was
# disproven (Caddy 0.05% CPU on a TLS bulk pull; Pi5 ARM HW-AES = TLS ~free).
GATEWAY="https://raspberrypi5lan:8443"
CURL="curl -fsS -k"

exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') ==="

# launchd starts agents with an empty PATH — train-head.py spawns
# ffprobe/ffmpeg subprocesses which would FileNotFoundError without
# this. Includes Homebrew (Apple Silicon default) and ~/.local/bin
# (where tv-detect lives if invoked via the Python path).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

# Markiert die Zeilen, die dieser Lauf in shadow-trend.jsonl schreibt, als
# NIGHTLY. Ein Handlauf setzt die Variable nicht und landet als "hand" in der
# Datei — das Registrierungs-Audit zaehlt nur Nightly-Zeilen als Serien-Nacht.
# Ohne diese Unterscheidung wuerde ein beliebiger Handlauf eine laufende Serie
# um eine Nacht "weiterzaehlen", ohne dass es jemandem auffaellt.
export TVD_LAUF=nightly

# launchd jobs can't reach the login keychain, so `gh` (used by the
# off-site model-anchor at deploy time) can't read its keyring-stored
# token → auto-anchor silently failed every night since 2026-06-02.
# Export GH_TOKEN from secrets.env (chmod 600); gh prefers the env var
# over the keychain. Fail-soft: if the file/key is absent the anchor
# stays a no-op, exactly as before.
SECRETS_ENV="$HOME/.config/tv-stack/secrets.env"
if [ -f "$SECRETS_ENV" ]; then
  GH_TOKEN="$(grep -E '^GH_TOKEN=' "$SECRETS_ENV" | head -1 | cut -d= -f2-)"
  [ -n "$GH_TOKEN" ] && export GH_TOKEN
fi

# Metadata snapshot: fetch all ads_user.json + cutlist + minute-prior
# from the gateway via HTTP into a local mirror at SNAPSHOT_DIR.
# train-head.py then runs with --hls-root pointing at the mirror,
# never touching SMB. Replaces the previous mount-pi-tv.sh dance.
if [ ! -x "$VENV_PY" ]; then
  echo "venv python missing at $VENV_PY"
  exit 1
fi
if [ ! -f "$SNAPSHOT_FETCH" ]; then
  echo "snapshot-fetch helper missing at $SNAPSHOT_FETCH"
  exit 1
fi
"$VENV_PY" "$SNAPSHOT_FETCH" --gateway-url "$GATEWAY" --out "$SNAPSHOT_DIR" || {
  echo "snapshot fetch failed — bailing"
  exit 1
}

# Auto re-detect after binary upgrade: when the tv-detect binary's
# mtime changes (= new build deployed), all existing .txt cutlists
# are stale relative to whatever the new binary would emit. Run a
# batch re-detect BEFORE training so the labels reflect the current
# pipeline. Skipped if mtime unchanged (= no new binary). The marker
# file lives in cache so it doesn't pollute the SMB share.
# DETECT_OFFLOAD=mac on the Pi gateway means re-detection happens via
# the HTTP daemon (~/bin/tv-thumbs-daemon.py). Pi's prewarm-loop
# notices missing/stale cutlist files and writes .detect-requested
# (test-set) + .detect-requested-low (rest) markers; daemon picks
# them up V2-style — high prio first, background backfill once idle.
# Nightly head changes propagate automatically.

# Source-cache hit-rate: train-head defaults --daemon-cache to
# ~/.cache/tv-detect-daemon/source/, populated by tv-thumbs-daemon
# (detect-driven + 30-min prefetch loop). With SOURCE_CACHE_MAX_GB=300
# the full Pi corpus fits locally — daemon-cache is the sole source
# for .ts files since the SMB-fallback was removed (2026-05-02).

# Conditional skip: only retrain when there are >=NEW_REVIEWS_THR
# fresh user reviews since the last deploy. Otherwise we'd burn 12+h
# of detect-drain post-deploy for no IoU gain. Threshold tunable via
# TVH_TRAIN_NEW_REVIEWS_MIN env (default 10; set 1 to always run).
NEW_REVIEWS_THR="${TVH_TRAIN_NEW_REVIEWS_MIN:-10}"
LAST_DEPLOY_TS=$(grep -oE '"ts": "[0-9T]+"' "$SNAPSHOT_DIR/.tvd-models/head.history.json" 2>/dev/null \
                  | tail -1 | grep -oE '[0-9T]+' \
                  | python3 -c "import sys, datetime
ts = sys.stdin.read().strip()
if ts:
    print(int(datetime.datetime.strptime(ts, '%Y%m%dT%H%M%S').timestamp()))
" 2>/dev/null)
if [ -n "$LAST_DEPLOY_TS" ]; then
  # Count ads_user.json files modified since last deploy
  N_NEW=$(find "$SNAPSHOT_DIR" -maxdepth 2 -name "ads_user.json" \
                -newermt "@$LAST_DEPLOY_TS" 2>/dev/null | wc -l | tr -d ' ')
  echo "  conditional check: $N_NEW reviews since last deploy "
  echo "  (threshold $NEW_REVIEWS_THR; override via TVH_TRAIN_NEW_REVIEWS_MIN=1)"
  if [ "$N_NEW" -lt "$NEW_REVIEWS_THR" ]; then
    echo "  → SKIP training (= not enough new data, save ~12h drain cycle)"
    exit 0
  fi
fi

# Training-active marker for the /learning page banner. Posted via
# HTTP (no SMB needed) so the gateway can show "Training läuft seit
# N min" — gateway writes the file on its own filesystem; trap fires
# the DELETE on exit (success OR crash) so a hung script doesn't
# leave the banner stuck on. The mtime tells the banner WHEN
# training started.
$CURL -X POST "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true
trap '$CURL -X DELETE "$GATEWAY/api/internal/training-active" >/dev/null 2>&1 || true' EXIT

# Pre-train drain check: after a head deploy, Smart-V2 invalidates
# ~250+ cutlists which the daemon re-detects via low-prio queue.
# When cron fires mid-drain, train-head sees a shrunk corpus and
# the gate rejects (= 2026-05-17 incident: 173 < 214 floor).
# Wait up to 30 min for the low-prio queue to drain below 30 before
# starting — daemon makes ~50/h so 30min is enough headroom for
# 25-30 pending markers. Cap at 30min so cron eventually runs even
# if drain stalls (= per-channel detect-spawn loop tends to recover
# but no guarantees).
WAIT_DEADLINE=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "$WAIT_DEADLINE" ]; do
  N=$($CURL "$GATEWAY/api/internal/detect-pending-low" 2>/dev/null \
       | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("pending",[])))' \
       2>/dev/null || echo 999)
  if [ "$N" -lt 30 ]; then
    echo "pre-train drain check: low-prio queue=$N (<30 threshold) — proceeding"
    break
  fi
  echo "pre-train drain check: low-prio queue=$N, waiting up to $((WAIT_DEADLINE - $(date +%s)))s..."
  sleep 60
done

TRAIN_START_TS=$(date +%s)

# Local output dir — head.bin + sidecars + archive/ all land here.
# Old SMB-mounted path was ~/mnt/pi-tv/hls/.tvd-models/ which broke
# when SMB unmounted; we now write locally and POST to Pi at end.
#
# ⚠️ NICHT nach /tmp. Dort lag es bis 2026-08-05, und damit war
# `archive/` — die EINZIGE Quelle vollständiger Bündel (head + Kanal-Karte
# + Kalibrierung) — nach jedem Mac-Neustart leer. Gemessen an dem Tag: das
# Pi-Archiv hielt 177 Köpfe zurück bis zum 25. April, aber ausschließlich
# `.bin` ohne Sidecars; lokal existierten noch DREI Zeitstempel. Die reale
# Rollback-Tiefe von `rollback-head.sh` war damit „seit dem letzten
# Neustart" statt eines halben Jahres — und das Werkzeug verweigert (zu
# Recht) einen Kopf ohne passende Kanal-Karte, weil falsch ausgerichtete
# Kanal-Spalten stille Fehlinferenz ergeben.
#
# Das Off-Site-Ankern via model-anchor.sh (GitHub-Releases) war intakt und
# hat den Fall gerettet — aber es ist der DR-Weg, nicht der Alltagsweg.
# Beide sollen tragen. ~/.cache liegt neben den übrigen Modell-Caches
# (tvd-features, tvd-train-archive) und überlebt Neustarts.
TRAIN_OUT="${TRAIN_OUT:-$HOME/.cache/tv-train-head-out}"
mkdir -p "$TRAIN_OUT"
LOCAL_BACKBONE="$HOME/.cache/tv-detect-daemon/backbone.onnx"

# Prefetch the CURRENTLY-DEPLOYED head + sidecars from the gateway so the
# run compares against / cleans with the REAL champion, not a stale local
# /tmp copy:
#   head.bin + head.channel-map.json — the deploy gate's head-to-head
#     re-scores this exact deployed head on the new test set (apples-to-
#     apples), and the label-hygiene teacher uses it too. Without seeding,
#     a wiped/stale $TRAIN_OUT (e.g. after a Mac reboot — /tmp is volatile)
#     leaves an unloadable head → BOTH silently fall back: the gate drops to
#     the lenient floor path (deploys "comparison invalidated" without ever
#     comparing to the champion = effectively a no-op) and the teacher is
#     skipped. 2026-06-22: this is exactly what bit the post-reboot run.
#   head.gate.bin — the train-only honest champion the gate prefers (404
#     until the first post-fix deploy uploads it → falls back to head.bin).
#   head.history.json / head.test-set.json — append to the existing trail +
#     fire the test-composition-changed reason against the real prev set.
# 404 just means none-yet-on-Pi → start fresh, fine.
for f in head.bin head.gate.bin head.channel-map.json \
         head.history.json head.test-set.json head.calibration.json \
         head.minute-prior.json; do
  $CURL -o "$TRAIN_OUT/$f" \
      "$GATEWAY/api/internal/detect-models/$f" || rm -f "$TRAIN_OUT/$f"
done

"$VENV_PY" "$SCRIPT" \
    --workers 4 \
    --backbone "$LOCAL_BACKBONE" \
    --logo-dir "$HOME/.cache/tv-detect-daemon/logos" \
    --output "$TRAIN_OUT/head.bin" \
    --hls-root "$SNAPSHOT_DIR" \
    --surface-uncertain 6 \
    --with-logo \
    --with-audio \
    --with-minute-prior \
    --with-self-training \
    --write-pseudo-labels \
    --train-archive "$HOME/.cache/tvd-train-archive" \
    --head-arch mlp32-channel-whisper-temporal-mp-wm \
    --shadow-eval \
    --ablate-minute-prior \
    --prod-seeds 3 \
    ${TVH_TRAIN_EXTRA_ARGS:-}
# 2026-08-06 (spaeter am Tag): der Temporal-Block waechst von 2 auf 3
# Spalten — dp, dn und NEU das Unruhe-Niveau (1s-Delta ueber 31 s
# gemittelt). Der Architektur-NAME bleibt gleich, der Header traegt die
# Zahl (n_temporal=3), das Modell beschreibt sich also selbst. Grund:
# Rang-AUC gegen die Labels ueber den Golden-Satz 0.637 (1s) gegen 0.880
# (31 s), Korrelation mit der Kopf-Ausgabe nur 0.625, und dort wo der Kopf
# unsicher ist (0.2<p<0.8) trennt es noch mit 0.784 — also neue
# Information, nicht bloss geglaettete alte.
#
# 31 s und nicht breiter, obwohl 61 s (0.932) und 181 s (0.977) hoeher
# messen: der Kopf laeuft chunkweise, ein Fenster sieht ueber die
# Chunk-Grenze nicht hinaus. Bei 4 Chunks (DETECT_PARALLEL=3) sind das bei
# 31 s ~4 % der Frames, bei 61 s 8 %, bei 181 s ueber 20 % — und ein
# Train/Serve-Bruch ist teurer als der Messgewinn.
#
# 2026-08-06: --head-arch auf -mp-wm (MLP5 v5) umgestellt — eine
# Indikatorspalte "Whisper vorhanden ja/nein". Grund: die
# Whisper-WAHRSCHEINLICHKEITS-Spalte faellt bei fehlenden Daten auf
# neutrale 0.5, womit "kein Ton-Signal" und "Ton sagt 50/50" dieselbe
# Eingabe waren — bei 300 von 557 Archiv-Aufnahmen. Der Kopf gewichtet
# die Spalte stark (Norm 15.5 gegen Median 1.49 der Backbone-Spalten),
# Aufnahmen ohne Whisper lagen 0.049 IoU tiefer (p=0.034), und dieselbe
# Aufnahme mit/ohne Feed schwankt bis 0.6. Nachholen geht nicht: 296 der
# 300 haben keine Quelle mehr und HLS-VOD existiert fuer keine.
#
# ⚠️ REIHENFOLGE: erst das Mac-Binary (~/.local/bin/tv-detect, kennt
# MLP5 seit a2deb9b), DANN diese Zeile. Umgekehrt laedt der Daemon einen
# MLP5-Kopf nicht und die Detection steht. Geprueft: das neue Binary
# liefert auf dem deployten MLP4-Kopf 2272/2272 NN-Werte identisch.
#
# Erste Nacht wie bei jedem Architekturwechsel: input_dim 1299 -> 1300,
# also skippt das Head-to-Head (die Meldung dazu war bis heute STILL und
# ist jetzt im Log). Es schuetzen der historische IoU-Boden und der
# Golden-Boden — der steht seit 20260806 bei 0.909 unter hsmm, mit
# --golden-floor 0.010 muss der Kandidat also >= 0.899 erreichen.
# Zurueck: diese Zeile auf ...-mp, sonst nichts.
#
# 2026-07-22: --head-arch auf -mp (MLP4 v4) migriert nach 3 positiven
# Shadow-Nächten (+0.021/+0.021/+0.016). Erste Nacht: dim-change →
# head-to-head skippt, Floor-Gate greift; head.minute-prior.json wird
# ab dann mit dem Bundle deployt (Go nn.go v4 + Daemon --start-ts in
# tandem deployt).
# --shadow-eval WIEDER AN seit 2026-07-15 für die minute-prior-Probe
# (cwt-Produktions-Replikat vs +P(ad|minute)-Spalte; Ziel: die 77%
# intra-show-FP-Fehlerklasse). Entscheidung nach einigen Nächten wie
# bei temporal: konsistent positiv → Migration, Rauschen → wieder raus.
# (Historie: erste Shadow-Serie 07-07..07-12 entschied MLP-64=Rauschen
# und +temporal=Produktion / MLP3 v3.)
rc=$?

# Boundary head (BNDR): a SECOND head that scores "is this frame an ad
# boundary?" — targets the dominant missed_bumper/boundary-drift failure
# mode the per-frame classifier can't fix (block position right, edge off
# by seconds → IoU drag). Reuses the feature cache train-head.py just
# populated (no re-extract) + ad-start/end frames from ads_user.json.
# Non-fatal: a failure here never blocks the main head deploy. Only after
# a successful main run (rc==0) so the feature cache is fresh.
if [ "$rc" -eq 0 ]; then
  echo "training boundary head…"
  "$VENV_PY" "$HOME/src/tv-detect/scripts/train-boundary-head.py" \
      --hls-root "$SNAPSHOT_DIR" \
      --feature-cache "$HOME/.cache/tvd-features" \
      --output "$TRAIN_OUT/boundary_head.bin" 2>&1 | sed 's/^/  /' \
    || echo "  boundary-head train FAILED (non-fatal, main head unaffected)"
fi

# Bundle head.bin + sidecars + archive/ into a tar.gz and POST to
# the gateway. Pi extracts atomically into /mnt/tv/hls/.tvd-models/
# (head.bin written LAST so daemons never see a half-deploy).
# Replaces the rsync over ssh — keeps everything on the unified
# Mac→HTTP→Pi path the rest of the stack uses.
if [ "$rc" -eq 0 ]; then
  echo "bundling + uploading head to gateway…"
  BUNDLE="/tmp/tv-train-head-bundle.tar.gz"
  ( cd "$TRAIN_OUT" && tar czf "$BUNDLE" \
      head.bin head.gate.bin boundary_head.bin boundary_head.history.json \
      head.*.json head.*.txt archive 2>/dev/null )
  size_mb=$(du -m "$BUNDLE" | cut -f1)
  echo "  bundle: ${size_mb} MB"
  resp=$($CURL -X POST --data-binary "@$BUNDLE" \
      -H "Content-Type: application/gzip" \
      "$GATEWAY/api/internal/head-bundle")
  echo "  upload response: $resp"
  if [ "$?" -ne 0 ]; then
      echo "  WARN: upload failed — head NOT deployed"
      rc=2
  fi
  rm -f "$BUNDLE"
  # Auto-anchor the just-deployed champion off-site so rollback never needs a
  # manual step. Fail-soft (gh/keychain may be unreachable under launchd) — the
  # local full-bundle archive + rollback-head.sh is the reliable path; this is
  # the off-site DR bonus. Only on a real deploy (rc==0, head changed).
  if [ "$rc" -eq 0 ]; then
    # Root-caused 2026-06-19: the nightly auto-anchor silently exit-1'd because
    # model-anchor.sh ran `gh release create` without --repo and launchd's cwd
    # is / (no git remote) → gh failed → set -e killed it before logging. Fixed
    # by pinning --repo "$ANCHOR_REPO" in model-anchor.sh. (The earlier
    # diagnostic echo here leaked GH_TOKEN into the log — removed.)
    echo "auto-anchor: invoking (rc=$rc)"
    TVD_MODELS_DIR="$TRAIN_OUT" \
      "$HOME/src/tv-detect/scripts/model-anchor.sh" auto 2>&1 | sed 's/^/  /'
    echo "auto-anchor: model-anchor.sh auto done (pipe-status=${PIPESTATUS[0]})"
  fi
elif [ "$rc" -eq 3 ]; then
  # Python rejected the candidate (regression vs last-deployed). Skip
  # bundle+upload — Pi keeps its current head. Without this gate, the
  # local /tmp/tv-train-head-out/head.bin (which can drift from Pi
  # after a manual rollback) would be bundled + pushed, silently
  # overwriting Pi with whatever stale candidate happened to be in
  # the local dir (14.05 incident: rejected 0.87 candidate but
  # uploaded the morning's 0.83 regression that was left in /tmp).
  echo "REJECTED by python — skipping bundle+upload (Pi keeps current head)"
  rc=0  # not a script-level failure, just a no-op deploy
fi

# Persist the wall-time of this run to the history file so the
# /learning banner can derive an ETA for future training runs as
# median(last-N-durations) - elapsed. JSON-lines format = append-
# only, gateway parses cheaply line-by-line. POSTed via HTTP =
# gateway appends server-side, no SMB write.
TRAIN_DUR=$(( $(date +%s) - TRAIN_START_TS ))
TRAIN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
$CURL -X POST -H "Content-Type: application/json" \
    -d "{\"ts\":\"$TRAIN_TS\",\"dur_s\":$TRAIN_DUR,\"rc\":$rc}" \
    "$GATEWAY/api/internal/training-duration" >/dev/null 2>&1 || true

# Push the LAST history entry — deploy OR reject — to the Pi as a standalone
# attempt record. The head-bundle upload above is gated to deploys only (it
# must never overwrite the Pi head with a rejected candidate), so a rejected
# run never reaches the Pi's head.history.json and would be invisible to
# /api/learning/last-training + HA. .last-attempt.json closes that gap without
# touching the deployed head. Python writes this run's entry as the last array
# element of the local head.history.json in BOTH the deploy and reject paths.
if [ -f "$TRAIN_OUT/head.history.json" ]; then
  "$VENV_PY" -c "import json,sys; h=json.load(open('$TRAIN_OUT/head.history.json')); sys.stdout.write(json.dumps(h[-1]) if h else '')" 2>/dev/null \
    | $CURL -X POST -H "Content-Type: application/json" --data-binary @- \
        "$GATEWAY/api/internal/training-attempt" >/dev/null 2>&1 || true
fi

# Cache hygiene: every recording experiment (different --with-* flags)
# leaves orphan .npy files in tvd-features/ — bumped cache key, old
# files unused but still on disk. Each is ~1-3 MB; over months of
# experiments the directory grows several GB. Prune anything that
# hasn't been read in 60 days. The current production flag set
# (--with-logo) is touched every nightly run so its caches stay
# alive; only stale experiment caches get reaped.
deleted=$(find "$HOME/.cache/tvd-features" -name "*.npy" -atime +60 -print -delete 2>/dev/null | wc -l)
if [ "$deleted" -gt 0 ]; then
  echo "cache cleanup: pruned $deleted stale .npy file(s)"
fi
# Per-show IoU snapshot. The new head.bin invalidates all auto-cutlists
# so the post-train daemon bulk-redetect must drain BEFORE the snapshot
# is meaningful (stale ads.json → IoU=0). Earlier this script tried to
# background a wait-loop here, but launchd reaps subshells when the
# parent train-head.sh exits. Instead we drop a marker file with the
# desired snapshot timestamp; the long-running tv-thumbs-daemon picks
# it up on its next poll cycle when detect queue == 0, fires the
# snapshot endpoint, deletes the marker. Stateless + survives daemon
# restarts.
if [ "$rc" -eq 0 ]; then
  marker="$HOME/.cache/tv-detect-daemon/snapshot-requested"
  mkdir -p "$(dirname "$marker")"
  date '+%Y%m%dT%H%M%S' > "$marker"
  echo "snapshot marker written → $marker" >> "$LOG"
fi

# ── Label-Audit (2026-07-27) ────────────────────────────────────────
# Recomputes the DEPLOYED head over the archive's own per-second features
# and flags every recording whose labels contradict its own signal — the
# head is confidently "ad" for 90s+ where the labels say show, or the
# reverse. No decode, so it covers the whole corpus in minutes.
#
# Runs AFTER training, not before. The archive .npz are rewritten by the run
# itself, so an audit placed ahead of it reports YESTERDAY's labels: on
# 2026-07-28 it listed six recordings whose labels the same run then fixed
# (16 -> 9 -> 4 once re-run by hand). A stale report is worse than none — it
# invites acting on a problem that is already gone.
#
# Report-only ON PURPOSE. The first run (07-27) found 16 of 486, of which
# 14 sat in the TRAIN split; three carried human labels, where a signal
# test disagreeing with a person is not authority to act. Acting is a
# judgement call (quarantine / redetect / leave alone) and belongs to
# whoever reads this log, not to a cron job.
#
# Why it runs every night rather than once: the golden-eval set decayed
# from 60 usable members to 23 without anyone noticing, precisely because
# nothing looked at it periodically. A one-off cleanup is an event; this
# makes it a property.
echo "=== label audit (report only) ==="
"$VENV_PY" "$HOME/src/tv-detect/scripts/corpus-label-audit.py" 2>&1 \
  | tail -30 || echo "label audit failed (non-fatal)"
echo "=== label audit end ==="

echo "train-head exit=$rc"
exit $rc
