# Notes for future Claude sessions

This file is project context for Claude Code. Read it before
suggesting changes.

## What this project is

A Go binary intended to replace the comskip-based ad-detection stage
in two upstream Python orchestrators:

- `~/bin/tv-live-comskip.py` — Mac launchd agent that does sliding-
  window scans on live HLS streams.
- `~/src/tvheadend/hls-gateway/service.py` — Pi container that runs
  comskip on completed DVR recordings.

Both currently shell out to `comskip --ini ... --output ... <src>` and
parse the `<basename>.txt` cutlist (frame-pair format). The intent for
tv-detect is to be a behaviourally identical drop-in: same cutlist
output, same Python parsers (`_rec_parse_comskip` in service.py,
`parse_comskip` in tv-live-comskip), no Python changes required.

## Architecture decisions worth remembering

- **No CGO.** Cross-compile for `darwin/arm64` and `linux/arm64` from
  any host. Anything that would need a C library (image processing,
  decoding) goes through ffmpeg as a subprocess instead. The cost is
  the IPC; the win is reproducible builds and a single static binary.

- **ffmpeg is the only runtime dep.** `ffprobe` for metadata, `ffmpeg`
  for raw rgb24 piping (video) and `silencedetect` filter (audio).
  Both already exist on every box this targets.

- **Chunk-parallelism, not frame-parallelism.** Each worker runs its
  OWN ffmpeg subprocess on a `-ss X -t Y` time slice, processes its
  frames sequentially, and the orchestrator merges per-chunk results.
  Splitting per-frame work across goroutines was rejected because the
  inter-frame state (scene-cut needs prev frame, blackframe needs
  consecutive runs) doesn't parallelise cleanly.

- **Silence runs as a single global subprocess**, not chunked. It's
  audio-only (low CPU), already runs concurrent with the video
  pipeline, and chunking it complicates merge with no benefit.

- **Scene-cut at chunk boundaries is suppressed**, because the first
  frame of a non-origin chunk has no real "previous frame" and would
  fire a spurious cut.

- **Blackframe runs that span chunk boundaries are coalesced** at
  merge time — a black flash at the seam shows up as two truncated
  events that we re-stitch.

- **Logo template format is comskip's** (the SaveLogoMaskData patch
  output: `logoMinX/MaxX/MinY/MaxY/picWidth/picHeight` header + ASCII
  edge mask). The active per-channel cache lives at
  `~/mnt/pi-tv/hls/.tvd-logos/<slug>.logo.txt` (channel-keyed,
  reused across all recordings of that channel). The legacy
  `~/mnt/pi-tv/hls/.logos/` path is from the comskip era — only kept
  as a fallback hint, not authoritative.

- **Logos are trained inside tv-detect** (`internal/logotrain` +
  `--auto-train N` in main.go) rather than parsing comskip's output.
  The trainer samples 1 frame/sec for N minutes, finds pixels that
  are edges in ≥ persistence (default 0.85) of frames, and bbox-caps
  to 5000 px² (`tr.ComputeAdaptive(5000, persist)`) by sweeping
  persistence upward if needed. This cap is the only safety net
  against the failure mode below.

- **NN evidence (Phase 7+)** is a frozen MobileNetV2 backbone +
  trainable Linear head, both at
  `~/mnt/pi-tv/hls/.tvd-models/{backbone.onnx,head.bin}`. The head
  hot-reloads on mtime change. Per-frame blend with logo controlled
  by `--nn-weight` (default 0.3) and `--nn-gate` (default 0.3 — only
  let NN vote when |conf - 0.5| ≥ gate, keeps logo authoritative
  when NN is unsure). Training is `scripts/train-head.py`, Python
  venv at `~/ml/tv-classifier/.venv` (NOT in this repo).

  **Head format is auto-detected by raw file size** in
  `internal/signals/nn.go`'s `reloadHead()`. Four shapes possible
  (each is float32-LE weights packed back-to-back, then 1 float32 bias):
    - 5124 B = LEGACY: 1280 backbone weights only.
    - 5128 B = +LOGO: backbone (1280) + logo conf (1).
    - 5148 B = +CHAN: backbone (1280) + channel one-hot (6).
    - 5152 B = +LOGO+CHAN: backbone + logo + channel = 1287 weights.
  Channel slot order is hard-coded as `nnChannels = [kabel-eins,
  prosieben, rtl, sat-1, sixx, vox]` and the same list lives in
  `scripts/train-head.py`. APPEND-ONLY: re-ordering or inserting
  silently mis-maps every previously-trained head's channel weights.
  As of this session, default deploy is +LOGO (5128 B).

- **Self-improving training loop** runs nightly via launchd:
  `~/Library/LaunchAgents/com.user.tv-train-head.plist` triggers
  `~/bin/tv-train-head.sh` at 03:30 → invokes train-head.py with
  `--with-logo --with-minute-prior --with-self-training
  --write-pseudo-labels --surface-uncertain 6`. Each run:
    1. Holds out 20 % of recordings (deterministic uuid hash) for
       per-show test metrics: per-frame Acc, F1, Block-IoU.
    2. Smart-merges `ads.json` (auto) + `ads_user.json` (user) per
       recording — user blocks override overlapping autos, explicit
       `deleted` entries kill auto-only false positives. For
       recordings with `pseudo_labels.json` AND no user file, the
       per-frame pseudo-labels are loaded as training data with a
       `frame_mask` filter (frames without an opinion are excluded,
       NOT defaulted to show).
    3. Drops recordings with merged ad-rate > 50 % (`--max-ad-rate`)
       — catches broken-template label artefacts.
    4. Label-hygiene pass: drops frames where the existing head
       (used as "teacher") strongly disagrees with the label,
       capped at 30 % drop-rate per recording so a busted teacher
       can't nuke its own future training data. Skipped for
       pseudo-labelled recordings (the conf+prior filter at write
       time already gates them).
    5. Fits LogisticRegression with `sample_weight`:
       - 2× user-confirmed recordings (`--user-weight`, default 2.0)
       - 1× auto-only recordings
       - 0.3× pseudo-labelled frames (`--pseudo-weight`)
       - 1.2× confirmed_show frames (active-learning ✓ Geprüft)
       - 1.5× skip-press frames (Werbung-Skip-Button → forced label=1)
       - Age decay: linear ramp 1.0 → 0.5 over [0, 90] days, 0.5 → 0
         over [90, 180], skipped beyond 180.
    6. **Champion-challenger** with two waivers:
       - If test-set composition changed (different rec count) →
         deploy + reset baseline.
       - If feature dim changed (e.g. `--with-channel` toggled) →
         deploy + reset baseline. Reads `n_features` from history;
         falls back to deployed head.bin file size when missing.
       - Otherwise: REJECT if IoU drops > 5 pp OR Acc > 3 pp vs
         last deployed.
    7. Writes `head.bin`, `archive/head.<ts>.bin`, `head.history.json`
       (now includes `n_features`), `head.uncertain.txt` (top-N
       split half between high-uncertainty + half between
       high-divergence-from-minute-prior frames; tagged `unc`/`div`/
       `both`), `head.confusion.txt` (with `--emit-confusion`).

- **Pseudo-label self-training (Phase A + B)**:
  - **Phase A** (`--with-self-training`): on the held-out test set,
    filter frames where the head is highly confident (p > 0.97 or
    p < 0.03) AND the wall-clock minute-prior agrees (same side of
    0.5). Report accuracy of those candidates vs truth. Verdict
    threshold ≥ 95 % = SAFE for Phase B.
  - **Phase B** (`--write-pseudo-labels`): walk every recording
    without `ads_user.json`, predict, apply same conf+prior filter,
    write `_rec_*/pseudo_labels.json`: `{frames: [i,...], labels:
    [0|1,...], n_total, threshold}`. Auto-deleted on next run if
    `ads_user.json` appears (user reviewed → real labels supersede).
  - **Bootstrap**: for an unreviewed recording with neither
    `ads.json` nor `pseudo_labels.json`, the loader includes it in
    `per_rec` with empty labels + all-False mask so Phase B can
    predict on its features. Zero training contribution this run,
    seeds the next.

- **Wall-clock minute-of-hour prior** (`--with-minute-prior`):
  empirical `P(ad | minute_of_hour)` per channel from labelled data,
  Bayesian-smoothed (5 virtual frames at channel mean per bucket).
  Cached as `<hls-root>/.minute_prior_by_channel.json`. Used for
  active-learning surfacing (high-confidence head predictions that
  disagree strongly with the prior become divergence-source targets,
  separate from uncertainty-source) AND as the sanity gate for
  pseudo-labels in Phase A/B. Privates (RTL/SAT.1/sixx/kabel-eins)
  show sharp peaks at fixed minutes; ARD/ZDF flat (no ad slots).

- **Co-training (analysis-only, `--co-train`)**: trains two extra
  heads — `head_logo` (backbone+logo+chan) and `head_audio`
  (rms+chan or yamnet+rms+chan if `--with-yamnet`) — on the same
  data, reports agreement. Phase 2 (use agreement on UNLABELLED
  frames as pseudo-labels) NOT wired in: the audio head is too
  weak (~78 % vs main head's ~89 %) to be a useful co-teacher. The
  disagreement frames could be future active-learning targets but
  are not currently surfaced.

- **Post-detection precision steps** in `internal/blocks/blocks.go`,
  applied after `refineBoundaryVoting` clusters all signals:
  - **`IFrameSnapS`** (default 5 s): snap to nearest encoder
    keyframe. Real ad inserts always align with IDR frames.
  - **`SceneCutSnapS`** (default 0 s, was briefly 1.5): snap to
    nearest hard scene cut (luma-histogram Bhattacharyya > 0.40).
    Off by default after empirical regression: snap-overrides of
    a correctly I-frame-aligned boundary onto a within-show scene
    cut hurts more than it helps on broadcasters that DON'T align
    cuts to keyframes consistently.
  - **`LogoCrossRefineS`** (default 2 s): snap to the precise
    sub-frame timestamp where logoConf crosses LogoThreshold.
    40 ms precision, free (re-uses the per-frame logo signal).

- **`scripts/model-anchor.sh`**: off-Pi semantic snapshots of the
  trained model. `create <name>` bundles `head.bin`, `backbone.onnx`,
  `head.history.json` from `~/mnt/pi-tv/hls/.tvd-models/` (auto-
  detected; falls back to scp-from-Pi if SMB is dead) into a
  GitHub release tagged `model-anchor-<name>`. Notes auto-include
  Block-IoU, Acc, n recs from the latest history entry plus a
  free-text `--notes` block. `install <name>` restores any anchor
  in one command (existing files saved as `*.bak.<ts>`); tv-detect
  picks up `head.bin` via the existing mtime watch, no restart.

  Use anchors at semantic milestones (letterbox fix, IoU
  thresholds, dataset-size jumps), NOT for every nightly retrain
  — the rolling `archive/head.<unix-ts>.bin` already covers that.
  First anchor: `letterbox-fix-2026-04-27` (post-cropdetect, IoU
  0.73 / Acc 96 %).

- **`scripts/bumper-detect.py`**: ffmpeg `blackdetect ∩
  silencedetect` helper. Outputs `_rec_*/bumpers.json` with
  proximity-matched (gap ≤ 1.5 s, NOT strict intersection — privates
  let silence start ~0.5 s before the black cut) ad-bumper
  timestamps. Used by `--with-bumpers` to boost sample weight on
  the ±2 s frame window around any ads.json boundary that has a
  bumper nearby. Empirical hit rate is low on private TV (Music
  beds in bumpers defeat silencedetect), so NOT in the nightly.

- **`ads_user.json` supports two on-disk formats** for backward
  compat (the gateway's `_read_user_ads` in service.py and
  train-head.py both accept either):
    - LEGACY list: `[[s,e], ...]` — each editor save before the
      smart-merge UI rewrite produced this.
    - DICT: `{"ads": [...], "deleted": [...]}` — current format.
      `deleted` blocks are auto-detected blocks the user explicitly
      removed; smart-merge suppresses any auto block that overlaps
      one. Without `deleted`, the smart-merge can't distinguish "user
      kept this auto unchanged" from "user deleted this auto", so the
      delete-from-UI button is the only way to surface a false
      positive that survives re-scans.

- **`/recording/<uuid>/ads`** returns the smart-merged view as
  `ads`, plus `auto`/`user`/`deleted` separately so the player UI
  can classify a click-delete (was it an auto block, hence record
  to `deleted`?) and `duration_s` so the scrub bar can render ad
  blocks before video metadata loads (~6 s saved on first paint).
  Also `uncertain: [{t, p}]` from the active-learning file —
  rendered as orange chevrons on the scrub bar that seek-and-play.

- **`LIVE_ADS_OFFLOAD=mac` on the Pi container** disables Pi-side
  scanning. The Mac runs `~/bin/tv-comskip.sh` / `tv-live-comskip.py`
  which shells out to `tv-detect`. Pi only does rec-thumbs. Means:
  `head.bin` and logo templates must live on the SMB share so the
  Mac can read them.

## Where the bodies are buried

- **Cached logo templates can silently break detection on a whole
  channel.** Two failure modes seen in the wild:
  1. *Bbox covers the whole frame* (e.g. cached `rtl.logo.txt`
     trained pre-cap or via a tool without the 5000 px² limit:
     bbox 129×576 = 74304 px², matched the letterbox edges → "logo
     present" on essentially every frame → no ad blocks ever
     detected, or one block in the wrong place).
  2. *Bbox in the wrong frame corner* (cached `sat-1.logo.txt` had
     a bottom-left bbox; SAT.1's actual logo is top-right →
     per-frame conf stuck in the 0.07–0.26 grayscale band, ~50 %
     present_rate throughout, "everything looks like ad").

  Diagnosis recipe:
  - `awk` the bbox dimensions out of the .logo.txt header — area >>
    5000 px² is automatically suspect.
  - Run `--emit-logo-csv` over a known-good recording, time-bucket
    the per-frame confidence: a healthy template shows clean
    bimodal show-vs-ad transitions (avg 0.8+ during show, < 0.2
    during ads); a broken template shows uniform mid-band noise.

  Fix: delete the cached template and rerun with `--auto-train 5`
  on a recording where minutes 0:30–5:30 are unambiguously show
  (no opening promo, no full-screen overlay magazine). The 5000
  px² adaptive shrink in `autoTrainLogo` will keep the new bbox
  honest.

- **Letterbox compensation via `--logo-y-offset N`.** Channels that
  air a 16:9 movie in their 4:3 broadcast container (RTL on Spielfilm
  blocks is the canonical case) put the logo at the same screen
  position relative to the visible 16:9 area — which means it sits
  N pixels DOWN in the 720x576 frame compared to non-letterboxed
  programmes. Without compensation the comskip-trained template lands
  in the top black bar, logoConf collapses to 0 throughout, and the
  block detector reverts to NN-only with massive false-positive rate.
  Symptom: a 2 h film produces 15+ ad blocks instead of the actual 5.

  The Mac daemon (`~/bin/tv-thumbs-daemon.py`) computes the offset
  per recording: `ffmpeg cropdetect` on a 5 s sample at the 60 s mark
  → take the last `crop=...:Y` y-value → `offset = max(0, Y - 20)`.
  The constant 20 is RTL-empirical: cropdetect finds the bar
  boundary, but the channel logo overhangs ~20 px into what should
  be visible content, so the bar y is too deep by that amount. Same
  cropdetect pass runs in `train-head.py` so cached training features
  match what inference produces (cache key bumped `-l1` → `-l2` to
  invalidate pre-letterbox-aware features). The 20-px overhang may
  not generalise to channels that don't air movies; revisit if
  another channel ever ships letterboxed content with a different
  logo position.

- **Per-frame logo signal works since Phase 5+.** The edge detector
  (Sobel 3×3) and template parser (comskip-compatible header +
  `-/|/+/space` glyphs) match correctly. The confidence is the
  fraction of expected-edge positions that fire in a frame. Block
  formation (`internal/blocks`) consumes it as the primary signal
  with optional NN blending.

- **`-ss X -t Y` seek is approximate.** ffmpeg seeks to the nearest
  I-frame ≤ X. On a 30-min recording with 8 chunks we observe
  ~15-20 extra frames total across all chunks — minor over-counting
  vs single-pipeline. Acceptable for our cutlist tolerance (±2
  frames per boundary).

- **`--max-ad-gap` (default 30 s)** post-merges adjacent ad blocks
  whose post-refinement gap is shorter than this. Catches the case
  where MinShowSegmentS is satisfied by a brief station-promo slate
  (e.g. VOX "Heute 20:15" with logo *centered* — corner detector
  says absent, NN backbone briefly says "show-like" because of big
  branded text) and the state machine prematurely closes one ad
  block to start a fresh one a few seconds later. Set to 0 to
  disable.

- **Per-channel logo smoothing** is configured in `tv-comskip.sh`
  (`LOGO_SMOOTH_PER_SLUG`) and the batch-redetect helper. Default
  `--logo-smooth 0` (no smoothing) — but channels with persistent
  lower-third graphics (ProSieben magazin formats: Galileo, Die
  Simpsons recap bumper) need ~5s smoothing or the state machine
  never satisfies its consecutive-present hysteresis. Empirical
  from 2026-04-25: `prosieben=5` won big (Galileo IoU 0.04 → 0.93
  after subsequent labelling), all other channels regressed when
  smoothed (RTL Wetzel/Unter uns lost up to 0.38 IoU). MIRROR THE
  MAP in both spawn paths if you change it.

- **`--logo-smooth` and `--nn-smooth`** (rolling-mean windows on
  the respective per-frame confidence streams, default 5 s and 10 s)
  exist for the same reason: backbone + template both produce
  sub-second flicker that the state machine's consecutive-present
  hysteresis can't filter — smoothing upstream is much cleaner than
  bumping `MinShowSegmentS` after the fact.

- **Frame timestamps are `index/fps`**, not container PTS. This
  matches comskip's convention. ffmpeg's `blackdetect` uses container
  PTS, so its event times sit ~0.5 s later than ours on MPEG-TS
  inputs (GOP-reorder offset). DON'T "fix" this — the downstream
  Python code expects the comskip convention.

- **Per-frame Acc and Block-IoU diverge**, sometimes spectacularly.
  Acc is dominated by the easy 90 % of frames (deep-show or
  deep-ad). IoU is dominated by the boundary 5–10 % of frames. A
  recording where the model gets boundaries off by 60 s on each
  end of one ad block: per-frame Acc moves maybe 4 pp, Block-IoU
  drops 0.50+. Always trust IoU as the primary metric; Acc is
  optimistic. Champion-Challenger thresholds reflect this:
  `--rollback-iou-drop=0.05` (5 pp) is tighter than `--rollback-
  acc-drop=0.03` (3 pp).

- **`--with-channel` hurts minority channels with sparse training
  data.** Empirical (2026-04-26): switching from no-channel
  (1281 dims) to channel-one-hot (1287 dims) on a 26-rec dataset
  with 1 hundkatzemaus (VOX) recording crashed VOX IoU 0.51 → 0.14.
  Channel-one-hot ISOLATES per-channel bias instead of sharing
  patterns across the pool — useful when each channel has 5+ recs,
  catastrophic when some have 0–1. Currently OFF in the nightly.
  Re-enable when every active channel has ≥ 3 train recs.

- **Catastrophic interference is real with N≈25 train recordings.**
  Adding 2–3 fresh user-confirmed recordings can shift the
  LogisticRegression's global decision boundary enough to drop
  per-show IoU on UNRELATED test shows by 0.30+. Not a bug, just
  the curse of small datasets with a linear model. Champion-
  Challenger catches it. Mitigation paths: more data, channel-
  one-hot (when sparsity allows), or per-show fingerprint auto-
  confirm (gateway-side) to grow the train set passively.

- **Pseudo-label files (`pseudo_labels.json`) are stale-prone.**
  Written by `--write-pseudo-labels` for any unreviewed recording;
  cleaned up automatically on the next training run when a real
  `ads_user.json` appears. Between runs, an existing pseudo file
  is shadowed by user labels at LOAD time (smart-merge prefers
  user). If you suspect orphans, `find _rec_*/ -name
  pseudo_labels.json -newer X | wc` and rely on the next nightly,
  or `rm` them — no data loss either way (they're regenerated).

- **Rebatch ↔ training has a feedback risk.** Rebuilding `tv-detect`
  changes its mtime → `tv-train-head.sh` runs `tv-detect-rebatch.sh`
  → ALL `ads.json` files get regenerated with the new binary. Smart-
  merge with user labels then changes which auto blocks survive in
  training, even when the user's edits are unchanged. Recovery: the
  old `ads_user.json` is still authoritative, retraining stabilises
  within 1–2 runs. NEVER delete `ads_user.json` to "force a clean
  start" — that throws away the only thing you can't reproduce.

## Patterns worth following

- **Each detector exposes `Push(idx, ...)` + `Events()`/`Cuts()`**
  rather than callbacks. Keeps the orchestrator code linear and
  testable.

- **Tests under `internal/signals/` use synthetic frames** (all-black,
  all-white, gradient) — no test fixtures, no golden files. Faster
  to maintain.

- **Use `quiet` flag for production paths** so the Python orchestrator
  gets clean stderr (only errors). Never write progress to stdout —
  stdout is reserved for the final summary/cutlist.

## When extending

- New per-frame signal → add file under `internal/signals/`, mirror
  the `BlackDetector` shape (constructor + `Push` + `Events`).
- New output format → add a `case` to the switch in `main.go`'s
  output dispatch. Keep stdout for data, stderr for diagnostics.
- New CLI flag → add to the `flag.*` block at the top of `main()`,
  document in `README.md`.

## When it breaks

- Most likely: ffmpeg version difference produces different stderr
  format for `silencedetect`. The regex in `internal/signals/silence.go`
  expects `silence_start: X` / `silence_end: Y | silence_duration: Z`.
- Second most likely: `decode_jpeg`-style ffmpeg API changes break
  `-pix_fmt rgb24` decoding. Known to be stable since ffmpeg 4.x but
  worth checking if frames come back the wrong size.
