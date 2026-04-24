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
  edge mask). Reusing the format means the existing per-channel
  trained `.logo.txt` cache (in `~/mnt/pi-tv/hls/.logos/`) works
  unchanged.

## Where the bodies are buried

- **Logo detection currently produces no usable confidences.** The
  edge detector uses Sobel 3x3 (correct convention), the template
  parser handles comskip's `.logo.txt` format including the binary
  marker byte and the `-`/`|`/`+`/space mask glyphs. Despite that,
  matching the cached comskip templates against decoded frames
  produces a bimodal 0/1 distribution with the logo "hit" frames
  scattered randomly — clearly NOT the actual logo positions.

  Investigated:
  - Padding-offset hypothesis (`picWidth=736` vs decoded 720): tried
    fixed dx in [-48..+48], no consistent best offset across frames.
  - Edge-magnitude threshold sweep (10..200): bimodal stays bimodal,
    just shifts how many frames hit "all 28 expected edges by chance".
  - The template's "edge expected" positions are heavily concentrated
    in a single row (e.g. row 25 of 34 for the VOX template), making
    even a 100% match a low-information signal.

  Likely root cause: comskip's training-time `width` variable (used
  for both edge buffer addressing and template coordinates) doesn't
  match the live-decode width. Source-side aspect-ratio handling,
  pillarboxing, or comskip's internal aspect normalization shifts
  logo position vs. the cached template by a non-constant amount.

  Path forward when this matters: train logos in tv-detect itself
  rather than parse comskip's cache. A `tv-detect train-logo`
  subcommand that processes 25 min of show content and emits its own
  edge mask in tv-detect's own coordinate space sidesteps the
  reverse-engineering problem entirely.

  Today's consequence: the block-formation state machine emits an
  empty list because it has no usable per-frame logo signal. Phases
  5b (validation against comskip cutlists) and 6 (Python integration)
  are blocked on either fixing this or shipping a logo-trainer.

- **`-ss X -t Y` seek is approximate.** ffmpeg seeks to the nearest
  I-frame ≤ X. On a 30-min recording with 8 chunks we observe
  ~15-20 extra frames total across all chunks — minor over-counting
  vs single-pipeline. Acceptable for our cutlist tolerance (±2
  frames per boundary).

- **Frame timestamps are `index/fps`**, not container PTS. This
  matches comskip's convention. ffmpeg's `blackdetect` uses container
  PTS, so its event times sit ~0.5 s later than ours on MPEG-TS
  inputs (GOP-reorder offset). DON'T "fix" this — the downstream
  Python code expects the comskip convention.

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
