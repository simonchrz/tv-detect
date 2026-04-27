# tv-detect

Multi-threaded ad-block detector — single Go binary, drop-in
replacement for the comskip detection stage.

## Why

`comskip` is venerable but single-threaded, written in C with mixed-era
ffmpeg bindings, and saturates one core for ~75 s on a typical
30-minute MPEG-2 recording. This binary does the same job ~8× faster
on a multi-core mac/Pi by chunk-splitting the input across N parallel
ffmpeg subprocesses, processing per-frame signals in goroutines, and
emitting the same frame-pair cutlist format so existing Python
parsers consume it unchanged.

## Status

| Phase | Item | State |
|---|---|---|
| 1 | ffmpeg decode pipeline | ✅ |
| 2 | Blackframe detector | ✅ matches `ffmpeg blackdetect` |
| 2 | Silence detector | ✅ parallel ffmpeg subprocess |
| 2 | Scene-cut detector | ✅ luma histogram Bhattacharyya |
| 2 | Logo detector | ✅ Sobel edge correlation vs trained template |
| 2 | Logo trainer (`tv-detect-train-logo`) | ✅ comskip-format template from N min of content |
| 3 | Multi-thread chunk pipeline | ✅ ~8× speedup vs comskip |
| 4 | Block-formation state machine | ✅ logo-gated cutlist output |
| 5 | Cross-compile | ✅ darwin-arm64 / linux-arm64 / linux-amd64 |
| 5 | Validation suite | ✅ 7/19 frame-perfect, 5/19 close, see [VALIDATION.md](VALIDATION.md) |
| 6a | Mac launchd agent swap (tv-comskip.sh → tv-detect) | ✅ full swap, comskip not invoked, see [PHASE6.md](PHASE6.md) |
| 6b | Pi container swap (hls-gateway/_rec_cskip_spawn) | ✅ |
| 7  | NN evidence source via ONNX (`signals.NNDetector`) | ✅ +LOGO head (1281 weights) deployed, blends with logo via `--nn-weight 0.3` |
| 8  | Letterbox-aware logo matching (`--logo-y-offset N`) | ✅ daemon runs cropdetect, shifts template y-coords for 16:9-in-4:3 broadcasts |

End-to-end output works once a per-channel template has been trained
by `tv-detect-train-logo`. The cached comskip templates don't align
with tv-detect's decode coordinates (see `CLAUDE.md` for the
investigation), but training an own template from 5-25 min of show
content produces a working detector: on a 50-min VOX CSI recording we
find the same 2 ad blocks comskip finds, boundaries within ~60 s.

## Requirements

- **Go 1.22+** to build.
- **`ffmpeg` and `ffprobe` on `$PATH` at runtime** — tv-detect itself
  is a thin orchestrator that shells out for video decode (`ffmpeg`
  raw rgb24 pipe), audio analysis (`ffmpeg` `silencedetect` filter),
  and metadata (`ffprobe`). No CGO, no libav linkage; the trade-off is
  that ffmpeg must be installed on every box you deploy to.

  Already present on every target host this binary expects (Mac via
  brew, the linuxserver/tvheadend image on the Pi, etc.).

## Build

```bash
make build              # native binary at build/tv-detect
make build-all          # cross-compiles darwin-arm64, linux-arm64, linux-amd64
make test               # unit tests across all packages
make install            # symlink build/tv-detect into /usr/local/bin
```

## Usage

```bash
# Train a per-channel logo template from any recording of that channel.
# 5-25 min of show content is enough; lowers persistence threshold if
# the recording has mid-roll ad breaks (default 0.85 assumes show-only).
tv-detect-train-logo --edge-threshold 40 \
  --output vox.logo.txt recording.ts

# Probe input metadata only.
tv-detect --probe path/to/recording.ts

# Full pipeline with a trained template, summary JSON to stdout.
tv-detect --workers 4 --logo vox.logo.txt recording.ts

# Cutlist output (comskip-compatible, line-delimited frame pairs).
tv-detect --workers 4 --output cutlist --logo vox.logo.txt recording.ts

# Letterbox-aware logo matching for 16:9 movies in 4:3 broadcast
# containers (e.g. RTL Spielfilm). Shifts the logo template's y-coords
# down by N px so the matcher hits the actual logo position inside the
# visible content area, not the top black bar. The Mac daemon computes
# N automatically via ffmpeg cropdetect; pass it manually for one-offs.
tv-detect --workers 4 --logo rtl.logo.txt --logo-y-offset 60 recording.ts

# Per-signal debug output (each independent of --output).
tv-detect --emit-blackframes input.ts
tv-detect --emit-silences input.ts
tv-detect --emit-scenes input.ts
tv-detect --emit-logo-csv --logo vox.logo.txt input.ts
```

## Layout

```
cmd/tv-detect/main.go          # CLI entry point
internal/decode/probe.go       # ffprobe wrapper
internal/decode/decode.go      # raw rgb24 frame stream from ffmpeg
internal/signals/blackframe.go # mean-luma threshold + run aggregator
internal/signals/silence.go    # ffmpeg silencedetect parser
internal/signals/scenecut.go   # per-frame luma histogram + Bhattacharyya
internal/signals/logo.go       # comskip-template edge correlation (WIP)
internal/blocks/blocks.go      # logo+black+silence → final cutlist
internal/pipeline/parallel.go  # chunk-split + N-worker orchestrator
pkg/logotemplate/template.go   # parser for comskip's .logo.txt format
```

`pkg/` is consumable by external Go packages; `internal/` is private.
The package boundary is intentional — logo template parsing is the
only piece anyone outside this binary might want to embed (e.g. a
training tool).

## Performance baseline

27-min MPEG-2 recording, 720x576, 25 fps, 41 034 frames, M5 Pro:

| Host | Configuration | Wall time | fps |
|---|---|---|---|
| M5 Pro Mac | `comskip` (reference) | ~75 s | ~550 |
| M5 Pro Mac | tv-detect, workers=1 | 27.2 s | 1 510 |
| M5 Pro Mac | tv-detect, workers=4 | 10.0 s | 4 109 |
| M5 Pro Mac | tv-detect, workers=8 |  9.2 s | 4 464 |
| Pi 5 (4-core) | tv-detect, workers=4 | 221 s | 197 |

Diminishing returns past 4 workers on Apple silicon because the
efficiency cores don't help video decode much. The Pi 5 is CPU-bound
even at workers=4 (197 fps vs Mac's 4109 fps), but still lands ~1.4×
faster than comskip on the same box.

## Testing

```bash
go test ./...
```

All packages have unit tests for the per-frame algorithms. End-to-end
validation against comskip cutlists on real recordings is Phase 5.
