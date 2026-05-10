# tv-detect

Multi-threaded ad-block detector for broadcast TV recordings — single
Go binary, no CGO, ffmpeg as the only runtime dependency.

## What it does

Given an MPEG-TS recording (DVB-T/C/S, IPTV, cable), tv-detect emits
a frame-pair cutlist of detected commercial blocks. Combines four
per-frame signals through a small neural-network head:

- **Edge-correlation logo matching** against a per-channel template.
- **Black-frame and silence** transitions (typical ad-bumper signature).
- **Scene-cut intensity** (luma-histogram Bhattacharyya).
- **MobileNetV2 backbone** + trainable MLP head (channel-aware,
  hot-reloadable on `head.bin` mtime change).

A nightly self-improving training loop with champion-challenger
gating, active-learning surfacing, and pseudo-label self-training keeps
the head fresh as new recordings come in. Current production: Block-IoU
0.92 / Test Acc 98.5 % at n=35 test recordings across 9 channels.

## Status

| Phase | Item | State |
|---|---|---|
| 1 | ffmpeg decode pipeline | ✅ |
| 2 | Blackframe detector | ✅ |
| 2 | Silence detector | ✅ parallel ffmpeg subprocess |
| 2 | Scene-cut detector | ✅ luma histogram Bhattacharyya |
| 2 | Logo detector | ✅ Sobel edge correlation vs trained template |
| 2 | Logo trainer (`tv-detect-train-logo`) | ✅ |
| 3 | Multi-thread chunk pipeline | ✅ |
| 4 | Block-formation state machine | ✅ |
| 5 | Cross-compile | ✅ darwin-arm64 / linux-arm64 / linux-amd64 |
| 6 | Production swap (replaces legacy comskip-based pipelines) | ✅ |
| 7 | NN evidence source via ONNX (`signals.NNDetector`) | ✅ MLP2 head (1290→32→1, channel one-hot) |
| 8 | Letterbox-aware logo matching (`--logo-y-offset N`) | ✅ |
| 9 | Self-improving training loop | ✅ nightly retrain, champion-challenger, active-learning, pseudo-label self-training |
| 10 | Whisper ad-classifier post-processor | ✅ German ASR refines block boundaries; +5.4 pp Block-IoU on n=9 eval |

## Requirements

- **Go 1.22+** to build.
- **`ffmpeg` and `ffprobe` on `$PATH` at runtime** — tv-detect itself
  is a thin orchestrator that shells out for video decode (raw rgb24
  pipe), audio analysis (`silencedetect` filter), and metadata
  (`ffprobe`). No CGO, no libav linkage; the trade-off is that
  ffmpeg must be installed on every box you deploy to.

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

# Cutlist output (frame-pair format, line-delimited).
tv-detect --workers 4 --output cutlist --logo vox.logo.txt recording.ts

# Letterbox-aware logo matching for 16:9 movies in 4:3 broadcast
# containers. Shifts the logo template's y-coords down by N px so the
# matcher hits the actual logo position inside the visible content
# area, not the top black bar.
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
internal/signals/logo.go       # edge-correlation logo matcher
internal/signals/nn.go         # ONNX MobileNetV2 + MLP head loader
internal/blocks/blocks.go      # logo+nn+black+silence → final cutlist
internal/pipeline/parallel.go  # chunk-split + N-worker orchestrator
pkg/logotemplate/template.go   # parser for logo template format
scripts/model-anchor.sh        # snapshot+restore trained models via GitHub releases
```

`pkg/` is consumable by external Go packages; `internal/` is private.

## Performance

27-min MPEG-2 recording, 720x576, 25 fps, 41 034 frames:

| Configuration | fps |
|---|---|
| `--workers 1` | 1 510 |
| `--workers 4` | 4 109 |
| `--workers 8` | 4 464 |

Diminishing returns past 4 workers on Apple silicon because efficiency
cores don't help video decode much.

## Testing

```bash
go test ./...
```

All packages have unit tests for the per-frame algorithms.
