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
| 2 | Logo detector | ⚠️ skeleton in place; matching quality broken |
| 3 | Multi-thread chunk pipeline | ✅ ~8× speedup vs comskip |
| 4 | Block-formation state machine | ✅ logo-gated cutlist output |
| 5 | Cross-compile + validation suite | ⏳ |
| 6 | Python integration (hls-gateway, tv-live-comskip) | ⏳ |

End-to-end output is empty until logo detection is fixed — the block
classifier needs per-frame logo presence as its primary signal.
Blackframe + silence alone don't separate ad/show reliably on
private DE-TV content.

## Build

```bash
go build -o build/tv-detect ./cmd/tv-detect

# Cross-compile for Pi container deployment (Phase 6):
GOOS=linux GOARCH=arm64 go build -o build/tv-detect-linux-arm64 ./cmd/tv-detect
```

## Usage

```bash
# Probe input metadata only.
tv-detect --probe path/to/recording.ts

# Full pipeline, summary JSON to stdout.
tv-detect --workers 4 --logo /path/to/.logos/vox.logo.txt input.ts

# Cutlist output (comskip-compatible, line-delimited frame pairs).
tv-detect --workers 4 --output cutlist input.ts > cutlist.txt

# Per-signal debug output (each independent of --output).
tv-detect --emit-blackframes input.ts
tv-detect --emit-silences input.ts
tv-detect --emit-scenes input.ts
tv-detect --emit-logo-csv --logo template.logo.txt input.ts
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

| Configuration | Wall time | fps |
|---|---|---|
| `comskip` (reference) | ~75 s | ~550 |
| tv-detect, workers=1 | 27.2 s | 1 510 |
| tv-detect, workers=4 | 10.0 s | 4 109 |
| tv-detect, workers=8 | 9.2 s | 4 464 |

Diminishing returns past 4 workers because efficiency cores on Apple
silicon don't help video decode much. Pi 5 (4 performance cores) will
likely scale linearly to 4.

## Testing

```bash
go test ./...
```

All packages have unit tests for the per-frame algorithms. End-to-end
validation against comskip cutlists on real recordings is Phase 5.
