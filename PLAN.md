# tv-detect

Multi-threaded ad-block detector als Go binary, designed um als
subprocess von der existing Python orchestration (`tv-live-comskip.py`,
`hls-gateway/service.py`) aufgerufen zu werden. Ersetzt langfristig die
comskip-Detection-Stufe; behält die Python-side cutlist-combinator
+ NN-classifier integration.

## Ziele

- **Performance**: echte multi-thread per-frame analysis. Ziel: 30 min
  HD recording in <30 s (vs comskip ~75 s).
- **Composability**: stdin/stdout JSON, einfach von Python aus zu
  pipen, kein in-process IPC.
- **Single binary**: cross-compile macOS arm64 + linux arm64, deploy
  per `scp`, keine runtime deps außer ffmpeg.
- **Comskip-compatible cutlist**: gleiches frame-paar format
  (`<start_frame>\t<end_frame>` per line) damit existing parsers
  weiter funktionieren.

## Was Go macht (was Python NICHT mehr macht)

- Decode-Pipeline (libavcodec via ffmpeg subprocess)
- Per-frame Logo-Correlation gegen .logo.txt template
- Blackframe-Detection (per-pixel mean luminance)
- Audio-Silence (parsing of ffmpeg silencedetect stderr)
- Scene-Cut (histogram bhattacharyya)
- Initial block-formation state machine (apply
  min_commercialbreak / min_show_segment_length / max_commercialbreak)
- Multi-thread orchestration: split input in N chunks, parallel workers

## Was Python WEITER macht

- Sliding-window scan cycles + per-channel scheduling
- Logo cache management (training delegated to first scan, persist
  the trained template for subsequent runs)
- NN classifier inference (if enabled)
- Final cutlist combinator (logo-refine, blackframe-extend,
  silence-extend, NN — picks best evidence per boundary)
- Persistence (.live_ads.json, ads.json, ads_user.json)
- API endpoints (live-ads-stream, recording/<uuid>/ads, ...)

## Datenfluss

```
                  Python orchestration
                         │
                         │ spawn subprocess
                         ▼
              ┌──────────────────────┐
              │  tv-detect <input>   │
              │  --logo X            │
              │  --workers N         │
              └──────────┬───────────┘
                         │
              ┌──────────┴───────────┐
              │                      │
              ▼                      ▼
       ffmpeg subprocess     ffmpeg subprocess
       (raw rgb24 frames)    (blackdetect+silencedetect filters)
              │                      │
              ▼                      ▼
       frame queue ──────────► aggregator
                                     │
                          worker pool (N goroutines):
                          per-frame logo + blackframe + scenecut
                                     │
                                     ▼
                          state machine: blocks
                                     │
                                     ▼
                          JSON to stdout
                                     │
                                     ▼
                            Python combinator
```

## CLI design

```bash
tv-detect [flags] <input.ts | input.m3u8>

flags:
  --logo <path>          # comskip .logo.txt template (skips logo training)
  --workers <n>          # parallel chunk workers (default: NumCPU)
  --output <fmt>         # summary | jsonlines | csv (default: summary)
  --min-block-sec <n>    # filter sub-N blocks (default: 60)
  --max-block-sec <n>    # split blocks longer than N (default: 900)
  --min-show-segment <n> # min show between blocks before merging (default: 120)
  --logo-threshold <f>   # 0..1, logo absent below this (default: 0.10)
  --blackframe-d <s>     # min duration for blackframe (default: 0.10)
  --silence-noise-db <n> # silence noise floor (default: -30)
  --silence-d <s>        # min silence duration (default: 0.50)
  --quiet                # suppress progress to stderr
```

Output formats:

- `summary` (default, for Python integration):
  ```json
  {
    "fps": 25.0,
    "frame_count": 45000,
    "duration_s": 1800.0,
    "blocks": [[start_sec, end_sec], ...],
    "stats": {"workers": 8, "elapsed_s": 28.4}
  }
  ```

- `jsonlines` (debugging / per-frame analysis):
  ```
  {"f":1,"t":0.04,"logo":0.51,"black":0,"silence":0,"scene":0.0}
  {"f":2,"t":0.08,"logo":0.51,"black":0,"silence":0,"scene":0.02}
  ...
  ```

- `csv` (analysis in pandas):
  Standard tabular for offline exploration.

## Implementation phases

### Phase 1 — Skeleton + decode (3-5 Tage)

- `go mod init github.com/simonchrz/tv-detect`
- ffmpeg subprocess wrapper: gibt rgb24-frames + duration metadata
- CLI grundstruktur mit cobra/spf13 oder flag std-lib
- Verify: read all frames, count, exit clean

### Phase 2 — Per-frame signal extractors (4-6 Tage)

- `internal/signals/blackframe.go`: per-pixel mean luminance < threshold
  - Test gegen ffmpeg blackdetect output: must match within 1 frame
- `internal/signals/silence.go`: parse ffmpeg silencedetect stderr
  - `silence_start: X` / `silence_end: Y` regex
- `internal/signals/scenecut.go`: histogram bhattacharyya distance
  vs. previous frame
- `internal/signals/logo.go`: parse comskip .logo.txt → mask
  - Edge-correlate per frame against mask
  - Output 0-1 confidence per frame
- Each unit-tested with sample frames at known states

### Phase 3 — Multi-thread orchestration (2-3 Tage)

- Chunk-split input via ffmpeg `-ss X -t Y` time ranges
- N goroutines process chunks in parallel
- Channel-based pipeline:
  decoder → frame channel → worker pool → result aggregator
- Merge per-chunk results, dedup overlap regions

### Phase 4 — Block formation state machine (2-3 Tage)

- Per-frame signal stream → stateful detector
- Apply heuristics:
  - logo present for ≥120 s = "show"
  - logo absent for ≥60 s = "ad"
  - merge blocks separated by <120 s show
  - validate_silence: confirm block boundaries with audio dip
- Output frame-pair cutlist matching comskip's format

### Phase 5 — CLI polish + tests (2-3 Tage)

- All output formats
- Progress bar via stderr (suppressable with --quiet)
- Cross-compile:
  ```bash
  GOOS=darwin GOARCH=arm64 go build -o build/tv-detect-darwin-arm64 ./cmd/tv-detect
  GOOS=linux  GOARCH=arm64 go build -o build/tv-detect-linux-arm64  ./cmd/tv-detect
  ```
- Validate gegen comskip output on 5+ test recordings:
  cutlist diffs ≤ 2 frames

### Phase 6 — Python integration (2-3 Tage)

- `tv-live-comskip.py`: replace `subprocess.run(['comskip', ...])`
  with `subprocess.run(['tv-detect', ...])`
- Shadow mode first: run BOTH, log diffs, only swap when matching ≥99%
- `hls-gateway/service.py`: same swap in `_rec_cskip_spawn` +
  `_live_ads_analyze`

**Total: 3-4 Wochen dedicated work.**

## Dependencies

- Go 1.22+ (modern toolchain, generics if useful)
- Standard library: `os/exec`, `encoding/json`, `image`, `math`,
  `bufio`, `flag`, `runtime`
- Optional: `gonum.org/v1/gonum/stat` für scene-cut histograms
  (or hand-rolled — minimal)
- No CGO — keeps cross-compilation trivial

## Build & deploy

```bash
# Mac dev
cd ~/src/tv-detect
go test ./...
go build -o build/tv-detect-darwin-arm64 ./cmd/tv-detect

# Cross-compile for Pi container
GOOS=linux GOARCH=arm64 go build -o build/tv-detect-linux-arm64 ./cmd/tv-detect

# Mac install
ln -s "$(pwd)/build/tv-detect-darwin-arm64" /usr/local/bin/tv-detect

# Pi install (in hls-gateway Dockerfile)
COPY tv-detect-linux-arm64 /usr/local/bin/tv-detect
```

## Open questions / risks

- **Logo template format**: comskip's `.logo.txt` is custom (logoMinX/Y,
  edge mask). Do we keep the format (compatible with existing cached
  templates) or design our own? → keep comskip format, parse it.
- **Logo training from scratch**: phase 2 specifies CONSUMING a logo
  template, not training one. For first-run channels without a template,
  Python still runs comskip in deep-init mode to train, then tv-detect
  takes over. Or: separate `tv-detect-train-logo` subcommand later.
- **Frame timing**: ffmpeg's PTS vs frame-count. Should match comskip
  exactly to avoid timestamp drift in cutlist.
- **Scene-cut threshold**: needs empirical tuning on real DE-private
  content. Will iterate.
- **Audio sync**: blackdetect runs on video, silencedetect on audio.
  Both ffmpeg subprocesses must agree on PTS reference.

## Test data

Use `/Users/simon/mnt/pi-tv/hls/_rec_*/` recordings. Match cutlist
output against:
- comskip's `.txt` output (golden reference)
- User-edited `ads_user.json` (preferred — closer to ground truth)

Validation suite: 10+ recordings spanning channels + content types
(reality, scripted, news). Goal: ≥99% boundary match within 2 frames
for each.
