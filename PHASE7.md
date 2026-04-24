# Phase 7: ONNX-based NN evidence source

`internal/signals/nn.go` integrates a frozen MobileNetV2 backbone (in
ONNX) plus a hot-reloadable linear head (raw float32 file) as a 4th
evidence source alongside logo / blackframe / silence / scene-cut.

## Architecture split

| Piece | Where | Lifecycle |
|---|---|---|
| MobileNetV2 backbone | `~/mnt/pi-tv/hls/.tvd-models/backbone.onnx` | exported once, immutable |
| Linear head (1280 weights + bias) | `~/mnt/pi-tv/hls/.tvd-models/head.bin` | retrained nightly, hot-reloaded |
| ONNX Runtime | system shared library (CGO) | `brew install onnxruntime` / `apt install libonnxruntime` |
| Per-frame inference | Go (`signals.NNDetector`) | called from pipeline, no Python at runtime |

The backbone (~9 MB) ships once. The head (~5 KB) is what online
learning updates — `tv-detect` watches its mtime and reloads whenever
the nightly trainer rewrites it.

## Per-frame cost

Measured on M5 Pro (CoreML execution provider):
- Preprocess (rgb24 → 224×224 float32 NCHW with ImageNet norm): ~1 ms
- Backbone forward pass: ~1.3 ms
- Linear head matmul + sigmoid: <1 µs
- Total: ~2.4 ms/frame → **420 fps** on a 50-min 720×576 recording

Pi 5 estimate (CPU only): ~10 ms backbone forward → ~40-60 fps. For a
30-min recording at 1 fps sampling = ~30 s.

## Setup

```bash
brew install onnxruntime          # Mac
# or apt install libonnxruntime    # Linux

cd /Users/simon/src/tv-detect
go get github.com/yalue/onnxruntime_go
make build                        # CGO_ENABLED=1 by default
```

Pi container: add `libonnxruntime-dev` to the Dockerfile install line
(or bind-mount the host's lib alongside `tv-detect`).

## Training the head

```bash
# 1. Export the backbone (once — file ships with the project,
#    re-run only if you switch architectures)
python3 scripts/export-backbone.py
# → ~/mnt/pi-tv/hls/.tvd-models/backbone.onnx  (~9 MB)

# 2. Train the head from labelled recordings
#    Walks ~/mnt/pi-tv/hls/_rec_*/ for ads_user.json (or ads.json
#    fallback), extracts features via the ONNX backbone, fits a
#    logistic regression, writes head.bin.
python3 scripts/train-head.py
# → ~/mnt/pi-tv/hls/.tvd-models/head.bin  (~5 KB)
```

The feature cache lives at `~/.cache/tvd-features/`; re-running the
trainer only re-extracts features for recordings whose source mtime
changed.

## Online learning loop

Schedule `train-head.py` as a nightly cron / launchd job. Each run
picks up newly-edited `ads_user.json` files, re-trains the head
incrementally, and writes a fresh `head.bin`. Every running
`tv-detect` process notices the new mtime and reloads its head
weights without restart.

`tv-detect` falls back to confidence 0.5 (= no signal) when the head
file is missing or has the wrong size; it still emits a cutlist based
on logo + blackframe + silence alone. The NN is purely additive
evidence.

## Status

- ✅ ONNX backbone export (legacy TorchScript exporter, single
      .onnx file — the new dynamo path produces an external-data
      sidecar that ORT 1.25 chokes on)
- ✅ Go `signals.NNDetector` with bilinear preprocess + ImageNet
      normalisation + ONNX session + linear head + hot-reload
- ✅ Smoke test (`tv-detect-nn-smoke`) confirms 420 fps end-to-end
- ✅ `train-head.py` produces a valid `head.bin` from 7 labelled
      recordings (24k frames, 24% ad fraction, 93% train accuracy)
- ⏳ Wire `NNDetector` into `pipeline.Run` as the 4th signal
- ⏳ Combine NN confidence with logo confidence in `blocks.Form`'s
      state machine
- ⏳ Pi: `apt install libonnxruntime` (or release tarball) +
      Dockerfile change
- ⏳ More training data: only 7 recordings have ads.json today, head
      will overfit until more recordings get auto-parsed or
      user-edited
