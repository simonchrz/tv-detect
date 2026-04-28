package signals

import (
	"fmt"
	"math"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// NNDetector wraps an ONNX MobileNetV2 backbone + a tiny in-Go linear
// head. The backbone is the heavy 2 MB-weights part and stays
// constant; the head is a 1280-float vector that can be hot-reloaded
// from disk so a nightly fine-tune can update predictions without
// restarting tv-detect.
//
// Per-frame cost on M-series Mac CoreML execution provider: ~1-2 ms
// for the backbone + a single fused multiply-add for the head.
type NNDetector struct {
	session  *ort.AdvancedSession
	inTensor *ort.Tensor[float32]
	outTensor *ort.Tensor[float32]
	frameW    int
	frameH    int
	headPath  string

	mu          sync.RWMutex
	headW       []float32 // 1280 / 1281 / 1286 / 1287 depending on format
	headBias    float32
	headMtime   int64 // last mtime; reload when it changes
	headLoaded  bool
	headWithLogo bool // true → weights[1280] = logo-conf coefficient
	headWithChan bool // true → weights[1280+(0|1) .. +6) = channel one-hot

	channelIdx int  // index into nnChannels for our recording, or -1
}

// Backbone tensor shape: (1, 3, 224, 224). Output: (1, 1280).
//
// Four head formats are supported, distinguished at load time by the
// raw-bytes length of head.bin. Weights are little-endian float32
// packed back-to-back, followed by a single float32 bias.
//
//   - LEGACY                   (5124 B): backbone only, 1280 weights.
//   - +LOGO                    (5128 B): backbone (1280) + logo conf (1).
//   - +CHAN                    (5148 B): backbone (1280) + channel one-hot (6).
//   - +LOGO+CHAN               (5152 B): backbone + logo + channel = 1287 weights.
//
// nnChannels MUST be appended-to-only — re-ordering or inserting
// breaks every previously trained head. The Python trainer
// (scripts/train-head.py) holds the same list verbatim.
const (
	nnInputW  = 224
	nnInputH  = 224
	nnFeatDim = 1280 // backbone output size
	nnBatch   = 8    // frames per ONNX inference call. CoreML on M-series benefits from batched matmul; see ConfidenceBatch. Sub-batches are zero-padded.
)

var nnChannels = []string{
	"kabel-eins", "prosieben", "rtl", "sat-1", "sixx", "vox",
}

// imagenet normalization (BT.601-ish luma is wrong; the backbone was
// trained on ImageNet RGB normalized stats, so we have to match)
var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

var ortInitOnce sync.Once
var ortInitErr error

// initOrtRuntime is called lazily on first NNDetector creation.
// We point the runtime at the system-installed shared library; on Mac
// brew installs it under /opt/homebrew/lib, on Linux it's typically
// /usr/lib or wherever the distro put it. Override via TVD_ORT_LIB
// env var.
func initOrtRuntime() error {
	ortInitOnce.Do(func() {
		if libPath := os.Getenv("TVD_ORT_LIB"); libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		} else {
			for _, p := range []string{
				"/opt/homebrew/lib/libonnxruntime.dylib",
				"/usr/local/lib/libonnxruntime.dylib",
				"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
				"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
				"/usr/lib/libonnxruntime.so",
			} {
				if _, err := os.Stat(p); err == nil {
					ort.SetSharedLibraryPath(p)
					break
				}
			}
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// NewNNDetector loads the ONNX backbone + the linear head weights.
// The backbone's path also implicitly anchors any external-data
// sidecar (PyTorch's exporter writes large weights to <name>.data;
// onnxruntime resolves that automatically by location).
//
// channelSlug, if it matches one of nnChannels, contributes a
// one-hot feature when the loaded head is a +CHAN format. Empty or
// unknown slugs map to all-zero one-hot — the head's bias still fires.
func NewNNDetector(backbonePath, headPath string, frameW, frameH int, channelSlug string) (*NNDetector, error) {
	if err := initOrtRuntime(); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}
	inShape := ort.NewShape(nnBatch, 3, nnInputH, nnInputW)
	outShape := ort.NewShape(nnBatch, nnFeatDim)
	inT, err := ort.NewEmptyTensor[float32](inShape)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	outT, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		inT.Destroy()
		return nil, fmt.Errorf("output tensor: %w", err)
	}
	// Cap intra-op threads at 2 — without this, ORT's CoreML execution
	// provider auto-grabs every core, which catastrophically over-
	// subscribes when the caller already runs N parallel chunk workers
	// (N × ~all-cores per session). Two threads keeps each session
	// modestly parallel while N workers stack cleanly to ~workers ×
	// 2 cores total.
	opts, err := ort.NewSessionOptions()
	if err != nil {
		inT.Destroy()
		outT.Destroy()
		return nil, fmt.Errorf("session opts: %w", err)
	}
	defer opts.Destroy()
	if err := opts.SetIntraOpNumThreads(2); err != nil {
		inT.Destroy()
		outT.Destroy()
		return nil, fmt.Errorf("set intra-op threads: %w", err)
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		inT.Destroy()
		outT.Destroy()
		return nil, fmt.Errorf("set inter-op threads: %w", err)
	}
	sess, err := ort.NewAdvancedSession(backbonePath,
		[]string{"frame"}, []string{"features"},
		[]ort.Value{inT}, []ort.Value{outT}, opts)
	if err != nil {
		inT.Destroy()
		outT.Destroy()
		return nil, fmt.Errorf("session: %w", err)
	}
	chanIdx := -1
	for i, s := range nnChannels {
		if s == channelSlug {
			chanIdx = i
			break
		}
	}
	d := &NNDetector{
		session: sess, inTensor: inT, outTensor: outT,
		frameW: frameW, frameH: frameH,
		headPath:   headPath,
		channelIdx: chanIdx,
	}
	if err := d.reloadHead(); err != nil {
		// Head missing is not fatal — the detector returns 0.5 (no
		// signal) until a head shows up. Useful for first-time runs
		// before any training has happened.
		fmt.Fprintf(os.Stderr, "nn: head not loaded (%v) — detector returns 0.5\n", err)
	}
	return d, nil
}

// reloadHead reads the binary weights file (1280 float32 weights +
// 1 float32 bias). Safe to call concurrently with Confidence.
func (d *NNDetector) reloadHead() error {
	st, err := os.Stat(d.headPath)
	if err != nil {
		return err
	}
	mtime := st.ModTime().UnixNano()
	d.mu.RLock()
	if d.headLoaded && d.headMtime == mtime {
		d.mu.RUnlock()
		return nil
	}
	d.mu.RUnlock()

	raw, err := os.ReadFile(d.headPath)
	if err != nil {
		return err
	}
	// Auto-detect head format by raw size. Four shapes possible —
	// see nnChannels comment block for the layout matrix.
	nC := len(nnChannels)
	legacyBytes  := (nnFeatDim + 1) * 4              // 5124
	withLogoBytes := (nnFeatDim + 1 + 1) * 4          // 5128
	withChanBytes := (nnFeatDim + nC + 1) * 4         // 5148
	withBothBytes := (nnFeatDim + 1 + nC + 1) * 4     // 5152
	var headDim int
	var withLogo, withChan bool
	switch len(raw) {
	case legacyBytes:
		headDim = nnFeatDim
	case withLogoBytes:
		headDim, withLogo = nnFeatDim+1, true
	case withChanBytes:
		headDim, withChan = nnFeatDim+nC, true
	case withBothBytes:
		headDim, withLogo, withChan = nnFeatDim+1+nC, true, true
	default:
		return fmt.Errorf("head file size %d, expected %d/%d/%d/%d "+
			"(legacy / +logo / +chan / +logo+chan)",
			len(raw), legacyBytes, withLogoBytes,
			withChanBytes, withBothBytes)
	}
	weights := make([]float32, headDim)
	for i := 0; i < headDim; i++ {
		weights[i] = floatLE(raw[i*4:])
	}
	bias := floatLE(raw[headDim*4:])

	d.mu.Lock()
	d.headW = weights
	d.headBias = bias
	d.headMtime = mtime
	d.headLoaded = true
	d.headWithLogo = withLogo
	d.headWithChan = withChan
	d.mu.Unlock()
	return nil
}

// MaybeReloadHead checks the head file's mtime and reloads if it
// changed. Cheap (one stat call) — call once per frame loop iteration
// or once per N frames as you prefer.
func (d *NNDetector) MaybeReloadHead() {
	st, err := os.Stat(d.headPath)
	if err != nil {
		return
	}
	d.mu.RLock()
	stale := !d.headLoaded || st.ModTime().UnixNano() != d.headMtime
	d.mu.RUnlock()
	if stale {
		_ = d.reloadHead()
	}
}

// Confidence returns the NN's ad-probability for one rgb24 frame.
// Returns 0.5 (= no signal) when the head hasn't been loaded yet.
//
// pixels is row-major rgb24 of length 3*frameW*frameH. Internally we
// resize to 224x224 by bilinear sampling (cheap pure-Go code) and
// normalize with ImageNet mean/std before the backbone forward pass.
//
// logoConf is the logo-template match confidence for the same frame
// (0..1). When the loaded head is a "with-logo" head (1281 weights),
// this is used as the 1281st input feature so the head can learn
// per-pattern logo trust. For a legacy 1280-weight head, logoConf is
// silently ignored — caller can still blend externally via NNWeight.
func (d *NNDetector) Confidence(pixels []byte, logoConf float64) float64 {
	r := d.ConfidenceBatch([][]byte{pixels}, []float64{logoConf})
	if len(r) == 0 {
		return 0.5
	}
	return r[0]
}

// ConfidenceBatch runs ONNX inference on up to nnBatch frames in
// one session.Run call (CoreML batches matmul efficiently on
// M-series GPUs). The session was created with a fixed batch
// dimension of nnBatch; partial batches are zero-padded and only
// the first len(framesPixels) results are returned. Caller must
// pass framesPixels and logoConfs of equal length, ≤ nnBatch.
func (d *NNDetector) ConfidenceBatch(framesPixels [][]byte, logoConfs []float64) []float64 {
	n := len(framesPixels)
	if n == 0 {
		return nil
	}
	if n > nnBatch {
		// Caller error — split into multiple calls upstream.
		n = nnBatch
		framesPixels = framesPixels[:nnBatch]
		logoConfs = logoConfs[:nnBatch]
	}
	in := d.inTensor.GetData()
	stride := 3 * nnInputH * nnInputW
	for i := 0; i < n; i++ {
		preprocess(framesPixels[i], d.frameW, d.frameH,
			in[i*stride:(i+1)*stride])
	}
	// Zero-pad unused slots so leftover data from a previous call
	// can't leak into this batch's results.
	for i := n; i < nnBatch; i++ {
		clear(in[i*stride : (i+1)*stride])
	}
	if err := d.session.Run(); err != nil {
		out := make([]float64, n)
		for i := range out {
			out[i] = 0.5
		}
		return out
	}
	feats := d.outTensor.GetData()
	d.mu.RLock()
	defer d.mu.RUnlock()
	out := make([]float64, n)
	if !d.headLoaded {
		for i := range out {
			out[i] = 0.5
		}
		return out
	}
	chanBase := nnFeatDim
	if d.headWithLogo {
		chanBase = nnFeatDim + 1
	}
	for i := 0; i < n; i++ {
		logit := d.headBias
		off := i * nnFeatDim
		for j := 0; j < nnFeatDim; j++ {
			logit += d.headW[j] * feats[off+j]
		}
		if d.headWithLogo {
			logit += d.headW[nnFeatDim] * float32(logoConfs[i])
		}
		if d.headWithChan && d.channelIdx >= 0 {
			logit += d.headW[chanBase+d.channelIdx]
		}
		out[i] = sigmoid(logit)
	}
	return out
}

// Close releases ORT resources. Call when shutting down the
// detector permanently.
func (d *NNDetector) Close() error {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.inTensor != nil {
		d.inTensor.Destroy()
	}
	if d.outTensor != nil {
		d.outTensor.Destroy()
	}
	return nil
}

// preprocess resizes srcW × srcH rgb24 into a 224×224 NCHW float32
// tensor with ImageNet normalization. Bilinear sample, single-pass.
func preprocess(src []byte, srcW, srcH int, dst []float32) {
	const tw = nnInputW
	const th = nnInputH
	scaleX := float32(srcW) / tw
	scaleY := float32(srcH) / th
	// Layout: dst[c*tw*th + y*tw + x]
	plane := tw * th
	for ty := 0; ty < th; ty++ {
		sy := float32(ty) * scaleY
		sy0 := int(sy)
		sy1 := sy0 + 1
		if sy1 >= srcH {
			sy1 = srcH - 1
		}
		fy := sy - float32(sy0)
		for tx := 0; tx < tw; tx++ {
			sx := float32(tx) * scaleX
			sx0 := int(sx)
			sx1 := sx0 + 1
			if sx1 >= srcW {
				sx1 = srcW - 1
			}
			fx := sx - float32(sx0)
			i00 := 3 * (sy0*srcW + sx0)
			i01 := 3 * (sy0*srcW + sx1)
			i10 := 3 * (sy1*srcW + sx0)
			i11 := 3 * (sy1*srcW + sx1)
			dstIdx := ty*tw + tx
			for c := 0; c < 3; c++ {
				v := (1-fx)*(1-fy)*float32(src[i00+c]) +
					fx*(1-fy)*float32(src[i01+c]) +
					(1-fx)*fy*float32(src[i10+c]) +
					fx*fy*float32(src[i11+c])
				v = (v/255.0 - imagenetMean[c]) / imagenetStd[c]
				dst[c*plane+dstIdx] = v
			}
		}
	}
}

// sigmoid returns 1 / (1 + e^-x).
func sigmoid(x float32) float64 {
	return 1.0 / (1.0 + math.Exp(-float64(x)))
}

// floatLE reads a little-endian float32 from b[0:4].
func floatLE(b []byte) float32 {
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return math.Float32frombits(bits)
}
