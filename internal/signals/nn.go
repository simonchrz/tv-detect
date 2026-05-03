package signals

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
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
	headW       []float32 // 1280..1288 weights, depending on format
	headBias    float32
	headMtime   int64 // last mtime; reload when it changes
	headLoaded  bool
	headWithLogo  bool // true → weights[1280] = logo-conf coefficient
	headWithChan  bool // true → weights[1280+(0|1) .. +6) = channel one-hot
	headWithAudio bool // true → trailing weight = audio RMS (per-second, [0,1] normalised)

	channelSlug string // recording's channel slug, stored for MLP re-resolution after sidecar reload
	channelIdx int     // index into nnChannels for our recording, or -1

	// MLP-head state (when headIsMLP=true; LogReg fields above are
	// unused). Loaded from a "MLP1" magic-prefixed head.bin as
	// specified in scripts/train-head.py write_mlp_head_v1.
	headIsMLP    bool
	mlpInDim     int       // total input dim (= mlpBackbone + mlpNLogo + mlpNAudio + mlpNChannel)
	mlpHidden    int       // hidden-layer size (e.g. 32)
	mlpOutDim    int       // = 1 today; v1 carries the field for fwd-compat
	mlpBackbone  int       // sanity vs nnFeatDim
	mlpNLogo     int       // 0 or 1
	mlpNAudio    int       // 0 or 1
	mlpNChannel  int       // size of channel one-hot block
	mlpW1        []float32 // (mlpInDim, mlpHidden) row-major: W1[i*mlpHidden+j]
	mlpB1        []float32 // mlpHidden
	mlpW2        []float32 // (mlpHidden, mlpOutDim) row-major
	mlpB2        []float32 // mlpOutDim
	mlpChanMap   map[string]int // slug → idx; loaded from <head>.channel-map.json sidecar
	mlpChanIdx   int            // resolved channelSlug→mlpChanMap idx, or -1 (= unknown slug, fallback to all-zero one-hot)
}

// Backbone tensor shape: (1, 3, 224, 224). Output: (1, 1280).
//
// Six head formats are supported, distinguished at load time by the
// raw-bytes length of head.bin. Weights are little-endian float32
// packed back-to-back, followed by a single float32 bias. Order
// matches scripts/train-head.py's featurize_recording: backbone,
// then optional logo, channel, audio (in that order).
//
//   - LEGACY                   (5124 B): backbone only, 1280 weights.
//   - +LOGO                    (5128 B): backbone (1280) + logo (1).
//   - +CHAN                    (5148 B): backbone (1280) + chan (6).
//   - +LOGO+CHAN               (5152 B): backbone + logo + chan = 1287 weights.
//   - +LOGO+AUDIO              (5132 B): backbone + logo + audio = 1282 weights.
//   - +LOGO+CHAN+AUDIO         (5156 B): backbone + logo + chan + audio = 1288 weights.
//
// Audio (= per-second normalised RMS, [0,1]) is only ever combined
// WITH logo because a bare +AUDIO would collide on size with +LOGO.
// Detection of the format is purely by file size — keep these
// distinct.
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
	// Enable CoreML execution provider on Apple Silicon (silently no-op on
	// non-Mac builds). MLProgram format gets the most ops onto Apple Neural
	// Engine + GPU; CPU provider remains as fallback for ops the EP can't
	// execute. Disable via TVD_NO_COREML env if it ever causes issues.
	if os.Getenv("TVD_NO_COREML") == "" {
		coremlOpts := map[string]string{
			"ModelFormat":     "MLProgram",
			"MLComputeUnits":  "ALL",
		}
		// V2 takes a map[string]string; ignore error to fall through to
		// CPU on platforms without the CoreML provider compiled in.
		_ = opts.AppendExecutionProviderCoreMLV2(coremlOpts)
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
		headPath:    headPath,
		channelSlug: channelSlug,
		channelIdx:  chanIdx,
		mlpChanIdx:  -1,
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
	// MLP1 magic-prefix detection — if the head bytes start with the
	// MLP v1 magic, dispatch to the MLP loader (different layout +
	// requires the channel-map sidecar). Falls through to the legacy
	// LogReg size-detection path otherwise. Magic is the 4-byte
	// little-endian uint32 0x31504C4D = ASCII "MLP1".
	if len(raw) >= 4 && raw[0] == 'M' && raw[1] == 'L' && raw[2] == 'P' && raw[3] == '1' {
		return d.loadMLPHead(raw, mtime)
	}
	// Auto-detect head format by raw size. Four shapes possible —
	// see nnChannels comment block for the layout matrix.
	nC := len(nnChannels)
	legacyBytes        := (nnFeatDim + 1) * 4                 // 5124
	withLogoBytes      := (nnFeatDim + 1 + 1) * 4             // 5128
	withChanBytes      := (nnFeatDim + nC + 1) * 4            // 5148
	withLogoChanBytes  := (nnFeatDim + 1 + nC + 1) * 4        // 5152
	withLogoAudioBytes := (nnFeatDim + 1 + 1 + 1) * 4         // 5132
	withAllBytes       := (nnFeatDim + 1 + nC + 1 + 1) * 4    // 5156
	var headDim int
	var withLogo, withChan, withAudio bool
	switch len(raw) {
	case legacyBytes:
		headDim = nnFeatDim
	case withLogoBytes:
		headDim, withLogo = nnFeatDim+1, true
	case withChanBytes:
		headDim, withChan = nnFeatDim+nC, true
	case withLogoChanBytes:
		headDim, withLogo, withChan = nnFeatDim+1+nC, true, true
	case withLogoAudioBytes:
		headDim, withLogo, withAudio = nnFeatDim+1+1, true, true
	case withAllBytes:
		headDim, withLogo, withChan, withAudio = nnFeatDim+1+nC+1, true, true, true
	default:
		return fmt.Errorf("head file size %d, expected %d/%d/%d/%d/%d/%d "+
			"(legacy / +logo / +chan / +logo+chan / +logo+audio / +all)",
			len(raw), legacyBytes, withLogoBytes, withChanBytes,
			withLogoChanBytes, withLogoAudioBytes, withAllBytes)
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
	d.headWithAudio = withAudio
	// Clear any stale MLP state from a previous load — a head.bin
	// switch from MLP1 → LogReg (e.g. emergency rollback) would
	// otherwise leave the MLP fwd-pass branch active with the wrong
	// shape data.
	d.headIsMLP = false
	d.mlpW1 = nil
	d.mlpB1 = nil
	d.mlpW2 = nil
	d.mlpB2 = nil
	d.mlpChanMap = nil
	d.mu.Unlock()
	return nil
}

// loadMLPHead parses an "MLP1"-magic head.bin (= written by
// scripts/train-head.py write_mlp_head_v1). Layout is documented at
// the top of train-head.py. Also loads the channel-map sidecar
// alongside head.bin (= same dir, name "head.channel-map.json" or
// "<basename>.channel-map.json" if the head file isn't named head.bin).
// Resolves the recording's channel slug → mlpChanIdx for the
// inference path. An unknown slug becomes mlpChanIdx=-1 → all-zero
// channel one-hot at inference (graceful degradation to a channel-
// agnostic prediction; never fails the load).
func (d *NNDetector) loadMLPHead(raw []byte, mtime int64) error {
	const headerLen = 36
	if len(raw) < headerLen {
		return fmt.Errorf("MLP head truncated: %d B < %d B header",
			len(raw), headerLen)
	}
	// Header layout (9 × uint32 LE): magic, version, input_dim,
	// hidden_dim, output_dim, backbone_dim, n_logo, n_audio, n_channel.
	u32 := func(off int) uint32 {
		return uint32(raw[off]) | uint32(raw[off+1])<<8 |
			uint32(raw[off+2])<<16 | uint32(raw[off+3])<<24
	}
	if u32(0) != 0x31504C4D { // "MLP1" little-endian
		return fmt.Errorf("MLP head magic mismatch: got 0x%08x, want 0x31504C4D",
			u32(0))
	}
	version := u32(4)
	if version != 1 {
		return fmt.Errorf("MLP head version %d unsupported (this build reads v1)",
			version)
	}
	inDim := int(u32(8))
	hidden := int(u32(12))
	outDim := int(u32(16))
	backbone := int(u32(20))
	nLogo := int(u32(24))
	nAudio := int(u32(28))
	nChan := int(u32(32))
	if backbone != nnFeatDim {
		return fmt.Errorf("MLP head backbone_dim %d != nnFeatDim %d "+
			"(rebuild head against the current backbone)",
			backbone, nnFeatDim)
	}
	if backbone+nLogo+nAudio+nChan != inDim {
		return fmt.Errorf("MLP head input_dim %d inconsistent with "+
			"backbone %d + logo %d + audio %d + chan %d",
			inDim, backbone, nLogo, nAudio, nChan)
	}
	expected := headerLen + (inDim*hidden+hidden+hidden*outDim+outDim)*4
	if len(raw) != expected {
		return fmt.Errorf("MLP head size %d != expected %d (in=%d hid=%d out=%d)",
			len(raw), expected, inDim, hidden, outDim)
	}
	// Read weight blocks back-to-back (W1 row-major, b1, W2 row-major, b2).
	off := headerLen
	readFloats := func(n int) []float32 {
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = floatLE(raw[off+i*4:])
		}
		off += n * 4
		return out
	}
	W1 := readFloats(inDim * hidden)
	b1 := readFloats(hidden)
	W2 := readFloats(hidden * outDim)
	b2 := readFloats(outDim)
	// Channel-map sidecar — same dir as head.bin, name derived by
	// replacing the head file's extension with ".channel-map.json".
	// E.g.  head.bin → head.channel-map.json
	//       head.mlp32-channel.bin → head.mlp32-channel.channel-map.json
	chanMap, mlpChanIdx, err := d.loadChannelMap(nChan)
	if err != nil {
		return fmt.Errorf("MLP head: %w", err)
	}
	d.mu.Lock()
	d.headIsMLP = true
	d.headLoaded = true
	d.headMtime = mtime
	d.mlpInDim = inDim
	d.mlpHidden = hidden
	d.mlpOutDim = outDim
	d.mlpBackbone = backbone
	d.mlpNLogo = nLogo
	d.mlpNAudio = nAudio
	d.mlpNChannel = nChan
	d.mlpW1 = W1
	d.mlpB1 = b1
	d.mlpW2 = W2
	d.mlpB2 = b2
	d.mlpChanMap = chanMap
	d.mlpChanIdx = mlpChanIdx
	// Clear LogReg fields so the legacy fwd-path doesn't fire on
	// stale data if anything reads them (defensive — ConfidenceBatch
	// already branches on headIsMLP).
	d.headW = nil
	d.headBias = 0
	d.headWithLogo = false
	d.headWithChan = false
	d.headWithAudio = false
	d.mu.Unlock()
	return nil
}

// loadChannelMap reads <head_dir>/<head_basename>.channel-map.json
// and resolves the recording's channel slug to a one-hot index.
// nChan is the size the MLP head expects; the sidecar's slug list
// must be at least that long (= the head was trained with that many
// channel columns). When nChan==0 the head doesn't condition on
// channel; sidecar lookup is skipped + map is nil.
func (d *NNDetector) loadChannelMap(nChan int) (map[string]int, int, error) {
	if nChan == 0 {
		return nil, -1, nil
	}
	dir := filepath.Dir(d.headPath)
	base := filepath.Base(d.headPath)
	// Strip a single trailing .bin so head.bin → head, head.foo.bin → head.foo
	stem := strings.TrimSuffix(base, ".bin")
	sidecar := filepath.Join(dir, stem+".channel-map.json")
	raw, err := os.ReadFile(sidecar)
	if err != nil {
		return nil, -1, fmt.Errorf("channel-map sidecar missing at %s: %w",
			sidecar, err)
	}
	var sc struct {
		Version int      `json:"version"`
		N       int      `json:"n"`
		Slugs   []string `json:"slugs"`
	}
	if err := json.Unmarshal(raw, &sc); err != nil {
		return nil, -1, fmt.Errorf("channel-map sidecar parse: %w", err)
	}
	if sc.Version != 1 {
		return nil, -1, fmt.Errorf("channel-map sidecar version %d != 1",
			sc.Version)
	}
	if len(sc.Slugs) != nChan {
		return nil, -1, fmt.Errorf("channel-map slug count %d != head n_channel %d",
			len(sc.Slugs), nChan)
	}
	m := make(map[string]int, len(sc.Slugs))
	for i, s := range sc.Slugs {
		m[s] = i
	}
	idx := -1
	if d.channelSlug != "" {
		if i, ok := m[d.channelSlug]; ok {
			idx = i
		}
		// Unknown slug → idx=-1; inference uses all-zero one-hot
		// (= channel-agnostic fallback). Logged once per load so a
		// silent mis-config (= recording on a channel never trained
		// against) is at least visible.
		if idx < 0 {
			fmt.Fprintf(os.Stderr,
				"nn: MLP head loaded but recording's channel slug %q "+
					"not in sidecar (%d known slugs) — using zero one-hot\n",
				d.channelSlug, len(sc.Slugs))
		}
	}
	return m, idx, nil
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
func (d *NNDetector) Confidence(pixels []byte, logoConf, rmsConf float64) float64 {
	r := d.ConfidenceBatch([][]byte{pixels}, []float64{logoConf}, []float64{rmsConf})
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
func (d *NNDetector) ConfidenceBatch(framesPixels [][]byte, logoConfs, rmsConfs []float64) []float64 {
	n := len(framesPixels)
	if n == 0 {
		return nil
	}
	if n > nnBatch {
		// Caller error — split into multiple calls upstream.
		n = nnBatch
		framesPixels = framesPixels[:nnBatch]
		logoConfs = logoConfs[:nnBatch]
		if rmsConfs != nil {
			rmsConfs = rmsConfs[:nnBatch]
		}
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
	if d.headIsMLP {
		return d.confidenceMLP(feats, logoConfs, rmsConfs, n)
	}
	// Layout (matches train-head.py featurize_recording order):
	//   [0..1280)        backbone
	//   [1280]           logo  (if headWithLogo)
	//   [1280+(0|1) ..]  chan one-hot (if headWithChan, len=nC)
	//   [tail]           audio (if headWithAudio) — appended LAST
	chanBase := nnFeatDim
	if d.headWithLogo {
		chanBase = nnFeatDim + 1
	}
	audioIdx := -1
	if d.headWithAudio {
		audioIdx = nnFeatDim
		if d.headWithLogo {
			audioIdx++
		}
		if d.headWithChan {
			audioIdx += len(nnChannels)
		}
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
		if d.headWithAudio {
			// Caller may pass nil rmsConfs for legacy heads — treat
			// as neutral 0.5 so a missing audio stream doesn't bias
			// predictions toward show or ad.
			rms := 0.5
			if rmsConfs != nil && i < len(rmsConfs) {
				rms = rmsConfs[i]
			}
			logit += d.headW[audioIdx] * float32(rms)
		}
		out[i] = sigmoid(logit)
	}
	return out
}

// confidenceMLP runs the v1 MLP forward pass for n frames. Caller
// holds d.mu.RLock — do NOT take it again here.
//
//	hidden_j = ReLU( b1_j + Σ_k x_k * W1[k*H+j] )      for j in [0, H)
//	logit    = b2_0 + Σ_j hidden_j * W2[j*1+0]         (= H mul-adds; output_dim=1)
//	prob     = 1 / (1 + exp(-logit))
//
// Input vector layout (must mirror train-head.py
// featurize_recording, which the writer asserts):
//
//	[0..1280)              backbone
//	[1280..1280+nLogo)     logo conf (= 0 or 1 entry)
//	[+nAudio entries)      audio RMS (= 0 or 1 entry)
//	[+nChannel entries)    channel one-hot (size from sidecar)
//
// Note: the LogReg layout puts channel BEFORE audio, but the MLP
// header uses backbone+logo+audio+channel order to match what
// write_mlp_head_v1 expects (sklearn flattens columns in the order
// they were concat'd in the augment step). Stay consistent with the
// header's contract — backbone, logo, audio, channel.
func (d *NNDetector) confidenceMLP(feats []float32, logoConfs, rmsConfs []float64, n int) []float64 {
	out := make([]float64, n)
	x := make([]float32, d.mlpInDim)
	hidden := make([]float32, d.mlpHidden)
	chanOff := nnFeatDim + d.mlpNLogo + d.mlpNAudio
	for i := 0; i < n; i++ {
		// Build the input vector for frame i. Reuse x (= zeroed before
		// each frame so unused channel one-hot slots default to 0).
		copy(x[:nnFeatDim], feats[i*nnFeatDim:(i+1)*nnFeatDim])
		off := nnFeatDim
		if d.mlpNLogo > 0 {
			x[off] = float32(logoConfs[i])
			off++
		}
		if d.mlpNAudio > 0 {
			rms := 0.5 // neutral when stream is missing
			if rmsConfs != nil && i < len(rmsConfs) {
				rms = rmsConfs[i]
			}
			x[off] = float32(rms)
			off++
		}
		// Channel one-hot: zero the block, then set the resolved idx
		// (if any) to 1.0. Cheap because mlpNChannel ≤ ~32 in practice.
		if d.mlpNChannel > 0 {
			for k := 0; k < d.mlpNChannel; k++ {
				x[chanOff+k] = 0
			}
			if d.mlpChanIdx >= 0 {
				x[chanOff+d.mlpChanIdx] = 1.0
			}
		}
		// Hidden layer: hidden_j = ReLU(b1_j + Σ_k x_k * W1[k*H+j])
		copy(hidden, d.mlpB1)
		for k := 0; k < d.mlpInDim; k++ {
			xk := x[k]
			if xk == 0 {
				continue // sparse one-hot inputs short-circuit
			}
			rowOff := k * d.mlpHidden
			for j := 0; j < d.mlpHidden; j++ {
				hidden[j] += xk * d.mlpW1[rowOff+j]
			}
		}
		for j := 0; j < d.mlpHidden; j++ {
			if hidden[j] < 0 {
				hidden[j] = 0
			}
		}
		// Output layer (output_dim=1): logit = b2_0 + Σ_j hidden_j * W2[j].
		// W2 is row-major (H, O); for O=1 the row stride is 1.
		logit := d.mlpB2[0]
		for j := 0; j < d.mlpHidden; j++ {
			logit += hidden[j] * d.mlpW2[j]
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
