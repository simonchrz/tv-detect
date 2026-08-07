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
	session   *ort.AdvancedSession
	inTensor  *ort.Tensor[float32]
	outTensor *ort.Tensor[float32]
	frameW    int
	frameH    int
	headPath  string

	mu            sync.RWMutex
	headW         []float32 // 1280..1288 weights, depending on format
	headBias      float32
	headMtime     int64 // last mtime; reload when it changes
	headLoaded    bool
	headWithLogo  bool // true → weights[1280] = logo-conf coefficient
	headWithChan  bool // true → weights[1280+(0|1) .. +6) = channel one-hot
	headWithAudio bool // true → trailing weight = audio RMS (per-second, [0,1] normalised)

	channelSlug string // recording's channel slug, stored for MLP re-resolution after sidecar reload
	channelIdx  int    // index into nnChannels for our recording, or -1

	// MLP-head state (when headIsMLP=true; LogReg fields above are
	// unused). Loaded from a "MLP1" magic-prefixed head.bin as
	// specified in scripts/train-head.py write_mlp_head_v1.
	headIsMLP       bool
	mlpInDim        int // total input dim (= mlpBackbone + mlpNLogo + mlpNAudio + mlpNChannel + mlpNWhisper + mlpNTemporal)
	mlpHidden       int // hidden-layer size (e.g. 32)
	mlpOutDim       int // = 1 today; format carries the field for fwd-compat
	mlpBackbone     int // sanity vs nnFeatDim
	mlpNLogo        int // 0 or 1
	mlpNAudio       int // 0 or 1
	mlpNChannel     int // size of channel one-hot block
	mlpNWhisper     int // 0 or 1 — per-frame whisper-prob slot (v2+)
	mlpNTemporal    int // 0 or 2 — L2 distance to prev/next frame (v3+)
	mlpNMinutePrior int // 0 or 1 — P(ad | minute-of-hour) slot (v4+)
	// 0 or 1 — whisper-PRESENCE slot (v5+). 1.0 when this recording has
	// whisper data at all, 0.0 when it does not. Distinct from
	// mlpNWhisper, which carries the probability: without this column a
	// missing whisper feed is indistinguishable from "audio says 50/50",
	// and that was true for half the 2026-08 training corpus.
	mlpNWhisperMask int
	mlpW1           []float32      // (mlpInDim, mlpHidden) row-major: W1[i*mlpHidden+j]
	mlpB1           []float32      // mlpHidden
	mlpW2           []float32      // (mlpHidden, mlpOutDim) row-major
	mlpB2           []float32      // mlpOutDim
	mlpChanMap      map[string]int // slug → idx; loaded from <head>.channel-map.json sidecar
	mlpChanIdx      int            // resolved channelSlug→mlpChanMap idx, or -1 (= unknown slug, fallback to all-zero one-hot)
	// Per-recording whisper-prob array (length = recording duration in
	// seconds). Set ONCE at recording start via SetWhisperProbs from
	// the per-recording whisper.json. Indexed at inference by frame
	// index (= per-second granularity matches the train-head feature
	// extraction). Only used when mlpNWhisper > 0; nil for v1 heads
	// or when the daemon hasn't supplied whisper data.
	mlpWhisperProbs []float64
	// Minute-of-hour prior (v4 heads): the recording channel's 60-bucket
	// P(ad | minute_of_hour) histogram resolved from the
	// <head>.minute-prior.json sidecar at load, plus the sidecar's
	// corpus-wide neutral fill. Frame → minute via startTS (wall-clock
	// recording start, set via SetStartTS / --start-ts) + frame offset.
	// nil prior or startTS==0 → the neutral value (matches train-head.py
	// _minuteprior_col exactly).
	mlpMinutePrior []float32
	mlpMPNeutral   float32
	startTS        int64
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
	nnBatch   = 32   // frames per ONNX inference call. CoreML on M-series benefits from batched matmul (= 2026-05-04 A/B 8/16/32/64 found 32 optimal: -20 % wall vs 8, -40 % backbone-phase sum). Sub-batches are zero-padded. Pairs 1:1 with pipeline/parallel.go nnBatchSize.
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
			"ModelFormat":    "MLProgram",
			"MLComputeUnits": "ALL",
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
	// MLP magic-prefix detection. Four known formats:
	//   "MLP1" (v1) — backbone + logo + audio + channel one-hot
	//   "MLP2" (v2) — v1 + per-frame whisper-prob input slot
	//   "MLP3" (v3) — v2 + L2-distance-to-prev/next-frame input slots
	//   "MLP4" (v4) — v3 + minute-of-hour-prior input slot
	//   "MLP5" (v5) — v4 + whisper-PRESENCE slot
	// Each gets its own loader because the header layout differs
	// (v5 is 52 B, v4 48 B, v3 44 B, v2 40 B, v1 36 B). Falls through to the
	// legacy LogReg size-detection path when no magic matches.
	if len(raw) >= 4 && raw[0] == 'M' && raw[1] == 'L' && raw[2] == 'P' {
		switch raw[3] {
		case '1':
			return d.loadMLPHead(raw, mtime)
		case '2':
			return d.loadMLPHeadV2(raw, mtime)
		case '3':
			return d.loadMLPHeadV3(raw, mtime)
		case '4':
			return d.loadMLPHeadV4(raw, mtime)
		case '5':
			return d.loadMLPHeadV5(raw, mtime)
		}
		// Unknown MLPx version → fall through to LogReg size
		// detection, which will fail with a clean error rather
		// than silently mis-parsing future-version weights.
	}
	// Auto-detect head format by raw size. Four shapes possible —
	// see nnChannels comment block for the layout matrix.
	nC := len(nnChannels)
	legacyBytes := (nnFeatDim + 1) * 4                // 5124
	withLogoBytes := (nnFeatDim + 1 + 1) * 4          // 5128
	withChanBytes := (nnFeatDim + nC + 1) * 4         // 5148
	withLogoChanBytes := (nnFeatDim + 1 + nC + 1) * 4 // 5152
	withLogoAudioBytes := (nnFeatDim + 1 + 1 + 1) * 4 // 5132
	withAllBytes := (nnFeatDim + 1 + nC + 1 + 1) * 4  // 5156
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
	d.mlpNWhisper = 0
	d.mlpNTemporal = 0
	d.mlpNMinutePrior = 0
	d.mlpNWhisperMask = 0
	d.mlpMinutePrior = nil
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
	// v1 → no whisper/temporal slots.
	d.mlpNWhisper = 0
	d.mlpNTemporal = 0
	d.mlpNMinutePrior = 0
	d.mlpNWhisperMask = 0
	d.mlpMinutePrior = nil
	d.mu.Unlock()
	return nil
}

// loadMLPHeadV2 parses an "MLP2"-magic head.bin (= written by
// scripts/train-head.py write_mlp_head_v2). Identical layout to v1
// except the header carries a 10th uint32 LE field `n_whisper`
// (0 or 1) at offset 36..39, growing the header from 36 to 40 bytes.
// input_dim must equal backbone+logo+audio+channel+whisper.
//
// At inference, the per-frame whisper-prob is appended LAST in the
// input vector (= AFTER the channel one-hot block), so chan-one-hot
// indices stay aligned with the channel-map sidecar from v1.
func (d *NNDetector) loadMLPHeadV2(raw []byte, mtime int64) error {
	const headerLen = 40
	if len(raw) < headerLen {
		return fmt.Errorf("MLP2 head truncated: %d B < %d B header",
			len(raw), headerLen)
	}
	u32 := func(off int) uint32 {
		return uint32(raw[off]) | uint32(raw[off+1])<<8 |
			uint32(raw[off+2])<<16 | uint32(raw[off+3])<<24
	}
	if u32(0) != 0x32504C4D {
		return fmt.Errorf("MLP2 head magic mismatch: got 0x%08x, want 0x32504C4D",
			u32(0))
	}
	version := u32(4)
	if version != 2 {
		return fmt.Errorf("MLP2 head version %d unsupported (this build reads v2)",
			version)
	}
	inDim := int(u32(8))
	hidden := int(u32(12))
	outDim := int(u32(16))
	backbone := int(u32(20))
	nLogo := int(u32(24))
	nAudio := int(u32(28))
	nChan := int(u32(32))
	nWhisper := int(u32(36))
	if backbone != nnFeatDim {
		return fmt.Errorf("MLP2 head backbone_dim %d != nnFeatDim %d "+
			"(rebuild head against the current backbone)",
			backbone, nnFeatDim)
	}
	if backbone+nLogo+nAudio+nChan+nWhisper != inDim {
		return fmt.Errorf("MLP2 head input_dim %d inconsistent with "+
			"backbone %d + logo %d + audio %d + chan %d + whisper %d",
			inDim, backbone, nLogo, nAudio, nChan, nWhisper)
	}
	expected := headerLen + (inDim*hidden+hidden+hidden*outDim+outDim)*4
	if len(raw) != expected {
		return fmt.Errorf("MLP2 head size %d != expected %d (in=%d hid=%d out=%d)",
			len(raw), expected, inDim, hidden, outDim)
	}
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
	chanMap, mlpChanIdx, err := d.loadChannelMap(nChan)
	if err != nil {
		return fmt.Errorf("MLP2 head: %w", err)
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
	d.mlpNWhisper = nWhisper
	// v2 → no temporal slots.
	d.mlpNTemporal = 0
	d.mlpW1 = W1
	d.mlpB1 = b1
	d.mlpW2 = W2
	d.mlpB2 = b2
	d.mlpChanMap = chanMap
	d.mlpChanIdx = mlpChanIdx
	d.headW = nil
	d.headBias = 0
	d.headWithLogo = false
	d.headWithChan = false
	d.headWithAudio = false
	d.mu.Unlock()
	return nil
}

// loadMLPHeadV3 parses an "MLP3"-magic head.bin (= written by
// scripts/train-head.py write_mlp_head_v3). Identical layout to v2
// except the header carries an 11th uint32 LE field `n_temporal`
// (0 or 2) at offset 40..43, growing the header from 40 to 44 bytes.
// input_dim must equal backbone+logo+audio+channel+whisper+temporal.
//
// At inference, the 2 temporal columns are appended LAST in the input
// vector (= AFTER whisper), so chan-one-hot and whisper indices stay
// aligned with v1/v2. See confidenceMLP for the matching computation
// (L2 distance to the previous/next frame's [backbone+logo+audio]
// vector — MUST mirror scripts/train-head.py's
// _augment_channel_whisper_temporal exactly, or the trained weights
// score against a differently-distributed input).
func (d *NNDetector) loadMLPHeadV3(raw []byte, mtime int64) error {
	const headerLen = 44
	if len(raw) < headerLen {
		return fmt.Errorf("MLP3 head truncated: %d B < %d B header",
			len(raw), headerLen)
	}
	u32 := func(off int) uint32 {
		return uint32(raw[off]) | uint32(raw[off+1])<<8 |
			uint32(raw[off+2])<<16 | uint32(raw[off+3])<<24
	}
	if u32(0) != 0x33504C4D {
		return fmt.Errorf("MLP3 head magic mismatch: got 0x%08x, want 0x33504C4D",
			u32(0))
	}
	version := u32(4)
	if version != 3 {
		return fmt.Errorf("MLP3 head version %d unsupported (this build reads v3)",
			version)
	}
	inDim := int(u32(8))
	hidden := int(u32(12))
	outDim := int(u32(16))
	backbone := int(u32(20))
	nLogo := int(u32(24))
	nAudio := int(u32(28))
	nChan := int(u32(32))
	nWhisper := int(u32(36))
	nTemporal := int(u32(40))
	if backbone != nnFeatDim {
		return fmt.Errorf("MLP3 head backbone_dim %d != nnFeatDim %d "+
			"(rebuild head against the current backbone)",
			backbone, nnFeatDim)
	}
	if backbone+nLogo+nAudio+nChan+nWhisper+nTemporal != inDim {
		return fmt.Errorf("MLP3 head input_dim %d inconsistent with "+
			"backbone %d + logo %d + audio %d + chan %d + whisper %d + "+
			"temporal %d",
			inDim, backbone, nLogo, nAudio, nChan, nWhisper, nTemporal)
	}
	expected := headerLen + (inDim*hidden+hidden+hidden*outDim+outDim)*4
	if len(raw) != expected {
		return fmt.Errorf("MLP3 head size %d != expected %d (in=%d hid=%d out=%d)",
			len(raw), expected, inDim, hidden, outDim)
	}
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
	chanMap, mlpChanIdx, err := d.loadChannelMap(nChan)
	if err != nil {
		return fmt.Errorf("MLP3 head: %w", err)
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
	d.mlpNWhisper = nWhisper
	d.mlpNTemporal = nTemporal
	d.mlpNMinutePrior = 0
	d.mlpNWhisperMask = 0
	d.mlpMinutePrior = nil
	d.mlpW1 = W1
	d.mlpB1 = b1
	d.mlpW2 = W2
	d.mlpB2 = b2
	d.mlpChanMap = chanMap
	d.mlpChanIdx = mlpChanIdx
	d.headW = nil
	d.headBias = 0
	d.headWithLogo = false
	d.headWithChan = false
	d.headWithAudio = false
	d.mu.Unlock()
	return nil
}

// loadMLPHeadV4 parses an "MLP4"-magic head.bin (= written by
// scripts/train-head.py write_mlp_head_v4). Identical layout to v3
// except the header carries a 12th uint32 LE field `n_minuteprior`
// (0 or 1) at offset 44..47, growing the header from 44 to 48 bytes.
// input_dim must equal backbone+logo+audio+channel+whisper+temporal
// +minuteprior.
//
// At inference the minute-prior column is appended LAST (= AFTER the
// temporal deltas), so the v1/v2/v3 column order stays a prefix. The
// per-channel P(ad | minute-of-hour) histogram comes from the
// <head>.minute-prior.json sidecar (resolved for channelSlug at load);
// frame → minute via startTS + frame offset — MUST mirror
// scripts/train-head.py's _minuteprior_col exactly.
func (d *NNDetector) loadMLPHeadV4(raw []byte, mtime int64) error {
	const headerLen = 48
	if len(raw) < headerLen {
		return fmt.Errorf("MLP4 head truncated: %d B < %d B header",
			len(raw), headerLen)
	}
	u32 := func(off int) uint32 {
		return uint32(raw[off]) | uint32(raw[off+1])<<8 |
			uint32(raw[off+2])<<16 | uint32(raw[off+3])<<24
	}
	if u32(0) != 0x34504C4D {
		return fmt.Errorf("MLP4 head magic mismatch: got 0x%08x, want 0x34504C4D",
			u32(0))
	}
	version := u32(4)
	if version != 4 {
		return fmt.Errorf("MLP4 head version %d unsupported (this build reads v4)",
			version)
	}
	inDim := int(u32(8))
	hidden := int(u32(12))
	outDim := int(u32(16))
	backbone := int(u32(20))
	nLogo := int(u32(24))
	nAudio := int(u32(28))
	nChan := int(u32(32))
	nWhisper := int(u32(36))
	nTemporal := int(u32(40))
	nMinutePrior := int(u32(44))
	if backbone != nnFeatDim {
		return fmt.Errorf("MLP4 head backbone_dim %d != nnFeatDim %d "+
			"(rebuild head against the current backbone)",
			backbone, nnFeatDim)
	}
	if backbone+nLogo+nAudio+nChan+nWhisper+nTemporal+nMinutePrior != inDim {
		return fmt.Errorf("MLP4 head input_dim %d inconsistent with "+
			"backbone %d + logo %d + audio %d + chan %d + whisper %d + "+
			"temporal %d + minuteprior %d",
			inDim, backbone, nLogo, nAudio, nChan, nWhisper, nTemporal,
			nMinutePrior)
	}
	expected := headerLen + (inDim*hidden+hidden+hidden*outDim+outDim)*4
	if len(raw) != expected {
		return fmt.Errorf("MLP4 head size %d != expected %d (in=%d hid=%d out=%d)",
			len(raw), expected, inDim, hidden, outDim)
	}
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
	chanMap, mlpChanIdx, err := d.loadChannelMap(nChan)
	if err != nil {
		return fmt.Errorf("MLP4 head: %w", err)
	}
	var mpPrior []float32
	mpNeutral := float32(0.25)
	if nMinutePrior > 0 {
		mpPrior, mpNeutral, err = d.loadMinutePrior()
		if err != nil {
			return fmt.Errorf("MLP4 head: %w", err)
		}
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
	d.mlpNWhisper = nWhisper
	d.mlpNTemporal = nTemporal
	d.mlpNMinutePrior = nMinutePrior
	d.mlpNWhisperMask = 0
	d.mlpMinutePrior = mpPrior
	d.mlpMPNeutral = mpNeutral
	d.mlpW1 = W1
	d.mlpB1 = b1
	d.mlpW2 = W2
	d.mlpB2 = b2
	d.mlpChanMap = chanMap
	d.mlpChanIdx = mlpChanIdx
	d.headW = nil
	d.headBias = 0
	d.headWithLogo = false
	d.headWithChan = false
	d.headWithAudio = false
	d.mu.Unlock()
	return nil
}

// loadMLPHeadV5 parses an "MLP5"-magic head.bin (= written by
// scripts/train-head.py write_mlp_head_v5). Identical layout to v4
// except the header carries a 13th uint32 LE field `n_whispermask`
// (0 or 1) at offset 48..51, growing the header from 48 to 52 bytes.
// input_dim must equal backbone+logo+audio+channel+whisper+temporal
// +minuteprior+whispermask.
//
// At inference the whisper-mask column is appended LAST (= AFTER the
// minute prior), so the v1..v4 column order stays a prefix.
//
// WHY the column exists. The whisper PROBABILITY column falls back to a
// neutral 0.5 when a recording has no whisper data, which makes "no
// audio evidence" and "audio says 50/50" the same input. On 2026-08-06
// that was 300 of 557 archived recordings — a June collapse in whisper
// coverage whose sources are long deleted, so it cannot be backfilled.
// The head weights the whisper column heavily (first-layer norm 15.5,
// top percentile) and recordings without it scored 0.049 IoU lower
// (p=0.034); toggling the feed on one recording moved it 0.341 -> 0.935.
// The mask lets the head tell the two apart instead of learning to
// half-trust a column that is real 93% of the time in production.
//
// MUST mirror scripts/train-head.py's _whisper_present exactly: 1.0 iff
// whisper data exists for the recording, 0.0 otherwise. On this side
// that is "the daemon passed --nn-whisper-json and it parsed", i.e.
// d.mlpWhisperProbs != nil.
func (d *NNDetector) loadMLPHeadV5(raw []byte, mtime int64) error {
	const headerLen = 52
	if len(raw) < headerLen {
		return fmt.Errorf("MLP5 head truncated: %d B < %d B header",
			len(raw), headerLen)
	}
	u32 := func(off int) uint32 {
		return uint32(raw[off]) | uint32(raw[off+1])<<8 |
			uint32(raw[off+2])<<16 | uint32(raw[off+3])<<24
	}
	if u32(0) != 0x35504C4D {
		return fmt.Errorf("MLP5 head magic mismatch: got 0x%08x, want 0x35504C4D",
			u32(0))
	}
	if version := u32(4); version != 5 {
		return fmt.Errorf("MLP5 head version %d unsupported (this build reads v5)",
			version)
	}
	inDim := int(u32(8))
	hidden := int(u32(12))
	outDim := int(u32(16))
	backbone := int(u32(20))
	nLogo := int(u32(24))
	nAudio := int(u32(28))
	nChan := int(u32(32))
	nWhisper := int(u32(36))
	nTemporal := int(u32(40))
	nMinutePrior := int(u32(44))
	nWhisperMask := int(u32(48))
	if backbone != nnFeatDim {
		return fmt.Errorf("MLP5 head backbone_dim %d != nnFeatDim %d "+
			"(rebuild head against the current backbone)",
			backbone, nnFeatDim)
	}
	if backbone+nLogo+nAudio+nChan+nWhisper+nTemporal+nMinutePrior+
		nWhisperMask != inDim {
		return fmt.Errorf("MLP5 head input_dim %d inconsistent with "+
			"backbone %d + logo %d + audio %d + chan %d + whisper %d + "+
			"temporal %d + minuteprior %d + whispermask %d",
			inDim, backbone, nLogo, nAudio, nChan, nWhisper, nTemporal,
			nMinutePrior, nWhisperMask)
	}
	expected := headerLen + (inDim*hidden+hidden+hidden*outDim+outDim)*4
	if len(raw) != expected {
		return fmt.Errorf("MLP5 head size %d != expected %d (in=%d hid=%d out=%d)",
			len(raw), expected, inDim, hidden, outDim)
	}
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
	chanMap, mlpChanIdx, err := d.loadChannelMap(nChan)
	if err != nil {
		return fmt.Errorf("MLP5 head: %w", err)
	}
	var mpPrior []float32
	mpNeutral := float32(0.25)
	if nMinutePrior > 0 {
		mpPrior, mpNeutral, err = d.loadMinutePrior()
		if err != nil {
			return fmt.Errorf("MLP5 head: %w", err)
		}
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
	d.mlpNWhisper = nWhisper
	d.mlpNTemporal = nTemporal
	d.mlpNMinutePrior = nMinutePrior
	d.mlpNWhisperMask = nWhisperMask
	d.mlpMinutePrior = mpPrior
	d.mlpMPNeutral = mpNeutral
	d.mlpW1 = W1
	d.mlpB1 = b1
	d.mlpW2 = W2
	d.mlpB2 = b2
	d.mlpChanMap = chanMap
	d.mlpChanIdx = mlpChanIdx
	d.headW = nil
	d.headBias = 0
	d.headWithLogo = false
	d.headWithChan = false
	d.headWithAudio = false
	d.mu.Unlock()
	return nil
}

// loadMinutePrior reads <head_dir>/<head_basename>.minute-prior.json
// and resolves the recording's channel slug to its 60-bucket
// P(ad | minute-of-hour) histogram. A missing slug is NOT an error —
// inference falls back to the sidecar's corpus-wide neutral value
// (same graceful degradation as an unknown channel one-hot). A
// missing/corrupt sidecar IS an error: a v4 head without its prior
// table would silently score against a differently-distributed input.
func (d *NNDetector) loadMinutePrior() ([]float32, float32, error) {
	dir := filepath.Dir(d.headPath)
	stem := strings.TrimSuffix(filepath.Base(d.headPath), ".bin")
	sidecar := filepath.Join(dir, stem+".minute-prior.json")
	raw, err := os.ReadFile(sidecar)
	if err != nil {
		return nil, 0, fmt.Errorf("minute-prior sidecar missing at %s: %w",
			sidecar, err)
	}
	var sc struct {
		Version int                  `json:"version"`
		Neutral float64              `json:"neutral"`
		Priors  map[string][]float64 `json:"priors"`
	}
	if err := json.Unmarshal(raw, &sc); err != nil {
		return nil, 0, fmt.Errorf("minute-prior sidecar parse: %w", err)
	}
	if sc.Version != 1 {
		return nil, 0, fmt.Errorf("minute-prior sidecar version %d != 1",
			sc.Version)
	}
	neutral := float32(sc.Neutral)
	if arr, ok := sc.Priors[d.channelSlug]; ok && len(arr) == 60 {
		prior := make([]float32, 60)
		for i, v := range arr {
			prior[i] = float32(v)
		}
		return prior, neutral, nil
	}
	if d.channelSlug != "" {
		fmt.Fprintf(os.Stderr,
			"nn: MLP4+ head loaded but channel slug %q has no minute-prior "+
				"histogram (%d known) — using neutral %.3f\n",
			d.channelSlug, len(sc.Priors), neutral)
	}
	return nil, neutral, nil
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

// EmbedBatch runs ONLY the backbone on up to nnBatch frames and returns a
// COPY of the embeddings (n * nnFeatDim float32, frame-major). Part of the
// two-phase inference split (2026-07-18): the head pass moved to
// ConfidenceChunk so time-dependent input columns (whisper, temporal
// deltas) can be built with correct ABSOLUTE timing over a whole chunk —
// inside a 32-frame batch neither is computable correctly (root cause of
// the production NN degradation found 2026-07-18: whisper indexed
// batch-locally = first 32 s repeated per batch, temporal deltas at
// consecutive-25fps-frame scale ≈ 25x smaller than the 1 s-spacing the
// head was trained on, zeroed at every batch edge).
// Returns nil on inference failure (caller substitutes neutral 0.5s).
func (d *NNDetector) EmbedBatch(framesPixels [][]byte) []float32 {
	n := len(framesPixels)
	if n == 0 {
		return nil
	}
	if n > nnBatch {
		n = nnBatch
		framesPixels = framesPixels[:nnBatch]
	}
	in := d.inTensor.GetData()
	stride := 3 * nnInputH * nnInputW
	for i := 0; i < n; i++ {
		preprocess(framesPixels[i], d.frameW, d.frameH,
			in[i*stride:(i+1)*stride])
	}
	for i := n; i < nnBatch; i++ {
		clear(in[i*stride : (i+1)*stride])
	}
	if err := d.session.Run(); err != nil {
		return nil
	}
	out := make([]float32, n*nnFeatDim)
	copy(out, d.outTensor.GetData()[:n*nnFeatDim])
	return out
}

// ConfidenceChunk runs the head pass over a whole chunk's embeddings with
// correctly-timed auxiliary inputs. embeds is frame-major (n * nnFeatDim,
// as returned by EmbedBatch calls concatenated in order); fps is the
// recording frame rate; chunkStartS the chunk's absolute start offset in
// the recording (for whisper's per-second indexing). A nil embeds or
// n == 0 returns all-neutral.
//
// Timing semantics (MUST mirror scripts/train-head.py, which trains on
// 1 fps rows):
//   - whisper column: whisperProbs[int(chunkStartS + i/fps)] — per-second
//     array indexed by the frame's absolute wall-clock second (same
//     mapping parallel.go uses for audio RMS).
//   - temporal deltas: dp[i] = ||base(i) - base(i-step)||, dn[i] =
//     ||base(i+step) - base(i)|| with step = round(fps) — i.e. the frame
//     ONE SECOND away, matching the 1 s row spacing of training's
//     _augment_channel_whisper_temporal. Frames within step of the chunk
//     edges get 0 (training zeroes recording edges; a chunked run zeroes
//     ~step frames per chunk boundary — ~2 % of frames at 18 chunks,
//     bounded and directionless).
func (d *NNDetector) ConfidenceChunk(embeds []float32, logoConfs, rmsConfs []float64, n int, fps, chunkStartS float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = 0.5
	}
	if n == 0 || embeds == nil || len(embeds) < n*nnFeatDim {
		return out
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.headLoaded {
		return out
	}
	if d.headIsMLP {
		return d.confidenceMLPChunk(embeds, logoConfs, rmsConfs, n, fps, chunkStartS)
	}
	// LogReg path — per-frame, no time-dependent inputs. Same math as
	// the historical ConfidenceBatch LogReg branch.
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
			logit += d.headW[j] * embeds[off+j]
		}
		if d.headWithLogo {
			logit += d.headW[nnFeatDim] * float32(logoConfs[i])
		}
		if d.headWithChan && d.channelIdx >= 0 {
			logit += d.headW[chanBase+d.channelIdx]
		}
		if d.headWithAudio {
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

// ConfidenceBatch runs backbone + head on up to nnBatch frames. Retained
// for callers WITHOUT chunk-timing context (single-frame Confidence, the
// nn-smoke tool): frames are treated as 1 s apart (fps=1, offset 0) —
// the training-row semantic. The production pipeline does NOT use this;
// it collects EmbedBatch results per chunk and calls ConfidenceChunk
// with real fps/offset so whisper + temporal columns are timed correctly.
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
	embeds := d.EmbedBatch(framesPixels)
	if embeds == nil {
		out := make([]float64, n)
		for i := range out {
			out[i] = 0.5
		}
		return out
	}
	return d.ConfidenceChunk(embeds, logoConfs, rmsConfs, n, 1, 0)
}

// churnWindowS ist die Fensterbreite (in Sekunden) der Unruhe-Spalte.
// Muss dem Vorgabewert von _churn_col in train-head.py entsprechen —
// eine Abweichung hier ist ein stiller Train/Serve-Bruch, kein Fehler.
//
// 2026-08-06 zunaechst 31, aus Sorge um die Chunk-Grenzen: der Kopf laeuft
// chunkweise, ein Fenster sieht die Nachbarschaft nicht, und bei 4 Chunks
// waeren das ~8 % der Frames bei 61 s. Diese Sorge war UNBEGRUENDET —
// 2026-08-07 nachgemessen (dieselben Aufnahmen, Unruhe einmal global und
// einmal chunkweise gerechnet, gleicher Kopf, gleicher Decoder):
//
//	Fenster   global   4 Chunks    Delta   betroffen
//	  31 s     0.936     0.936    -0.000      0/37
//	  61 s     0.931     0.930    -0.000      1/37
//	 121 s     0.934     0.934    +0.000      0/37
//
// Die Unruhe ist eine geglaettete Groesse, und der Dauer-Prior des
// Decoders buegelt lokale Stoerungen weg. Deshalb jetzt 61 — dort trennt
// das Merkmal deutlich besser (Rang-AUC 0.932 gegen 0.880).
//
// ⚠️ Die Breite aendert input_dim NICHT. Das ist Absicht: so bleibt das
// naechtliche Head-to-Head funktionsfaehig und kann die Aenderung sofort
// beurteilen, statt wie bei einem Spaltenzuwachs auszufallen.
const churnWindowS = 61

// confidenceMLPChunk is the chunk-scope MLP forward pass with correctly-
// timed auxiliary inputs (see ConfidenceChunk for the timing contract).
// Caller holds d.mu.RLock. Replaces the batch-scope confidenceMLP, whose
// batch-local indexing was the 2026-07-18 root cause (whisper read the
// first 32 seconds for every batch; temporal deltas were consecutive-
// frame-scale and zeroed at every 32-frame boundary).
func (d *NNDetector) confidenceMLPChunk(embeds []float32, logoConfs, rmsConfs []float64, n int, fps, chunkStartS float64) []float64 {
	out := make([]float64, n)
	x := make([]float32, d.mlpInDim)
	hidden := make([]float32, d.mlpHidden)
	chanOff := nnFeatDim + d.mlpNLogo + d.mlpNAudio
	whisperOff := chanOff + d.mlpNChannel
	temporalOff := whisperOff + d.mlpNWhisper
	minutePriorOff := temporalOff + d.mlpNTemporal
	whisperMaskOff := minutePriorOff + d.mlpNMinutePrior
	whisperPerSec := d.mlpWhisperProbs
	// v5: 1.0 wenn fuer diese Aufnahme ueberhaupt Whisper-Daten
	// vorliegen. Konstant ueber die ganze Aufnahme — der Daemon
	// uebergibt --nn-whisper-json entweder oder nicht. Muss
	// _whisper_present in train-head.py entsprechen (Datei existiert),
	// sonst sieht der Kopf im Betrieb eine andere Spalte als im
	// Training.
	whisperMask := float32(0)
	if whisperPerSec != nil {
		whisperMask = 1
	}
	if fps <= 0 {
		fps = 1
	}
	step := int(fps + 0.5)
	if step < 1 {
		step = 1
	}
	// Temporal deltas at 1-second spacing over the WHOLE chunk. Base
	// vector per frame = [backbone, logo?, audio?] — mirrors training's
	// X rows (train-head.py _augment_channel_whisper_temporal computes
	// np.linalg.norm(X[t] - X[t-1]) on 1 fps rows of exactly that
	// layout).
	var dpTemporal, dnTemporal, churnTemporal []float32
	if d.mlpNTemporal > 0 {
		baseDim := nnFeatDim + d.mlpNLogo + d.mlpNAudio
		base := make([]float32, n*baseDim)
		for i := 0; i < n; i++ {
			copy(base[i*baseDim:], embeds[i*nnFeatDim:(i+1)*nnFeatDim])
			off := i*baseDim + nnFeatDim
			if d.mlpNLogo > 0 {
				base[off] = float32(logoConfs[i])
				off++
			}
			if d.mlpNAudio > 0 {
				rms := 0.5
				if rmsConfs != nil && i < len(rmsConfs) {
					rms = rmsConfs[i]
				}
				base[off] = float32(rms)
			}
		}
		dpTemporal = make([]float32, n)
		dnTemporal = make([]float32, n)
		for i := step; i < n; i++ {
			a := base[i*baseDim : (i+1)*baseDim]
			b := base[(i-step)*baseDim : (i-step+1)*baseDim]
			var sumSq float32
			for k := 0; k < baseDim; k++ {
				diff := a[k] - b[k]
				sumSq += diff * diff
			}
			dist := float32(math.Sqrt(float64(sumSq)))
			dpTemporal[i] = dist
			dnTemporal[i-step] = dist
		}
		// Dritte Spalte (v5, 2026-08-06): das UNRUHE-NIVEAU — derselbe
		// 1s-Abstand, aber ueber churnWindowS Sekunden gemittelt.
		//
		// Warum: der Einzelsprung ist verrauscht. Rang-AUC gegen die
		// Labels ueber den Golden-Satz: 1s-Delta 0.637, ueber 31 s
		// gemittelt 0.880. Und es ist NEUE Information — Korrelation mit
		// der Kopf-Ausgabe nur 0.625, und dort wo der Kopf unsicher ist
		// (0.2<p<0.8) trennt es noch mit 0.784.
		//
		// MUSS _churn_col in train-head.py spiegeln: Mittel ueber 31
		// Werte im 1-SEKUNDEN-Abstand, am Rand auf die tatsaechlich
		// vorhandenen normiert (NICHT mit Nullen auffuellen — das zoege
		// die Unruhe am Rand nach unten und saehe wie "Sendung" aus,
		// also eine gerichtete Verzerrung statt nur mehr Rauschen).
		//
		// ⚠️ Der Kopf laeuft CHUNKWEISE, das Fenster sieht die
		// Nachbarschaft ueber die Chunk-Grenze hinweg nicht. Bei 31 s und
		// 4 Chunks (Produktion, DETECT_PARALLEL=3) betrifft das ~4 % der
		// Frames — dieselbe Groessenordnung wie die ~2 %, die die
		// 1s-Deltas schon an den Chunk-Raendern kosten. Deshalb 31 und
		// nicht 61 (dort 8 %) oder 181 (ueber 20 %), obwohl die breiteren
		// Fenster hoeher messen.
		if d.mlpNTemporal >= 3 {
			churnTemporal = make([]float32, n)
			const halb = churnWindowS / 2
			for i := 0; i < n; i++ {
				var sum float32
				var cnt float32
				for k := -halb; k <= halb; k++ {
					j := i + k*step
					if j < 0 || j >= n {
						continue
					}
					sum += dpTemporal[j]
					cnt++
				}
				if cnt > 0 {
					churnTemporal[i] = sum / cnt
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		copy(x[:nnFeatDim], embeds[i*nnFeatDim:(i+1)*nnFeatDim])
		off := nnFeatDim
		if d.mlpNLogo > 0 {
			x[off] = float32(logoConfs[i])
			off++
		}
		if d.mlpNAudio > 0 {
			rms := 0.5
			if rmsConfs != nil && i < len(rmsConfs) {
				rms = rmsConfs[i]
			}
			x[off] = float32(rms)
			off++
		}
		if d.mlpNChannel > 0 {
			for k := 0; k < d.mlpNChannel; k++ {
				x[chanOff+k] = 0
			}
			if d.mlpChanIdx >= 0 {
				x[chanOff+d.mlpChanIdx] = 1.0
			}
		}
		// Whisper: per-second array indexed by the frame's ABSOLUTE
		// wall-clock second (chunk offset + frame-local time) — the
		// same mapping parallel.go applies to audio RMS.
		if d.mlpNWhisper > 0 {
			wp := float32(0.5)
			if whisperPerSec != nil {
				absSec := int(chunkStartS + float64(i)/fps)
				if absSec >= 0 && absSec < len(whisperPerSec) {
					wp = float32(whisperPerSec[absSec])
				}
			}
			x[whisperOff] = wp
		}
		if d.mlpNTemporal > 0 {
			x[temporalOff] = dpTemporal[i]
			x[temporalOff+1] = dnTemporal[i]
			if d.mlpNTemporal >= 3 && churnTemporal != nil {
				x[temporalOff+2] = churnTemporal[i]
			}
		}
		// Minute-of-hour prior (v4): wall-clock minute of this frame =
		// recording start + absolute in-recording offset. Mirrors
		// train-head.py _minuteprior_col: (start + t) // 60 % 60. No
		// startTS or no per-channel histogram → the sidecar's neutral.
		if d.mlpNMinutePrior > 0 {
			mp := d.mlpMPNeutral
			if d.startTS > 0 && d.mlpMinutePrior != nil {
				absS := float64(d.startTS) + chunkStartS + float64(i)/fps
				minute := int(absS/60) % 60
				if minute >= 0 && minute < 60 {
					mp = d.mlpMinutePrior[minute]
				}
			}
			x[minutePriorOff] = mp
		}
		if d.mlpNWhisperMask > 0 {
			x[whisperMaskOff] = whisperMask
		}
		copy(hidden, d.mlpB1)
		for k := 0; k < d.mlpInDim; k++ {
			xk := x[k]
			if xk == 0 {
				continue
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
		logit := d.mlpB2[0]
		for j := 0; j < d.mlpHidden; j++ {
			logit += hidden[j] * d.mlpW2[j]
		}
		out[i] = sigmoid(logit)
	}
	return out
}

// SetWhisperProbs supplies the per-second whisper-prob array for
// the current recording. Indexed by ABSOLUTE wall-clock second in
// ConfidenceChunk (per-second granularity matches train-head
// extraction). Only consumed when the loaded head is MLP2/MLP3-format
// with n_whisper>0; other formats ignore the call. Pass nil to clear
// (= back to neutral 0.5 fallback).
// SetStartTS supplies the recording's wall-clock start (unix seconds,
// = the DVR grid's start_real; the daemon passes it via --start-ts).
// Only consumed when the loaded head is MLP4-format with
// n_minuteprior>0; other formats ignore it. 0 (unset) → the neutral
// fill, matching train-head.py's missing-start_ts fallback.
func (d *NNDetector) SetStartTS(ts int64) {
	d.mu.Lock()
	d.startTS = ts
	d.mu.Unlock()
}

func (d *NNDetector) SetWhisperProbs(probs []float64) {
	d.mu.Lock()
	d.mlpWhisperProbs = probs
	d.mu.Unlock()
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
