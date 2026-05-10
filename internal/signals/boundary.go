package signals

import (
	"fmt"
	"os"
	"sync"
)

// BoundaryDetector scores each frame on "is this a commercial-block
// boundary?" using a small MLP fed with a 3-frame temporal context
// of MobileNetV2 embeddings. Orthogonal to NNDetector (which scores
// "is this frame ad/show?") — boundaries are events at the transition
// between ad and show, while NNDetector classifies the steady state.
//
// Architecture rationale: the user-facing failure mode that dominates
// is missed_bumper — i.e. the rough block position is right but the
// precise start/end frame is off. A single per-frame ad/show score
// has no notion of "this frame looks like an opening bumper" vs "this
// frame is deep in an ad break"; both score 1. The boundary head
// learns the visual derivative across a ±1s window, which is the
// signal the comskip-era heuristics (blackframe + scenecut + silence
// near the edge) tried to approximate without the visual context.
//
// Reuses the NNDetector's backbone embeddings — no second ONNX pass.
// Caller extracts embeddings once via NNDetector.EmbedBatch, then
// passes them to BoundaryScores. Per-frame cost: 3840 × 64 mul-add
// for hidden layer + 64 for output = ~250 µs on M-series.
type BoundaryDetector struct {
	headPath string

	mu        sync.RWMutex
	loaded    bool
	headMtime int64

	inDim     int       // = boundaryWindowSize × nnFeatDim (3840 in v1)
	hidden    int       // hidden-layer width (64 in v1)
	w1        []float32 // (inDim, hidden) row-major
	b1        []float32 // hidden
	w2        []float32 // (hidden, 1)
	b2        float32   // output bias
}

// Window offsets used at training and inference: t-1s, t, t+1s
// (stride 1 frame at fps=1). MUST stay in sync with WINDOW_OFFSETS in
// scripts/train-boundary-head.py — the head weights are trained to
// expect this exact concatenation order.
var boundaryWindowOffsets = [...]int{-1, 0, 1}

const boundaryHeaderLen = 16 // 4-byte magic + 3 × uint32 LE

// NewBoundaryDetector loads a BNDR v1 head from disk. Returns nil and
// nil error when headPath is empty (= caller didn't request boundary
// detection); returns nil and a real error when the path is set but
// the file is missing/corrupt.
func NewBoundaryDetector(headPath string) (*BoundaryDetector, error) {
	if headPath == "" {
		return nil, nil
	}
	d := &BoundaryDetector{headPath: headPath}
	if err := d.reloadHead(); err != nil {
		return nil, fmt.Errorf("boundary head: %w", err)
	}
	return d, nil
}

// reloadHead reads the BNDR v1 head.bin format (header + W1, b1, W2, b2).
// Format spec mirrors scripts/train-boundary-head.py write_boundary_head_v1:
//
//	0..3   "BNDR"           magic
//	4..7   uint32 LE        version (= 1)
//	8..11  uint32 LE        input_dim
//	12..15 uint32 LE        hidden_dim
//	16..   float32 LE       W1 row-major (input_dim × hidden_dim)
//	       float32 LE       b1 (hidden_dim)
//	       float32 LE       W2 row-major (hidden_dim × 1)
//	       float32 LE       b2 (1)
func (d *BoundaryDetector) reloadHead() error {
	raw, err := os.ReadFile(d.headPath)
	if err != nil {
		return err
	}
	stat, err := os.Stat(d.headPath)
	if err != nil {
		return err
	}
	mtime := stat.ModTime().Unix()
	if len(raw) < boundaryHeaderLen {
		return fmt.Errorf("BNDR head truncated: %d B < %d B header",
			len(raw), boundaryHeaderLen)
	}
	if string(raw[0:4]) != "BNDR" {
		return fmt.Errorf("BNDR magic mismatch: got %q", string(raw[0:4]))
	}
	u32 := func(off int) uint32 {
		return uint32(raw[off]) | uint32(raw[off+1])<<8 |
			uint32(raw[off+2])<<16 | uint32(raw[off+3])<<24
	}
	version := u32(4)
	if version != 1 {
		return fmt.Errorf("BNDR version %d unsupported (this build reads v1)",
			version)
	}
	inDim := int(u32(8))
	hidden := int(u32(12))
	expectedDim := len(boundaryWindowOffsets) * nnFeatDim
	if inDim != expectedDim {
		return fmt.Errorf("BNDR input_dim %d != expected %d (%d × backbone %d)",
			inDim, expectedDim, len(boundaryWindowOffsets), nnFeatDim)
	}
	expectedSize := boundaryHeaderLen + (inDim*hidden+hidden+hidden+1)*4
	if len(raw) != expectedSize {
		return fmt.Errorf("BNDR head size %d != expected %d (in=%d hid=%d)",
			len(raw), expectedSize, inDim, hidden)
	}
	off := boundaryHeaderLen
	readFloats := func(n int) []float32 {
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = floatLE(raw[off+i*4:])
		}
		off += n * 4
		return out
	}
	w1 := readFloats(inDim * hidden)
	b1 := readFloats(hidden)
	w2 := readFloats(hidden)
	b2 := readFloats(1)[0]
	d.mu.Lock()
	d.loaded = true
	d.headMtime = mtime
	d.inDim = inDim
	d.hidden = hidden
	d.w1 = w1
	d.b1 = b1
	d.w2 = w2
	d.b2 = b2
	d.mu.Unlock()
	return nil
}

// MaybeReloadHead checks the file mtime and re-reads on change. Same
// hot-reload pattern as NNDetector — the nightly retrain bumps mtime,
// next call picks up the new weights without a tv-detect restart.
func (d *BoundaryDetector) MaybeReloadHead() {
	if d == nil {
		return
	}
	stat, err := os.Stat(d.headPath)
	if err != nil {
		return
	}
	d.mu.RLock()
	current := d.headMtime
	d.mu.RUnlock()
	if stat.ModTime().Unix() != current {
		_ = d.reloadHead()
	}
}

// BoundaryScores returns a per-frame boundary probability for the
// supplied per-frame backbone embeddings. The temporal context
// (boundaryWindowOffsets = -1, 0, +1) is built by mirror-padding at
// the recording boundaries (same as the training script).
//
// embeds: slice of per-frame []float32, each length nnFeatDim (1280).
// Returns []float64 of length len(embeds), each in [0, 1].
//
// Reuses the NNDetector's already-extracted embeddings — does NOT run
// ONNX. Caller is responsible for getting the embeds via the existing
// backbone pass (Phase 4 wires this into the pipeline).
func (d *BoundaryDetector) BoundaryScores(embeds [][]float32) []float64 {
	if d == nil {
		return nil
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.loaded {
		return nil
	}
	n := len(embeds)
	out := make([]float64, n)
	x := make([]float32, d.inDim)
	hidden := make([]float32, d.hidden)
	for i := 0; i < n; i++ {
		// Build the temporal context input: concat embeds[i+offset]
		// for each offset in boundaryWindowOffsets. Out-of-range
		// neighbours mirror to the nearest in-range frame (= same
		// padding scheme as train-boundary-head.py).
		for w, off := range boundaryWindowOffsets {
			j := i + off
			if j < 0 {
				j = 0
			} else if j >= n {
				j = n - 1
			}
			copy(x[w*nnFeatDim:(w+1)*nnFeatDim], embeds[j])
		}
		// Hidden layer: h_j = ReLU(b1_j + Σ_k x_k · W1[k*H + j]).
		// Same row-major convention as NNDetector.confidenceMLP — keep
		// it consistent so the training-side .npy weights round-trip.
		copy(hidden, d.b1)
		for k := 0; k < d.inDim; k++ {
			xk := x[k]
			if xk == 0 {
				continue
			}
			rowOff := k * d.hidden
			for j := 0; j < d.hidden; j++ {
				hidden[j] += xk * d.w1[rowOff+j]
			}
		}
		for j := 0; j < d.hidden; j++ {
			if hidden[j] < 0 {
				hidden[j] = 0
			}
		}
		logit := d.b2
		for j := 0; j < d.hidden; j++ {
			logit += hidden[j] * d.w2[j]
		}
		out[i] = sigmoid(logit)
	}
	return out
}

// IsLoaded reports whether the head was successfully loaded. Useful
// for the pipeline to decide whether to compute embeddings only when
// they will actually be consumed.
func (d *BoundaryDetector) IsLoaded() bool {
	if d == nil {
		return false
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.loaded
}
