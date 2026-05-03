package signals

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestConfidenceMLP_HandComputed verifies the MLP forward pass
// against a hand-computed expected value. Bypasses loadMLPHead so
// it can use a tiny hidden size; nnFeatDim is fixed at 1280 by the
// constants block, so the input dim has to match (most weights zero,
// only the first two inputs carry signal).
func TestConfidenceMLP_HandComputed(t *testing.T) {
	const H = 2
	W1 := make([]float32, nnFeatDim*H)
	W1[0*H+0] = 0.1 // input 0 → hidden 0
	W1[0*H+1] = 0.2 // input 0 → hidden 1
	W1[1*H+0] = 0.3 // input 1 → hidden 0
	W1[1*H+1] = 0.4 // input 1 → hidden 1
	d := &NNDetector{
		headLoaded:   true,
		headIsMLP:    true,
		mlpInDim:     nnFeatDim,
		mlpHidden:    H,
		mlpOutDim:    1,
		mlpBackbone:  nnFeatDim,
		mlpW1:        W1,
		mlpB1:        []float32{0, 0},
		mlpW2:        []float32{0.5, 0.6},
		mlpB2:        []float32{0},
		mlpChanIdx:   -1,
	}
	feats := make([]float32, nnFeatDim)
	feats[0] = 1.0
	feats[1] = 2.0

	out := d.confidenceMLP(feats, nil, nil, 1)
	if len(out) != 1 {
		t.Fatalf("len(out)=%d, want 1", len(out))
	}
	// hidden[0] = ReLU(0 + 1.0*0.1 + 2.0*0.3) = 0.7
	// hidden[1] = ReLU(0 + 1.0*0.2 + 2.0*0.4) = 1.0
	// logit    = 0 + 0.7*0.5 + 1.0*0.6 = 0.95
	// prob     = 1 / (1 + e^-0.95) = 0.72111519...
	want := 1.0 / (1.0 + math.Exp(-0.95))
	if math.Abs(out[0]-want) > 1e-6 {
		t.Errorf("out[0]=%.8f, want %.8f (Δ=%.2e)",
			out[0], want, out[0]-want)
	}
}

// TestConfidenceMLP_ReLUClamps verifies ReLU truly zeros negative
// pre-activations rather than passing them through (=  catches a
// missed activation step in the forward pass).
func TestConfidenceMLP_ReLUClamps(t *testing.T) {
	const H = 2
	W1 := make([]float32, nnFeatDim*H)
	// Hidden 0 unused; hidden 1 driven negative via input 0.
	W1[0*H+1] = -10.0
	d := &NNDetector{
		headLoaded:  true,
		headIsMLP:   true,
		mlpInDim:    nnFeatDim,
		mlpHidden:   H,
		mlpOutDim:   1,
		mlpBackbone: nnFeatDim,
		mlpW1:       W1,
		mlpB1:       []float32{0, 0},
		// Output picks up hidden[1] only — without ReLU it would
		// drag logit deeply negative.
		mlpW2:      []float32{0, 1.0},
		mlpB2:      []float32{0},
		mlpChanIdx: -1,
	}
	feats := make([]float32, nnFeatDim)
	feats[0] = 1.0 // pushes hidden[1] pre-act to -10

	out := d.confidenceMLP(feats, nil, nil, 1)
	// With ReLU: hidden[1] = max(0, -10) = 0; logit = 0; prob = 0.5.
	// Without ReLU: prob ≈ sigmoid(-10) ≈ 4.5e-5 (= drastically off).
	if math.Abs(out[0]-0.5) > 1e-6 {
		t.Errorf("out[0]=%.8f, want 0.5 (ReLU likely missing)", out[0])
	}
}

// TestConfidenceMLP_LogoAudioChannelInputs covers the optional input
// blocks (logo, audio, channel one-hot). Builds a head that uses
// each block exactly once with a known weight; checks the per-block
// contribution lands at the right input slot.
func TestConfidenceMLP_LogoAudioChannelInputs(t *testing.T) {
	const H = 1
	const nLogo = 1
	const nAudio = 1
	const nChan = 4
	inDim := nnFeatDim + nLogo + nAudio + nChan
	W1 := make([]float32, inDim*H)
	// Identity-style: each contributing slot carries a unique weight
	// so we can read off which slot got summed into the hidden unit.
	W1[(nnFeatDim+0)*H+0] = 7.0  // logo slot
	W1[(nnFeatDim+nLogo+0)*H+0] = 11.0 // audio slot
	W1[(nnFeatDim+nLogo+nAudio+2)*H+0] = 13.0 // channel idx 2
	d := &NNDetector{
		headLoaded:  true,
		headIsMLP:   true,
		mlpInDim:    inDim,
		mlpHidden:   H,
		mlpOutDim:   1,
		mlpBackbone: nnFeatDim,
		mlpNLogo:    nLogo,
		mlpNAudio:   nAudio,
		mlpNChannel: nChan,
		mlpW1:       W1,
		mlpB1:       []float32{0},
		mlpW2:       []float32{1.0},
		mlpB2:       []float32{0},
		// Channel idx 2 = the slot we put weight 13 into.
		mlpChanIdx: 2,
	}
	feats := make([]float32, nnFeatDim) // all zero — backbone contributes nothing
	logoConfs := []float64{1.0}
	rmsConfs := []float64{1.0}

	out := d.confidenceMLP(feats, logoConfs, rmsConfs, 1)
	// hidden[0] = 0 + 1.0*7 + 1.0*11 + 1.0*13 = 31  (channel idx 2 hot)
	// logit = 31; prob = sigmoid(31) ≈ 1.0 (saturated)
	if out[0] < 0.999 {
		t.Errorf("out[0]=%.8f, want ≈1.0 (slot inputs not summing)",
			out[0])
	}
}

// TestConfidenceMLP_BatchIndependence: two-frame batch where frame 0
// has a logo signal and frame 1 doesn't. Validates that per-frame
// inputs aren't accidentally shared/leaking across iterations.
func TestConfidenceMLP_BatchIndependence(t *testing.T) {
	const H = 1
	inDim := nnFeatDim + 1 // +1 logo
	W1 := make([]float32, inDim*H)
	W1[nnFeatDim*H+0] = 100.0 // logo weight pushes to saturation
	d := &NNDetector{
		headLoaded:  true,
		headIsMLP:   true,
		mlpInDim:    inDim,
		mlpHidden:   H,
		mlpOutDim:   1,
		mlpBackbone: nnFeatDim,
		mlpNLogo:    1,
		mlpW1:       W1,
		mlpB1:       []float32{0},
		mlpW2:       []float32{1.0},
		mlpB2:       []float32{0},
		mlpChanIdx:  -1,
	}
	feats := make([]float32, 2*nnFeatDim) // 2 frames, all zero
	logoConfs := []float64{1.0, 0.0}      // frame 0: logo present, frame 1: absent

	out := d.confidenceMLP(feats, logoConfs, nil, 2)
	if len(out) != 2 {
		t.Fatalf("len(out)=%d, want 2", len(out))
	}
	if out[0] < 0.999 {
		t.Errorf("frame 0 (logo=1.0) prob=%.8f, want ≈1.0", out[0])
	}
	if math.Abs(out[1]-0.5) > 1e-6 {
		t.Errorf("frame 1 (logo=0.0) prob=%.8f, want 0.5", out[1])
	}
}

// writeTestMLPHead: synthetic head.bin with caller-supplied dims +
// weights, in the v1 format. Used by the loadMLPHead roundtrip
// test below — keeps fixtures inline rather than checking in a
// binary blob that drifts from the format spec.
func writeTestMLPHead(t *testing.T, path string,
	inDim, hidden, outDim, nLogo, nAudio, nChan int,
	W1, b1, W2, b2 []float32) {
	t.Helper()
	if len(W1) != inDim*hidden ||
		len(b1) != hidden ||
		len(W2) != hidden*outDim ||
		len(b2) != outDim {
		t.Fatalf("writeTestMLPHead: weight shape mismatch")
	}
	header := make([]byte, 36)
	binary.LittleEndian.PutUint32(header[0:], 0x31504C4D) // "MLP1"
	binary.LittleEndian.PutUint32(header[4:], 1)
	binary.LittleEndian.PutUint32(header[8:], uint32(inDim))
	binary.LittleEndian.PutUint32(header[12:], uint32(hidden))
	binary.LittleEndian.PutUint32(header[16:], uint32(outDim))
	binary.LittleEndian.PutUint32(header[20:], uint32(nnFeatDim))
	binary.LittleEndian.PutUint32(header[24:], uint32(nLogo))
	binary.LittleEndian.PutUint32(header[28:], uint32(nAudio))
	binary.LittleEndian.PutUint32(header[32:], uint32(nChan))
	body := make([]byte, 0, (len(W1)+len(b1)+len(W2)+len(b2))*4)
	for _, v := range append(append(append([]float32{}, W1...), b1...), append(W2, b2...)...) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		body = append(body, b[:]...)
	}
	if err := os.WriteFile(path, append(header, body...), 0o644); err != nil {
		t.Fatalf("write head: %v", err)
	}
}

// TestLoadMLPHead_Roundtrip: write a synthetic v1 head + sidecar to
// a temp dir, load it via the production reloadHead path, verify
// every field round-trips correctly + an unknown channel slug
// degrades gracefully to mlpChanIdx=-1.
func TestLoadMLPHead_Roundtrip(t *testing.T) {
	dir := t.TempDir()
	headPath := filepath.Join(dir, "head.bin")

	const H = 3
	const nChan = 4
	inDim := nnFeatDim + 1 + 1 + nChan // logo + audio + chan
	W1 := make([]float32, inDim*H)
	for i := range W1 {
		W1[i] = float32(i) * 0.001
	}
	b1 := []float32{0.1, 0.2, 0.3}
	W2 := []float32{1.5, 2.5, 3.5}
	b2 := []float32{0.05}
	writeTestMLPHead(t, headPath, inDim, H, 1, 1, 1, nChan, W1, b1, W2, b2)

	// Sidecar — slug list must be exactly nChan entries.
	sidecarPath := filepath.Join(dir, "head.channel-map.json")
	sidecar := `{"version":1,"n":4,"slugs":["alpha","beta","gamma","delta"]}`
	if err := os.WriteFile(sidecarPath, []byte(sidecar), 0o644); err != nil {
		t.Fatal(err)
	}

	// Recording on channel "gamma" (= idx 2 in the alphabetical list).
	d := &NNDetector{headPath: headPath, channelSlug: "gamma", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("reloadHead: %v", err)
	}
	if !d.headIsMLP {
		t.Fatal("headIsMLP=false, want true after MLP1 magic load")
	}
	if d.mlpInDim != inDim || d.mlpHidden != H ||
		d.mlpOutDim != 1 || d.mlpNChannel != nChan {
		t.Errorf("dims: in=%d hid=%d out=%d nChan=%d, "+
			"want in=%d hid=%d out=1 nChan=%d",
			d.mlpInDim, d.mlpHidden, d.mlpOutDim, d.mlpNChannel,
			inDim, H, nChan)
	}
	if d.mlpChanIdx != 2 {
		t.Errorf("mlpChanIdx=%d, want 2 (= 'gamma' is alphabetical idx 2)",
			d.mlpChanIdx)
	}
	for i, w := range W1 {
		if d.mlpW1[i] != w {
			t.Fatalf("mlpW1[%d]=%g, want %g", i, d.mlpW1[i], w)
			break
		}
	}
	if d.mlpB1[0] != 0.1 || d.mlpB1[2] != 0.3 {
		t.Errorf("b1 mismatch: %v", d.mlpB1)
	}
	if d.mlpW2[0] != 1.5 || d.mlpW2[2] != 3.5 {
		t.Errorf("W2 mismatch: %v", d.mlpW2)
	}
	if d.mlpB2[0] != 0.05 {
		t.Errorf("b2[0]=%g, want 0.05", d.mlpB2[0])
	}

	// Unknown slug → graceful -1 (= channel-agnostic).
	d2 := &NNDetector{headPath: headPath, channelSlug: "neverexisted",
		mlpChanIdx: -1}
	if err := d2.reloadHead(); err != nil {
		t.Fatalf("reloadHead unknown slug: %v", err)
	}
	if d2.mlpChanIdx != -1 {
		t.Errorf("unknown slug mlpChanIdx=%d, want -1 (= zero one-hot fallback)",
			d2.mlpChanIdx)
	}
}

// TestLoadMLPHead_BadMagic: file starts with junk bytes that look
// MLP-ish but aren't the exact magic → should NOT enter the MLP
// loader (= falls through to the legacy LogReg size detector,
// which then errors on the unrecognised size).
func TestLoadMLPHead_BadMagic(t *testing.T) {
	dir := t.TempDir()
	headPath := filepath.Join(dir, "head.bin")
	// 100 random bytes; first 4 are not "MLP1".
	junk := make([]byte, 100)
	junk[0] = 'X'
	junk[1] = 'M'
	junk[2] = 'L'
	junk[3] = '1'
	if err := os.WriteFile(headPath, junk, 0o644); err != nil {
		t.Fatal(err)
	}
	d := &NNDetector{headPath: headPath, mlpChanIdx: -1}
	err := d.reloadHead()
	if err == nil {
		t.Fatal("reloadHead accepted junk bytes; want size-mismatch error")
	}
	if d.headIsMLP {
		t.Errorf("headIsMLP=true on junk bytes; want false (no MLP1 magic)")
	}
}
