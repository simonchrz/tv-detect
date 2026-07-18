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
		headLoaded:  true,
		headIsMLP:   true,
		mlpInDim:    nnFeatDim,
		mlpHidden:   H,
		mlpOutDim:   1,
		mlpBackbone: nnFeatDim,
		mlpW1:       W1,
		mlpB1:       []float32{0, 0},
		mlpW2:       []float32{0.5, 0.6},
		mlpB2:       []float32{0},
		mlpChanIdx:  -1,
	}
	feats := make([]float32, nnFeatDim)
	feats[0] = 1.0
	feats[1] = 2.0

	out := d.confidenceMLPChunk(feats, nil, nil, 1, 1, 0)
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

	out := d.confidenceMLPChunk(feats, nil, nil, 1, 1, 0)
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
	W1[(nnFeatDim+0)*H+0] = 7.0               // logo slot
	W1[(nnFeatDim+nLogo+0)*H+0] = 11.0        // audio slot
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

	out := d.confidenceMLPChunk(feats, logoConfs, rmsConfs, 1, 1, 0)
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

	out := d.confidenceMLPChunk(feats, logoConfs, nil, 2, 1, 0)
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

// writeTestMLPHeadV2: synthetic MLP2 head.bin (40-byte header,
// includes n_whisper). Mirrors writeTestMLPHead's layout decisions.
func writeTestMLPHeadV2(t *testing.T, path string,
	inDim, hidden, outDim, nLogo, nAudio, nChan, nWhisper int,
	W1, b1, W2, b2 []float32) {
	t.Helper()
	if len(W1) != inDim*hidden ||
		len(b1) != hidden ||
		len(W2) != hidden*outDim ||
		len(b2) != outDim {
		t.Fatalf("writeTestMLPHeadV2: weight shape mismatch")
	}
	header := make([]byte, 40)
	binary.LittleEndian.PutUint32(header[0:], 0x32504C4D) // "MLP2"
	binary.LittleEndian.PutUint32(header[4:], 2)
	binary.LittleEndian.PutUint32(header[8:], uint32(inDim))
	binary.LittleEndian.PutUint32(header[12:], uint32(hidden))
	binary.LittleEndian.PutUint32(header[16:], uint32(outDim))
	binary.LittleEndian.PutUint32(header[20:], uint32(nnFeatDim))
	binary.LittleEndian.PutUint32(header[24:], uint32(nLogo))
	binary.LittleEndian.PutUint32(header[28:], uint32(nAudio))
	binary.LittleEndian.PutUint32(header[32:], uint32(nChan))
	binary.LittleEndian.PutUint32(header[36:], uint32(nWhisper))
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

// TestLoadMLPHeadV2_Roundtrip: parses an MLP2 file with whisper
// slot, validates header decoding incl. n_whisper, then exercises
// the forward pass through the whisper-input slot via SetWhisperProbs.
func TestLoadMLPHeadV2_Roundtrip(t *testing.T) {
	dir := t.TempDir()
	headPath := filepath.Join(dir, "head.bin")

	const H = 1
	const nChan = 2
	const nLogo = 1
	const nAudio = 1
	const nWhisper = 1
	inDim := nnFeatDim + nLogo + nAudio + nChan + nWhisper
	W1 := make([]float32, inDim*H)
	// Only the whisper slot carries weight; everything else zero so
	// the test can pin output to f(whisper_input).
	whisperInputCol := nnFeatDim + nLogo + nAudio + nChan
	W1[whisperInputCol*H+0] = 4.0
	b1 := []float32{0}
	W2 := []float32{1.0}
	b2 := []float32{0}
	writeTestMLPHeadV2(t, headPath,
		inDim, H, 1, nLogo, nAudio, nChan, nWhisper,
		W1, b1, W2, b2)
	sidecarPath := filepath.Join(dir, "head.channel-map.json")
	if err := os.WriteFile(sidecarPath,
		[]byte(`{"version":1,"n":2,"slugs":["alpha","beta"]}`),
		0o644); err != nil {
		t.Fatal(err)
	}

	d := &NNDetector{headPath: headPath, channelSlug: "alpha", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("reloadHead MLP2: %v", err)
	}
	if !d.headIsMLP || d.mlpNWhisper != 1 {
		t.Fatalf("MLP2 load: headIsMLP=%v mlpNWhisper=%d, want true/1",
			d.headIsMLP, d.mlpNWhisper)
	}
	if d.mlpInDim != inDim || d.mlpNChannel != nChan {
		t.Errorf("dims wrong: in=%d nChan=%d", d.mlpInDim, d.mlpNChannel)
	}

	// Two frames; supply whisper-prob 1.0 + 0.0 → expect saturated
	// sigmoid(4) ≈ 0.982 vs sigmoid(0) = 0.5.
	d.SetWhisperProbs([]float64{1.0, 0.0})
	feats := make([]float32, 2*nnFeatDim)
	out := d.confidenceMLPChunk(feats, []float64{0, 0}, []float64{0.5, 0.5}, 2, 1, 0)
	if out[0] < 0.97 {
		t.Errorf("frame 0 (whisper=1) prob=%.4f, want ≈0.98 (saturated)",
			out[0])
	}
	if math.Abs(out[1]-0.5) > 1e-6 {
		t.Errorf("frame 1 (whisper=0) prob=%.4f, want 0.5", out[1])
	}

	// Nil whisper-probs → fallback to 0.5 → sigmoid(4*0.5) ≈ 0.881.
	d.SetWhisperProbs(nil)
	out = d.confidenceMLPChunk(feats, []float64{0, 0}, []float64{0.5, 0.5}, 2, 1, 0)
	if math.Abs(out[0]-0.8807970779778823) > 1e-6 {
		t.Errorf("nil whisper fallback prob=%.6f, want sigmoid(2)=0.8808",
			out[0])
	}
}

// TestMLPHeader_BackwardCompat: an MLP1 file still loads correctly
// after the dispatch was extended for MLP2 — no regression on the
// production format.
func TestMLPHeader_BackwardCompat(t *testing.T) {
	dir := t.TempDir()
	headPath := filepath.Join(dir, "head.bin")
	const H = 2
	const nChan = 2
	inDim := nnFeatDim + 1 + 1 + nChan
	W1 := make([]float32, inDim*H)
	b1 := []float32{0, 0}
	W2 := []float32{1, 1}
	b2 := []float32{0}
	writeTestMLPHead(t, headPath, inDim, H, 1, 1, 1, nChan, W1, b1, W2, b2)
	sidecarPath := filepath.Join(dir, "head.channel-map.json")
	os.WriteFile(sidecarPath,
		[]byte(`{"version":1,"n":2,"slugs":["a","b"]}`), 0o644)
	d := &NNDetector{headPath: headPath, channelSlug: "a", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("reloadHead MLP1 after MLP2 dispatch: %v", err)
	}
	if !d.headIsMLP || d.mlpNWhisper != 0 {
		t.Errorf("MLP1 load: headIsMLP=%v mlpNWhisper=%d, want true/0",
			d.headIsMLP, d.mlpNWhisper)
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

// writeTestMLPHeadV3: synthetic MLP3 head.bin (44-byte header,
// includes n_whisper AND n_temporal). Mirrors writeTestMLPHeadV2's
// layout decisions.
func writeTestMLPHeadV3(t *testing.T, path string,
	inDim, hidden, outDim, nLogo, nAudio, nChan, nWhisper, nTemporal int,
	W1, b1, W2, b2 []float32) {
	t.Helper()
	if len(W1) != inDim*hidden ||
		len(b1) != hidden ||
		len(W2) != hidden*outDim ||
		len(b2) != outDim {
		t.Fatalf("writeTestMLPHeadV3: weight shape mismatch")
	}
	header := make([]byte, 44)
	binary.LittleEndian.PutUint32(header[0:], 0x33504C4D) // "MLP3"
	binary.LittleEndian.PutUint32(header[4:], 3)
	binary.LittleEndian.PutUint32(header[8:], uint32(inDim))
	binary.LittleEndian.PutUint32(header[12:], uint32(hidden))
	binary.LittleEndian.PutUint32(header[16:], uint32(outDim))
	binary.LittleEndian.PutUint32(header[20:], uint32(nnFeatDim))
	binary.LittleEndian.PutUint32(header[24:], uint32(nLogo))
	binary.LittleEndian.PutUint32(header[28:], uint32(nAudio))
	binary.LittleEndian.PutUint32(header[32:], uint32(nChan))
	binary.LittleEndian.PutUint32(header[36:], uint32(nWhisper))
	binary.LittleEndian.PutUint32(header[40:], uint32(nTemporal))
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

// TestLoadMLPHeadV3_Roundtrip: parses an MLP3 file with whisper AND
// temporal slots, validates header decoding incl. n_temporal, then
// exercises the forward pass through the whisper-input slot (same
// shape as the V2 test) to confirm the extra header field didn't
// shift any existing offsets.
func TestLoadMLPHeadV3_Roundtrip(t *testing.T) {
	dir := t.TempDir()
	headPath := filepath.Join(dir, "head.bin")

	const H = 1
	const nChan = 2
	const nLogo = 1
	const nAudio = 1
	const nWhisper = 1
	const nTemporal = 2
	inDim := nnFeatDim + nLogo + nAudio + nChan + nWhisper + nTemporal
	W1 := make([]float32, inDim*H)
	whisperInputCol := nnFeatDim + nLogo + nAudio + nChan
	W1[whisperInputCol*H+0] = 4.0
	b1 := []float32{0}
	W2 := []float32{1.0}
	b2 := []float32{0}
	writeTestMLPHeadV3(t, headPath,
		inDim, H, 1, nLogo, nAudio, nChan, nWhisper, nTemporal,
		W1, b1, W2, b2)
	sidecarPath := filepath.Join(dir, "head.channel-map.json")
	if err := os.WriteFile(sidecarPath,
		[]byte(`{"version":1,"n":2,"slugs":["alpha","beta"]}`),
		0o644); err != nil {
		t.Fatal(err)
	}

	d := &NNDetector{headPath: headPath, channelSlug: "alpha", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("reloadHead MLP3: %v", err)
	}
	if !d.headIsMLP || d.mlpNWhisper != 1 || d.mlpNTemporal != 2 {
		t.Fatalf("MLP3 load: headIsMLP=%v mlpNWhisper=%d mlpNTemporal=%d, "+
			"want true/1/2", d.headIsMLP, d.mlpNWhisper, d.mlpNTemporal)
	}
	if d.mlpInDim != inDim || d.mlpNChannel != nChan {
		t.Errorf("dims wrong: in=%d nChan=%d", d.mlpInDim, d.mlpNChannel)
	}

	// Zero temporal deltas (2 identical frames) should leave the
	// whisper-only forward pass unaffected — same expected values as
	// TestLoadMLPHeadV2_Roundtrip.
	d.SetWhisperProbs([]float64{1.0, 0.0})
	feats := make([]float32, 2*nnFeatDim)
	out := d.confidenceMLPChunk(feats, []float64{0, 0}, []float64{0.5, 0.5}, 2, 1, 0)
	if out[0] < 0.97 {
		t.Errorf("frame 0 (whisper=1) prob=%.4f, want ≈0.98 (saturated)",
			out[0])
	}
	if math.Abs(out[1]-0.5) > 1e-6 {
		t.Errorf("frame 1 (whisper=0) prob=%.4f, want 0.5", out[1])
	}
}

// TestConfidenceMLP_TemporalDeltaMatchesPython is the Go↔Python
// parity check promised for the mlp32-channel-whisper-temporal
// migration (2026-07-12): the L2-distance-to-prev/next-frame values
// computed here MUST match scripts/train-head.py's
// _augment_channel_whisper_temporal exactly, or a deployed head
// scores against a differently-distributed input than the one the
// gate validated during training.
//
// Golden values cross-checked against a literal numpy transcription
// of the Python formula (`np.linalg.norm(X[1:] - X[:-1], axis=1)`,
// same edge-zero convention) on the identical 3-frame
// [backbone0, logo, audio] example:
//
//	frame0 = [1.0, 0.2, 0.3]
//	frame1 = [4.0, 0.6, 0.7]
//	frame2 = [4.0, 0.6, 0.7]  (repeat of frame1 → zero delta)
//
// numpy: dp = [0, 3.0528674, 0], dn = [3.0528674, 0, 0].
func TestConfidenceMLP_TemporalDeltaMatchesPython(t *testing.T) {
	const H = 1
	const nLogo = 1
	const nAudio = 1
	const nTemporal = 2
	inDim := nnFeatDim + nLogo + nAudio + nTemporal
	temporalOff := nnFeatDim + nLogo + nAudio // no channel block in this test
	W1 := make([]float32, inDim*H)
	// Isolate dp+dn into hidden[0]; output = ReLU(dp+dn) (both ≥0,
	// so ReLU is a no-op here — chosen so sign never needs checking).
	W1[temporalOff*H+0] = 1.0
	W1[(temporalOff+1)*H+0] = 1.0
	d := &NNDetector{
		headLoaded:   true,
		headIsMLP:    true,
		mlpInDim:     inDim,
		mlpHidden:    H,
		mlpOutDim:    1,
		mlpBackbone:  nnFeatDim,
		mlpNLogo:     nLogo,
		mlpNAudio:    nAudio,
		mlpNChannel:  0,
		mlpNWhisper:  0,
		mlpNTemporal: nTemporal,
		mlpW1:        W1,
		mlpB1:        []float32{0},
		mlpW2:        []float32{1.0},
		mlpB2:        []float32{0},
		mlpChanIdx:   -1,
	}
	feats := make([]float32, 3*nnFeatDim)
	feats[0*nnFeatDim] = 1.0
	feats[1*nnFeatDim] = 4.0
	feats[2*nnFeatDim] = 4.0
	logoConfs := []float64{0.2, 0.6, 0.6}
	rmsConfs := []float64{0.3, 0.7, 0.7}

	out := d.confidenceMLPChunk(feats, logoConfs, rmsConfs, 3, 1, 0)
	if len(out) != 3 {
		t.Fatalf("len(out)=%d, want 3", len(out))
	}
	const dist = 3.0528674 // numpy golden value, see doc comment
	want := []float64{
		1.0 / (1.0 + math.Exp(-(0 + dist))), // frame0: dp=0, dn=dist
		1.0 / (1.0 + math.Exp(-(dist + 0))), // frame1: dp=dist, dn=0
		1.0 / (1.0 + math.Exp(-(0 + 0))),    // frame2: dp=0, dn=0 → 0.5
	}
	for i := range want {
		if math.Abs(out[i]-want[i]) > 1e-5 {
			t.Errorf("frame %d: out=%.8f, want %.8f (Δ=%.2e) — "+
				"Go temporal-delta diverges from the Python golden value",
				i, out[i], want[i], out[i]-want[i])
		}
	}
	if math.Abs(out[2]-0.5) > 1e-6 {
		t.Errorf("frame 2 (identical to frame 1, zero delta) prob=%.6f, "+
			"want 0.5", out[2])
	}
}

// TestConfidenceMLPChunk_TemporalAt25fps is the parity test the 07-12
// migration was missing: at production frame rates the temporal deltas
// must be computed against the frame ONE SECOND away (step = fps), not
// the neighbouring frame. The old batch-scope pass compared consecutive
// 25fps frames (≈25x smaller deltas, zeroed each 32-frame batch edge) —
// the root cause of the 2026-07-18 production NN degradation.
func TestConfidenceMLPChunk_TemporalAt25fps(t *testing.T) {
	const H = 1
	const nTemporal = 2
	const fps = 25.0
	const n = 60 // 2.4 s at 25fps
	inDim := nnFeatDim + nTemporal
	temporalOff := nnFeatDim
	W1 := make([]float32, inDim*H)
	W1[temporalOff*H+0] = 1.0     // dp
	W1[(temporalOff+1)*H+0] = 1.0 // dn
	d := &NNDetector{
		headLoaded:   true,
		headIsMLP:    true,
		mlpInDim:     inDim,
		mlpHidden:    H,
		mlpOutDim:    1,
		mlpBackbone:  nnFeatDim,
		mlpNTemporal: nTemporal,
		mlpW1:        W1,
		mlpB1:        []float32{0},
		mlpW2:        []float32{1.0},
		mlpB2:        []float32{0},
		mlpChanIdx:   -1,
	}
	// Embedding[0] ramps by +1.0 per FRAME → the delta to the frame 25
	// indices away is exactly 25.0; consecutive-frame deltas would be 1.0.
	feats := make([]float32, n*nnFeatDim)
	for i := 0; i < n; i++ {
		feats[i*nnFeatDim] = float32(i)
	}
	out := d.confidenceMLPChunk(feats, nil, nil, n, fps, 0)

	// Frame 30 (≥ step from both edges): dp = dn = 25 → logit 50 →
	// saturated ≈ 1.0. With the OLD consecutive-frame bug it would be
	// sigmoid(2) ≈ 0.88 — the test discriminates sharply.
	if out[30] < 0.999999 {
		t.Errorf("frame 30 prob=%.8f, want ≈1.0 (dp=dn=25 at 1s spacing; "+
			"consecutive-frame deltas would give sigmoid(2)≈0.88)", out[30])
	}
	// Frame 10 (< step from chunk start): dp must be 0, dn = 25 →
	// logit 25 → still saturated but distinguishable via a weaker head…
	// simpler: frame n-10 (< step from end): dp = 25, dn = 0 → logit 25.
	// Both edges: verify frame 0 has dp=0 (logit = dn = 25 → ≈1.0) and
	// that a WOULD-BE consecutive delta of 1.0 never appears alone:
	// check frame 55 (dp=25, dn=0 → sigmoid(25)) vs frame 30 equality.
	if math.Abs(out[55]-out[30]) > 1e-9 && out[55] < 0.999999 {
		t.Errorf("frame 55 prob=%.8f, want saturated (dp=25, dn=0)", out[55])
	}
}

// TestConfidenceMLPChunk_WhisperAbsoluteSeconds pins the whisper column
// to ABSOLUTE wall-clock seconds (chunk offset + frame/fps). The old
// batch-scope pass indexed the per-second array with the batch-local
// frame index — every 32-frame batch replayed the first 32 seconds of
// whisper data (second bug in the 2026-07-18 root-cause set).
func TestConfidenceMLPChunk_WhisperAbsoluteSeconds(t *testing.T) {
	const H = 1
	const nWhisper = 1
	const fps = 25.0
	const chunkStart = 100.0 // chunk begins at t=100 s in the recording
	const n = 50             // 2 s of frames
	inDim := nnFeatDim + nWhisper
	whisperOff := nnFeatDim
	W1 := make([]float32, inDim*H)
	W1[whisperOff*H+0] = 4.0
	d := &NNDetector{
		headLoaded:  true,
		headIsMLP:   true,
		mlpInDim:    inDim,
		mlpHidden:   H,
		mlpOutDim:   1,
		mlpBackbone: nnFeatDim,
		mlpNWhisper: nWhisper,
		mlpW1:       W1,
		mlpB1:       []float32{0},
		mlpW2:       []float32{1.0},
		mlpB2:       []float32{0},
		mlpChanIdx:  -1,
	}
	// Whisper per-second array: 1.0 at seconds 100+101, 0.0 elsewhere.
	probs := make([]float64, 200)
	probs[100] = 1.0
	probs[101] = 1.0
	d.mlpWhisperProbs = probs

	feats := make([]float32, n*nnFeatDim)
	out := d.confidenceMLPChunk(feats, nil, nil, n, fps, chunkStart)

	// ALL 50 frames live in seconds 100-101 → whisper=1.0 → sigmoid(4).
	want := 1.0 / (1.0 + math.Exp(-4.0))
	for _, i := range []int{0, 24, 25, 49} {
		if math.Abs(out[i]-want) > 1e-6 {
			t.Errorf("frame %d prob=%.6f, want %.6f (whisper at abs sec "+
				"%d — batch-local indexing would read sec %d = 0.0)",
				i, out[i], want, int(chunkStart)+i/25, i)
		}
	}
}
