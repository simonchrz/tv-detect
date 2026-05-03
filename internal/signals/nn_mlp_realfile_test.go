package signals

import (
	"os"
	"testing"
)

// TestLoadRealMLPHead loads the actual Python-written head.bin from
// the local Phase-C test run if present. Skip if the file isn't on
// disk (= test is opportunistic). Validates the Python format spec
// matches what Go expects on real artefacts, not just synthetic.
func TestLoadRealMLPHead(t *testing.T) {
	const path = "/tmp/tv-train-mlp-test/head.bin"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("real MLP head not present at %s — skipping", path)
	}
	d := &NNDetector{headPath: path, channelSlug: "rtl", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("load real head: %v", err)
	}
	if !d.headIsMLP {
		t.Fatal("real head loaded but headIsMLP=false")
	}
	if d.mlpInDim != 1290 || d.mlpHidden != 32 || d.mlpOutDim != 1 ||
		d.mlpNLogo != 1 || d.mlpNAudio != 1 || d.mlpNChannel != 8 {
		t.Errorf("dims: in=%d hid=%d out=%d logo=%d audio=%d chan=%d, "+
			"want 1290/32/1/1/1/8",
			d.mlpInDim, d.mlpHidden, d.mlpOutDim,
			d.mlpNLogo, d.mlpNAudio, d.mlpNChannel)
	}
	if d.mlpChanIdx != 3 { // "rtl" is alphabetical idx 3 in the 8-slug list
		t.Errorf("rtl slug → idx %d, want 3", d.mlpChanIdx)
	}
	// Run forward pass on a synthetic feature vector (= zeros + idx-3
	// channel hot via sidecar). The output should be a valid sigmoid
	// (= in (0,1)), proving the matmul works against real weights.
	feats := make([]float32, nnFeatDim)
	out := d.confidenceMLP(feats, []float64{0}, []float64{0.5}, 1)
	if len(out) != 1 || out[0] <= 0 || out[0] >= 1 {
		t.Errorf("forward pass output %v, want sigmoid in (0,1)", out)
	}
	t.Logf("real MLP head loaded; forward pass on zero-feats + rtl chan → prob %.4f", out[0])
}
