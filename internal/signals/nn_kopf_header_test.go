package signals

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Header values the forward pass cannot represent must fail the load.
// Before the check they loaded: output_dim 2 read W2 with the wrong
// stride, n_temporal 1 overwrote the next block's column, n_logo 2 left a
// column at 0 — all silent — and output_dim 0 panicked at mlpB2[0].
func TestMLPKopfHeaderWirdGeprueft(t *testing.T) {
	const H = 1
	mk := func(n int) ([]float32, []float32, []float32, []float32) {
		return make([]float32, n*H), make([]float32, H), make([]float32, H*1), make([]float32, 1)
	}
	cases := map[string]func(t *testing.T, p string){
		"output_dim 2": func(t *testing.T, p string) {
			in := nnFeatDim
			W1, b1, _, _ := mk(in)
			writeTestMLPHead(t, p, in, H, 2, 0, 0, 0, W1, b1, make([]float32, 2*H), make([]float32, 2))
		},
		"output_dim 0": func(t *testing.T, p string) {
			in := nnFeatDim
			W1, b1, _, _ := mk(in)
			writeTestMLPHead(t, p, in, H, 0, 0, 0, 0, W1, b1, nil, nil)
		},
		"n_logo 2": func(t *testing.T, p string) {
			in := nnFeatDim + 2
			W1, b1, W2, b2 := mk(in)
			writeTestMLPHead(t, p, in, H, 1, 2, 0, 0, W1, b1, W2, b2)
		},
		"n_audio 2": func(t *testing.T, p string) {
			in := nnFeatDim + 2
			W1, b1, W2, b2 := mk(in)
			writeTestMLPHead(t, p, in, H, 1, 0, 2, 0, W1, b1, W2, b2)
		},
		"n_temporal 1": func(t *testing.T, p string) {
			in := nnFeatDim + 1
			W1, b1, W2, b2 := mk(in)
			writeTestMLPHeadV3(t, p, in, H, 1, 0, 0, 0, 0, 1, W1, b1, W2, b2)
		},
		"n_temporal 4": func(t *testing.T, p string) {
			in := nnFeatDim + 4
			W1, b1, W2, b2 := mk(in)
			writeTestMLPHeadV3(t, p, in, H, 1, 0, 0, 0, 0, 4, W1, b1, W2, b2)
		},
		"n_whispermask 2": func(t *testing.T, p string) {
			in := nnFeatDim + 2
			W1, b1, W2, b2 := mk(in)
			writeTestMLPHeadV5(t, p, in, H, 1, 0, 0, 0, 0, 0, 0, 2, W1, b1, W2, b2)
		},
	}
	for name, write := range cases {
		t.Run(name, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "head.bin")
			write(t, p)
			d := &NNDetector{headPath: p, mlpChanIdx: -1}
			var err error
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("load panicked: %v", r)
					}
				}()
				err = d.reloadHead()
			}()
			if err == nil {
				t.Fatalf("header accepted (in=%d out=%d logo=%d audio=%d temporal=%d)",
					d.mlpInDim, d.mlpOutDim, d.mlpNLogo, d.mlpNAudio, d.mlpNTemporal)
			}
			if !strings.Contains(err.Error(), "erlaubt") {
				t.Errorf("error %q is not the header check", err)
			}
		})
	}
	// The valid shapes still load (n_temporal 3 = churn column).
	p := filepath.Join(t.TempDir(), "head.bin")
	in := nnFeatDim + 1 + 1 + 3
	W1, b1, W2, b2 := mk(in)
	writeTestMLPHeadV3(t, p, in, H, 1, 1, 1, 0, 0, 3, W1, b1, W2, b2)
	if err := (&NNDetector{headPath: p, mlpChanIdx: -1}).reloadHead(); err != nil {
		t.Fatalf("valid v3 head (n_temporal 3) rejected: %v", err)
	}
}

// Hot reload v5 → v2: loadMLPHeadV2 must clear the minute-prior, mask and
// OCR state of the previous head. It did not; the forward pass then kept
// writing those columns past the v2 input vector.
func TestReloadAufV2SetztZusatzspaltenZurueck(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "head.bin")
	const H = 1
	in5 := nnFeatDim + 1 + 1 + 0 + 1 + 2 + 1 + 1 // logo audio whisper temporal minuteprior mask
	writeTestMLPHeadV5(t, p, in5, H, 1, 1, 1, 0, 1, 2, 1, 1,
		make([]float32, in5*H), []float32{0}, []float32{1}, []float32{0})
	prior := `{"version":1,"neutral":0.25,"priors":{"rtl":[` +
		strings.TrimSuffix(strings.Repeat("0.5,", 60), ",") + `]}}`
	if err := os.WriteFile(filepath.Join(dir, "head.minute-prior.json"), []byte(prior), 0o644); err != nil {
		t.Fatal(err)
	}
	d := &NNDetector{headPath: p, channelSlug: "rtl", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatal(err)
	}
	if d.mlpNMinutePrior != 1 || d.mlpNWhisperMask != 1 || d.mlpMinutePrior == nil {
		t.Fatalf("v5 setup: minuteprior=%d mask=%d", d.mlpNMinutePrior, d.mlpNWhisperMask)
	}
	d.mlpNOCR = 3 // as a v6 head would have left it

	in2 := nnFeatDim + 1 + 1 + 0 + 1
	writeTestMLPHeadV2(t, p, in2, H, 1, 1, 1, 0, 1,
		make([]float32, in2*H), []float32{0}, []float32{1}, []float32{0})
	later := time.Now().Add(time.Hour) // reloadHead keys on mtime
	if err := os.Chtimes(p, later, later); err != nil {
		t.Fatal(err)
	}
	if err := d.reloadHead(); err != nil {
		t.Fatal(err)
	}
	if d.mlpNMinutePrior != 0 || d.mlpNWhisperMask != 0 || d.mlpNOCR != 0 || d.mlpMinutePrior != nil {
		t.Fatalf("after v2 reload: minuteprior=%d mask=%d ocr=%d prior=%v — stale v5/v6 state",
			d.mlpNMinutePrior, d.mlpNWhisperMask, d.mlpNOCR, d.mlpMinutePrior != nil)
	}
	d.SetStartTS(1_700_000_000)
	d.confidenceMLPChunk(make([]float32, nnFeatDim), []float64{0.5}, []float64{0.5}, 1, 1, 0)
}
