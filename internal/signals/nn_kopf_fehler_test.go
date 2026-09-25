package signals

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// A head that is named but not loadable must fail the detector. Before
// the fix NewNNDetector only printed a line and returned a detector that
// answered 0.5 for every frame — the HSMM decoded a flat emission and the
// run exited 0 with a written ads.json.
//
// The backbone path is bogus on purpose: the head is checked first, so
// the error must name the head (the old code failed later, at the ORT
// session, with an unrelated message — or not at all with a real
// backbone).
func TestNNKopfNichtLadbarIstFatal(t *testing.T) {
	mlp1 := func(t *testing.T, dir string, nChan int) string {
		const H = 1
		inDim := nnFeatDim + 1 + 1 + nChan
		p := filepath.Join(dir, "head.bin")
		writeTestMLPHead(t, p, inDim, H, 1, 1, 1, nChan,
			make([]float32, inDim*H), []float32{0}, []float32{1}, []float32{0})
		return p
	}
	cases := map[string]func(t *testing.T, dir string) string{
		"fehlt": func(t *testing.T, dir string) string {
			return filepath.Join(dir, "head.bin")
		},
		"abgeschnitten": func(t *testing.T, dir string) string {
			p := mlp1(t, dir, 0)
			raw, _ := os.ReadFile(p)
			if err := os.WriteFile(p, raw[:len(raw)-100], 0o644); err != nil {
				t.Fatal(err)
			}
			return p
		},
		"channel-map fehlt": func(t *testing.T, dir string) string {
			return mlp1(t, dir, 2)
		},
		"minute-prior fehlt": func(t *testing.T, dir string) string {
			const H = 1
			inDim := nnFeatDim + 1 + 1 + 0 + 1 + 2 + 1
			p := filepath.Join(dir, "head.bin")
			writeTestMLPHeadV4(t, p, inDim, H, 1, 1, 1, 0, 1, 2, 1,
				make([]float32, inDim*H), []float32{0}, []float32{1}, []float32{0})
			return p
		},
	}
	for name, mk := range cases {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			head := mk(t, dir)
			d, err := NewNNDetector(filepath.Join(dir, "gibt-es-nicht.onnx"),
				head, 64, 64, "rtl")
			if err == nil {
				d.Close()
				t.Fatal("NewNNDetector accepted an unloadable head")
			}
			if !strings.Contains(err.Error(), "nn head") {
				t.Fatalf("error %q does not name the head — it failed "+
					"somewhere else, the head problem went unnoticed", err)
			}
		})
	}
}

type fehlerSession struct{}

func (fehlerSession) Run() error     { return errors.New("coreml: kaputt") }
func (fehlerSession) Destroy() error { return nil }

// A failing backbone pass must reach the caller as an error. It used to
// come back as nil, which runChunk turned into a silent all-0.5 chunk.
func TestEmbedBatchMeldetBackboneFehler(t *testing.T) {
	if err := initOrtRuntime(); err != nil {
		t.Skipf("onnxruntime nicht verfügbar: %v", err)
	}
	in, err := ort.NewEmptyTensor[float32](ort.NewShape(nnBatch, 3, nnInputH, nnInputW))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Destroy()
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(nnBatch, nnFeatDim))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Destroy()
	d := &NNDetector{session: fehlerSession{}, inTensor: in, outTensor: out,
		frameW: 4, frameH: 4}
	emb, err := d.EmbedBatch([][]byte{make([]byte, 4*4*3)})
	if err == nil {
		t.Fatalf("EmbedBatch returned %d floats and no error on a failed Run", len(emb))
	}
}
