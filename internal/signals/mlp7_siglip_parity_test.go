package signals

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Paritaet fuer den MLP7-Kopf mit SigLIP-Spalten (docs/siglip-spur-design.md):
// .bin aus dem ECHTEN write_mlp_head_v7, Spalten aus der ECHTEN
// siglip_spalten.py, Spur als float16-.npy wie siglip-spur.py sie schreibt
// (scripts/make_mlp7_siglip_parity_fixture.py). Geladen wird ueber den
// Produktions-Loader, die Spur ueber LadeSigLIPSpur + SetSigLIPSpur.
//
// Die Spur ist kuerzer als die Aufnahme: dahinter muss der Kopf die
// "ohne Spur"-Werte liefern, so wie im Training.
func TestMLP7SigLIPParitaetMitTraining(t *testing.T) {
	td := filepath.Join("testdata", "mlp7-siglip")
	rohJ, err := os.ReadFile(td + "-parity.json")
	if err != nil {
		t.Fatalf("Fixture fehlt (%v) — scripts/make_mlp7_siglip_parity_fixture.py", err)
	}
	var fx struct {
		N            int       `json:"n"`
		NSpur        int       `json:"n_spur"`
		Embeds       []float64 `json:"embeds"`
		Logo         []float64 `json:"logo"`
		Rms          []float64 `json:"rms"`
		Erwartet     []float64 `json:"erwartet"`
		ErwartetOhne []float64 `json:"erwartet_ohne_spur"`
		SpaltenT5    []float64 `json:"spalten_t5"`
	}
	if err := json.Unmarshal(rohJ, &fx); err != nil {
		t.Fatal(err)
	}
	d := &NNDetector{headPath: td + ".bin", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("Produktions-Loader lehnt MLP7 ab: %v", err)
	}
	if !d.headIsMLP || d.mlpNOCR != 3 || d.mlpNSigLIP != SigLIPKomponenten+1 || !d.BrauchtSigLIP() {
		t.Fatalf("headIsMLP=%v mlpNOCR=%d mlpNSigLIP=%d", d.headIsMLP, d.mlpNOCR, d.mlpNSigLIP)
	}
	sp, err := LadeSigLIPSpur(td + "-spur.npy")
	if err != nil {
		t.Fatal(err)
	}
	if sp.N != fx.NSpur {
		t.Fatalf("Spur %d Zeilen, will %d", sp.N, fx.NSpur)
	}
	// Spalten einzeln (Projektion + siglip_da), bevor der Kopf sie mischt
	x := make([]float32, SigLIPKomponenten+1)
	sp.spalten(5, d.mlpSigMu, d.mlpSigV, x)
	for j, want := range fx.SpaltenT5 {
		if math.Abs(float64(x[j])-want) > 1e-3*math.Max(1, math.Abs(want)) {
			t.Fatalf("Spalte %d bei t=5: Go %.6f ≠ Python %.6f", j, x[j], want)
		}
	}
	embeds := make([]float32, len(fx.Embeds))
	for i, v := range fx.Embeds {
		embeds[i] = float32(v)
	}
	vergleiche := func(name string, got, want []float64) {
		t.Helper()
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-5 {
				t.Fatalf("%s Frame %d: Go %.7f ≠ Python %.7f", name, i, got[i], want[i])
			}
		}
	}
	d.SetSigLIPSpur(sp)
	vergleiche("mit Spur", d.ConfidenceChunk(embeds, fx.Logo, fx.Rms, fx.N, 1.0, 0), fx.Erwartet)
	d.SetSigLIPSpur(nil)
	vergleiche("ohne Spur", d.ConfidenceChunk(embeds, fx.Logo, fx.Rms, fx.N, 1.0, 0), fx.ErwartetOhne)
}

// float16 → float32 muss bitgenau zu numpy passen (die Spur liegt als f2).
func TestHalbZuFloat(t *testing.T) {
	for h, want := range map[uint16]float32{
		0x0000: 0, 0x3c00: 1, 0xc000: -2, 0x3555: 0.33325195, 0x0001: 5.9604645e-08,
		0x7bff: 65504, 0x0400: 6.1035156e-05,
	} {
		if got := halbZuFloat(h); got != want {
			t.Errorf("halbZuFloat(0x%04x) = %g, will %g", h, got, want)
		}
	}
}

// Ein v6-Kopf ignoriert die Spur (liest keine SigLIP-Spalten).
func TestMLP6BrauchtKeinSigLIP(t *testing.T) {
	d := &NNDetector{headPath: filepath.Join("testdata", "mlp6-ocr.bin"), mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatal(err)
	}
	if d.BrauchtSigLIP() {
		t.Fatal("v6-Kopf meldet SigLIP-Bedarf")
	}
}
