package signals

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Paritäts-Fixture für den NACKTEN MLP1-Kopf — der Architekturwechsel-
// Kandidat seit O2/O6/O7/O8 (2026-08-12/13): input = backbone + logo +
// audio, KEIN Kanal-Block, kein Whisper, kein Temporal.
//
// Die .bin stammt aus dem ECHTEN write_mlp_head_v1 des Trainers
// (scripts/make_mlp1_bare_parity_fixture.py, Seed 42), die Erwartungswerte
// aus der Python-Vorwärtsrechnung. Dieser Test lädt sie über den
// PRODUKTIONS-Loader (reloadHead → loadMLPHead) und rechnet über
// ConfidenceChunk nach — das ist der L5-Vertrag: Python-Fit und
// Go-Inferenz müssen dieselben Zahlen liefern, BEVOR die Architektur
// wechselt.
func TestMLP1BareParitaetMitTraining(t *testing.T) {
	td := filepath.Join("testdata", "mlp1-bare")
	rohJ, err := os.ReadFile(td + "-parity.json")
	if err != nil {
		t.Skipf("Fixture fehlt (%v) — scripts/make_mlp1_bare_parity_fixture.py", err)
	}
	var fx struct {
		N        int       `json:"n"`
		Backbone int       `json:"backbone"`
		Embeds   []float64 `json:"embeds"`
		Logo     []float64 `json:"logo"`
		Rms      []float64 `json:"rms"`
		Erwartet []float64 `json:"erwartet"`
	}
	if err := json.Unmarshal(rohJ, &fx); err != nil {
		t.Fatal(err)
	}
	if fx.Backbone != nnFeatDim {
		t.Fatalf("Fixture backbone %d != nnFeatDim %d — Fixture neu erzeugen",
			fx.Backbone, nnFeatDim)
	}

	d := &NNDetector{headPath: td + ".bin", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("Produktions-Loader lehnt die bare .bin ab: %v", err)
	}
	if !d.headIsMLP {
		t.Fatal("headIsMLP=false — die Fixture lief in den LogReg-Pfad")
	}
	if d.mlpNChannel != 0 {
		t.Fatalf("mlpNChannel=%d, will 0 — der nackte Kopf hat keinen "+
			"Kanal-Block", d.mlpNChannel)
	}

	embeds := make([]float32, len(fx.Embeds))
	for i, v := range fx.Embeds {
		embeds[i] = float32(v)
	}
	got := d.ConfidenceChunk(embeds, fx.Logo, fx.Rms, fx.N, 1.0, 0)
	if len(got) != fx.N {
		t.Fatalf("ConfidenceChunk lieferte %d Werte, will %d", len(got), fx.N)
	}
	for i := range got {
		if diff := math.Abs(got[i] - fx.Erwartet[i]); diff > 1e-6 {
			t.Errorf("Frame %d: Go %.9f vs Python %.9f (Δ %.2e) — "+
				"Paritaet gebrochen", i, got[i], fx.Erwartet[i], diff)
		}
	}
}
