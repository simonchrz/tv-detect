package signals

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Paritäts-Fixture für den ENSEMBLE-Kopf (2026-08-27).
//
// Der Nightly liefert seit dem 2026-08-27 nicht mehr einen ausgewählten
// Init-Seed aus, sondern k Seeds zusammengelegt zu EINEM Kopf, der ihren
// Logit-Mittelwert rechnet. Das ist bewusst kein neues Dateiformat, sondern
// ein normaler MLP1-v1-Kopf mit hidden_dim = k*32 — die ganze Umstellung
// hängt daran, dass die Go-Seite hidden_dim wirklich aus dem Header liest
// und nirgends 32 annimmt.
//
// Die .bin stammt aus dem ECHTEN write_mlp_head_v1 des Trainers, gefüttert
// mit dem ECHTEN merge_mlp_ensemble
// (scripts/make_mlp1_ensemble_parity_fixture.py, Seed 4242). Die
// Erwartungswerte kommen aus den DREI EINZELKÖPFEN, nicht aus dem Merge —
// dieser Test prüft also beides zugleich: dass die Zusammenlegung wirklich
// den Mittelwert rechnet, und dass der Produktions-Loader den breiteren
// Kopf korrekt liest.
func TestMLP1EnsembleParitaetMitTraining(t *testing.T) {
	td := filepath.Join("testdata", "mlp1-ensemble")
	rohJ, err := os.ReadFile(td + "-parity.json")
	if err != nil {
		t.Skipf("Fixture fehlt (%v) — scripts/make_mlp1_ensemble_parity_fixture.py", err)
	}
	var fx struct {
		N        int       `json:"n"`
		Backbone int       `json:"backbone"`
		Seeds    int       `json:"seeds"`
		Hidden   int       `json:"hidden"`
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
		t.Fatalf("Produktions-Loader lehnt den Ensemble-Kopf ab: %v — "+
			"das ist der Fall, den die Umstellung nicht haben darf", err)
	}
	if !d.headIsMLP {
		t.Fatal("headIsMLP=false — die Fixture lief in den LogReg-Pfad")
	}
	// Der Kern: hidden kommt aus dem Header, nicht aus einer Annahme.
	if d.mlpHidden != fx.Hidden {
		t.Fatalf("mlpHidden=%d, will %d (= %d Seeds x 32) — die Go-Seite "+
			"nimmt eine feste Breite an", d.mlpHidden, fx.Hidden, fx.Seeds)
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
			t.Errorf("Frame %d: Go %.9f vs Mittel der Einzelkoepfe %.9f "+
				"(Δ %.2e) — Ensemble-Paritaet gebrochen",
				i, got[i], fx.Erwartet[i], diff)
		}
	}
}
