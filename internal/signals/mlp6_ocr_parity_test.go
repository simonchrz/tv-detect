package signals

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Paritaet fuer den MLP6-Kopf mit OCR-Spalten (O26, L5-Vertrag): .bin aus
// dem ECHTEN write_mlp_head_v6, Spalten aus der ECHTEN ocr_spalten.py,
// Erwartung aus der Python-Vorwaertsrechnung
// (scripts/make_mlp6_ocr_parity_fixture.py). Geladen wird ueber den
// Produktions-Loader, die Spur ueber SetOCRSpalten wie im Detect.
//
// Zweiter Teil: OHNE Spur muss der Kopf die "ohne OCR"-Werte liefern —
// genau das, was er im Training fuer Aufnahmen ohne Spur gesehen hat.
func TestMLP6OCRParitaetMitTraining(t *testing.T) {
	td := filepath.Join("testdata", "mlp6-ocr")
	rohJ, err := os.ReadFile(td + "-parity.json")
	if err != nil {
		t.Fatalf("Fixture fehlt (%v) — scripts/make_mlp6_ocr_parity_fixture.py", err)
	}
	var fx struct {
		N            int             `json:"n"`
		Backbone     int             `json:"backbone"`
		Spur         json.RawMessage `json:"spur"`
		Embeds       []float64       `json:"embeds"`
		Logo         []float64       `json:"logo"`
		Rms          []float64       `json:"rms"`
		Erwartet     []float64       `json:"erwartet"`
		ErwartetOhne []float64       `json:"erwartet_ohne_ocr"`
	}
	if err := json.Unmarshal(rohJ, &fx); err != nil {
		t.Fatal(err)
	}
	d := &NNDetector{headPath: td + ".bin", mlpChanIdx: -1}
	if err := d.reloadHead(); err != nil {
		t.Fatalf("Produktions-Loader lehnt MLP6 ab: %v", err)
	}
	if !d.headIsMLP || d.mlpNOCR != 3 {
		t.Fatalf("headIsMLP=%v mlpNOCR=%d, will true/3", d.headIsMLP, d.mlpNOCR)
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
	o, err := OCRSpaltenAusSpur(fx.Spur, fx.N)
	if err != nil {
		t.Fatal(err)
	}
	d.SetOCRSpalten(o)
	vergleiche("mit Spur", d.ConfidenceChunk(embeds, fx.Logo, fx.Rms, fx.N, 1.0, 0), fx.Erwartet)
	d.SetOCRSpalten(nil)
	vergleiche("ohne Spur", d.ConfidenceChunk(embeds, fx.Logo, fx.Rms, fx.N, 1.0, 0), fx.ErwartetOhne)
}
