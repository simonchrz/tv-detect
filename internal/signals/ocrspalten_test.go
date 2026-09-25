package signals

import (
	"encoding/json"
	"os"
	"testing"
)

// Die OCR-Spalten muessen bitgleich zu scripts/ocr_spalten.py sein
// (Goldwerte aus scripts/gen-ocr-paritaet.py). Ein Versatz um eine Sekunde
// oder eine andere Rundung am Kachelende ware ein stiller Train/Serve-Bruch.
func TestOCRSpaltenParitaetMitPython(t *testing.T) {
	raw, err := os.ReadFile("testdata/ocr_paritaet.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Fenster  int             `json:"fenster"`
		NSek     int             `json:"n_sek"`
		Spur     json.RawMessage `json:"spur"`
		Erwartet [][3]float32    `json:"erwartet"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.Fenster != OCRFenster {
		t.Fatalf("Fenster Python %d ≠ Go %d", f.Fenster, OCRFenster)
	}
	o, err := OCRSpaltenAusSpur(f.Spur, f.NSek)
	if err != nil {
		t.Fatal(err)
	}
	abw := 0
	for s := 0; s < f.NSek; s++ {
		h, w, d := o.wert(s)
		e := f.Erwartet[s]
		if h != e[0] || w != e[1] || d != e[2] {
			if abw < 5 {
				t.Errorf("Sekunde %d: Go (%v,%v,%v) ≠ Python %v", s, h, w, d, e)
			}
			abw++
		}
	}
	if abw > 0 {
		t.Fatalf("%d von %d Sekunden weichen ab", abw, f.NSek)
	}
}

func TestOCRSpaltenOhneSpurNull(t *testing.T) {
	var o *OCRSpalten
	if h, w, d := o.wert(5); h+w+d != 0 {
		t.Fatal("nil-Spur liefert nicht 0")
	}
}
