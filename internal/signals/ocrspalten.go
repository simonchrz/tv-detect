package signals

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// OCR-Spalten fuer den Kopf (MLP6, O26 ERFUELLT 2026-09-25).
//
// Drei Werte je Sekunde aus der flaechendeckenden OCR-Spur (tv-ocr-spur):
// hinweis_nah, werbung_nah (Treffer in +-OCRFenster s) und spur_da (Sekunde
// liegt in einer abgetasteten Kachel). Keine Spur → alle drei 0.
//
// ⚠️ MUSS scripts/ocr_spalten.py (aus_spur) exakt entsprechen — das
// Training baut die Spalten dort, hier baut der Detect sie nach. Eine
// Abweichung waere ein stiller Train/Serve-Bruch: der Kopf saehe im
// Betrieb eine andere Spalte als im Training, und das aeussert sich als
// "etwas schlechter", nie als Fehler. ocrspalten_test.go haelt beide ueber
// Goldwerte aus scripts/gen-ocr-paritaet.py zusammen.

// OCRFenster ist das Halbfenster in Sekunden (O26-Registrierung).
const OCRFenster = 10

// OCRSpalten haelt die drei Spalten je absoluter Sekunde.
type OCRSpalten struct {
	Hinweis, Werbung, Da []float32
}

type ocrSpurDatei struct {
	DauerS     float64 `json:"dauer_s"`
	Abgetastet []struct {
		Von float64 `json:"von"`
		Bis float64 `json:"bis"`
	} `json:"abgetastet"`
	Funde []OCRFund `json:"funde"`
}

// OCRSpaltenAusSpur baut die Spalten fuer nSek Sekunden aus dem JSON einer
// Spur. Gleiche Rundung wie Python: int() schneidet ab (fuer die hier
// vorkommenden nicht-negativen Werte = Abrunden), ceil fuer das Kachelende.
func OCRSpaltenAusSpur(raw []byte, nSek int) (*OCRSpalten, error) {
	var s ocrSpurDatei
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("ocr-spur: %w", err)
	}
	if nSek < 0 {
		nSek = 0
	}
	o := &OCRSpalten{
		Hinweis: make([]float32, nSek),
		Werbung: make([]float32, nSek),
		Da:      make([]float32, nSek),
	}
	for _, k := range s.Abgetastet {
		a := max(0, int(k.Von))
		b := min(nSek, int(math.Ceil(k.Bis)))
		for t := a; t < b; t++ {
			o.Da[t] = 1
		}
	}
	for _, f := range s.Funde {
		t := int(f.TimeS)
		lo, hi := max(0, t-OCRFenster), min(nSek, t+OCRFenster+1)
		for i := lo; i < hi; i++ {
			if f.Hinweis {
				o.Hinweis[i] = 1
			}
			if f.Werbemarker {
				o.Werbung[i] = 1
			}
		}
	}
	return o, nil
}

// LadeOCRSpur liest eine Spur-Datei. Die Laenge folgt aus dauer_s (+1 als
// Rand); Sekunden dahinter liest der Kopf als 0 — wie Python, das nur bis
// zur Zahl der Merkmalszeilen fuellt.
func LadeOCRSpur(pfad string) (*OCRSpalten, error) {
	raw, err := os.ReadFile(pfad)
	if err != nil {
		return nil, err
	}
	var kopf struct {
		DauerS float64 `json:"dauer_s"`
	}
	if err := json.Unmarshal(raw, &kopf); err != nil {
		return nil, fmt.Errorf("ocr-spur %s: %w", pfad, err)
	}
	return OCRSpaltenAusSpur(raw, int(math.Ceil(kopf.DauerS))+1)
}

// wert liefert die drei Spalten an Sekunde t; ausserhalb → 0.
func (o *OCRSpalten) wert(t int) (h, w, d float32) {
	if o == nil || t < 0 || t >= len(o.Da) {
		return 0, 0, 0
	}
	return o.Hinweis[t], o.Werbung[t], o.Da[t]
}
