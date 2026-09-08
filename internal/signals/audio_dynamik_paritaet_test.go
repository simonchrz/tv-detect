package signals

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Die Audio-Dynamik ist die erste Aenderung, die BEIDE Seiten betrifft:
// Python trainiert den Kopf darauf, Go liefert die Zahl zur Inferenzzeit.
// Weichen sie ab, stuerzt nichts ab — der Kopf bekommt nur still eine
// andere Zahl, als er gelernt hat, und die Bloecke werden schlechter.
// Genau die Sorte Fehler, die in diesem Projekt schon zweimal Monate
// gekostet hat (merkmalsspalte_verschoben_auffueller, logo NaN).
//
// Die Fixture schreibt die PYTHON-Seite (scripts/train-head.py,
// audio_dynamik). Dieser Test haelt die Go-Seite dagegen. Wer eine der
// beiden Rechnungen aendert, muss die Fixture neu erzeugen — und merkt
// dabei, dass es zwei Seiten sind.
func TestAudioDynamikParitaet(t *testing.T) {
	roh, err := os.ReadFile("testdata/audio_dynamik_paritaet.json")
	if err != nil {
		t.Fatalf("Fixture fehlt: %v", err)
	}
	var faelle []struct {
		Name    string    `json:"name"`
		Fenster int       `json:"fenster"`
		Ein     []float64 `json:"ein"`
		Soll    []float64 `json:"soll"`
	}
	if err := json.Unmarshal(roh, &faelle); err != nil {
		t.Fatal(err)
	}
	if len(faelle) == 0 {
		t.Fatal("Fixture ist leer")
	}
	for _, f := range faelle {
		t.Run(f.Name, func(t *testing.T) {
			ein := make([]float32, len(f.Ein))
			for i, v := range f.Ein {
				ein[i] = float32(v)
			}
			ist := AudioDynamik(ein, f.Fenster)
			if len(ist) != len(f.Soll) {
				t.Fatalf("Laenge %d, erwartet %d", len(ist), len(f.Soll))
			}
			for i := range ist {
				// 1e-5 ist grosszuegig gegenueber float32-Rundung und
				// eng genug, um eine andere FORMEL zu fangen (falscher
				// Nenner, verschobenes Fenster, aufgefuellter Rand).
				if math.Abs(float64(ist[i])-f.Soll[i]) > 1e-5 {
					t.Fatalf("Index %d: Go %.6f, Python %.6f",
						i, ist[i], f.Soll[i])
				}
			}
		})
	}
}
