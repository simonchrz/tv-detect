package signals_test

import (
	"math/rand"
	"os"
	"testing"

	"github.com/simonchrz/tv-detect/internal/signals"
)

func TestBoundaryDetectorRealFile(t *testing.T) {
	// Loads the deployed boundary_head.bin (= 961 KB) and runs forward
	// pass on synthetic embeddings. Validates BNDR-format parsing and
	// shape/dim sanity. Real-quality eval happens via the per-recording
	// recall metric in train-boundary-head.py.
	//
	// ⚠️ Haengt an einem Artefakt in /tmp, das kein Bauschritt erzeugt —
	// nach einem Neustart ist es weg und der Test dauerhaft rot. Eine immer
	// rote Suite ist keine Absicherung, sondern Rauschen: sie hat am
	// 2026-08-07 verdeckt, dass ein ZWEITER Test (die Unruhe-Spalte) durch
	// eine echte Aenderung gebrochen war. Fehlt die Datei, wird deshalb
	// uebersprungen statt gemeldet.
	const kopf = "/tmp/boundary_head.bin"
	if _, err := os.Stat(kopf); err != nil {
		t.Skipf("kein %s — der Boundary-Head ist ohnehin deaktiviert "+
			"(Memory boundary_head_activated)", kopf)
	}
	d, err := signals.NewBoundaryDetector(kopf)
	if err != nil {
		t.Fatalf("NewBoundaryDetector: %v", err)
	}
	if d == nil || !d.IsLoaded() {
		t.Fatal("head not loaded")
	}
	r := rand.New(rand.NewSource(42))
	embeds := make([][]float32, 60)
	for i := range embeds {
		embeds[i] = make([]float32, 1280)
		for k := range embeds[i] {
			embeds[i][k] = r.Float32()
		}
	}
	scores := d.BoundaryScores(embeds, 1.0)
	if len(scores) != 60 {
		t.Fatalf("scores len %d != 60", len(scores))
	}
	for i, s := range scores {
		if s < 0 || s > 1 {
			t.Errorf("frame %d score %v out of [0,1]", i, s)
		}
	}
	t.Logf("60-frame run ok; first 5 scores: %.3f %.3f %.3f %.3f %.3f",
		scores[0], scores[1], scores[2], scores[3], scores[4])
}
