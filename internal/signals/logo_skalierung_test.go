package signals

import (
	"math/rand"
	"testing"

	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

// oldConfidence is the pre-2026-10-18 walk (template coordinates used
// unscaled), kept here as the parity reference.
func oldConfidence(t *logotemplate.Template, pixels []byte, frameW, frameH, edgeTh, yOffset int) float64 {
	matches := 0
	for ry := range t.Height() {
		fy := t.MinY + ry + yOffset
		if fy <= 0 || fy >= frameH-1 {
			continue
		}
		for rx := range t.Width() {
			if !t.Mask[ry][rx] {
				continue
			}
			fx := t.MinX + rx
			if fx <= 0 || fx >= frameW-1 {
				continue
			}
			if sobelMag(pixels, 3*frameW, fx, fy) > edgeTh {
				matches++
			}
		}
	}
	return float64(matches) / float64(t.EdgePositions)
}

func testTemplate(rng *rand.Rand, minX, minY, w, h, picW, picH int) *logotemplate.Template {
	t := &logotemplate.Template{MinX: minX, MaxX: minX + w, MinY: minY, MaxY: minY + h, PicWidth: picW, PicHeight: picH}
	for range h {
		row := make([]bool, w)
		for x := range row {
			if rng.Intn(3) == 0 {
				row[x] = true
				t.EdgePositions++
			}
		}
		t.Mask = append(t.Mask, row)
	}
	return t
}

// At equal frame and template size the detector must give bit-identical
// values to the old walk — that covers every broadcast 720x576 recording.
func TestLogoSkalierungParitaetGleicheGroesse(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const W, H = 720, 576
	for _, yOff := range []int{0, 12} {
		tmpl := testTemplate(rng, 600, 30, 80, 40, W, H)
		d, err := NewLogoDetector(tmpl, W, H, 0, yOff)
		if err != nil {
			t.Fatal(err)
		}
		for range 20 {
			px := make([]byte, 3*W*H)
			rng.Read(px)
			got := d.Confidence(px)
			want := oldConfidence(tmpl, px, W, H, defaultEdgeThresh, yOff)
			if got != want {
				t.Fatalf("yOffset %d: %v != alt %v", yOff, got, want)
			}
		}
	}
}

// Header without picWidth/picHeight (0) → unscaled, as before.
func TestLogoSkalierungOhneKopfgroesse(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	tmpl := testTemplate(rng, 600, 30, 80, 40, 0, 0)
	d, err := NewLogoDetector(tmpl, 1046, 576, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	px := make([]byte, 3*1046*576)
	rng.Read(px)
	if got, want := d.Confidence(px), oldConfidence(tmpl, px, 1046, 576, defaultEdgeThresh, 0); got != want {
		t.Fatalf("%v != %v", got, want)
	}
}

// A 720-wide template on a 1046-wide frame: an edge drawn at the scaled
// position of every mask cell must be found (confidence 1), and the old
// unscaled position must no longer be what is read.
func TestLogoSkalierungBreiteQuelle(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const W, H = 1046, 576
	tmpl := testTemplate(rng, 600, 30, 60, 30, 720, 576)
	// sparse mask: every 6th column, every 3rd row, so the drawn edges
	// of neighbouring cells don't cancel each other's gradient
	tmpl.EdgePositions = 0
	for ry, row := range tmpl.Mask {
		for rx := range row {
			row[rx] = rx%6 == 0 && ry%3 == 0
			if row[rx] {
				tmpl.EdgePositions++
			}
		}
	}
	d, err := NewLogoDetector(tmpl, W, H, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	px := make([]byte, 3*W*H)
	for _, p := range d.pts {
		// bright pixel right of the cell → strong horizontal gradient
		for c := range 3 {
			px[p.y*3*W+3*(p.x+1)+c] = 255
		}
	}
	if got := d.Confidence(px); got != 1 {
		t.Fatalf("skaliert: Konfidenz %v, erwartet 1", got)
	}
	if p := d.pts[0]; p.x < 870 {
		t.Fatalf("erste Zelle bei x=%d, erwartet ~600*1046/720=872", p.x)
	}
	if old := oldConfidence(tmpl, px, W, H, defaultEdgeThresh, 0); old > 0.2 {
		t.Fatalf("alte Lesart traf zufaellig %v", old)
	}
}
