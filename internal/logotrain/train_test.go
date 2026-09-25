package logotrain

import "testing"

// One sampled frame without any edge: before the fix the threshold was
// uint32(1*0.85) = 0, every pixel qualified and the bbox covered the whole
// frame. No edge in the frame → no logo.
func TestEinFrameErgibtNichtDieGanzeFlaeche(t *testing.T) {
	const w, h = 40, 30
	tr := New(Opts{FrameW: w, FrameH: h})
	tr.Push(make([]byte, w*h*3)) // flat black, no edges
	r := tr.Compute()
	if r.HasLogo {
		t.Fatalf("flat frame produced a logo: bbox (%d,%d)-(%d,%d), %d edge pixels",
			r.MinX, r.MinY, r.MaxX, r.MaxY, r.EdgePixels)
	}
}

// Control: a real edge in that single frame is still found.
func TestEinFrameMitKante(t *testing.T) {
	const w, h = 40, 30
	px := make([]byte, w*h*3)
	for y := range h {
		for x := 20; x < w; x++ {
			i := (y*w + x) * 3
			px[i], px[i+1], px[i+2] = 255, 255, 255
		}
	}
	tr := New(Opts{FrameW: w, FrameH: h})
	tr.Push(px)
	r := tr.Compute()
	if !r.HasLogo || r.MinX > 20 || r.MaxX < 19 || r.MaxX-r.MinX > 20 {
		t.Fatalf("vertical edge at x=20 not found tightly: %+v", r)
	}
}
