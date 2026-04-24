package signals

import "testing"

func TestSceneDetectorIdenticalNoCut(t *testing.T) {
	d := NewSceneDetector(25, 0.10)
	pix := make([]byte, 3*16*16)
	for i := range pix {
		pix[i] = 128
	}
	for i := 0; i < 5; i++ {
		d.Push(i, pix)
	}
	if len(d.Cuts()) != 0 {
		t.Fatalf("identical frames should produce no cuts, got %+v", d.Cuts())
	}
}

func TestSceneDetectorBlackToWhite(t *testing.T) {
	d := NewSceneDetector(25, 0.10)
	black := make([]byte, 3*16*16) // luma 0
	white := make([]byte, 3*16*16)
	for i := range white {
		white[i] = 255
	}
	d.Push(0, black)
	d.Push(1, white) // huge histogram shift → cut
	cuts := d.Cuts()
	if len(cuts) != 1 {
		t.Fatalf("want 1 cut on black→white, got %d", len(cuts))
	}
	if cuts[0].Frame != 1 {
		t.Errorf("cut at frame 1 expected, got %d", cuts[0].Frame)
	}
	if cuts[0].Distance < 0.99 {
		t.Errorf("expected dist ≈ 1 for opposite histograms, got %f", cuts[0].Distance)
	}
}
