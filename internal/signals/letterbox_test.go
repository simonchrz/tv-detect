package signals

import "testing"

const tw, th = 32, 32 // tiny test frames
const trows = 4       // 4-row bars

func makeFrame(letterboxed bool) []byte {
	pix := make([]byte, tw*th*3)
	for i := range pix {
		pix[i] = 200 // bright content default
	}
	if letterboxed {
		// Zero out top and bottom 4 rows.
		rowBytes := tw * 3
		for y := 0; y < trows; y++ {
			for x := 0; x < rowBytes; x++ {
				pix[y*rowBytes+x] = 0
			}
		}
		for y := th - trows; y < th; y++ {
			for x := 0; x < rowBytes; x++ {
				pix[y*rowBytes+x] = 0
			}
		}
	}
	return pix
}

func TestLetterboxOnsetEmitted(t *testing.T) {
	d := NewLetterboxDetector(25, tw, th, trows, 16)
	// 30 frames clean, 30 letterboxed → must emit exactly one onset.
	for i := 0; i < 30; i++ {
		d.Push(i, makeFrame(false))
	}
	for i := 30; i < 60; i++ {
		d.Push(i, makeFrame(true))
	}
	ev := d.Events()
	if len(ev) != 1 {
		t.Fatalf("want 1 event, got %d: %+v", len(ev), ev)
	}
	if !ev[0].Onset {
		t.Errorf("expected onset, got offset")
	}
	// Onset frame should land near 30 (first letterboxed frame), not at
	// the hysteresis-confirmation frame.
	if ev[0].Frame < 28 || ev[0].Frame > 32 {
		t.Errorf("onset frame %d not near 30", ev[0].Frame)
	}
}

func TestLetterboxNoFlickerOnDarkScene(t *testing.T) {
	d := NewLetterboxDetector(25, tw, th, trows, 16)
	clean := makeFrame(false)
	letter := makeFrame(true)
	// 20 clean, 3 letterboxed (transient), 20 clean → should NOT emit.
	for i := 0; i < 20; i++ {
		d.Push(i, clean)
	}
	for i := 20; i < 23; i++ {
		d.Push(i, letter)
	}
	for i := 23; i < 43; i++ {
		d.Push(i, clean)
	}
	if len(d.Events()) != 0 {
		t.Fatalf("transient letterbox under hysteresis should not emit, got %+v", d.Events())
	}
}

func TestLetterboxOnsetThenOffset(t *testing.T) {
	d := NewLetterboxDetector(25, tw, th, trows, 16)
	for i := 0; i < 30; i++ {
		d.Push(i, makeFrame(false))
	}
	for i := 30; i < 90; i++ {
		d.Push(i, makeFrame(true))
	}
	for i := 90; i < 120; i++ {
		d.Push(i, makeFrame(false))
	}
	ev := d.Events()
	if len(ev) != 2 {
		t.Fatalf("want 2 events (onset+offset), got %d: %+v", len(ev), ev)
	}
	if !ev[0].Onset || ev[1].Onset {
		t.Errorf("expected onset then offset, got %+v", ev)
	}
}
