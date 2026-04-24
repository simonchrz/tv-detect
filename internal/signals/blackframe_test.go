package signals

import (
	"math"
	"testing"
)

func TestIsBlackFrame(t *testing.T) {
	w, h := 16, 16
	rgb := make([]byte, 3*w*h)

	// All-zero pixels → black.
	if !IsBlackFrame(rgb, 0, 0) {
		t.Fatal("all-zero frame should be black")
	}

	// All bright white → not black.
	for i := range rgb {
		rgb[i] = 255
	}
	if IsBlackFrame(rgb, 0, 0) {
		t.Fatal("all-white frame should not be black")
	}

	// Threshold edge: 98% of pixels at luma=20 (below 25 default), 2% white.
	// Should be classified as black.
	for i := range rgb {
		rgb[i] = 20
	}
	rgb[0], rgb[1], rgb[2] = 255, 255, 255
	rgb[3], rgb[4], rgb[5] = 255, 255, 255
	rgb[6], rgb[7], rgb[8] = 255, 255, 255
	rgb[9], rgb[10], rgb[11] = 255, 255, 255
	rgb[12], rgb[13], rgb[14] = 255, 255, 255 // 5/256 ≈ 2% white, rest black
	if !IsBlackFrame(rgb, 0, 0) {
		t.Fatal("frame with 98% black pixels should be black")
	}
}

func TestBlackDetectorAggregate(t *testing.T) {
	d := NewBlackDetector(25.0, 0.10, 0, 0)
	w, h := 8, 8
	black := make([]byte, 3*w*h)
	white := make([]byte, 3*w*h)
	for i := range white {
		white[i] = 255
	}

	// Frames 0-5 white, 6-15 black, 16-30 white, 31-32 black (too short).
	for i := 0; i < 33; i++ {
		switch {
		case i >= 6 && i <= 15:
			d.Push(i, black)
		case i >= 31:
			d.Push(i, black)
		default:
			d.Push(i, white)
		}
	}
	d.Finish()

	events := d.Events()
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d: %+v", len(events), events)
	}
	want := BlackEvent{StartS: 6.0 / 25, EndS: 16.0 / 25, DurationS: 10.0 / 25}
	if !nearly(events[0].StartS, want.StartS) ||
		!nearly(events[0].EndS, want.EndS) ||
		!nearly(events[0].DurationS, want.DurationS) {
		t.Fatalf("want %+v, got %+v", want, events[0])
	}
}

func nearly(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
