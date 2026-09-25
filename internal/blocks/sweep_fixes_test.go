package blocks

import (
	"testing"

	"github.com/simonchrz/tv-detect/internal/signals"
)

// A long block whose only blackframes sit near its edges must NOT be
// split there: the split left a sliver below MinBlockS (here 50 s and
// 20 s) that no later step removes.
func TestSplitErzeugtKeineWinzbloecke(t *testing.T) {
	black := []signals.BlackEvent{
		{StartS: 19.5, EndS: 20.5, DurationS: 1},
		{StartS: 949.5, EndS: 950.5, DurationS: 1},
	}
	got := splitLongBlocks([]Block{{0, 1000}}, 900, 60, black)
	for _, b := range got {
		if b.Duration() < 60 {
			t.Fatalf("split produced a %.0f s block (< MinBlockS 60): %+v", b.Duration(), got)
		}
	}
	// A blackframe with room on both sides is still used.
	black = append(black, signals.BlackEvent{StartS: 499.5, EndS: 500.5, DurationS: 1})
	got = splitLongBlocks([]Block{{0, 1000}}, 900, 60, black)
	if len(got) != 2 || got[0].EndS != 500 {
		t.Fatalf("want split at 500, got %+v", got)
	}
}

// Step 5 with crossing limits: a block shorter than MinBlockS next to its
// neighbour. The old clamp order let the min-duration clamp win — the
// second block started before the first ended and ran past the recording.
func TestExtendUeberlapptNieUndBleibtInDerAufnahme(t *testing.T) {
	const fps = 25.0
	nFrames := int(145 * fps)
	in := []Block{{0, 100}, {110, 140}}
	got := extendBlocks(in, Opts{FPS: fps, MinBlockS: 60, StartExtendS: 5, EndExtendS: 5}, nFrames)
	for i := 1; i < len(got); i++ {
		if got[i].StartS < got[i-1].EndS {
			t.Errorf("block %d starts %.1f before block %d ends %.1f", i, got[i].StartS, i-1, got[i-1].EndS)
		}
	}
	for _, b := range got {
		if b.EndS > 145 || b.StartS > b.EndS {
			t.Errorf("block %+v leaves the 145 s recording or is inverted", b)
		}
	}
}

// Unknown recording length (nFrames 0) is no bound — it used to clamp
// every last block to StartS+MinBlockS.
func TestExtendOhneLaengeKuerztNicht(t *testing.T) {
	got := extendBlocks([]Block{{100, 400}}, Opts{FPS: 25, MinBlockS: 60, EndExtendS: 10}, 0)
	if got[0].EndS != 410 {
		t.Fatalf("EndS = %v, want 410", got[0].EndS)
	}
}

// --max-ad-gap 0 must mean "off", as the flag help and CLAUDE.md say.
// defaults() turned 0 into 30, so two blocks 20 s apart were merged.
func TestMaxAdGapNullIstAus(t *testing.T) {
	const fps = 1.0
	nFrames := 600
	// ads 200-300 and 320-420, show otherwise
	logo := makeLogo(nFrames, [][2]int{{0, 200}, {300, 320}, {420, nFrames}})
	base := Opts{FPS: fps, MinShowSegmentS: 10, MinAbsentToOpenS: 5}
	aus := base
	aus.MaxAdGapS = 0
	if got := Form(aus, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames); len(got) != 2 {
		t.Errorf("MaxAdGapS=0: %d blocks %+v, want 2 (no merge)", len(got), got)
	}
	an := base
	an.MaxAdGapS = 30 // the CLI default: production behaviour unchanged
	if got := Form(an, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames); len(got) != 1 {
		t.Errorf("MaxAdGapS=30: %d blocks %+v, want 1 (merged)", len(got), got)
	}
}

// A fractional MinBlockS: the search used int(60.5)=60 and found a 60 s ad
// segment, the backtrace then dropped it (60 < 60.5) — the block vanished.
func TestHSMMGebrocheneMinBlockS(t *testing.T) {
	p := make([]float64, 900)
	for i := range p {
		p[i] = 0.001
	}
	for i := 400; i < 460; i++ { // exactly 60 s of clear ad
		p[i] = 0.999
	}
	got := FormHSMM(p, HSMMOpts{MinBlockS: 60.5, MaxBlockS: 900, DurW: 1})
	if len(got) != 1 || got[0].Duration() < 60.5 {
		t.Fatalf("got %+v, want one block >= 60.5 s around 400-460", got)
	}
}
