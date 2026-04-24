package blocks

import (
	"testing"

	"github.com/simonchrz/tv-detect/internal/signals"
)

// makeLogo builds a logoConf array where positions in `presentRanges`
// (start, end frame pairs) have conf 1.0, all others 0.0.
func makeLogo(nFrames int, presentRanges [][2]int) []float64 {
	out := make([]float64, nFrames)
	for _, r := range presentRanges {
		for i := r[0]; i < r[1] && i < nFrames; i++ {
			out[i] = 1.0
		}
	}
	return out
}

func TestFormBasicAdBlock(t *testing.T) {
	// 30 min recording at 25 fps:
	//   show 0..600 s (frames 0..15000)
	//   ad   600..900 s (frames 15000..22500)
	//   show 900..1800 s (frames 22500..45000)
	fps := 25.0
	nFrames := 45000
	logo := makeLogo(nFrames,
		[][2]int{{0, 15000}, {22500, nFrames}})

	blocks := Form(Opts{FPS: fps}, logo, nil, nil, nFrames)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d: %+v", len(blocks), blocks)
	}
	if !near(blocks[0].StartS, 600, 0.5) || !near(blocks[0].EndS, 900, 0.5) {
		t.Errorf("block bounds want [600,900], got %+v", blocks[0])
	}
}

func TestFormFiltersShortBlocks(t *testing.T) {
	// 30s ad block — below default 60s min, should be filtered.
	fps := 25.0
	nFrames := 25 * 1800
	logo := makeLogo(nFrames,
		[][2]int{{0, 25 * 600}, {25 * 630, nFrames}})

	blocks := Form(Opts{FPS: fps}, logo, nil, nil, nFrames)
	if len(blocks) != 0 {
		t.Errorf("30s gap should be filtered, got %+v", blocks)
	}
}

func TestFormBoundaryRefineToBlackframe(t *testing.T) {
	// Logo says ad runs 600..900. The real hard-cut blackframes are at
	// 603 (just after the rough start — typical: logo fades a few seconds
	// before the actual cut) and 897 (just before the rough end — typical:
	// we wait MinShowSegmentS of confirmed-present before declaring end).
	// Refiner should snap to both.
	fps := 25.0
	nFrames := 25 * 1800
	logo := makeLogo(nFrames,
		[][2]int{{0, 25 * 600}, {25 * 900, nFrames}})
	black := []signals.BlackEvent{
		{StartS: 602.5, EndS: 603.0, DurationS: 0.5},
		{StartS: 897.0, EndS: 897.5, DurationS: 0.5},
	}

	blocks := Form(Opts{FPS: fps, RefineWindowS: 10}, logo, black, nil, nFrames)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %+v", blocks)
	}
	if !near(blocks[0].StartS, 603.0, 0.001) {
		t.Errorf("start should snap forward to black end 603.0, got %f", blocks[0].StartS)
	}
	if !near(blocks[0].EndS, 897.0, 0.001) {
		t.Errorf("end should snap backward to black start 897.0, got %f", blocks[0].EndS)
	}
}

func near(a, b, eps float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}
