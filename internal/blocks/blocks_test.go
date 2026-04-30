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

	blocks := Form(Opts{FPS: fps}, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames)
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

	blocks := Form(Opts{FPS: fps}, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames)
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

	blocks := Form(Opts{FPS: fps, RefineWindowS: 10}, logo, nil, nil, nil, nil, black, nil, nil, nil, nil, nFrames)
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

func TestSnapToIFrame(t *testing.T) {
	iframes := []float64{10.0, 22.5, 60.0, 600.0}
	cases := []struct {
		name    string
		t       float64
		radius  float64
		wantT   float64
	}{
		{"exact match", 22.5, 5, 22.5},
		{"snap forward", 21.0, 5, 22.5},
		{"snap backward", 24.0, 5, 22.5},
		{"out of radius — return unchanged", 100.0, 5, 100.0},
		{"prefer closer of two", 41.0, 30, 22.5}, // 41-22.5=18.5 vs 60-41=19
		{"empty iframes — passthrough", 50.0, 5, 50.0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fr := iframes
			if c.name == "empty iframes — passthrough" {
				fr = nil
			}
			got := snapToIFrame(c.t, fr, c.radius)
			if !near(got, c.wantT, 1e-6) {
				t.Errorf("snapToIFrame(%.1f, _, %.1f) = %.3f, want %.3f",
					c.t, c.radius, got, c.wantT)
			}
		})
	}
}

func TestSnapToBumper(t *testing.T) {
	// 25 fps; bumper peaks 0.95 at frames 100..120 (= 4.0 .. 4.8 s),
	// background at 0.10 elsewhere. snapToBumper should return the
	// FIRST frame after the last peak: frame 121 / 25 = 4.84 s.
	fps := 25.0
	conf := make([]float64, 250)
	for i := 100; i <= 120; i++ {
		conf[i] = 0.95
	}
	cases := []struct {
		name      string
		t         float64
		radius    float64
		threshold float64
		want      float64
	}{
		{"end inside bumper window", 4.5, 5, 0.85, 4.84},
		{"end after bumper window", 6.0, 5, 0.85, 4.84},
		{"end before bumper window", 3.0, 5, 0.85, 4.84},
		{"radius too small — passthrough", 7.0, 0.5, 0.85, 7.0},
		{"threshold too high — passthrough", 4.5, 5, 0.99, 4.5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := snapToBumper(c.t, conf, fps, c.radius, c.threshold)
			if !near(got, c.want, 0.05) {
				t.Errorf("snapToBumper(%.1f, _, %.1f, %.2f) = %.3f, want %.3f",
					c.t, c.radius, c.threshold, got, c.want)
			}
		})
	}
}

func TestSnapToBumperStart(t *testing.T) {
	// Start-bumper peaks 0.95 at frames 100..120 (= 4.0 .. 4.8 s).
	// snapToBumperStart returns the EARLIEST high-conf frame in
	// window: frame 100 / 25 = 4.0 s — that's the first ad frame.
	fps := 25.0
	conf := make([]float64, 250)
	for i := 100; i <= 120; i++ {
		conf[i] = 0.95
	}
	cases := []struct {
		name      string
		t         float64
		radius    float64
		threshold float64
		want      float64
	}{
		{"start inside bumper window", 4.5, 5, 0.85, 4.0},
		{"start before bumper window", 2.0, 5, 0.85, 4.0},
		{"start after bumper window", 7.0, 5, 0.85, 4.0},
		{"radius too small — passthrough", 9.0, 0.5, 0.85, 9.0},
		{"threshold too high — passthrough", 4.5, 5, 0.99, 4.5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := snapToBumperStart(c.t, conf, fps, c.radius, c.threshold)
			if !near(got, c.want, 0.05) {
				t.Errorf("snapToBumperStart(%.1f, _, %.1f, %.2f) = %.3f, want %.3f",
					c.t, c.radius, c.threshold, got, c.want)
			}
		})
	}
}

func TestRefineBoundaryVotingMultiSignalAgreement(t *testing.T) {
	// All three sources cluster around t=120: blackframe at 120.0,
	// silence ending at 120.5, scene-cut at 119.8. Weighted vote
	// total = 4.5 — clearly the winner. A lone scene-cut at 100s
	// (weight 1.5) inside the same radius shouldn't beat it.
	black := []signals.BlackEvent{
		{StartS: 120.0, EndS: 120.3, DurationS: 0.3},
	}
	silence := []signals.SilenceEvent{
		{StartS: 119.8, EndS: 120.5, DurationS: 0.7},
	}
	scenes := []signals.SceneCut{
		{TimeS: 119.8, Distance: 0.5},
		{TimeS: 100.0, Distance: 0.5},
	}
	// rough boundary 121s, isStart=false (so anchor preference is BACKWARD)
	got := refineBoundaryVoting(121.0, 30.0, black, silence, scenes, nil, false)
	if !near(got, 119.8, 0.6) && !near(got, 120.0, 0.6) && !near(got, 120.5, 0.6) {
		t.Errorf("multi-signal cluster around t≈120 expected, got %.3f", got)
	}
}

func TestRefineBoundaryVotingFallthroughEmpty(t *testing.T) {
	// No candidates at all → return roughS unchanged.
	got := refineBoundaryVoting(500.0, 30, nil, nil, nil, nil, true)
	if got != 500.0 {
		t.Errorf("empty signals: want passthrough 500, got %.3f", got)
	}
}

func TestRefineBoundaryVotingDirectionalPenalty(t *testing.T) {
	// isStart=true (ad-start): real cut is typically AFTER the rough
	// boundary. Two equidistant blackframes at 100 (before) and 110
	// (after) for roughS=105 — the forward one should win even
	// though backward distance is identical, because the penalty
	// doubles the backward distance.
	black := []signals.BlackEvent{
		{StartS: 100.0, EndS: 100.5}, // anchor for isStart = EndS = 100.5
		{StartS: 110.0, EndS: 110.5}, // anchor = 110.5
	}
	got := refineBoundaryVoting(105.0, 30.0, black, nil, nil, nil, true)
	if !near(got, 110.5, 0.1) {
		t.Errorf("forward bias for isStart: want ≈110.5, got %.3f", got)
	}
}

func TestMaxBlockFractionDropsRunaway(t *testing.T) {
	// 30 min recording where the entire show stretch is logo-absent
	// (= sixx-style washout pathology). The state machine opens one
	// giant block covering ~95 % of the recording. With the default
	// MaxBlockFraction=0.5 cap this block must be dropped, leaving an
	// empty block list rather than emitting a 28-min false positive.
	fps := 25.0
	nFrames := 25 * 1800
	// Logo only present in the first and last 30 s — everything in
	// between is "absent" (= washout).
	logo := makeLogo(nFrames, [][2]int{{0, 25 * 30}, {25 * 1770, nFrames}})
	blocks := Form(Opts{FPS: fps}, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames)
	if len(blocks) != 0 {
		t.Errorf("runaway block (>50%% of recording) must be dropped, got %+v", blocks)
	}
	// Sanity: with the cap loosened, the same input should yield
	// exactly the runaway block — proves the only thing changed is
	// the new filter, not some other behaviour.
	blocks = Form(Opts{FPS: fps, MaxBlockFraction: 1.0},
		logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames)
	if len(blocks) != 1 {
		t.Fatalf("with cap loosened, want 1 block, got %d", len(blocks))
	}
	if blocks[0].Duration() < 1500 {
		t.Errorf("loosened-cap block should span most of recording, got %.0fs",
			blocks[0].Duration())
	}
}

func TestStartEndExtendCappedToNeighbours(t *testing.T) {
	fps := 25.0
	nFrames := 25 * 1000
	// One ad block 600..700 (logo absent), show on either side.
	logo := makeLogo(nFrames, [][2]int{{0, 25 * 600}, {25 * 700, nFrames}})
	// Extend both ends by 50s — neither neighbour caps in this
	// single-block case (totalS=1000, no prior block).
	blocks := Form(Opts{
		FPS:          fps,
		StartExtendS: 50,
		EndExtendS:   50,
		MinBlockS:    50, // allow shorter blocks for the test
	}, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	if !near(blocks[0].StartS, 550, 1.0) {
		t.Errorf("start should pull back to 550, got %.1f", blocks[0].StartS)
	}
	if !near(blocks[0].EndS, 750, 1.0) {
		t.Errorf("end should push forward to 750, got %.1f", blocks[0].EndS)
	}
}

func TestStartEndExtendCappedAtBoundaries(t *testing.T) {
	fps := 25.0
	nFrames := 25 * 1800 // 30 min
	// Two ad blocks: 60..120 and 200..260. Show outside.
	logo := makeLogo(nFrames, [][2]int{
		{0, 25 * 60}, {25 * 120, 25 * 200}, {25 * 260, nFrames}})
	// Extend each by 60s — block A end (120+60=180) would normally
	// run into block B start (200), but cap at next.StartS = 200.
	// Block B start (200-60=140) would run back into block A end
	// (180+extension) — cap at last.EndS.
	blocks := Form(Opts{
		FPS:          fps,
		StartExtendS: 60,
		EndExtendS:   60,
		MinBlockS:    50,
	}, logo, nil, nil, nil, nil, nil, nil, nil, nil, nil, nFrames)
	if len(blocks) != 2 {
		t.Fatalf("want 2 blocks, got %d", len(blocks))
	}
	if blocks[0].EndS > blocks[1].StartS+0.01 {
		t.Errorf("blocks must not overlap after extension: %v", blocks)
	}
	if !near(blocks[0].EndS, blocks[1].StartS, 1.0) {
		t.Errorf("expected end/start to butt or near, got end=%.1f start=%.1f",
			blocks[0].EndS, blocks[1].StartS)
	}
}

func TestLogoCrossingRefineStart(t *testing.T) {
	// 25-fps logo conf: present (0.9) for 100 frames, then drops to
	// 0.0 starting at exactly frame 100 (= 4.0 s). Threshold 0.10.
	// Rough boundary is 4.5 s — refinement should snap exactly to
	// 4.0 s = frame 100 (first absent frame).
	fps := 25.0
	logo := make([]float64, 250)
	for i := 0; i < 100; i++ {
		logo[i] = 0.9
	}
	got := logoCrossingRefine(4.5, 2.0, logo, 0.10, fps, true)
	want := 100.0 / fps // = 4.0
	if !near(got, want, 0.05) {
		t.Errorf("isStart crossing: want %.3f, got %.3f", want, got)
	}
}

func TestLogoCrossingRefineEnd(t *testing.T) {
	// Inverse: absent for 100 frames, then present at frame 100.
	// isStart=false → snap to first present frame.
	fps := 25.0
	logo := make([]float64, 250)
	for i := 100; i < 250; i++ {
		logo[i] = 0.9
	}
	got := logoCrossingRefine(3.5, 2.0, logo, 0.10, fps, false)
	want := 100.0 / fps
	if !near(got, want, 0.05) {
		t.Errorf("isEnd crossing: want %.3f, got %.3f", want, got)
	}
}

func TestLogoCrossingRefineNoCrossingPassthrough(t *testing.T) {
	// All-present signal, no crossing in window → roughS unchanged.
	fps := 25.0
	logo := make([]float64, 250)
	for i := range logo {
		logo[i] = 0.9
	}
	got := logoCrossingRefine(5.0, 2.0, logo, 0.10, fps, true)
	if got != 5.0 {
		t.Errorf("monotone signal: want passthrough 5.0, got %.3f", got)
	}
}

func TestLogoCrossingRefinePrefersClosestCrossing(t *testing.T) {
	// Two crossings in window — prefer the one closer to rough boundary.
	// Frame 50: 0.9 → 0.0 (start crossing)
	// Frame 150: 0.0 → 0.9 (end crossing — wrong direction for isStart)
	// Frame 200: 0.9 → 0.0 (start crossing again)
	// rough = frame 180 (= 7.2s). Closest START crossing is frame 200 (= 8.0s).
	fps := 25.0
	logo := make([]float64, 300)
	for i := 0; i < 50; i++ { logo[i] = 0.9 }
	for i := 150; i < 200; i++ { logo[i] = 0.9 }
	got := logoCrossingRefine(7.2, 2.0, logo, 0.10, fps, true)
	want := 200.0 / fps // = 8.0
	if !near(got, want, 0.05) {
		t.Errorf("closest-start: want %.3f, got %.3f", want, got)
	}
}

func TestSmoothMeanWindow(t *testing.T) {
	// 9 ones surrounded by zeros — half-window 1 (= 3-element window)
	// should bleed the spike to the immediate neighbours.
	x := []float64{0, 0, 0, 1, 0, 0, 0}
	got := smoothMean(x, 1)
	// Index 2 sees x[1..3] = (0+0+1)/3 = 0.333
	// Index 3 sees x[2..4] = (0+1+0)/3 = 0.333
	// Index 4 sees x[3..5] = (1+0+0)/3 = 0.333
	for _, i := range []int{2, 3, 4} {
		if !near(got[i], 0.3333, 0.001) {
			t.Errorf("smoothMean idx %d: want 0.333, got %.3f", i, got[i])
		}
	}
	if !near(got[0], 0.0, 1e-9) || !near(got[6], 0.0, 1e-9) {
		t.Errorf("edges should stay 0 (no leak past 1 step), got %v", got)
	}
}
