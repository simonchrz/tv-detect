// Letterbox transitions are a deterministic show↔ad signal on
// broadcasters that air 16:9 promos inside a 4:3 container (RTL,
// RTLZWEI, partly VOX): the moment thick black bars appear at top +
// bottom is a programme→promo cut, and the moment they disappear is
// promo→programme. Unlike scene-cut detection (which can miss
// luma-similar transitions) letterbox state is geometric — pure
// presence/absence of black bars at frame edges, no thresholding of
// content similarity.
package signals

// LetterboxEvent is one letterbox state transition.
//
//	Onset  = first frame where letterbox bars appeared (black-bars
//	         visible) — a programme→promo cut on 4:3 broadcasters.
//	Offset = first frame where bars went away (full-frame content
//	         resumed) — a promo→programme cut.
//
// TimeS is the timestamp of that transition frame.
type LetterboxEvent struct {
	Frame  int
	TimeS  float64
	Onset  bool // true = bars appeared, false = bars disappeared
}

// LetterboxDetector tracks the per-frame "are top+bottom rows mostly
// black" boolean, with hysteresis to suppress single-frame flickers
// caused by encoder noise on dark scenes.
type LetterboxDetector struct {
	fps       float64
	width     int
	height    int
	barRows   int     // rows at top/bottom to inspect
	lumaTh    int     // row pixel counts as black if luma <= this
	rowFill   float64 // fraction of row pixels that must be black to count the row as a black bar (default 0.95)
	hysteresis int    // frames of consistent state before flipping (default = ~0.5s @ fps)

	pendingState  bool // candidate new state
	pendingFrames int  // consecutive frames in candidate state
	curState      bool // confirmed letterbox state
	hasFirst      bool // first frame seen — don't emit on initial confirm

	events []LetterboxEvent
}

// NewLetterboxDetector creates a detector for a frame stream of the
// given dimensions. barRows = how many rows at top + bottom to test.
// lumaThreshold = pixel counts as black when luma <= this (default 16).
func NewLetterboxDetector(fps float64, width, height, barRows, lumaThreshold int) *LetterboxDetector {
	if barRows <= 0 {
		// 4:3 (576) holding 16:9 → bar height ≈ (576-405)/2 = 85.
		// 32 is a safe lower bound that even small letterbox crops trip.
		barRows = 32
	}
	if lumaThreshold <= 0 {
		lumaThreshold = 16
	}
	hyst := int(fps * 0.5)
	if hyst < 1 {
		hyst = 1
	}
	return &LetterboxDetector{
		fps:        fps,
		width:      width,
		height:     height,
		barRows:    barRows,
		lumaTh:     lumaThreshold,
		rowFill:    0.95,
		hysteresis: hyst,
	}
}

// Push inspects the top + bottom rows of an rgb24 frame and updates
// the letterbox state machine.
func (d *LetterboxDetector) Push(idx int, pixels []byte) {
	if d.width <= 0 || d.height <= 0 || d.barRows*2 >= d.height {
		return
	}
	state := d.framedAsLetterbox(pixels)
	if !d.hasFirst {
		d.curState = state
		d.pendingState = state
		d.pendingFrames = d.hysteresis
		d.hasFirst = true
		return
	}
	if state == d.curState {
		d.pendingState = state
		d.pendingFrames = 0
		return
	}
	// state differs from confirmed — count toward a flip.
	if state != d.pendingState {
		d.pendingState = state
		d.pendingFrames = 1
		return
	}
	d.pendingFrames++
	if d.pendingFrames >= d.hysteresis {
		// Flip confirmed. Emission timestamp is the first frame of the
		// candidate run (idx - hysteresis + 1) — that is the actual
		// transition frame, not the hysteresis-confirmation frame.
		flipFrame := idx - d.hysteresis + 1
		if flipFrame < 0 {
			flipFrame = 0
		}
		d.events = append(d.events, LetterboxEvent{
			Frame: flipFrame,
			TimeS: float64(flipFrame) / d.fps,
			Onset: state, // state==true means we just turned letterbox ON
		})
		d.curState = state
		d.pendingFrames = 0
	}
}

// Events returns all confirmed letterbox transitions in frame order.
func (d *LetterboxDetector) Events() []LetterboxEvent { return d.events }

// framedAsLetterbox returns true when both the top barRows rows AND
// the bottom barRows rows are >= rowFill black. Both must be black
// (single-side darkness = dark scene, not letterbox).
func (d *LetterboxDetector) framedAsLetterbox(pixels []byte) bool {
	rowBytes := d.width * 3
	if len(pixels) < rowBytes*d.height {
		return false
	}
	if !d.rowsAreBlack(pixels, 0, d.barRows, rowBytes) {
		return false
	}
	if !d.rowsAreBlack(pixels, d.height-d.barRows, d.height, rowBytes) {
		return false
	}
	return true
}

func (d *LetterboxDetector) rowsAreBlack(pixels []byte, yLo, yHi, rowBytes int) bool {
	totalPx := (yHi - yLo) * d.width
	threshold := int(float64(totalPx) * d.rowFill)
	blackCount := 0
	nonBlackCount := 0
	maxNonBlack := totalPx - threshold
	for y := yLo; y < yHi; y++ {
		base := y * rowBytes
		for x := 0; x < d.width; x++ {
			p := base + x*3
			y8 := (77*int(pixels[p]) + 150*int(pixels[p+1]) + 29*int(pixels[p+2])) >> 8
			if y8 <= d.lumaTh {
				blackCount++
			} else {
				nonBlackCount++
				if nonBlackCount > maxNonBlack {
					return false
				}
			}
		}
	}
	return blackCount >= threshold
}
