// Package signals contains per-frame analysers used by the block
// formation state machine: blackframe, logo correlation, scene cut,
// audio silence (silence is parsed from a sibling ffmpeg subprocess
// and lives in its own file).
package signals

// Defaults match ffmpeg's blackdetect filter so output is directly
// comparable. ffmpeg: pix_th=0.10, pic_th=0.98, d=2.0.
const (
	defaultPixTh = 0.10 // pixel is "black" if luma <= pix_th
	defaultPicTh = 0.98 // frame is "black" if pic_th fraction of pixels are
)

// IsBlackFrame returns true when at least picTh fraction of pixels
// (0..1) have luma <= pixTh*255. pixels is row-major rgb24 (length
// 3*w*h). Uses BT.601 integer-approx luma to avoid float math
// per-pixel — ~50µs per 720x576 frame on M-series.
func IsBlackFrame(pixels []byte, pixTh, picTh float64) bool {
	if pixTh <= 0 {
		pixTh = defaultPixTh
	}
	if picTh <= 0 {
		picTh = defaultPicTh
	}
	thresh := int(pixTh*255 + 0.5)
	nPixels := len(pixels) / 3
	black := 0
	for i := 0; i < len(pixels); i += 3 {
		// Y = 0.299*R + 0.587*G + 0.114*B  (BT.601)
		// fixed-point: (77*R + 150*G + 29*B) >> 8 — sums to 256
		y := (77*int(pixels[i]) + 150*int(pixels[i+1]) + 29*int(pixels[i+2])) >> 8
		if y <= thresh {
			black++
		}
	}
	return float64(black) >= picTh*float64(nPixels)
}

// BlackEvent is a contiguous run of black frames lasting at least
// MinDurS seconds. Times are in seconds from input start.
type BlackEvent struct {
	StartS    float64
	EndS      float64
	DurationS float64
}

// BlackDetector accumulates per-frame black/non-black classifications
// into BlackEvents. Configure with NewBlackDetector, then Push every
// frame in order, then call Events to drain.
type BlackDetector struct {
	fps     float64
	minDur  float64
	pixTh   float64
	picTh   float64
	events  []BlackEvent
	inBlack bool
	startF  int
	lastF   int
}

// NewBlackDetector configures the run-aggregator. minDurS filters out
// blink-length events (typical comskip default: 0.10s). pixTh/picTh
// override IsBlackFrame thresholds; pass 0 for ffmpeg-compatible
// defaults.
func NewBlackDetector(fps, minDurS, pixTh, picTh float64) *BlackDetector {
	return &BlackDetector{
		fps:    fps,
		minDur: minDurS,
		pixTh:  pixTh,
		picTh:  picTh,
	}
}

// Push classifies one frame and updates run state.
func (d *BlackDetector) Push(idx int, pixels []byte) {
	d.lastF = idx
	if IsBlackFrame(pixels, d.pixTh, d.picTh) {
		if !d.inBlack {
			d.startF = idx
			d.inBlack = true
		}
		return
	}
	if d.inBlack {
		d.flush(idx)
	}
}

// Finish flushes any open black run as if a non-black frame followed.
// Call once after the last frame.
func (d *BlackDetector) Finish() {
	if d.inBlack {
		d.flush(d.lastF + 1)
	}
}

func (d *BlackDetector) flush(endIdx int) {
	startS := float64(d.startF) / d.fps
	endS := float64(endIdx) / d.fps
	if endS-startS >= d.minDur {
		d.events = append(d.events, BlackEvent{
			StartS:    startS,
			EndS:      endS,
			DurationS: endS - startS,
		})
	}
	d.inBlack = false
}

// Events returns the accumulated run events. Safe to call after Finish.
func (d *BlackDetector) Events() []BlackEvent { return d.events }
