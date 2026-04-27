// Channel station-id bumpers (e.g. RTL's "Mein RTL" card at the end
// of every ad break) are pre-rendered animations that recur with
// pixel-near-identical layout — a deterministic ad-end marker far
// stronger than any learned signal. Comskip's "cutscene fingerprint"
// concept; we run multiple per-channel templates and take the max
// match. Colors typically blink across appearances; luma binarisation
// makes the matcher color-invariant while keeping layout sensitive.
package signals

import (
	"fmt"
	"image"
	_ "image/png"
	"os"
)

// BumperTemplate is one reference frame, pre-converted to a luma
// binary mask matching the runtime frame size. Multiple templates per
// channel cover color-variant animations.
type BumperTemplate struct {
	Name      string
	Mask      []bool // length = w*h
	WhiteSet  int    // popcount(mask) — used for IoU denominator
}

// BumperDetector matches one or more templates against per-frame
// rgb24 pixels. Confidence returns the max IoU across all loaded
// templates. Per-frame cost ≈ frame_pixels × (luma + bool ops),
// ~0.5 ms per template at 720x576 on Apple Silicon.
type BumperDetector struct {
	templates []*BumperTemplate
	frameW    int
	frameH    int
	lumaTh    int
}

// NewBumperDetector loads the given PNG paths, resizes-by-decoding
// to (frameW, frameH), and converts each to a luma binary mask.
// Returns nil if no templates were given (caller should skip).
func NewBumperDetector(paths []string, frameW, frameH, lumaThresh int) (*BumperDetector, error) {
	if len(paths) == 0 {
		return nil, nil
	}
	if lumaThresh <= 0 {
		lumaThresh = 80
	}
	d := &BumperDetector{frameW: frameW, frameH: frameH, lumaTh: lumaThresh}
	for _, p := range paths {
		t, err := loadBumperTemplate(p, frameW, frameH, lumaThresh)
		if err != nil {
			return nil, fmt.Errorf("bumper template %s: %w", p, err)
		}
		d.templates = append(d.templates, t)
	}
	return d, nil
}

// Confidence returns the max IoU (intersection over union) across
// all templates — best match wins. Range 0..1.
// pixels is row-major rgb24, length 3*frameW*frameH.
func (d *BumperDetector) Confidence(pixels []byte) float64 {
	if d == nil || len(d.templates) == 0 {
		return 0
	}
	frameMask := make([]bool, d.frameW*d.frameH)
	frameWhite := 0
	for i := 0; i < d.frameW*d.frameH; i++ {
		luma := lumaAt(pixels, i*3)
		if luma > d.lumaTh {
			frameMask[i] = true
			frameWhite++
		}
	}
	best := 0.0
	for _, t := range d.templates {
		inter := 0
		for i, b := range t.Mask {
			if b && frameMask[i] {
				inter++
			}
		}
		union := t.WhiteSet + frameWhite - inter
		if union == 0 {
			continue
		}
		iou := float64(inter) / float64(union)
		if iou > best {
			best = iou
		}
	}
	return best
}

// loadBumperTemplate reads a PNG, resamples it to (frameW, frameH)
// via nearest-neighbor (good enough for binary masks), converts to
// the luma binary representation, returns the cached template.
func loadBumperTemplate(path string, frameW, frameH, lumaThresh int) (*BumperTemplate, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	srcB := img.Bounds()
	srcW, srcH := srcB.Dx(), srcB.Dy()
	mask := make([]bool, frameW*frameH)
	white := 0
	for fy := 0; fy < frameH; fy++ {
		sy := fy * srcH / frameH
		for fx := 0; fx < frameW; fx++ {
			sx := fx * srcW / frameW
			r, g, b, _ := img.At(srcB.Min.X+sx, srcB.Min.Y+sy).RGBA()
			// RGBA() returns 16-bit; downscale to 8-bit then BT.601 luma
			r8, g8, b8 := int(r>>8), int(g>>8), int(b>>8)
			luma := (77*r8 + 150*g8 + 29*b8) >> 8
			if luma > lumaThresh {
				mask[fy*frameW+fx] = true
				white++
			}
		}
	}
	return &BumperTemplate{
		Name:     path,
		Mask:     mask,
		WhiteSet: white,
	}, nil
}
