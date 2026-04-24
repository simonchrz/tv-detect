package signals

import (
	"fmt"

	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

// Default edge-detection threshold (sum of |dx|+|dy| on luma in 0..255*2).
// Tuned empirically against comskip output; can be overridden.
const defaultEdgeThresh = 30

// LogoDetector matches a comskip-trained logo template against per-frame
// pixels. Returns 0..1 confidence per frame: fraction of template's
// "edge expected" positions that actually have an edge in the frame.
type LogoDetector struct {
	tmpl   *logotemplate.Template
	frameW int
	frameH int
	edgeTh int
	stride int // bytes per row in source frame = 3*frameW
	// Per-detection scratch — reused across calls to avoid allocation.
	edgeBuf []bool
}

// NewLogoDetector validates that the template's training resolution
// matches the decode resolution and prepares scratch buffers.
// edgeThresh of 0 selects the default (30).
func NewLogoDetector(tmpl *logotemplate.Template, frameW, frameH, edgeThresh int) (*LogoDetector, error) {
	if tmpl.PicWidth != frameW || tmpl.PicHeight != frameH {
		return nil, fmt.Errorf(
			"logo template trained at %dx%d but frames are %dx%d — re-train or scale",
			tmpl.PicWidth, tmpl.PicHeight, frameW, frameH)
	}
	if edgeThresh <= 0 {
		edgeThresh = defaultEdgeThresh
	}
	return &LogoDetector{
		tmpl:    tmpl,
		frameW:  frameW,
		frameH:  frameH,
		edgeTh:  edgeThresh,
		stride:  3 * frameW,
		edgeBuf: make([]bool, tmpl.Width()*tmpl.Height()),
	}, nil
}

// Confidence returns the fraction of expected-edge positions in the
// template that have an actual edge in this frame's ROI. 0..1.
//
// Pixels is row-major rgb24, length 3*frameW*frameH. The ROI is
// extracted at the template's bounding box. Edge presence at each
// pixel = (|dx_luma| + |dy_luma|) > edgeThresh, where dx and dy are
// 1-pixel forward differences on BT.601 luma.
func (d *LogoDetector) Confidence(pixels []byte) float64 {
	w, h := d.tmpl.Width(), d.tmpl.Height()
	matches := 0
	// Walk ROI pixels. We need the luma at (x, y), (x+1, y), (x, y+1)
	// to compute the gradient; clip to bbox so we don't read past the
	// last column/row.
	x0, y0 := d.tmpl.MinX, d.tmpl.MinY
	for ry := 0; ry < h; ry++ {
		fy := y0 + ry
		if fy+1 >= d.frameH {
			break
		}
		row := d.tmpl.Mask[ry]
		for rx := 0; rx < w; rx++ {
			if !row[rx] {
				continue // template doesn't expect an edge here
			}
			fx := x0 + rx
			if fx+1 >= d.frameW {
				continue
			}
			i := fy*d.stride + 3*fx
			yC := luma(pixels[i], pixels[i+1], pixels[i+2])
			yX := luma(pixels[i+3], pixels[i+4], pixels[i+5])
			yY := luma(pixels[i+d.stride], pixels[i+d.stride+1], pixels[i+d.stride+2])
			dx := yX - yC
			if dx < 0 {
				dx = -dx
			}
			dy := yY - yC
			if dy < 0 {
				dy = -dy
			}
			if dx+dy > d.edgeTh {
				matches++
			}
		}
	}
	return float64(matches) / float64(d.tmpl.EdgePositions)
}

// luma returns BT.601 8-bit luma from R,G,B (integer-approx, sums to 256).
func luma(r, g, b byte) int {
	return (77*int(r) + 150*int(g) + 29*int(b)) >> 8
}
