package signals

import (
	"fmt"
	"os"

	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

// Default Sobel edge-magnitude threshold (|Gx| + |Gy| on luma in
// 0..2040). Comskip's effective threshold is in the same ballpark
// once you account for kernel scaling. Tuned empirically against
// known logo-present recordings; configurable via NewLogoDetector.
const defaultEdgeThresh = 80

// LogoDetector matches a comskip-trained logo template against per-frame
// pixels. Returns 0..1 confidence per frame: fraction of template's
// "edge expected" positions that have an actual Sobel edge in the
// frame at the same position.
//
// The template's bounding box (logoMinX..MaxX, logoMinY..MaxY) is used
// directly as the source ROI. Comskip pads its working frame to a
// multiple of 16 columns (so PAL 720 → 736 in templates), but the
// padding goes on the RIGHT, so the bounding-box X coordinates are
// equally valid in the native 720-wide frame as long as logoMaxX
// stays within frameW. We range-check rather than fail outright so a
// trimmed/cropped frame just under-counts gracefully.
type LogoDetector struct {
	tmpl    *logotemplate.Template
	frameW  int
	frameH  int
	edgeTh  int
	stride  int
	yOffset int // shift template y-coords (= letterbox top-bar height)
}

// NewLogoDetector creates a detector for the given template + frame
// dimensions. edgeThresh of 0 selects the default. yOffset shifts
// the template's Y coordinates downward (positive) — used when the
// channel airs a 16:9 program in a 4:3 broadcast container with
// letterbox bars: the template was trained against the full frame,
// but the actual logo appears further down in the visible area.
// Returns an error only if the bounding box can't fit in the frame.
func NewLogoDetector(tmpl *logotemplate.Template, frameW, frameH, edgeThresh, yOffset int) (*LogoDetector, error) {
	// Hard requirement: bbox X must fit in frame width — there's no
	// equivalent x-offset compensation, so an X-overflow really is a
	// configuration error worth failing on.
	if tmpl.MaxX > frameW {
		return nil, fmt.Errorf(
			"logo bbox (%d,%d)-(%d,%d) MaxX > frameW %d",
			tmpl.MinX, tmpl.MinY, tmpl.MaxX, tmpl.MaxY, frameW)
	}
	// If the requested Y offset would push the bbox past the bottom
	// of the frame, ignore it and log a warning. Common case: logo
	// templates trained with the logo at the bottom of the frame
	// (rtlzwei: y=494-552 in a 720x576 frame; vox; sixx) can't
	// physically shift down by the cropdetect-derived offset — the
	// offset is a letterbox-bar height, sensible only for top-of-
	// frame logos. Recording proceeds with the trained bbox; we lose
	// the per-recording letterbox compensation but don't crash the
	// pipeline. Pre-2026-05-04 this returned an error, which combined
	// with the daemon's no-retry-clear policy stalled the entire
	// re-detect queue on the first rtlzwei recording in a batch.
	if tmpl.MaxY+yOffset > frameH {
		fmt.Fprintf(os.Stderr,
			"logo: y-offset %d would push bbox bottom (%d) past frameH (%d) — clamping to 0 (logo template likely bottom-positioned, letterbox compensation N/A)\n",
			yOffset, tmpl.MaxY+yOffset, frameH)
		yOffset = 0
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
		yOffset: yOffset,
	}, nil
}

// Confidence returns the fraction of expected-edge positions in the
// template that have a Sobel edge in this frame's ROI (range 0..1).
// pixels is row-major rgb24, length 3*frameW*frameH.
func (d *LogoDetector) Confidence(pixels []byte) float64 {
	w, h := d.tmpl.Width(), d.tmpl.Height()
	x0, y0 := d.tmpl.MinX, d.tmpl.MinY
	matches := 0
	for ry := 0; ry < h; ry++ {
		fy := y0 + ry + d.yOffset
		if fy <= 0 || fy >= d.frameH-1 {
			continue // need fy-1 and fy+1 for the Sobel kernel
		}
		row := d.tmpl.Mask[ry]
		for rx := 0; rx < w; rx++ {
			if !row[rx] {
				continue
			}
			fx := x0 + rx
			if fx <= 0 || fx >= d.frameW-1 {
				continue
			}
			if sobelMag(pixels, d.stride, fx, fy) > d.edgeTh {
				matches++
			}
		}
	}
	return float64(matches) / float64(d.tmpl.EdgePositions)
}

// sobelMag returns |Gx| + |Gy| of BT.601 luma for the 3x3 neighbourhood
// centred on (x,y). Caller must ensure x and y are at least 1 away from
// the frame edges.
//
//	Gx = [-1 0 +1; -2 0 +2; -1 0 +1]   (detects vertical edges)
//	Gy = [-1 -2 -1;  0 0 0; +1 +2 +1]  (detects horizontal edges)
func sobelMag(pixels []byte, stride, x, y int) int {
	c := y*stride + 3*x
	tl := lumaAt(pixels, c-stride-3)
	t := lumaAt(pixels, c-stride)
	tr := lumaAt(pixels, c-stride+3)
	l := lumaAt(pixels, c-3)
	r := lumaAt(pixels, c+3)
	bl := lumaAt(pixels, c+stride-3)
	b := lumaAt(pixels, c+stride)
	br := lumaAt(pixels, c+stride+3)

	gx := -tl + tr - 2*l + 2*r - bl + br
	gy := -tl - 2*t - tr + bl + 2*b + br
	if gx < 0 {
		gx = -gx
	}
	if gy < 0 {
		gy = -gy
	}
	return gx + gy
}

func lumaAt(p []byte, i int) int {
	return (77*int(p[i]) + 150*int(p[i+1]) + 29*int(p[i+2])) >> 8
}
