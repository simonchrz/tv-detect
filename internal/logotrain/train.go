// Package logotrain learns a logo template from N frames of channel
// content. The output format is comskip-compatible so the same file
// can be consumed by either tool (or by tv-detect's signals.LogoDetector).
//
// Algorithm: for each pixel, track how often a horizontal-edge
// response (Sobel Gy) and a vertical-edge response (Sobel Gx) fire.
// A pixel is part of the logo if either response holds for >= P
// fraction of frames (default 0.85). The bounding box is the
// inclusive min/max of all such pixels, expanded by a small margin
// so the mask captures the logo cleanly. Within the bbox, each pixel
// gets one of:
//
//	' '  no consistent edge
//	'-'  horizontal edge consistent
//	'|'  vertical edge consistent
//	'+'  both
//
// (Same encoding as comskip's SaveLogoMaskData in mpeg2dec.c.)
package logotrain

import (
	"fmt"
	"io"
	"os"
)

// Opts controls the training algorithm.
type Opts struct {
	FrameW, FrameH int
	EdgeThresh     int     // Sobel |G| above which an edge fires (0 = 80)
	Persistence    float64 // fraction of frames a pixel must be edge to count (0 = 0.85)
	BboxMargin     int     // extend bbox by this many pixels in each direction (0 = 4)
}

// Trainer accumulates per-pixel edge presence across many frames,
// then emits a logo template via WriteTemplate. Push frames one by
// one (any sampling cadence works; 1 fps is plenty since the logo
// doesn't move).
type Trainer struct {
	opts        Opts
	stride      int
	hCount      []uint32 // per-pixel horizontal-edge count (Sobel Gy)
	vCount      []uint32 // per-pixel vertical-edge count   (Sobel Gx)
	frameCount  uint32
}

// New creates a Trainer for FrameW x FrameH frames. Allocates two
// per-pixel uint32 buffers (~3 MB for 720x576 — negligible).
func New(opts Opts) *Trainer {
	defaults(&opts)
	n := opts.FrameW * opts.FrameH
	return &Trainer{
		opts:   opts,
		stride: 3 * opts.FrameW,
		hCount: make([]uint32, n),
		vCount: make([]uint32, n),
	}
}

// Push folds one frame's edge map into the running counts.
// pixels is row-major rgb24 of length 3*FrameW*FrameH.
func (t *Trainer) Push(pixels []byte) {
	w, h := t.opts.FrameW, t.opts.FrameH
	thresh := t.opts.EdgeThresh
	stride := t.stride
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			c := y*stride + 3*x
			tl := lumaAt(pixels, c-stride-3)
			ttop := lumaAt(pixels, c-stride)
			tr := lumaAt(pixels, c-stride+3)
			l := lumaAt(pixels, c-3)
			r := lumaAt(pixels, c+3)
			bl := lumaAt(pixels, c+stride-3)
			b := lumaAt(pixels, c+stride)
			br := lumaAt(pixels, c+stride+3)
			gx := -tl + tr - 2*l + 2*r - bl + br
			gy := -tl - 2*ttop - tr + bl + 2*b + br
			if gx < 0 {
				gx = -gx
			}
			if gy < 0 {
				gy = -gy
			}
			idx := y*w + x
			if gx > thresh {
				t.vCount[idx]++ // strong horizontal-direction gradient = vertical edge
			}
			if gy > thresh {
				t.hCount[idx]++
			}
		}
	}
	t.frameCount++
}

// Result holds what training produced.
type Result struct {
	FrameCount  uint32
	MinX, MaxX  int
	MinY, MaxY  int
	EdgePixels  int
	HasLogo     bool // false if no pixel met the persistence threshold
}

// RawCounts exposes the accumulated edge histograms (length FrameW*FrameH).
// Indexed as y*FrameW + x. Intended for debug/diagnosis only.
type RawCounts struct {
	H []uint32
	V []uint32
}

func (t *Trainer) RawCounts() RawCounts { return RawCounts{H: t.hCount, V: t.vCount} }

// Compute scans the accumulated counts, finds the logo bounding box,
// and returns a Result (without writing anything to disk).
func (t *Trainer) Compute() Result {
	if t.frameCount == 0 {
		return Result{}
	}
	w, h := t.opts.FrameW, t.opts.FrameH
	threshold := uint32(float64(t.frameCount) * t.opts.Persistence)
	minX, minY := w, h
	maxX, maxY := -1, -1
	edgePixels := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if t.hCount[idx] >= threshold || t.vCount[idx] >= threshold {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
				edgePixels++
			}
		}
	}
	if maxX < 0 {
		return Result{FrameCount: t.frameCount}
	}
	// Expand bbox a touch so the eventual mask captures soft edges
	// at the logo perimeter; clip to frame.
	margin := t.opts.BboxMargin
	minX = max0(minX - margin)
	minY = max0(minY - margin)
	maxX = clip(maxX+margin, w-1)
	maxY = clip(maxY+margin, h-1)
	return Result{
		FrameCount: t.frameCount,
		MinX:       minX, MaxX: maxX,
		MinY: minY, MaxY: maxY,
		EdgePixels: edgePixels,
		HasLogo:    true,
	}
}

// WriteTemplate emits the trained mask in comskip's .logo.txt format.
// File layout:
//
//	logoMinX/MaxX/MinY/MaxY  (bbox in source pixel coords, inclusive)
//	picWidth/picHeight       (training frame size)
//	"Combined Logo Mask"     (literal marker)
//	0x82 \n                  (single binary byte the format demands)
//	one row per Y in [MinY..MaxY], MaxX-MinX+1 chars per row
//
// Returns an error if no logo was found.
func (t *Trainer) WriteTemplate(w io.Writer) error {
	r := t.Compute()
	if !r.HasLogo {
		return fmt.Errorf("no pixels met %.0f%% persistence over %d frames",
			t.opts.Persistence*100, t.frameCount)
	}
	threshold := uint32(float64(t.frameCount) * t.opts.Persistence)
	fmt.Fprintf(w, "logoMinX=%d\n", r.MinX)
	fmt.Fprintf(w, "logoMaxX=%d\n", r.MaxX)
	fmt.Fprintf(w, "logoMinY=%d\n", r.MinY)
	fmt.Fprintf(w, "logoMaxY=%d\n", r.MaxY)
	fmt.Fprintf(w, "picWidth=%d\n", t.opts.FrameW)
	fmt.Fprintf(w, "picHeight=%d\n", t.opts.FrameH)
	fmt.Fprint(w, "\nCombined Logo Mask\n")
	if _, err := w.Write([]byte{0x82, '\n'}); err != nil {
		return err
	}
	fw := t.opts.FrameW
	for y := r.MinY; y <= r.MaxY; y++ {
		for x := r.MinX; x <= r.MaxX; x++ {
			idx := y*fw + x
			h := t.hCount[idx] >= threshold
			v := t.vCount[idx] >= threshold
			switch {
			case h && v:
				fmt.Fprint(w, "+")
			case v:
				fmt.Fprint(w, "|")
			case h:
				fmt.Fprint(w, "-")
			default:
				fmt.Fprint(w, " ")
			}
		}
		fmt.Fprintln(w)
	}
	return nil
}

// SaveTemplate is a convenience wrapper that opens a file and writes
// the template to it.
func (t *Trainer) SaveTemplate(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return t.WriteTemplate(f)
}

func defaults(o *Opts) {
	if o.EdgeThresh <= 0 {
		o.EdgeThresh = 80
	}
	if o.Persistence <= 0 {
		o.Persistence = 0.85
	}
	if o.BboxMargin <= 0 {
		o.BboxMargin = 4
	}
}

func lumaAt(p []byte, i int) int {
	return (77*int(p[i]) + 150*int(p[i+1]) + 29*int(p[i+2])) >> 8
}

func max0(x int) int {
	if x < 0 {
		return 0
	}
	return x
}

func clip(x, hi int) int {
	if x > hi {
		return hi
	}
	return x
}
