// tv-detect-train-logo — trains a comskip-compatible .logo.txt
// template from a recording. See ../../internal/logotrain for the
// algorithm.
//
// Usage:
//
//	tv-detect-train-logo [flags] <input.ts | input.m3u8>
//
// Outputs to --output (default <input-basename>.logo.txt) a template
// matching comskip's SaveLogoMaskData format. Both this binary and
// the regular tv-detect detector consume that same format, so a
// trained template plugs straight into the detection pipeline.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/logotrain"
)

func main() {
	var (
		output       = flag.String("output", "", "destination .logo.txt path (default: <input-basename>.logo.txt)")
		minutes      = flag.Float64("minutes", 5, "minutes of input to train on. 5 min stays inside the typical first-show segment; longer windows often span an ad break and prevent any pixel from meeting --persistence")
		samplesPerS  = flag.Float64("samples-per-sec", 1, "frames per second to sample (1 = 1 frame/sec)")
		edgeThresh   = flag.Int("edge-threshold", 80, "Sobel |Gx|+|Gy| above which a pixel-frame counts as edge")
		persistence  = flag.Float64("persistence", 0.85, "fraction of sampled frames a pixel must show an edge to be logo")
		bboxMargin   = flag.Int("bbox-margin", 4, "extend bbox by N pixels in each direction")
		maxBboxArea  = flag.Int("max-bbox-area", 5000, "if trained mask bbox area exceeds this many px² at the requested persistence, sweep persistence upward by 0.02 until the mask fits or persistence > 0.99 (0 = off)")
		startS       = flag.Float64("start", 30, "skip this many seconds of leading content before sampling (typical pre-show fade-in)")
		decodeWidth  = flag.Int("decode-width", 0, "scale frames to width before training (0 = native)")
		decodeHeight = flag.Int("decode-height", 0, "scale frames to height (0 = native)")
		quiet        = flag.Bool("quiet", false, "suppress progress to stderr")
		debugTopN    = flag.Int("debug-top", 0, "print top-N pixels by edge-count to diagnose persistence (0 = off)")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: tv-detect-train-logo [flags] <input.ts | input.m3u8>")
		flag.PrintDefaults()
		os.Exit(2)
	}
	input := flag.Arg(0)

	if *output == "" {
		base := filepath.Base(input)
		ext := filepath.Ext(base)
		*output = strings.TrimSuffix(base, ext) + ".logo.txt"
	}

	ctx, cancel := signalContext()
	defer cancel()

	d, err := decode.NewDecoder(ctx, decode.DecodeOpts{
		Input:  input,
		Width:  *decodeWidth,
		Height: *decodeHeight,
		StartS: *startS,
		DurS:   *minutes * 60,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(1)
	}
	defer d.Close()

	if !*quiet {
		fmt.Fprintf(os.Stderr,
			"training: %dx%d  fps=%.3f  start=%.0fs  duration=%.0fs  sample=%.1ffps\n",
			d.Width, d.Height, d.FPS, *startS, *minutes*60, *samplesPerS)
	}

	tr := logotrain.New(logotrain.Opts{
		FrameW:      d.Width,
		FrameH:      d.Height,
		EdgeThresh:  *edgeThresh,
		Persistence: *persistence,
		BboxMargin:  *bboxMargin,
	})

	// Sample one frame every (FPS / samplesPerS) frames.
	stride := int(d.FPS/(*samplesPerS) + 0.5)
	if stride < 1 {
		stride = 1
	}
	t0 := time.Now()
	read, sampled := 0, 0
	for f := range d.Frames() {
		read++
		if read%stride != 0 {
			continue
		}
		tr.Push(f.Pixels)
		sampled++
		if !*quiet && sampled%30 == 0 {
			fmt.Fprintf(os.Stderr, "\rtrain: %d sampled (%d read, %.0fs elapsed)",
				sampled, read, time.Since(t0).Seconds())
		}
	}
	if err := d.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "\ndecode error:", err)
		os.Exit(1)
	}
	if !*quiet {
		fmt.Fprintf(os.Stderr, "\ntrain done: %d frames sampled in %.1fs\n",
			sampled, time.Since(t0).Seconds())
	}

	if *debugTopN > 0 {
		printTopPixels(tr, *debugTopN, sampled, d.Width)
		return
	}
	res, finalPersist := tr.ComputeAdaptive(*maxBboxArea, *persistence)
	if !res.HasLogo {
		fmt.Fprintf(os.Stderr,
			"no logo found — no pixel met %.0f%% persistence threshold over %d frames. "+
				"Try lowering --persistence or --edge-threshold.\n",
			*persistence*100, sampled)
		os.Exit(1)
	}
	if !*quiet {
		note := ""
		if finalPersist > *persistence+1e-6 {
			note = fmt.Sprintf(" (persistence bumped %.2f → %.2f to fit --max-bbox-area %d)",
				*persistence, finalPersist, *maxBboxArea)
		}
		fmt.Fprintf(os.Stderr,
			"logo bbox: (%d,%d)-(%d,%d)  %dx%d  edge pixels=%d%s\n",
			res.MinX, res.MinY, res.MaxX, res.MaxY,
			res.MaxX-res.MinX+1, res.MaxY-res.MinY+1, res.EdgePixels, note)
	}

	if err := tr.SaveTemplateAt(*output, finalPersist); err != nil {
		fmt.Fprintln(os.Stderr, "write template:", err)
		os.Exit(1)
	}
	if !*quiet {
		fmt.Fprintf(os.Stderr, "wrote %s\n", *output)
	}
}

func printTopPixels(tr *logotrain.Trainer, n, sampled, w int) {
	type pix struct {
		x, y int
		h, v uint32
	}
	all := tr.RawCounts()
	hCount, vCount := all.H, all.V
	pixels := make([]pix, 0, len(hCount)/100)
	for i := range hCount {
		score := hCount[i]
		if vCount[i] > score {
			score = vCount[i]
		}
		if score == 0 {
			continue
		}
		pixels = append(pixels, pix{i % w, i / w, hCount[i], vCount[i]})
	}
	// Simple top-N via partial sort.
	for sel := 0; sel < n && sel < len(pixels); sel++ {
		bestI := sel
		for i := sel + 1; i < len(pixels); i++ {
			si := pixels[i].h
			if pixels[i].v > si {
				si = pixels[i].v
			}
			ss := pixels[bestI].h
			if pixels[bestI].v > ss {
				ss = pixels[bestI].v
			}
			if si > ss {
				bestI = i
			}
		}
		pixels[sel], pixels[bestI] = pixels[bestI], pixels[sel]
		p := pixels[sel]
		max := p.h
		if p.v > max {
			max = p.v
		}
		fmt.Fprintf(os.Stderr, "  (%d,%d) h=%d v=%d  best=%d/%d (%.0f%%)\n",
			p.x, p.y, p.h, p.v, max, sampled, 100*float64(max)/float64(sampled))
	}
}

func signalContext() (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
		<-ch
		cancel()
	}()
	return ctx, cancel
}
