// tv-detect — multi-threaded ad-block detector. See PLAN.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"encoding/json"

	"github.com/simonchrz/tv-detect/internal/blocks"
	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/signals"
	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

func main() {
	var (
		logoPath       = flag.String("logo", "", "path to comskip .logo.txt template")
		workers        = flag.Int("workers", runtime.NumCPU(), "parallel chunk workers")
		output         = flag.String("output", "summary", "output format: summary | jsonlines | csv")
		minBlockSec    = flag.Float64("min-block-sec", 60, "filter sub-N-second blocks")
		maxBlockSec    = flag.Float64("max-block-sec", 900, "split blocks longer than N seconds")
		minShowSegSec  = flag.Float64("min-show-segment", 120, "min show between blocks before merging")
		logoThreshold  = flag.Float64("logo-threshold", 0.10, "logo absent below this confidence")
		blackframeDur  = flag.Float64("blackframe-d", 0.10, "min duration for blackframe (seconds)")
		silenceNoiseDB = flag.Float64("silence-noise-db", -30, "silence noise floor (dB)")
		silenceDur     = flag.Float64("silence-d", 0.50, "min silence duration (seconds)")
		quiet          = flag.Bool("quiet", false, "suppress progress output to stderr")
		// Phase 1/2 sanity flags — will fold into pipeline orchestration later.
		probeOnly       = flag.Bool("probe", false, "only print stream metadata and exit")
		decodeWidth     = flag.Int("decode-width", 0, "scale frames to width (0 = native)")
		decodeHeight    = flag.Int("decode-height", 0, "scale frames to height (0 = native)")
		emitBlackframes = flag.Bool("emit-blackframes", false, "print detected blackframe events to stdout (one per line)")
		emitLogoCSV     = flag.Bool("emit-logo-csv", false, "print logo confidence per second to stdout (idx,t,conf)")
		emitSilences    = flag.Bool("emit-silences", false, "print detected silence events to stdout (one per line)")
		emitScenes      = flag.Bool("emit-scenes", false, "print detected scene cuts to stdout (one per line)")
		sceneThreshold  = flag.Float64("scene-threshold", 0.40, "histogram bhattacharyya distance above which a frame pair is a scene cut")
	)
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: tv-detect [flags] <input.ts | input.m3u8>")
		flag.PrintDefaults()
		os.Exit(2)
	}
	input := flag.Arg(0)

	// TODO Phase 2: signal extractors (logo, blackframe, silence, scenecut)
	// TODO Phase 3: chunk-parallel pipeline
	// TODO Phase 4: block-formation state machine
	// TODO Phase 5: output formatter
	_ = logoPath
	_ = workers
	_ = output
	_ = minBlockSec
	_ = maxBlockSec
	_ = minShowSegSec
	_ = logoThreshold
	_ = blackframeDur
	_ = silenceNoiseDB
	_ = silenceDur

	info, err := decode.Probe(input)
	if err != nil {
		fmt.Fprintln(os.Stderr, "probe:", err)
		os.Exit(1)
	}
	if !*quiet {
		fmt.Fprintf(os.Stderr,
			"probe: %dx%d  fps=%.3f  duration=%.1fs  frames=%d\n",
			info.Width, info.Height, info.FPS, info.DurationS, info.FrameCount)
	}
	if *probeOnly {
		return
	}

	ctx, cancel := signalContext()
	defer cancel()

	d, err := decode.NewDecoder(ctx, decode.DecodeOpts{
		Input:  input,
		Width:  *decodeWidth,
		Height: *decodeHeight,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(1)
	}
	defer d.Close()

	// Silence runs in its own ffmpeg subprocess that reads only the
	// audio stream — no contention with the video decoder. Kick it off
	// in a goroutine; join after the video pipeline completes.
	type silenceResult struct {
		events []signals.SilenceEvent
		err    error
	}
	silenceCh := make(chan silenceResult, 1)
	go func() {
		ev, err := signals.DetectSilence(ctx, signals.SilenceOpts{
			Input:   input,
			NoiseDB: *silenceNoiseDB,
			MinDurS: *silenceDur,
		})
		silenceCh <- silenceResult{events: ev, err: err}
	}()

	black := signals.NewBlackDetector(d.FPS, *blackframeDur, 0, 0)
	scene := signals.NewSceneDetector(d.FPS, *sceneThreshold)

	var logo *signals.LogoDetector
	var logoConfs []float64 // per-frame confidences when logo is enabled
	if *logoPath != "" {
		tmpl, err := logotemplate.Load(*logoPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "logo:", err)
			os.Exit(1)
		}
		logo, err = signals.NewLogoDetector(tmpl, d.Width, d.Height, 0)
		if err != nil {
			fmt.Fprintln(os.Stderr, "logo:", err)
			os.Exit(1)
		}
		if !*quiet {
			fmt.Fprintf(os.Stderr,
				"logo: %dx%d bbox at (%d,%d) — %d edge positions\n",
				tmpl.Width(), tmpl.Height(), tmpl.MinX, tmpl.MinY, tmpl.EdgePositions)
		}
	}

	t0 := time.Now()
	count := 0
	logoSum, logoCount := 0.0, 0
	progressEvery := 1000
	for f := range d.Frames() {
		black.Push(f.Index, f.Pixels)
		scene.Push(f.Index, f.Pixels)
		if logo != nil {
			c := logo.Confidence(f.Pixels)
			logoSum += c
			logoCount++
			logoConfs = append(logoConfs, c)
		}
		count++
		if !*quiet && count%progressEvery == 0 {
			elapsed := time.Since(t0).Seconds()
			fmt.Fprintf(os.Stderr,
				"\rdecode: %d frames  %.1f fps", count, float64(count)/elapsed)
		}
	}
	black.Finish()
	if err := d.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "\ndecode error:", err)
		os.Exit(1)
	}
	silenceRes := <-silenceCh
	if silenceRes.err != nil {
		fmt.Fprintln(os.Stderr, "silence:", silenceRes.err)
		os.Exit(1)
	}
	elapsed := time.Since(t0).Seconds()
	if !*quiet {
		fmt.Fprintf(os.Stderr,
			"\rdecode: %d frames in %.1fs (%.0f fps avg)  blackframes=%d  silences=%d  scenes=%d",
			count, elapsed, float64(count)/elapsed,
			len(black.Events()), len(silenceRes.events), len(scene.Cuts()))
		if logo != nil && logoCount > 0 {
			fmt.Fprintf(os.Stderr, "  logo_avg=%.3f", logoSum/float64(logoCount))
		}
		fmt.Fprintln(os.Stderr)
	}
	if *emitBlackframes {
		for _, e := range black.Events() {
			fmt.Printf("blackframe start=%.3f end=%.3f duration=%.3f\n",
				e.StartS, e.EndS, e.DurationS)
		}
	}
	if *emitSilences {
		for _, e := range silenceRes.events {
			fmt.Printf("silence start=%.3f end=%.3f duration=%.3f\n",
				e.StartS, e.EndS, e.DurationS)
		}
	}
	if *emitScenes {
		for _, c := range scene.Cuts() {
			fmt.Printf("scene frame=%d t=%.3f dist=%.3f\n", c.Frame, c.TimeS, c.Distance)
		}
	}
	if *emitLogoCSV {
		fmt.Println("idx,time_s,confidence")
		for i, c := range logoConfs {
			fmt.Printf("%d,%.3f,%.4f\n", i, float64(i)/d.FPS, c)
		}
	}

	// Block formation + final output. Requires logo confidences; without
	// logo we emit an empty block list (blackframe/silence alone don't
	// classify ad vs show reliably enough).
	blockList := blocks.Form(blocks.Opts{
		FPS:             d.FPS,
		MinBlockS:       *minBlockSec,
		MaxBlockS:       *maxBlockSec,
		MinShowSegmentS: *minShowSegSec,
		LogoThreshold:   *logoThreshold,
	}, logoConfs, black.Events(), silenceRes.events, count)

	switch *output {
	case "summary":
		writeSummary(d, count, elapsed, blockList)
	case "cutlist":
		writeCutlist(d.FPS, count, blockList)
	case "jsonlines", "csv":
		fmt.Fprintf(os.Stderr, "output format %q not yet implemented\n", *output)
	default:
		fmt.Fprintf(os.Stderr, "unknown output format: %q\n", *output)
		os.Exit(2)
	}
}

func writeSummary(d *decode.Decoder, frames int, elapsedS float64, bl []blocks.Block) {
	type summaryOut struct {
		FPS        float64        `json:"fps"`
		Width      int            `json:"width"`
		Height     int            `json:"height"`
		FrameCount int            `json:"frame_count"`
		DurationS  float64        `json:"duration_s"`
		Blocks     [][2]float64   `json:"blocks"`
		Stats      map[string]any `json:"stats"`
	}
	out := summaryOut{
		FPS:        d.FPS,
		Width:      d.Width,
		Height:     d.Height,
		FrameCount: frames,
		DurationS:  float64(frames) / d.FPS,
		Blocks:     make([][2]float64, len(bl)),
		Stats: map[string]any{
			"elapsed_s": elapsedS,
			"fps_proc":  float64(frames) / elapsedS,
		},
	}
	for i, b := range bl {
		out.Blocks[i] = [2]float64{b.StartS, b.EndS}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}

// writeCutlist prints comskip-compatible frame-pair output so existing
// Python parsers (hls-gateway _rec_parse_comskip, tv-live-comskip
// parse_comskip) can consume it unchanged.
func writeCutlist(fps float64, frameCount int, bl []blocks.Block) {
	fmt.Printf("FILE PROCESSING COMPLETE  %d FRAMES AT  %d\n",
		frameCount, int(fps*100+0.5))
	fmt.Println("-------------------")
	for _, b := range bl {
		sf := int(b.StartS*fps + 0.5)
		ef := int(b.EndS*fps + 0.5)
		fmt.Printf("%d\t%d\n", sf, ef)
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
