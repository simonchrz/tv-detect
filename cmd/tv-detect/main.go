// tv-detect — multi-threaded ad-block detector. See PLAN.md.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/simonchrz/tv-detect/internal/blocks"
	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/pipeline"
	"github.com/simonchrz/tv-detect/internal/signals"
	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

func main() {
	var (
		logoPath       = flag.String("logo", "", "path to comskip .logo.txt template")
		workers        = flag.Int("workers", runtime.NumCPU(), "parallel chunk workers (1 = single pipeline)")
		output         = flag.String("output", "summary", "output format: summary | cutlist")
		minBlockSec    = flag.Float64("min-block-sec", 60, "filter sub-N-second blocks")
		maxBlockSec    = flag.Float64("max-block-sec", 900, "split blocks longer than N seconds")
		minShowSegSec  = flag.Float64("min-show-segment", 120, "min show between blocks before merging")
		logoThreshold  = flag.Float64("logo-threshold", 0.10, "logo absent below this confidence")
		blackframeDur  = flag.Float64("blackframe-d", 0.10, "min duration for blackframe (seconds)")
		silenceNoiseDB = flag.Float64("silence-noise-db", -30, "silence noise floor (dB)")
		silenceDur     = flag.Float64("silence-d", 0.50, "min silence duration (seconds)")
		sceneThreshold = flag.Float64("scene-threshold", 0.40, "histogram distance above which a frame pair is a scene cut")
		quiet          = flag.Bool("quiet", false, "suppress progress output to stderr")
		probeOnly      = flag.Bool("probe", false, "only print stream metadata and exit")
		decodeWidth    = flag.Int("decode-width", 0, "scale frames to width (0 = native)")
		decodeHeight   = flag.Int("decode-height", 0, "scale frames to height (0 = native)")
		emitBlackframes = flag.Bool("emit-blackframes", false, "print detected blackframe events to stdout")
		emitSilences    = flag.Bool("emit-silences", false, "print detected silence events to stdout")
		emitScenes      = flag.Bool("emit-scenes", false, "print detected scene cuts to stdout")
		emitLogoCSV     = flag.Bool("emit-logo-csv", false, "print per-frame logo confidence as CSV")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: tv-detect [flags] <input.ts | input.m3u8>")
		flag.PrintDefaults()
		os.Exit(2)
	}
	input := flag.Arg(0)

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

	// Optional logo template. Shared read-only across all chunk workers.
	var tmpl *logotemplate.Template
	if *logoPath != "" {
		tmpl, err = logotemplate.Load(*logoPath)
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

	// Silence runs as a single parallel ffmpeg subprocess on the audio
	// stream — no benefit from chunking it further (low CPU).
	type silenceResult struct {
		events []signals.SilenceEvent
		err    error
	}
	silenceCh := make(chan silenceResult, 1)
	go func() {
		ev, err := signals.DetectSilence(ctx, signals.SilenceOpts{
			Input: input, NoiseDB: *silenceNoiseDB, MinDurS: *silenceDur,
		})
		silenceCh <- silenceResult{events: ev, err: err}
	}()

	// Video decode + per-frame signal extraction, split across N chunks.
	t0 := time.Now()
	res, err := pipeline.Run(ctx, pipeline.Opts{
		Input:          input,
		Workers:        *workers,
		DecodeWidth:    *decodeWidth,
		DecodeHeight:   *decodeHeight,
		BlackframeDurS: *blackframeDur,
		SceneThreshold: *sceneThreshold,
		LogoTemplate:   tmpl,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "pipeline:", err)
		os.Exit(1)
	}
	sil := <-silenceCh
	if sil.err != nil {
		fmt.Fprintln(os.Stderr, "silence:", sil.err)
		os.Exit(1)
	}
	elapsed := time.Since(t0).Seconds()
	if !*quiet {
		fmt.Fprintf(os.Stderr,
			"pipeline: %d frames in %.1fs (%.0f fps) workers=%d  blackframes=%d  silences=%d  scenes=%d",
			res.FrameCount, elapsed, float64(res.FrameCount)/elapsed, *workers,
			len(res.Blackframes), len(sil.events), len(res.SceneCuts))
		if res.LogoConfs != nil {
			sum := 0.0
			for _, c := range res.LogoConfs {
				sum += c
			}
			fmt.Fprintf(os.Stderr, "  logo_avg=%.3f", sum/float64(len(res.LogoConfs)))
		}
		fmt.Fprintln(os.Stderr)
	}

	// Debug emitters — write to stdout independent of --output format.
	if *emitBlackframes {
		for _, e := range res.Blackframes {
			fmt.Printf("blackframe start=%.3f end=%.3f duration=%.3f\n",
				e.StartS, e.EndS, e.DurationS)
		}
	}
	if *emitSilences {
		for _, e := range sil.events {
			fmt.Printf("silence start=%.3f end=%.3f duration=%.3f\n",
				e.StartS, e.EndS, e.DurationS)
		}
	}
	if *emitScenes {
		for _, c := range res.SceneCuts {
			fmt.Printf("scene frame=%d t=%.3f dist=%.3f\n", c.Frame, c.TimeS, c.Distance)
		}
	}
	if *emitLogoCSV {
		fmt.Println("idx,time_s,confidence")
		for i, c := range res.LogoConfs {
			fmt.Printf("%d,%.3f,%.4f\n", i, float64(i)/res.FPS, c)
		}
	}

	// Block formation + final output. Without logo confidences the
	// classifier has no primary signal and returns an empty list.
	blockList := blocks.Form(blocks.Opts{
		FPS:             res.FPS,
		MinBlockS:       *minBlockSec,
		MaxBlockS:       *maxBlockSec,
		MinShowSegmentS: *minShowSegSec,
		LogoThreshold:   *logoThreshold,
	}, res.LogoConfs, res.Blackframes, sil.events, res.FrameCount)

	switch *output {
	case "summary":
		writeSummary(res, elapsed, blockList)
	case "cutlist":
		writeCutlist(res.FPS, res.FrameCount, blockList)
	default:
		fmt.Fprintf(os.Stderr, "unknown output format: %q\n", *output)
		os.Exit(2)
	}
}

func writeSummary(res *pipeline.Result, elapsedS float64, bl []blocks.Block) {
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
		FPS:        res.FPS,
		Width:      res.Width,
		Height:     res.Height,
		FrameCount: res.FrameCount,
		DurationS:  float64(res.FrameCount) / res.FPS,
		Blocks:     make([][2]float64, len(bl)),
		Stats: map[string]any{
			"elapsed_s": elapsedS,
			"fps_proc":  float64(res.FrameCount) / elapsedS,
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
