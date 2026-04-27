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

	"path/filepath"
	"strings"

	"github.com/simonchrz/tv-detect/internal/blocks"
	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/logotrain"
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
		minShowSegSec  = flag.Float64("min-show-segment", 60, "min show between blocks before merging — also sets how long logo-present is required to declare block end")
		maxAdGapSec    = flag.Float64("max-ad-gap", 30, "post-refine merge: glue adjacent ad blocks if the gap between them is shorter than this (catches promo slates between ad break halves; 0 disables)")
		startExtendS   = flag.Float64("start-extend", 0, "pull each detected block's start back by N seconds (channel-specific systematic correction from user-feedback boundary drift; capped at the prior block's end)")
		endExtendS     = flag.Float64("end-extend", 0, "push each detected block's end forward by N seconds (channel-specific sponsor-tail trim from user-feedback drift; capped at the next block's start)")
		iframeSnapS    = flag.Float64("iframe-snap", 5, "post-refine snap each boundary to the nearest I-frame within ±N seconds. 0 = off. Real ad-inserts always align with encoder I-frames.")
		logoCrossS     = flag.Float64("logo-cross-refine", 2, "post-refine snap each boundary to the precise frame where logoConf crosses --logo-threshold within ±N seconds. 0 = off. Sub-frame precision (40 ms) using the existing 25-fps logo signal — no extra decode.")
		sceneCutSnapS  = flag.Float64("scene-cut-snap", 0, "post-refine snap each boundary to the nearest hard scene cut within ±N seconds. 0 = off (default — empirical test showed regressions on some shows when the I-frame snap was authoritative for those broadcasters). Set 1.5 to enable.")
		minAbsentS     = flag.Float64("min-absent-open", 5, "seconds of continuous logo-absent before a candidate block opens (filters single-frame flickers in the show)")
		refineWindowS  = flag.Float64("refine-window", 90, "search radius (s) for snapping rough block boundaries to a blackframe / silence")
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
		emitBumperCSV   = flag.Bool("emit-bumper-csv", false, "print per-frame bumper match score as CSV (max IoU across templates)")
		bumperTemplates = flag.String("bumper-templates", "", "comma-separated list of PNG paths used as channel-bumper reference frames (e.g. RTL's 'Mein RTL' end-of-ad-block card). All templates are matched per frame; max score wins. Color variants of the same animation should be added together.")
		bumperSnapS     = flag.Float64("bumper-snap", 10, "post-refine snap each ad-block END to the latest bumper-match peak within ±this seconds. 0 = off. Strongest deterministic ad-end signal when --bumper-templates is set; overrides logo/scene-cut/I-frame refinement for the END boundary.")
		bumperThresh    = flag.Float64("bumper-threshold", 0.85, "bumper match score required for a snap (default 0.85). Above all observed show-content false positives in validation.")
		nnBackbone      = flag.String("nn-backbone", "", "path to ONNX MobileNetV2 backbone (enables NN evidence). Empty = NN off.")
		nnHead          = flag.String("nn-head", "", "path to head.bin (1280 weights + 1 bias as float32 LE). Auto-finds <backbone-dir>/head.bin if empty.")
		nnChannelSlug   = flag.String("channel-slug", "", "channel slug (kabel-eins/prosieben/rtl/sat-1/sixx/vox) — only used if the loaded head.bin is a +CHAN format (5148 or 5152 B). Empty / unknown slugs are silently treated as all-zero one-hot.")
		nnWeight        = flag.Float64("nn-weight", 0.3, "blend weight of NN evidence vs logo (0 = logo only, 1 = NN only)")
		nnGate          = flag.Float64("nn-gate", 0.3, "ignore NN where |conf - 0.5| < this (0 = always use NN, 0.3 = only when conf < 0.2 or > 0.8)")
		nnSmoothS       = flag.Float64("nn-smooth", 10, "rolling-mean window (s, total) on NN confidences before blending. 0 = off. Single-frame backbone is noisy; 10s smoothing gives clean block boundaries.")
		logoSmoothS     = flag.Float64("logo-smooth", 5, "rolling-mean window (s, total) on logo confidences before block formation. 0 = off. Catches sub-second logo flickers caused by lower-third graphics that prevent the state machine from closing blocks (e.g. ProSieben/Galileo with persistent banner overlays).")
		autoTrain       = flag.Float64("auto-train", 0, "if --logo not provided, train one from the first N minutes of input and cache as <input-dir>/<basename>.trained.logo.txt")
		logoYOffset     = flag.Int("logo-y-offset", 0, "shift the logo template's Y coordinates by N pixels (= letterbox top-bar height). Use when a 16:9 program airs in a 4:3 broadcast container — the actual logo sits below the template's trained position because the visible content is pushed down by the letterbox bar.")
		autoTrainEdge   = flag.Int("auto-train-edge", 40, "Sobel edge threshold during auto-training")
		autoTrainPersist = flag.Float64("auto-train-persist", 0.85, "persistence threshold during auto-training (0.85 = pixel must be edge in 85% of sampled frames)")
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

	// Optional logo template. Resolved in order:
	//   1. --logo <path>                           — explicit
	//   2. --auto-train + cached <basename>.trained.logo.txt
	//   3. --auto-train + (train from first N min, cache, then use)
	// Shared read-only across all chunk workers.
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
	} else if *autoTrain > 0 {
		base := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
		trainedPath := filepath.Join(filepath.Dir(input), base+".trained.logo.txt")
		if _, err := os.Stat(trainedPath); err == nil {
			tmpl, err = logotemplate.Load(trainedPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "logo (cached %s): %v\n", trainedPath, err)
				os.Exit(1)
			}
			if !*quiet {
				fmt.Fprintf(os.Stderr,
					"logo (cached): %s — %dx%d %d edges\n",
					trainedPath, tmpl.Width(), tmpl.Height(), tmpl.EdgePositions)
			}
		} else {
			tmpl, err = autoTrainLogo(ctx, input, info, trainedPath, *autoTrain,
				*autoTrainEdge, *autoTrainPersist, *quiet)
			if err != nil {
				if !*quiet {
					fmt.Fprintf(os.Stderr,
						"auto-train: %v — proceeding without logo (empty cutlist)\n", err)
				}
				tmpl = nil
			}
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

	// I-frame timestamps: tiny ffprobe call, runs alongside the
	// heavy decode so it costs zero wall-time. Used by the block-
	// formation step to snap refined boundaries to actual encoder
	// cuts (commercial inserts always align with I-frames).
	type iframeResult struct {
		times []float64
		err   error
	}
	iframeCh := make(chan iframeResult, 1)
	go func() {
		t, err := signals.IFrameTimes(ctx, input)
		iframeCh <- iframeResult{times: t, err: err}
	}()

	// Resolve NN head path if backbone is set but head wasn't passed.
	if *nnBackbone != "" && *nnHead == "" {
		*nnHead = filepath.Join(filepath.Dir(*nnBackbone), "head.bin")
	}
	if *nnBackbone != "" && !*quiet {
		fmt.Fprintf(os.Stderr, "nn: backbone=%s  head=%s  weight=%.2f\n",
			*nnBackbone, *nnHead, *nnWeight)
	}

	// Video decode + per-frame signal extraction, split across N chunks.
	t0 := time.Now()
	res, err := pipeline.Run(ctx, pipeline.Opts{
		Input:          input,
		Workers:        *workers,
		DecodeWidth:    *decodeWidth,
		DecodeHeight:   *decodeHeight,
		BlackframeDurS: *blackframeDur,
		SceneThreshold: *sceneThreshold,
		LogoTemplate:    tmpl,
		LogoYOffset:     *logoYOffset,
		BumperTemplates: parseBumperTemplates(*bumperTemplates),
		NNBackbonePath: *nnBackbone,
		NNHeadPath:     *nnHead,
		NNChannelSlug:  *nnChannelSlug,
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
	ifr := <-iframeCh
	if ifr.err != nil && !*quiet {
		// Soft failure: fall through with no I-frames; block formation
		// just skips the I-frame snap step.
		fmt.Fprintln(os.Stderr, "iframes:", ifr.err)
	}
	res.IFrames = ifr.times
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
	if *emitBumperCSV {
		fmt.Println("idx,time_s,bumper_score")
		for i, c := range res.BumperConfs {
			fmt.Printf("%d,%.3f,%.4f\n", i, float64(i)/res.FPS, c)
		}
	}

	// Block formation + final output. Without logo confidences the
	// classifier has no primary signal and returns an empty list.
	blockList := blocks.Form(blocks.Opts{
		FPS:              res.FPS,
		MinBlockS:        *minBlockSec,
		MaxBlockS:        *maxBlockSec,
		MinShowSegmentS:  *minShowSegSec,
		MinAbsentToOpenS: *minAbsentS,
		LogoThreshold:    *logoThreshold,
		RefineWindowS:    *refineWindowS,
		NNWeight:         *nnWeight,
		NNGate:           *nnGate,
		NNSmoothS:        *nnSmoothS,
		LogoSmoothS:      *logoSmoothS,
		MaxAdGapS:        *maxAdGapSec,
		StartExtendS:     *startExtendS,
		EndExtendS:       *endExtendS,
		IFrameSnapS:      *iframeSnapS,
		LogoCrossRefineS: *logoCrossS,
		SceneCutSnapS:    *sceneCutSnapS,
		BumperSnapS:      *bumperSnapS,
		BumperThreshold:  *bumperThresh,
	}, res.LogoConfs, res.NNConfs, res.BumperConfs, res.Blackframes, sil.events,
		res.SceneCuts, res.IFrames, res.FrameCount)

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

// autoTrainLogo runs a one-pass training over the first minutes of
// input, samples 1 frame per second, and writes the result to
// trainedPath. Returns the loaded template ready for the detector.
// On failure (no consistently-edged pixels), returns an error and
// the caller proceeds without a logo (empty cutlist).
func autoTrainLogo(ctx context.Context, input string, info decode.Info,
	trainedPath string, minutes float64, edgeThresh int,
	persist float64, quiet bool) (*logotemplate.Template, error) {

	if !quiet {
		fmt.Fprintf(os.Stderr,
			"auto-train: sampling first %.0f min @ 1 fps from %s\n",
			minutes, filepath.Base(input))
	}
	d, err := decode.NewDecoder(ctx, decode.DecodeOpts{
		Input: input, StartS: 30, DurS: minutes * 60,
	})
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	defer d.Close()

	tr := logotrain.New(logotrain.Opts{
		FrameW: d.Width, FrameH: d.Height,
		EdgeThresh: edgeThresh, Persistence: persist,
	})
	stride := int(d.FPS + 0.5) // 1 frame per second
	if stride < 1 {
		stride = 1
	}
	read := 0
	for f := range d.Frames() {
		if read%stride == 0 {
			tr.Push(f.Pixels)
		}
		read++
	}
	if err := d.Err(); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	// Adaptive sweep: bump persistence upward until the bbox fits in
	// 5000 px² (typical real-logo size) or we run out of room. Same
	// behaviour as tv-detect-train-logo's default --max-bbox-area=5000.
	res, finalPersist := tr.ComputeAdaptive(5000, persist)
	if !res.HasLogo {
		return nil, fmt.Errorf("no pixel met persistence (sampled %d frames)",
			read/stride)
	}
	// Try to persist next to source; if that fails (e.g. recordings
	// dir is mounted read-only in a container), fall back to /tmp so
	// at least the in-memory template can be loaded back. Either way
	// the caller gets a usable template.
	cachedTo := trainedPath
	if err := tr.SaveTemplateAt(trainedPath, finalPersist); err != nil {
		alt := filepath.Join(os.TempDir(),
			"tvd-"+strings.ReplaceAll(filepath.Base(trainedPath), " ", "_"))
		if err2 := tr.SaveTemplateAt(alt, finalPersist); err2 != nil {
			return nil, fmt.Errorf("save template: %w (also tried %s: %v)",
				err, alt, err2)
		}
		if !quiet {
			fmt.Fprintf(os.Stderr,
				"auto-train: source dir read-only (%v) — cached to %s\n", err, alt)
		}
		cachedTo = alt
	}
	tmpl, err := logotemplate.Load(cachedTo)
	if err != nil {
		return nil, fmt.Errorf("reload trained template: %w", err)
	}
	if !quiet {
		note := ""
		if finalPersist > persist+1e-6 {
			note = fmt.Sprintf(" (persistence bumped %.2f → %.2f)", persist, finalPersist)
		}
		fmt.Fprintf(os.Stderr,
			"auto-train: bbox (%d,%d)-(%d,%d) %dx%d %d edges%s → %s\n",
			res.MinX, res.MinY, res.MaxX, res.MaxY,
			res.MaxX-res.MinX+1, res.MaxY-res.MinY+1, res.EdgePixels,
			note, cachedTo)
	}
	return tmpl, nil
}

// parseBumperTemplates splits a comma-separated paths list and drops
// empty/whitespace entries. Returns nil for an empty/whitespace input
// so the pipeline can skip the bumper detector entirely.
func parseBumperTemplates(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
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
