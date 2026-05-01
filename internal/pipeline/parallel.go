// Package pipeline splits a video input across N goroutines, each
// running its own ffmpeg decode subprocess on a time-range slice.
// Per-chunk signal results are merged back into a single full-file
// event list and per-frame logo-confidence array.
package pipeline

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/signals"
	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

// Opts configures a parallel run.
type Opts struct {
	Input          string
	Workers        int     // number of parallel chunk workers (>= 1)
	DecodeWidth    int     // 0 = native
	DecodeHeight   int     // 0 = native
	BlackframeDurS float64
	SceneThreshold float64
	LogoTemplate    *logotemplate.Template // nil = skip logo
	LogoYOffset     int                    // shift template y-coords by N pixels (letterbox correction)
	BumperTemplates      []string          // PNG paths for END-of-ad-block reference frames; nil/empty = skip
	BumperStartTemplates []string          // PNG paths for START-of-ad-block reference frames; nil/empty = skip. Independent template set + per-frame conf stream so a start-bumper hit can't pull a block end and vice versa.
	BumperStride         int               // run bumper IoU every Nth frame (default 1 = every frame). Boundary snap only needs ~200ms precision; stride 5 at 25fps gives 5× speedup on bumper matching.
	WithAudio            bool              // extract per-second audio RMS once at pipeline start and pass into NN.ConfidenceBatch as the (rms) feature. Required for +AUDIO heads (1282/1288 weights). Skipped when the loaded head doesn't have an audio dim — caller can leave this false to save the ffmpeg pass.
	NNBackbonePath  string                 // "" = skip NN
	NNHeadPath      string                 // ignored if backbone is empty
	NNChannelSlug   string                 // for +CHAN heads — set the per-recording one-hot input
}

// Result is the merged output across all chunks.
type Result struct {
	FPS         float64
	Width       int
	Height      int
	FrameCount  int // total frames processed (sum of chunk frame counts)
	Blackframes []signals.BlackEvent
	SceneCuts   []signals.SceneCut
	IFrames     []float64 // ascending I-frame timestamps from ffprobe
	LogoConfs   []float64 // per-frame confidences in original order, nil if no logo
	NNConfs     []float64 // per-frame NN ad-confidence, nil if NN disabled
	BumperConfs      []float64 // per-frame max END-bumper match score, nil if no templates
	BumperStartConfs []float64 // per-frame max START-bumper match score, nil if no start templates
	Letterbox        []signals.LetterboxEvent
}

// chunkPlan describes one worker's time-range slice.
type chunkPlan struct {
	index  int
	startS float64 // -ss seek offset
	durS   float64 // -t duration; 0 means "rest of file"
}

// chunkRes is what a worker produces; frame indices are LOCAL to the chunk.
type chunkRes struct {
	index       int
	startS      float64
	frameCount  int
	blackframes []signals.BlackEvent
	sceneCuts   []signals.SceneCut
	logoConfs   []float64
	nnConfs     []float64
	bumperConfs      []float64
	bumperStartConfs []float64
	letterbox        []signals.LetterboxEvent
	err         error
}

// Run probes the input, plans chunks, spawns workers, merges results.
// If Workers == 1 the parallel machinery is still used but with a
// single chunk — same code path, easier to reason about than two
// branches.
func Run(ctx context.Context, opts Opts) (*Result, error) {
	info, err := decode.Probe(opts.Input)
	if err != nil {
		return nil, err
	}
	if info.FPS <= 0 || info.DurationS <= 0 {
		return nil, fmt.Errorf("probe returned fps=%f duration=%f — cannot plan chunks",
			info.FPS, info.DurationS)
	}
	w := opts.Workers
	if w < 1 {
		w = 1
	}
	plans := planChunks(info.DurationS, w)

	// Audio RMS is per-recording (one ffmpeg pass over the whole
	// stream) — extract it once here, then each chunk worker indexes
	// into the same slice. Cheap (~5-10 s on a 1 h recording) and
	// avoids 4× duplicated ffmpeg passes if every chunk did its own.
	var audioRMS []float32
	if opts.WithAudio {
		audioRMS = signals.ExtractAudioRMSPerSecond(
			ctx, opts.Input, int(info.DurationS)+1)
	}

	resCh := make(chan chunkRes, len(plans))
	var wg sync.WaitGroup
	for _, p := range plans {
		wg.Add(1)
		go func(p chunkPlan) {
			defer wg.Done()
			r := runChunk(ctx, opts, p, info, audioRMS)
			resCh <- r
		}(p)
	}
	wg.Wait()
	close(resCh)

	results := make([]chunkRes, 0, len(plans))
	for r := range resCh {
		if r.err != nil {
			return nil, fmt.Errorf("chunk %d: %w", r.index, r.err)
		}
		results = append(results, r)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].index < results[j].index })

	return merge(results, info,
		opts.LogoTemplate != nil,
		opts.NNBackbonePath != "",
		len(opts.BumperTemplates) > 0,
		len(opts.BumperStartTemplates) > 0), nil
}

func planChunks(totalS float64, workers int) []chunkPlan {
	plans := make([]chunkPlan, workers)
	chunkDur := totalS / float64(workers)
	for i := 0; i < workers; i++ {
		plans[i].index = i
		plans[i].startS = float64(i) * chunkDur
		if i == workers-1 {
			plans[i].durS = 0 // read to EOF — avoids float rounding leaving a tail
		} else {
			plans[i].durS = chunkDur
		}
	}
	return plans
}

func runChunk(ctx context.Context, opts Opts, p chunkPlan, info decode.Info, audioRMS []float32) chunkRes {
	out := chunkRes{index: p.index, startS: p.startS}
	d, err := decode.NewDecoder(ctx, decode.DecodeOpts{
		Input:  opts.Input,
		Width:  opts.DecodeWidth,
		Height: opts.DecodeHeight,
		StartS: p.startS,
		DurS:   p.durS,
	})
	if err != nil {
		out.err = err
		return out
	}
	defer d.Close()

	black := signals.NewBlackDetector(d.FPS, opts.BlackframeDurS, 0, 0)
	scene := signals.NewSceneDetector(d.FPS, opts.SceneThreshold)
	letterbox := signals.NewLetterboxDetector(d.FPS, d.Width, d.Height, 0, 0)
	var logo *signals.LogoDetector
	if opts.LogoTemplate != nil {
		logo, err = signals.NewLogoDetector(opts.LogoTemplate, d.Width, d.Height, 0, opts.LogoYOffset)
		if err != nil {
			out.err = err
			return out
		}
	}
	var bumper *signals.BumperDetector
	if len(opts.BumperTemplates) > 0 {
		bumper, err = signals.NewBumperDetector(opts.BumperTemplates, d.Width, d.Height, 0)
		if err != nil {
			out.err = err
			return out
		}
	}
	var bumperStart *signals.BumperDetector
	if len(opts.BumperStartTemplates) > 0 {
		bumperStart, err = signals.NewBumperDetector(opts.BumperStartTemplates, d.Width, d.Height, 0)
		if err != nil {
			out.err = err
			return out
		}
	}
	var nn *signals.NNDetector
	if opts.NNBackbonePath != "" {
		nn, err = signals.NewNNDetector(opts.NNBackbonePath, opts.NNHeadPath, d.Width, d.Height, opts.NNChannelSlug)
		if err != nil {
			out.err = err
			return out
		}
		defer nn.Close()
	}

	count := 0
	// NN is the dominant per-frame cost — batch it. CoreML on M-series
	// gets 2-4× throughput from batched matmul. We accumulate up to
	// nnBatchSize frames + their logoConfs, run inference once, append
	// confidences in order. Frame pixels must be COPIED into the buffer
	// because the decoder reuses its slice for the next frame.
	const nnBatchSize = 8
	var (
		nnPxBuf   [][]byte
		nnLogoBuf []float64
		nnRmsBuf  []float64
	)
	flushNN := func() {
		if nn == nil || len(nnPxBuf) == 0 {
			return
		}
		// Pass nnRmsBuf only when audio is in play; nil signals the
		// nn detector to use a neutral 0.5 if its head expects an
		// audio dim but we have no data (=  recording with no audio
		// stream).
		var rmsArg []float64
		if len(audioRMS) > 0 {
			rmsArg = nnRmsBuf
		}
		out.nnConfs = append(out.nnConfs,
			toNNConfs(nn.ConfidenceBatch(nnPxBuf, nnLogoBuf, rmsArg))...)
		nnPxBuf = nnPxBuf[:0]
		nnLogoBuf = nnLogoBuf[:0]
		nnRmsBuf = nnRmsBuf[:0]
	}
	for f := range d.Frames() {
		black.Push(f.Index, f.Pixels)
		scene.Push(f.Index, f.Pixels)
		letterbox.Push(f.Index, f.Pixels)
		// Compute logo conf first; the NN may consume it (with-logo
		// head format passes the same per-frame logoConf as the 1281st
		// input feature). For a legacy head it's silently ignored.
		var logoConf float64 = 0.5
		if logo != nil {
			logoConf = logo.Confidence(f.Pixels)
			out.logoConfs = append(out.logoConfs, logoConf)
		}
		if nn != nil {
			pxCopy := make([]byte, len(f.Pixels))
			copy(pxCopy, f.Pixels)
			nnPxBuf = append(nnPxBuf, pxCopy)
			nnLogoBuf = append(nnLogoBuf, logoConf)
			// Per-frame audio RMS = the per-second value at the
			// frame's wall-clock second. audioRMS is indexed by
			// absolute seconds across the recording; chunk's startS
			// + frame-local seconds gives that. Out-of-range falls
			// back to 0.5 (matches Python neutral fallback).
			rms := 0.5
			if len(audioRMS) > 0 {
				absSec := int(p.startS + float64(f.Index)/d.FPS)
				if absSec >= 0 && absSec < len(audioRMS) {
					rms = float64(audioRMS[absSec])
				}
			}
			nnRmsBuf = append(nnRmsBuf, rms)
			if len(nnPxBuf) == nnBatchSize {
				flushNN()
			}
		}
		if bumper != nil || bumperStart != nil {
			// Subsample: stride>1 means we only compute bumper IoU on
			// every Nth frame; the rest get 0 (= no match). Boundary
			// snap walks the array looking for peaks, so as long as
			// the bumper window (~2-3s) hits at least one sampled
			// frame, snap still works. At stride 5 + 25fps, the worst-
			// case sampling phase miss is 200ms — well below the
			// snap radius (90s) we configure in the daemon.
			s := opts.BumperStride
			if s <= 0 {
				s = 1
			}
			compute := count%s == 0
			if bumper != nil {
				if compute {
					out.bumperConfs = append(out.bumperConfs, bumper.Confidence(f.Pixels))
				} else {
					out.bumperConfs = append(out.bumperConfs, 0)
				}
			}
			if bumperStart != nil {
				if compute {
					out.bumperStartConfs = append(out.bumperStartConfs, bumperStart.Confidence(f.Pixels))
				} else {
					out.bumperStartConfs = append(out.bumperStartConfs, 0)
				}
			}
		}
		count++
	}
	flushNN() // any tail frames waiting in the batch buffer
	black.Finish()
	if err := d.Err(); err != nil {
		out.err = err
		return out
	}
	out.frameCount = count
	out.blackframes = black.Events()
	out.sceneCuts = scene.Cuts()
	out.letterbox = letterbox.Events()
	return out
}

// toNNConfs is a small adapter — the NN detector returns []float64
// already, but we keep this boundary so future detectors can return
// other shapes (e.g. probability + uncertainty) without touching
// every caller.
func toNNConfs(in []float64) []float64 { return in }

// merge stitches chunk-local results into full-file-timeline events
// and a single logo-confidence array. Blackframe runs that span a
// chunk boundary are reunited; suspicious scene-cuts at the very
// first frame of chunks 2..N are dropped (they're artifacts of the
// decoder starting fresh, not real content changes).
func merge(chunks []chunkRes, info decode.Info, hasLogo, hasNN, hasBumper, hasBumperStart bool) *Result {
	r := &Result{
		FPS:    info.FPS,
		Width:  info.Width,
		Height: info.Height,
	}
	for i, c := range chunks {
		r.FrameCount += c.frameCount
		// Shift blackframes into full-file timeline.
		for _, e := range c.blackframes {
			r.Blackframes = append(r.Blackframes, signals.BlackEvent{
				StartS:    e.StartS + c.startS,
				EndS:      e.EndS + c.startS,
				DurationS: e.DurationS,
			})
		}
		// Shift scene cuts; drop the very first cut of chunks 2..N
		// (scene-cut requires a previous frame, and the "previous" of
		// the first frame in a non-origin chunk comes from a different
		// chunk's decoder state — artifact).
		sc := c.sceneCuts
		if i > 0 && len(sc) > 0 && sc[0].Frame == 1 {
			sc = sc[1:]
		}
		for _, s := range sc {
			r.SceneCuts = append(r.SceneCuts, signals.SceneCut{
				Frame:    s.Frame,
				TimeS:    s.TimeS + c.startS,
				Distance: s.Distance,
			})
		}
		if hasLogo {
			r.LogoConfs = append(r.LogoConfs, c.logoConfs...)
		}
		if hasNN {
			r.NNConfs = append(r.NNConfs, c.nnConfs...)
		}
		if hasBumper {
			r.BumperConfs = append(r.BumperConfs, c.bumperConfs...)
		}
		if hasBumperStart {
			r.BumperStartConfs = append(r.BumperStartConfs, c.bumperStartConfs...)
		}
		// Drop the very first letterbox event of chunks 2..N — the
		// detector emits a state-confirmation as soon as it has seen
		// `hysteresis` frames of consistent state, which is meaningless
		// for a chunk that started mid-stream. (We don't know the prior
		// chunk's letterbox state without crossing the boundary.)
		lb := c.letterbox
		if i > 0 && len(lb) > 0 && lb[0].Frame < int(0.6*info.FPS) {
			lb = lb[1:]
		}
		for _, e := range lb {
			r.Letterbox = append(r.Letterbox, signals.LetterboxEvent{
				Frame: e.Frame,
				TimeS: e.TimeS + c.startS,
				Onset: e.Onset,
			})
		}
	}
	// Reunite adjacent blackframes split across a chunk boundary.
	r.Blackframes = coalesceBlack(r.Blackframes, 1.0/info.FPS+1e-3)
	return r
}

// coalesceBlack merges consecutive blackframe events with a gap <= gapS.
// Used to reunite a black run that was split at a chunk boundary.
func coalesceBlack(events []signals.BlackEvent, gapS float64) []signals.BlackEvent {
	if len(events) < 2 {
		return events
	}
	out := make([]signals.BlackEvent, 0, len(events))
	cur := events[0]
	for i := 1; i < len(events); i++ {
		e := events[i]
		if e.StartS-cur.EndS <= gapS {
			cur.EndS = e.EndS
			cur.DurationS = cur.EndS - cur.StartS
			continue
		}
		out = append(out, cur)
		cur = e
	}
	out = append(out, cur)
	return out
}
