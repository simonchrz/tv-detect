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
	LogoTemplate   *logotemplate.Template // nil = skip logo
	LogoYOffset    int                    // shift template y-coords by N pixels (letterbox correction)
	NNBackbonePath string                 // "" = skip NN
	NNHeadPath     string                 // ignored if backbone is empty
	NNChannelSlug  string                 // for +CHAN heads — set the per-recording one-hot input
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

	resCh := make(chan chunkRes, len(plans))
	var wg sync.WaitGroup
	for _, p := range plans {
		wg.Add(1)
		go func(p chunkPlan) {
			defer wg.Done()
			r := runChunk(ctx, opts, p, info)
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

	return merge(results, info, opts.LogoTemplate != nil, opts.NNBackbonePath != ""), nil
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

func runChunk(ctx context.Context, opts Opts, p chunkPlan, info decode.Info) chunkRes {
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
	var logo *signals.LogoDetector
	if opts.LogoTemplate != nil {
		logo, err = signals.NewLogoDetector(opts.LogoTemplate, d.Width, d.Height, 0, opts.LogoYOffset)
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
	for f := range d.Frames() {
		black.Push(f.Index, f.Pixels)
		scene.Push(f.Index, f.Pixels)
		// Compute logo conf first; the NN may consume it (with-logo
		// head format passes the same per-frame logoConf as the 1281st
		// input feature). For a legacy head it's silently ignored.
		var logoConf float64 = 0.5
		if logo != nil {
			logoConf = logo.Confidence(f.Pixels)
			out.logoConfs = append(out.logoConfs, logoConf)
		}
		if nn != nil {
			out.nnConfs = append(out.nnConfs, nn.Confidence(f.Pixels, logoConf))
		}
		count++
	}
	black.Finish()
	if err := d.Err(); err != nil {
		out.err = err
		return out
	}
	out.frameCount = count
	out.blackframes = black.Events()
	out.sceneCuts = scene.Cuts()
	return out
}

// merge stitches chunk-local results into full-file-timeline events
// and a single logo-confidence array. Blackframe runs that span a
// chunk boundary are reunited; suspicious scene-cuts at the very
// first frame of chunks 2..N are dropped (they're artifacts of the
// decoder starting fresh, not real content changes).
func merge(chunks []chunkRes, info decode.Info, hasLogo, hasNN bool) *Result {
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
