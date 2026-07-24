// Package pipeline splits a video input across N goroutines, each
// running its own ffmpeg decode subprocess on a time-range slice.
// Per-chunk signal results are merged back into a single full-file
// event list and per-frame logo-confidence array.
package pipeline

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/signals"
	"github.com/simonchrz/tv-detect/pkg/logotemplate"
)

// Opts configures a parallel run.
type Opts struct {
	Input        string
	Workers      int // number of parallel chunk workers (>= 1)
	DecodeWidth  int // 0 = native
	DecodeHeight int // 0 = native
	// FFmpegExtraInputArgs are injected before -i in the decode
	// subprocess. Use for error-tolerance flags on corrupt IPTV
	// streams: ["-err_detect", "ignore_err", "-fflags", "+discardcorrupt"]
	// keeps frames flowing through h264 PPS / packet-loss errors.
	FFmpegExtraInputArgs []string
	BlackframeDurS       float64
	SceneThreshold       float64
	LogoTemplate         *logotemplate.Template // nil = skip logo
	LogoYOffset          int                    // shift template y-coords by N pixels (letterbox correction)
	LogoEdgeThresh       int                    // Sobel |Gx|+|Gy| above which a frame pixel counts as edge. 0 = use defaultEdgeThresh (80). Raise per-channel for visually-busy channels where ad content scores false-positive logo-conf at the default — VOX/Nick/RTL ads have high edge density everywhere, so the asymmetric "fraction of template-edges with frame-edge" metric trips at 1.0 even when logo absent. Higher threshold = fewer frame pixels qualify as edge = harder for non-logo content to match.
	LogoCNNPath          string                 // path to per-channel CNN ONNX (= scripts/train_logo_cnn.py output). When set + LogoTemplate also set, the CNN replaces the edge-template Sobel signal but keeps the template's bbox for the same crop region. CNN is trained on user-labeled show vs ad frames cropped to bbox+margin and gives 87-100% per-channel accuracy on hard channels (vs ~50% random for edge-template).
	LogoCNNMargin        int                    // padding (px) around the template bbox before cropping for CNN input. Must match scripts/extract_logo_dataset.py LOGO_MARGIN_PX (= 50) used to build the training set.
	BumperTemplates      []string               // PNG paths for END-of-ad-block reference frames; nil/empty = skip
	BumperStartTemplates []string               // PNG paths for START-of-ad-block reference frames; nil/empty = skip. Independent template set + per-frame conf stream so a start-bumper hit can't pull a block end and vice versa.
	BumperStride         int                    // run bumper IoU every Nth frame (default 1 = every frame). Boundary snap only needs ~200ms precision; stride 5 at 25fps gives 5× speedup on bumper matching.
	WithAudio            bool                   // extract per-second audio RMS once at pipeline start and pass into NN.ConfidenceBatch as the (rms) feature. Required for +AUDIO heads (1282/1288 weights). Skipped when the loaded head doesn't have an audio dim — caller can leave this false to save the ffmpeg pass.
	NNBackbonePath       string                 // "" = skip NN
	NNHeadPath           string                 // ignored if backbone is empty
	NNChannelSlug        string                 // for +CHAN heads — set the per-recording one-hot input
	NNWhisperJSON        string                 // optional path to ~/.cache/tv-whisper/<uuid>.whisper.json. Loaded into the NNDetector via SetWhisperProbs when the head is MLP2 v2 (= n_whisper>0). Other formats ignore the data; missing file → neutral 0.5 fallback at inference.
	NNStartTS            int64                  // recording wall-clock start (unix s, = DVR start_real). Feeds the MLP4 minute-of-hour prior; 0 → neutral fallback.
	BoundaryHead         bool                   // load boundary_head.bin (sibling of NNHeadPath) and score per-frame boundary confidences into Result.BoundaryConfs. Set only when the caller intends to use them (--boundary-snap > 0), since the BNDR forward pass costs a second MLP per frame. No-op if the sibling file is absent.
}

// Result is the merged output across all chunks.
type Result struct {
	FPS              float64
	Width            int
	Height           int
	FrameCount       int // total frames processed (sum of chunk frame counts)
	Blackframes      []signals.BlackEvent
	SceneCuts        []signals.SceneCut
	IFrames          []float64 // ascending I-frame timestamps from ffprobe
	LogoConfs        []float64 // per-frame confidences in original order, nil if no logo
	NNConfs          []float64 // per-frame NN ad-confidence, nil if NN disabled
	BumperConfs      []float64 // per-frame max END-bumper match score, nil if no templates
	BumperStartConfs []float64 // per-frame max START-bumper match score, nil if no start templates
	BoundaryConfs    []float64 // per-frame BNDR boundary-head score, nil unless the boundary head loaded (opts.BoundarySnap)
	Letterbox        []signals.LetterboxEvent
	// Per-phase wall-time SUMMED across all chunk workers. Wall-time
	// not CPU-time, so parallel chunks stack — divide by Workers for
	// per-chunk feel. Driven by --quiet=false log emit in main.go.
	LogoNs   int64
	NNNs     int64
	BumperNs int64
	OtherNs  int64
}

// chunkPlan describes one worker's time-range slice.
type chunkPlan struct {
	index  int
	startS float64 // -ss seek offset
	durS   float64 // -t duration; 0 means "rest of file"
}

// chunkRes is what a worker produces; frame indices are LOCAL to the chunk.
type chunkRes struct {
	index            int
	startS           float64
	frameCount       int
	blackframes      []signals.BlackEvent
	sceneCuts        []signals.SceneCut
	logoConfs        []float64
	nnConfs          []float64
	bumperConfs      []float64
	bumperStartConfs []float64
	boundaryConfs    []float64
	letterbox        []signals.LetterboxEvent
	err              error
	// Per-phase cumulative wall-time for THIS chunk's worker. Summed
	// across chunks at merge time + emitted as a "pipeline-timing:"
	// line so a single-detect log shows where the budget went. Useful
	// to confirm/refute optimisation hypotheses (= bumper-stride
	// bump, nnBatch tuning, decode-pass speedup) before changing
	// production parameters. Wall-time, NOT CPU-time, so parallel
	// chunks' totals overlap — divide by Workers for an apples-to-
	// apples per-chunk view.
	logoNs   int64
	nnNs     int64
	bumperNs int64
	otherNs  int64 // black + scene + letterbox + decode-iter waits
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
		len(opts.BumperStartTemplates) > 0,
		opts.BoundaryHead), nil
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
		Input:          opts.Input,
		Width:          opts.DecodeWidth,
		Height:         opts.DecodeHeight,
		StartS:         p.startS,
		DurS:           p.durS,
		ExtraInputArgs: opts.FFmpegExtraInputArgs,
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
	var logoCNN *signals.LogoCNNDetector
	if opts.LogoTemplate != nil {
		if opts.LogoCNNPath != "" {
			// CNN logo signal — keeps template's bbox as the crop region
			// but ignores its edge-mask data; predictions come from
			// the per-channel CNN trained on user-labeled show/ad
			// crops. Falls back to edge-template if CNN load fails
			// (= missing file, ORT not initialised, bbox too small).
			t := opts.LogoTemplate
			margin := opts.LogoCNNMargin
			if margin == 0 {
				margin = 50
			}
			logoCNN, err = signals.NewLogoCNNDetector(opts.LogoCNNPath,
				d.Width, d.Height,
				t.MinX, t.MinY, t.MaxX, t.MaxY, margin)
			if err != nil {
				// Non-fatal: edge-template is the fallback path.
				logoCNN = nil
			}
		}
		if logoCNN == nil {
			logo, err = signals.NewLogoDetector(opts.LogoTemplate, d.Width, d.Height, opts.LogoEdgeThresh, opts.LogoYOffset)
			if err != nil {
				out.err = err
				return out
			}
		} else {
			defer logoCNN.Close()
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
		// Optional per-recording whisper-prob feed (MLP2 v2 heads).
		// Loader is best-effort: missing file or parse error → no-op,
		// the head's forward pass falls back to neutral 0.5 per frame.
		if opts.NNStartTS > 0 {
			nn.SetStartTS(opts.NNStartTS)
		}
		if opts.NNWhisperJSON != "" {
			if probs, err := signals.LoadWhisperPerSecond(opts.NNWhisperJSON); err == nil {
				nn.SetWhisperProbs(probs)
			} else {
				fmt.Fprintf(os.Stderr,
					"nn: whisper-json load failed (%v) — head sees 0.5 fallback\n",
					err)
			}
		}
	}

	// Boundary head (BNDR): loaded only when the caller intends to snap
	// (opts.BoundaryHead == --boundary-snap>0). Sibling of the NN head; a
	// nil detector (file absent / load error) degrades to neutral scores so
	// Result.BoundaryConfs stays length-aligned with the frame timeline.
	// Reuses the NN backbone embeddings below — no second ONNX pass.
	var boundary *signals.BoundaryDetector
	if nn != nil && opts.BoundaryHead {
		boundary, _ = signals.NewBoundaryDetector(
			filepath.Join(filepath.Dir(opts.NNHeadPath), "boundary_head.bin"))
	}

	count := 0
	// NN is the dominant per-frame cost — batch it. CoreML on M-series
	// gets 2-4× throughput from batched matmul. We accumulate up to
	// nnBatchSize frames + their logoConfs, run inference once, append
	// confidences in order. Frame pixels must be COPIED into the buffer
	// because the decoder reuses its slice for the next frame.
	//
	// 32 measured optimal on M-series CoreML (= 2026-05-04 A/B against
	// 8/16/64 on a real recording: 8→44.0 s wall, 16→42.1 s, 32→35.1 s,
	// 64→48.1 s — CoreML throughput curves up to 32 then regresses at
	// 64 due to memory or kernel-launch overhead). Backbone shrinks
	// from 95 s to 57 s sum (-40 %), wall -20 %. Pairs 1:1 with the
	// nn.go session-creation nnBatch constant — both must match.
	const nnBatchSize = 32
	// Two-phase NN (2026-07-18): phase 1 streams 32-frame batches through
	// the BACKBONE only (EmbedBatch), buffering embeddings + logo + rms
	// per frame for the whole chunk; phase 2 (after the frame loop) runs
	// the head over the chunk via ConfidenceChunk with real fps + chunk
	// offset. Required because whisper (per-second, absolute time) and
	// the temporal deltas (1-second spacing) cannot be built correctly
	// inside a 32-frame batch — the old batch-scope head pass fed the
	// head mistimed inputs (root cause of the production NN degradation:
	// whisper column replayed the first 32 s for every batch, temporal
	// deltas were ~25x smaller than trained + zeroed each batch edge).
	// Memory: ~1280 floats/frame ≈ 13 MB per typical chunk — fine.
	var (
		nnPxBuf     [][]byte
		nnEmbeds    []float32
		nnLogoAll   []float64
		nnRmsAll    []float64
		nnEmbedFail bool
	)
	flushNN := func() {
		if nn == nil || len(nnPxBuf) == 0 {
			return
		}
		tNN := time.Now()
		defer func() { out.nnNs += time.Since(tNN).Nanoseconds() }()
		emb := nn.EmbedBatch(nnPxBuf)
		if emb == nil {
			// Backbone inference failure — mark so phase 2 falls back to
			// neutral for the whole chunk (embedding offsets would
			// otherwise desync from logo/rms indices).
			nnEmbedFail = true
		} else {
			nnEmbeds = append(nnEmbeds, emb...)
		}
		nnPxBuf = nnPxBuf[:0]
	}
	for f := range d.Frames() {
		tFrame := time.Now()
		black.Push(f.Index, f.Pixels)
		scene.Push(f.Index, f.Pixels)
		letterbox.Push(f.Index, f.Pixels)
		out.otherNs += time.Since(tFrame).Nanoseconds()
		// Compute logo conf first; the NN may consume it (with-logo
		// head format passes the same per-frame logoConf as the 1281st
		// input feature). For a legacy head it's silently ignored.
		var logoConf float64 = 0.5
		if logoCNN != nil {
			tLogo := time.Now()
			logoConf = logoCNN.Confidence(f.Pixels)
			out.logoNs += time.Since(tLogo).Nanoseconds()
			out.logoConfs = append(out.logoConfs, logoConf)
		} else if logo != nil {
			tLogo := time.Now()
			logoConf = logo.Confidence(f.Pixels)
			out.logoNs += time.Since(tLogo).Nanoseconds()
			out.logoConfs = append(out.logoConfs, logoConf)
		}
		if nn != nil {
			pxCopy := make([]byte, len(f.Pixels))
			copy(pxCopy, f.Pixels)
			nnPxBuf = append(nnPxBuf, pxCopy)
			nnLogoAll = append(nnLogoAll, logoConf)
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
			nnRmsAll = append(nnRmsAll, rms)
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
			tBump := time.Now()
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
			if compute {
				out.bumperNs += time.Since(tBump).Nanoseconds()
			}
		}
		count++
	}
	flushNN() // any tail frames waiting in the batch buffer
	// Phase 2: head pass over the whole chunk with correctly-timed
	// whisper/temporal inputs (see the two-phase comment above).
	if nn != nil {
		tNN := time.Now()
		nFrames := len(nnLogoAll)
		var rmsArg []float64
		if len(audioRMS) > 0 {
			rmsArg = nnRmsAll
		}
		if nnEmbedFail || len(nnEmbeds) != nFrames*1280 {
			// Backbone failed somewhere — neutral chunk, same behaviour
			// as the old per-batch failure path.
			neutral := make([]float64, nFrames)
			for i := range neutral {
				neutral[i] = 0.5
			}
			out.nnConfs = append(out.nnConfs, neutral...)
		} else {
			out.nnConfs = append(out.nnConfs, toNNConfs(
				nn.ConfidenceChunk(nnEmbeds, nnLogoAll, rmsArg,
					nFrames, d.FPS, p.startS))...)
		}
		out.nnNs += time.Since(tNN).Nanoseconds()
		// Boundary scores off the SAME backbone embeddings (zero-copy
		// per-frame views). Neutral 0 when the head is absent or the
		// backbone failed — keeps the per-chunk length == nFrames so the
		// merged timeline never desyncs.
		if opts.BoundaryHead {
			bc := make([]float64, nFrames)
			if boundary != nil && !nnEmbedFail && len(nnEmbeds) == nFrames*1280 {
				embs := make([][]float32, nFrames)
				for i := 0; i < nFrames; i++ {
					embs[i] = nnEmbeds[i*1280 : (i+1)*1280]
				}
				if s := boundary.BoundaryScores(embs, d.FPS); len(s) == nFrames {
					bc = s
				}
			}
			out.boundaryConfs = append(out.boundaryConfs, bc...)
		}
	}
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
func merge(chunks []chunkRes, info decode.Info, hasLogo, hasNN, hasBumper, hasBumperStart, hasBoundary bool) *Result {
	r := &Result{
		FPS:    info.FPS,
		Width:  info.Width,
		Height: info.Height,
	}
	for i, c := range chunks {
		r.FrameCount += c.frameCount
		r.LogoNs += c.logoNs
		r.NNNs += c.nnNs
		r.BumperNs += c.bumperNs
		r.OtherNs += c.otherNs
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
		if hasBoundary {
			r.BoundaryConfs = append(r.BoundaryConfs, c.boundaryConfs...)
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
