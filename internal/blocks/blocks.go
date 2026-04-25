// Package blocks turns per-frame signals into a final commercial-block
// cutlist. Mirrors comskip's logic at a high level: logo presence is
// the primary signal, blackframe + silence events refine the rough
// boundaries to the actual frame where the cut happens.
package blocks

import (
	"github.com/simonchrz/tv-detect/internal/signals"
)

// Block is one detected commercial segment.
type Block struct {
	StartS float64
	EndS   float64
}

// Duration in seconds.
func (b Block) Duration() float64 { return b.EndS - b.StartS }

// Opts configures the formation thresholds.
type Opts struct {
	FPS              float64
	MinBlockS        float64 // ignore blocks shorter than this (default 60)
	MaxBlockS        float64 // split blocks longer than this (default 900)
	MinShowSegmentS  float64 // merge ad blocks separated by less show than this (default 60)
	MinAbsentToOpenS float64 // require this many seconds of continuous logo-absent to open a block (default 5)
	LogoThreshold    float64 // per-frame logo confidence < this counts as "absent" (default 0.10)
	RefineWindowS    float64 // search radius for blackframe/silence snap (default 90)
	NNWeight         float64 // 0..1 weight of NN evidence vs logo (default 0). When > 0 and NNConfs supplied to Form, the per-frame ad-likelihood is a weighted blend.
}

// Form combines per-frame logo confidences and (optionally) NN
// confidences with the auxiliary event lists into a final block list.
//
// logoConf is the primary signal: high = logo present (= show). Pass
// nil/empty to opt out of detection (returns no blocks).
//
// nnConf is the optional NN ad-confidence (high = ad). When supplied
// and Opts.NNWeight > 0, the per-frame "show-likelihood" used by the
// state machine is the weighted blend
//
//	show = (1 - NNWeight) * logoConf + NNWeight * (1 - nnConf)
//
// so NNWeight=0 reproduces the logo-only behaviour and NNWeight=1
// makes the NN authoritative. Length of nnConf must match logoConf
// (the pipeline guarantees this when both detectors run).
func Form(opts Opts, logoConf, nnConf []float64, black []signals.BlackEvent,
	silence []signals.SilenceEvent, nFrames int) []Block {

	defaults(&opts)
	if logoConf == nil || len(logoConf) == 0 {
		return nil
	}

	// Step 1: per-frame "is the logo (or NN-equivalent show signal)
	// present" boolean. Single threshold over the (optionally blended)
	// confidence — keeps the downstream state machine identical
	// regardless of whether NN is in play.
	useNN := opts.NNWeight > 0 && len(nnConf) == len(logoConf)
	wL := 1.0 - opts.NNWeight
	wN := opts.NNWeight
	present := make([]bool, len(logoConf))
	for i, c := range logoConf {
		score := c
		if useNN {
			score = wL*c + wN*(1-nnConf[i])
		}
		present[i] = score >= opts.LogoThreshold
	}

	// Step 2: state machine over per-frame present/absent.
	//
	// Two hysteresis thresholds:
	//   minOpenFrames    — N consecutive absent frames needed to OPEN
	//                      a candidate block (filters single-frame
	//                      logo-flickers in the middle of a show)
	//   minPresentFrames — M consecutive present frames needed to CLOSE
	//                      the block (so brief logo-back moments inside
	//                      an ad break — e.g. station promo with logo —
	//                      don't prematurely terminate the block)
	//
	// minBlockFrames filters the closed block by total absence span.
	minOpenFrames := int(opts.MinAbsentToOpenS * opts.FPS)
	minPresentFrames := int(opts.MinShowSegmentS * opts.FPS)
	minBlockFrames := int(opts.MinBlockS * opts.FPS)

	type run struct {
		startF, endF int
	}
	var raw []run
	openStart := -1                  // first absent frame of the candidate run, -1 = closed
	pendingStart := -1               // first absent frame of an unconfirmed run
	consecutiveAbsent := 0
	consecutivePresent := 0
	for i := 0; i < len(present); i++ {
		if !present[i] {
			consecutivePresent = 0
			if openStart >= 0 {
				continue // already inside a confirmed open block
			}
			if pendingStart < 0 {
				pendingStart = i
			}
			consecutiveAbsent++
			if consecutiveAbsent >= minOpenFrames {
				openStart = pendingStart
			}
			continue
		}
		// logo present
		consecutiveAbsent = 0
		pendingStart = -1
		if openStart < 0 {
			continue
		}
		consecutivePresent++
		if consecutivePresent < minPresentFrames {
			continue
		}
		end := i - consecutivePresent + 1
		if end-openStart >= minBlockFrames {
			raw = append(raw, run{openStart, end})
		}
		openStart = -1
		consecutivePresent = 0
	}
	// Trailing absent run that runs to EOF still counts.
	if openStart >= 0 && len(present)-openStart >= minBlockFrames {
		raw = append(raw, run{openStart, len(present)})
	}

	// Step 3: convert to seconds + refine boundaries with blackframe /
	// silence events (prefer blackframe — it's a hard cut, silence is
	// typically 100-300 ms wider than the actual visual transition).
	blocks := make([]Block, 0, len(raw))
	for _, r := range raw {
		startS := float64(r.startF) / opts.FPS
		endS := float64(r.endF) / opts.FPS
		startS = refineBoundary(startS, opts.RefineWindowS, black, silence, true)
		endS = refineBoundary(endS, opts.RefineWindowS, black, silence, false)
		if endS-startS < opts.MinBlockS {
			continue
		}
		blocks = append(blocks, Block{StartS: startS, EndS: endS})
	}

	// Step 4: split overly long blocks at internal hard cuts (blackframe
	// boundaries inside the block). Keeps comskip's max_commercialbreak
	// behaviour so a 20-min "no logo" stretch (e.g. live news) doesn't
	// emit one giant block.
	if opts.MaxBlockS > 0 {
		blocks = splitLongBlocks(blocks, opts.MaxBlockS, black)
	}
	return blocks
}

// refineBoundary snaps a rough boundary timestamp to the best matching
// blackframe within radiusS, falling back to silence if no blackframe
// fits. The snap is directional:
//
//   - isStart=true: logo confidence drops EARLY (logo fades out before
//     the actual hard cut). The real ad-start blackframe is therefore
//     LATER than the rough boundary. Prefer blackframes at or after
//     roughS, but allow a small backward window in case the cut
//     happened a few seconds before logo-loss (rare).
//   - isStart=false: logo confidence rises LATE (we wait minPresent
//     frames of "logo present" before declaring ad-end). The real
//     ad-end blackframe is therefore EARLIER than the rough boundary.
//     Prefer blackframes at or before roughS.
//
// Within the preferred direction, take the closest blackframe.
func refineBoundary(roughS, radiusS float64, black []signals.BlackEvent,
	silence []signals.SilenceEvent, isStart bool) float64 {
	const backTolerance = 5.0 // small slack into the "wrong" direction

	best := roughS
	bestDist := radiusS + 1
	for _, e := range black {
		var anchor float64
		if isStart {
			anchor = e.EndS // ad starts where the black ended
		} else {
			anchor = e.StartS // ad ends where the black started
		}
		d := anchor - roughS
		var dist float64
		if isStart {
			// Prefer anchor >= roughS (forward); allow small backward.
			if d >= 0 {
				dist = d
			} else if -d <= backTolerance {
				dist = -d * 2 // backward penalised
			} else {
				continue
			}
		} else {
			// Prefer anchor <= roughS (backward); allow small forward.
			if d <= 0 {
				dist = -d
			} else if d <= backTolerance {
				dist = d * 2
			} else {
				continue
			}
		}
		if dist <= radiusS && dist < bestDist {
			best = anchor
			bestDist = dist
		}
	}
	if bestDist <= radiusS {
		return best
	}
	// No blackframe nearby — fall back to silence boundary.
	for _, e := range silence {
		var anchor float64
		if isStart {
			anchor = e.EndS
		} else {
			anchor = e.StartS
		}
		d := anchor - roughS
		var dist float64
		if isStart {
			if d >= 0 {
				dist = d
			} else if -d <= backTolerance {
				dist = -d * 2
			} else {
				continue
			}
		} else {
			if d <= 0 {
				dist = -d
			} else if d <= backTolerance {
				dist = d * 2
			} else {
				continue
			}
		}
		if dist <= radiusS && dist < bestDist {
			best = anchor
			bestDist = dist
		}
	}
	return best
}

func splitLongBlocks(blocks []Block, maxS float64, black []signals.BlackEvent) []Block {
	out := make([]Block, 0, len(blocks))
	for _, b := range blocks {
		if b.Duration() <= maxS {
			out = append(out, b)
			continue
		}
		// Find blackframes inside the block to split on. Take the one
		// closest to the midpoint; recurse on the two halves.
		mid := (b.StartS + b.EndS) / 2
		var splitAt float64
		bestDist := b.Duration()
		for _, e := range black {
			anchor := (e.StartS + e.EndS) / 2
			if anchor <= b.StartS || anchor >= b.EndS {
				continue
			}
			d := abs(anchor - mid)
			if d < bestDist {
				splitAt = anchor
				bestDist = d
			}
		}
		if splitAt == 0 {
			out = append(out, b)
			continue
		}
		left := []Block{{b.StartS, splitAt}}
		right := []Block{{splitAt, b.EndS}}
		out = append(out, splitLongBlocks(left, maxS, black)...)
		out = append(out, splitLongBlocks(right, maxS, black)...)
	}
	return out
}

func defaults(o *Opts) {
	if o.FPS <= 0 {
		o.FPS = 25
	}
	if o.MinBlockS <= 0 {
		o.MinBlockS = 60
	}
	if o.MaxBlockS <= 0 {
		o.MaxBlockS = 900
	}
	if o.MinShowSegmentS <= 0 {
		o.MinShowSegmentS = 60
	}
	if o.MinAbsentToOpenS <= 0 {
		o.MinAbsentToOpenS = 5
	}
	if o.LogoThreshold <= 0 {
		o.LogoThreshold = 0.10
	}
	if o.RefineWindowS <= 0 {
		o.RefineWindowS = 90
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
