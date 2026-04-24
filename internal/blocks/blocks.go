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
	MinShowSegmentS  float64 // merge ad blocks separated by less show than this (default 120)
	LogoThreshold    float64 // per-frame logo confidence < this counts as "absent" (default 0.10)
	RefineWindowS    float64 // search radius for blackframe/silence snap (default 10)
}

// Form combines per-frame logo confidences (one entry per frame, len
// must equal NFrames) and the auxiliary event lists into a final block
// list. logoConf may be nil — then the result is empty (no detection
// possible without logo). Boundaries are refined to the nearest
// blackframe within RefineWindowS, falling back to silence end.
func Form(opts Opts, logoConf []float64, black []signals.BlackEvent,
	silence []signals.SilenceEvent, nFrames int) []Block {

	defaults(&opts)
	if logoConf == nil || len(logoConf) == 0 {
		return nil
	}

	// Step 1: per-frame logo presence boolean.
	present := make([]bool, len(logoConf))
	for i, c := range logoConf {
		present[i] = c >= opts.LogoThreshold
	}

	// Step 2: collapse to runs of "absent" (potential ad) longer than
	// MinBlockS. The state machine demands MinBlockS of continuous
	// absence to start a block; once started, it ends as soon as the
	// logo returns for MinShowSegmentS continuously.
	minAbsentFrames := int(opts.MinBlockS * opts.FPS)
	minPresentFrames := int(opts.MinShowSegmentS * opts.FPS)

	type run struct {
		startF, endF int
	}
	var raw []run
	var openStart = -1
	consecutivePresent := 0
	for i := 0; i < len(present); i++ {
		if !present[i] {
			if openStart < 0 {
				// scan ahead to confirm we'll have minAbsentFrames absent in the next window
				openStart = i
			}
			consecutivePresent = 0
			continue
		}
		// logo present
		if openStart < 0 {
			continue
		}
		consecutivePresent++
		if consecutivePresent < minPresentFrames {
			continue
		}
		// closed: openStart .. (i - consecutivePresent + 1) is the absence run.
		end := i - consecutivePresent + 1
		if end-openStart >= minAbsentFrames {
			raw = append(raw, run{openStart, end})
		}
		openStart = -1
		consecutivePresent = 0
	}
	// Trailing absent run that runs to EOF still counts.
	if openStart >= 0 && len(present)-openStart >= minAbsentFrames {
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
