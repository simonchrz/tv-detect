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
	NNGate           float64 // 0..0.5: ignore NN where |conf - 0.5| < NNGate (default 0 = always use NN). Set to e.g. 0.3 to only let NN vote when it's confident (conf < 0.2 or > 0.8) — keeps logo-only behaviour when NN is unsure.
	NNSmoothS        float64 // rolling-mean window (in seconds, total) applied to nnConf before blending. 0 = off. A 10s mean (NNSmoothS=10) collapses single-frame backbone noise so the gating doesn't flip-flop boundaries; the deployed state machine already does temporal hysteresis but smoothing the raw NN signal recovers a much cleaner block-level result (per-frame acc stays similar, block-IoU jumps).
	LogoSmoothS      float64 // rolling-mean window (s, total) applied to logoConf. 0 = off. Some channels (ProSieben/Galileo) show constant lower-third graphics that intermittently cover or noise-up the logo ROI — sub-second absent flickers prevent the state machine from ever satisfying its consecutive-present hysteresis, so a 5s mean kills the flicker without smearing real (minutes-long) ad transitions.
	MaxAdGapS        float64 // post-refine merge: glue two adjacent ad blocks together if the gap between them is shorter than this (default 30). Catches station promo slates between ad break halves where MinShowSegmentS already let the state machine close, but the resulting "show" gap is too short to be real show.
	StartExtendS     float64 // pull each block's StartS back by this many seconds. Channel-specific systematic correction: users on some channels (RTL) consistently shift block-starts earlier than auto-detected, suggesting tv-detect's logo-loss latency runs late at the head of an ad break. Set per-channel from boundary-drift feedback; capped at the previous block's end (or 0 for the first block) so blocks never overlap.
	EndExtendS       float64 // push each block's EndS forward by this many seconds. Same idea for the tail: VOX/RTL show ~30-40s of sponsor cards / "Heute 20:15" slates after the actual ad break that users systematically want skipped. Capped at the next block's start (or recording duration for the last block).
	BumperSnapS      float64 // post-refine snap each block END to the latest bumper-match peak (channel station-id card) within ±this seconds. 0 = off (default 10). Strongest single ad-end signal we have when a per-channel bumper template is loaded — unlike logo/scene-cut/I-frame refinement, this is a deterministic channel-specific marker, not statistical inference. Block STARTS are NOT snapped (bumpers mark ad→show transitions, not show→ad).
	BumperThreshold  float64 // bumper match score required for a snap (default 0.85). Above all observed show-content false-positive levels in validation, well below the 0.95+ peak of an actual bumper. Only applied when BumperConfs is non-empty.
	IFrameSnapS      float64 // post-refine snap each boundary to the nearest I-frame within ±this seconds. 0 = off (default 5). Real ad-inserts almost always align with encoder I-frames; snapping here gives sub-second precision regardless of how coarse the rough boundary was.
	LogoCrossRefineS float64 // post-refine snap each boundary to the precise frame where the per-frame logoConf crosses LogoThreshold within ±this seconds. 0 = off (default 2). Uses the existing 25-fps logo signal — sub-frame precision (40 ms) without any extra decode. Falls through silently when the crossing is ambiguous.
	SceneCutSnapS    float64 // post-refine, pre-IFrame snap to the nearest hard scene cut (luma-histogram Bhattacharyya distance > SceneThreshold). 0 = off (default 1.5). Show→Ad transitions are by definition a complete shot change; the SceneDetector already feeds the voting cluster but the cluster centre often sits between the scene cut and a nearby black/silence anchor. This step forcibly moves the boundary to the exact scene-cut frame when one exists in the radius — the subsequent I-frame snap usually leaves it in place because broadcaster ad-inserts align scene cut + keyframe at the same PTS.
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
func Form(opts Opts, logoConf, nnConf, bumperConf []float64,
	black []signals.BlackEvent,
	silence []signals.SilenceEvent, scenes []signals.SceneCut,
	iFrames []float64, nFrames int) []Block {

	defaults(&opts)
	if logoConf == nil || len(logoConf) == 0 {
		return nil
	}

	// Step 0: optional rolling-mean smooth of NN + logo confidences.
	// Both signals are noisy at sub-second resolution — the MobileNetV2
	// backbone is single-frame and dips on cuts, the logo template
	// match dips when lower-third graphics or compression artifacts
	// touch the ROI. Without smoothing the state machine's consecutive-
	// present hysteresis can never close on shows with persistent
	// graphics overlay (e.g. ProSieben/Galileo), and NN-edge cases
	// fragment block boundaries. Smoothing the raw signals upstream
	// of the gating gives a much steadier downstream behaviour.
	useNN := opts.NNWeight > 0 && len(nnConf) == len(logoConf)
	if useNN && opts.NNSmoothS > 0 {
		halfW := int(opts.NNSmoothS * opts.FPS / 2)
		if halfW > 0 {
			nnConf = smoothMean(nnConf, halfW)
		}
	}
	if opts.LogoSmoothS > 0 {
		halfW := int(opts.LogoSmoothS * opts.FPS / 2)
		if halfW > 0 {
			logoConf = smoothMean(logoConf, halfW)
		}
	}

	// Step 1: per-frame "is the logo (or NN-equivalent show signal)
	// present" boolean. Single threshold over the (optionally blended)
	// confidence — keeps the downstream state machine identical
	// regardless of whether NN is in play.
	wL := 1.0 - opts.NNWeight
	wN := opts.NNWeight
	gate := opts.NNGate
	present := make([]bool, len(logoConf))
	for i, c := range logoConf {
		score := c
		if useNN {
			nn := nnConf[i]
			// If NN is unsure (close to 0.5), skip its vote and fall
			// back to logo-only — avoids the noisy-NN-wins-by-tiny-
			// margin failure mode.
			confident := gate <= 0 || abs(nn-0.5) >= gate
			if confident {
				score = wL*c + wN*(1-nn)
			}
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

	// Step 3: convert to seconds + refine boundaries via multi-signal
	// voting. All three independently extracted boundary signals
	// (blackframe, silence, scene-cut) cluster around the actual
	// hard cut within ~1-2 s; picking the candidate with the
	// HIGHEST SUM of source weights is much more robust than the
	// old "blackframe-then-silence-fallback" two-tier rule. After
	// voting we snap to the nearest encoder I-frame — real ad
	// inserts always align there, so this gives sub-second
	// precision basically for free.
	iFrameSnap := opts.IFrameSnapS
	if iFrameSnap < 0 {
		iFrameSnap = 0
	}
	blocks := make([]Block, 0, len(raw))
	for _, r := range raw {
		startS := float64(r.startF) / opts.FPS
		endS := float64(r.endF) / opts.FPS
		startS = refineBoundaryVoting(startS, opts.RefineWindowS,
			black, silence, scenes, true)
		endS = refineBoundaryVoting(endS, opts.RefineWindowS,
			black, silence, scenes, false)
		if iFrameSnap > 0 && len(iFrames) > 0 {
			startS = snapToIFrame(startS, iFrames, iFrameSnap)
			endS = snapToIFrame(endS, iFrames, iFrameSnap)
		}
		// Scene-cut snap AFTER I-frame snap: I-frame gives coarse
		// alignment to encoder boundaries (which are nearby for ad
		// inserts), then scene-cut snap pulls onto the exact visual
		// shot change. When the encoder forced an IDR at the cut both
		// already agree (no movement). When they disagree (encoder
		// placed an IDR at a regular GOP boundary off the cut), scene
		// cut is the semantically correct ad boundary.
		if opts.SceneCutSnapS > 0 && len(scenes) > 0 {
			startS = snapToSceneCut(startS, scenes, opts.SceneCutSnapS)
			endS = snapToSceneCut(endS, scenes, opts.SceneCutSnapS)
		}
		if opts.LogoCrossRefineS > 0 {
			startS = logoCrossingRefine(startS, opts.LogoCrossRefineS,
				logoConf, opts.LogoThreshold, opts.FPS, true)
			endS = logoCrossingRefine(endS, opts.LogoCrossRefineS,
				logoConf, opts.LogoThreshold, opts.FPS, false)
		}
		// Bumper snap is deterministic — overrides everything above for
		// the END boundary when a high-confidence channel bumper sits
		// nearby. Only END (bumper marks ad→show transition).
		if opts.BumperSnapS > 0 && len(bumperConf) > 0 {
			endS = snapToBumper(endS, bumperConf,
				opts.FPS, opts.BumperSnapS, opts.BumperThreshold)
		}
		if endS-startS < opts.MinBlockS {
			continue
		}
		blocks = append(blocks, Block{StartS: startS, EndS: endS})
	}

	// Step 3.5: merge adjacent blocks that ended up close together after
	// boundary refinement. The state-machine MinShowSegmentS check is
	// done on raw frame indices; refinement can move a block end forward
	// or a start backward by up to RefineWindowS, which can collapse a
	// borderline 60s "show" gap to a few seconds. Also catches station
	// promo slates (e.g. VOX "Heute 20:15") that the NN briefly
	// classified as show — the slate is short enough that the gap to
	// the next ad detection is below MaxAdGapS.
	if opts.MaxAdGapS > 0 && len(blocks) > 1 {
		merged := blocks[:1]
		for _, b := range blocks[1:] {
			last := &merged[len(merged)-1]
			if b.StartS-last.EndS <= opts.MaxAdGapS {
				last.EndS = b.EndS
				continue
			}
			merged = append(merged, b)
		}
		blocks = merged
	}

	// Step 4: split overly long blocks at internal hard cuts (blackframe
	// boundaries inside the block). Keeps comskip's max_commercialbreak
	// behaviour so a 20-min "no logo" stretch (e.g. live news) doesn't
	// emit one giant block.
	if opts.MaxBlockS > 0 {
		blocks = splitLongBlocks(blocks, opts.MaxBlockS, black)
	}

	// Step 5: per-block boundary extension from learned channel drift.
	// Cap to neighbouring blocks (or 0 / total duration) so we never
	// produce overlap or negative starts.
	if opts.StartExtendS > 0 || opts.EndExtendS > 0 {
		totalS := float64(nFrames) / opts.FPS
		for i := range blocks {
			ns := blocks[i].StartS - opts.StartExtendS
			minStart := 0.0
			if i > 0 {
				minStart = blocks[i-1].EndS
			}
			if ns < minStart {
				ns = minStart
			}
			blocks[i].StartS = ns

			ne := blocks[i].EndS + opts.EndExtendS
			maxEnd := totalS
			if i+1 < len(blocks) {
				maxEnd = blocks[i+1].StartS
			}
			if ne > maxEnd {
				ne = maxEnd
			}
			blocks[i].EndS = ne
		}
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
	if o.MaxAdGapS <= 0 {
		o.MaxAdGapS = 30
	}
	// BumperSnapS / BumperThreshold: only meaningful when the caller
	// passed a non-empty bumperConf slice. Defaults are no-op safe (a
	// zero-length conf slice short-circuits in snapToBumper).
	if o.BumperSnapS <= 0 {
		o.BumperSnapS = 10
	}
	if o.BumperThreshold <= 0 {
		o.BumperThreshold = 0.85
	}
	// IFrameSnapS, LogoCrossRefineS, MaxAdGapS, *SmoothS all use
	// "0 = off, positive = on" semantics. Production defaults live
	// at the CLI flag declarations in cmd/tv-detect/main.go so test
	// callers can opt out by passing 0 here without surprise.
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// refineBoundaryVoting picks the boundary candidate with the highest
// sum of source-weights. Sources are weighted blackframe=2.0,
// scene-cut=1.5, silence=1.0 — the same actual cut showing up in
// multiple sources is much more reliable than any single source.
//
// Candidates within ±2 s cluster into the same vote; the best cluster
// (by total weight, then by directional preference, then by distance
// to roughS) wins. Falls through to roughS unchanged if no candidate
// landed inside the window.
//
// Direction handling matches the legacy refineBoundary: for ad-start
// the real cut is typically AFTER the rough boundary (logo fades a
// few seconds before the cut); for ad-end it's BEFORE.
func refineBoundaryVoting(roughS, radiusS float64,
	black []signals.BlackEvent, silence []signals.SilenceEvent,
	scenes []signals.SceneCut, isStart bool) float64 {
	const backTolerance = 5.0
	const wBlack = 2.0
	const wScene = 1.5
	const wSilence = 1.0
	const clusterRadius = 2.0

	type cand struct {
		anchor float64
		weight float64
	}
	cands := make([]cand, 0, len(black)+len(silence)+len(scenes))
	add := func(anchor, weight float64) {
		d := anchor - roughS
		// Directional gating + radius filter.
		if isStart {
			if d < 0 && -d > backTolerance {
				return
			}
		} else {
			if d > 0 && d > backTolerance {
				return
			}
		}
		if abs(d) > radiusS {
			return
		}
		cands = append(cands, cand{anchor, weight})
	}
	for _, e := range black {
		if isStart {
			add(e.EndS, wBlack)
		} else {
			add(e.StartS, wBlack)
		}
	}
	for _, e := range silence {
		if isStart {
			add(e.EndS, wSilence)
		} else {
			add(e.StartS, wSilence)
		}
	}
	for _, c := range scenes {
		add(c.TimeS, wScene)
	}
	if len(cands) == 0 {
		return roughS
	}
	// Cluster: a candidate's effective score = sum of all weights
	// within ±clusterRadius. Pick the strongest, breaking ties by
	// directional preference (closer in the favoured direction wins).
	bestScore := -1.0
	bestAnchor := roughS
	bestDist := radiusS + 1
	for _, c := range cands {
		score := 0.0
		for _, o := range cands {
			if abs(o.anchor-c.anchor) <= clusterRadius {
				score += o.weight
			}
		}
		d := c.anchor - roughS
		var dist float64
		if isStart {
			if d >= 0 {
				dist = d
			} else {
				dist = -d * 2 // backward penalised
			}
		} else {
			if d <= 0 {
				dist = -d
			} else {
				dist = d * 2
			}
		}
		if score > bestScore || (score == bestScore && dist < bestDist) {
			bestScore = score
			bestAnchor = c.anchor
			bestDist = dist
		}
	}
	return bestAnchor
}

// snapToSceneCut returns the scene-cut timestamp closest to t within
// ±r, or t unchanged if no scene cut lies in the window. Scenes is
// expected to be in chronological order (the SceneDetector produces
// them in frame order, which is monotonic in TimeS). Linear scan —
// per-recording scene-cut counts are typically <500, so the cost is
// negligible vs the binary search complexity of snapToIFrame.
func snapToSceneCut(t float64, scenes []signals.SceneCut, r float64) float64 {
	if len(scenes) == 0 || r <= 0 {
		return t
	}
	bestT := t
	bestD := r + 1
	for _, c := range scenes {
		d := abs(c.TimeS - t)
		if d <= r && d < bestD {
			bestD = d
			bestT = c.TimeS
		}
		if c.TimeS-t > r {
			break // chronological — further entries are farther
		}
	}
	return bestT
}

// snapToIFrame returns the I-frame timestamp closest to t within ±r,
// snapToBumper looks for the LATEST frame in [t-r, t+r] where the
// bumper match score exceeds threshold and returns its position + 1
// frame (= first show frame after the bumper). Bumpers span 2-3
// seconds; we want the END of the bumper window, not the start of
// the match. If no peak is found, returns t unchanged.
func snapToBumper(t float64, bumperConf []float64, fps, r, threshold float64) float64 {
	if len(bumperConf) == 0 || r <= 0 || fps <= 0 {
		return t
	}
	iCenter := int(t * fps)
	iLo := iCenter - int(r*fps)
	iHi := iCenter + int(r*fps)
	if iLo < 0 {
		iLo = 0
	}
	if iHi >= len(bumperConf) {
		iHi = len(bumperConf) - 1
	}
	for i := iHi; i >= iLo; i-- {
		if bumperConf[i] > threshold {
			return float64(i+1) / fps
		}
	}
	return t
}

// or t unchanged if no I-frame falls inside the window. iFrames is
// expected to be sorted ascending — caller responsibility.
func snapToIFrame(t float64, iFrames []float64, r float64) float64 {
	if len(iFrames) == 0 || r <= 0 {
		return t
	}
	// Binary search for first iFrame >= t.
	lo, hi := 0, len(iFrames)
	for lo < hi {
		mid := (lo + hi) / 2
		if iFrames[mid] < t {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	bestT := t
	bestD := r + 1
	for _, idx := range []int{lo - 1, lo} {
		if idx < 0 || idx >= len(iFrames) {
			continue
		}
		d := abs(iFrames[idx] - t)
		if d <= r && d < bestD {
			bestD = d
			bestT = iFrames[idx]
		}
	}
	return bestT
}

// logoCrossingRefine snaps a boundary timestamp to the precise frame
// where the per-frame logoConf transitions across the logo-presence
// threshold within ±radiusS of roughS. Sub-frame precision (40 ms at
// 25 fps) using data we already have — no extra decode.
//
// Direction handling:
//
//	isStart=true  → ad starts where logo went absent. Snap to the
//	                LAST frame inside the window where conf was still
//	                ≥ threshold; boundary = next frame (= first absent).
//	isStart=false → ad ends where logo returned. Snap to the LAST
//	                frame where conf was still < threshold; boundary
//	                = next frame (= first present).
//
// Returns roughS unchanged if logoConf is empty, the window is empty,
// or the window is monotone (no crossing — caller's voted boundary
// stays). The threshold check uses the ALREADY-SMOOTHED logoConf
// when LogoSmoothS was applied upstream — that's intentional: the
// smoothed signal is what the state machine reasoned about, so we
// snap to the same view of "present" the rest of the pipeline sees.
func logoCrossingRefine(roughS, radiusS float64, logoConf []float64,
	threshold, fps float64, isStart bool) float64 {
	if len(logoConf) == 0 || fps <= 0 {
		return roughS
	}
	center := int(roughS * fps)
	radius := int(radiusS * fps)
	lo := center - radius
	if lo < 0 {
		lo = 0
	}
	hi := center + radius
	if hi >= len(logoConf) {
		hi = len(logoConf) - 1
	}
	if hi <= lo {
		return roughS
	}
	// Find every threshold crossing in the window; pick the one
	// closest to the rough boundary. Walking left-to-right and
	// recording transitions covers ad-start (present→absent) and
	// ad-end (absent→present) symmetrically; the directional filter
	// keeps only the relevant kind.
	bestFrame := -1
	bestDist := radius + 1
	for i := lo + 1; i <= hi; i++ {
		prevPresent := logoConf[i-1] >= threshold
		currPresent := logoConf[i] >= threshold
		if prevPresent == currPresent {
			continue
		}
		// Crossing here.
		var crossDirOK bool
		if isStart {
			crossDirOK = prevPresent && !currPresent
		} else {
			crossDirOK = !prevPresent && currPresent
		}
		if !crossDirOK {
			continue
		}
		d := i - center
		if d < 0 {
			d = -d
		}
		if d < bestDist {
			bestDist = d
			bestFrame = i
		}
	}
	if bestFrame < 0 {
		return roughS
	}
	return float64(bestFrame) / fps
}

// smoothMean returns a centered rolling-mean over x with a half-window
// of halfW samples (so total window = 2*halfW+1 samples). Edges use
// the truncated window — no padding, no NaN.
func smoothMean(x []float64, halfW int) []float64 {
	n := len(x)
	if n == 0 || halfW <= 0 {
		return x
	}
	cs := make([]float64, n+1)
	for i, v := range x {
		cs[i+1] = cs[i] + v
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		lo := i - halfW
		if lo < 0 {
			lo = 0
		}
		hi := i + halfW + 1
		if hi > n {
			hi = n
		}
		out[i] = (cs[hi] - cs[lo]) / float64(hi-lo)
	}
	return out
}
