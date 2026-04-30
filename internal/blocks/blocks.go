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
	MaxBlockFraction float64 // 0..1; reject any block longer than this fraction of total recording duration (default 0.5; pass a large value like 1.0 to effectively disable). Catches state-machine runaway false-positives where a sustained logo washout (e.g. white sixx logo over white sky for several minutes) opens a block that never closes — splitLongBlocks can only split when blackframes exist inside, so a pure-show stretch with no internal cuts stays as one giant block. Real ad blocks essentially never exceed 50 % of a recording, so dropping anything bigger is safe.
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
	BumperSnapS      float64 // post-refine snap each block boundary to the bumper-match peak within ±this seconds. 0 = off (default 10). Strongest deterministic boundary signal — a per-channel bumper template ("WIE SIXX IST DAS DENN?", "Mein RTL", etc) is a programmer-placed semantic marker, not statistical inference. Used for BOTH ends: end-bumpers (passed via the bumperConf positional arg) snap block END to the LATEST high-conf frame; start-bumpers (passed via startBumperConf) snap block START to the EARLIEST high-conf frame. Same window/threshold for both kinds — the distinction is only which boundary they target.
	BumperThreshold  float64 // bumper match score required for a snap (default 0.85). Above all observed show-content false-positive levels in validation, well below the 0.95+ peak of an actual bumper. Applied to both end- and start-bumper-conf streams.
	SpeakerWeight   float64  // 0..1 weight of speaker-fingerprint evidence vs logo+NN (default 0; pass --speaker-weight 0.3 to engage). When > 0 and SpeakerConfs is len(logoConf), per-frame "show-likelihood" is re-blended:  score = (1-SpeakerWeight)*old_score + SpeakerWeight*speaker — orthogonal signal to logo/NN/letterbox. Useful for shows with persistent recurring speakers (soap operas, court shows) where ad voice-overs are reliably distinct from the show's cast.
	IFrameSnapS      float64 // post-refine snap each boundary to the nearest I-frame within ±this seconds. 0 = off (default 5). Real ad-inserts almost always align with encoder I-frames; snapping here gives sub-second precision regardless of how coarse the rough boundary was.
	LogoCrossRefineS float64 // post-refine snap each boundary to the precise frame where the per-frame logoConf crosses LogoThreshold within ±this seconds. 0 = off (default 2). Uses the existing 25-fps logo signal — sub-frame precision (40 ms) without any extra decode. Falls through silently when the crossing is ambiguous.
	SceneCutSnapS    float64 // post-refine, pre-IFrame snap to the nearest hard scene cut (luma-histogram Bhattacharyya distance > SceneThreshold). 0 = off (default 1.5). Show→Ad transitions are by definition a complete shot change; the SceneDetector already feeds the voting cluster but the cluster centre often sits between the scene cut and a nearby black/silence anchor. This step forcibly moves the boundary to the exact scene-cut frame when one exists in the radius — the subsequent I-frame snap usually leaves it in place because broadcaster ad-inserts align scene cut + keyframe at the same PTS.
	LetterboxSnapS   float64 // post-refine snap to the nearest letterbox transition (onset for START, offset for END) within ±this seconds. 0 = off (default 5). Deterministic geometric signal on broadcasters that air 16:9 promos in 4:3 container (RTL/RTLZWEI) — overrides scene-cut/I-frame for the boundary it covers. No-op when LetterboxEvents is empty.
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
func Form(opts Opts, logoConf, nnConf, bumperConf, startBumperConf, speakerConf []float64,
	black []signals.BlackEvent,
	silence []signals.SilenceEvent, scenes []signals.SceneCut,
	letterbox []signals.LetterboxEvent,
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
	useSpeaker := opts.SpeakerWeight > 0 && len(speakerConf) == len(logoConf)
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
		// Speaker-fingerprint blend: ADDITIVE CENTERED nudge, not a
		// re-weighted average. Speaker conf in [0,1] is centered around
		// 0.5 (neutral = no info / non-speech / silence) so we subtract
		// 0.5 before scaling — values > 0.5 boost present-likelihood
		// (show-like voice), values < 0.5 reduce it (ad-like voice).
		// Re-weighted average would BOOST every frame because (1+cos)/2
		// is always > 0 for any non-opposite vectors, killing the
		// state machine's ability to open blocks.
		if useSpeaker {
			score += opts.SpeakerWeight * (speakerConf[i] - 0.5)
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
			black, silence, scenes, letterbox, true)
		endS = refineBoundaryVoting(endS, opts.RefineWindowS,
			black, silence, scenes, letterbox, false)
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
		// Letterbox snap is deterministic for 4:3-broadcasters airing
		// 16:9 promos: an onset = programme→promo cut, offset =
		// promo→programme. Runs after the scene-cut snap because
		// letterbox transitions are channel-geometric (no luma/chroma
		// dependence) and beat any scene-cut on broadcasters where it
		// fires. No-op on broadcasters that always air same aspect.
		if opts.LetterboxSnapS > 0 && len(letterbox) > 0 {
			startS = snapToLetterbox(startS, letterbox, opts.LetterboxSnapS, true)
			endS = snapToLetterbox(endS, letterbox, opts.LetterboxSnapS, false)
		}
		if opts.LogoCrossRefineS > 0 {
			startS = logoCrossingRefine(startS, opts.LogoCrossRefineS,
				logoConf, opts.LogoThreshold, opts.FPS, true)
			endS = logoCrossingRefine(endS, opts.LogoCrossRefineS,
				logoConf, opts.LogoThreshold, opts.FPS, false)
		}
		// Bumper snap is deterministic — overrides everything above for
		// the matching boundary when a high-confidence channel bumper
		// sits nearby. End-bumpers (e.g. sixx "WIE SIXX IST DAS DENN?")
		// mark ad→show; start-bumpers (e.g. sixx "WERBUNG"-card) mark
		// show→ad. Each kind only snaps its own boundary — using a
		// start-bumper hit to pull the block END would be semantically
		// wrong even if the match is high-confidence.
		if opts.BumperSnapS > 0 && len(bumperConf) > 0 {
			endS = snapToBumper(endS, bumperConf,
				opts.FPS, opts.BumperSnapS, opts.BumperThreshold)
		}
		if opts.BumperSnapS > 0 && len(startBumperConf) > 0 {
			startS = snapToBumperStart(startS, startBumperConf,
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

	// Step 3.6: drop runaway false-positive blocks BEFORE the long-block
	// split. State-machine pathology — a sustained logo-absence stretch
	// (sixx white-on-white washout, long outdoor scenes on light
	// background, etc) opens a block that never closes. If we let
	// splitLongBlocks run first, a 2400 s runaway gets carved into 3 ×
	// 800 s sub-blocks via blackframes; each survives the per-block cap
	// individually, and we end up with 3 false-positives instead of one.
	// Cutting BEFORE the split kills the whole runaway tree at once.
	// Real ad blocks essentially never exceed 50 % of a recording.
	if opts.MaxBlockFraction > 0 && nFrames > 0 {
		totalS := float64(nFrames) / opts.FPS
		cap := totalS * opts.MaxBlockFraction
		filtered := blocks[:0]
		for _, b := range blocks {
			if b.Duration() > cap {
				continue
			}
			filtered = append(filtered, b)
		}
		blocks = filtered
	}

	// Step 4: split overly long blocks at internal hard cuts (blackframe
	// boundaries inside the block). Keeps comskip's max_commercialbreak
	// behaviour so a 20-min "no logo" stretch (e.g. live news) doesn't
	// emit one giant block.
	if opts.MaxBlockS > 0 {
		blocks = splitLongBlocks(blocks, opts.MaxBlockS, black)
	}

	// Step 5: per-block boundary correction from learned channel/show
	// drift. SIGNED — positive StartExtendS pulls the block START
	// EARLIER (extend backward), negative pushes it LATER (shrink
	// from the front). Same symmetry for EndExtendS: positive extends
	// forward, negative pulls back. Per-show drift learning may set
	// either sign depending on which way the user systematically
	// trims.
	//
	// Clamps:
	//   START — between previous block's END (or 0) and current END
	//           minus MinBlockS (don't shrink below the min-block size).
	//   END   — between current START plus MinBlockS and next block's
	//           START (or recording duration).
	if opts.StartExtendS != 0 || opts.EndExtendS != 0 {
		totalS := float64(nFrames) / opts.FPS
		minDur := opts.MinBlockS
		if minDur <= 0 {
			minDur = 1.0
		}
		for i := range blocks {
			ns := blocks[i].StartS - opts.StartExtendS
			minStart := 0.0
			if i > 0 {
				minStart = blocks[i-1].EndS
			}
			maxStart := blocks[i].EndS - minDur
			if ns < minStart {
				ns = minStart
			}
			if ns > maxStart {
				ns = maxStart
			}
			blocks[i].StartS = ns

			ne := blocks[i].EndS + opts.EndExtendS
			maxEnd := totalS
			if i+1 < len(blocks) {
				maxEnd = blocks[i+1].StartS
			}
			minEnd := blocks[i].StartS + minDur
			if ne > maxEnd {
				ne = maxEnd
			}
			if ne < minEnd {
				ne = minEnd
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
	if o.MaxBlockFraction <= 0 {
		o.MaxBlockFraction = 0.5
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
	if o.LetterboxSnapS <= 0 {
		o.LetterboxSnapS = 5
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
	scenes []signals.SceneCut, letterbox []signals.LetterboxEvent,
	isStart bool) float64 {
	const backTolerance = 5.0
	const wBlack = 2.0
	const wLetter = 2.0 // deterministic geometric signal — same weight as blackframe
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
	for _, e := range letterbox {
		// Onset = programme→promo (ad-start). Offset = promo→programme
		// (ad-end). Wrong-direction events are filtered: their geometric
		// meaning doesn't match the boundary type.
		if isStart && !e.Onset {
			continue
		}
		if !isStart && e.Onset {
			continue
		}
		add(e.TimeS, wLetter)
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
// snapToLetterbox snaps a boundary timestamp to the EARLIEST letterbox
// onset (for START) or LATEST offset (for END) within the snap window.
// Picking the extreme rather than the nearest catches the case where
// the broadcast's RTL-branded Werbung-promo period (logo still visible
// in the badge → state machine stays closed) actually started 30-50 s
// before the rough boundary the state machine produced. The chain of
// letterbox events between true ad-start and logo-loss collapses to a
// single snap-target = the first letterbox onset.
//
// A small forward-tolerance (5 s) into the "wrong" direction is allowed
// so that a letterbox event slightly past the rough boundary (typical
// hysteresis lag) still matches.
func snapToLetterbox(t float64, events []signals.LetterboxEvent, r float64, isStart bool) float64 {
	if len(events) == 0 || r <= 0 {
		return t
	}
	const fwdTolerance = 5.0
	bestT := t
	found := false
	for _, e := range events {
		if isStart && !e.Onset {
			continue
		}
		if !isStart && e.Onset {
			continue
		}
		d := e.TimeS - t // negative = before t
		if isStart {
			// Window: [t-r, t+fwdTolerance]. Pick EARLIEST.
			if d < -r || d > fwdTolerance {
				continue
			}
			if !found || e.TimeS < bestT {
				bestT = e.TimeS
				found = true
			}
		} else {
			// Window: [t-fwdTolerance, t+r]. Pick LATEST.
			if d < -fwdTolerance || d > r {
				continue
			}
			if !found || e.TimeS > bestT {
				bestT = e.TimeS
				found = true
			}
		}
	}
	return bestT
}

// snapToBumperStart looks for the EARLIEST frame in [t-r, t+r] where
// a start-bumper match score exceeds threshold and returns its
// position (= first ad frame). Start-bumpers (announcer cards like
// sixx "WERBUNG"-with-presenter) are typically the very first visual
// of the ad break, so the real ad-start IS the bumper-start frame —
// no +1 offset like the end-bumper case. Returns t unchanged when no
// peak is found in the window.
func snapToBumperStart(t float64, bumperConf []float64, fps, r, threshold float64) float64 {
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
	for i := iLo; i <= iHi; i++ {
		if bumperConf[i] > threshold {
			return float64(i) / fps
		}
	}
	return t
}

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
