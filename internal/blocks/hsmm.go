package blocks

import "math"

// Explicit-duration HMM (HSMM) block formation — THE PRODUCTION DECODER
// since 2026-07-29 (with DurW=15; Form remains the no-NN fallback and the
// second opinion).
//
// What settled it: every corpus measurement below was poisoned by label
// provenance — labels descend from Form's output (review edits only what a
// human notices), so Form was being scored against its own echo. A blind
// edge test (24 largest Form-vs-HSMM edge disputes, contact sheets without
// decoder markings, the user names the transition second) came back 22/23
// for the HSMM, median error 4 s vs Form's 90 s, and both of the blocks only
// the HSMM emitted turned out to be real ads (one 52 s long — below Form's
// 60 s minimum, structurally invisible to it). Against the blind-corrected
// labels: HSMM +0.061 IoU [+0.032,+0.090] over Form, n=68 — the first
// significant decoder result in this file's history. Everything below is
// kept as the record of how the wrong conclusion was reached honestly.
//
// Why this exists. Form scores each frame independently and then repairs the
// resulting edges with deterministic snaps (bumper / scene-cut / I-frame /
// letterbox). The HSMM instead scores whole SEGMENTS: a candidate block's cost
// is the sum of its per-second log-likelihoods plus a log-normal duration
// prior, and Viterbi picks the segmentation that maximises the total. Block
// length becomes learned structure rather than the hand-set MinBlockS /
// MaxBlockS constants.
//
// What was actually measured, because the port must ship exactly that:
//
//	                     training(22)   holdout(20)   golden(24)   ALL 58
//	  always production      0.839         0.843        0.843       0.845
//	  always HSMM            0.862         0.916        0.851       0.866
//	  delta                 +0.023        +0.073       +0.008      +0.022
//
// Read the LAST column, not the holdout. The holdout was pre-registered
// (docs/hsmm-holdout-preregistration.md) and H1 passed there at +0.073, but
// that sample got lucky in both directions and the figure did not survive
// contact with more data. Composition is not the explanation — the share of
// recordings where production already scores >= 0.90 is 35% / 38% / 36%
// across the three sets. The holdout simply drew mild ceiling losses (4 of 7
// wins above 0.90, against 5 of 21 overall).
//
// The stable shape across every set measured so far:
//
//	  production < 0.90   n=37   +0.083   28/37 wins
//	  production >= 0.90  n=21   -0.085    5/21
//
// HSMM raises floors and lowers ceilings. Individual losses reach -0.404
// (Micky Maus) and -0.365 (GZSZ, where production was perfect). Whether a
// +0.022 mean is worth that trade is a product decision, not a measurement —
// which is why this decoder is OFF by default and selected with --decoder.
//
// H2 (a label-free proxy gate that beats ungated HSMM) FAILED at -0.003, so
// there is deliberately no per-channel or per-show switch either: the same
// show runs in opposite directions (Galileo 0.762 -> +0.035, Galileo 360
// 0.976 -> -0.207). The discriminator is per-recording production quality,
// which is not knowable at detect time.
//
// Two things this decoder does NOT do, both load-bearing:
//
//  1. It consumes the RAW per-second NN probability, not Form's gated
//     logo/NN blend. Feeding it the blend measured negative on rtl (-0.033)
//     where the raw signal measures +0.151 — RTL's nn_gate sends 10.7% of
//     frames to a weak logo fallback (AUC 0.827 vs NN 0.973).
//  2. It applies NO deterministic snaps. Production's +-90 s bumper window on
//     an already-correct HSMM edge costs -0.158: the snaps are compensation
//     for weak block forming, not independent value. If someone later wants
//     snapping here, it has to be re-measured with narrow windows, not
//     inherited from Form's config.
//
// Kept numerically identical to scripts/hmm-decode-proto.py's viterbi_hsmm,
// including tie-breaking direction and the NEG sentinel, so the parity test
// can compare block lists exactly rather than approximately.

// HSMMOpts are the segment-model parameters. The defaults used for every
// measurement above are set by hsmmDefaults.
type HSMMOpts struct {
	AdMuS     float64 // median ad-block length (s)
	AdSD      float64 // log-space spread of ad lengths
	ShowMuS   float64 // median show-segment length (s)
	ShowSD    float64 // log-space spread of show lengths
	EmitW     float64 // weight on the per-second emission term
	DurW      float64 // weight on the duration prior
	MinBlockS float64 // shortest ad block that may be emitted
	MaxBlockS float64 // longest ad segment the decoder will consider

	// Optional per-second boundary evidence (stage-2b experiment, 2026-07-29):
	// log-score added when an AD segment ends (EndBoundaryLP) or starts
	// (StartBoundaryLP) at that second — soft bumper/ident evidence the
	// Viterbi can weigh against emission and duration, instead of Form's hard
	// post-hoc snap. Index t = boundary after second t-1 (len T+1). nil = off;
	// nil inputs keep the decoder byte-identical to the parity reference.
	EndBoundaryLP   []float64
	StartBoundaryLP []float64
}

func hsmmDefaults(o *HSMMOpts) {
	if o.AdMuS <= 0 {
		o.AdMuS = 4 * 60
	}
	if o.AdSD <= 0 {
		o.AdSD = 0.55
	}
	if o.ShowMuS <= 0 {
		o.ShowMuS = 12 * 60
	}
	if o.ShowSD <= 0 {
		o.ShowSD = 0.9
	}
	if o.EmitW <= 0 {
		o.EmitW = 1.0
	}
	if o.DurW <= 0 {
		o.DurW = 60.0
	}
	if o.MinBlockS <= 0 {
		o.MinBlockS = 60
	}
	if o.MaxBlockS <= 0 {
		o.MaxBlockS = 15 * 60
	}
}

// Shortest show segment the decoder will consider. Not exposed: it bounds the
// SEARCH, not the output.
//
// There is deliberately no matching MAXIMUM. The reference carried a 45-minute
// cap, and because the two states strictly alternate, any longer recording then
// could not be covered by show segments alone — the decoder was FORCED to
// invent an ad block to bridge two of them. Found 2026-07-27 on
// dvr-one-hd-1781285100: 55 minutes of ARD, ad-free, not one second anywhere
// above p=0.5, and it still emitted a 60 s block at the least-bad position.
// Every ad-free recording over 45 minutes had a phantom block, which is exactly
// the public-broadcaster material that needs no cutlist at all.
//
// Fixed by lifting the cap, NOT by allowing a show→show self-transition: that
// was tried first and let the decoder split one show into two for free to
// collect the duration prior twice, which made it drop a genuine 60 s ad break
// (the exact-min parity case). The duration prior already penalises an
// over-long show segment; it does not need a hard wall as well.
const hsmmShowMinS = 30

// PerSecondMean reduces a per-frame signal to per-second means — the rate the
// head was trained at. Non-overlapping windows, trailing partial second
// dropped, mirroring the reference implementation's to_seconds().
func PerSecondMean(x []float64, fps float64) []float64 {
	if fps <= 0 || len(x) == 0 {
		return nil
	}
	n := int(float64(len(x)) / fps)
	if n <= 0 {
		return nil
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		lo := int(float64(i) * fps)
		hi := int(float64(i+1) * fps)
		if hi > len(x) {
			hi = len(x)
		}
		if hi <= lo {
			continue
		}
		s := 0.0
		for j := lo; j < hi; j++ {
			s += x[j]
		}
		out[i] = s / float64(hi-lo)
	}
	return out
}

// FormHSMM decodes per-second ad probabilities into ad blocks (in seconds).
//
// p[i] is P(ad) for second i. Returns blocks as [startS, endS) with
// endS-startS >= MinBlockS, in ascending order.
func FormHSMM(p []float64, o HSMMOpts) []Block {
	hsmmDefaults(&o)
	T := len(p)
	if T == 0 {
		return nil
	}
	const (
		eps = 1e-6
		neg = -1e18
	)

	// Log-likelihood of "ad" and "show" per second, prefix-summed so a
	// segment's emission total is one subtraction.
	cumAd := make([]float64, T+1)
	cumShow := make([]float64, T+1)
	for i, v := range p {
		if v < eps {
			v = eps
		} else if v > 1-eps {
			v = 1 - eps
		}
		cumAd[i+1] = cumAd[i] + math.Log(v)*o.EmitW
		cumShow[i+1] = cumShow[i] + math.Log(1-v)*o.EmitW
	}

	logAdMu := math.Log(o.AdMuS)
	logShowMu := math.Log(o.ShowMuS)
	durLP := func(d int, logMu, sd float64) float64 {
		z := (math.Log(float64(d)) - logMu) / sd
		return -0.5 * z * z * o.DurW
	}

	// dp[t][k] = best score for a prefix of length t whose last segment is in
	// state k (0 = show, 1 = ad). bk[t][k] is that segment's start, bp[t][k]
	// the state it followed.
	dp := make([][2]float64, T+1)
	bk := make([][2]int, T+1)
	bp := make([][2]int, T+1)
	for t := 1; t <= T; t++ {
		dp[t][0], dp[t][1] = neg, neg
	}

	type stateCfg struct {
		dmin, dmax int
		cum        []float64
		logMu, sd  float64
	}
	cfg := [2]stateCfg{
		{hsmmShowMinS, T, cumShow, logShowMu, o.ShowSD},
		{int(o.MinBlockS), int(o.MaxBlockS), cumAd, logAdMu, o.AdSD},
	}

	for t := 1; t <= T; t++ {
		for k := 0; k < 2; k++ {
			c := cfg[k]
			prevs := []int{1 - k}
			best, bd, bpv := neg, 0, 0
			lo := t - c.dmax
			if lo < 0 {
				lo = 0
			}
			// Descend from the longest-allowed start to the shortest. With a
			// strict > comparison this keeps the LATEST start on ties, which
			// is what the reference does — do not relax to >=.
			for st := t - c.dmin; st >= lo; st-- {
				if st < 0 {
					break
				}
				for _, prev := range prevs {
					base := dp[st][prev]
					if st == 0 {
						base = 0.0
					}
					if base <= neg/2 {
						continue
					}
					sc := base + (c.cum[t] - c.cum[st]) + durLP(t-st, c.logMu, c.sd)
					if k == 1 { // ad segment: soft boundary evidence
						if o.EndBoundaryLP != nil && t < len(o.EndBoundaryLP) {
							sc += o.EndBoundaryLP[t]
						}
						if o.StartBoundaryLP != nil && st < len(o.StartBoundaryLP) {
							sc += o.StartBoundaryLP[st]
						}
					}
					if sc > best {
						best, bd, bpv = sc, st, prev
					}
				}
			}
			dp[t][k], bk[t][k], bp[t][k] = best, bd, bpv
		}
	}

	k := 0
	if dp[T][1] > dp[T][0] {
		k = 1
	}
	if dp[T][k] <= neg/2 {
		return nil
	}

	var rev []Block
	for t := T; t > 0; {
		st, prev := bk[t][k], bp[t][k]
		if k == 1 && float64(t-st) >= o.MinBlockS {
			rev = append(rev, Block{StartS: float64(st), EndS: float64(t)})
		}
		t, k = st, prev
	}
	out := make([]Block, 0, len(rev))
	for i := len(rev) - 1; i >= 0; i-- {
		out = append(out, rev[i])
	}
	return out
}

// OverlapIoU is the second-level intersection-over-union of two block lists.
//
// Deliberately SYMMETRIC, unlike the block_iou used for scoring against ground
// truth (which averages the best match per GT block and therefore depends on
// which list is "truth"). Here neither list is truth — the point is to measure
// how far two decoders disagree about the same recording, and a measure that
// changed when the arguments were swapped would be useless for that.
//
// Returns 1.0 when both lists are empty: two decoders that both found no ads
// agree completely.
func OverlapIoU(a, b []Block) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	inter := 0.0
	for _, x := range a {
		for _, y := range b {
			lo, hi := x.StartS, x.EndS
			if y.StartS > lo {
				lo = y.StartS
			}
			if y.EndS < hi {
				hi = y.EndS
			}
			if hi > lo {
				inter += hi - lo
			}
		}
	}
	total := func(bs []Block) float64 {
		s := 0.0
		for _, x := range bs {
			s += x.Duration()
		}
		return s
	}
	union := total(a) + total(b) - inter
	if union <= 0 {
		return 0.0
	}
	return inter / union
}
