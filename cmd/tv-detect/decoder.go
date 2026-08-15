package main

import (
	"fmt"
	"os"

	"github.com/simonchrz/tv-detect/internal/blocks"
	"github.com/simonchrz/tv-detect/internal/signals"
)

// Selects which block-formation decoder runs. Both paths (full decode and
// --replay-signals) go through here so a sweep and a production run can never
// disagree about which decoder was used.
const (
	decoderForm = "form"
	decoderHSMM = "hsmm"
	// Stage-2/3 experiment variants. Measured 2026-07-29 on the converged
	// corpus: none beats Form with confidence (see hsmm_refine.go for the
	// numbers) — kept for re-evaluation, not production:
	decoderHSMMBlend  = "hsmm-blend"  // emission = Form's logo/NN blend
	decoderHSMMRefine = "hsmm-refine" // + narrow deterministic edge snaps
	decoderHSMMFull   = "hsmm-full"   // blend + refine
)

// hsmmPrior carries the --hsmm-* duration-prior flags (zero values = built-in
// defaults, resolved by hsmmDefaults). Deliberately NOT consulted by
// secondOpinion — see the comment at the flag definitions.
var hsmmPrior blocks.HSMMOpts

// hsmmBumperW is the --hsmm-bumper-w flag: weight of soft bumper boundary
// evidence in the hsmm* decoders (0 = off). See boundaryLP.
var hsmmBumperW float64

// boundaryLP turns a per-frame bumper-template confidence stream into
// per-second boundary log-scores for FormHSMM (stage-2b, 2026-07-29). Instead
// of Form's hard ±90 s snap, a template hit becomes evidence the Viterbi can
// WEIGH: lp[s+1] = w * (maxConf(second s) - thresh) / (1 - thresh), zero
// below thresh.
//
// guard (nil = off) is the per-second smoothed NN ad-probability, and applies
// the same veto as snapToBumperGuarded: an END-ident whose second the NN
// already reads as show (< 0.45) sits AFTER the block's closing trailers —
// under the trailer-= -show convention that is the WRONG boundary, so the hit
// contributes nothing. Start bumpers are unguarded, mirroring production.
func boundaryLP(conf, guard []float64, fps, thresh, w float64, nSec int) []float64 {
	lp := make([]float64, nSec+1)
	if thresh >= 1 || fps <= 0 {
		return lp
	}
	for s := 0; s < nSec; s++ {
		lo, hi := int(float64(s)*fps), int(float64(s+1)*fps)
		if hi > len(conf) {
			hi = len(conf)
		}
		m := 0.0
		for i := lo; i < hi; i++ {
			if conf[i] > m {
				m = conf[i]
			}
		}
		if m < thresh {
			continue
		}
		if guard != nil && s < len(guard) && guard[s] < 0.45 {
			continue
		}
		lp[s+1] = w * (m - thresh) / (1 - thresh)
	}
	return lp
}

// smoothForGuard reproduces Form's step-0 NN smoothing so the bumper guard in
// RefineHSMM sees the same stream Form's own guard would.
func smoothForGuard(nnConf []float64, opts blocks.Opts) []float64 {
	if opts.NNSmoothS <= 0 || opts.FPS <= 0 {
		return nnConf
	}
	halfW := int(opts.NNSmoothS * opts.FPS / 2)
	if halfW <= 0 {
		return nnConf
	}
	return blocks.SmoothMean(nnConf, halfW)
}

// formBlocks dispatches on --decoder.
//
// The hsmm branch deliberately ignores almost everything Form consumes. It
// takes the RAW per-second NN probability and nothing else: no logo blend, no
// bumper/scene/I-frame/letterbox snaps, no start/end extends. That is not an
// oversight, it is what was measured — see internal/blocks/hsmm.go for the
// numbers and docs/hsmm-holdout-preregistration.md for the pre-registered
// holdout. Feeding it the gated blend, or letting Form's ±90 s snap windows
// touch its edges, both measured NEGATIVE.
func formBlocks(decoder string, opts blocks.Opts,
	logoConf, nnConf, bumperConf, startBumperConf, speakerConf, boundaryConf []float64,
	black []signals.BlackEvent, silence []signals.SilenceEvent,
	scenes []signals.SceneCut, letterbox []signals.LetterboxEvent,
	iFrames []float64, nFrames int) []blocks.Block {

	switch decoder {
	case decoderHSMM, decoderHSMMBlend, decoderHSMMRefine, decoderHSMMFull:
		if len(nnConf) == 0 {
			// Production safety (2026-07-29 decoder switch): a recording
			// without NN confidences (old dump, exotic channel) must still
			// produce a cutlist. Form is the only decoder that works from
			// logo alone — fall back loudly instead of failing the detect.
			fmt.Fprintln(os.Stderr,
				"decoder-fallback form (no NN confidences for --decoder "+decoder+")")
			return blocks.Form(opts, logoConf, nnConf, bumperConf, startBumperConf,
				speakerConf, boundaryConf, black, silence, scenes, letterbox,
				iFrames, nFrames)
		}
		// Emission choice. "blend" mirrors Form's per-frame score so channels
		// where the LOGO carries the decision (Disney: NN wrong for six
		// minutes straight, logo right) are not invisible to the HSMM:
		//   ad = 1 - ((1-w)*logo + w*(1-nn))  with the channel's nn_weight.
		// Pure NN stays the default emission — it is what every published
		// HSMM number was measured with.
		//
		// ⚠️ Damit sind unter plain "hsmm" die per-Show/per-Kanal gesetzten
		// nn_weight und nn_gate WIRKUNGSLOS: die Emission ist reines NN, das
		// Logo geht gar nicht ein. Das ist kein Versehen, aber es hat schon
		// einen Fix still getötet — die Let's-Dance-Konfiguration vom
		// 2026-07-28 (nn_gate 0 / nn_weight 1 gegen das Logo-Ausblenden) tut
		// seit dem Decoder-Wechsel am 07-29 nichts, und niemandem fiel es auf,
		// bis eine Kantenmessung 623 s Fehler in genau dieser Sendung fand
		// (Ledger §3af). Deshalb warnt der Lauf jetzt laut, statt es
		// schweigend zu ignorieren.
		if decoder == decoderHSMM {
			if opts.NNGate > 0 || (opts.NNWeight > 0 && opts.NNWeight != 1) {
				fmt.Fprintf(os.Stderr,
					"warn: --nn-gate=%.2f/--nn-weight=%.2f haben unter --decoder hsmm "+
						"KEINE Wirkung (Emission ist reines NN, Logo geht nicht ein). "+
						"Wirksam sind hier nur --min/--max-block-sec und die --hsmm-* Flags.\n",
					opts.NNGate, opts.NNWeight)
			}
		}
		emit := nnConf
		if (decoder == decoderHSMMBlend || decoder == decoderHSMMFull) &&
			len(logoConf) == len(nnConf) && opts.NNWeight > 0 {
			w := opts.NNWeight
			emit = make([]float64, len(nnConf))
			for i := range emit {
				emit[i] = 1 - ((1-w)*logoConf[i] + w*(1-nnConf[i]))
			}
		}
		sec := blocks.PerSecondMean(emit, opts.FPS)
		ho := blocks.HSMMOpts{
			// Only the two block-length bounds are taken from the shared
			// flags, because those are the ones a caller legitimately
			// varies per channel. The segment-model parameters stay at
			// the built-in defaults unless --hsmm-* flags override them.
			AdMuS:     hsmmPrior.AdMuS,
			AdSD:      hsmmPrior.AdSD,
			ShowMuS:   hsmmPrior.ShowMuS,
			ShowSD:    hsmmPrior.ShowSD,
			DurW:      hsmmPrior.DurW,
			AdBiasLP:  hsmmPrior.AdBiasLP,
			MinBlockS: opts.MinBlockS,
			MaxBlockS: opts.MaxBlockS,
		}
		if hsmmBumperW > 0 && len(bumperConf) > 0 {
			guard := blocks.PerSecondMean(smoothForGuard(nnConf, opts), opts.FPS)
			ho.EndBoundaryLP = boundaryLP(bumperConf, guard, opts.FPS,
				opts.BumperThreshold, hsmmBumperW, len(sec))
			if len(startBumperConf) > 0 {
				ho.StartBoundaryLP = boundaryLP(startBumperConf, nil, opts.FPS,
					opts.BumperThreshold, hsmmBumperW, len(sec))
			}
		}
		bl := blocks.FormHSMM(sec, ho)
		if decoder == decoderHSMMRefine || decoder == decoderHSMMFull {
			bl = blocks.RefineHSMM(bl, opts.FPS, bumperConf, startBumperConf,
				smoothForGuard(nnConf, opts), scenes, letterbox, iFrames)
		}
		return bl
	case decoderForm, "":
		return blocks.Form(opts, logoConf, nnConf, bumperConf, startBumperConf,
			speakerConf, boundaryConf, black, silence, scenes, letterbox,
			iFrames, nFrames)
	default:
		fmt.Fprintf(os.Stderr, "unknown --decoder %q (want %s or %s)\n",
			decoder, decoderForm, decoderHSMM)
		os.Exit(2)
		return nil
	}
}

// secondOpinion runs the OTHER decoder and reports how far the two disagree,
// as second-level IoU (1.0 = identical, 0.0 = no overlap at all).
//
// It costs one extra pass over the per-second probabilities — 0.81 s on an
// 8.7-hour recording, against minutes of decode — it cannot alter a single
// output byte, and where the two decoders diverge sharply is precisely where
// block forming is doing something questionable. That makes it a
// review-triage signal, and being wrong about triage is cheap.
//
// Since the 2026-07-29 production switch to --decoder hsmm the roles are
// inverted: production runs the HSMM, so the second opinion is Form (with the
// full production opts — snaps, extends, blend). When production runs Form
// (fallback, sweeps), the second opinion is the HSMM at pinned defaults, the
// same pairing every agreement number before the switch used. Either way the
// reported figure stays "Form vs HSMM disagreement".
//
// What it is NOT: a quality score. Two decoders can agree and both be wrong
// (see dvr-rtl-1781909700, where both miss an ad block that starts at t=0 and
// therefore agree perfectly at IoU 0.000 against the labels).
func secondOpinion(decoder string, opts blocks.Opts, produced []blocks.Block,
	logoConf, nnConf, bumperConf, startBumperConf, speakerConf, boundaryConf []float64,
	black []signals.BlackEvent, silence []signals.SilenceEvent,
	scenes []signals.SceneCut, letterbox []signals.LetterboxEvent,
	iFrames []float64, nFrames int) (float64, int, bool) {

	if opts.FPS <= 0 {
		return 0, 0, false
	}
	var alt []blocks.Block
	switch decoder {
	case decoderHSMM, decoderHSMMBlend, decoderHSMMRefine, decoderHSMMFull:
		alt = blocks.Form(opts, logoConf, nnConf, bumperConf, startBumperConf,
			speakerConf, boundaryConf, black, silence, scenes, letterbox,
			iFrames, nFrames)
	default:
		if len(nnConf) == 0 {
			return 0, 0, false
		}
		alt = blocks.FormHSMM(blocks.PerSecondMean(nnConf, opts.FPS),
			blocks.HSMMOpts{MinBlockS: opts.MinBlockS, MaxBlockS: opts.MaxBlockS})
	}
	return blocks.OverlapIoU(produced, alt), len(alt), true
}
