package blocks

import "github.com/simonchrz/tv-detect/internal/signals"

// Edge refinement for HSMM output — stage 2 of the replace-the-state-machine
// plan (2026-07-28).
//
// The HSMM finds WHERE the blocks are, but its edges are only as sharp as the
// smoothed per-second NN transition (±10-30 s). Form's deterministic snaps
// (bumper / scene cut / I-frame / letterbox) give frame-exact edges. The two
// have never been combined properly:
//
//   - The 07-24 measurement "snaps cost -0.158 on HSMM edges" used Form's
//     production ±90 s windows, which drag an already-correct HSMM edge up to
//     90 s away. The same note records that with NARROW windows the snaps were
//     neutral (scene ±2 s: 0.928 vs 0.927 unsnapped).
//   - The bumper end-snap has since grown the NN guard (2026-07-28), which
//     removes the trailer-hit failure mode that made wide bumper windows
//     dangerous in the first place.
//
// So this applies the deterministic chain with DELIBERATELY NARROW radii, and
// with the guarded bumper snap. The radii are compile-time constants rather
// than reusing Opts.*SnapS: the production values are tuned for state-machine
// edges that can be tens of seconds off, and inheriting them here would
// recreate exactly the -0.158 mistake.
//
// MEASURED 2026-07-29, converged corpus (trailer-start convention), n=68
// human-labelled dumps, all decoders with the recording's real production
// flags:
//
//	prod (Form, guarded)  mean 0.875  median 0.926
//	hsmm                  Δ +0.009 [-0.015,+0.033]
//	hsmm-refine           Δ +0.011 [-0.012,+0.034]   <- this file
//	hsmm-blend            Δ +0.004 (one catastrophic loss: super-rtl -0.39)
//	hsmm-full             Δ +0.004
//	fitted duration priors on top: all deltas SHRINK (~+0.005) — not a lever.
//
// Refine does sharpen HSMM edges (median 0.904 -> 0.913 vs bare hsmm).
//
// SUPERSEDED 2026-07-29 (same day): the numbers above were measured against
// form-echo labels (see hsmm.go header). After the blind edge test and label
// correction, bare hsmm (DurW=15) measures +0.061 over Form and refine adds
// nothing on top (+0.053 < +0.061) — production runs bare hsmm; refine stays
// an unused experiment.
const (
	hsmmRefineSceneS  = 3.0  // scene cut: shot change at the real boundary
	hsmmRefineIFrameS = 3.0  // encoder I-frame: ad inserts align here
	hsmmRefineBumperS = 20.0 // guarded bumper: strongest semantic marker, but
	// only trusted near the HSMM edge — a genuine bumper sits at the break
	// boundary the HSMM already found to within its NN blur.
	hsmmRefineLetterboxS = 10.0
)

// RefineHSMM sharpens HSMM block edges with the deterministic snap chain.
// nnConf must be the same smoothed per-frame ad-probability stream the guard
// expects (Form's step-0 smoothing); pass nil to skip the bumper guard.
func RefineHSMM(bl []Block, fps float64,
	bumperConf, startBumperConf, nnConf []float64,
	scenes []signals.SceneCut, letterbox []signals.LetterboxEvent,
	iFrames []float64) []Block {

	out := make([]Block, len(bl))
	for i, b := range bl {
		startS, endS := b.StartS, b.EndS
		// Same ordering as Form: geometric/encoder signals first, bumper
		// last so the semantic marker wins when present.
		if len(scenes) > 0 {
			startS = snapToSceneCut(startS, scenes, hsmmRefineSceneS)
			endS = snapToSceneCut(endS, scenes, hsmmRefineSceneS)
		}
		if len(letterbox) > 0 {
			startS = snapToLetterbox(startS, letterbox, hsmmRefineLetterboxS, true)
			endS = snapToLetterbox(endS, letterbox, hsmmRefineLetterboxS, false)
		}
		if len(iFrames) > 0 {
			startS = snapToIFrame(startS, iFrames, hsmmRefineIFrameS)
			endS = snapToIFrame(endS, iFrames, hsmmRefineIFrameS)
		}
		if len(bumperConf) > 0 {
			endS = snapToBumperGuarded(endS, bumperConf, nnConf,
				fps, hsmmRefineBumperS, 0.75)
		}
		if len(startBumperConf) > 0 {
			startS = snapToBumperStart(startS, startBumperConf,
				fps, hsmmRefineBumperS, 0.75)
		}
		if endS <= startS { // a snap collapsed the block; keep the original
			startS, endS = b.StartS, b.EndS
		}
		out[i] = Block{StartS: startS, EndS: endS}
	}
	return out
}
