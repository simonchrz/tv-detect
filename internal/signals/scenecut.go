package signals

import "math"

// SceneCut is one detected hard cut between two adjacent frames.
// Frame is the index of the SECOND frame (the new shot starts here).
// Distance is the Bhattacharyya distance 0..1 between the two frames'
// joint RGB histograms (1 = max difference).
type SceneCut struct {
	Frame    int
	TimeS    float64
	Distance float64
}

// rgbBins is the per-channel quantisation: 8 bins per R/G/B → 512
// joint bins. Captures both luma and chroma shifts — luma-only missed
// cuts where two scenes share the same brightness distribution but
// differ in colour cast (e.g. warm-courtroom → dark-purple-stage works
// luma-only, but cold-outdoor → warm-subway-promo doesn't).
const rgbBins = 8
const rgbHistSize = rgbBins * rgbBins * rgbBins

// SceneDetector tracks the joint RGB histogram of the previous frame
// and flags pairs whose Bhattacharyya distance exceeds Threshold.
// Threshold of 0 selects 0.40 (matches the legacy luma-only setting
// that callers tune against; with RGB histograms the same nominal
// number is slightly stricter — most real ad-cuts still score >0.5).
type SceneDetector struct {
	fps       float64
	threshold float64
	prev      []float64 // previous-frame normalised RGB histogram (rgbHistSize)
	hasPrev   bool
	cuts      []SceneCut
}

// NewSceneDetector creates a detector with the given threshold (0..1).
// Pass 0 for the default (0.40).
func NewSceneDetector(fps, threshold float64) *SceneDetector {
	if threshold <= 0 {
		threshold = 0.40
	}
	return &SceneDetector{fps: fps, threshold: threshold,
		prev: make([]float64, rgbHistSize)}
}

// Push computes the joint RGB histogram for the frame and, if the
// distance to the previous frame exceeds the threshold, records a
// SceneCut at idx.
func (d *SceneDetector) Push(idx int, pixels []byte) {
	h := make([]int, rgbHistSize)
	nPx := len(pixels) / 3
	// 3-bit quantisation per channel: top 3 bits give the bin index.
	for i := 0; i < nPx; i++ {
		r := int(pixels[i*3]) >> 5
		g := int(pixels[i*3+1]) >> 5
		b := int(pixels[i*3+2]) >> 5
		h[(r<<6)|(g<<3)|b]++
	}
	hn := make([]float64, rgbHistSize)
	if nPx > 0 {
		inv := 1.0 / float64(nPx)
		for i, c := range h {
			hn[i] = float64(c) * inv
		}
	}
	if d.hasPrev {
		dist := bhattacharyyaDist(d.prev, hn)
		if dist > d.threshold {
			d.cuts = append(d.cuts, SceneCut{
				Frame:    idx,
				TimeS:    float64(idx) / d.fps,
				Distance: dist,
			})
		}
	}
	d.prev = hn
	d.hasPrev = true
}

// Cuts returns the recorded scene cuts. Safe to call any time.
func (d *SceneDetector) Cuts() []SceneCut { return d.cuts }

// bhattacharyyaDist returns 1 - BC where BC is the Bhattacharyya
// coefficient Σ sqrt(p_i * q_i). Both histograms must already be
// normalised so they sum to ~1. Range is [0, 1].
func bhattacharyyaDist(p, q []float64) float64 {
	var bc float64
	for i := range p {
		bc += math.Sqrt(p[i] * q[i])
	}
	if bc > 1 {
		bc = 1
	}
	return 1 - bc
}
