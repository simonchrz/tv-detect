package signals

import "math"

// SceneCut is one detected hard cut between two adjacent frames.
// Frame is the index of the SECOND frame (the new shot starts here).
// Distance is the Bhattacharyya distance 0..1 between the two frames'
// luma histograms (1 = max difference).
type SceneCut struct {
	Frame    int
	TimeS    float64
	Distance float64
}

// SceneDetector tracks the luma histogram of the previous frame and
// flags pairs whose histogram distance exceeds Threshold. Threshold
// of 0 selects 0.40 (approximately matching ffmpeg's select='gt(scene,0.4)'
// which is the conventional hard-cut sensitivity).
type SceneDetector struct {
	fps       float64
	threshold float64
	prev      [256]float64 // previous-frame normalised luma histogram
	hasPrev   bool
	cuts      []SceneCut
}

// NewSceneDetector creates a detector with the given threshold (0..1).
// Pass 0 for the default (0.40).
func NewSceneDetector(fps, threshold float64) *SceneDetector {
	if threshold <= 0 {
		threshold = 0.40
	}
	return &SceneDetector{fps: fps, threshold: threshold}
}

// Push computes the histogram for the frame and, if the distance to
// the previous frame exceeds the threshold, records a SceneCut at idx.
func (d *SceneDetector) Push(idx int, pixels []byte) {
	var h [256]int
	for i := 0; i < len(pixels); i += 3 {
		y := (77*int(pixels[i]) + 150*int(pixels[i+1]) + 29*int(pixels[i+2])) >> 8
		h[y]++
	}
	total := float64(len(pixels) / 3)
	var hn [256]float64
	if total > 0 {
		for i, c := range h {
			hn[i] = float64(c) / total
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
func bhattacharyyaDist(p, q [256]float64) float64 {
	var bc float64
	for i := 0; i < 256; i++ {
		bc += math.Sqrt(p[i] * q[i])
	}
	if bc > 1 {
		bc = 1
	}
	return 1 - bc
}
