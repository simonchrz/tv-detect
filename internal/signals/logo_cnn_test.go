package signals

import "testing"

// A failed CNN inference must be neutral. It returned 0 = "logo absent",
// which votes AD for every frame the CNN could not score.
func TestLogoCNNFehlerIstNeutral(t *testing.T) {
	const w, h = 32, 32
	d := &LogoCNNDetector{
		session: fehlerSession{},
		frameW:  w, frameH: h, stride: w * 3,
		cropX: 0, cropY: 0, cropW: w, cropH: h,
		inputBuf: make([]float32, cnnInputChans*cnnInputSize*cnnInputSize),
	}
	for range 2 { // second call: already reported, same answer
		if got := d.Confidence(make([]byte, w*h*3)); got != 0.5 {
			t.Fatalf("Confidence after failed Run = %v, want neutral 0.5", got)
		}
	}
	var closed *LogoCNNDetector
	if got := closed.Confidence(nil); got != 0.5 {
		t.Errorf("nil detector = %v, want 0.5", got)
	}
}
