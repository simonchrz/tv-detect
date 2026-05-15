package signals

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// LogoCNNDetector replaces the edge-template logo signal with a tiny
// per-channel CNN trained on user-labeled show vs ad frames cropped
// to the logo region.
//
// Why: edge-template matching computes "fraction of template-edge
// positions that have a Sobel edge in the frame ROI" — busy ad
// content (modern broadcasters with quick cuts, product overlays,
// dynamic backgrounds) trivially satisfies every template-edge
// position regardless of whether the actual logo is present, so
// VOX/Nick/RTL recordings score 0.95+ false-positive logo conf
// during ads. The 2026-05-15 investigation showed neither IoU
// metric nor higher edge-thresholds fix this — Sobel-edge signals
// don't separate cleanly for these channels' content.
//
// CNN approach: 30K-param 3-conv net trained on per-channel labeled
// crops (see scripts/extract_logo_dataset.py + train_logo_cnn.py).
// Inference cost ~1-2 ms per frame on M-series Mac CoreML.
// Validation accuracy 87-100% per channel vs ~50% random for the
// edge-template on hard channels.
//
// Per-frame flow:
//   1. Crop frame to (bbox + margin) region — same crop as training
//   2. Bilinear-resize to 64x64
//   3. Convert RGB-uint8 → CHW float32, normalise (x-0.5)/0.25
//   4. ONNX forward → 1 logit
//   5. Sigmoid → probability of "logo present" in [0, 1]
type LogoCNNDetector struct {
	session   *ort.AdvancedSession
	inTensor  *ort.Tensor[float32]
	outTensor *ort.Tensor[float32]
	frameW    int
	frameH    int
	stride    int
	// Crop region = bbox + margin, clamped to frame. Pre-computed at
	// construction so per-frame Confidence() doesn't repeat clamp/clip.
	cropX, cropY, cropW, cropH int
	// Pre-allocated buffer for the resized 64x64x3 RGB float32 input,
	// avoiding per-frame allocation. Layout: CHW (= ONNX-standard for
	// vision nets). [3*64*64].
	inputBuf []float32
}

const (
	cnnInputSize  = 64
	cnnInputChans = 3
)

// NewLogoCNNDetector loads a per-channel CNN model and configures the
// crop/resize parameters. bboxMin*/bboxMax* match the .logo.txt-style
// pixel coordinates used by the edge-template detector; the CNN
// crops the same region + marginPx padding (= same as
// extract_logo_dataset.py LOGO_MARGIN_PX). Returns nil if onnxPath
// can't be loaded — caller falls back to template detector.
func NewLogoCNNDetector(onnxPath string, frameW, frameH int,
	bboxMinX, bboxMinY, bboxMaxX, bboxMaxY, marginPx int) (*LogoCNNDetector, error) {

	if err := initOrtRuntime(); err != nil {
		return nil, fmt.Errorf("logo-cnn: ORT init: %w", err)
	}
	x := bboxMinX - marginPx
	y := bboxMinY - marginPx
	x2 := bboxMaxX + marginPx
	y2 := bboxMaxY + marginPx
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x2 > frameW {
		x2 = frameW
	}
	if y2 > frameH {
		y2 = frameH
	}
	if x2-x < 4 || y2-y < 4 {
		return nil, fmt.Errorf("logo-cnn: bbox+margin clamped to %dx%d, too small",
			x2-x, y2-y)
	}

	inShape := ort.NewShape(1, cnnInputChans, cnnInputSize, cnnInputSize)
	outShape := ort.NewShape(1)
	in, err := ort.NewEmptyTensor[float32](inShape)
	if err != nil {
		return nil, fmt.Errorf("logo-cnn: input tensor: %w", err)
	}
	out, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		in.Destroy()
		return nil, fmt.Errorf("logo-cnn: output tensor: %w", err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		in.Destroy(); out.Destroy()
		return nil, fmt.Errorf("logo-cnn: session opts: %w", err)
	}
	defer opts.Destroy()
	sess, err := ort.NewAdvancedSession(onnxPath,
		[]string{"frame"}, []string{"logit"},
		[]ort.Value{in}, []ort.Value{out}, opts)
	if err != nil {
		in.Destroy(); out.Destroy()
		return nil, fmt.Errorf("logo-cnn: load %s: %w", onnxPath, err)
	}
	return &LogoCNNDetector{
		session:   sess,
		inTensor:  in,
		outTensor: out,
		frameW:    frameW,
		frameH:    frameH,
		stride:    frameW * 3,
		cropX:     x, cropY: y, cropW: x2 - x, cropH: y2 - y,
		inputBuf: in.GetData(),
	}, nil
}

// Close releases ORT resources. Safe to call multiple times.
func (d *LogoCNNDetector) Close() {
	if d == nil {
		return
	}
	if d.session != nil {
		d.session.Destroy()
		d.session = nil
	}
	if d.inTensor != nil {
		d.inTensor.Destroy()
		d.inTensor = nil
	}
	if d.outTensor != nil {
		d.outTensor.Destroy()
		d.outTensor = nil
	}
}

// Confidence returns P(logo present) ∈ [0, 1] for this frame.
// pixels is row-major rgb24, length 3*frameW*frameH.
func (d *LogoCNNDetector) Confidence(pixels []byte) float64 {
	if d == nil || d.session == nil {
		return 0
	}
	d.fillInput(pixels)
	if err := d.session.Run(); err != nil {
		return 0
	}
	logit := float64(d.outTensor.GetData()[0])
	return 1.0 / (1.0 + math.Exp(-logit))
}

// fillInput crops the frame to (cropX, cropY, cropW, cropH), bilinear-
// resizes to 64x64, converts RGB-uint8 to CHW float32, and normalises
// (x/255 - 0.5) / 0.25 — matching the training preprocess in
// train_logo_cnn.py LogoDataset.transform.
func (d *LogoCNNDetector) fillInput(pixels []byte) {
	const N = cnnInputSize
	// Bilinear sampling in source coords for each output pixel.
	// scaleX/Y map output [0..N-1] back to source [0..cropW/H-1].
	scaleX := float32(d.cropW-1) / float32(N-1)
	scaleY := float32(d.cropH-1) / float32(N-1)
	// CHW layout: channel c starts at c*N*N
	const planeSz = N * N
	for oy := 0; oy < N; oy++ {
		sy := float32(oy) * scaleY
		y0 := int(sy)
		y1 := y0 + 1
		if y1 >= d.cropH {
			y1 = d.cropH - 1
		}
		fy := sy - float32(y0)
		for ox := 0; ox < N; ox++ {
			sx := float32(ox) * scaleX
			x0 := int(sx)
			x1 := x0 + 1
			if x1 >= d.cropW {
				x1 = d.cropW - 1
			}
			fx := sx - float32(x0)
			// Compute byte offsets in the source frame. Crop origin
			// (cropX, cropY) shifts everything.
			p00 := ((d.cropY+y0)*d.frameW + (d.cropX + x0)) * 3
			p10 := ((d.cropY+y0)*d.frameW + (d.cropX + x1)) * 3
			p01 := ((d.cropY+y1)*d.frameW + (d.cropX + x0)) * 3
			p11 := ((d.cropY+y1)*d.frameW + (d.cropX + x1)) * 3
			outIdx := oy*N + ox
			for c := 0; c < cnnInputChans; c++ {
				v00 := float32(pixels[p00+c])
				v10 := float32(pixels[p10+c])
				v01 := float32(pixels[p01+c])
				v11 := float32(pixels[p11+c])
				top := v00*(1-fx) + v10*fx
				bot := v01*(1-fx) + v11*fx
				v := top*(1-fy) + bot*fy
				// (x/255 - 0.5) / 0.25 = (x/255 - 0.5) * 4
				d.inputBuf[c*planeSz+outIdx] = (v/255.0 - 0.5) * 4.0
			}
		}
	}
}

