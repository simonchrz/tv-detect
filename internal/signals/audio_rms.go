package signals

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ExtractAudioRMSPerSecond mirrors scripts/train-head.py's
// extract_audio_rms_per_second so the per-frame audio feature seen
// by training matches what tv-detect produces at inference. Runs a
// single ffmpeg pass that pipes the recording's first audio stream
// through asetnsamples + astats + ametadata, prints one
// "lavfi.astats.Overall.RMS_level=<dB>" line per second to stderr.
//
// Output is normalised the SAME way as the Python side:
//
//	norm = clip((rms_dB + 60) / 60, 0, 1)
//
// so a -60 dB silent block → 0.0 and a 0 dB full-scale clip → 1.0.
// Returns (nSeconds,) values; on any failure returns a neutral 0.5
// array of length nSeconds (matches the Python fallback exactly so
// recordings with no audio stream silently skip the feature).
//
// nSeconds is the number of seconds we want output for — usually
// recording duration in seconds, ceiled. ffmpeg may emit fewer or
// more lines than expected (depends on stream length); we left-pad
// the result to nSeconds with neutral 0.5.
func ExtractAudioRMSPerSecond(ctx context.Context, src string, nSeconds int) []float32 {
	neutral := make([]float32, nSeconds)
	for i := range neutral {
		neutral[i] = 0.5
	}
	if nSeconds <= 0 {
		return neutral
	}
	const sampleRate = 48000
	args := []string{
		"-nostdin", "-nostats",
		"-i", src,
		"-map", "0:a:0",
		"-ac", "1", "-ar", strconv.Itoa(sampleRate),
		"-af", fmt.Sprintf(
			"asetnsamples=n=%d,"+
				"astats=metadata=1:reset=1,"+
				"ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
			sampleRate),
		"-f", "null", "-",
	}
	c, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(c, "ffmpeg", args...)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return neutral
	}
	if err := cmd.Start(); err != nil {
		return neutral
	}
	out := make([]float32, 0, nSeconds)
	sc := bufio.NewScanner(stderr)
	// astats writes one line per second-window; per-line scan keeps
	// memory bounded even for very long recordings (3-h movie ≈
	// 11k lines).
	const prefix = "lavfi.astats.Overall.RMS_level="
	for sc.Scan() {
		line := sc.Text()
		i := strings.Index(line, prefix)
		if i < 0 {
			continue
		}
		val := strings.TrimSpace(line[i+len(prefix):])
		// ffmpeg writes "-inf" for digital silence — clamp to -90 dB.
		var dB float64
		if val == "-inf" || val == "-Inf" || val == "-INF" {
			dB = -90.0
		} else {
			parsed, err := strconv.ParseFloat(val, 64)
			if err != nil {
				dB = -90.0
			} else if math.IsInf(parsed, 0) || math.IsNaN(parsed) {
				dB = -90.0
			} else {
				dB = parsed
			}
		}
		// Same normalisation as Python: -60 dB → 0, 0 dB → 1.
		n := (dB + 60.0) / 60.0
		if n < 0 {
			n = 0
		} else if n > 1 {
			n = 1
		}
		out = append(out, float32(n))
	}
	_ = cmd.Wait()
	if len(out) == 0 {
		return neutral
	}
	if len(out) >= nSeconds {
		return out[:nSeconds]
	}
	// ffmpeg emitted fewer entries than the requested duration —
	// pad with neutral so the index space is still right.
	padded := make([]float32, nSeconds)
	copy(padded, out)
	for i := len(out); i < nSeconds; i++ {
		padded[i] = 0.5
	}
	return padded
}
