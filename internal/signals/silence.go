package signals

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"strconv"
)

// SilenceEvent is a contiguous run of audio silence as detected by
// ffmpeg's silencedetect filter. Times are in seconds from input start.
type SilenceEvent struct {
	StartS    float64
	EndS      float64
	DurationS float64
}

// SilenceOpts configures the silencedetect ffmpeg subprocess.
type SilenceOpts struct {
	Input   string
	NoiseDB float64 // noise floor (negative dB; e.g. -30)
	MinDurS float64 // min silence duration to emit (e.g. 0.5)
}

// DetectSilence runs a dedicated ffmpeg subprocess that reads only the
// audio stream and pipes silencedetect events to stderr. We parse those
// lines and return an aggregated event list.
//
// This intentionally lives outside the video Decoder pipeline: the
// audio analyser and the video decoder share the same input file but
// run in parallel processes, fully exploiting cores. The orchestrator
// kicks both off as goroutines and joins.
var (
	silenceStartRE = regexp.MustCompile(`silence_start:\s*(-?\d+(?:\.\d+)?)`)
	silenceEndRE   = regexp.MustCompile(`silence_end:\s*(-?\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(-?\d+(?:\.\d+)?)`)
)

func DetectSilence(ctx context.Context, opts SilenceOpts) ([]SilenceEvent, error) {
	if opts.NoiseDB == 0 {
		opts.NoiseDB = -30
	}
	if opts.MinDurS == 0 {
		opts.MinDurS = 0.5
	}
	filter := fmt.Sprintf("silencedetect=noise=%.1fdB:d=%.3f", opts.NoiseDB, opts.MinDurS)
	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-hide_banner", "-nostdin",
		"-i", opts.Input,
		"-vn",                 // discard video — we only want the audio analyser
		"-af", filter,
		"-f", "null", "-")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start ffmpeg silencedetect: %w", err)
	}
	events, parseErr := parseSilenceStream(stderr)
	waitErr := cmd.Wait()
	if parseErr != nil {
		return nil, parseErr
	}
	if waitErr != nil {
		return events, fmt.Errorf("ffmpeg silencedetect: %w", waitErr)
	}
	return events, nil
}

// parseSilenceStream reads lines until EOF and returns the aggregated
// events. Exposed for testability — pass any reader carrying the
// canonical "silence_start: X" / "silence_end: Y | silence_duration: Z"
// format ffmpeg emits.
func parseSilenceStream(r io.Reader) ([]SilenceEvent, error) {
	var events []SilenceEvent
	var pending SilenceEvent
	pendingOpen := false
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if m := silenceStartRE.FindStringSubmatch(line); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			pending.StartS = v
			pendingOpen = true
			continue
		}
		if m := silenceEndRE.FindStringSubmatch(line); m != nil && pendingOpen {
			pending.EndS, _ = strconv.ParseFloat(m[1], 64)
			pending.DurationS, _ = strconv.ParseFloat(m[2], 64)
			events = append(events, pending)
			pending = SilenceEvent{}
			pendingOpen = false
		}
	}
	if err := scanner.Err(); err != nil {
		return events, err
	}
	return events, nil
}
