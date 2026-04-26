package signals

import (
	"bufio"
	"context"
	"os/exec"
	"strconv"
	"strings"
)

// IFrameTimes returns the ascending list of I-frame timestamps (PTS,
// in seconds) for the video stream of input. Used by the block
// formation step to snap refined boundaries to actual encoder cuts —
// commercial inserts almost always align with I-frames, so this
// gives sub-second boundary precision regardless of how coarse the
// rough boundary was.
//
// Implementation: ffprobe with `-skip_frame nokey` only emits
// keyframes (I-frames in our MPEG-TS / h264 case). Cheap — ~1-2 s
// for an hour-long recording, no full decode.
//
// Returns empty slice on any error or empty stream; the caller
// treats that as "I-frame snap unavailable" and falls through.
func IFrameTimes(ctx context.Context, input string) ([]float64, error) {
	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-skip_frame", "nokey",
		"-show_entries", "frame=pts_time",
		"-of", "csv=p=0", input)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	var out []float64
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			continue
		}
		out = append(out, v)
	}
	_ = cmd.Wait()
	// ffprobe usually emits PTS in source order which is also
	// chronological for I-frames; sort defensively just in case.
	for i := 1; i < len(out); i++ {
		if out[i] < out[i-1] {
			// rare; do an insertion-sort pass
			j := i
			for j > 0 && out[j] < out[j-1] {
				out[j], out[j-1] = out[j-1], out[j]
				j--
			}
		}
	}
	return out, nil
}
