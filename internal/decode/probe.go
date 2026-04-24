// Package decode wraps ffmpeg/ffprobe to deliver a stream of raw
// video frames + the metadata needed to interpret them.
package decode

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Info describes the first video stream of an input file.
type Info struct {
	Width      int
	Height     int
	FPS        float64
	DurationS  float64
	FrameCount int // 0 if ffprobe couldn't determine it (live, some MPEG-TS)
}

type probeOut struct {
	Streams []struct {
		CodecType string `json:"codec_type"`
		Width     int    `json:"width"`
		Height    int    `json:"height"`
		RFrameRate string `json:"r_frame_rate"`
		AvgFrameRate string `json:"avg_frame_rate"`
		NbFrames  string `json:"nb_frames"`
		Duration  string `json:"duration"`
	} `json:"streams"`
	Format struct {
		Duration string `json:"duration"`
	} `json:"format"`
}

// Probe runs ffprobe on input and returns metadata for the first video stream.
func Probe(input string) (Info, error) {
	cmd := exec.Command("ffprobe",
		"-v", "error",
		"-show_streams", "-show_format",
		"-of", "json",
		input)
	out, err := cmd.Output()
	if err != nil {
		return Info{}, fmt.Errorf("ffprobe: %w", err)
	}
	var p probeOut
	if err := json.Unmarshal(out, &p); err != nil {
		return Info{}, fmt.Errorf("parse ffprobe json: %w", err)
	}
	for _, s := range p.Streams {
		if s.CodecType != "video" {
			continue
		}
		fps := parseFPS(s.RFrameRate)
		if fps == 0 {
			fps = parseFPS(s.AvgFrameRate)
		}
		dur := parseFloat(s.Duration)
		if dur == 0 {
			dur = parseFloat(p.Format.Duration)
		}
		nbFrames, _ := strconv.Atoi(s.NbFrames)
		// MPEG-TS rarely carries nb_frames — derive from duration*fps.
		if nbFrames == 0 && dur > 0 && fps > 0 {
			nbFrames = int(dur*fps + 0.5)
		}
		return Info{
			Width:      s.Width,
			Height:     s.Height,
			FPS:        fps,
			DurationS:  dur,
			FrameCount: nbFrames,
		}, nil
	}
	return Info{}, fmt.Errorf("no video stream in %s", input)
}

// parseFPS handles ffprobe's "num/den" rational ("25/1", "30000/1001").
func parseFPS(s string) float64 {
	if s == "" || s == "0/0" {
		return 0
	}
	if i := strings.IndexByte(s, '/'); i >= 0 {
		num, err1 := strconv.ParseFloat(s[:i], 64)
		den, err2 := strconv.ParseFloat(s[i+1:], 64)
		if err1 == nil && err2 == nil && den != 0 {
			return num / den
		}
		return 0
	}
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func parseFloat(s string) float64 {
	v, _ := strconv.ParseFloat(s, 64)
	return v
}
