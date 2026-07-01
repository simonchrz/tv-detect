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
		Size     string `json:"size"`
	} `json:"format"`
}

// minPlausibleVideoKbps is the size-implied bitrate below which a
// container's reported duration is treated as inflated. Real SD/HD
// broadcast video sustains well above this; a wrapped/discontinuous
// MPEG-TS PTS makes ffprobe report a duration many times the real one
// (seen ~16x on rtl/vox), dragging the implied bitrate to ~170 kbit/s.
const minPlausibleVideoKbps = 500

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
		// Prefer avg_frame_rate over r_frame_rate. For interlaced sources
		// (= 50i SD broadcast: r=50/1, avg=25/1), r is the FIELD rate while
		// the decoder yields PROGRESSIVE frames at avg rate — using r
		// halves every downstream timestamp (= back-half-NaN in
		// extract_logo's CSV, 2026-05-23 incident). For progressive
		// sources both fields are equal, so this is a no-op.
		fps := parseFPS(s.AvgFrameRate)
		if fps == 0 {
			fps = parseFPS(s.RFrameRate)
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
		// Guard against an inflated container duration (MPEG-TS PTS wrap /
		// discontinuity). A wrapped PTS makes the reported duration many
		// times the real one, which balloons the chunk plan + decode into
		// ~1M phantom (CFR-padded) frames and blows the detect timeout. If
		// the size-implied bitrate is implausibly low for real video,
		// trust an authoritative packet-count over the duration.
		if fps > 0 && dur > 0 {
			if size := parseFloat(p.Format.Size); size > 0 &&
				size*8/dur/1000 < minPlausibleVideoKbps {
				if real := countVideoPackets(input); real > 0 {
					nbFrames = real
					dur = float64(real) / fps
				}
			}
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

// countVideoPackets returns the true number of demuxed video packets
// (≈ coded frames) by reading the whole file. Only called as a fallback
// when the container duration looks inflated, since it costs a full read.
// Returns 0 on any error so the caller keeps the duration-derived estimate.
func countVideoPackets(input string) int {
	// Corrupt inputs (the exact case this fires on) make ffprobe exit
	// non-zero while still printing the count to stdout, so parse stdout
	// regardless of exit status rather than bailing on the error.
	out, _ := exec.Command("ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-count_packets",
		"-show_entries", "stream=nb_read_packets",
		"-of", "csv=p=0",
		input).Output()
	// ffprobe can print the count more than once (e.g. duplicate stream
	// entries), so take the first integer token rather than the whole
	// trimmed string, which strconv can't parse across newlines.
	fields := strings.Fields(string(out))
	if len(fields) == 0 {
		return 0
	}
	n, _ := strconv.Atoi(fields[0])
	return n
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
