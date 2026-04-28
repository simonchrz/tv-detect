// Speaker fingerprinting — orthogonal signal to logo/NN/letterbox.
//
// Per-window (1 Hz typical) speaker-embedding-vs-show-centroid cosine
// similarity, remapped to [0, 1] where 1 = matches the show's voice
// profile. Pre-computed offline by scripts/compute-speaker-confs.py;
// tv-detect just reads the CSV and expands to per-frame.
//
// Useful for shows with persistent recurring speakers (soap operas, court
// shows, news magazines) where ad-block voice-overs are reliably distinct
// from the show's cast — works even when logo is present in promo badges
// (RTL Werbung) or absent in show graphics flickers (Pro7 lower-thirds).
package signals

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// SpeakerWindow is one fixed-size analysis window of speaker confidence.
// Times are recording-start-relative seconds; HasSpeech=false means the
// window was below the energy-VAD threshold so Conf is unreliable (silence
// or music) and downstream should treat as neutral (0.5) rather than as
// "different speaker" (0.0).
type SpeakerWindow struct {
	TimeS     float64
	Conf      float64 // 0..1; 1 = identical to show centroid, 0 = orthogonal
	HasSpeech bool
}

// LoadSpeakerCSV reads a CSV with header "time_s,speaker_conf,has_speech".
// Returns windows in time order. Empty/missing file is an error — callers
// that want to opt out should not pass --speaker-csv at all.
func LoadSpeakerCSV(path string) ([]SpeakerWindow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	var ws []SpeakerWindow
	first := true
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if first {
			first = false
			continue
		}
		if len(row) < 3 {
			return nil, fmt.Errorf("speaker csv: expected 3 cols, got %d: %v", len(row), row)
		}
		t, err := strconv.ParseFloat(row[0], 64)
		if err != nil {
			return nil, fmt.Errorf("speaker csv time: %w", err)
		}
		c, err := strconv.ParseFloat(row[1], 64)
		if err != nil {
			return nil, fmt.Errorf("speaker csv conf: %w", err)
		}
		ws = append(ws, SpeakerWindow{
			TimeS:     t,
			Conf:      c,
			HasSpeech: row[2] == "1" || row[2] == "true",
		})
	}
	return ws, nil
}

// ExpandSpeakerToFrames produces a per-frame conf array of length nFrames
// from window-rate measurements. Each frame at time t = i/fps gets the conf
// of the window covering it (nearest-window lookup; assumes uniform hop).
// Non-speech windows yield 0.5 (neutral — no speaker information). Frames
// outside the windowed range get 0.5 too.
func ExpandSpeakerToFrames(windows []SpeakerWindow, fps float64, nFrames int) []float64 {
	out := make([]float64, nFrames)
	for i := range out {
		out[i] = 0.5
	}
	if len(windows) == 0 || fps <= 0 {
		return out
	}
	hop := 1.0
	if len(windows) > 1 {
		hop = windows[1].TimeS - windows[0].TimeS
	}
	if hop <= 0 {
		return out
	}
	for i := 0; i < nFrames; i++ {
		t := float64(i) / fps
		idx := int(t / hop)
		if idx < 0 || idx >= len(windows) {
			continue
		}
		w := windows[idx]
		if w.HasSpeech {
			out[i] = w.Conf
		}
	}
	return out
}
