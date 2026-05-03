package signals

import (
	"encoding/json"
	"fmt"
	"os"
)

// whisperJSON is the on-disk shape of ~/.cache/tv-whisper/<uuid>.whisper.json,
// produced by tv-whisper-classify (Mac side). Only the fields the per-second
// resampler needs are decoded; the bulk transcription text is ignored at
// load time.
type whisperJSON struct {
	DurationS float64           `json:"duration_s"`
	WindowS   int               `json:"window_s"`
	StrideS   int               `json:"stride_s"`
	Windows   []whisperWindow   `json:"windows"`
}

type whisperWindow struct {
	T    float64 `json:"t"`
	Prob float64 `json:"prob"`
}

// LoadWhisperPerSecond reads a whisper.json file and returns a per-second
// ad-prob array. Each second is averaged across the windows that contain
// it (windows are 60 s long at 30 s stride → 2× overlap typical). Length
// = ceil(duration_s) so the array can be indexed by frame index at 1 fps
// without bounds checks; per-frame consumers cap their access at the
// returned length anyway.
//
// Returns (nil, err) if the file is missing, malformed, or has zero
// windows. Caller should treat err as informational — the NN forward
// pass falls back to neutral 0.5 per frame when whisper data is absent.
func LoadWhisperPerSecond(path string) ([]float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("whisper read: %w", err)
	}
	var d whisperJSON
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, fmt.Errorf("whisper parse: %w", err)
	}
	if len(d.Windows) == 0 {
		return nil, fmt.Errorf("whisper file has no windows")
	}
	winS := d.WindowS
	if winS <= 0 {
		winS = 60 // matches the train-head + tv-whisper-classify default
	}
	// Determine output length. Prefer the explicit duration_s; fall back to
	// the latest window-end if duration is missing or zero.
	n := int(d.DurationS + 0.5)
	if n <= 0 {
		for _, w := range d.Windows {
			end := int(w.T) + winS
			if end > n {
				n = end
			}
		}
	}
	if n <= 0 {
		return nil, fmt.Errorf("whisper file has no usable duration")
	}
	sums := make([]float64, n)
	counts := make([]int, n)
	for _, w := range d.Windows {
		t0 := int(w.T)
		if t0 < 0 {
			t0 = 0
		}
		t1 := t0 + winS
		if t1 > n {
			t1 = n
		}
		for s := t0; s < t1; s++ {
			sums[s] += w.Prob
			counts[s]++
		}
	}
	out := make([]float64, n)
	for s := 0; s < n; s++ {
		if counts[s] > 0 {
			out[s] = sums[s] / float64(counts[s])
		} else {
			out[s] = 0.5 // gap → neutral
		}
	}
	return out, nil
}
