package signals

import (
	"os"
	"testing"
)

// TestLoadWhisperPerSecond_RealFile is opportunistic: skips if no
// real whisper.json is in the local Mac cache. When present, it
// validates that the loader produces a per-second array of plausible
// length + all values in [0,1].
func TestLoadWhisperPerSecond_RealFile(t *testing.T) {
	dir, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	path := dir + "/.cache/tv-whisper/00c30a668a62722a7b7de8dc2cda849d.whisper.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("real whisper.json not present at %s — skipping", path)
	}
	probs, err := LoadWhisperPerSecond(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(probs) < 60 {
		t.Errorf("got %d per-second probs, want at least 60 (= 1 minute "+
			"of recording)", len(probs))
	}
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Fatalf("probs[%d]=%g out of [0,1]", i, p)
		}
	}
	t.Logf("loaded %d per-second probs from real whisper.json", len(probs))
}

// TestLoadWhisperPerSecond_Synthetic builds a deterministic 3-window
// whisper.json in a temp dir + verifies the per-second averaging.
func TestLoadWhisperPerSecond_Synthetic(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/test.whisper.json"
	body := `{
		"duration_s": 90,
		"window_s": 60,
		"stride_s": 30,
		"windows": [
			{"t": 0,  "prob": 0.2},
			{"t": 30, "prob": 0.8},
			{"t": 60, "prob": 0.5}
		]
	}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	probs, err := LoadWhisperPerSecond(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(probs) != 90 {
		t.Errorf("len(probs)=%d, want 90", len(probs))
	}
	// Coverage windows:
	//   sec 0..29   : only window 0 (prob 0.2)
	//   sec 30..59  : windows 0 + 1 → mean 0.5
	//   sec 60..89  : windows 1 + 2 → mean 0.65
	// Allow 1e-6 fp slack.
	check := func(s int, want float64) {
		if probs[s] < want-1e-6 || probs[s] > want+1e-6 {
			t.Errorf("probs[%d]=%g, want %g", s, probs[s], want)
		}
	}
	check(0, 0.2)
	check(29, 0.2)
	check(30, 0.5)
	check(59, 0.5)
	check(60, 0.65)
	check(89, 0.65)
}
