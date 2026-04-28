package signals

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSpeakerCSV(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "spk.csv")
	if err := os.WriteFile(p, []byte("time_s,speaker_conf,has_speech\n"+
		"0.000,0.825,1\n"+
		"1.000,0.412,0\n"+
		"2.000,0.901,1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := LoadSpeakerCSV(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(ws) != 3 {
		t.Fatalf("want 3 windows, got %d", len(ws))
	}
	if ws[1].HasSpeech || !ws[0].HasSpeech {
		t.Errorf("speech flags wrong: %+v", ws)
	}
	if ws[2].Conf != 0.901 {
		t.Errorf("conf parse wrong: %f", ws[2].Conf)
	}
}

func TestExpandSpeakerToFrames(t *testing.T) {
	ws := []SpeakerWindow{
		{TimeS: 0, Conf: 0.8, HasSpeech: true},
		{TimeS: 1, Conf: 0.3, HasSpeech: false}, // silence → 0.5
		{TimeS: 2, Conf: 0.9, HasSpeech: true},
	}
	out := ExpandSpeakerToFrames(ws, 25, 75) // 3 sec @ 25fps
	if len(out) != 75 {
		t.Fatalf("len wrong: %d", len(out))
	}
	if out[0] != 0.8 || out[24] != 0.8 {
		t.Errorf("first window mismapped: %v %v", out[0], out[24])
	}
	if out[25] != 0.5 || out[49] != 0.5 {
		t.Errorf("silence window not neutral: %v %v", out[25], out[49])
	}
	if out[50] != 0.9 || out[74] != 0.9 {
		t.Errorf("third window mismapped: %v %v", out[50], out[74])
	}
}

func TestExpandSpeakerOutOfRangeNeutral(t *testing.T) {
	ws := []SpeakerWindow{{TimeS: 0, Conf: 0.8, HasSpeech: true}}
	out := ExpandSpeakerToFrames(ws, 25, 100) // 4s but only 1 window
	// First 25 frames covered; rest get neutral 0.5
	if out[0] != 0.8 {
		t.Errorf("first frame wrong: %f", out[0])
	}
	if out[50] != 0.5 {
		t.Errorf("out-of-range frame should be neutral, got %f", out[50])
	}
}
