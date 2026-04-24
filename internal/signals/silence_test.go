package signals

import (
	"strings"
	"testing"
)

func TestParseSilenceStream(t *testing.T) {
	const sample = `frame=    1 fps=0.0 q=-0.0 size=N/A time=00:00:00.04 bitrate=N/A speed=42x
[silencedetect @ 0xff00] silence_start: 12.345
frame=  100 fps=0.0 q=-0.0 size=N/A time=00:00:04.00 bitrate=N/A speed=42x
[silencedetect @ 0xff00] silence_end: 13.567 | silence_duration: 1.222
[silencedetect @ 0xff00] silence_start: 100.5
[silencedetect @ 0xff00] silence_end: 102.0 | silence_duration: 1.5
[silencedetect @ 0xff00] silence_start: 200
[silencedetect @ 0xff00] silence_end: 200.7 | silence_duration: 0.7
`
	events, err := parseSilenceStream(strings.NewReader(sample))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 3 {
		t.Fatalf("want 3 events, got %d: %+v", len(events), events)
	}
	want := []SilenceEvent{
		{12.345, 13.567, 1.222},
		{100.5, 102.0, 1.5},
		{200, 200.7, 0.7},
	}
	for i, w := range want {
		if events[i] != w {
			t.Errorf("event %d: want %+v, got %+v", i, w, events[i])
		}
	}
}

func TestParseSilenceUnclosedStart(t *testing.T) {
	// silence_start without matching silence_end — ffmpeg can do this if
	// the file ends mid-silence. Should be ignored, not produce a half event.
	const sample = `[silencedetect @ 0xff00] silence_start: 50.0
[silencedetect @ 0xff00] silence_end: 60.0 | silence_duration: 10.0
[silencedetect @ 0xff00] silence_start: 90.0
`
	events, err := parseSilenceStream(strings.NewReader(sample))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].StartS != 50.0 {
		t.Fatalf("want 1 closed event, got %+v", events)
	}
}
