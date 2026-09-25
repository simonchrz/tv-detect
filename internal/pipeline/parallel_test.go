package pipeline

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/signals"
)

// installFakeFFmpeg puts an ffprobe reporting a 4x2@25fps, 10 s stream
// and an ffmpeg running ffmpegBody (sh) first on PATH — enough to drive
// Run end-to-end without real video.
func installFakeFFmpeg(t *testing.T, ffmpegBody string) {
	t.Helper()
	dir := t.TempDir()
	probe := `#!/bin/sh
cat <<'JSON'
{"streams":[{"codec_type":"video","width":4,"height":2,"r_frame_rate":"25/1","avg_frame_rate":"25/1","duration":"10.0"}],
 "format":{"duration":"10.0","size":"100000000"}}
JSON
`
	if err := os.WriteFile(filepath.Join(dir, "ffprobe"), []byte(probe), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ffmpeg"),
		[]byte("#!/bin/sh\n"+ffmpegBody+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

// SIGTERM mid-decode cancels ctx; every chunk then ends early WITHOUT a
// decode error (own cancel). Before the fix Run returned that truncated
// result as a success, and main wrote cutlist + signals dump from it.
func TestRunNachAbbruchIstFehler(t *testing.T) {
	installFakeFFmpeg(t, "head -c 24 /dev/zero; exec sleep 5")
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(300 * time.Millisecond)
		cancel()
	}()
	res, err := Run(ctx, Opts{Input: "x.ts", Workers: 2})
	if err == nil {
		t.Fatalf("Run after cancel returned a result (%d frames) and no error",
			res.FrameCount)
	}
}

// Counterpart: an uncancelled run over the same fake source succeeds.
func TestRunOhneAbbruch(t *testing.T) {
	installFakeFFmpeg(t, "head -c 240 /dev/zero; exit 0") // 10 frames per chunk
	res, err := Run(context.Background(), Opts{Input: "x.ts", Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	if res.FrameCount == 0 {
		t.Fatal("no frames")
	}
}

func konst(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// Chunk 0 decoded 5 frames too few, chunk 1 six too many (inexact -ss).
// Plain appending shifted chunk 2 by +1 frame and chunk 1 by −5 against
// the absolute event axis; re-anchoring puts every chunk's first frame at
// round(startS*fps) and pads/cuts at the tail.
func TestMergeVerankertChunksNeu(t *testing.T) {
	const fps = 25.0
	mk := func(idx int, startS float64, n int, v float64) chunkRes {
		return chunkRes{index: idx, startS: startS, frameCount: n,
			logoConfs: konst(n, v), nnConfs: konst(n, v),
			bumperConfs: konst(n, v), bumperStartConfs: konst(n, v),
			boundaryConfs: konst(n, v)}
	}
	chunks := []chunkRes{mk(0, 0, 245, 0.1), mk(1, 10, 256, 0.2), mk(2, 20, 100, 0.3)}
	r := merge(chunks, decode.Info{FPS: fps}, true, true, true, true, true)

	if r.FrameCount != 600 {
		t.Errorf("FrameCount = %d, want 600 (= 500 anchored + 100 of the last chunk)", r.FrameCount)
	}
	arrays := map[string]struct {
		a    []float64
		fill float64
	}{
		"logo": {r.LogoConfs, fillLogo}, "nn": {r.NNConfs, fillNN},
		"bumper": {r.BumperConfs, fillNoMatch}, "bumperStart": {r.BumperStartConfs, fillNoMatch},
		"boundary": {r.BoundaryConfs, fillNoMatch},
	}
	for name, x := range arrays {
		if len(x.a) != r.FrameCount {
			t.Errorf("%s: len %d != FrameCount %d", name, len(x.a), r.FrameCount)
			continue
		}
		for _, c := range []struct {
			at   int
			want float64
		}{{244, 0.1}, {245, x.fill}, {249, x.fill}, {250, 0.2}, {499, 0.2}, {500, 0.3}, {599, 0.3}} {
			if x.a[c.at] != c.want {
				t.Errorf("%s[%d] = %v, want %v", name, c.at, x.a[c.at], c.want)
			}
		}
	}
}

// Scene cuts and letterbox transitions carry GLOBAL frame numbers, and
// the events right after a chunk start are real (see the next test) —
// the old merge kept chunk-local frames and dropped them.
func TestMergeEreignisseGlobalUndVollstaendig(t *testing.T) {
	const fps = 25.0
	chunks := []chunkRes{
		{index: 0, startS: 0, frameCount: 250,
			sceneCuts: []signals.SceneCut{{Frame: 100, TimeS: 4}}},
		{index: 1, startS: 10, frameCount: 256,
			sceneCuts: []signals.SceneCut{{Frame: 1, TimeS: 0.04}, {Frame: 252, TimeS: 10.08}},
			letterbox: []signals.LetterboxEvent{{Frame: 3, TimeS: 0.12, Onset: true}}},
		{index: 2, startS: 20, frameCount: 50},
	}
	r := merge(chunks, decode.Info{FPS: fps}, false, false, false, false, false)
	var got []int
	for _, sc := range r.SceneCuts {
		got = append(got, sc.Frame)
	}
	// 252 lies in chunk 1's cut-off tail (it owns 250 frames) → dropped with them.
	if want := []int{100, 251}; !slices.Equal(got, want) {
		t.Errorf("scene-cut frames = %v, want %v", got, want)
	}
	if len(r.Letterbox) != 1 || r.Letterbox[0].Frame != 253 {
		t.Errorf("letterbox = %+v, want one event at global frame 253", r.Letterbox)
	}
}

// Why the chunk-start filters were wrong: neither detector emits anything
// just because a chunk starts. Its first frame only sets state.
func TestDetektorenMeldenAmChunkStartNichts(t *testing.T) {
	const w, h, fps = 64, 96, 25.0
	lb := make([]byte, w*h*3) // all black = letterbox bars top+bottom
	for i := w * 40 * 3; i < w*56*3; i++ {
		lb[i] = 200 // bright picture in the middle
	}
	letter := signals.NewLetterboxDetector(fps, w, h, 0, 0)
	scene := signals.NewSceneDetector(fps, 0)
	for i := range 30 {
		letter.Push(i, lb)
		scene.Push(i, lb)
	}
	if ev := letter.Events(); len(ev) != 0 {
		t.Errorf("letterbox emitted %+v for a chunk that starts letterboxed", ev)
	}
	if c := scene.Cuts(); len(c) != 0 {
		t.Errorf("scene detector emitted %+v without a previous frame", c)
	}
	// Control: the frames really read as letterboxed — a full picture
	// afterwards is an offset.
	full := make([]byte, w*h*3)
	for i := range full {
		full[i] = 200
	}
	for i := 30; i < 60; i++ {
		letter.Push(i, full)
	}
	if ev := letter.Events(); len(ev) != 1 || ev[0].Onset {
		t.Errorf("control: letterbox events %+v, want exactly one offset", ev)
	}
}
