package pipeline

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
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
