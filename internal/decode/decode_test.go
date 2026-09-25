package decode

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Fake ffprobe/ffmpeg on PATH: the tests pin the Decoder's END-OF-STREAM
// semantics (exit status vs. pipe EOF), which do not depend on real
// video. A 4x2 frame is 24 bytes of rgb24.
const fakeFrameBytes = 4 * 2 * 3

// installFakeFFmpeg puts an ffprobe reporting a 4x2@25fps stream and an
// ffmpeg running ffmpegBody (sh) first on PATH.
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

func drain(d *Decoder) int {
	n := 0
	for range d.Frames() {
		n++
	}
	return n
}

// An ffmpeg that dies mid-chunk closes stdout exactly like one that
// finished. Before the fix the exit status was discarded and this chunk
// came back error-free with 2 of its frames — every later signal shifted.
func TestDecoderMeldetAbbruchVonFFmpeg(t *testing.T) {
	installFakeFFmpeg(t, "head -c 48 /dev/zero; echo 'Conversion failed!' >&2; exit 1")
	d, err := NewDecoder(context.Background(), DecodeOpts{Input: "x.ts"})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if n := drain(d); n != 2 {
		t.Fatalf("frames = %d, want 2", n)
	}
	err = d.Err()
	if err == nil {
		t.Fatal("Err() = nil nach ffmpeg exit 1 — abgebrochener Chunk gilt als vollständig")
	}
	if !strings.Contains(err.Error(), "Conversion failed!") {
		t.Errorf("Err() = %q, want ffmpeg's stderr tail in the message", err)
	}
}

// A truncated final frame with a clean exit stays a normal end (TS files
// cut mid-stream) — the fix must not turn that into a failure.
func TestDecoderAbgeschnittenerLetzterFrameIstNormal(t *testing.T) {
	installFakeFFmpeg(t, "head -c 60 /dev/zero; exit 0") // 2.5 frames
	d, err := NewDecoder(context.Background(), DecodeOpts{Input: "x.ts"})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if n := drain(d); n != 2 {
		t.Fatalf("frames = %d, want 2", n)
	}
	if err := d.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil for clean exit", err)
	}
}

// Our own cancel kills ffmpeg (non-zero exit) — that is expected and must
// not surface as a decode error; the caller sees its own ctx instead.
func TestDecoderEigenerAbbruchIstKeinFehler(t *testing.T) {
	installFakeFFmpeg(t, "head -c 24 /dev/zero; exec sleep 5")
	d, err := NewDecoder(context.Background(), DecodeOpts{Input: "x.ts"})
	if err != nil {
		t.Fatal(err)
	}
	<-d.Frames()
	d.Close()
	drain(d)
	if err := d.Err(); err != nil {
		t.Fatalf("Err() = %v after own Close(), want nil", err)
	}
}
