package decode

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Echte Datei, echtes ffmpeg: der Videostrom beginnt 10 s nach dem
// Container (Ton ab 0), jede Sekunde hat eine eigene Helligkeit (20*s).
// Ein Teilstueck ab Sekunde 5 muss die Helligkeit von Sekunde 5 liefern.
// Vor dem Fix sprang `-ss 5` ab Container-Beginn, landete vor dem Video
// und lieferte Sekunde 0 (dvr-kabel-eins-1779893460, Versatz 2516 s).
func TestTeilstueckSpringtAufVideoZeitachse(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("kein ffmpeg")
	}
	// Zwei Programme wie in der PID-Union-Zeit: Ton-Programm ab PTS ~1.4 s,
	// Video-Programm (eigene PIDs) ab ~11.4 s, hintereinandergehaengt.
	// (-itsoffset/setpts helfen nicht: ffmpeg gleicht die Starts beim Muxen an.)
	dir := t.TempDir()
	v, a, f := filepath.Join(dir, "v.ts"), filepath.Join(dir, "a.ts"), filepath.Join(dir, "versatz.ts")
	for _, c := range [][]string{
		{"-f", "lavfi", "-i", "nullsrc=s=32x18:r=25:d=20,geq=lum='20*floor(T)':cb=128:cr=128",
			"-c:v", "mpeg2video", "-q:v", "1", "-g", "5", "-output_ts_offset", "10",
			"-mpegts_start_pid", "0x200", "-mpegts_service_id", "2", v},
		{"-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=1", "-c:a", "mp2", a},
	} {
		if out, err := exec.Command("ffmpeg", append([]string{"-v", "error", "-y"}, c...)...).CombinedOutput(); err != nil {
			t.Fatalf("Testdatei: %v %s", err, out)
		}
	}
	ab, _ := os.ReadFile(a)
	vb, _ := os.ReadFile(v)
	if err := os.WriteFile(f, append(ab, vb...), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := Probe(f)
	if err != nil {
		t.Fatal(err)
	}
	if info.SeekOffsetS < 9 || info.SeekOffsetS > 11 {
		t.Fatalf("SeekOffsetS = %.2f, erwartet ~10", info.SeekOffsetS)
	}
	d, err := NewDecoder(context.Background(), DecodeOpts{Input: f, StartS: 5, DurS: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	var erstes []byte
	for fr := range d.Frames() {
		if erstes == nil {
			erstes = fr.Pixels
		}
	}
	if erstes == nil {
		t.Fatal("keine Bilder")
	}
	// rgb24 von Graustufe: Y=100 (limited range) ist grob 98..104 im Mittel
	sum := 0
	for _, b := range erstes {
		sum += int(b)
	}
	mittel := sum / len(erstes)
	if mittel < 80 || mittel > 120 {
		t.Fatalf("Helligkeit %d — erwartet ~100 (Sekunde 5), 0 hiesse: Sprung vor den Videobeginn", mittel)
	}
}
