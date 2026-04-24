package logotemplate

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadVOX(t *testing.T) {
	// Use the actual cached vox template if available, otherwise skip.
	path := filepath.Join(os.Getenv("HOME"), "mnt/pi-tv/hls/.logos/vox.logo.txt")
	if _, err := os.Stat(path); err != nil {
		t.Skip("no vox.logo.txt cached")
	}
	tmpl, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if tmpl.MinX != 117 || tmpl.MaxX != 159 {
		t.Errorf("X bbox: want 117-159, got %d-%d", tmpl.MinX, tmpl.MaxX)
	}
	if tmpl.MinY != 39 || tmpl.MaxY != 73 {
		t.Errorf("Y bbox: want 39-73, got %d-%d", tmpl.MinY, tmpl.MaxY)
	}
	if tmpl.PicWidth != 736 || tmpl.PicHeight != 576 {
		t.Errorf("pic size: want 736x576, got %dx%d", tmpl.PicWidth, tmpl.PicHeight)
	}
	if len(tmpl.Mask) != tmpl.Height() {
		t.Errorf("mask rows: want %d, got %d", tmpl.Height(), len(tmpl.Mask))
	}
	if tmpl.EdgePositions == 0 {
		t.Error("expected non-zero edge positions in vox template")
	}
	t.Logf("vox template: %dx%d bbox, %d edge positions", tmpl.Width(), tmpl.Height(), tmpl.EdgePositions)
}
