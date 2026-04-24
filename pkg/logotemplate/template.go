// Package logotemplate parses comskip's .logo.txt files into an
// edge-mask grid the signal extractor can match per-frame.
//
// File format (after the SaveLogoMaskData patch in mpeg2dec.c):
//
//	logoMinX=117
//	logoMaxX=159
//	logoMinY=39
//	logoMaxY=73
//	picWidth=736
//	picHeight=576
//
//	Combined Logo Mask
//	<single binary marker byte> <newline>
//	<row 0 of mask, picWidth chars or width of bbox + padding>
//	<row 1 ...>
//	...
//
// Each mask row is one Y position in the logo bounding box. Within a
// row, every non-space character marks "an edge is expected here";
// space marks "no edge expected". The exact glyph (+,-,|) tells comskip
// the edge orientation, but for confidence scoring we only care
// whether the position expects an edge.
package logotemplate

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Template is a parsed comskip .logo.txt with the mask flattened to
// a per-pixel "edge expected" boolean grid.
type Template struct {
	MinX, MaxX     int  // logo bounding-box columns in source pixel coords
	MinY, MaxY     int  // logo bounding-box rows
	PicWidth       int  // source frame width when the template was trained
	PicHeight      int  // source frame height when the template was trained
	Mask           [][]bool // [row 0..H-1][col 0..W-1], row-major
	EdgePositions  int  // number of true cells in Mask
}

// Width returns the bounding-box width in pixels.
func (t *Template) Width() int { return t.MaxX - t.MinX }

// Height returns the bounding-box height in pixels.
func (t *Template) Height() int { return t.MaxY - t.MinY }

// Load parses path and returns the template. Returns a wrapped error
// if the file is malformed or the mask is empty.
func Load(path string) (*Template, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open logo template: %w", err)
	}
	defer f.Close()

	t := &Template{}
	br := bufio.NewReader(f)

	headerWanted := map[string]*int{
		"logoMinX":  &t.MinX,
		"logoMaxX":  &t.MaxX,
		"logoMinY":  &t.MinY,
		"logoMaxY":  &t.MaxY,
		"picWidth":  &t.PicWidth,
		"picHeight": &t.PicHeight,
	}
	headerSeen := 0
	// Read the leading "key=value" lines, then the "Combined Logo Mask"
	// marker, then the binary byte after it. Anything else before the
	// mask body is ignored.
	for headerSeen < len(headerWanted) {
		line, err := br.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("logo header truncated: %w", err)
		}
		line = strings.TrimRight(line, "\r\n")
		if eq := strings.IndexByte(line, '='); eq > 0 {
			key, val := line[:eq], line[eq+1:]
			if dst, ok := headerWanted[key]; ok {
				v, err := strconv.Atoi(val)
				if err != nil {
					return nil, fmt.Errorf("logo header %q: %w", key, err)
				}
				*dst = v
				headerSeen++
			}
		}
	}
	// Skip until "Combined Logo Mask" line.
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("logo mask marker missing: %w", err)
		}
		if strings.HasPrefix(line, "Combined Logo Mask") {
			break
		}
	}
	// Discard the single binary byte + its trailing newline. Comskip
	// writes one non-printable byte (count or version marker) on its
	// own line right after the marker.
	if _, err := br.ReadString('\n'); err != nil {
		return nil, fmt.Errorf("logo binary marker line: %w", err)
	}

	w, h := t.Width(), t.Height()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("logo bbox invalid: w=%d h=%d", w, h)
	}
	t.Mask = make([][]bool, h)
	for r := 0; r < h; r++ {
		line, err := br.ReadString('\n')
		if err != nil && line == "" {
			return nil, fmt.Errorf("logo mask truncated at row %d/%d: %w", r, h, err)
		}
		row := make([]bool, w)
		// Trim newline, leave embedded spaces.
		line = strings.TrimRight(line, "\r\n")
		for c := 0; c < w && c < len(line); c++ {
			if line[c] != ' ' {
				row[c] = true
				t.EdgePositions++
			}
		}
		t.Mask[r] = row
	}
	if t.EdgePositions == 0 {
		return nil, fmt.Errorf("logo template empty (no edge positions)")
	}
	return t, nil
}
