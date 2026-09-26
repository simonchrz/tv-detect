package signals

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
)

// SigLIPDim ist die Breite eines SigLIP-2-Bildmerkmals (base, NaFlex).
const SigLIPDim = 768

// SigLIPKomponenten: so viele Hauptkomponenten haengt ein MLP7-Kopf an,
// dazu EINE Spalte siglip_da (Entwurf docs/siglip-spur-design.md).
const SigLIPKomponenten = 64

// SigLIPSpur haelt die rohen Merkmale einer Aufnahme (scripts/siglip-spur.py):
// Zeile i = Sekunde i ab Dateianfang. Die Projektion auf die Kopf-Spalten
// steht IM Kopf (mlpSigMu/mlpSigV), die Spur bleibt kopfunabhaengig — wie
// die OCR-Spur rohe Treffer haelt.
type SigLIPSpur struct {
	N int
	E []float32 // N*SigLIPDim, zeilenweise
}

var npyShape = regexp.MustCompile(`'shape':\s*\((\d+),\s*(\d+)\)`)

// LadeSigLIPSpur liest <uuid>.npy (float16 oder float32, C-Reihenfolge,
// Form (n, 768)). Alles andere ist ein Fehler: eine falsch gelesene Spur
// saehe wie ein etwas schlechterer Kopf aus, nicht wie ein Absturz.
func LadeSigLIPSpur(pfad string) (*SigLIPSpur, error) {
	raw, err := os.ReadFile(pfad)
	if err != nil {
		return nil, fmt.Errorf("siglip-spur %s: %w", pfad, err)
	}
	if len(raw) < 10 || string(raw[1:6]) != "NUMPY" {
		return nil, fmt.Errorf("siglip-spur %s: kein .npy", pfad)
	}
	var hlen, off int
	switch raw[6] {
	case 1:
		hlen, off = int(binary.LittleEndian.Uint16(raw[8:10])), 10
	case 2, 3:
		if len(raw) < 12 {
			return nil, fmt.Errorf("siglip-spur %s: Kopf abgeschnitten", pfad)
		}
		hlen, off = int(binary.LittleEndian.Uint32(raw[8:12])), 12
	default:
		return nil, fmt.Errorf("siglip-spur %s: npy-Version %d", pfad, raw[6])
	}
	if off+hlen > len(raw) {
		return nil, fmt.Errorf("siglip-spur %s: Kopf abgeschnitten", pfad)
	}
	kopf := string(raw[off : off+hlen])
	daten := raw[off+hlen:]
	m := npyShape.FindStringSubmatch(kopf)
	if m == nil {
		return nil, fmt.Errorf("siglip-spur %s: Form nicht lesbar (%q)", pfad, kopf)
	}
	n, _ := strconv.Atoi(m[1])
	d, _ := strconv.Atoi(m[2])
	if d != SigLIPDim {
		return nil, fmt.Errorf("siglip-spur %s: Breite %d, will %d", pfad, d, SigLIPDim)
	}
	if regexp.MustCompile(`'fortran_order':\s*True`).MatchString(kopf) {
		return nil, fmt.Errorf("siglip-spur %s: Fortran-Reihenfolge", pfad)
	}
	s := &SigLIPSpur{N: n, E: make([]float32, n*d)}
	switch {
	case regexp.MustCompile(`'descr':\s*'<f2'`).MatchString(kopf):
		if len(daten) != n*d*2 {
			return nil, fmt.Errorf("siglip-spur %s: %d B Daten, will %d", pfad, len(daten), n*d*2)
		}
		for i := range s.E {
			s.E[i] = halbZuFloat(binary.LittleEndian.Uint16(daten[2*i:]))
		}
	case regexp.MustCompile(`'descr':\s*'<f4'`).MatchString(kopf):
		if len(daten) != n*d*4 {
			return nil, fmt.Errorf("siglip-spur %s: %d B Daten, will %d", pfad, len(daten), n*d*4)
		}
		for i := range s.E {
			s.E[i] = math.Float32frombits(binary.LittleEndian.Uint32(daten[4*i:]))
		}
	default:
		return nil, fmt.Errorf("siglip-spur %s: dtype nicht float16/float32 (%q)", pfad, kopf)
	}
	return s, nil
}

// halbZuFloat: IEEE-754 binary16 → float32 (inkl. Subnormale, Inf, NaN).
func halbZuFloat(h uint16) float32 {
	vz := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if man == 0 {
			return math.Float32frombits(vz)
		}
		f := float32(man) / 1024 * float32(math.Pow(2, -14))
		if vz != 0 {
			return -f
		}
		return f
	case 0x1f:
		return math.Float32frombits(vz | 0x7f800000 | man<<13)
	}
	return math.Float32frombits(vz | (exp+112)<<23 | man<<13)
}

// spalten schreibt die SigLIP-Spalten fuer Sekunde t nach x (Laenge
// SigLIPKomponenten+1): ((e - mu) @ V), dann siglip_da. Ohne Spur oder
// ausserhalb der Spur alles 0 — genau wie scripts/siglip_spalten.py.
func (s *SigLIPSpur) spalten(t int, mu, V []float32, x []float32) {
	for j := range x {
		x[j] = 0
	}
	if s == nil || t < 0 || t >= s.N {
		return
	}
	e := s.E[t*SigLIPDim : (t+1)*SigLIPDim]
	for i := 0; i < SigLIPDim; i++ {
		v := e[i] - mu[i]
		if v == 0 {
			continue
		}
		row := V[i*SigLIPKomponenten : (i+1)*SigLIPKomponenten]
		for j := 0; j < SigLIPKomponenten; j++ {
			x[j] += v * row[j]
		}
	}
	x[SigLIPKomponenten] = 1
}
