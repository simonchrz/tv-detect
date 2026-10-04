package blocks

import "testing"

// 10 min show, 4 min ad with a 50 s show-looking island in the middle,
// 10 min show. Default decoder splits the break; InnerShowMinS=90 keeps it whole.
func innerIslandSignal() []float64 {
	var p []float64
	add := func(n int, v float64) {
		for range n {
			p = append(p, v)
		}
	}
	add(600, 0.05)
	add(100, 0.99)
	add(50, 0.01)
	add(100, 0.99)
	add(600, 0.05)
	return p
}

func TestHSMMInnerShowMinBridgesIsland(t *testing.T) {
	p := innerIslandSignal()
	base := FormHSMM(p, HSMMOpts{DurW: 15})
	if len(base) != 2 {
		t.Fatalf("Ausgangslage: erwartet geteilter Block, bekam %v", base)
	}
	got := FormHSMM(p, HSMMOpts{DurW: 15, InnerShowMinS: 90})
	if len(got) != 1 || got[0].StartS != 600 || got[0].EndS != 850 {
		t.Fatalf("mit InnerShowMinS=90: erwartet [600,850), bekam %v", got)
	}
}

func TestHSMMInnerShowMinKeepsEdgesAndRealGaps(t *testing.T) {
	// Show am Anfang/Ende kuerzer als 90 s bleibt erlaubt; eine echte
	// Sendungsluecke von 5 min bleibt eine Luecke.
	var p []float64
	add := func(n int, v float64) {
		for range n {
			p = append(p, v)
		}
	}
	add(45, 0.05)
	add(200, 0.95)
	add(300, 0.05)
	add(200, 0.95)
	add(45, 0.05)
	a := FormHSMM(p, HSMMOpts{DurW: 15})
	b := FormHSMM(p, HSMMOpts{DurW: 15, InnerShowMinS: 90})
	if len(a) != len(b) {
		t.Fatalf("Raender/echte Luecke veraendert: %v vs %v", a, b)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("Raender/echte Luecke veraendert: %v vs %v", a, b)
		}
	}
}
