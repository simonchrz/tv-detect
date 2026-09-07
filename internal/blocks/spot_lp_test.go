package blocks

import (
	"testing"

	"github.com/simonchrz/tv-detect/internal/signals"
)

// Die Index-Konvention ist der einzige Ort, an dem man hier still um eine
// Sekunde daneben liegen kann -- und eine Sekunde ist bei einem Snap, der
// auf den Frame genau sein soll, der ganze Gewinn.
func TestSpotBoundaryLPIndexKonvention(t *testing.T) {
	anchors := []signals.SpotAnchor{{StartS: 843.39, EndS: 863.39, FamilyID: 1, FamilySize: 5}}
	s, e := SpotBoundaryLP(anchors, 2.0, 1000)
	if len(s) != 1001 || len(e) != 1001 {
		t.Fatalf("Laenge %d/%d, erwartet nSec+1 = 1001", len(s), len(e))
	}
	// Start 843.39 -> Grenze VOR Sekunde 843 -> Index 843.
	if s[843] != 2.0 {
		t.Errorf("startLP[843] = %v, erwartet 2.0", s[843])
	}
	if s[842] != 0 || s[844] != 0 {
		t.Errorf("Start-Bonus streut: [842]=%v [844]=%v", s[842], s[844])
	}
	// Ende 863.39 -> Grenze NACH Sekunde 863 -> Index 864 (aufgerundet).
	if e[864] != 2.0 {
		t.Errorf("endLP[864] = %v, erwartet 2.0", e[864])
	}
	if e[863] != 0 {
		t.Errorf("End-Bonus eine Sekunde zu frueh: [863]=%v", e[863])
	}
}

func TestSpotBoundaryLPInnereGrenzenSindNurBoni(t *testing.T) {
	// Vier Spots Ruecken an Ruecken -- der reale Fall aus der Aufnahme
	// vom 2026-09-07. Jeder Anfang bekommt einen Start-Bonus, jedes Ende
	// einen End-Bonus. Dass die inneren wirkungslos sind, entscheidet der
	// HSMM (Zustand schon "ad"); hier wird nur festgehalten, dass sie
	// gesetzt werden und sich nicht aufaddieren.
	anchors := []signals.SpotAnchor{
		{StartS: 843.39, EndS: 863.39}, {StartS: 858.11, EndS: 878.11},
		{StartS: 881.02, EndS: 901.02}, {StartS: 901.04, EndS: 921.04},
	}
	s, e := SpotBoundaryLP(anchors, 1.0, 1000)
	gesetzt := 0
	for _, v := range s {
		if v > 1.0 {
			t.Fatalf("Start-Bonus ueber w: %v -- Anker duerfen sich nicht aufaddieren", v)
		}
		if v > 0 {
			gesetzt++
		}
	}
	if gesetzt != 4 {
		t.Errorf("%d Start-Boni, erwartet 4", gesetzt)
	}
	if e[922] != 1.0 {
		t.Errorf("End-Bonus des letzten Spots fehlt bei 922: %v", e[922])
	}
}

func TestSpotBoundaryLPAusIstAus(t *testing.T) {
	anchors := []signals.SpotAnchor{{StartS: 10, EndS: 30}}
	if s, e := SpotBoundaryLP(anchors, 0, 100); s != nil || e != nil {
		t.Error("w=0 muss nil liefern -- nil haelt den Dekoder byte-identisch")
	}
	if s, e := SpotBoundaryLP(nil, 1.0, 100); s != nil || e != nil {
		t.Error("keine Anker muss nil liefern")
	}
	// Ein Anker ausserhalb der Aufnahme wird ignoriert, nicht abgeschnitten.
	s, e := SpotBoundaryLP([]signals.SpotAnchor{{StartS: 5000, EndS: 5020}}, 1.0, 100)
	for i := range s {
		if s[i] != 0 || e[i] != 0 {
			t.Fatalf("Anker ausserhalb hat Bonus gesetzt bei %d", i)
		}
	}
}

func TestSpotBoundaryLPUngueltigerAnker(t *testing.T) {
	// Ende vor Anfang: kein Bonus, kein Panic.
	s, e := SpotBoundaryLP([]signals.SpotAnchor{{StartS: 50, EndS: 40}}, 1.0, 100)
	for i := range s {
		if s[i] != 0 || e[i] != 0 {
			t.Fatalf("ungueltiger Anker hat Bonus gesetzt bei %d", i)
		}
	}
}

func TestAddLP(t *testing.T) {
	if got := AddLP(nil, nil); got != nil {
		t.Error("nil+nil muss nil bleiben (Dekoder-Parity)")
	}
	b := []float64{0, 1, 0}
	if got := AddLP(nil, b); len(got) != 3 || got[1] != 1 {
		t.Errorf("nil+b = %v", got)
	}
	a := []float64{1, 1}
	got := AddLP(a, b)
	want := []float64{1, 2, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("AddLP = %v, erwartet %v", got, want)
		}
	}
	// Bumper und Spot an DERSELBEN Grenze: beide zaehlen.
	bump := []float64{0, 0.5, 0}
	spot := []float64{0, 2.0, 0}
	if AddLP(bump, spot)[1] != 2.5 {
		t.Error("Evidenzen an derselben Grenze muessen sich addieren")
	}
}
