package signals

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Der Temporal-Block entsteht zweimal: hier in temporalSpalten fuer den
// Betrieb, und in train-head.py zusatzspalten/_churn_col fuer das Training.
// Zusammenlegen geht nicht — die eine Seite rechnet in numpy auf
// Sekundenzeilen, die andere in Go auf Frames. Also wird die Naht wie beim
// hsmm-Decoder mit Goldwerten festgenagelt.
//
// ⚠️ Ein Auseinanderlaufen erzeugt KEINEN Fehler, sondern ein leicht
// schlechteres Modell — unsichtbar gegen die Nacht-zu-Nacht-Streuung der
// IoU-Zahlen. Deshalb dieser Test.
//
// Goldwerte neu erzeugen: python3 scripts/gen-augment-parity.py

type augmentFall struct {
	Name         string    `json:"name"`
	Beschreibung string    `json:"beschreibung"`
	N            int       `json:"n"`
	BaseDim      int       `json:"baseDim"`
	Step         int       `json:"step"`
	NTemporal    int       `json:"nTemporal"`
	Base         []float32 `json:"base"`
	DP           []float32 `json:"dp"`
	DN           []float32 `json:"dn"`
	Churn        []float32 `json:"churn"`
}

type augmentGold struct {
	ChurnWindowS int           `json:"churnWindowS"`
	Faelle       []augmentFall `json:"faelle"`
}

func ladeGold(t *testing.T) augmentGold {
	t.Helper()
	roh, err := os.ReadFile("testdata/augment-parity.json")
	if err != nil {
		t.Fatalf("Goldwerte nicht lesbar (python3 scripts/gen-augment-parity.py): %v", err)
	}
	var g augmentGold
	if err := json.Unmarshal(roh, &g); err != nil {
		t.Fatalf("Goldwerte nicht parsebar: %v", err)
	}
	if len(g.Faelle) == 0 {
		t.Fatal("Goldwert-Datei ohne Faelle")
	}
	return g
}

// Die Fensterbreite steht in beiden Sprachen. Laeuft sie auseinander, sind
// alle anderen Vergleiche hier wertlos — deshalb zuerst.
func TestChurnFensterbreiteStimmtMitPython(t *testing.T) {
	g := ladeGold(t)
	if g.ChurnWindowS != churnWindowS {
		t.Fatalf("Fensterbreite: Go %d, train-head.py %d — stiller "+
			"Train/Serve-Bruch", churnWindowS, g.ChurnWindowS)
	}
}

func TestTemporalSpaltenParitaet(t *testing.T) {
	g := ladeGold(t)
	for _, f := range g.Faelle {
		t.Run(f.Name, func(t *testing.T) {
			if len(f.Base) != f.N*f.BaseDim {
				t.Fatalf("Goldwert kaputt: %d Basiswerte, erwartet %d",
					len(f.Base), f.N*f.BaseDim)
			}
			dp, dn, churn := temporalSpalten(f.Base, f.N, f.BaseDim, f.Step, f.NTemporal)
			vergleiche(t, "dp", dp, f.DP)
			vergleiche(t, "dn", dn, f.DN)
			vergleiche(t, "churn", churn, f.Churn)
		})
	}
}

// Toleranz: Python summiert die Unruhe ueber Praefixsummen in float64, Go
// addiert der Reihe nach in float32. Bei gleichem Ergebnis unterscheiden
// sich die letzten Bits — das ist Rundung, kein Auseinanderlaufen. Alles
// darueber ist eine andere Rechnung.
const augmentTol = 2e-5

func vergleiche(t *testing.T, was string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d Werte, erwartet %d", was, len(got), len(want))
	}
	schlimmster, wo := 0.0, -1
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		// relativ messen, wo die Werte gross sind
		if m := math.Abs(float64(want[i])); m > 1 {
			d /= m
		}
		if d > schlimmster {
			schlimmster, wo = d, i
		}
	}
	if schlimmster > augmentTol {
		t.Errorf("%s laeuft auseinander: groesste Abweichung %.3g bei Index %d "+
			"(Go %.6f, Python %.6f)", was, schlimmster, wo, got[wo], want[wo])
	}
}

// nTemporal steuert, WIE VIELE Spalten entstehen. Eine Spalte zu viel faellt
// beim Laden des Kopfes als Breitenfehler auf; eine zu wenig nicht — dann
// sitzt der Minute-Prior auf dem Platz der Unruhe, und der Kopf rechnet
// stillschweigend Unsinn.
func TestTemporalSpaltenAnzahl(t *testing.T) {
	base := make([]float32, 40*3)
	for i := range base {
		base[i] = float32(i) * 0.25
	}
	if dp, dn, ch := temporalSpalten(base, 40, 3, 1, 2); dp == nil || dn == nil || ch != nil {
		t.Errorf("nTemporal=2 muss dp und dn liefern, aber keine Unruhe")
	}
	if dp, dn, ch := temporalSpalten(base, 40, 3, 1, 3); dp == nil || dn == nil || ch == nil {
		t.Errorf("nTemporal=3 muss alle drei Spalten liefern")
	}
	if dp, _, _ := temporalSpalten(base, 40, 3, 1, 0); dp != nil {
		t.Errorf("nTemporal=0 darf gar nichts liefern")
	}
	if dp, _, _ := temporalSpalten(nil, 0, 3, 1, 3); dp != nil {
		t.Errorf("n=0 darf gar nichts liefern")
	}
}
