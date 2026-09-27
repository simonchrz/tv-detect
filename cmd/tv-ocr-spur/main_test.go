package main

import "testing"

// Die Kacheln muessen [0, dauer) lueckenlos und ohne Ueberlappung decken:
// eine Luecke waere "nie hingesehen", eine Ueberlappung zaehlte Treffer
// doppelt.
func TestKachelnDeckenLueckenlos(t *testing.T) {
	for _, dauer := range []float64{1, 179.9, 180, 181, 3600, 4105.3, 4320.14, 1800.36} {
		k := kacheln(dauer, 90, 2)
		if len(k) == 0 || k[0].Von != 0 {
			t.Fatalf("dauer %v: beginnt nicht bei 0: %v", dauer, k)
		}
		for i := 1; i < len(k); i++ {
			if k[i].Von != k[i-1].Bis {
				t.Fatalf("dauer %v: Luecke/Ueberlappung bei %d: %v", dauer, i, k)
			}
		}
		if last := k[len(k)-1].Bis; last != dauer {
			t.Fatalf("dauer %v: endet bei %v", dauer, last)
		}
		for _, s := range k {
			if s.Bis-s.Von > 182+1e-9 || s.Bis <= s.Von {
				t.Fatalf("dauer %v: Kachel %v ausserhalb 0..182 s", dauer, s)
			}
		}
	}
}

// Ein Rest unter einem Abtastschritt enthaelt kein Bild und zaehlte als
// Fehlschlag, der die ganze Spur verwarf (rtl-1790502300, 4320.14 s).
func TestKeineWinzigeEndkachel(t *testing.T) {
	for _, dauer := range []float64{4320.14, 1800.36, 4320.54} {
		k := kacheln(dauer, 90, 2)
		if last := k[len(k)-1]; last.Bis-last.Von < 2 {
			t.Fatalf("dauer %v: End-Kachel %v kuerzer als ein Abtastschritt", dauer, last)
		}
	}
	if k := kacheln(1, 90, 2); len(k) != 1 || k[0].Bis != 1 {
		t.Fatalf("kurze Aufnahme: %v", k)
	}
}
