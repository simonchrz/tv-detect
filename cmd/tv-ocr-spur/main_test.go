package main

import "testing"

// Die Kacheln muessen [0, dauer) lueckenlos und ohne Ueberlappung decken:
// eine Luecke waere "nie hingesehen", eine Ueberlappung zaehlte Treffer
// doppelt.
func TestKachelnDeckenLueckenlos(t *testing.T) {
	for _, dauer := range []float64{1, 179.9, 180, 181, 3600, 4105.3} {
		k := kacheln(dauer, 90)
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
			if s.Bis-s.Von > 180+1e-9 || s.Bis <= s.Von {
				t.Fatalf("dauer %v: Kachel %v ausserhalb 0..180 s", dauer, s)
			}
		}
	}
}
