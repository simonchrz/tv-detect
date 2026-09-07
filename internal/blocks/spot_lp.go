package blocks

import "github.com/simonchrz/tv-detect/internal/signals"

// SpotBoundaryLP macht aus bekannten Werbespots Uebergangs-Evidenz fuer den
// HSMM-Dekoder -- dieselbe Form wie boundaryLP fuer Bumper: ein Log-Bonus
// je Sekundengrenze, Index t = Grenze bei Sekunde t, Laenge nSec+1.
//
// # WARUM ALS LOG-PROB UND NICHT ALS SNAP
//
// Produktion faehrt bare hsmm; RefineHSMM (die Snap-Kette) ist ein
// ungenutztes Experiment (hsmm_refine.go, 2026-07-29). Ein Snap in Form()
// oder RefineHSMM waere in Produktion tot -- die Klasse aus
// train_gate_mass_den_falschen_decoder. Im Dekoder selbst wiegt der Viterbi
// den Bonus gegen Emission und Dauer-Prior ab, statt nachtraeglich zu
// ziehen.
//
// Und es loest das Problem der INNEREN Spotgrenzen von selbst: ein Block aus
// vier Spots hat vier Spot-Anfaenge, aber nur EINEN Blockstart. Ein
// show->ad-Bonus an einer Stelle, an der der Zustand schon "ad" ist, kann
// keinen Uebergang erzeugen -- er ist wirkungslos. Ein Snap muesste das
// ausdruecklich erkennen.
//
// # WAS DER BONUS TUT UND NICHT TUT
//
// Start-Bonus am Spot-Anfang: ist die Emission davor schon "ad" (Ident,
// Trailer -- per Konvention Werbung), passiert der Uebergang dort und der
// Bonus greift nicht. War die NN zu SPAET (erster Spot sah aus wie
// Sendung), zieht der Bonus den Uebergang auf den Spot-Anfang.
//
// ⚠️ KORRIGIERT 2026-09-07 (O19). Hier stand, der Block werde "ausgedehnt,
// nie beschnitten". Das ist falsch. Ein UEBERGANGS-Bonus an Sekunde t
// macht eine Grenze bei t attraktiver, gleich aus welcher Richtung sie
// sonst gekommen waere -- er zieht also auch eine Kante nach INNEN, wenn
// der Ankerrand innerhalb des Blocks liegt. Gemessen an 47 Aufnahmen
// wandern mit den Bild-Ankern (scripts/wiederholung.py) bei Gewicht 1
// eine Kante nach aussen und fuenf nach innen, bei Gewicht 2 zwei gegen
// neun. Der Grund: 78.9 % dieser Anker liegen GANZ im Block, ihre Raender
// sitzen im Blockinneren.
//
// Die urspruengliche Begruendung gilt nur fuer einen Anker, dessen Rand
// AUSSERHALB des Blocks liegt. Die Audio-Anker sind kuerzer und sitzen
// naeher am Blockrand, sie verhalten sich deshalb wie beschrieben
// (Gewicht 1: acht nach aussen, eine nach innen).
//
// O19-Ergebnis: --spot-lp-w bleibt 0. Bis Gewicht 2 verbessert der Bonus
// einzelne Aufnahmen und verschlechtert keine, ab Gewicht 4 ueberholt der
// Schaden den Nutzen. Siehe docs/o19-spot-lp-preregistration.md.
//
// Das Gewicht ist je Anker konstant (w). Die Familiengroesse geht bewusst
// nicht ein: ab der Mindestgroesse (tv-recorder: 3) ist ein Spot ein Spot.
func SpotBoundaryLP(anchors []signals.SpotAnchor, w float64, nSec int) (startLP, endLP []float64) {
	if w <= 0 || len(anchors) == 0 || nSec <= 0 {
		return nil, nil
	}
	startLP = make([]float64, nSec+1)
	endLP = make([]float64, nSec+1)
	for _, a := range anchors {
		if a.EndS <= a.StartS {
			continue
		}
		// Grenze bei Sekunde t liegt zwischen Sekunde t-1 und t: der
		// Spot-Anfang bei 843.4 s ist die Grenze VOR Sekunde 843, also
		// Index 843; das Ende bei 863.4 s ist die Grenze nach Sekunde 863,
		// also Index 864 (aufgerundet).
		s := int(a.StartS)
		e := int(a.EndS + 0.999)
		if s >= 0 && s <= nSec && startLP[s] < w {
			startLP[s] = w
		}
		if e >= 0 && e <= nSec && endLP[e] < w {
			endLP[e] = w
		}
	}
	return startLP, endLP
}

// AddLP addiert zwei Boundary-LP-Vektoren elementweise; nil zaehlt als
// Null. So kommen Bumper- und Spot-Evidenz an derselben Grenze zusammen,
// statt dass die eine die andere ueberschreibt.
func AddLP(a, b []float64) []float64 {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]float64, n)
	for i := range out {
		if i < len(a) {
			out[i] += a[i]
		}
		if i < len(b) {
			out[i] += b[i]
		}
	}
	return out
}
