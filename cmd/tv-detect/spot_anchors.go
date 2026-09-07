package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/simonchrz/tv-detect/internal/signals"
)

// spotAnchorList haelt die bekannten Werbespots der aktuellen Aufnahme
// (--spot-anchors). Paketweit, damit main.go den Typ nicht selbst nennen
// muss und buildOpts wie writeSignalsJSON dieselbe Liste sehen.
var spotAnchorList []signals.SpotAnchor

// ladeSpotAnker liest entweder die Antwort von tv-recorder
// GET /api/internal/spot-fp/cluster-anchored/{uuid} (Objekt mit "anchored")
// oder ein nacktes Array. Beides, weil der Daemon die Antwort roh
// durchreicht und ein Handlauf gern nur das Array schreibt.
func ladeSpotAnker(pfad string) error {
	b, err := os.ReadFile(pfad)
	if err != nil {
		return err
	}
	var huelle struct {
		Anchored []signals.SpotAnchor `json:"anchored"`
	}
	if err := json.Unmarshal(b, &huelle); err == nil && huelle.Anchored != nil {
		spotAnchorList = huelle.Anchored
		return nil
	}
	var nackt []signals.SpotAnchor
	if err := json.Unmarshal(b, &nackt); err != nil {
		return fmt.Errorf("%s: weder {anchored:[...]} noch [...]: %w", pfad, err)
	}
	spotAnchorList = nackt
	return nil
}
