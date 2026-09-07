package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Die Bild-Anker aus scripts/wiederholung.py tragen bewusst dieselben
// Feldnamen wie die Antwort von tv-recorder. Dieser Test haelt fest, dass
// der Go-Leser sie ohne eine Zeile Anpassung liest -- sonst faellt es
// erst auf, wenn ein Lauf still ohne Anker fuehrt.
func TestLadeSpotAnkerBildform(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bild.json")
	// family_id = -1: Bild-Anker bilden keinen Familienverbund.
	inhalt := `{"uuid":"u1","anchored":[
	  {"window_start_s":843.0,"end_s":863.0,"family_id":-1,"family_size":5},
	  {"window_start_s":881.0,"end_s":901.0,"family_id":-1,"family_size":13}]}`
	if err := os.WriteFile(p, []byte(inhalt), 0o644); err != nil {
		t.Fatal(err)
	}
	spotAnchorList = nil
	if err := ladeSpotAnker(p); err != nil {
		t.Fatalf("ladeSpotAnker: %v", err)
	}
	if len(spotAnchorList) != 2 {
		t.Fatalf("%d Anker gelesen, erwartet 2", len(spotAnchorList))
	}
	a := spotAnchorList[0]
	if a.StartS != 843.0 || a.EndS != 863.0 {
		t.Errorf("Zeiten falsch gelesen: %.1f-%.1f", a.StartS, a.EndS)
	}
	if a.FamilyID != -1 {
		t.Errorf("family_id = %d, erwartet -1 (Bild-Anker haben keinen Verbund)", a.FamilyID)
	}
	if a.FamilySize != 5 {
		t.Errorf("family_size = %d, erwartet 5", a.FamilySize)
	}
}
