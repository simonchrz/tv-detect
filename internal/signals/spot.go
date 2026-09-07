package signals

// SpotAnchor ist ein bekannter Werbespot in dieser Aufnahme: ein Intervall,
// dessen Audio-/Bild-Fingerprint zu einer Familie mit >= N Mitgliedern
// gehoert -- derselbe Spot lief also nachweislich mehrfach.
//
// Warum das der haerteste Anker ist, den es gibt: Werbung WIEDERHOLT sich.
// Ein Werbeblock ist eine Folge bekannter Spots, und Spotgrenzen sind harte
// Schnitte mit festen Laengen (15/20/30 s). Ein Blockstart liegt bei oder
// VOR dem ersten bekannten Spot, ein Blockende bei oder NACH dem letzten --
// auf den Frame genau, ohne ein einziges menschliches Label. So arbeiten
// Werbemonitoring und Zuschauermessung; der Stapel hatte die Familien seit
// Monaten (tv-recorder spot.go), nutzte sie aber nur als Ja/Nein-Zaehler.
//
// Quelle: GET /api/internal/spot-fp/cluster-anchored/{uuid} auf tv-recorder,
// Feld "anchored". JSON-Tags entsprechen dessen Ausgabe 1:1.
type SpotAnchor struct {
	StartS     float64 `json:"window_start_s"`
	EndS       float64 `json:"end_s"`
	FamilyID   int64   `json:"family_id"`
	FamilySize int     `json:"family_size"`
}
