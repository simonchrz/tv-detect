package signals

import "testing"

// Die Muster sind der ganze Wert des Signals — ein zu grosszuegiges
// verwandelt jede Nachrichtenlaufschrift in einen Werbeblock, ein zu
// enges laesst die Haelfte der Trailer liegen. Beide Richtungen werden
// hier an echten OCR-Ausgaben gemessen (2026-08-15 aus Frames gezogen).
func TestBewerte(t *testing.T) {
	faelle := []struct {
		text             string
		hinweis, werbung bool
		warum            string
	}{
		// --- echte Trailer-Ausgaben ---
		{"Samstag 20:15 Joyn -", true, false, "Wochentag + Uhrzeit"},
		{"Heute (20:40 Uhr | Woozle Goozle", true, false, "Heute zaehlt als Tagesangabe"},
		{"GHOSTS MO-FR 19:15", true, false, "Sendeplatz-Kuerzel"},
		{"GIVE PEACE | A CHANCE | JEDEN TAG 14:00", true, false, "jeden Tag"},
		{"COMEDY CENTRAL LIVE | FREITAG 00:00", true, false, "Mitternacht ist eine gueltige Zeit"},
		{"Mogicompers | Heute 7:45 | Neu!", true, false, "einstellige Stunde"},

		// --- echte Werbe-Kennzeichnungen ---
		{"Werbung | Werbung", false, true, ""},
		{"O head& | shoulders | WERBUNG | Let's | Dance", false, true, "Split-Screen bei Let's Dance"},

		// --- beides gleichzeitig kommt vor ---
		{"Morgen 20:15 | Werbung", true, true, "Trailer im gekennzeichneten Block"},

		// --- echte Sendungs-Ausgaben, duerfen NICHT treffen ---
		{"Viel Lärm um ein Wasserbecken!", false, false, ""},
		{"DATES", false, false, ""},
		{"Zeuge Harald Speiser (34) / saß 5 Jahre im Gefängnis", false, false, ""},
		{"Das Neueste auf einen Blick: RTL.de", false, false, "Laufband ohne Zeit"},

		// --- die Grenzfaelle, wegen derer die Muster so eng sind ---
		{"Am Montag rief sie an", false, false, "Wochentag OHNE Uhrzeit ist kein Hinweis"},
		{"Es ist jetzt 20:15", false, false, "Uhrzeit OHNE Tagesangabe ist kein Hinweis"},
		{"Nur 29.99 Euro statt 49.99", false, false, "Preis ist keine Uhrzeit"},
		{"Montag 29:99", false, false, "unmoegliche Uhrzeit"},
		{"01379-70 70 90", false, false, "Gewinnspiel-Nummer ist keine Uhrzeit"},
		{"Werbungskosten absetzen", false, false, "Teilwort zaehlt nicht"},
	}
	for _, f := range faelle {
		h, w := Bewerte(f.text)
		if h != f.hinweis || w != f.werbung {
			t.Errorf("Bewerte(%q) = (%v,%v), erwartet (%v,%v)  %s",
				f.text, h, w, f.hinweis, f.werbung, f.warum)
		}
	}
}

// Ueberlappende Fenster muessen verschmelzen, sonst decodiert ffmpeg
// dieselben Sekunden mehrfach — bei kurzen Bloecken war das in der
// Handprobe ueber die Haelfte der Frames.
func TestFasseZusammen(t *testing.T) {
	got := fasseZusammen([]float64{100, 150, 400}, 90, 1000)
	if len(got) != 2 {
		t.Fatalf("erwartet 2 Fenster, bekam %d: %+v", len(got), got)
	}
	if got[0].von != 10 || got[0].bis != 240 {
		t.Errorf("erstes Fenster %+v, erwartet {10 240}", got[0])
	}
	if got[1].von != 310 || got[1].bis != 490 {
		t.Errorf("zweites Fenster %+v, erwartet {310 490}", got[1])
	}
}

func TestFasseZusammenGrenzen(t *testing.T) {
	// Kante am Anfang darf nicht negativ werden, Kante am Ende nicht
	// ueber die Laufzeit hinaus — ffmpeg liefert sonst ein leeres Fenster
	// und der Lauf verliert das Signal genau dort, wo Werbebloecke oft
	// sitzen (Anfang und Ende der Aufnahme).
	got := fasseZusammen([]float64{20, 980}, 90, 1000)
	if len(got) != 2 {
		t.Fatalf("erwartet 2 Fenster, bekam %+v", got)
	}
	if got[0].von != 0 {
		t.Errorf("Anfang %v, erwartet 0", got[0].von)
	}
	if got[1].bis != 1000 {
		t.Errorf("Ende %v, erwartet 1000", got[1].bis)
	}
}

func TestFasseZusammenLeer(t *testing.T) {
	if got := fasseZusammen(nil, 90, 1000); got != nil {
		t.Errorf("ohne Kanten erwartet nil, bekam %+v", got)
	}
}
