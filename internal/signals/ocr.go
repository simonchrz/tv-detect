package signals

// Bildschirm-Text als eigenes Signal.
//
// Warum das ueberhaupt existiert (gemessen 2026-08-15, Ledger §3af): die
// Kantenfehler sitzen ganz ueberwiegend dort, wo NN und Logo GEMEINSAM
// versagen — und zwar systematisch, nicht zufaellig:
//
//   * Programmtrailer tragen das Senderlogo und zeigen Sendungsmaterial.
//     Beide Signale lesen "Sendung", es ist aber der Rand des Werbeblocks.
//   * Split-Screen-Werbung laesst die Show in einer Ecke weiterlaufen.
//     Beide Signale lesen "Sendung", es ist aber Werbung.
//
// In beiden Faellen steht die Antwort als TEXT im Bild ("Montag 20:15",
// "Werbung" mit Countdown) — und der Backbone kann sie nicht lesen: er
// bekommt 224x224 aus 1920x1080 (nn.go), eine Hinweisleiste ist darin ~8 px
// hoch. Die Ueberanpassungs-Luecke train/test belegt das: +0.251 bei
// Trailern gegen +0.023 bei Sendung und +0.007 bei Werbung — der Kopf
// MERKT sich Trailer, statt ihr Merkmal zu lernen, weil das Merkmal gar
// nicht ankommt.
//
// OCR rechnet also nichts um, was schon da ist (das war die Sackgasse aller
// Zusatzspalten, Ledger §3w), sondern holt verworfene Information zurueck.
// Gemessen deckt es 26 % der Kantenfehler-Masse ab — bei nur 2 Frames je
// Fehlerstelle, also eine Untergrenze.
//
// KANTEN-LOKAL, nicht flaechendeckend: 32 ms/Frame heisst 134 s fuer eine
// 70-Minuten-Aufnahme bei 1 fps — das waere so teuer wie der Backbone
// selbst. Um die Blockgrenzen herum sind es ~23 s = +17 %.

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// OCRFund ist ein abgetasteter Zeitpunkt mit dem, was dort im Bild stand.
type OCRFund struct {
	TimeS       float64 `json:"time_s"`
	Hinweis     bool    `json:"hinweis"`     // Programmhinweis: Wochentag + Uhrzeit
	Werbemarker bool    `json:"werbemarker"` // Werbe-Kennzeichnung
	Text        string  `json:"text,omitempty"`
}

// OCROpts steuert Reichweite und Dichte der Abtastung.
type OCROpts struct {
	Helfer   string  // Pfad zum tv-ocr-Binary (Vision-basiert, macOS)
	FensterS float64 // Halbfenster um jede Kante
	SchrittS float64 // Abstand zwischen zwei Abtastpunkten
	Breite   int     // Skalierungsbreite der Frames (Text muss lesbar bleiben)
}

func (o *OCROpts) setzeVorgaben() {
	if o.FensterS <= 0 {
		o.FensterS = 90
	}
	if o.SchrittS <= 0 {
		o.SchrittS = 2
	}
	// 960 statt Originalbreite: Vision liest die Leisten dort zuverlaessig,
	// und die JPEG-Groesse (und damit die Laufzeit) sinkt deutlich. Kleiner
	// ist nicht ratsam — bei 640 fielen in der Handprobe Countdown-Ziffern aus.
	if o.Breite <= 0 {
		o.Breite = 960
	}
}

var (
	// Wochentage, "heute"/"morgen" und die Sendeplatz-Kuerzel, die in
	// Programmhinweisen vorkommen. Bewusst ohne "sonntags"-Formen: die
	// stehen auch in Sendungstiteln, und der Treffer braucht ohnehin
	// zusaetzlich eine Uhrzeit.
	reTag = regexp.MustCompile(
		`(?i)(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag` +
			`|heute|morgen|mo-fr|mo-so|di-do|jeden tag|täglich|taeglich)`)
	// Uhrzeit im deutschen TV-Format. Der Stundenteil ist auf 0..23 begrenzt,
	// damit Preise ("29.99") und Telefonnummern nicht mitzaehlen.
	reZeit = regexp.MustCompile(`\b([01]?\d|2[0-3])[:.][0-5]\d\b`)
	// Werbe-Kennzeichnung. Als eigenes Wort, sonst treffen
	// "Werbungskosten" und Sendungstitel mit.
	reWerbung = regexp.MustCompile(`(?i)\bwerbung\b`)
)

// Bewerte ordnet einen OCR-Text den beiden Mustern zu.
//
// Der Programmhinweis verlangt BEIDES — Tagesangabe und Uhrzeit. Ein
// Wochentag allein steht in jeder zweiten Sendung ("Am Montag rief sie
// an"), eine Uhrzeit allein auf jedem Nachrichten-Laufband.
func Bewerte(text string) (hinweis, werbemarker bool) {
	return reTag.MatchString(text) && reZeit.MatchString(text),
		reWerbung.MatchString(text)
}

// OCRUmKanten tastet die Umgebung der uebergebenen Kanten ab und liefert je
// Abtastpunkt, was dort zu lesen war.
//
// Ueberlappende Fenster werden zusammengelegt, bevor ffmpeg laeuft — bei
// eng beieinander liegenden Kanten (kurze Bloecke) sparte das in der
// Handprobe ueber die Haelfte der Frames.
func OCRUmKanten(quelle string, kanten []float64, dauerS float64, o OCROpts) ([]OCRFund, error) {
	o.setzeVorgaben()
	if o.Helfer == "" {
		return nil, fmt.Errorf("kein OCR-Helfer angegeben")
	}
	if _, err := os.Stat(o.Helfer); err != nil {
		return nil, fmt.Errorf("OCR-Helfer nicht ausfuehrbar: %w", err)
	}
	fenster := fasseZusammen(kanten, o.FensterS, dauerS)
	if len(fenster) == 0 {
		return nil, nil
	}

	tmp, err := os.MkdirTemp("", "tvd-ocr-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)

	var funde []OCRFund
	for i, f := range fenster {
		muster := filepath.Join(tmp, fmt.Sprintf("w%02d_%%04d.jpg", i))
		cmd := exec.Command("ffmpeg", "-nostdin", "-loglevel", "error",
			"-ss", strconv.FormatFloat(f.von, 'f', 2, 64),
			"-t", strconv.FormatFloat(f.bis-f.von, 'f', 2, 64),
			"-i", quelle,
			"-vf", fmt.Sprintf("fps=1/%g,scale=%d:-1", o.SchrittS, o.Breite),
			"-q:v", "4", muster)
		if out, err := cmd.CombinedOutput(); err != nil {
			// Ein kaputtes Fenster darf den Lauf nicht kosten — der Rest
			// der Aufnahme bleibt auswertbar, das Signal wird dort nur
			// duenner. Lautlos waere es allerdings falsch.
			fmt.Fprintf(os.Stderr, "ocr: ffmpeg-Fenster %.0f-%.0f fehlgeschlagen: %v (%s)\n",
				f.von, f.bis, err, strings.TrimSpace(string(out)))
			continue
		}
		bilder, _ := filepath.Glob(filepath.Join(tmp, fmt.Sprintf("w%02d_*.jpg", i)))
		if len(bilder) == 0 {
			continue
		}
		sort.Strings(bilder)
		texte, err := liesText(o.Helfer, bilder)
		if err != nil {
			return nil, err
		}
		for _, b := range bilder {
			n, ok := laufnummer(b)
			if !ok {
				continue
			}
			t := f.von + float64(n-1)*o.SchrittS
			h, w := Bewerte(texte[b])
			if !h && !w {
				continue // nur Treffer aufheben, der Rest ist Rauschen
			}
			funde = append(funde, OCRFund{TimeS: t, Hinweis: h, Werbemarker: w,
				Text: kuerze(texte[b])})
		}
		for _, b := range bilder {
			_ = os.Remove(b)
		}
	}
	sort.Slice(funde, func(a, b int) bool { return funde[a].TimeS < funde[b].TimeS })
	return funde, nil
}

type spanne struct{ von, bis float64 }

// fasseZusammen macht aus Kanten disjunkte Abtastfenster.
func fasseZusammen(kanten []float64, halb, dauerS float64) []spanne {
	if len(kanten) == 0 {
		return nil
	}
	s := append([]float64(nil), kanten...)
	sort.Float64s(s)
	var out []spanne
	for _, k := range s {
		von, bis := k-halb, k+halb
		if von < 0 {
			von = 0
		}
		if dauerS > 0 && bis > dauerS {
			bis = dauerS
		}
		if bis <= von {
			continue
		}
		if n := len(out); n > 0 && von <= out[n-1].bis {
			if bis > out[n-1].bis {
				out[n-1].bis = bis
			}
			continue
		}
		out = append(out, spanne{von, bis})
	}
	return out
}

// liesText ruft den Vision-Helfer einmal fuer alle Bilder eines Fensters.
// Ein Aufruf je Bild kostete in der Messung ~4x so viel (Prozessstart).
func liesText(helfer string, bilder []string) (map[string]string, error) {
	cmd := exec.Command(helfer, bilder...)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ocr-helfer: %w", err)
	}
	m := make(map[string]string, len(bilder))
	for _, z := range strings.Split(string(out), "\n") {
		if z == "" {
			continue
		}
		pfad, text, _ := strings.Cut(z, "\t")
		m[pfad] = text
	}
	return m, nil
}

func laufnummer(pfad string) (int, bool) {
	b := strings.TrimSuffix(filepath.Base(pfad), ".jpg")
	_, num, ok := strings.Cut(b, "_")
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(num)
	return n, err == nil
}

func kuerze(s string) string {
	const max = 160
	if len(s) <= max {
		return s
	}
	return s[:max]
}

// OCRAlsJSON ist die Form, in der die Funde im Signal-Dump landen.
func OCRAlsJSON(f []OCRFund) ([]byte, error) { return json.Marshal(f) }
