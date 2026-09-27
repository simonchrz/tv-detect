// tv-ocr-spur: Bildschirm-Text ueber die GANZE Aufnahme, nicht nur um Kanten.
//
// Wozu: eine OCR-Eingabespalte fuer den Kopf braucht Werte, die weder vom
// Kopf noch vom Label abhaengen. Die Produktion (tv-detect --ocr-marker)
// tastet nur um die VORHERGESAGTEN Kanten des gerade deployten Kopfs ab —
// jede Nacht ein anderes Fenster, eine Nachrechnung ueber Stunden wuerde
// nie fertig. Um die LABEL-Kanten abzutasten waere ein Leck: schon dass an
// einer Stelle OCR-Werte existieren, verriete dem Kopf die Menschenkante.
// Flaechendeckend ist beides nicht.
//
// Gekachelt in Fenster von 2*halb Sekunden (Vorgabe 180 s, wie ein
// Produktionsfenster), jedes mit eigenem -ss. OCRUmKanten setzt die Zeit
// eines Bildes als von + (n-1)*schritt; ueber eine Stunde an einem Stueck
// driftet das bei .ts (Memory frames_tragen_erwartete_zeit), je Kachel
// neu verankert bleibt der Fehler so klein wie in der Produktion.
//
// Die Ausgabe nennt die abgetasteten Spannen ausdruecklich: OCRUmKanten
// behaelt nur TREFFER, und ohne die Spannen waere "kein Treffer" von
// "nie hingesehen" nicht zu unterscheiden.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/signals"
)

type spanne struct {
	Von float64 `json:"von"`
	Bis float64 `json:"bis"`
}

type spur struct {
	Quelle      string   `json:"quelle"`
	QuelleBytes int64    `json:"quelle_bytes"`
	QuelleMtime int64    `json:"quelle_mtime"`
	DauerS      float64  `json:"dauer_s"`
	SchrittS    float64  `json:"schritt_s"`
	Breite      int      `json:"breite"`
	Abgetastet  []spanne `json:"abgetastet"`
	// Kacheln, deren ffmpeg-Aufruf scheiterte: NICHT abgetastet. Ohne diese
	// Liste saehe eine Luecke aus wie "kein Text im Bild".
	Fehlgeschlagen []spanne          `json:"fehlgeschlagen"`
	Funde          []signals.OCRFund `json:"funde"`
	Erstellt       string            `json:"erstellt"`
	LaufzeitS      float64           `json:"laufzeit_s"`
}

// mitStderr faengt waehrend fn alles ab, was auf os.Stderr geschrieben wird.
// Gelesen wird nebenlaeufig, damit ein volles Pipe-Puffer fn nicht blockiert.
func mitStderr(fn func()) string {
	r, w, err := os.Pipe()
	if err != nil {
		fn()
		return ""
	}
	alt := os.Stderr
	os.Stderr = w
	fertig := make(chan string)
	go func() {
		b, _ := io.ReadAll(r)
		fertig <- string(b)
	}()
	fn()
	w.Close()
	os.Stderr = alt
	return <-fertig
}

// kacheln zerlegt [0, dauer) in Fenster der Breite 2*halb. Die Mitte jeder
// Kachel ist die "Kante", die OCRUmKanten bekommt.
// kacheln teilt [0, dauer) in Fenster von 2*halb Sekunden.
//
// ⚠️ Ein Rest kuerzer als `mindest` (ein Abtastschritt) wird an die
// vorige Kachel angehaengt. Bis 2026-09-27 war er eine eigene Kachel:
// 4320.14 s ergab ein 0.14-s-Fenster ohne ein einziges Bild, das als
// "fehlgeschlagen" zaehlte — und der Daemon verwirft eine Spur mit
// Fehlschlaegen GANZ. Der erste MLP6-Detect im Alltag (rtl-1790502300)
// lief deshalb ohne OCR-Spalten; betroffen waren 4 von 334 Spuren.
func kacheln(dauer, halb, mindest float64) []spanne {
	var out []spanne
	for von := 0.0; von < dauer; von += 2 * halb {
		bis := von + 2*halb
		if bis > dauer {
			bis = dauer
		}
		out = append(out, spanne{von, bis})
	}
	if n := len(out); n > 1 && out[n-1].Bis-out[n-1].Von < mindest {
		out[n-2].Bis = out[n-1].Bis
		out = out[:n-1]
	}
	return out
}

func main() {
	var (
		quelle  = flag.String("quelle", "", "Aufnahme (.ts)")
		aus     = flag.String("aus", "", "Ziel-JSON")
		halb    = flag.Float64("halb", 90, "Halbfenster je Kachel (s)")
		schritt = flag.Float64("schritt", 2, "Abstand der Abtastpunkte (s), wie Produktion")
		helfer  = flag.String("helfer", "", "tv-ocr-Binary (Vorgabe: neben dieser Datei)")
	)
	flag.Parse()
	if *quelle == "" || *aus == "" {
		fmt.Fprintln(os.Stderr, "--quelle und --aus sind Pflicht")
		os.Exit(2)
	}
	h := *helfer
	if h == "" {
		if exe, err := os.Executable(); err == nil {
			h = filepath.Join(filepath.Dir(exe), "tv-ocr")
		}
	}
	st, err := os.Stat(*quelle)
	if err != nil {
		fmt.Fprintln(os.Stderr, "quelle:", err)
		os.Exit(1)
	}
	info, err := decode.Probe(*quelle)
	if err != nil || info.DurationS <= 0 {
		fmt.Fprintln(os.Stderr, "dauer nicht bestimmbar:", err)
		os.Exit(1)
	}

	t0 := time.Now()
	s := spur{
		Quelle: *quelle, QuelleBytes: st.Size(), QuelleMtime: st.ModTime().Unix(),
		DauerS: info.DurationS, SchrittS: *schritt, Breite: 960,
		Funde: []signals.OCRFund{}, Fehlgeschlagen: []spanne{},
	}
	for _, k := range kacheln(info.DurationS, *halb, *schritt) {
		mitte := (k.Von + k.Bis) / 2
		// Halbfenster = halbe Kachelbreite: genau diese Kachel, keine
		// Ueberlappung, fasseZusammen verschmilzt daher nichts.
		var f []signals.OCRFund
		meldung := mitStderr(func() {
			f, err = signals.OCRUmKanten(*quelle, []float64{mitte}, info.DurationS,
				signals.OCROpts{Helfer: h, FensterS: (k.Bis - k.Von) / 2,
					SchrittS: *schritt, Breite: 960, UnteresTeilbild: 1})
		})
		if meldung != "" {
			fmt.Fprint(os.Stderr, meldung)
		}
		// OCRUmKanten ueberspringt ein Fenster mit kaputtem ffmpeg-Aufruf
		// und sagt es nur auf stderr — hier wird daraus eine Luecke in der
		// Spur statt eines stillen "nichts gefunden".
		if strings.Contains(meldung, "ffmpeg-Fenster") {
			s.Fehlgeschlagen = append(s.Fehlgeschlagen, k)
			continue
		}
		if err != nil {
			// Der Helfer selbst ist kaputt — kein Teilergebnis schreiben,
			// eine halbe Spur saehe vollstaendig aus.
			fmt.Fprintln(os.Stderr, "ocr:", err)
			os.Exit(1)
		}
		s.Abgetastet = append(s.Abgetastet, k)
		s.Funde = append(s.Funde, f...)
	}
	s.Erstellt = time.Now().Format(time.RFC3339)
	s.LaufzeitS = time.Since(t0).Seconds()

	b, _ := json.Marshal(s)
	tmp := *aus + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "schreiben:", err)
		os.Exit(1)
	}
	if err := os.Rename(tmp, *aus); err != nil {
		fmt.Fprintln(os.Stderr, "umbenennen:", err)
		os.Exit(1)
	}
	var nh, nw int
	for _, f := range s.Funde {
		if f.Hinweis {
			nh++
		}
		if f.Werbemarker {
			nw++
		}
	}
	fmt.Printf("%s: %.0f s abgetastet in %d Kacheln, %d Hinweis, %d Werbung, %.0f s\n",
		filepath.Base(*quelle), info.DurationS, len(s.Abgetastet), nh, nw, s.LaufzeitS)
}
