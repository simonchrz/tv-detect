package signals

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ExtractAudioRMSPerSecond mirrors scripts/train-head.py's
// extract_audio_rms_per_second so the per-frame audio feature seen
// by training matches what tv-detect produces at inference. Runs a
// single ffmpeg pass that pipes the recording's first audio stream
// through asetnsamples + astats + ametadata, prints one
// "lavfi.astats.Overall.RMS_level=<dB>" line per second to stderr.
//
// Output is normalised the SAME way as the Python side:
//
//	norm = clip((rms_dB + 60) / 60, 0, 1)
//
// so a -60 dB silent block → 0.0 and a 0 dB full-scale clip → 1.0.
// Returns (nSeconds,) values; on any failure returns a neutral 0.5
// array of length nSeconds (matches the Python fallback exactly so
// recordings with no audio stream silently skip the feature).
//
// nSeconds is the number of seconds we want output for — usually
// recording duration in seconds, ceiled. ffmpeg may emit fewer or
// more lines than expected (depends on stream length); we left-pad
// the result to nSeconds with neutral 0.5.
func ExtractAudioRMSPerSecond(ctx context.Context, src string, nSeconds int) []float32 {
	neutral := make([]float32, nSeconds)
	for i := range neutral {
		neutral[i] = 0.5
	}
	if nSeconds <= 0 {
		return neutral
	}
	const sampleRate = 48000
	args := []string{
		"-nostdin", "-nostats",
		"-i", src,
		"-map", "0:a:0",
		"-ac", "1", "-ar", strconv.Itoa(sampleRate),
		"-af", fmt.Sprintf(
			"asetnsamples=n=%d,"+
				"astats=metadata=1:reset=1,"+
				"ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
			sampleRate),
		"-f", "null", "-",
	}
	c, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(c, "ffmpeg", args...)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return neutral
	}
	if err := cmd.Start(); err != nil {
		return neutral
	}
	out := make([]float32, 0, nSeconds)
	sc := bufio.NewScanner(stderr)
	// astats writes one line per second-window; per-line scan keeps
	// memory bounded even for very long recordings (3-h movie ≈
	// 11k lines).
	const prefix = "lavfi.astats.Overall.RMS_level="
	for sc.Scan() {
		line := sc.Text()
		i := strings.Index(line, prefix)
		if i < 0 {
			continue
		}
		val := strings.TrimSpace(line[i+len(prefix):])
		// ffmpeg writes "-inf" for digital silence — clamp to -90 dB.
		var dB float64
		if val == "-inf" || val == "-Inf" || val == "-INF" {
			dB = -90.0
		} else {
			parsed, err := strconv.ParseFloat(val, 64)
			if err != nil {
				dB = -90.0
			} else if math.IsInf(parsed, 0) || math.IsNaN(parsed) {
				dB = -90.0
			} else {
				dB = parsed
			}
		}
		// Same normalisation as Python: -60 dB → 0, 0 dB → 1.
		n := (dB + 60.0) / 60.0
		if n < 0 {
			n = 0
		} else if n > 1 {
			n = 1
		}
		out = append(out, float32(n))
	}
	_ = cmd.Wait()
	if len(out) == 0 {
		return neutral
	}
	if len(out) >= nSeconds {
		return out[:nSeconds]
	}
	// ffmpeg emitted fewer entries than the requested duration —
	// pad with neutral so the index space is still right.
	padded := make([]float32, nSeconds)
	copy(padded, out)
	for i := len(out); i < nSeconds; i++ {
		padded[i] = 0.5
	}
	return padded
}

// AudioDynamik ersetzt die Lautheit durch ihre gleitende
// Standardabweichung — das Gegenstueck zu audio_dynamik() in
// scripts/train-head.py. Beide Seiten MUESSEN dieselbe Zahl liefern.
//
// # WARUM
//
// Der Kommentar an ExtractAudioRMSPerSecond begruendet die Spalte damit,
// dass Werbung 6 bis 10 dB lauter laufe. Am 2026-09-08 ueber 293576
// Sekunden aus 98 Aufnahmen gemessen: 1.23 dB. Die EU-Lautheits-
// regulierung hat den alten Trick erledigt, und die Permutations-
// Wichtigkeit am deployten Kopf zeigte die Spalte folgerichtig als
// unbenutzt (Verlust 0.0021 gegen 0.2954 beim Logo).
//
// Was traegt, ist die SCHWANKUNG: Werbung ist stark komprimiert und
// haelt ihren Pegel, Sendung hat Dialog, Musik und Stille. AUC innerhalb
// jeder Aufnahme, Median ueber 98 Aufnahmen: Lautheit 0.592, Schwankung
// ueber 30 s 0.726 (nuetzlich in 81 % der Aufnahmen).
//
// # DIE DEFINITION, VERBINDLICH FUER BEIDE SEITEN
//
//	d[i] = Populations-Standardabweichung von
//	       rms[max(0, i-fenster/2) : min(n, i+fenster/2+1)]
//
// Zentriertes Fenster, an den Raendern BESCHNITTEN statt aufgefuellt,
// Nenner n und nicht n-1. scripts/test_audio_dynamik.py nagelt genau
// diese Kanten fest; TestAudioDynamikParitaet prueft sie hier gegen.
//
// # UEBER DIE GANZE AUFNAHME, NIE UEBER EIN STUECK
//
// Ein Fenster, das an einer Chunk-Grenze abgeschnitten wird, erzeugt
// dort einen Sprung, den es nicht gibt — dieselbe Klasse wie die
// Temporal-Deltas, die bis 2026-07-18 an jeder 32-Frame-Grenze auf null
// fielen. Deshalb steht diese Funktion HIER, wo das Array fuer die ganze
// Aufnahme entsteht, und nicht im Chunk-Pfad des Dekoders.
func AudioDynamik(rms []float32, fenster int) []float32 {
	n := len(rms)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	if fenster < 1 {
		fenster = 1
	}
	// Praefixsummen, damit jedes Fenster in konstanter Zeit faellt.
	c1 := make([]float64, n+1)
	c2 := make([]float64, n+1)
	for i, v := range rms {
		x := float64(v)
		c1[i+1] = c1[i] + x
		c2[i+1] = c2[i] + x*x
	}
	half := fenster / 2
	for i := 0; i < n; i++ {
		lo := i - half
		if lo < 0 {
			lo = 0
		}
		hi := i + half + 1
		if hi > n {
			hi = n
		}
		m := float64(hi - lo)
		mu := (c1[hi] - c1[lo]) / m
		v := (c2[hi]-c2[lo])/m - mu*mu
		if v < 0 {
			v = 0
		}
		out[i] = float32(math.Sqrt(v))
	}
	return out
}
