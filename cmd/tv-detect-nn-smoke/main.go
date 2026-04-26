// tv-detect-nn-smoke — load the ONNX backbone + a head, run inference
// on every frame of an input file and print the average confidence.
// Throwaway harness for verifying ORT integration during development.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/simonchrz/tv-detect/internal/decode"
	"github.com/simonchrz/tv-detect/internal/signals"
)

func main() {
	backbone := flag.String("backbone", os.ExpandEnv("$HOME/mnt/pi-tv/hls/.tvd-models/backbone.onnx"), "ONNX backbone path")
	head := flag.String("head", os.ExpandEnv("$HOME/mnt/pi-tv/hls/.tvd-models/head.bin"), "binary head weights path (1280×float32 + 1×float32 bias = 5124 B)")
	maxFrames := flag.Int("max", 0, "stop after N frames (0 = full input)")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: tv-detect-nn-smoke <input.ts>")
		os.Exit(2)
	}
	input := flag.Arg(0)

	d, err := decode.NewDecoder(context.Background(), decode.DecodeOpts{Input: input})
	if err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(1)
	}
	defer d.Close()

	nn, err := signals.NewNNDetector(*backbone, *head, d.Width, d.Height, "")
	if err != nil {
		fmt.Fprintln(os.Stderr, "nn init:", err)
		os.Exit(1)
	}
	defer nn.Close()

	t0 := time.Now()
	count := 0
	sum := 0.0
	for f := range d.Frames() {
		c := nn.Confidence(f.Pixels, 0.5) // smoke tool has no logo signal — pass neutral 0.5
		sum += c
		count++
		if count%500 == 0 {
			fmt.Fprintf(os.Stderr,
				"\r%d frames  avg=%.3f  %.0f fps",
				count, sum/float64(count), float64(count)/time.Since(t0).Seconds())
		}
		if *maxFrames > 0 && count >= *maxFrames {
			break
		}
	}
	dt := time.Since(t0).Seconds()
	fmt.Fprintf(os.Stderr,
		"\n%d frames in %.1fs (%.0f fps)  avg confidence=%.3f\n",
		count, dt, float64(count)/dt, sum/float64(count))
}
