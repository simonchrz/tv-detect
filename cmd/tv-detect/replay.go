package main

// replay.go — --emit-signals-json / --replay-signals: decouples the
// expensive decode+detection pass from cheap block(re)formation. Written
// for train-head.py's "realistic eval" (2026-07-14): the held-out
// evaluation used to threshold the classifier's raw probability and run a
// naive contiguous-run grouper, never exercising blocks.Form()'s logo
// blend / nn-gate / bumper-snap / letterbox-snap / I-frame-snap refinement
// that production actually applies — meaning both the printed IoU numbers
// AND the deploy-gate's head-to-head decision were evaluating a different,
// simpler pipeline than what ships. Caching a recording's decode-derived
// signals once (logo/bumper/black/silence/scenes/letterbox/iframes — all
// independent of which classifier head is loaded) lets eval swap in a
// fresh candidate's nn_confs and call the REAL blocks.Form() per candidate,
// per test recording, without re-running ffmpeg every time.

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/simonchrz/tv-detect/internal/blocks"
	"github.com/simonchrz/tv-detect/internal/pipeline"
	"github.com/simonchrz/tv-detect/internal/signals"
)

// signalsDump is the --emit-signals-json/--replay-signals wire format —
// every argument blocks.Form takes besides Opts and the optional speaker
// stream (loaded separately via --speaker-csv, same as the normal path).
type signalsDump struct {
	FPS              float64                  `json:"fps"`
	FrameCount       int                      `json:"frame_count"`
	LogoConfs        []float64                `json:"logo_confs"`
	NNConfs          []float64                `json:"nn_confs"`
	BumperConfs      []float64                `json:"bumper_confs"`
	BumperStartConfs []float64                `json:"bumper_start_confs"`
	BoundaryConfs    []float64                `json:"boundary_confs,omitempty"`
	Blackframes      []signals.BlackEvent     `json:"blackframes"`
	Silences         []signals.SilenceEvent   `json:"silences"`
	SceneCuts        []signals.SceneCut       `json:"scene_cuts"`
	Letterbox        []signals.LetterboxEvent `json:"letterbox"`
	IFrames          []float64                `json:"iframes"`
}

func writeSignalsJSON(path string, res *pipeline.Result, silences []signals.SilenceEvent) error {
	d := signalsDump{
		FPS: res.FPS, FrameCount: res.FrameCount,
		LogoConfs: res.LogoConfs, NNConfs: res.NNConfs,
		BumperConfs: res.BumperConfs, BumperStartConfs: res.BumperStartConfs,
		BoundaryConfs: res.BoundaryConfs,
		Blackframes:   res.Blackframes, Silences: silences,
		SceneCuts: res.SceneCuts, Letterbox: res.Letterbox, IFrames: res.IFrames,
	}
	b, err := json.Marshal(d)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

// loadReplayNNCSV reads the idx,time_s,nn_confidence format --emit-nn-csv
// writes, returning confidences ordered by idx (gaps filled with 0.5 —
// callers should size the CSV to match the cached frame_count exactly).
func loadReplayNNCSV(path string, frameCount int) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	out := make([]float64, frameCount)
	for i := range out {
		out[i] = 0.5
	}
	for _, row := range rows {
		if len(row) < 3 {
			continue
		}
		idx, err := strconv.Atoi(row[0])
		if err != nil || idx < 0 || idx >= frameCount {
			continue // header row or out-of-range
		}
		conf, err := strconv.ParseFloat(row[2], 64)
		if err != nil {
			continue
		}
		out[idx] = conf
	}
	return out, nil
}

// runReplay is the --replay-signals entry point: load a cached decode's
// signals, optionally swap in a fresh nn_confs CSV (a different
// classifier candidate scored against the SAME cached decode), form
// blocks, and print via the normal --output path. No ffmpeg involved.
func runReplay(signalsPath, nnCSVPath, speakerCSVPath, output string, buildOpts func(fps float64) blocks.Opts) {
	b, err := os.ReadFile(signalsPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "replay-signals:", err)
		os.Exit(1)
	}
	var d signalsDump
	if err := json.Unmarshal(b, &d); err != nil {
		fmt.Fprintln(os.Stderr, "replay-signals: parse:", err)
		os.Exit(1)
	}
	nnConfs := d.NNConfs
	if nnCSVPath != "" {
		nnConfs, err = loadReplayNNCSV(nnCSVPath, d.FrameCount)
		if err != nil {
			fmt.Fprintln(os.Stderr, "replay-nn-csv:", err)
			os.Exit(1)
		}
	}
	var speakerConfFrames []float64
	if speakerCSVPath != "" {
		ws, err := signals.LoadSpeakerCSV(speakerCSVPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "speaker-csv:", err)
			os.Exit(1)
		}
		speakerConfFrames = signals.ExpandSpeakerToFrames(ws, d.FPS, d.FrameCount)
	}

	// boundary_confs are dumped at emit time (small per-frame scalars, unlike
	// the 1280-dim embeddings they're computed from), so the sweep can vary
	// --boundary-snap against a fixed dump. nil in older dumps → snap no-op.
	blockList := blocks.Form(buildOpts(d.FPS),
		d.LogoConfs, nnConfs, d.BumperConfs, d.BumperStartConfs, speakerConfFrames,
		d.BoundaryConfs, d.Blackframes, d.Silences, d.SceneCuts, d.Letterbox, d.IFrames, d.FrameCount)

	switch output {
	case "summary":
		writeReplaySummary(d.FPS, d.FrameCount, blockList)
	case "cutlist":
		writeCutlist(d.FPS, d.FrameCount, blockList)
	default:
		fmt.Fprintf(os.Stderr, "unknown output format: %q\n", output)
		os.Exit(2)
	}
}

// writeReplaySummary mirrors writeSummary's JSON shape (fps/frame_count/
// duration_s/blocks) without needing a full pipeline.Result — replay mode
// never decodes, so width/height/per-phase timing don't exist.
func writeReplaySummary(fps float64, frameCount int, bl []blocks.Block) {
	out := struct {
		FPS        float64      `json:"fps"`
		FrameCount int          `json:"frame_count"`
		DurationS  float64      `json:"duration_s"`
		Blocks     [][2]float64 `json:"blocks"`
	}{
		FPS: fps, FrameCount: frameCount,
		DurationS: float64(frameCount) / fps,
		Blocks:    make([][2]float64, len(bl)),
	}
	for i, blk := range bl {
		out.Blocks[i] = [2]float64{blk.StartS, blk.EndS}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}
