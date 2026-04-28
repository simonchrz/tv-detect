package decode

import (
	"context"
	"fmt"
	"io"
	"os/exec"
	"runtime"
)

// Frame is one decoded video frame.
type Frame struct {
	Index  int     // 0-based frame number
	TimeS  float64 // PTS in seconds, derived as Index/FPS
	Pixels []byte  // raw rgb24 row-major, len = 3*Width*Height
}

// DecodeOpts controls the ffmpeg decode subprocess.
type DecodeOpts struct {
	Input    string
	Width    int     // target width  (0 = native)
	Height   int     // target height (0 = native)
	StartS   float64 // -ss seek offset (0 = beginning)
	DurS     float64 // -t duration limit (0 = full input)
}

// Decoder streams decoded frames over a channel.
//
// Spawns one ffmpeg subprocess that pipes raw rgb24 to stdout. The
// reader goroutine slices the byte stream into Frame structs of
// exactly W*H*3 bytes each. Closing the returned channel signals EOF;
// any decode/exec error is delivered via Err() after Close.
type Decoder struct {
	Width  int
	Height int
	FPS    float64

	cmd      *exec.Cmd
	cancel   context.CancelFunc
	frames   chan Frame
	err      error
	bytesPer int
}

// NewDecoder probes the input, then spawns ffmpeg with the requested
// scale/crop. Caller must Range-receive on Frames() until it closes,
// then check Err().
func NewDecoder(ctx context.Context, opts DecodeOpts) (*Decoder, error) {
	info, err := Probe(opts.Input)
	if err != nil {
		return nil, err
	}
	w, h := info.Width, info.Height
	if opts.Width > 0 {
		w = opts.Width
	}
	if opts.Height > 0 {
		h = opts.Height
	}

	args := []string{"-hide_banner", "-loglevel", "error", "-nostdin"}
	// VideoToolbox MPEG-2 / H.264 hardware decode on macOS is 5-10×
	// faster than libavcodec software decode on M-series. Output still
	// piped to user-space as rgb24 (we need pixel access), but the
	// decode itself runs on the GPU's media engine. Linux containers
	// fall through to software (would need v4l2m2m on the Pi which is
	// flaky for MPEG-2). The flag must come BEFORE -i.
	if runtime.GOOS == "darwin" {
		args = append(args, "-hwaccel", "videotoolbox")
	}
	if opts.StartS > 0 {
		args = append(args, "-ss", fmt.Sprintf("%.3f", opts.StartS))
	}
	args = append(args, "-i", opts.Input)
	if opts.DurS > 0 {
		args = append(args, "-t", fmt.Sprintf("%.3f", opts.DurS))
	}
	args = append(args,
		"-map", "0:v:0",
		"-f", "rawvideo",
		"-pix_fmt", "rgb24")
	if opts.Width > 0 || opts.Height > 0 {
		args = append(args, "-vf", fmt.Sprintf("scale=%d:%d", w, h))
	}
	args = append(args, "-")

	cctx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(cctx, "ffmpeg", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start ffmpeg: %w", err)
	}

	d := &Decoder{
		Width:    w,
		Height:   h,
		FPS:      info.FPS,
		cmd:      cmd,
		cancel:   cancel,
		frames:   make(chan Frame, 4),
		bytesPer: w * h * 3,
	}
	go d.reader(stdout)
	return d, nil
}

func (d *Decoder) reader(r io.Reader) {
	defer close(d.frames)
	defer d.cmd.Wait()
	buf := make([]byte, d.bytesPer)
	idx := 0
	for {
		_, err := io.ReadFull(r, buf)
		if err == io.EOF {
			return
		}
		if err == io.ErrUnexpectedEOF {
			// truncated final frame — typical for TS files cut mid-stream
			return
		}
		if err != nil {
			d.err = fmt.Errorf("read frame %d: %w", idx, err)
			return
		}
		// Copy pixels: the receiver may stash them past the next iteration.
		pix := make([]byte, d.bytesPer)
		copy(pix, buf)
		d.frames <- Frame{
			Index:  idx,
			TimeS:  float64(idx) / d.FPS,
			Pixels: pix,
		}
		idx++
	}
}

// Frames returns the receive-only frame channel. Closes on EOF or error.
func (d *Decoder) Frames() <-chan Frame { return d.frames }

// Err returns any decode error after Frames() closes.
func (d *Decoder) Err() error { return d.err }

// Close cancels the ffmpeg subprocess; safe to call multiple times.
func (d *Decoder) Close() error {
	d.cancel()
	return nil
}
