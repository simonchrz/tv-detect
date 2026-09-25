package decode

import (
	"context"
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"strings"
)

// Frame is one decoded video frame.
type Frame struct {
	Index  int     // 0-based frame number
	TimeS  float64 // PTS in seconds, derived as Index/FPS
	Pixels []byte  // raw rgb24 row-major, len = 3*Width*Height
}

// DecodeOpts controls the ffmpeg decode subprocess.
type DecodeOpts struct {
	Input  string
	Width  int     // target width  (0 = native)
	Height int     // target height (0 = native)
	StartS float64 // -ss seek offset (0 = beginning)
	DurS   float64 // -t duration limit (0 = full input)
	// ExtraInputArgs is injected BEFORE -i so it applies to the
	// input stream. Use for error-tolerance flags on corrupt IPTV
	// streams: ["-err_detect", "ignore_err", "-fflags", "+discardcorrupt"]
	// keeps ffmpeg producing frames through h264 PPS / packet-loss
	// errors instead of aborting (which would leave the caller with a
	// silent fallback for the affected range).
	ExtraInputArgs []string
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
	ctx      context.Context // the decoder's own context — cancelled by Close() or the caller
	cancel   context.CancelFunc
	stderr   *tailBuffer
	frames   chan Frame
	err      error
	nFrames  int // frames delivered so far (reader goroutine only)
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
	// Error-tolerance + similar input-side flags must come BEFORE -i.
	if len(opts.ExtraInputArgs) > 0 {
		args = append(args, opts.ExtraInputArgs...)
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
	// Keep the END of ffmpeg's stderr for the error message: the cause of
	// an abort is the last line, while error-tolerant runs on corrupt TS
	// can print thousands of harmless decode errors before it.
	stderr := &tailBuffer{max: 2048}
	cmd.Stderr = stderr
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
		ctx:      cctx,
		cancel:   cancel,
		stderr:   stderr,
		frames:   make(chan Frame, 4),
		bytesPer: w * h * 3,
	}
	go d.reader(stdout)
	return d, nil
}

func (d *Decoder) reader(r io.Reader) {
	defer close(d.frames)
	d.err = d.readFrames(r)
	if d.err != nil {
		d.cancel() // nobody reads the pipe any more — don't let Wait hang on a blocked ffmpeg
	}
	waitErr := d.cmd.Wait()
	// ⚠️ EOF on the pipe is NOT proof that ffmpeg finished: a crash or an
	// abort mid-chunk also closes stdout, and the reader alone cannot tell
	// that from a clean end. Until 2026-09-25 the exit status was dropped
	// (`defer d.cmd.Wait()`), so a chunk cut short by a dying ffmpeg came
	// back as a complete, error-free chunk with fewer frames — and every
	// signal after it silently shifted. Only our own cancel (Close() or the
	// caller's ctx) is an expected non-zero exit; the caller learns about
	// that from its ctx, not from here.
	if d.err == nil && waitErr != nil && d.ctx.Err() == nil {
		d.err = fmt.Errorf("ffmpeg after %d frames: %w: %s",
			d.nFrames, waitErr, d.stderr.String())
	}
}

// readFrames slices the pipe into frames until EOF. A truncated final
// frame (TS cut mid-stream) is a normal end; whether ffmpeg itself ended
// cleanly is decided by the exit status in reader.
func (d *Decoder) readFrames(r io.Reader) error {
	buf := make([]byte, d.bytesPer)
	for {
		_, err := io.ReadFull(r, buf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read frame %d: %w", d.nFrames, err)
		}
		// Copy pixels: the receiver may stash them past the next iteration.
		pix := make([]byte, d.bytesPer)
		copy(pix, buf)
		d.frames <- Frame{
			Index:  d.nFrames,
			TimeS:  float64(d.nFrames) / d.FPS,
			Pixels: pix,
		}
		d.nFrames++
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

// tailBuffer is an io.Writer that keeps only the last max bytes.
type tailBuffer struct {
	max int
	buf []byte
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.buf = append(t.buf, p...)
	if over := len(t.buf) - t.max; over > 0 {
		t.buf = t.buf[over:]
	}
	return len(p), nil
}

func (t *tailBuffer) String() string { return strings.TrimSpace(string(t.buf)) }
