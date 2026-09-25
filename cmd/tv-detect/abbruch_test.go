package main

import (
	"os"
	"syscall"
	"testing"
	"time"
)

// SIGTERM → signalContext cancels → abgebrochen() refuses. main calls it
// before writing any output; without it a killed detect wrote a cutlist
// from truncated signals and exited 0.
func TestSIGTERMVerhindertAusgabe(t *testing.T) {
	ctx, cancel := signalContext()
	defer cancel()
	if err := abgebrochen(ctx); err != nil {
		t.Fatalf("before the signal: %v", err)
	}
	if err := syscall.Kill(os.Getpid(), syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ctx.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("ctx not cancelled by SIGTERM")
	}
	if err := abgebrochen(ctx); err == nil {
		t.Fatal("abgebrochen() = nil after SIGTERM — outputs would be written")
	}
}
