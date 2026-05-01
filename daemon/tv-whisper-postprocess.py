#!/usr/bin/env python3
"""Post-process a comskip-style cutlist using whisper-prob windows.

Daemon helper: stdin = raw cutlist, stdout = refined cutlist. Applies
the FP-killer + boundary-extend + missed-adder rules from
tv-whisper-eval.py. Does nothing destructive — on any error (missing
whisper.json, malformed cutlist, classifier crash) it passes the
input through unchanged so the surrounding pipeline always sees a
valid cutlist.

Cutlist format (comskip-compat):
    FILE PROCESSING COMPLETE  <n_frames> FRAMES AT  <fps*100>
    -------------------
    <start_frame>\\t<end_frame>
    ...
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

# Reuse rules from the eval script (same dir under symlink)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_eval_path = Path(__file__).resolve().parent / "tv-whisper-eval.py"
_spec = importlib.util.spec_from_file_location("tv_whisper_eval", _eval_path)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)


def parse_cutlist(text: str):
    """Returns (header_lines, fps, blocks). blocks = [(start_frame, end_frame), ...].
    Returns (None, None, None) if format is not recognisable."""
    fps = None
    header = []
    blocks = []
    line_re = re.compile(r"(\d+)\s+(\d+)\s*$")
    for line in text.splitlines():
        clean = line.replace("\x00", "").rstrip()
        if "FRAMES AT" in clean:
            try:
                fps = float(clean.split()[-1]) / 100.0
                if fps <= 0: fps = 25.0
            except ValueError:
                fps = 25.0
            header.append(clean)
            continue
        if clean.startswith("-") or not clean.strip():
            header.append(clean)
            continue
        m = line_re.search(clean)
        if m:
            blocks.append((int(m.group(1)), int(m.group(2))))
    if fps is None:
        return None, None, None
    return header, fps, blocks


def build_cutlist(header_lines, fps: float, blocks_seconds) -> str:
    """Inverse of parse_cutlist. Rounds to whole frames."""
    out = list(header_lines)
    for s, e in blocks_seconds:
        sf = max(0, int(round(s * fps)))
        ef = max(sf + 1, int(round(e * fps)))
        out.append(f"{sf}\t{ef}")
    # Trailing newline matches comskip output convention
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", type=Path, required=True,
                    help="Path to whisper.json (output of tv-whisper-classify)")
    ap.add_argument("--diag-out", type=Path, default=None,
                    help="Optional: write JSON diagnostic (removed/"
                         "extended/added) to this path.")
    args = ap.parse_args()

    raw = sys.stdin.read()

    # Pass-through fallback paths — daemon expects valid cutlist on stdout
    if not args.whisper.is_file():
        sys.stdout.write(raw)
        return

    header, fps, blocks = parse_cutlist(raw)
    if header is None or not blocks:
        sys.stdout.write(raw)
        return

    try:
        with open(args.whisper) as fp:
            wd = json.load(fp)
        windows = wd["windows"]
        window_s = wd["window_s"]
        stride_s = wd["stride_s"]
    except Exception as e:
        print(f"whisper-postprocess: load err: {e}", file=sys.stderr)
        sys.stdout.write(raw)
        return

    blocks_s = [[a / fps, b / fps] for a, b in blocks]
    try:
        refined_s, diag = _ev.apply_postprocess(blocks_s, windows, window_s,
                                                  stride_s)
    except Exception as e:
        print(f"whisper-postprocess: apply err: {e}", file=sys.stderr)
        sys.stdout.write(raw)
        return

    if args.diag_out:
        try:
            args.diag_out.write_text(json.dumps({
                "raw_blocks_s": blocks_s,
                "refined_blocks_s": refined_s,
                "fps": fps,
                **diag,
            }, indent=1))
        except Exception:
            pass

    sys.stdout.write(build_cutlist(header, fps, refined_s))


if __name__ == "__main__":
    main()
