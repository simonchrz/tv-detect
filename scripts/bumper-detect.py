#!/usr/bin/env python3
"""Detect ad-bumper boundaries (black-frame + silence overlap) in a .ts.

German private TV reliably surrounds every commercial block with a
brief bumper: ~0.3-1.5s of near-black video AND near-silent audio,
simultaneously. Detecting that intersection gives sub-second
boundary candidates without needing a trained model — pure signal
processing via ffmpeg's blackdetect + silencedetect filters.

Output: bumpers.json next to the input, format:
  [{"t": <center_seconds>, "dur": <overlap_seconds>}]
sorted by t. Empty list = clean run, no bumpers found (probably an
ad-free public broadcaster, or a recording with no commercials).

Used by train-head.py as a teacher signal: boundaries in ads.json
that align with a bumper (±2s) get higher sample weight, treating
them as auto-confirmed by an independent signal. Also useful for
sub-second cutlist refinement at inference time."""

import argparse, json, re, subprocess, sys
from pathlib import Path


def detect_bumpers(src: Path,
                   black_min_dur: float = 0.10,
                   silence_min_dur: float = 0.15,
                   pix_th: float = 0.10,
                   db_th: float = -22.0,
                   proximity_max: float = 1.5,
                   timeout: int = 600):
    """Run a single ffmpeg pass with both filters chained, parse the
    intervals from stderr, and return the list of black∩silence
    overlap windows.

    Defaults tuned for German private TV: -22 dB silence threshold
    (broadcasters compress audio hot, -30 dB misses everything),
    1.5s proximity window (silence often starts 0.3-1s before the
    black cut as the show audio fades, so strict intersection misses
    most real bumpers — proximity catches both intersections and
    near-misses where the two events neighbor each other)."""
    # Two parallel analysis chains (split video+audio); filter logs go to
    # stderr, no actual encode happens because of -f null. Both chains
    # must be in the filter_complex graph or one disables the other —
    # using -vf together with -af with -an breaks audio decoding.
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner",
        "-i", str(src),
        "-filter_complex",
        f"[0:v]blackdetect=d={black_min_dur}:pix_th={pix_th}[v];"
        f"[0:a]silencedetect=n={db_th}dB:d={silence_min_dur}[a]",
        "-map", "[v]", "-map", "[a]",
        "-f", "null", "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log = p.stderr

    blacks = []
    for m in re.finditer(
            r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)",
            log):
        blacks.append((float(m.group(1)), float(m.group(2))))

    silences = []
    cur_start = None
    for line in log.splitlines():
        m = re.search(r"silence_start:\s*([-\d.]+)", line)
        if m:
            cur_start = float(m.group(1)); continue
        m = re.search(r"silence_end:\s*([-\d.]+)", line)
        if m and cur_start is not None:
            silences.append((cur_start, float(m.group(1))))
            cur_start = None

    # Proximity match: every black event paired with any silence whose
    # interval is within proximity_max of the black interval. We keep
    # the gap (negative if overlapping, positive if neighbouring) so
    # downstream consumers can weight by tightness.
    bumpers = []
    for b_s, b_e in blacks:
        b_c = (b_s + b_e) / 2
        best = None
        for s_s, s_e in silences:
            # gap = signed distance between intervals (0 = touching, <0 = overlap)
            if s_e < b_s:
                gap = b_s - s_e
            elif s_s > b_e:
                gap = s_s - b_e
            else:
                gap = -min(b_e, s_e) + max(b_s, s_s)  # negative = overlap depth
            if gap > proximity_max:
                continue
            if best is None or gap < best[0]:
                best = (gap, s_s, s_e)
        if best is None:
            continue
        gap, s_s, s_e = best
        # Bumper center = midpoint of the union (covers both events)
        u_s = min(b_s, s_s); u_e = max(b_e, s_e)
        bumpers.append({"t": round((u_s + u_e) / 2, 3),
                        "dur": round(u_e - u_s, 3),
                        "gap": round(gap, 3)})
    return bumpers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="path to .ts recording")
    ap.add_argument("--out", help="output json path "
                                  "(default: bumpers.json next to --rec-dir or src)")
    ap.add_argument("--rec-dir", help="if set, write to <rec_dir>/bumpers.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"src not found: {src}", file=sys.stderr); sys.exit(2)

    bumpers = detect_bumpers(src)

    if args.out:
        out = Path(args.out)
    elif args.rec_dir:
        out = Path(args.rec_dir) / "bumpers.json"
    else:
        out = src.with_suffix(".bumpers.json")
    out.write_text(json.dumps(bumpers))
    if not args.quiet:
        print(f"{src.name}: {len(bumpers)} bumper(s) → {out}")


if __name__ == "__main__":
    main()
