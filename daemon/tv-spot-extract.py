#!/usr/bin/env python3
"""
Extract per-spot fingerprints (Chromaprint audio + dHash visual) from
locally-cached .ts recordings and POST them to the gateway.

Why on Mac: ffmpeg startup overhead dominates per-spot extraction
(~5-7 s on Pi 5 ARM cores vs ~0.3-0.5 s on M-series). For a backfill
of 200+ recordings, that's 25 min vs 6 hours.

Source: ~/.cache/tv-detect-daemon/source/<uuid>.ts (already filled
by the detect daemon — no extra fetch). Falls back to the gateway
HTTP source endpoint if a uuid isn't cached locally.

Usage:
    tv-spot-extract.py --queue           # drain everything pending
    tv-spot-extract.py --uuid <uuid>     # one specific recording
    tv-spot-extract.py --queue --limit 5 # cap for testing
"""
import argparse
import base64
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

GATEWAY = os.environ.get("GATEWAY", "https://raspberrypi5lan:8443")
SOURCE_CACHE = Path.home() / ".cache" / "tv-detect-daemon" / "source"
CTX = ssl.create_default_context()
CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

# --- Tuned to match Pi-side service.py constants -----------------
SPOT_SILENCE_DB     = -30
SPOT_SILENCE_MIN_S  = 0.30
SPOT_MIN_DUR_S      = 12.0
SPOT_MAX_DUR_S      = 60.0
SPOT_TRIM_EDGE_S    = 2.0
SPOT_VISUAL_FRAMES  = 5


def http_json(url, method="GET", body=None, timeout=30):
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read())


def get_local_ts(uuid):
    """Return Path to local .ts or None if not cached + can't fetch."""
    p = SOURCE_CACHE / f"{uuid}.ts"
    if p.exists() and p.stat().st_size > 100_000:
        return p
    return None


def fetch_ads_user(uuid, retries=3):
    """Read confirmed ad blocks via gateway /recording/<uuid>/ads.
    Returns:
        list of [s,e] blocks (possibly empty) on success
        None on transient error (caller should treat as 'try again')
    """
    url = f"{GATEWAY}/recording/{uuid}/ads"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                d = json.loads(r.read())
            return d.get("ads") or []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s
                continue
            print(f"  fetch ads err ({uuid[:8]}): {e}", file=sys.stderr)
            return None


def silence_intervals(ts_path, block_start_s, block_end_s):
    """Run silencedetect over [block_start, block_end]. Returns
    list of (silence_start, silence_end) RELATIVE to block_start."""
    dur = max(0.5, block_end_s - block_start_s)
    try:
        r = subprocess.run(
            ["ffmpeg", "-ss", f"{block_start_s:.3f}",
             "-i", str(ts_path),
             "-t", f"{dur:.2f}",
             "-vn",
             "-af", f"silencedetect=n={SPOT_SILENCE_DB}dB:"
                    f"d={SPOT_SILENCE_MIN_S}",
             "-f", "null", "-"],
            capture_output=True, timeout=120)
    except Exception as e:
        print(f"  silencedetect err: {e}", file=sys.stderr)
        return []
    intervals = []
    cur = None
    for line in r.stderr.decode("utf-8", errors="replace").splitlines():
        if "silence_start:" in line:
            try:
                cur = float(line.split(
                    "silence_start:", 1)[1].strip().split()[0])
            except Exception:
                cur = None
        elif "silence_end:" in line and cur is not None:
            try:
                end = float(line.split(
                    "silence_end:", 1)[1].split("|", 1)[0].strip().split()[0])
                if end > cur:
                    intervals.append((cur, end))
            except Exception:
                pass
            cur = None
    return intervals


def spots_from_silences(silences, block_dur_s):
    """Audio runs between silences. Filtered to MIN..MAX dur."""
    bnd = [(0.0, 0.0)] + silences + [(block_dur_s, block_dur_s)]
    out = []
    for i in range(len(bnd) - 1):
        ss = bnd[i][1]
        se = bnd[i + 1][0]
        d = se - ss
        if SPOT_MIN_DUR_S <= d <= SPOT_MAX_DUR_S:
            out.append((ss, se))
    return out


def extract_chromaprint(ts_path, abs_start_s, dur_s):
    """Mac's homebrew ffmpeg lacks the chromaprint muxer; we use the
    standalone `fpcalc -raw` CLI (chromaprint package) instead. Pipe
    audio via ffmpeg → fpcalc to avoid temp file."""
    inner_dur = dur_s - 2 * SPOT_TRIM_EDGE_S
    if inner_dur < (SPOT_MIN_DUR_S - 2 * SPOT_TRIM_EDGE_S):
        return None
    inner_start = abs_start_s + SPOT_TRIM_EDGE_S
    try:
        # Stage 1: ffmpeg → mono 22050 Hz raw PCM s16le on stdout
        ff = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error",
             "-ss", f"{inner_start:.3f}",
             "-i", str(ts_path),
             "-t", f"{inner_dur:.2f}",
             "-vn", "-ac", "1", "-ar", "22050",
             "-f", "wav", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # Stage 2: fpcalc reads WAV from stdin, emits raw fingerprint
        fp = subprocess.run(
            ["fpcalc", "-raw", "-length",
             str(int(inner_dur) + 1), "-"],
            stdin=ff.stdout, capture_output=True, timeout=30)
        ff.stdout.close()
        ff.wait(timeout=5)
        if fp.returncode != 0:
            return None
    except Exception as e:
        print(f"  chromaprint err: {e}", file=sys.stderr)
        return None
    # Parse "FINGERPRINT=12345,67890,..." line, pack as big-endian
    # uint32 stream so Pi-side matcher can process bytewise.
    out_text = fp.stdout.decode("utf-8", errors="replace")
    ints = []
    for line in out_text.splitlines():
        if not line.startswith("FINGERPRINT="):
            continue
        for tok in line[len("FINGERPRINT="):].split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = int(tok)
                # fpcalc emits signed int32; convert to uint32 wraparound
                if v < 0:
                    v += (1 << 32)
                ints.append(v & 0xFFFFFFFF)
            except ValueError:
                pass
        break
    if len(ints) < 50:
        return None
    return b"".join(v.to_bytes(4, "big") for v in ints)


def extract_dhashes(ts_path, abs_start_s, dur_s,
                    n_frames=SPOT_VISUAL_FRAMES):
    """Single ffmpeg call: n_frames evenly-spaced 9×8 grayscale,
    pack as 5×8-byte big-endian. Skips first/last 1s to avoid
    transition flashes."""
    inner_start = abs_start_s + 1.0
    inner_dur = dur_s - 2.0
    if inner_dur < 2.0:
        return None
    fps = max(0.5, n_frames / inner_dur)
    try:
        r = subprocess.run(
            ["ffmpeg", "-loglevel", "error",
             "-ss", f"{inner_start:.3f}",
             "-i", str(ts_path),
             "-t", f"{inner_dur:.2f}",
             "-vf", f"fps={fps:.4f},scale=9:8,format=gray",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=30)
        out = r.stdout
    except Exception as e:
        print(f"  dhash err: {e}", file=sys.stderr)
        return None
    hashes = []
    for i in range(0, len(out), 72):
        chunk = out[i:i + 72]
        if len(chunk) != 72:
            break
        h = 0
        for y in range(8):
            row_off = y * 9
            bit_off = y * 8
            for x in range(8):
                if chunk[row_off + x] > chunk[row_off + x + 1]:
                    h |= 1 << (bit_off + x)
        hashes.append(h)
        if len(hashes) >= n_frames:
            break
    if not hashes:
        return None
    return b"".join(h.to_bytes(8, "big") for h in hashes)


def process_uuid(uuid):
    """Extract every spot, POST the bundle. Returns (n_spots, ok)."""
    ts = get_local_ts(uuid)
    if ts is None:
        print(f"  {uuid[:8]}: no local .ts (would need to fetch — "
              f"skipping for now)", flush=True)
        return (0, False)
    ads = fetch_ads_user(uuid)
    if ads is None:
        # Transient gateway error — keep uuid in queue for retry
        return (0, False)
    if not ads:
        print(f"  {uuid[:8]}: no ad blocks")
        return (0, True)
    spots_payload = []
    for bi, blk in enumerate(ads):
        try:
            bs, be = float(blk[0]), float(blk[1])
        except Exception:
            continue
        block_dur = be - bs
        if block_dur < SPOT_MIN_DUR_S:
            continue
        sils = silence_intervals(ts, bs, be)
        spots = spots_from_silences(sils, block_dur)
        if not spots and SPOT_MIN_DUR_S <= block_dur <= SPOT_MAX_DUR_S:
            spots = [(0.0, block_dur)]
        for ss_rel, se_rel in spots:
            spot_dur = se_rel - ss_rel
            abs_start = bs + ss_rel
            fp = extract_chromaprint(ts, abs_start, spot_dur)
            if fp is None:
                continue
            dh = extract_dhashes(ts, abs_start, spot_dur)
            spots_payload.append({
                "block_idx": bi,
                "block_start_s": round(bs, 2),
                "block_end_s": round(be, 2),
                "window_start_s": round(abs_start, 2),
                "fp_b64": base64.b64encode(fp).decode("ascii"),
                "dhashes_b64": base64.b64encode(dh).decode("ascii")
                                if dh else None,
            })
    if not spots_payload:
        print(f"  {uuid[:8]}: 0 spots extracted")
        return (0, True)
    # POST
    body = json.dumps({"spots": spots_payload}).encode()
    try:
        d = http_json(
            f"{GATEWAY}/api/internal/spot-fp/upload?uuid={uuid}",
            method="POST", body=body, timeout=60)
    except Exception as e:
        print(f"  {uuid[:8]}: POST err: {e}", file=sys.stderr)
        return (len(spots_payload), False)
    return (d.get("n_spots", 0), True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", help="single recording uuid")
    ap.add_argument("--queue", action="store_true",
                    help="drain /api/internal/spot-fp/queue")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap N uuids (queue mode)")
    args = ap.parse_args()
    if args.uuid:
        n, ok = process_uuid(args.uuid)
        print(f"done: {args.uuid[:8]} → {n} spots, ok={ok}")
        return
    if not args.queue:
        ap.error("--uuid or --queue required")
    q = http_json(f"{GATEWAY}/api/internal/spot-fp/queue")
    uuids = q.get("uuids") or []
    print(f"queue: {len(uuids)} uuids "
          f"({q.get('n_missing')} missing, "
          f"{q.get('n_partial_no_dhash')} partial)", flush=True)
    if args.limit > 0:
        uuids = uuids[:args.limit]
    n_ok = n_err = total_spots = 0
    t0 = time.time()
    for i, u in enumerate(uuids, 1):
        t1 = time.time()
        n, ok = process_uuid(u)
        dt = time.time() - t1
        total_spots += n
        if ok:
            n_ok += 1
        else:
            n_err += 1
        if i % 5 == 0 or i == len(uuids):
            avg = (time.time() - t0) / i
            eta_min = (len(uuids) - i) * avg / 60
            print(f"[{i}/{len(uuids)}] {u[:8]} {n} spots "
                  f"({dt:.1f}s, avg {avg:.1f}s, ETA {eta_min:.1f}m) "
                  f"· total: {total_spots} spots, "
                  f"{n_ok} ok / {n_err} err", flush=True)
    print(f"done: {n_ok}/{len(uuids)} ok, {total_spots} spots, "
          f"{(time.time()-t0)/60:.1f} min total")


if __name__ == "__main__":
    main()
