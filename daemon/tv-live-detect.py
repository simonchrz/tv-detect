#!/opt/homebrew/bin/python3.13
"""Mac-side live-buffer ad detection for the Pi's HLS streams.

Drop-in replacement for the _live_adskip_loop + _live_ads_analyze
path in simonchrz/tvheadend/hls-gateway/service.py. SMB-free as of
2026-05-01 — all gateway state goes via HTTP:

  - .m3u8 + segments via Caddy file_server fast path on :8443
    (bypasses Flask, so doesn't bump channels[slug]["last_seen"]
    and keep an idle ffmpeg artificially warm)
  - per-channel config + block-length prior + cached-logo URL via
    /api/internal/live-config/<slug> (TTL-cached in-process)
  - logo template downloaded once per day to LOGO_CACHE_DIR
  - .live_ads.json read/written via /api/internal/live-ads
  - keeps a local /tmp segment cache per slug — only fetches
    segments that aren't already cached (after warmup, ~30 new
    segments per scan instead of all 1800)

Logic ported from service.py. Keep this file in lockstep:
  parse_comskip ← _rec_parse_comskip
  blackframe_extend_ads ← _blackframe_extend_ads
  analyze ← _live_ads_analyze
  main_loop ← _live_adskip_loop
"""

import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

TVD             = Path.home() / ".local" / "bin" / "tv-detect"


# HTTP gateway for live-config / state — SMB-free per-channel config
# fetch (Pi serves combined JSON: logo_smooth_s + start_lag +
# sponsor_duration + cached logo URL). Replaces three separate SMB
# reads with one HTTP round-trip per slug per scan.
GATEWAY_HTTP    = "http://raspberrypi5lan:8080"
_LIVE_CFG_CACHE = {}  # slug -> (cfg_dict, fetched_at)
_LIVE_CFG_TTL_S = 300


def _live_config(slug):
    if not slug:
        return {}
    now = time.time()
    cached = _LIVE_CFG_CACHE.get(slug)
    if cached and now - cached[1] < _LIVE_CFG_TTL_S:
        return cached[0]
    try:
        with urllib.request.urlopen(
                f"{GATEWAY_HTTP}/api/internal/live-config/{slug}",
                timeout=5) as r:
            cfg = json.loads(r.read())
        _LIVE_CFG_CACHE[slug] = (cfg, now)
        return cfg
    except Exception as ex:
        log(f"live-config fetch err for {slug}: {ex}")
        return cached[0] if cached else {}


def channel_logo_smooth_s(slug):
    return float(_live_config(slug).get("logo_smooth_s", 0))
# launchd starts agents with an empty PATH, so bare "ffmpeg" raises
# FileNotFoundError when subprocess.run looks it up. Resolve once at
# import via the parent shell's PATH (or the Homebrew default), then
# pass an absolute path to every subprocess call.
FFMPEG          = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
WORK_DIR        = Path("/tmp/tv-live-scan")
# Local cache for per-channel logo templates downloaded via HTTP
# (cached_logo_url from /api/internal/live-config). Reused across
# scans; refreshed when the gateway reports a different cached_logo_url
# (= channel logo template was retrained on the Pi).
LOGO_CACHE_DIR  = WORK_DIR / ".logos"
CADDY_BASE      = "https://raspberrypi5lan:8443"
SEGMENT_TIME    = 1
WINDOW_SECONDS  = 2 * 3600
ACTIVE_MTIME_S  = 60
MIN_BUFFER_SEGS = 120
# WINDOW_SIZE applies only to the initial scan per channel (or after a
# state gap where prev's latest_seg expired from the buffer). 3600 ≈ 1 h
# is a compromise between cold-start latency (~120 s fetch + ~30 s
# comskip per channel) and useful immediate coverage when the user
# scrubs backwards. Bump to 7200 for full 2 h coverage at the cost of
# ~4 min cold-start per channel. Subsequent scans are incremental.
WINDOW_SIZE     = 3600
# Backward overlap on incremental scans — must be ≥ the longest expected
# ad block (max_commercialbreak=900) so any block that crosses the seam
# between scans is fully visible to comskip on the next pass and gets
# re-detected with correct boundaries (otherwise we'd see a single block
# split into "open block from prev scan" + "new block this scan").
OVERLAP_SECONDS = 900
# Minimum new segments required to trigger an incremental scan. Below
# this, the scan would buy us nothing — comskip needs both endpoints of
# an ad block visible to detect it, so a few seconds of new content
# can't add a new block, only confirm the latest one.
MIN_INCREMENTAL_NEW = 5
RESCAN_SECONDS  = 720
LOOP_SLEEP      = 30
HTTP_TIMEOUT    = 30
# How many channel scans run concurrently. M-series Macs have plenty of
# cores idle here — comskip itself is single-threaded but pinning one
# scan per active channel parallelises naturally. 3 covers our typical
# ALWAYS_WARM set without over-saturating Caddy fetch bandwidth.
MAX_PARALLEL_SCANS = 3

# Caddy's tls-internal CA isn't in the system trust store. We're on the
# LAN, talking to a known IP — disable verification rather than wire up
# the cert for one consumer.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode   = ssl.CERT_NONE

SPONSOR_DURATION_DEFAULT = 20.0
SPONSOR_DURATION_BY_CHANNEL = {
    "vox":        25.0,
    "rtl":        20.0,
    "rtlzwei":    20.0,
    "prosieben":  20.0,
    "sat-1":      20.0,
    "kabel-eins": 20.0,
    "sixx":       20.0,
    "super-rtl":  20.0,
    "nitro":      20.0,
    "ntv":        20.0,
    "sport1":     20.0,
}

# Per-channel minimum backward extension when blackframe-extend
# finds nothing usable. Channels with soft logo fades (logo dims
# rather than hard-cuts) leave comskip 15-20 s behind the visual
# ad-start; without a blackframe to snap to we'd otherwise show
# the skip button that late. Witnessed:
#   rtlzwei 12:13:38 vs 12:13:57 detection = 19 s
#   rtlzwei 12:39:19 vs 12:39:38 detection = 19 s
START_LAG_FALLBACK = {
    "rtlzwei": 20.0,
}

def effective_start_lag(slug):
    """Learned overrides via HTTP gateway, hardcoded fallback."""
    cfg = _live_config(slug)
    if cfg.get("start_lag_s", 0) > 0:
        return float(cfg["start_lag_s"])
    return START_LAG_FALLBACK.get(slug or "", 0)


def effective_sponsor_duration(slug):
    """SPONSOR_DURATION_BY_CHANNEL lookup with learned overrides."""
    cfg = _live_config(slug)
    if cfg.get("sponsor_duration_s", 0) > 0:
        return float(cfg["sponsor_duration_s"])
    return SPONSOR_DURATION_BY_CHANNEL.get(
        slug or "", SPONSOR_DURATION_DEFAULT)

LIVE_ADSKIP_SLUGS = {
    "prosieben", "sat-1", "kabel-eins", "kabeleins", "prosiebenmaxx",
    "rtl", "vox", "rtlzwei", "nitro", "rtlup", "superrtl",
    "tele-5", "tele5", "sport1", "dmax", "sixx",
}


def log(msg):
    print(f"{datetime.now().strftime('%F %T')} {msg}", flush=True)


def parse_pdt(s):
    """Parse a #EXT-X-PROGRAM-DATE-TIME value → Unix timestamp.
    Python 3.9's fromisoformat() is picky: `+0200` (no colon) and
    `Z` suffix both fail; 3.11+ handles them natively. Normalize
    here so this works on /usr/bin/python3 (macOS ships 3.9)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', s)
    return datetime.fromisoformat(s).timestamp()


def load_live_ads():
    """GET initial state via HTTP — Pi serves from local disk."""
    try:
        with urllib.request.urlopen(
                f"{GATEWAY_HTTP}/api/internal/live-ads",
                timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        log(f"load live-ads via HTTP: {e}")
        return {}


_save_lock = threading.Lock()


def save_live_ads(state):
    """POST full state to gateway — atomic-rename happens on Pi side.
    Lock serialises concurrent writes from parallel scan workers."""
    with _save_lock:
        try:
            body = json.dumps(state).encode()
            req = urllib.request.Request(
                f"{GATEWAY_HTTP}/api/internal/live-ads",
                data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
        except Exception as e:
            log(f"save live-ads via HTTP: {e}")


def parse_comskip(out_dir):
    """Parse comskip's default .txt output → list of [start_s, stop_s].
    Adjacent blocks separated by a short sponsor card (≤25 s of "show")
    are merged."""
    # Exclude *.logo.txt — comskip writes that file alongside the
    # cutlist (since the SaveLogoMaskData patch in mpeg2dec.c) and
    # filesystem ordering can put it ahead of the actual cutlist.
    txts = [p for p in out_dir.glob("*.txt")
            if not p.name.endswith(".logo.txt")]
    if not txts:
        return []
    try:
        lines = txts[0].read_text().splitlines()
    except Exception:
        return []
    fps = 25.0
    ads = []
    # Comskip's .txt occasionally writes a long run of NUL bytes
    # before the first frame-range line. After str.strip() those NULs
    # stay — match the trailing frame pair via regex so the leading
    # garbage doesn't break parsing.
    line_re = re.compile(r"(\d+)\s+(\d+)\s*$")
    for line in lines:
        line = line.strip().replace("\x00", "")
        if not line or line.startswith("-"):
            continue
        if "FRAMES AT" in line:
            try:
                fps = float(line.split()[-1]) / 100.0
                if fps <= 0:
                    fps = 25.0
            except Exception:
                pass
            continue
        m = line_re.search(line)
        if not m:
            continue
        try:
            a = float(m.group(1)) / fps
            b = float(m.group(2)) / fps
            if b - a >= 10:
                ads.append([round(a, 2), round(b, 2)])
        except Exception:
            continue
    seen = set()
    dedup = []
    for a, b in ads:
        key = (round(a, 1), round(b, 1))
        if key not in seen:
            seen.add(key)
            dedup.append([a, b])
    dedup.sort(key=lambda x: x[0])
    MERGE_GAP = 25.0
    merged = []
    for a, b in dedup:
        if merged and a - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def blackframe_extend_ads(video_path, ads, channel_slug=None,
                           sponsor_duration=None, max_extend=None):
    """ffmpeg blackdetect-based post-processor for comskip output.
    Three cases after each ad-end:
      (1) two blacks → [ad]·(black)·[sponsor]·(black)·[show] — extend to 2nd.
      (2) one black near end → [ad]·(black)·[sponsor]·hardcut — extend by sponsor_duration.
      (3) no black → no extension.
    Plus a backward-extension for ad-start if a black sits within 25 s before."""
    if sponsor_duration is None:
        sponsor_duration = effective_sponsor_duration(channel_slug)
    if max_extend is None:
        max_extend = sponsor_duration + 10.0
    if not ads or not video_path or sponsor_duration <= 0:
        return ads
    scan_window = max(sponsor_duration + 8.0, 20.0)
    # Backward window for ad-START. Two blackframes typically precede
    # comskip's logo-loss-based detection on DE private TV:
    #   show-end (BF1) → ~25 s "Programmhinweis" sponsor card
    #   (logo still visible, looks like content) → (BF2) → real ad
    # We snap back to the EARLIEST blackframe in the window — 75 s
    # catches both: BF2 (~30-40 s back) when no promo, BF1 (~50-65 s
    # back) when there is one. Safely smaller than
    # min_show_segment_length=120 s so we won't pull a blackframe
    # from inside the show.
    START_SCAN = 75.0
    START_MAX_EXTEND = 75.0
    out = []
    for start, end in ads:
        ss = max(0, end - 1.0)
        new_end = end
        blacks_after = []
        try:
            proc = subprocess.run(
                [FFMPEG, "-hide_banner", "-nostats",
                 "-ss", str(ss), "-i", str(video_path),
                 "-t", str(scan_window + 2),
                 "-vf", "blackdetect=d=0.04:pix_th=0.30:pic_th=0.80",
                 "-an", "-f", "null", "-"],
                capture_output=True, text=True, timeout=25)
            for line in proc.stderr.splitlines():
                if "blackdetect" in line and "black_end:" in line:
                    try:
                        rel = float(line.split("black_end:")[1].split()[0])
                        abs_t = ss + rel
                        if end < abs_t <= end + scan_window:
                            blacks_after.append(abs_t)
                    except Exception:
                        pass
        except Exception:
            pass
        extend_to = end
        if len(blacks_after) >= 2 and (blacks_after[-1] - blacks_after[0]) >= 5.0:
            extend_to = blacks_after[-1] + 0.5
        elif blacks_after and blacks_after[0] <= end + 5.0:
            extend_to = blacks_after[0] + sponsor_duration
        elif blacks_after:
            extend_to = blacks_after[-1] + 0.5
        new_end = max(new_end, min(end + max_extend, extend_to))

        # ffmpeg fast-seek (-ss BEFORE -i) drops blackdetect output
        # for the first ~6 s after the seek-point — filter warm-up.
        # Pre-roll 12 s before our window so any blackframe at the
        # very start of the actual window survives. Blackframes still
        # filtered to [start - START_SCAN, start - 1].
        WARMUP = 12.0
        ss2 = max(0, start - START_SCAN - WARMUP)
        new_start = start
        blacks_before = []
        try:
            proc = subprocess.run(
                [FFMPEG, "-hide_banner", "-nostats",
                 "-ss", str(ss2), "-i", str(video_path),
                 "-t", str(START_SCAN + WARMUP + 2),
                 "-vf", "blackdetect=d=0.04:pix_th=0.30:pic_th=0.80",
                 "-an", "-f", "null", "-"],
                capture_output=True, text=True, timeout=25)
            for line in proc.stderr.splitlines():
                if "blackdetect" in line and "black_start:" in line:
                    try:
                        rel = float(line.split("black_start:")[1].split()[0])
                        abs_t = ss2 + rel
                        if start - START_SCAN <= abs_t < start - 1.0:
                            blacks_before.append(abs_t)
                    except Exception:
                        pass
        except Exception:
            pass
        if blacks_before:
            earliest = min(blacks_before)
            if start - earliest <= START_MAX_EXTEND:
                new_start = earliest
        # Fallback: if backward-extend found no earlier blackframe and
        # the channel is known to have a soft logo fade (comskip lags
        # by N s), shift back by at least that much.
        min_back = effective_start_lag(channel_slug)
        if min_back > 0 and new_start >= start - 1.0:
            new_start = max(0.0, start - min_back)
        out.append([round(new_start, 2), round(new_end, 2)])
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] <= 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def silence_extend_ads(video_path, ads, channel_slug=None,
                        max_shift_sec=75.0, noise_db=-30, min_dur=0.5):
    """ffmpeg silencedetect post-processor — analog zu blackframe_extend
    aber für Audio. Snappt Block-Grenzen an die nächste Stille im
    [start - max_shift, start - 1] / [end + 1, end + max_shift] window.
    Nützlich für Channels mit harten Cuts ohne Schwarzbild — die
    Audio-Stille zwischen Show-Ende und Werbung ist meist da auch wenn
    visuell kein blackframe existiert.

    Returns (refined_ads, moved_flags). Mover-Logik analog zu
    refine_ads_by_logo / blackframe_extend_ads."""
    if not ads or not video_path:
        return ads, [(False, False)] * len(ads)
    out = []
    moved = []
    for start, end in ads:
        new_start, new_end = start, end
        # Backward window
        ss_b = max(0, start - max_shift_sec - 4)
        try:
            proc = subprocess.run(
                [FFMPEG, "-hide_banner", "-nostats",
                 "-ss", str(ss_b), "-i", str(video_path),
                 "-t", str(max_shift_sec + 6),
                 "-vn",
                 "-af", f"silencedetect=n={noise_db}dB:d={min_dur}",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=25)
            silences = []
            for line in proc.stderr.splitlines():
                if "silence_start:" in line:
                    try:
                        rel = float(line.split("silence_start:")[1].split()[0])
                        abs_t = ss_b + rel
                        if start - max_shift_sec <= abs_t < start - 1:
                            silences.append(abs_t)
                    except Exception:
                        pass
            if silences:
                new_start = min(silences)  # earliest silence in window
        except Exception:
            pass
        # Forward window
        ss_f = max(0, end - 1)
        try:
            proc = subprocess.run(
                [FFMPEG, "-hide_banner", "-nostats",
                 "-ss", str(ss_f), "-i", str(video_path),
                 "-t", str(max_shift_sec + 4),
                 "-vn",
                 "-af", f"silencedetect=n={noise_db}dB:d={min_dur}",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=25)
            silences = []
            for line in proc.stderr.splitlines():
                if "silence_end:" in line:
                    try:
                        rel = float(line.split("silence_end:")[1].split()[0])
                        abs_t = ss_f + rel
                        if end + 1 < abs_t <= end + max_shift_sec:
                            silences.append(abs_t)
                    except Exception:
                        pass
            if silences:
                new_end = max(end, min(silences))
        except Exception:
            pass
        out.append([round(new_start, 2), round(new_end, 2)])
        moved.append((new_start != start, new_end != end))
    return out, moved


_ACTIVE_CACHE = {"set": set(), "fetched_at": 0.0}
_ACTIVE_TTL_S = 10  # batched fetch — one HTTP per cycle, not per slug


def _refresh_active():
    now = time.time()
    if now - _ACTIVE_CACHE["fetched_at"] < _ACTIVE_TTL_S:
        return
    try:
        with urllib.request.urlopen(
                f"{GATEWAY_HTTP}/api/internal/active-channels",
                timeout=5) as r:
            _ACTIVE_CACHE["set"] = set(json.loads(r.read()).get("active", []))
            _ACTIVE_CACHE["fetched_at"] = now
    except Exception as ex:
        log(f"active-channels fetch err: {ex}")


def channel_is_active(slug):
    """HTTP gateway call — Pi serves the freshness check from local
    disk (no SMB on Mac side). Batched: one fetch per ≤10 s, all
    slugs share."""
    _refresh_active()
    return slug in _ACTIVE_CACHE["set"]


def parse_playlist(text):
    """Extract first PROGRAM-DATE-TIME and ordered segment names from
    an HLS m3u8. Discontinuity markers don't reset our PDT — we always
    anchor on the first PDT and assume SEGMENT_TIME-second segments
    (matches the Pi's ffmpeg config: `-hls_time 1`)."""
    first_pdt = None
    seg_names = []
    for line in text.splitlines():
        line = line.strip()
        if first_pdt is None and line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            try:
                first_pdt = parse_pdt(line.split(":", 1)[1])
            except Exception:
                pass
        elif line and not line.startswith("#"):
            seg_names.append(line)
    return seg_names, first_pdt


def fetch_segment_http(slug, name, dest):
    """Pull one segment from Caddy's file_server (no Flask = no
    last_seen bump). Returns bytes written, raises on failure."""
    url = f"{CADDY_BASE}/hls/{slug}/{name}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=HTTP_TIMEOUT) as r:
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=256 * 1024)
    return dest.stat().st_size


_LOGO_CACHE_TTL_S = 86400


def cached_logo_path(slug):
    """Download per-channel logo template via HTTP and cache locally.
    Returns the local Path if available, or None if the gateway has
    no logo for this channel (fresh channel, never auto-trained).
    File-mtime-based TTL so a Pi-side retrain (= different content)
    propagates within a day; force-refresh by deleting the local file."""
    if not slug:
        return None
    cfg = _live_config(slug)
    if not cfg.get("cached_logo_url"):
        return None
    LOGO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOGO_CACHE_DIR / f"{slug}.logo.txt"
    if dest.is_file() and (
            time.time() - dest.stat().st_mtime < _LOGO_CACHE_TTL_S):
        return dest
    try:
        with urllib.request.urlopen(
                f"{GATEWAY_HTTP}{cfg['cached_logo_url']}",
                timeout=15) as r:
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(r.read())
            tmp.replace(dest)
        return dest
    except Exception as e:
        log(f"[{slug}] logo fetch: {e}")
        return dest if dest.is_file() else None


def analyze(slug, state, window_end_seg=None, window_size=WINDOW_SIZE):
    """Incremental + HTTP-fetch port of _live_ads_analyze."""
    # m3u8 fetched via Caddy file_server fast path (same auth-free
    # static path as segment fetches below). 30 KB per scan = trivial.
    try:
        with urllib.request.urlopen(
                f"{CADDY_BASE}/hls/{slug}/index.m3u8",
                context=_SSL_CTX, timeout=HTTP_TIMEOUT) as r:
            if r.status != 200:
                return False
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        log(f"[{slug}] m3u8 fetch: {e}")
        return False

    all_segs, first_pdt = parse_playlist(text)
    if first_pdt is None:
        log(f"[{slug}] skip: no PROGRAM-DATE-TIME in playlist")
        return False
    if len(all_segs) < MIN_BUFFER_SEGS:
        return False

    if window_end_seg is not None:
        try:
            end_idx = min(all_segs.index(window_end_seg) + 1, len(all_segs))
        except ValueError:
            end_idx = len(all_segs)
    else:
        end_idx = len(all_segs)

    prev = state.get(slug) or {}
    prev_latest_seg = prev.get("latest_seg")
    latest_name = all_segs[end_idx - 1]

    # Skip if nothing new — only meaningful for unbounded (live) scans.
    if window_end_seg is None and prev_latest_seg == latest_name:
        return False

    # Sliding-window: re-scan only the tail since prev_latest_seg
    # (plus OVERLAP_SECONDS backwards so ad-blocks crossing the seam
    # get re-detected with correct boundaries). Older ads are kept
    # via the `retained` array below — no reprocessing needed.
    incr_start = None
    if (window_end_seg is None
            and prev_latest_seg
            and prev_latest_seg in all_segs):
        prev_idx = all_segs.index(prev_latest_seg)
        new_segs = end_idx - prev_idx - 1
        if new_segs >= MIN_INCREMENTAL_NEW:
            overlap_segs = OVERLAP_SECONDS // SEGMENT_TIME
            incr_start = max(0, prev_idx - overlap_segs + 1)

    if incr_start is not None:
        start_idx = incr_start
        scan_mode = "incr"
    else:
        # Cold init scan. If we have no cached logo template for this
        # channel yet, expand the window to the full 2 h buffer so
        # tv-detect's auto-train has the maximum amount of show-with-
        # logo material to converge on a stable template. Logo training
        # quality drops sharply when a 1 h window happens to land in an
        # ad-heavy slot (lots of logo-less frames). One-time cost per
        # channel session — subsequent inits use the cached --logo and
        # fall back to the regular window_size.
        cached_logo_check = cached_logo_path(slug)
        if cached_logo_check and cached_logo_check.stat().st_size > 0:
            init_size = window_size
            scan_mode = "init"
        else:
            init_size = 7200
            scan_mode = "init-deep"
            log(f"[{slug}] no cached logo — using 2 h deep init scan")
        start_idx = max(0, end_idx - init_size)
    window = all_segs[start_idx:end_idx]

    window_first_pdt = first_pdt + start_idx * SEGMENT_TIME

    # ---- segment cache: only fetch what we don't already have ----
    cache_dir = WORK_DIR / slug / "segs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {p.name for p in cache_dir.glob("seg_*.ts")}
    needed = set(window)
    to_fetch = [n for n in window if n not in cached]
    to_delete = cached - needed

    for n in to_delete:
        try:
            (cache_dir / n).unlink()
        except Exception:
            pass

    fetched_n = 0
    fetched_bytes = 0
    fetch_t0 = time.time()
    for n in to_fetch:
        try:
            fetched_bytes += fetch_segment_http(slug, n, cache_dir / n)
            fetched_n += 1
        except Exception as e:
            log(f"[{slug}] fetch {n}: {e}")
    fetch_dt = time.time() - fetch_t0

    # ---- concat from local cache ----
    work = WORK_DIR / slug
    for f in list(work.glob("*.txt")) + list(work.glob("*.csv")):
        try:
            f.unlink()
        except Exception:
            pass
    merged = work / "merged.ts"
    merged.unlink(missing_ok=True)
    try:
        with open(merged, "wb") as out:
            for n in window:
                p = cache_dir / n
                if p.exists():
                    out.write(p.read_bytes())
    except Exception as e:
        log(f"[{slug}] cat: {e}")
        return False
    if not merged.exists() or merged.stat().st_size < 1_000_000:
        return False

    # ---- tv-detect detection ----
    # Migrated from comskip 2026-04-25: tv-detect now drives the live
    # buffer scan too. Output format is unchanged (comskip-compatible
    # cutlist), so parse_comskip below still works. tv-detect already
    # does multi-signal voting + I-frame snap + per-channel smoothing
    # internally — the legacy logo/blackframe/silence post-processing
    # below still runs as belt-and-suspenders (mostly no-ops on
    # already-snapped tv-detect boundaries; CSV-based logo refine
    # gracefully degrades because tv-detect doesn't write --csvout).
    cached_logo = cached_logo_path(slug)
    smooth_s = channel_logo_smooth_s(slug)
    scan_t0 = time.time()
    out_txt = work / f"{merged.stem}.txt"
    cmd = [str(TVD), "--quiet", "--workers", "4",
           "--logo-smooth", str(smooth_s),
           "--output", "cutlist"]
    if cached_logo and cached_logo.is_file() and cached_logo.stat().st_size > 0:
        cmd += ["--logo", str(cached_logo)]
    else:
        cmd += ["--auto-train", "5"]
    if slug:
        cmd += ["--channel-slug", slug]
    # Per-channel block-length prior — gateway returns it via
    # /api/internal/live-config (computed from user-confirmed
    # ads_user.json edits, ≥5 samples + σ ≥ 30 s). 0 = no prior,
    # tv-detect falls back to library defaults.
    cfg = _live_config(slug) if slug else {}
    if cfg.get("min_block_s", 0) > 0 and cfg.get("max_block_s", 0) > 0:
        cmd += ["--min-block-sec", str(cfg["min_block_s"]),
                "--max-block-sec", str(cfg["max_block_s"])]
    cmd += [str(merged)]
    try:
        proc = subprocess.run(cmd, timeout=600, check=False,
                                capture_output=True, text=True)
        out_txt.write_text(proc.stdout)
    except Exception as e:
        log(f"[{slug}] tv-detect: {e}")
    ads_raw = parse_comskip(work)
    # tv-detect already does multi-signal voting (blackframe + silence
    # + scene-cut) and I-frame snap internally — its boundaries are
    # already as good as it gets. We keep blackframe + silence
    # post-processing here as belt-and-suspenders: occasionally they
    # nudge a boundary another second or two when tv-detect was
    # uncertain. Logo-CSV refinement is gone — tv-detect doesn't
    # write --csvout, and its in-process logo confidence already
    # contributes to the multi-signal vote.
    ads_bf = blackframe_extend_ads(str(merged), ads_raw, channel_slug=slug)
    ads_si, moved_si = silence_extend_ads(str(merged), ads_raw,
                                            channel_slug=slug)
    # Silence wins where it moved (audio is precise on hard cuts);
    # else blackframe; else tv-detect's own boundary.
    ads_sec = []
    for (rs, re_), (bs, be), (ss, se), (ssi, esi) in zip(
            ads_raw, ads_bf, ads_si, moved_si):
        new_s = ss if ssi else bs
        new_e = se if esi else be
        ads_sec.append([new_s, new_e])
    # Quality stats for the per-channel health dashboard. Logo-shift
    # counter retired — tv-detect handles logo internally, so the
    # health signal worth tracking is "did blackframe/silence actually
    # move anything vs trusting tv-detect's own boundary?".
    logo_shifts = 0
    silence_shifts = sum(1 for ssi, esi in moved_si if ssi) \
                     + sum(1 for ssi, esi in moved_si if esi)
    bf_shifts = sum(1 for (rs, re_), (bs, be) in zip(ads_raw, ads_bf)
                    if abs(rs - bs) > 0.05 or abs(re_ - be) > 0.05)
    tail_extended = False
    # If the last block runs through the end of the scan window, comskip
    # can't see where it actually ends — neither logo-refine nor
    # blackframe-extend can extend past the available frames. Without
    # any extension the saved ad-end equals scan-end, which makes the
    # player show "ads finished" while the real ad is still running for
    # several more minutes. Extend optimistically by 10 min so the skip
    # button stays visible until the next scan re-detects the block
    # with proper boundaries (the retained-ads merge will overwrite
    # this entry in-place).
    if ads_sec:
        win_end_sec = len(window) * SEGMENT_TIME
        last_start, last_end = ads_sec[-1]
        if win_end_sec - last_end < 1.0:
            ads_sec[-1] = [last_start, last_end + 600.0]
            tail_extended = True
    # Refresh the cached logo template if comskip wrote a fresh one and
    # Logo-cache writeback removed — per-channel canonical templates
    # in /mnt/tv/hls/.tvd-logos/ are trained from completed recordings
    # (one-shot per channel, much higher quality than per-scan training)
    # and pulled by cached_logo_path() via HTTP.
    merged.unlink(missing_ok=True)
    scan_dt = time.time() - scan_t0

    ads_wall = [[round(window_first_pdt + a, 1), round(window_first_pdt + b, 1)]
                for a, b in ads_sec]

    buffer_cutoff = time.time() - WINDOW_SECONDS
    scan_start = window_first_pdt
    scan_end = window_first_pdt + len(window) * SEGMENT_TIME
    prev_ads = state.get(slug, {}).get("ads", [])
    retained = [a for a in prev_ads
                if (a[1] <= scan_start or a[0] >= scan_end)
                and a[1] > buffer_cutoff]
    merged_ads = sorted(retained + ads_wall)

    # Quality score 0-100. 100 = clean detection. Penalties:
    #   -25 if no ads found in init/init-deep scan with > 25-min window
    #         (KiKa runs ad-free so don't penalise short scans)
    #   -15 if last block was tail-extended (open block, won't trust end)
    #   -10 if logo-refine moved 0 boundaries but blackframe/silence
    #         had to (= cached logo template likely broken)
    #   -2 per very short block (< 30s) — likely a false positive
    durations = [b - a for a, b in ads_sec]
    very_short = sum(1 for d in durations if d < 30)
    score = 100
    if scan_mode in ("init", "init-deep") and len(window) >= 1500 \
            and not ads_sec:
        score -= 25
    if tail_extended:
        score -= 15
    if logo_shifts == 0 and (bf_shifts + silence_shifts) > 0:
        score -= 10
    score -= 2 * very_short
    score = max(0, min(100, score))
    quality = {
        "score": score,
        "scan_mode": scan_mode,
        "scan_dur_s": round(scan_dt, 1),
        "ads_in_scan": len(ads_sec),
        "logo_shifts": logo_shifts,
        "silence_shifts": silence_shifts,
        "bf_shifts": bf_shifts,
        "tail_extended": tail_extended,
        "very_short_blocks": very_short,
    }
    state[slug] = {"generated": time.time(), "ads": merged_ads,
                   "latest_seg": latest_name, "quality": quality}
    save_live_ads(state)
    log(f"[{slug}] {scan_mode} {len(window)} segs | "
        f"{len(ads_wall)} new, {len(retained)} retained, "
        f"{len(merged_ads)} total | fetched {fetched_n} "
        f"({fetched_bytes // (1024*1024)} MB in {fetch_dt:.1f}s), "
        f"scan {scan_dt:.1f}s")
    return True


def gc_stale_caches():
    """Drop /tmp/tv-live-scan/<slug>/ dirs whose corresponding playlist
    is no longer fresh (channel went idle). Keeps /tmp from accumulating
    after channel switches."""
    if not WORK_DIR.exists():
        return
    for d in WORK_DIR.iterdir():
        if not d.is_dir():
            continue
        slug = d.name
        if channel_is_active(slug):
            continue
        # Inactive channel — drop its segment cache
        segs = d / "segs"
        if segs.exists():
            try:
                shutil.rmtree(segs)
            except Exception:
                pass


_inflight = set()
_inflight_lock = threading.Lock()


def _run_scan(slug, state):
    """ThreadPoolExecutor worker: run analyze() and clear inflight on
    exit so the loop can re-submit this slug on the next cycle."""
    try:
        analyze(slug, state)
    except Exception as e:
        log(f"[{slug}] analyse: {e}")
    finally:
        with _inflight_lock:
            _inflight.discard(slug)


def main_loop():
    log(f"starting; tv-detect={TVD}, gateway={GATEWAY_HTTP}, "
        f"caddy={CADDY_BASE}, parallel={MAX_PARALLEL_SCANS}")
    # SMB-mount sanity removed — script no longer reads/writes SMB.
    # Gateway HTTP is the sole source of truth for state + config.
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load state ONCE at startup, then keep it in-memory for the
    # lifetime of the process. Pi-side _live_adskip_loop is disabled
    # when LIVE_ADS_OFFLOAD is set, so we're the only writer — no
    # need to re-read from gateway after we've started.
    state = load_live_ads()

    pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS,
                              thread_name_prefix="scan")
    last_gc = 0
    # Heartbeat is now synthesised on the Pi from /api/internal/
    # active-channels poll timestamps — no SMB write needed.
    while True:
        try:
            now = time.time()
            if now - last_gc > 600:
                gc_stale_caches()
                last_gc = time.time()
            candidates = [s for s in LIVE_ADSKIP_SLUGS
                          if channel_is_active(s)]
            recent = {s for s, v in state.items()
                      if time.time() - v.get("generated", 0) < RESCAN_SECONDS}
            with _inflight_lock:
                busy = set(_inflight)
            todo = [s for s in candidates
                    if s not in recent and s not in busy]
            free = MAX_PARALLEL_SCANS - len(busy)
            for slug in todo[:free]:
                with _inflight_lock:
                    _inflight.add(slug)
                    active = len(_inflight)
                log(f"[{slug}] analysing "
                    f"({active}/{MAX_PARALLEL_SCANS} active, "
                    f"{max(0, len(todo) - free)} queued)")
                pool.submit(_run_scan, slug, state)
        except Exception as e:
            log(f"loop: {e}")
        time.sleep(LOOP_SLEEP)


if __name__ == "__main__":
    main_loop()
