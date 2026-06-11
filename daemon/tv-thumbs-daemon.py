#!/usr/bin/env python3
"""Polls Pi gateway via HTTP for pending thumbnail AND HLS-remux jobs.
Runs ffmpeg locally on the .ts (fetched via HTTP — 2-3× faster than
SMB), POSTs results back as a tar (sequential streaming, 2-3× faster
than SMB writes).

Two job types share the same poll loop, both via Pi-internal API:
  GET  /api/internal/thumbs-pending       → thumbnail jobs
  GET  /api/internal/hls-pending          → HLS-remux jobs
  GET  /recording/<uuid>/source           → range-streamed .ts
  POST /api/internal/thumbs-uploaded/<uuid>  → tar of JPGs
  POST /api/internal/hls-uploaded/<uuid>     → tar of HLS bundle

Why HTTP instead of SMB:
- macOS TCC (Transparency Consent Control) restricts launchd Aqua
  agents from enumerating network mounts. The interactive shell
  works (TCC has user-grant), the daemon does not (would need
  per-binary Full Disk Access, security trade-off).
- HTTP saturates gigabit (~110 MB/s) vs smbfs ~30-50 MB/s, AND
  doesn't have per-file metadata round-trips that kill bulk
  small-file writes (HLS bundles have hundreds of segments).

Triggered by launchd at boot via
~/Library/LaunchAgents/com.user.tv-thumbs-daemon.plist."""

import io, json, os, re, ssl, subprocess, sys, tarfile, tempfile, threading, time
import urllib.error, urllib.request
from pathlib import Path

POLL_INTERVAL_S = 5
TIMEOUT_S = 1800           # HLS-remux of 90-min recording can take ~30s
GATEWAY = os.environ.get("GATEWAY", "https://raspberrypi5lan:8443")  # Caddy; CTX below

# Concurrency: how many detect jobs to run in parallel. Default 1 (legacy
# sequential). Set DETECT_PARALLEL=3 to overlap downloads + decode + NN.
# Each tv-detect run uses --workers (8 // DETECT_PARALLEL) CPU cores so
# total per-job CPU stays bounded — a Mac with 8 perf cores running 3
# parallel detects with 4 workers each is fine; the 8th core handles
# Python + ffmpeg overhead. Memory bound: each detect carries ~1-2 GB
# (ffmpeg decode buffers + NN backbone), so 3-parallel ≈ 4-6 GB peak.
DETECT_PARALLEL = max(1, int(os.environ.get("DETECT_PARALLEL", "1")))
TVD_WORKERS = max(2, 12 // DETECT_PARALLEL)  # 2026-05-11: 18-core Mac, bumped from 8 → 12 divisor
# HLS+thumbs jobs (process_recording) used to run sequential in the main
# poll loop with a `break` after the first hls — meant Pi-fallback fired
# whenever the queue grew past ~15 markers (= 30 min × 1 per ~2 min).
# Mac M5 Pro VideoToolbox HW-encoder handles 2-3 parallel HLS encodes
# without thermal/CPU concern; thumbs is cheap (~10s) and rides along.
HLS_PARALLEL = max(1, int(os.environ.get("HLS_PARALLEL", "2")))
# Detect priority vs on-demand HLS remux. A VOD remux is latency-critical
# (a user is staring at a loading bar); detect is best-effort background.
# Two levers keep a remux from queuing behind detect (separate thread pools,
# but they contend for CPU + GPU/ANE):
#   * DETECT_NICE — run tv-detect (+ its ffmpeg/onnx children, which inherit
#     niceness) at a lower CPU priority so the remux encode wins the cores.
#   * gate (in main loop) — don't START new detect jobs while HLS remuxes are
#     pending or in flight, so the latency-critical window has fewer GPU/ANE
#     competitors. In-flight detects finish; detect resumes once HLS drains.
DETECT_NICE = max(0, int(os.environ.get("DETECT_NICE", "5")))
# Watchdog: if a uuid sits in detect_in_flight longer than this AND no
# subprocess for it exists, the worker thread leaked the slot (= hung
# HTTP POST, segfault in ONNX/ffmpeg ext, etc) and never decremented.
# 30 min comfortably above worst-case detect+whisper combined runtime.
IN_FLIGHT_STUCK_S = 30 * 60
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
TVD = os.path.expanduser("~/.local/bin/tv-detect")
SAFE_VIDEO = {"h264", "hevc"}

# Sub-process environment with a sane PATH. launchd starts agents
# with PATH=/usr/bin:/bin, so any tool spawn that internally exec's
# a binary by NAME (= ffmpeg, fpcalc, ECAPA's audio loader inside
# extract-speaker-embeddings) raises FileNotFoundError. The daemon
# uses absolute paths for its OWN ffmpeg/ffprobe calls, but the
# Python helpers it spawns (= speaker-extract, whisper-classify,
# centroid-build, postprocess) all rely on PATH lookups internally.
# Pass SPAWN_ENV to every subprocess.run/Popen for these helpers.
# Includes ~/.local/bin (tv-detect), brew (Apple Silicon default),
# and the standard system dirs.
SPAWN_ENV = {**os.environ, "PATH": (
    f"{os.path.expanduser('~/.local/bin')}:"
    f"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")}

# Local cache for model files (head.bin, backbone.onnx) — refreshed
# on size change so a nightly retrain auto-propagates without daemon
# restart. Keyed by remote ETag/size; head.bin is small (~5 KB),
# backbone.onnx is ~9 MB.
MODEL_CACHE = Path.home() / ".cache" / "tv-detect-daemon"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# Local .ts cache. First fetch via HTTP, subsequent uses read from
# Mac NVMe (~3 GB/s) instead of LAN gigabit (~110 MB/s) — 30× faster
# for any re-detect (head-bin invalidation, accidental marker, etc).
# LRU-evicted at SOURCE_CACHE_MAX_GB so the cache doesn't fill the SSD.
SNAPSHOT_MARKER = Path.home() / ".cache" / "tv-detect-daemon" / "snapshot-requested"
SOURCE_CACHE = Path.home() / ".cache" / "tv-detect-daemon" / "source"
SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
SOURCE_CACHE_MAX_GB = int(os.environ.get("SOURCE_CACHE_MAX_GB", "60"))

# Speaker-fingerprint orchestration (default OFF). When SPEAKER_ENABLE=1
# the daemon runs three Python helpers around each detect:
#   1. extract-speaker-embeddings.py — once per recording, ECAPA-TDNN
#      embeddings cached as <uuid>.npz (~30KB-1MB per recording, never
#      invalidated since audio doesn't change)
#   2. update-show-centroid.py — rebuilt per show on every detect of an
#      episode of that show (cheap once embeddings exist; sub-second)
#   3. compute-speaker-confs.py — per-recording speaker.csv combining
#      this recording's embeddings with the show centroid
# tv-detect then runs with --speaker-csv + --speaker-weight.
# Disabled by default until empirically validated to net-improve IoU.
SPEAKER_ENABLE = os.environ.get("SPEAKER_ENABLE", "0") == "1"
SPEAKER_WEIGHT = float(os.environ.get("SPEAKER_WEIGHT", "0.3"))
EMB_CACHE = Path.home() / ".cache" / "tv-detect-daemon" / "embeddings"
CENTROID_CACHE = Path.home() / ".cache" / "tv-detect-daemon" / "centroids"
SPK_CSV_CACHE = Path.home() / ".cache" / "tv-detect-daemon" / "speaker-csv"
if SPEAKER_ENABLE:
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    CENTROID_CACHE.mkdir(parents=True, exist_ok=True)
    SPK_CSV_CACHE.mkdir(parents=True, exist_ok=True)
SPEAKER_PYTHON = os.environ.get(
    "SPEAKER_PYTHON",
    "/Users/simon/ml/tv-classifier/.venv/bin/python")
SPEAKER_SCRIPTS = Path("/Users/simon/src/tv-detect/scripts")

# Whisper post-processor orchestration (default OFF). When WHISPER_ENABLE=1
# the daemon does an extra step after tv-detect:
#   1. tv-whisper-classify.py: .ts → per-window ad-prob JSON
#      (~50 s on M-series Mac with base model + 4 workers)
#   2. tv-whisper-postprocess.py: pipes the raw cutlist through, applies
#      FP-killer / boundary-extend / missed-adder rules from
#      tv-whisper-eval.py, emits refined cutlist for upload
# Empirically validated 2026-05-01 on n=9 reviewed: +5.4% mean symmetric
# Block-IoU, 3 helped / 0 hurt / 6 unchanged. Most gain from boundary-
# extend on truncated ad-block ends (Charmed +10.6%, The Middle +21.7%).
# Pass-through-safe: any error preserves the raw cutlist.
WHISPER_ENABLE = os.environ.get("WHISPER_ENABLE", "0") == "1"
# Spot-fingerprint extraction (audio Chromaprint + visual dHash).
# Default ON — Mac is the source of truth for spot extraction
# (Pi worker disabled, Mac fast). Drains the gateway queue
# whenever the daemon is idle (= no detect jobs in flight).
SPOT_EXTRACT_ENABLE = os.environ.get("SPOT_EXTRACT_ENABLE", "1") == "1"
SPOT_EXTRACT_SCRIPT = Path(__file__).parent / "tv-spot-extract.py"
SPOT_EXTRACT_BATCH  = int(os.environ.get("SPOT_EXTRACT_BATCH", "5"))
WHISPER_PYTHON = os.environ.get(
    "WHISPER_PYTHON",
    "/Users/simon/ml/tv-classifier/.venv/bin/python")
WHISPER_SCRIPTS = Path("/Users/simon/src/tv-detect/daemon")
WHISPER_CACHE = Path.home() / ".cache" / "tv-whisper"
if WHISPER_ENABLE:
    WHISPER_CACHE.mkdir(parents=True, exist_ok=True)


def _maybe_whisper_refine(uuid, src_path, raw_cutlist):
    """Run whisper-classify (idempotent — cached per-uuid) then pipe the
    raw cutlist through tv-whisper-postprocess. Returns refined cutlist
    on success, raw cutlist on any error. Adds ~50 s wallclock per
    detect on the first run; cached runs cost ~3 s (postprocess only)."""
    if not WHISPER_ENABLE:
        return raw_cutlist
    if not src_path or not Path(src_path).is_file():
        return raw_cutlist
    whisper_path = WHISPER_CACHE / f"{uuid}.whisper.json"
    classify_py = WHISPER_SCRIPTS / "tv-whisper-classify.py"
    postprocess_py = WHISPER_SCRIPTS / "tv-whisper-postprocess.py"
    if not classify_py.is_file() or not postprocess_py.is_file():
        return raw_cutlist
    # Step 1: ensure whisper.json exists (idempotent, returns 0 if fresh)
    try:
        r = subprocess.run(
            [WHISPER_PYTHON, str(classify_py),
             "--src", str(src_path), "--output", str(whisper_path)],
            capture_output=True, timeout=300, env=SPAWN_ENV)
        if r.returncode != 0 or not whisper_path.is_file():
            err = r.stderr.decode(errors='replace') if r.stderr else ''
            out = r.stdout.decode(errors='replace') if r.stdout else ''
            print(f"  detect {uuid[:8]} whisper-classify rc={r.returncode}\n"
                  f"    stderr (last 600): {err[-600:]}\n"
                  f"    stdout (last 200): {out[-200:]}",
                  flush=True)
            return raw_cutlist
    except Exception as e:
        print(f"  detect {uuid[:8]} whisper-classify err: {e}", flush=True)
        return raw_cutlist
    # Push the whisper.json to the gateway so /search can index it.
    # Best-effort — failure here doesn't affect cutlist refinement.
    try:
        body = whisper_path.read_bytes()
        req = urllib.request.Request(
            f"{GATEWAY}/api/internal/whisper-reindex?uuid={uuid}",
            data=body, method="POST",
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10, context=CTX).read()
    except Exception as e:
        print(f"  detect {uuid[:8]} whisper-reindex push err: {e}",
              flush=True)
    # Step 2: pipe raw cutlist through postprocess
    try:
        diag_path = WHISPER_CACHE / f"{uuid}.postprocess.json"
        r = subprocess.run(
            [WHISPER_PYTHON, str(postprocess_py),
             "--whisper", str(whisper_path), "--diag-out", str(diag_path)],
            input=raw_cutlist.encode(),
            capture_output=True, timeout=30, env=SPAWN_ENV)
        if r.returncode != 0 or not r.stdout:
            return raw_cutlist
        # Diagnostic log line — only when something actually changed
        try:
            d = json.loads(diag_path.read_text())
            n_rm = len(d.get("removed", []))
            n_ext = len(d.get("extended", []))
            n_add = len(d.get("added", []))
            if n_rm or n_ext or n_add:
                print(f"  detect {uuid[:8]} whisper-refined: "
                      f"−{n_rm}fp, ⤇{n_ext}ext, +{n_add}new",
                      flush=True)
        except Exception:
            pass
        return r.stdout.decode(errors="replace")
    except Exception as e:
        print(f"  detect {uuid[:8]} whisper-postprocess err: {e}",
              flush=True)
        return raw_cutlist


def _maybe_evict_source_cache():
    files = sorted(SOURCE_CACHE.glob("*.ts"),
                   key=lambda p: p.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    cap = SOURCE_CACHE_MAX_GB * 1024 ** 3
    while total > cap and files:
        oldest = files.pop(0)
        # Sole-copy guard: if Pi no longer has the source (= drop-pi-
        # source already fired for this uuid), the T7 cache is the
        # ONLY remaining .ts. Evicting it loses the recording's
        # source forever — re-extraction during the next train cycle
        # then drops the recording and silently shrinks the corpus.
        # 404 from /recording/<uuid>/source means dual-existence is
        # already broken; skip evict and try the next-oldest dup.
        uuid = oldest.stem
        try:
            req = urllib.request.Request(
                f"{GATEWAY}/recording/{uuid}/source", method="HEAD")
            with urllib.request.urlopen(req, timeout=4, context=CTX) as r:
                pi_has = r.status == 200
        except urllib.error.HTTPError as e:
            pi_has = (e.code != 404)
        except Exception:
            # Gateway unreachable → don't risk evicting; pause cycle.
            break
        if not pi_has:
            continue  # protected — try older dups instead
        sz = oldest.stat().st_size
        try: oldest.unlink(); total -= sz
        except Exception: pass


_last_orphan_gc = 0.0
ORPHAN_GC_INTERVAL_S = 3600  # once per hour
# Quarantine grace before deleting a cached .ts that's gone from the Pi's
# recording-uuids list. For a disk-dedup'd recording the cache is the ONLY
# copy (Pi .ts pruned), so an immediate delete on a TRANSIENT list-gap — a
# tv-receiver restart mid-prune, or a wrong prune later corrected by
# re-import — is unrecoverable. With a grace window the cache survives until
# the recording is genuinely, durably gone. (Cost: a truly-deleted recording's
# cache lingers up to GRACE; LRU still reclaims it under cap pressure.)
ORPHAN_GRACE_S = 36 * 3600
ORPHAN_PENDING = MODEL_CACHE / "orphan-pending.json"  # uuid -> first-seen-orphaned epoch (local, survives T7 unmount)

# Source-cache prefetch — fills SOURCE_CACHE with .ts files for
# recordings the user hasn't viewed/redetected recently. Without this,
# train-head.py falls back to SMB for the long tail of older reviewed
# recordings (~46/161 in this corpus). Cycle params chosen to keep
# wall-clock impact gentle:
#   * 1800 s between cycles = at most ~25 GB/h pulled (5 files × 5 GB)
#   * 3 parallel HTTP fetches = saturates Gigabit cleanly while
#     leaving headroom for live HLS streaming
#   * skip when ANY detect job in flight (= they have priority on
#     bandwidth + Pi disk IO)
#   * skip when SOURCE_CACHE is within 10 GB of cap (LRU would just
#     evict our fresh prefetches; let the active workload manage)
PREFETCH_INTERVAL_S = 1800
PREFETCH_PARALLEL   = 3
PREFETCH_PER_CYCLE  = 5
PREFETCH_HEADROOM_GB = 10
DROP_SWEEP_PER_CYCLE = 25   # re-attempt drop-pi-source for N aged cached recs/cycle
_last_prefetch = 0.0
# Layer-3 integrity sweep: periodically report the Mac-side health signals
# (stale source caches, orphan-pending) to the Pi's /api/integrity. The
# get_source guard FIXES stale caches reactively; this COUNTS them proactively
# so the dashboard/HA can flag accumulation before a nightly trains on them.
INTEGRITY_INTERVAL_S = 1800
_last_integrity = 0.0
# Host-stats push for the app's parent-diagnostics panel: cpu/disk every 30s to
# the Pi's /api/internal/host-stats (merged into GET /api/status/host). One Mac
# process (this daemon) instead of a separate launchd agent.
HOST_STATS_INTERVAL_S = 30
_last_host_stats = 0.0
# uuids that 404 on /source (= no raw .ts on Pi; HLS-VOD-only orphan).
# Skipped in prefetch so we stop re-probing them every cycle.
_known_orphans = set()
# uuids confirmed dropped (or permanently un-droppable). Skipped in the
# drop-sweep so it advances through the backlog instead of re-hitting them.
_dropped_uuids = set()


def _maybe_gc_orphans():
    """Drop cached .ts whose recording has been deleted on the Pi.
    LRU alone would evict eventually but only when the cap is hit;
    explicit orphan removal frees space sooner and keeps the cache
    aligned with Pi-side reality. Runs at most once per hour.

    Also triggers Pi-side cleanup of orphaned _rec_<uuid>/ dirs
    (= bundle-folders whose tvh DVR entry was deleted but the dir
    stayed behind, leaving thumbs/.requested in an infinite retry
    loop) and clears the local whisper-cache for deleted uuids."""
    global _last_orphan_gc
    now = time.time()
    if now - _last_orphan_gc < ORPHAN_GC_INTERVAL_S:
        return
    _last_orphan_gc = now
    try:
        valid = set(http_get_json(
            f"{GATEWAY}/api/internal/recording-uuids").get("uuids", []))
    except Exception as e:
        print(f"  orphan-gc: pi unreachable: {e}", flush=True); return
    if not valid:
        return  # don't wipe the cache if Pi returned nothing (= safety)
    # Quarantine ledger: a cache is only deleted once it's been continuously
    # orphaned for ORPHAN_GRACE_S. A uuid that reappears in `valid` (e.g. was
    # re-imported) is forgiven and never deleted.
    try:
        pending = json.loads(ORPHAN_PENDING.read_text())
        if not isinstance(pending, dict):
            pending = {}
    except Exception:
        pending = {}
    n_removed = 0
    bytes_removed = 0
    quarantined = 0
    next_pending = {}
    for f in SOURCE_CACHE.glob("*.ts"):
        uuid = f.stem
        if uuid in valid:
            continue  # valid → forgiven (drops out of the ledger)
        first = pending.get(uuid, now)  # first time orphaned → start the clock now
        if now - first < ORPHAN_GRACE_S:
            next_pending[uuid] = first  # still in grace → keep, don't delete
            quarantined += 1
            continue
        try:  # orphaned past the grace window → genuinely gone, reclaim
            bytes_removed += f.stat().st_size
            f.unlink()
            n_removed += 1
        except Exception:
            next_pending[uuid] = first
    try:
        ORPHAN_PENDING.write_text(json.dumps(next_pending))
    except Exception:
        pass
    if n_removed:
        print(f"  orphan-gc: removed {n_removed} cached .ts "
              f"({bytes_removed/1e6:.0f} MB)", flush=True)
    if quarantined:
        print(f"  orphan-gc: {quarantined} orphaned .ts in grace "
              f"({ORPHAN_GRACE_S // 3600}h before delete)", flush=True)
    # Whisper-cache: each recording leaves a <uuid>.whisper.json +
    # <uuid>.postprocess.json. Clean both when the recording's gone.
    whisper_dir = Path.home() / ".cache" / "tv-whisper"
    if whisper_dir.is_dir():
        n_w = 0
        for f in whisper_dir.glob("*.json"):
            uuid = f.name.split(".", 1)[0]
            if uuid not in valid and len(uuid) >= 16:
                try:
                    f.unlink()
                    n_w += 1
                except Exception:
                    pass
        if n_w:
            print(f"  orphan-gc: removed {n_w} whisper-cache files",
                  flush=True)
    # Pi-side: trigger HLS-bundle cleanup for deleted DVR entries.
    # Gateway diffs against tvh's grid + only deletes _rec_/ dirs
    # whose mtime is >1h old (= avoid race with in-flight finalize).
    try:
        req = urllib.request.Request(
            f"{GATEWAY}/api/internal/cleanup-orphans", method="POST")
        r = json.loads(urllib.request.urlopen(
            req, timeout=30, context=CTX).read())
        if r.get("n_removed"):
            print(f"  orphan-gc: pi removed {r['n_removed']} _rec_/ dirs",
                  flush=True)
    except Exception as e:
        print(f"  orphan-gc: pi cleanup err: {e}", flush=True)


def _maybe_prefetch_sources(in_flight_n):
    """Background-fill SOURCE_CACHE with recordings train-head would
    otherwise have to read via SMB. Skipped when any detect job is in
    flight or the cache is already near cap."""
    global _last_prefetch
    if in_flight_n > 0:
        return
    now = time.time()
    if now - _last_prefetch < PREFETCH_INTERVAL_S:
        return
    try:
        valid = http_get_json(
            f"{GATEWAY}/api/internal/recording-uuids").get("uuids", [])
    except Exception as e:
        print(f"  prefetch: pi unreachable: {e}", flush=True); return
    if not valid:
        return
    _last_prefetch = now
    cache_files = list(SOURCE_CACHE.glob("*.ts"))
    cached = {p.stem for p in cache_files}
    # Drop-sweep FIRST — runs regardless of cache fullness, since deleting
    # Pi-original .ts files frees the Pi disk without touching the T7 cache.
    # This catches recordings whose drop was skipped at cache time (too
    # fresh) and never retried once they aged past the gateway's 3-day gate.
    _drop_sweep(cached)
    # Prefetch only if the T7 cache has headroom.
    used_b = sum(f.stat().st_size for f in cache_files)
    if used_b > (SOURCE_CACHE_MAX_GB - PREFETCH_HEADROOM_GB) * 1024**3:
        return
    # Skip known HLS-VOD-only orphans (404 on /source) so we don't re-probe
    # recordings whose raw .ts is gone every cycle.
    missing = [u for u in valid if u not in cached and u not in _known_orphans]
    if not missing:
        return
    todo = missing[:PREFETCH_PER_CYCLE]
    print(f"  prefetch: {len(missing)} uncached on pi, fetching "
          f"{len(todo)} ({PREFETCH_PARALLEL} parallel)", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PREFETCH_PARALLEL) as ex:
        list(ex.map(get_source, todo))
    print(f"  prefetch: cycle done in {time.time()-t0:.0f}s", flush=True)


def _maybe_integrity_sweep():
    """Report the Mac-side health half to the Pi (Layer-3 integrity). Counts
    stale source caches (Pi /source size != cached size = re-filtered, the
    get_source guard's reactive fix made visible proactively) + orphan-pending
    + cache stats, POSTed to /api/integrity. Throttled to once per
    INTEGRITY_INTERVAL_S; best-effort (never blocks the work loop)."""
    global _last_integrity
    now = time.time()
    if now - _last_integrity < INTEGRITY_INTERVAL_S:
        return
    _last_integrity = now
    try:
        files = list(SOURCE_CACHE.glob("*.ts"))
    except Exception:
        return
    total = 0
    for f in files:
        try: total += f.stat().st_size
        except Exception: pass

    # Bulk size map (one request) instead of one HEAD per cached uuid — the
    # per-uuid HEADs hit every dedup-dropped recording with a 404 twice an
    # hour, forever (Caddy-log noise). A uuid absent from the map = Pi has no
    # source (dropped / in-progress) = not stale, same verdict the old
    # 404→None path produced.
    pi_sizes = None
    try:
        with urllib.request.urlopen(
                f"{GATEWAY}/api/internal/source-sizes",
                timeout=30, context=CTX) as r:
            pi_sizes = json.loads(r.read()).get("sizes", {})
    except Exception:
        pi_sizes = None  # old recorder / unreachable → per-uuid fallback

    def _is_stale(f):
        try:
            cur = f.stat().st_size
            if pi_sizes is not None:
                ps = pi_sizes.get(f.stem)
            else:
                ps = _pi_source_size(f.stem)
            return ps is not None and abs(ps - cur) > 8192
        except Exception:
            return False

    stale = 0
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            stale = sum(1 for r in ex.map(_is_stale, files) if r)
    except Exception:
        pass
    pend = 0
    try:
        pend = len(json.loads(ORPHAN_PENDING.read_text()))
    except Exception:
        pass
    report = {
        "stale_source_caches": stale,
        "orphan_pending": pend,
        "source_cache_count": len(files),
        "source_cache_gb": round(total / 1e9, 1),
    }
    try:
        req = urllib.request.Request(
            f"{GATEWAY}/api/internal/integrity-report",
            data=json.dumps(report).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15, context=CTX).close()
        print(f"  integrity: {stale} stale, {pend} orphan-pending, "
              f"{len(files)} cached ({total/1e9:.0f}GB) → reported", flush=True)
    except Exception as e:
        print(f"  integrity report err: {e}", flush=True)


def _cpu_util_pct():
    """Real CPU utilization (100 − idle%) from a single `top` sample. macOS has
    no /proc/stat and we avoid psutil, so parse top's "CPU usage:" line:
      "CPU usage: 54.46% user, 9.53% sys, 36.00% idle"
    Returns None on any failure → caller falls back to the load proxy. This is
    UTILIZATION (how busy the cores are), distinct from load average (how much
    work is queued) — a detect batch spikes load past core-count while a third of
    the CPU is still idle, so load-as-percent cried "100% am Limit" falsely."""
    try:
        out = subprocess.run(["top", "-l", "1", "-n", "0"],
                             capture_output=True, text=True, timeout=8).stdout
        for line in out.splitlines():
            if "CPU usage" in line:
                idle = float(line.split("% idle")[0].split()[-1])
                return round(max(0.0, min(100.0, 100.0 - idle)), 1)
    except Exception:
        pass
    return None


def _maybe_host_stats():
    """Push Mac CPU/disk to the Pi every 30s for the app's host-stats panel.
    cpu_pct = real utilization (top), with a load-average fallback; cpu_load1 +
    cpu_count let the app show queue saturation separately. disk = the Mac's
    root volume. Best-effort; never blocks the work loop."""
    global _last_host_stats
    now = time.time()
    if now - _last_host_stats < HOST_STATS_INTERVAL_S:
        return
    _last_host_stats = now
    try:
        import shutil
        load1 = os.getloadavg()[0]
        cnt = os.cpu_count() or 1
        du = shutil.disk_usage("/")
        util = _cpu_util_pct()
        if util is None:  # top unavailable — fall back to the (coarse) load proxy
            util = round(min(100.0, load1 / cnt * 100.0), 1)
        report = {
            "cpu_load1": round(load1, 2),
            "cpu_count": cnt,
            "cpu_pct": util,
            "disk_used_gb": round(du.used / 1e9, 1),
            "disk_total_gb": round(du.total / 1e9, 1),
        }
        req = urllib.request.Request(
            f"{GATEWAY}/api/internal/host-stats",
            data=json.dumps(report).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10, context=CTX).close()
    except Exception:
        pass


def _drop_pi_source(uuid):
    """Tell gateway it can delete the Pi-original .ts for a T7-cached
    recording. Gateway has its own safety guards (sched_status==completed,
    age>72h, HLS-VOD exists). Returns a status the drop-sweep uses to
    decide whether to retry:
      'dropped' — .ts deleted (returns freed MB via the print)
      'done'    — nothing to do (already gone / unknown uuid); stop retrying
      'retry'   — guard fired (too fresh / in progress); eligible later
      'error'   — transient failure; retry later
    Best-effort — never raises."""
    try:
        req = urllib.request.Request(
            f"{GATEWAY}/api/internal/drop-pi-source/{uuid}",
            method="POST")
        with urllib.request.urlopen(req, timeout=10, context=CTX) as r:
            resp = json.loads(r.read())
        if resp.get("ok"):
            freed_mb = resp.get("deleted_bytes", 0) / 1e6
            print(f"  drop-pi-source {uuid[:8]}: freed {freed_mb:.0f} MB "
                  f"on Pi (age={resp.get('age_h')}h)", flush=True)
            return "dropped"
        # ok:False at HTTP 200 = already gone / no filename → nothing to retry
        return "done"
    except urllib.error.HTTPError as e:
        # 409 = guard fired (too fresh / recording in progress / no HLS yet).
        # Normal for new recordings — becomes eligible once they age.
        if e.code == 409:
            return "retry"
        if e.code == 404:           # unknown uuid → will never drop
            return "done"
        print(f"  drop-pi-source {uuid[:8]}: HTTP {e.code}", flush=True)
        return "error"
    except Exception as e:
        print(f"  drop-pi-source {uuid[:8]}: err {e}", flush=True)
        return "error"


def _drop_sweep(cached):
    """Re-attempt drop-pi-source for already-T7-cached recordings whose
    Pi-original .ts may still be sitting on the Pi. The drop is fired once
    at cache time (in get_source), but if the recording was <3 days old
    then the gateway's age-gate returned 409 and nothing ever retried it —
    so the .ts accumulates forever. This sweeps a batch per cycle; the
    gateway's guards still decide eligibility. Sorted + _dropped_uuids
    bookkeeping make it advance through the backlog without re-hitting
    already-handled recordings."""
    candidates = sorted(cached - _dropped_uuids)
    if not candidates:
        return
    batch = candidates[:DROP_SWEEP_PER_CYCLE]
    dropped = 0
    for u in batch:
        status = _drop_pi_source(u)
        if status in ("dropped", "done"):
            _dropped_uuids.add(u)
            if status == "dropped":
                dropped += 1
        # 'retry'/'error' stay candidates for a future cycle.
    if dropped:
        print(f"  drop-sweep: dropped {dropped} aged Pi-source(s); "
              f"{len(candidates) - len(batch)} cached still to check",
              flush=True)


# train-head's derived caches (same Mac) — must be invalidated together with a
# stale source, else the nightly re-uses wrong-channel features/labels. See
# memory recovery_churn_poisons_training.
TVD_FEATURES = Path.home() / ".cache" / "tvd-features"
TVD_ARCHIVE  = Path.home() / ".cache" / "tvd-train-archive"


def _pi_source_size(uuid):
    """HEAD the Pi's CURRENT /source and return its Content-Length, or None if
    the size can't be determined (Pi unreachable / 404 dropped / 425 recording /
    no header). None means 'unknown' → caller must NOT evict (fail-safe: a Pi
    hiccup never wipes the cache)."""
    try:
        req = urllib.request.Request(
            f"{GATEWAY}/recording/{uuid}/source", method="HEAD")
        with urllib.request.urlopen(req, timeout=10, context=CTX) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


def _invalidate_derived(uuid):
    """Drop stale feature .npy + archive .npz for a uuid so the nightly
    re-extracts from the fresh source instead of training on wrong-channel
    features (the cache key is mtime-based, but the archive pins the old .npy
    path, so both must go). Returns count removed."""
    n = 0
    try:
        for p in TVD_FEATURES.glob(f"{uuid}-*.npy"):
            try: p.unlink(); n += 1
            except Exception: pass
    except Exception: pass
    z = TVD_ARCHIVE / f"{uuid}.npz"
    try:
        if z.exists(): z.unlink(); n += 1
    except Exception: pass
    return n


def get_source(uuid):
    """Return local .ts path. Cached: serve from disk. Cold: HTTP-fetch
    + cache for next time. Falls back to None on any error — caller
    should use the HTTP URL directly as a last resort."""
    cache_path = SOURCE_CACHE / f"{uuid}.ts"
    if cache_path.exists() and cache_path.stat().st_size > 100_000_000:
        # Source-freshness guard: if the Pi re-filtered/trimmed this recording,
        # its .ts content (and size) changed but our cache still holds the OLD
        # video → stale wrong-channel features poison the nightly. Verify the
        # cached size against the Pi's current source; on a CONFIRMED mismatch
        # (>8 KB, same slack as the truncation check), evict the cache + the
        # derived feature/archive caches and fall through to re-fetch. Fail-safe:
        # a None size (unreachable/404/425) keeps the cache untouched.
        cur = cache_path.stat().st_size
        pi_size = _pi_source_size(uuid)
        if pi_size is not None and abs(pi_size - cur) > 8192:
            print(f"  source {uuid[:8]}: changed on Pi "
                  f"({cur}->{pi_size} bytes) — evicting stale cache + features",
                  flush=True)
            try: cache_path.unlink()
            except Exception: pass
            ninv = _invalidate_derived(uuid)
            if ninv:
                print(f"  invalidated {ninv} derived cache file(s) "
                      f"for {uuid[:8]}", flush=True)
            # fall through to the cold-fetch path below
        else:
            try: cache_path.touch()  # update atime for LRU
            except Exception: pass
            return cache_path
    if cache_path.exists():
        try: cache_path.unlink()  # stub from a half-finished fetch
        except Exception: pass
    src_url = f"{GATEWAY}/recording/{uuid}/source"
    # Unique tmp per fetch. get_source runs from BOTH the prefetch ThreadPool
    # and the detect/hls executors, which can target the same uuid at once. A
    # shared "{uuid}.tmp" let them race on open()/rename() — on the T7's
    # FSKit-exfat driver the loser surfaced as EPERM/ENOENT ("cache-fill err:
    # Operation not permitted"). Give each fetch its own tmp via pid+thread id.
    # NB: do NOT use tempfile.mkstemp here — its O_EXCL + 0o600 open raises
    # EPERM on the FSKit exfat driver; the plain open(tmp,"wb") below works.
    tmp = cache_path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
    t0 = time.time()
    try:
        with urllib.request.urlopen(src_url, timeout=600,
                                       context=CTX) as r:
            expected = r.headers.get("Content-Length")
            expected = int(expected) if expected and expected.isdigit() else None
            with open(tmp, "wb") as f:
                import shutil
                shutil.copyfileobj(r, f, length=1 << 20)
        # Content-Length sanity: if the gateway sent a length and we
        # have fewer bytes, the connection died mid-stream and the .ts
        # is truncated. Don't cache it — caller falls back to HTTP URL
        # or the next cycle retries. Tolerate <8 KB delta (= TCP
        # tail-loss / final-segment slack).
        actual = tmp.stat().st_size
        if expected is not None and expected - actual > 8192:
            tmp.unlink()
            print(f"  cache-fill TRUNCATED {uuid[:8]}: "
                  f"{actual}/{expected} bytes "
                  f"({100*actual/expected:.1f}%), discarded", flush=True)
            return None
        try:
            os.replace(tmp, cache_path)  # atomic
        except OSError:
            # A concurrent fetch of the same uuid won the race and already
            # produced a valid cache file — reuse it, drop our tmp.
            if cache_path.exists() and cache_path.stat().st_size > 100_000_000:
                try: tmp.unlink()
                except Exception: pass
                return cache_path
            raise
        size_mb = cache_path.stat().st_size / 1e6
        print(f"  cached {uuid[:8]} ({size_mb:.0f} MB in "
              f"{time.time()-t0:.0f}s)", flush=True)
        _maybe_evict_source_cache()
        # T7 has it now → ask gateway to dedup the Pi-original .ts.
        # Gateway's safety guards (age, HLS-VOD presence, sched_status)
        # decide eligibility; daemon just signals "we have it cached".
        _drop_pi_source(uuid)
        return cache_path
    except urllib.error.HTTPError as e:
        try: tmp.unlink()
        except Exception: pass
        if e.code == 425:
            # tvh still writing — gateway guard says retry later. Set
            # cooldown so we don't hammer, but DO NOT tick the persistent
            # retry counter: this is a transient state, not a broken
            # recording. After cooldown the next cycle re-tries and will
            # succeed once sched_status flips to completed.
            print(f"  source {uuid[:8]}: recording in progress (425), "
                  f"cooldown {FAIL_COOLDOWN_S}s", flush=True)
            _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
            return None
        if e.code == 404:
            # No raw .ts on the Pi for this uuid: HLS-VOD-only orphan (the
            # original was already dropped and never cached here). Nothing
            # to fetch — remember it so the prefetch loop stops re-probing
            # it every cycle. Logged once; silent thereafter.
            if uuid not in _known_orphans:
                _known_orphans.add(uuid)
                print(f"  source {uuid[:8]}: 404 — HLS-VOD-only orphan, "
                      f"skipping in future prefetch", flush=True)
            return None
        print(f"  cache-fill err: HTTP {e.code} {e.reason}", flush=True)
        return None
    except Exception as e:
        print(f"  cache-fill err: {e}", flush=True)
        try: tmp.unlink()
        except Exception: pass
        return None

CTX = ssl.create_default_context()
CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

# Client-side cooldown so a single broken recording doesn't burn
# the daemon in a tight retry loop. After ffmpeg fails on a uuid,
# we skip it for FAIL_COOLDOWN_S; the Pi-fallback timer in
# _rec_hls_spawn picks it up locally during that window. After
# cooldown we try again (transient failures self-heal).
FAIL_COOLDOWN_S = 600  # 10 min
_failed_until = {}  # uuid -> unix_ts when we may retry

# Persistent failure-counter file. Survives daemon restart, unlike the
# in-memory _failed_until cooldown which resets on relaunch and lets a
# broken recording immediately retry → infinitely block the queue.
# Write atomically (= tmp + rename) so a kill mid-write can't corrupt.
DETECT_RETRY_FILE = MODEL_CACHE / "detect-retries.json"
MAX_DETECT_RETRIES = 3   # 3 attempts then give up + clear marker via gateway


def _load_detect_retries():
    try:
        return json.loads(DETECT_RETRY_FILE.read_text())
    except Exception:
        return {}


def _save_detect_retries(d):
    try:
        tmp = DETECT_RETRY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(DETECT_RETRY_FILE)
    except Exception:
        pass


def _record_detect_failure(uuid, force=False):
    """Increment the persistent retry count. Returns the new count.
    When count reaches MAX_DETECT_RETRIES (or force=True for an
    unrecoverable failure like a corrupt source), POSTs to the gateway's
    detect-give-up endpoint to clear both markers — without that the
    daemon's low-prio queue (= line ~956: only fetched when high-prio
    empty) can stay blocked indefinitely on a single broken recording."""
    retries = _load_detect_retries()
    n = retries.get(uuid, 0) + 1
    retries[uuid] = n
    _save_detect_retries(retries)
    if force or n >= MAX_DETECT_RETRIES:
        print(f"  detect {uuid[:8]}: giving up after {n} failure(s)"
              f"{' (unrecoverable)' if force else ''}, "
              f"asking gateway to clear marker", flush=True)
        try:
            req = urllib.request.Request(
                f"{GATEWAY}/api/internal/detect-give-up/{uuid}",
                method="POST")
            urllib.request.urlopen(req, timeout=10, context=CTX).read()
            retries.pop(uuid, None)
            _save_detect_retries(retries)
        except Exception as e:
            print(f"  detect-give-up err: {e}", flush=True)
    return n


def _record_detect_success(uuid):
    """Drop any persisted retry count after a successful detect (=
    next failure starts the counter from 0 again)."""
    retries = _load_detect_retries()
    if uuid in retries:
        retries.pop(uuid, None)
        _save_detect_retries(retries)


def http_get_json(url):
    with urllib.request.urlopen(url, timeout=15, context=CTX) as r:
        return json.loads(r.read())


def http_post_stream(url, fileobj, content_type="application/x-tar",
                       size=None):
    """POST a file-like object as the request body. Streams the upload
    so we don't hold the whole tar in RAM (HLS bundles can be 1-3 GB)."""
    headers = {"Content-Type": content_type}
    if size is not None:
        headers["Content-Length"] = str(size)
    req = urllib.request.Request(url, data=fileobj, method="POST",
                                  headers=headers)
    with urllib.request.urlopen(req, timeout=600, context=CTX) as r:
        return r.read()


def http_download(url, dest_path):
    """Download URL → file. Refreshes only if the remote
    Content-Length differs from the local size — head.bin nightly
    retrain bumps size when feature dim changes (5128 ↔ 5152 B)
    and content otherwise; for backbone.onnx the size never moves."""
    headers = {}
    if dest_path.exists():
        # Cheap check: HEAD request, compare size
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                remote_size = int(r.headers.get("Content-Length", 0))
            if remote_size > 0 and remote_size == dest_path.stat().st_size:
                return  # cache hit
        except Exception:
            pass
    with urllib.request.urlopen(url, timeout=120, context=CTX) as r:
        dest_path.write_bytes(r.read())


def _upload_tar(url, files, arcname_fn=None):
    """Tar `files` in-memory, POST to `url`. Used for thumbs (small
    bundles, ~600 KB each). HLS bundles use _upload_files_put
    instead — same wire-time, no Pi-side tar parser."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in files:
            arcname = arcname_fn(f) if arcname_fn else f.name
            tf.add(str(f), arcname=arcname)
    buf.seek(0)
    http_post_stream(url, buf.read())


def _upload_files_put(url_template, files):
    """PUT each file to `url_template.format(name=...)` over a single
    keep-alive HTTP connection. Sends raw bytes — Pi just streams to
    disk, no Python-loop parsing → no CPU spike on the gateway side.

    Throughput on gigabit ~110 MB/s sustained for back-to-back PUTs;
    HTTPSConnection reuses the TLS session so per-file overhead is
    just headers + ack (~1-2 ms)."""
    import http.client, urllib.parse
    parsed = urllib.parse.urlparse(url_template.format(name=""))
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(parsed.hostname, parsed.port,
                                             timeout=120, context=CTX)
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                            timeout=120)
    try:
        for f in files:
            url_path = urllib.parse.urlparse(
                url_template.format(name=urllib.parse.quote(f.name))
            ).path
            size = f.stat().st_size
            with open(f, "rb") as fh:
                conn.request("PUT", url_path, body=fh,
                              headers={"Content-Length": str(size),
                                        "Content-Type": "application/octet-stream"})
                r = conn.getresponse()
                r.read()  # drain
                if r.status >= 300:
                    raise RuntimeError(
                        f"PUT {f.name} → {r.status}")
    finally:
        conn.close()


def process_recording(uuid, do_hls, do_thumbs):
    """Combined job: single ffmpeg pass produces HLS bundle AND
    thumbs from one .ts download. Source resolved via local cache
    first (NVMe ~3 GB/s); HTTP fallback only on cache miss."""
    local = get_source(uuid)
    src_url = str(local) if local else f"{GATEWAY}/recording/{uuid}/source"
    # Stub-source short-circuit: ffmpeg can't extract thumbnails from
    # a 500KB-1MB broken recording (tuner cut mid-stream), produces 0
    # jpgs, daemon retried forever (= 2026-05-18+19 incidents). Probe
    # source size via HTTP HEAD (cheap, ~50ms) and skip with cooldown
    # if <2 MB. Recordings legitimately that small don't exist — even
    # 1 min of DVB-C ~25 Mbit/s is 180 MB.
    try:
        if local:
            src_size = local.stat().st_size
        else:
            import urllib.request
            with urllib.request.urlopen(
                    urllib.request.Request(src_url, method="HEAD"),
                    timeout=5, context=CTX) as r:
                src_size = int(r.headers.get("Content-Length") or 0)
        if 0 < src_size < 2 * 1024 * 1024:
            print(f"  skipping {uuid[:8]}: source is stub "
                  f"({src_size} bytes, recording likely cut mid-stream)",
                  flush=True)
            _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
            return False
    except Exception:
        pass  # probe failed → fall through to ffmpeg, normal failure path will catch
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
               "-i", src_url]
        if do_hls:
            try:
                probe = subprocess.run(
                    [FFPROBE, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name", "-of",
                     "default=nokey=1:noprint_wrappers=1", src_url],
                    capture_output=True, text=True, timeout=30,
                    env=SPAWN_ENV)
                vcodec = (probe.stdout or "").strip()
            except Exception:
                vcodec = ""
            if vcodec in SAFE_VIDEO:
                v_opts = ["-c:v", "copy"]
            else:
                # libx264 ultrafast on M5 Pro encodes 90× realtime
                # (a 90-min recording finalises in ~1 min wallclock)
                # at the cost of bursting to ~8 cores. Tested
                # h264_videotoolbox 2026-04 — 5× less CPU but 3× SLOWER
                # wallclock. Wallclock wins for snappier "play-now"
                # availability after recording ends.
                # Breite auf 16er-Vielfache runden (statt nur gerade): der
                # iOS-VT-Super-Resolution-Scaler verlangt 16-aligned Breiten
                # im process (Mac-Probe + A19-Log 2026-06-10/11) — bei z.B.
                # 1046 px musste die App-Bridge croppen+kopieren; mit 1040
                # trifft der Decoder-Buffer die VT-Session exakt (Zero-Copy-
                # Direct-Pass). AR-Fehler durch round ≤ ~0,8 %, unsichtbar.
                v_opts = ["-vf",
                          "scale=round(iw*sar/16)*16:trunc(ih/2)*2,setsar=1",
                          "-c:v", "libx264", "-preset", "ultrafast",
                          "-profile:v", "main", "-pix_fmt", "yuv420p",
                          "-g", "50"]
            cmd += [
                "-map", "0:v:0", "-map", "0:a:0?",
                *v_opts, "-c:a", "aac", "-b:a", "128k",
                "-f", "hls",
                "-hls_time", "6",
                "-hls_list_size", "0",
                "-hls_playlist_type", "event",
                "-hls_base_url", f"/hls/_rec_{uuid}/",
                "-hls_segment_filename", str(td_p / "seg_%05d.ts"),
                str(td_p / "index.m3u8")]
        if do_thumbs:
            cmd += [
                "-map", "0:v:0",
                "-vf", "fps=1/30,scale=160:-2",
                "-q:v", "6",
                str(td_p / "t%05d.jpg")]
        t0 = time.time()
        # Stream HLS segments to the Pi WHILE ffmpeg encodes, instead of
        # batch-uploading after it finishes — so /progress's segment count grows
        # throughout the remux (the app shows a smooth bar instead of 0% then a
        # jump) and the upload overlaps the encode. Only segments are streamed;
        # the playlist is uploaded LAST (after ffmpeg), so the Pi flips to
        # "ready" only when the VOD is complete. ffmpeg stderr → a temp file so
        # polling never deadlocks a PIPE. The final batch below re-uploads
        # anything the stream missed, so this can only help, never lose a seg.
        seg_url = f"{GATEWAY}/api/internal/hls-segment/{uuid}/{{name}}"
        uploaded = set()
        errf = open(td_p / "ffmpeg.stderr", "w+")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=errf, env=SPAWN_ENV)
        while True:
            rc = proc.poll()
            if do_hls:
                segs = sorted(td_p.glob("seg_*.ts"))
                # While running, the highest-numbered segment is still being
                # written → stream all but the last; after exit all are done.
                ready = segs if rc is not None else segs[:-1]
                new = [s for s in ready if s.name not in uploaded]
                if new:
                    try:
                        _upload_files_put(seg_url, new)
                        uploaded.update(s.name for s in new)
                    except Exception as e:
                        print(f"  hls stream err: {e}", flush=True)
            if rc is not None:
                break
            if time.time() - t0 > TIMEOUT_S:
                proc.kill(); proc.wait(); rc = -9
                break
            time.sleep(1.0)
        errf.seek(0); err_tail = errf.read()[-500:]; errf.close()
        encode_s = time.time() - t0
        if rc != 0:
            print(f"  ffmpeg {uuid[:8]} rc={rc} — last stderr:\n"
                  f"{err_tail}", flush=True)
            _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
            return False
        ok_hls = ok_thumbs = True
        if do_hls:
            playlist = td_p / "index.m3u8"
            if not playlist.exists():
                print(f"  no playlist for {uuid[:8]}", flush=True)
                ok_hls = False
            else:
                # Upload anything the stream missed (the in-flight last segment)
                # + the playlist LAST (readiness signal) + hls-done.
                segs = sorted(td_p.glob("seg_*.ts"))
                remaining = [s for s in segs if s.name not in uploaded]
                size_mb = sum(f.stat().st_size for f in segs) / 1e6
                t1 = time.time()
                try:
                    if remaining:
                        _upload_files_put(seg_url, remaining)
                    _upload_files_put(seg_url, [playlist])
                    http_post_stream(
                        f"{GATEWAY}/api/internal/hls-done/{uuid}",
                        b"")
                    print(f"  hls {uuid[:8]}: {len(segs)+1} files "
                          f"({size_mb:.0f} MB), encode {encode_s:.0f}s "
                          f"+ tail {time.time()-t1:.0f}s", flush=True)
                except Exception as e:
                    print(f"  hls upload err: {e}", flush=True); ok_hls = False
        if do_thumbs:
            jpgs = sorted(td_p.glob("t*.jpg"))
            if not jpgs:
                print(f"  no jpgs for {uuid[:8]}", flush=True)
                ok_thumbs = False
            else:
                try:
                    _upload_tar(
                        f"{GATEWAY}/api/internal/thumbs-uploaded/{uuid}",
                        jpgs)
                    print(f"  thumbs {uuid[:8]}: {len(jpgs)} jpgs",
                          flush=True)
                except Exception as e:
                    print(f"  thumbs upload err: {e}", flush=True)
                    ok_thumbs = False
        # ffmpeg-rc=0 but post-processing failed (no jpgs, upload err)
        # also needs cooldown — otherwise daemon spins on this uuid
        # every poll cycle (= the stub-thumbs retry loop from same
        # incidents above). The rc!=0 branch above already sets it.
        if not (ok_hls and ok_thumbs):
            _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
        return ok_hls and ok_thumbs


# Channels broadcast 16:9 movies in a 4:3 container with letterbox
# bars; the logo template (trained against the full frame) then sits
# in the top black bar instead of on the visible image. cropdetect
# finds the visible-content top edge — the logo sits ~20px above that
# (RTL-empirical for Jungle Cruise: cropdetect=80, optimal offset=60).
# May need per-channel calibration later; for now one constant.
LETTERBOX_LOGO_OVERHANG = 20


# Per-recording cropdetect cache. cropdetect runs ffmpeg for 5 s of
# decode at the 60 s mark — the underlying letterbox geometry is a
# property of the recording, not the detect parameters, so the result
# is stable across re-detects (= head deploys, manual /redetect
# triggers). Cache file = {uuid: y_offset_int} JSON; file content is
# the entire cache, atomic-rename on update. Cache is keyed by uuid
# (= source content hash). Survives daemon restart.
LETTERBOX_CACHE_FILE = MODEL_CACHE / "letterbox-offsets.json"


def _load_letterbox_cache() -> dict:
    try:
        return json.loads(LETTERBOX_CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_letterbox_cache(d: dict):
    try:
        tmp = LETTERBOX_CACHE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(LETTERBOX_CACHE_FILE)
    except Exception:
        pass


def detect_letterbox_offset(src, uuid=None):
    """Quick cropdetect on a 5s sample 60s into the recording (skips
    intro/promo). Returns recommended --logo-y-offset, or 0 if no
    meaningful letterbox. src may be a Path or HTTP URL.

    When uuid is supplied the result is cached per-uuid in
    ~/.cache/tv-detect-daemon/letterbox-offsets.json; subsequent
    detects of the same recording hit the cache instead of paying the
    5 s ffmpeg pass. Letterbox geometry is invariant w.r.t. the
    detect run (= depends on the recording's pixel content, not the
    head/template config), so this cache never needs explicit
    invalidation."""
    cache = _load_letterbox_cache() if uuid else {}
    if uuid and uuid in cache:
        return int(cache[uuid])
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "info",
             "-ss", "60", "-t", "5", "-i", str(src),
             "-vf", "cropdetect=24:16:0",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
            env=SPAWN_ENV)
    except Exception as e:
        print(f"  cropdetect err: {e}", flush=True)
        return 0
    # stderr lines look like: [...] crop=720:416:0:80
    # cropdetect refines the box over time — last match is the most
    # confident aggregate.
    ys = re.findall(r"crop=\d+:\d+:\d+:(\d+)", r.stderr)
    if not ys:
        offset = 0
    else:
        y = int(ys[-1])
        if y < 8:
            offset = 0  # no meaningful letterbox
        else:
            offset = max(0, y - LETTERBOX_LOGO_OVERHANG)
    if uuid:
        cache[uuid] = offset
        _save_letterbox_cache(cache)
    return offset


def _slugify_show(title: str) -> str:
    s = (title or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _ensure_speaker_artifacts(uuid: str, src_path: str, show_title: str):
    """Three-step pre-detect: extract embeddings, update show centroid,
    compute per-recording speaker.csv. Returns CSV path or None on any
    failure (so the caller can fall through to non-speaker detect).

    Cached aggressively: embeddings never re-extracted (audio doesn't
    change), centroid + csv re-computed each call (sub-second once
    embeddings exist) so newly user-edited episodes are picked up
    without explicit invalidation.
    """
    if not SPEAKER_ENABLE:
        return None
    show_slug = _slugify_show(show_title)
    if not show_slug:
        return None
    emb_path = EMB_CACHE / f"{uuid}.npz"
    centroid_path = CENTROID_CACHE / f"{show_slug}.npz"
    csv_path = SPK_CSV_CACHE / f"{uuid}.speaker.csv"

    # 1) Extract embeddings (once per recording)
    if not emb_path.exists():
        print(f"  detect {uuid[:8]}: extracting speaker embeddings "
              f"(~10-15min wallclock for 30min recording)", flush=True)
        t0 = time.time()
        try:
            r = subprocess.run(
                [SPEAKER_PYTHON,
                 str(SPEAKER_SCRIPTS / "extract-speaker-embeddings.py"),
                 src_path, str(emb_path)],
                capture_output=True, text=True, timeout=2400,
                env=SPAWN_ENV)
            if r.returncode != 0:
                # Show TAIL of stderr — the script logs progress to
                # stderr and the failure is at the END, but a 300-char
                # head truncation showed only the "loading audio from"
                # progress line, hiding the actual traceback.
                err_tail = (r.stderr or "")[-1500:]
                print(f"  detect {uuid[:8]}: embedding extract failed "
                      f"(rc={r.returncode}):\n{err_tail}", flush=True)
                return None
            print(f"  detect {uuid[:8]}: embeddings extracted in "
                  f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  detect {uuid[:8]}: embedding extract err: {e}",
                  flush=True)
            return None

    # 2) Rebuild show centroid from all available episodes
    try:
        r = subprocess.run(
            [SPEAKER_PYTHON,
             str(SPEAKER_SCRIPTS / "update-show-centroid.py"),
             show_slug, str(centroid_path),
             "--embeddings-dir", str(EMB_CACHE),
             "--gateway", GATEWAY,
             "--show-title", show_title],
            capture_output=True, text=True, timeout=120,
            env=SPAWN_ENV)
        if r.returncode == 2:
            # Insufficient data (no edited episodes yet) — expected for
            # cold-start shows. Run detect without speaker.
            return None
        if r.returncode != 0:
            print(f"  detect {uuid[:8]}: centroid build failed "
                  f"(rc={r.returncode}): {r.stderr[:300]}", flush=True)
            return None
    except Exception as e:
        print(f"  detect {uuid[:8]}: centroid build err: {e}", flush=True)
        return None

    # 3) Compute per-recording speaker.csv
    try:
        r = subprocess.run(
            [SPEAKER_PYTHON,
             str(SPEAKER_SCRIPTS / "compute-speaker-confs.py"),
             str(emb_path), str(centroid_path), str(csv_path)],
            capture_output=True, text=True, timeout=60,
            env=SPAWN_ENV)
        if r.returncode != 0:
            print(f"  detect {uuid[:8]}: speaker-csv compute failed "
                  f"(rc={r.returncode}): {r.stderr[:300]}", flush=True)
            return None
    except Exception as e:
        print(f"  detect {uuid[:8]}: speaker-csv err: {e}", flush=True)
        return None

    return csv_path


def process_detect(uuid):
    """Run tv-detect on a recording (HTTP-streamed source) and POST
    the cutlist back. Always passes --nn-backbone + --nn-head + the
    full per-channel knob set — Pi-side tv-detect doesn't currently
    pass NN flags so this offload also fixes the NN-not-actually-used
    bug in production detection."""
    cfg = http_get_json(f"{GATEWAY}/api/internal/detect-config/{uuid}")
    local = get_source(uuid)
    src_url = str(local) if local else f"{GATEWAY}{cfg['src_url']}"
    slug = cfg.get("channel_slug") or ""

    # Phase D optimisation: kick off the speaker-fingerprint pipeline
    # as a background thread RIGHT NOW so its 5-10 s (cached) /
    # 60-90 s (first-detect) work overlaps with the bumper + model
    # downloads + logo + cropdetect that follow. Sequential previously
    # added speaker-pipeline wallclock on top of the setup phase;
    # parallel absorbs it. Joined just before the tv-detect spawn
    # below — if speaker hasn't finished by then we still wait, but
    # in the typical case the future is already done.
    from concurrent.futures import ThreadPoolExecutor
    spk_executor = ThreadPoolExecutor(max_workers=1)
    spk_future = spk_executor.submit(
        _ensure_speaker_artifacts,
        uuid, src_url, cfg.get("show_title") or "")

    # Refresh model cache (small files, only re-downloads on size change)
    head_path = MODEL_CACHE / "head.bin"
    backbone_path = MODEL_CACHE / "backbone.onnx"
    try:
        http_download(f"{GATEWAY}{cfg['head_url']}", head_path)
        http_download(f"{GATEWAY}{cfg['backbone_url']}", backbone_path)
    except Exception as e:
        print(f"  detect {uuid[:8]}: model fetch err: {e}", flush=True)
        return False
    # MLP1 head.bin needs the channel-map sidecar to resolve the
    # recording's channel slug to a one-hot column. Fetch alongside;
    # legacy LogReg heads ignore it. 404 from the gateway means the
    # current head was trained without the sidecar (= pre-MLP era);
    # leave any stale local copy in place — Go loader gracefully
    # degrades to mlpChanIdx=-1 if the slug isn't in whatever map
    # is on disk.
    sidecar_path = MODEL_CACHE / "head.channel-map.json"
    sidecar_url = f"{GATEWAY}/api/internal/detect-models/head.channel-map.json"
    try:
        http_download(sidecar_url, sidecar_path)
    except Exception as e:
        # Don't bail — LogReg path doesn't need the sidecar.
        msg = str(e)
        if "404" not in msg:
            print(f"  detect {uuid[:8]}: channel-map sidecar fetch "
                  f"err (non-fatal): {e}", flush=True)

    # Per-channel logo. Resolution-aware: HD recordings (= width >=1280)
    # need an HD-trained logo because the bbox pixel-coords differ
    # between SD (720x576) and HD (1920x1080). Probe source width
    # once; pass ?res=hd to the gateway logo endpoint. Gateway falls
    # back to SD logo if no HD-trained file exists yet.
    logo_path = None
    if cfg.get("cached_logo_url"):
        src_w = 0
        if local and Path(local).is_file():
            try:
                pr = subprocess.run(
                    [FFPROBE, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width", "-of",
                     "default=nokey=1:noprint_wrappers=1", str(local)],
                    capture_output=True, text=True, timeout=15,
                    env=SPAWN_ENV)
                src_w = int((pr.stdout or "0").strip().splitlines()[0] or 0)
            except Exception:
                src_w = 0
        is_hd = src_w >= 1280
        suffix = ".hd" if is_hd else ""
        logo_path = MODEL_CACHE / "logos" / f"{slug}.logo{suffix}.txt"
        logo_path.parent.mkdir(exist_ok=True)
        url = f"{GATEWAY}{cfg['cached_logo_url']}"
        if is_hd:
            url += ("&" if "?" in url else "?") + "res=hd"
        try:
            http_download(url, logo_path)
        except Exception as e:
            print(f"  detect {uuid[:8]}: logo fetch err: {e}",
                  flush=True)
            logo_path = None

    # Per-channel logo CNN. Gateway sets cached_logo_cnn_url only when:
    # (a) channel-config opts in via "logo_cnn": true, AND
    # (b) Pi-side .tvd-logo-cnn/<slug>.logo-cnn.onnx exists.
    # tv-detect prefers CNN over edge-template when both are passed;
    # silent fallback if the ONNX load fails inside the Go binary.
    # Replaces the broken edge-template Sobel signal on busy channels
    # (VOX/Nick/RTL) where ad content false-positives at 0.95+ conf.
    logo_cnn_path = None
    if cfg.get("cached_logo_cnn_url") and slug:
        logo_cnn_path = MODEL_CACHE / "logo-cnn" / f"{slug}.logo-cnn.onnx"
        logo_cnn_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            http_download(f"{GATEWAY}{cfg['cached_logo_cnn_url']}",
                          logo_cnn_path)
        except Exception as e:
            print(f"  detect {uuid[:8]}: logo-cnn fetch err: {e}",
                  flush=True)
            logo_cnn_path = None

    # Per-channel bumper templates (channel station-id cards). One Pi
    # endpoint lists all PNGs configured for the slug; we download each
    # into MODEL_CACHE / "bumpers" / slug / <name>.png and pass the
    # comma-separated paths to tv-detect. Two independent kinds:
    #   - end-bumpers   (channel-root or .../<slug>/end/) → --bumper-templates
    #   - start-bumpers (.../<slug>/start/)               → --bumper-templates-start
    # Each goes through its own per-frame conf stream in tv-detect so
    # a start-bumper hit can't pull a block end and vice versa.
    bumper_paths = []
    bumper_start_paths = []
    if slug:
        try:
            blist = http_get_json(
                f"{GATEWAY}/api/internal/detect-bumpers/{slug}")
            bdir = MODEL_CACHE / "bumpers" / slug
            bdir.mkdir(parents=True, exist_ok=True)
            # Per-file mtime+size compare (= no HEAD request needed).
            # Pre-2026-05-12 daemon did 1 HEAD per template per detect
            # (= 2000+ network roundtrips, ~30-90s wallclock for ProSieben).
            # Now: gateway lists size+mtime in the bumper-list response;
            # daemon stat()s local file + skips download if both match.
            n_fetched = n_cached = 0
            def _ensure_local(entry, kind, paths_list):
                nonlocal n_fetched, n_cached
                local = bdir / kind / entry["name"]
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if local.is_file():
                        st = local.stat()
                        if (entry.get("size", -1) == st.st_size
                                and entry.get("mtime", -1) <= int(st.st_mtime)):
                            paths_list.append(str(local))
                            n_cached += 1
                            return
                    with urllib.request.urlopen(
                            f"{GATEWAY}{entry['url']}",
                            timeout=30, context=CTX) as r:
                        local.write_bytes(r.read())
                    paths_list.append(str(local))
                    n_fetched += 1
                except Exception as e:
                    print(f"  detect {uuid[:8]}: bumper {kind}/{entry['name']} "
                          f"err: {e}", flush=True)
            for entry in blist.get("templates", []):
                _ensure_local(entry, "end", bumper_paths)
            for entry in blist.get("start_templates", []):
                _ensure_local(entry, "start", bumper_start_paths)
            if n_fetched:
                print(f"  detect {uuid[:8]}: {n_cached} cached + "
                      f"{n_fetched} fetched bumper(s) for {slug}",
                      flush=True)
        except Exception as e:
            print(f"  detect {uuid[:8]}: bumper-list err: {e}",
                  flush=True)

    cmd = [TVD, "--quiet", "--workers", str(TVD_WORKERS),
           "--nn-backbone", str(backbone_path),
           "--nn-head",     str(head_path),
           "--nn-weight",   "0.3",
           "--logo-smooth", str(cfg.get("logo_smooth_s") or 0),]
    # --with-audio when the deployed head was trained with audio
    # (gateway tells us via the cfg flag; based on head.bin size).
    # Adds ~5-10 s ffmpeg pass per recording but unlocks the audio
    # feature in NN inference.
    if cfg.get("with_audio"):
        cmd += ["--with-audio"]
    # Per-recording whisper-prob feed for MLP2 v2 heads. tv-detect
    # only consumes this when the loaded head has n_whisper>0; for
    # MLP1 / LogReg heads the flag is silently ignored. The Mac-
    # local whisper cache is populated by tv-whisper-classify before
    # detect runs (= same daemon, earlier in the recording's
    # post-processing pipeline). Missing file → tv-detect logs
    # "whisper-json load failed" and the head sees neutral 0.5
    # per frame (= still works, just loses the +0.075 IoU gain).
    whisper_path = Path.home() / ".cache" / "tv-whisper" / f"{uuid}.whisper.json"
    if whisper_path.is_file():
        cmd += ["--nn-whisper-json", str(whisper_path)]
    cmd += [
           # Letterbox-snap with 90s window catches the RTL "Werbung"-
           # promo period that precedes the actual logo-loss: during
           # 16:9-letterboxed RTL self-promos the small RTL badge keeps
           # the logo signal "present" so the state machine opens the
           # block 30-50 s late. Snapping to the EARLIEST letterbox
           # onset within ±90 s rewinds the START to the true ad cut
           # (validated on GZSZ: 1146→1099, user truth 1099.82).
           # No-op on broadcasters that always air same aspect.
           "--letterbox-snap", "90"]
    # Both extend values are now SIGNED — positive grows the block in
    # the natural direction (earlier start / later end), negative
    # shrinks. Forward to tv-detect verbatim if non-zero.
    if cfg.get("start_extend_s", 0) != 0:
        cmd += ["--start-extend", str(cfg["start_extend_s"])]
    if cfg.get("end_extend_s", 0) != 0:
        cmd += ["--end-extend", str(cfg["end_extend_s"])]
    if cfg.get("min_block_s") and cfg.get("max_block_s"):
        cmd += ["--min-block-sec", str(cfg["min_block_s"]),
                "--max-block-sec", str(cfg["max_block_s"])]
    if slug:
        cmd += ["--channel-slug", slug]
    if logo_path and logo_path.exists() and logo_path.stat().st_size > 0:
        cmd += ["--logo", str(logo_path)]
        y_off = detect_letterbox_offset(src_url, uuid=uuid)
        if y_off > 0:
            cmd += ["--logo-y-offset", str(y_off)]
            print(f"  detect {uuid[:8]}: letterbox y-offset={y_off}",
                  flush=True)
        if logo_cnn_path and logo_cnn_path.exists() and logo_cnn_path.stat().st_size > 1024:
            # Pass alongside --logo so tv-detect keeps the bbox from
            # the template but uses the CNN for confidence. Falls back
            # to template internally if ONNX load fails.
            cmd += ["--logo-cnn", str(logo_cnn_path)]
            print(f"  detect {uuid[:8]}: using CNN logo for {slug}",
                  flush=True)
    else:
        cmd += ["--auto-train", "5"]
    if bumper_paths or bumper_start_paths:
        # Snap radius 90s — empirically the model under-predicts ad-end
        # by ~60-65s on RTL Spielfilm; 60 was JUST too tight to catch
        # 3 of 4 JC bumpers. 90 catches all four cleanly. Same window
        # used for both kinds — the snap direction differs (end picks
        # latest match, start picks earliest), not the radius.
        # Threshold per-channel: gateway returns 0.75 default (= helps
        # missed_bumper cases) with channel-specific overrides for
        # senders whose template set is too generic to tolerate the
        # lower bar (RTL). Backwards-compatible default 0.75 if the
        # gateway hasn't been updated yet.
        bt = float(cfg.get("bumper_threshold", 0.75))
        cmd += ["--bumper-snap", "90", "--bumper-threshold", str(bt)]
        if bumper_paths:
            cmd += ["--bumper-templates", ",".join(bumper_paths)]
        if bumper_start_paths:
            cmd += ["--bumper-templates-start", ",".join(bumper_start_paths)]
        print(f"  detect {uuid[:8]}: {len(bumper_paths)} end + "
              f"{len(bumper_start_paths)} start bumper(s) for {slug}",
              flush=True)

    # Join the speaker-fingerprint future kicked off at the top of
    # process_detect. If the pipeline finished while bumpers / model
    # files were downloading we get spk_csv immediately; otherwise we
    # block here until done. Either way, the speaker work no longer
    # serially adds to the per-detect wall — it's absorbed by the
    # setup phase's existing IO time.
    try:
        spk_csv = spk_future.result(timeout=2400)
    except Exception as e:
        print(f"  detect {uuid[:8]}: speaker future err: {e}", flush=True)
        spk_csv = None
    finally:
        spk_executor.shutdown(wait=False)
    if spk_csv:
        cmd += ["--speaker-csv", str(spk_csv),
                "--speaker-weight", str(SPEAKER_WEIGHT)]
        print(f"  detect {uuid[:8]}: speaker fingerprint engaged "
              f"(weight={SPEAKER_WEIGHT})", flush=True)

    cmd += ["--output", "cutlist", src_url]
    # Run detect below an on-demand HLS remux in CPU priority (children
    # inherit the niceness). No-op-safe if /usr/bin/nice is missing.
    if DETECT_NICE:
        cmd = ["nice", "-n", str(DETECT_NICE)] + cmd

    # tv-detect spawns ffprobe internally — uses the shared SPAWN_ENV
    # which sets PATH for launchd-spawned subprocesses.
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT_S, env=SPAWN_ENV)
    if result.returncode != 0:
        stderr = result.stderr or ""
        print(f"  detect {uuid[:8]} rc={result.returncode}: "
              f"{stderr[-300:]}", flush=True)
        # Unrecoverable source (corrupt H.264 → ffprobe can't read duration →
        # tv-detect "cannot plan chunks"): retrying never helps, and because
        # the high-prio queue isn't drained until the marker clears, a single
        # such recording head-of-line-blocks ALL background detects. Give up
        # on the first failure instead of burning 3×FAIL_COOLDOWN_S (30 min).
        corrupt = ("cannot plan chunks" in stderr
                   or "duration=0.000000" in stderr)
        _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
        _record_detect_failure(uuid, force=corrupt)
        return False
    # Surface tv-detect's per-phase wall-time line to the daemon log
    # (= even with --quiet the binary emits "pipeline-timing: ..."
    # unconditionally so we can profile production drains without
    # re-running them by hand). Cheap grep over the captured stderr.
    for ln in (result.stderr or "").splitlines():
        if ln.startswith("pipeline-timing"):
            print(f"  detect {uuid[:8]}: {ln}", flush=True)
            break
    cutlist = result.stdout
    # Optional Whisper post-processor (no-op when WHISPER_ENABLE=0).
    # Re-fetch `local` via get_source: the bumper download loop above
    # rebinds `local` to the LAST bumper PNG path it processed, which
    # is wrong here. get_source is cached/idempotent so the second call
    # is a stat() + return.
    #
    # Conditional skip (2026-05-12): for older recordings (>14d, NOT in
    # test-set) the whisper-refinement gain is marginal vs the 20-60s
    # wallclock cost. Mirror the gateway's Smart-V2 invalidation logic
    # to drop the cost on bulk-redetect cycles. Test-set + recent stay
    # whisper-refined for full quality where it matters.
    rec_ts = cfg.get("start_real") or cfg.get("start") or 0
    is_recent = (time.time() - rec_ts) < 14 * 86400 if rec_ts else True
    is_test_uuid = cfg.get("is_test_uuid", False)
    if WHISPER_ENABLE and (is_recent or is_test_uuid):
        local_ts = get_source(uuid)
        if local_ts:
            cutlist = _maybe_whisper_refine(uuid, str(local_ts), cutlist)
    try:
        http_post_stream(
            f"{GATEWAY}/api/internal/cutlist-uploaded/{uuid}",
            cutlist.encode())
    except Exception as e:
        # Cutlist-upload failures (= gateway 404, network blip, etc.)
        # used to fall through with no failure-tracking → daemon
        # picked the same uuid up next cycle, ran tv-detect again
        # (~6 min wall), failed upload again → endless loop.
        # Treat as a detect failure: cool down + count toward the
        # MAX_DETECT_RETRIES cap so persistent uploaders eventually
        # get given-up via /api/internal/detect-give-up.
        print(f"  detect upload err: {e}", flush=True)
        _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
        _record_detect_failure(uuid)
        return False
    n_blocks = sum(1 for ln in cutlist.splitlines() if ln.strip()
                    and "\t" in ln and not ln.startswith("FILE"))
    print(f"  detect {uuid[:8]}: {n_blocks} blocks, "
          f"{time.time()-t0:.0f}s", flush=True)
    _record_detect_success(uuid)
    return True


def main():
    print(f"tv-thumbs-daemon (HTTP, thumbs+hls) started, "
          f"gateway={GATEWAY}", flush=True)
    # Detect concurrency: in-flight set of UUIDs currently being processed
    # by a worker thread. Avoids re-submitting the same recording on the
    # next poll cycle while it's still in progress. HLS/thumbs jobs stay
    # sequential (they're fast and contention-free).
    from concurrent.futures import ThreadPoolExecutor
    import threading
    detect_executor = ThreadPoolExecutor(max_workers=DETECT_PARALLEL)
    detect_in_flight = set()
    detect_in_flight_since = {}  # uuid → time.time() at slot acquisition
    detect_lock = threading.Lock()
    # HLS+thumbs concurrency: same idea as detect but separate pool +
    # in_flight set. process_recording does HLS+thumbs in one call, so
    # a single worker owns both for a uuid.
    hls_executor = ThreadPoolExecutor(max_workers=HLS_PARALLEL,
                                       thread_name_prefix="hls")
    hls_in_flight = set()
    hls_lock = threading.Lock()

    def _gc_stuck_in_flight():
        """Free slots leaked by abnormally-exited worker threads (= the
        try/except/finally in _run_detect should always discard, but a
        thread killed by a C-extension segfault or stuck on a hung HTTP
        POST never reaches finally). For each long-stale uuid: if no
        subprocess has it in argv, the slot is dead — discard it so new
        work can flow."""
        if not detect_in_flight:
            return
        now = time.time()
        with detect_lock:
            stale = [u for u, t in detect_in_flight_since.items()
                     if now - t > IN_FLIGHT_STUCK_S]
        if not stale:
            return
        freed = []
        for uuid in stale:
            try:
                r = subprocess.run(
                    ["pgrep", "-f", uuid], capture_output=True, timeout=5)
            except Exception:
                continue
            if r.returncode == 0 and r.stdout.strip():
                continue  # live subprocess still running for this uuid
            freed.append(uuid)
        if not freed:
            return
        with detect_lock:
            for uuid in freed:
                detect_in_flight.discard(uuid)
                detect_in_flight_since.pop(uuid, None)
        for uuid in freed:
            print(f"  [watchdog] in_flight stuck slot freed: {uuid[:8]} "
                  f"(no live subprocess, > {IN_FLIGHT_STUCK_S//60} min)",
                  flush=True)

    def _run_detect(uuid):
        # Tell the gateway we just picked this job up so the recordings
        # page can render the 🔍 badge. Best-effort — the cutlist-uploaded
        # call at the end clears the entry, and stale entries auto-expire
        # gateway-side after 10 min.
        try:
            req = urllib.request.Request(
                f"{GATEWAY}/api/internal/detect-started/{uuid}",
                method="POST")
            urllib.request.urlopen(req, timeout=5, context=CTX).read()
        except Exception:
            pass
        try:
            process_detect(uuid)
        except Exception as e:
            print(f"  detect {uuid[:8]}: unhandled err: {e}", flush=True)
        finally:
            with detect_lock:
                detect_in_flight.discard(uuid)
                detect_in_flight_since.pop(uuid, None)

    def _maybe_fire_snapshot():
        """Per-show IoU snapshot trigger. Set by tv-train-head.sh dropping
        a marker file with a desired timestamp; we fire the gateway
        endpoint once the bulk-redetect after the head update is done
        (= no detect-pending AND no in-flight). Stateless — marker
        deleted on success so the next retrain just drops it again.
        """
        if not SNAPSHOT_MARKER.exists():
            return
        try:
            ts = SNAPSHOT_MARKER.read_text().strip() or time.strftime("%Y%m%dT%H%M%S")
        except Exception:
            ts = time.strftime("%Y%m%dT%H%M%S")
        try:
            req = urllib.request.Request(
                f"{GATEWAY}/api/internal/snapshot-per-show-iou?ts={ts}",
                method="POST")
            r = urllib.request.urlopen(req, timeout=30, context=CTX).read().decode()
            print(f"  snapshot fired (ts={ts}): {r[:100]}", flush=True)
            SNAPSHOT_MARKER.unlink()
        except Exception as e:
            print(f"  snapshot fire err: {e}", flush=True)

    cycle = 0
    _hls_gate = [False]  # edge-trigger for the "detect paused/resumed" log
    while True:
        cycle += 1
        _maybe_gc_orphans()
        _gc_stuck_in_flight()
        thumbs = hls = detect = detect_low = []
        try:
            thumbs = http_get_json(
                f"{GATEWAY}/api/internal/thumbs-pending").get("pending", [])
            hls = http_get_json(
                f"{GATEWAY}/api/internal/hls-pending").get("pending", [])
            detect = http_get_json(
                f"{GATEWAY}/api/internal/detect-pending").get("pending", [])
        except Exception as e:
            if cycle % 12 == 1:
                print(f"  [cycle {cycle}] gateway unreachable: {e}",
                      flush=True)
            time.sleep(30); continue
        with detect_lock:
            in_flight_n = len(detect_in_flight)
        # Source-cache prefetch — runs only when nothing else is
        # consuming Pi disk IO / link bandwidth (in_flight_n == 0).
        # Internal interval gate caps it at one cycle per 30 min.
        _maybe_prefetch_sources(in_flight_n)
        # Layer-3 integrity report to the Pi (own 30-min interval gate).
        _maybe_integrity_sweep()
        # Host-stats push for the app panel (own 30s interval gate).
        _maybe_host_stats()
        # V2 low-prio queue: fetch when there is no UNCLAIMED high-
        # prio work AND we still have a parallel slot free.
        # "Unclaimed" = the high-prio entry isn't the uuid we're
        # already running (= the gateway only clears markers after
        # cutlist upload, so a high-prio job in-flight keeps showing
        # in detect-pending until it finishes). Without the in_flight
        # subtraction, DETECT_PARALLEL=2 collapses to 1 the moment a
        # high-prio job starts: bg gate fires `not detect` → False →
        # bg skipped, the 2nd slot sits empty for the entire
        # high-prio detect duration.
        with detect_lock:
            new_high = [r for r in detect
                        if r.get("uuid") not in detect_in_flight]
        if not new_high and in_flight_n < DETECT_PARALLEL:
            try:
                detect_low = http_get_json(
                    f"{GATEWAY}/api/internal/detect-pending-low"
                    ).get("pending", [])
            except Exception:
                detect_low = []
        # Per-show IoU snapshot — only when BOTH queues fully drained
        # (otherwise the snapshot would record stale-ads.json IoU=0 for
        # most rows). Marker file dropped by tv-train-head.sh.
        if not detect and not detect_low and in_flight_n == 0:
            _maybe_fire_snapshot()
        # Spot-fingerprint backfill drain — runs only when detect/HLS
        # queues are quiet AND nothing in flight (= avoid CPU + Pi
        # link contention). Throttled to one batch per ~30 s by the
        # cycle gate. Subprocess so we keep one ffmpeg ↔ fpcalc
        # pipeline cleanly contained.
        if (SPOT_EXTRACT_ENABLE and not detect and not detect_low
                and not thumbs and not hls
                and in_flight_n == 0
                and cycle % 6 == 0
                and SPOT_EXTRACT_SCRIPT.is_file()):
            try:
                q = http_get_json(
                    f"{GATEWAY}/api/internal/spot-fp/queue")
                qlen = q.get("n_missing", 0) + q.get("n_partial_no_dhash", 0)
            except Exception:
                qlen = 0
            if qlen > 0:
                print(f"  spot-fp queue: {qlen} → drain {SPOT_EXTRACT_BATCH}",
                      flush=True)
                try:
                    subprocess.Popen(
                        [sys.executable, str(SPOT_EXTRACT_SCRIPT),
                         "--queue", "--limit", str(SPOT_EXTRACT_BATCH)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                        env=SPAWN_ENV)
                except Exception as e:
                    print(f"  spot-fp drain spawn err: {e}", flush=True)
        if (thumbs or hls or detect or detect_low or in_flight_n
                or cycle % 12 == 1):
            print(f"  [cycle {cycle}] thumbs={len(thumbs)} "
                  f"hls={len(hls)} detect={len(detect)}"
                  f"+{len(detect_low)}bg "
                  f"in_flight={in_flight_n}/{DETECT_PARALLEL}",
                  flush=True)
        thumb_uuids = {j["uuid"] for j in thumbs}
        hls_uuids = {j["uuid"] for j in hls}
        detect_uuids = {j["uuid"] for j in detect}
        # Failure cooldown — Pi-fallback timer picks up locally
        now = time.time()
        cooled = {u for u, t in _failed_until.items() if t > now}
        for u in [u for u, t in _failed_until.items() if t <= now]:
            _failed_until.pop(u, None)

        # HLS-priority gate: don't START new detect jobs while a VOD remux is
        # pending or in flight. Detect is background; a remux is latency-
        # critical (user waiting on a loading bar) and the two contend for
        # CPU + GPU/ANE. In-flight detects finish; new ones wait until HLS
        # drains (= "VODs first, detect after"). HLS is bursty + fast, so this
        # only pauses detect briefly; the nightly bulk re-detect runs when HLS
        # is idle. Logged once per pause-edge to avoid spam.
        # A remux stuck in failure-cooldown (bad source) keeps its marker, so
        # it'd block detect forever — exclude cooled jobs so the gate only
        # honours remuxes that can actually make progress.
        hls_live = [j for j in hls if j.get("uuid") not in cooled]
        with hls_lock:
            n_hls_inflight = len(hls_in_flight)
            hls_active = bool(hls_live) or n_hls_inflight > 0
        if hls_active and (detect or detect_low) and not _hls_gate[0]:
            print(f"  [cycle {cycle}] detect paused — "
                  f"{len(hls_live)} hls pending / {n_hls_inflight} in flight "
                  f"have CPU/GPU priority", flush=True)
            _hls_gate[0] = True
        elif not hls_active and _hls_gate[0]:
            print(f"  [cycle {cycle}] detect resumed (hls drained)", flush=True)
            _hls_gate[0] = False
        # Submit detect jobs to the pool until either the pool is full or
        # the queue is exhausted. Each pool worker runs process_detect in
        # parallel; tv-detect inside scales itself to TVD_WORKERS cores.
        with detect_lock:
            available = 0 if hls_active else DETECT_PARALLEL - len(detect_in_flight)
            already = set(detect_in_flight)
        for j in detect:
            if available <= 0:
                break
            uuid = j["uuid"]
            if uuid in already or uuid in cooled:
                continue
            with detect_lock:
                detect_in_flight.add(uuid)
                detect_in_flight_since[uuid] = time.time()
            print(f"  → {uuid[:8]} detect=True (parallel slot)", flush=True)
            detect_executor.submit(_run_detect, uuid)
            available -= 1
        # V2 low-prio: fill remaining slots with bg jobs. Earlier
        # took only `detect_low[0]` per cycle which left slots empty
        # if [0] was the in-flight uuid or in cooldown — iterate to
        # find the next eligible candidate(s) instead. Still gated
        # on `not detect` so any new high-prio marker pauses bg
        # uptake on the next cycle.
        if available > 0 and detect_low and not detect:
            for j in detect_low:
                if available <= 0:
                    break
                uuid = j["uuid"]
                if uuid in already or uuid in cooled:
                    continue
                with detect_lock:
                    detect_in_flight.add(uuid)
                    detect_in_flight_since[uuid] = time.time()
                already.add(uuid)
                print(f"  → {uuid[:8]} detect=True (bg slot)", flush=True)
                detect_executor.submit(_run_detect, uuid)
                available -= 1

        # HLS+thumbs: parallel via hls_executor. Each uuid gets one
        # worker-thread that runs process_recording (= HLS+thumbs in one
        # shot). Pool cap = HLS_PARALLEL. Skip if already in-flight on
        # this pool OR on the detect pool (= same uuid being touched).
        def _run_hls(uuid, do_hls, do_thumbs):
            try:
                process_recording(uuid, do_hls, do_thumbs)
            except Exception as e:
                print(f"  hls {uuid[:8]}: unhandled err: {e}", flush=True)
            finally:
                with hls_lock:
                    hls_in_flight.discard(uuid)
        for uuid in sorted((hls_uuids | thumb_uuids) - cooled):
            do_hls = uuid in hls_uuids
            do_thumbs = uuid in thumb_uuids
            with detect_lock:
                if uuid in detect_in_flight:
                    continue
            with hls_lock:
                if uuid in hls_in_flight:
                    continue
                if len(hls_in_flight) >= HLS_PARALLEL:
                    break  # pool full, try again next cycle
                hls_in_flight.add(uuid)
            print(f"  → {uuid[:8]} hls={do_hls} thumbs={do_thumbs} "
                  f"(pool {len(hls_in_flight)}/{HLS_PARALLEL})",
                  flush=True)
            hls_executor.submit(_run_hls, uuid, do_hls, do_thumbs)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
