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

import io, json, os, re, ssl, subprocess, sys, tarfile, tempfile, time
import urllib.request
from pathlib import Path

POLL_INTERVAL_S = 5
TIMEOUT_S = 1800           # HLS-remux of 90-min recording can take ~30s
GATEWAY = os.environ.get("GATEWAY", "http://raspberrypi5lan:8080")

# Concurrency: how many detect jobs to run in parallel. Default 1 (legacy
# sequential). Set DETECT_PARALLEL=3 to overlap downloads + decode + NN.
# Each tv-detect run uses --workers (8 // DETECT_PARALLEL) CPU cores so
# total per-job CPU stays bounded — a Mac with 8 perf cores running 3
# parallel detects with 4 workers each is fine; the 8th core handles
# Python + ffmpeg overhead. Memory bound: each detect carries ~1-2 GB
# (ffmpeg decode buffers + NN backbone), so 3-parallel ≈ 4-6 GB peak.
DETECT_PARALLEL = max(1, int(os.environ.get("DETECT_PARALLEL", "1")))
TVD_WORKERS = max(2, 8 // DETECT_PARALLEL)
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
TVD = os.path.expanduser("~/.local/bin/tv-detect")
SAFE_VIDEO = {"h264", "hevc"}

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


def _maybe_evict_source_cache():
    files = sorted(SOURCE_CACHE.glob("*.ts"),
                   key=lambda p: p.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    cap = SOURCE_CACHE_MAX_GB * 1024 ** 3
    while total > cap and files:
        oldest = files.pop(0)
        sz = oldest.stat().st_size
        try: oldest.unlink(); total -= sz
        except Exception: pass


_last_orphan_gc = 0.0
ORPHAN_GC_INTERVAL_S = 3600  # once per hour


def _maybe_gc_orphans():
    """Drop cached .ts whose recording has been deleted on the Pi.
    LRU alone would evict eventually but only when the cap is hit;
    explicit orphan removal frees space sooner and keeps the cache
    aligned with Pi-side reality. Runs at most once per hour."""
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
    n_removed = 0
    bytes_removed = 0
    for f in SOURCE_CACHE.glob("*.ts"):
        uuid = f.stem
        if uuid not in valid:
            try:
                bytes_removed += f.stat().st_size
                f.unlink()
                n_removed += 1
            except Exception:
                pass
    if n_removed:
        print(f"  orphan-gc: removed {n_removed} cached .ts "
              f"({bytes_removed/1e6:.0f} MB)", flush=True)


def get_source(uuid):
    """Return local .ts path. Cached: serve from disk. Cold: HTTP-fetch
    + cache for next time. Falls back to None on any error — caller
    should use the HTTP URL directly as a last resort."""
    cache_path = SOURCE_CACHE / f"{uuid}.ts"
    if cache_path.exists() and cache_path.stat().st_size > 100_000:
        try: cache_path.touch()  # update atime for LRU
        except Exception: pass
        return cache_path
    src_url = f"{GATEWAY}/recording/{uuid}/source"
    tmp = cache_path.with_suffix(".tmp")
    t0 = time.time()
    try:
        with urllib.request.urlopen(src_url, timeout=600,
                                       context=CTX) as r:
            with open(tmp, "wb") as f:
                import shutil
                shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(cache_path)
        size_mb = cache_path.stat().st_size / 1e6
        print(f"  cached {uuid[:8]} ({size_mb:.0f} MB in "
              f"{time.time()-t0:.0f}s)", flush=True)
        _maybe_evict_source_cache()
        return cache_path
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
                    capture_output=True, text=True, timeout=30)
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
                v_opts = ["-vf",
                          "scale=trunc(iw*sar/2)*2:trunc(ih/2)*2,setsar=1",
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
        # Suppress ffmpeg's mpeg2video MV warnings (broadcasters drop
        # GOPs all the time; ffmpeg recovers fine, the noise just
        # buries our own log lines). Capture stderr for failure
        # diagnosis.
        result = subprocess.run(cmd, timeout=TIMEOUT_S,
                                  capture_output=True, text=True)
        rc = result.returncode
        encode_s = time.time() - t0
        if rc != 0:
            err_tail = (result.stderr or "")[-500:]
            print(f"  ffmpeg {uuid[:8]} rc={rc} — last stderr:\n"
                  f"{err_tail}", flush=True)
            # Remove markers so the daemon doesn't loop on this uuid
            # forever. The next prewarm cycle will re-create them
            # if the recording still needs processing — gives the
            # underlying issue (corrupt .ts, gateway disconnect)
            # a chance to clear before we retry.
            _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
            return False
        ok_hls = ok_thumbs = True
        if do_hls:
            playlist = td_p / "index.m3u8"
            if not playlist.exists():
                print(f"  no playlist for {uuid[:8]}", flush=True)
                ok_hls = False
            else:
                # Segments first, playlist last — Pi sees index.m3u8
                # only when all segments have arrived (player polls
                # for it as the readiness signal).
                segments = sorted(td_p.glob("*.ts"))
                files = segments + [playlist]
                size_mb = sum(f.stat().st_size for f in files) / 1e6
                t1 = time.time()
                try:
                    _upload_files_put(
                        f"{GATEWAY}/api/internal/hls-segment/{uuid}/{{name}}",
                        files)
                    http_post_stream(
                        f"{GATEWAY}/api/internal/hls-done/{uuid}",
                        b"")
                    print(f"  hls {uuid[:8]}: {len(files)} files "
                          f"({size_mb:.0f} MB), encode {encode_s:.0f}s "
                          f"+ upload {time.time()-t1:.0f}s", flush=True)
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
        return ok_hls and ok_thumbs


# Channels broadcast 16:9 movies in a 4:3 container with letterbox
# bars; the logo template (trained against the full frame) then sits
# in the top black bar instead of on the visible image. cropdetect
# finds the visible-content top edge — the logo sits ~20px above that
# (RTL-empirical for Jungle Cruise: cropdetect=80, optimal offset=60).
# May need per-channel calibration later; for now one constant.
LETTERBOX_LOGO_OVERHANG = 20


def detect_letterbox_offset(src):
    """Quick cropdetect on a 5s sample 60s into the recording (skips
    intro/promo). Returns recommended --logo-y-offset, or 0 if no
    meaningful letterbox. src may be a Path or HTTP URL."""
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "info",
             "-ss", "60", "-t", "5", "-i", str(src),
             "-vf", "cropdetect=24:16:0",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"  cropdetect err: {e}", flush=True)
        return 0
    # stderr lines look like: [...] crop=720:416:0:80
    # cropdetect refines the box over time — last match is the most
    # confident aggregate.
    ys = re.findall(r"crop=\d+:\d+:\d+:(\d+)", r.stderr)
    if not ys:
        return 0
    y = int(ys[-1])
    if y < 8:
        return 0  # no meaningful letterbox
    return max(0, y - LETTERBOX_LOGO_OVERHANG)


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
                capture_output=True, text=True, timeout=2400)
            if r.returncode != 0:
                print(f"  detect {uuid[:8]}: embedding extract failed "
                      f"(rc={r.returncode}): {r.stderr[:300]}", flush=True)
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
            capture_output=True, text=True, timeout=120)
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
            capture_output=True, text=True, timeout=60)
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

    # Refresh model cache (small files, only re-downloads on size change)
    head_path = MODEL_CACHE / "head.bin"
    backbone_path = MODEL_CACHE / "backbone.onnx"
    try:
        http_download(f"{GATEWAY}{cfg['head_url']}", head_path)
        http_download(f"{GATEWAY}{cfg['backbone_url']}", backbone_path)
    except Exception as e:
        print(f"  detect {uuid[:8]}: model fetch err: {e}", flush=True)
        return False

    # Per-channel logo (auto-train fallback if absent)
    logo_path = None
    if cfg.get("cached_logo_url"):
        logo_path = MODEL_CACHE / "logos" / f"{slug}.logo.txt"
        logo_path.parent.mkdir(exist_ok=True)
        try:
            http_download(f"{GATEWAY}{cfg['cached_logo_url']}",
                          logo_path)
        except Exception as e:
            print(f"  detect {uuid[:8]}: logo fetch err: {e}",
                  flush=True)
            logo_path = None

    # Per-channel bumper templates (channel station-id cards). One Pi
    # endpoint lists all PNGs configured for the slug; we download each
    # into MODEL_CACHE / "bumpers" / slug / <name>.png and pass the
    # comma-separated paths to tv-detect. No-op if the channel has no
    # templates configured.
    bumper_paths = []
    if slug:
        try:
            blist = http_get_json(
                f"{GATEWAY}/api/internal/detect-bumpers/{slug}"
            ).get("templates", [])
            bdir = MODEL_CACHE / "bumpers" / slug
            bdir.mkdir(parents=True, exist_ok=True)
            for entry in blist:
                local = bdir / entry["name"]
                try:
                    http_download(f"{GATEWAY}{entry['url']}", local)
                    bumper_paths.append(str(local))
                except Exception as e:
                    print(f"  detect {uuid[:8]}: bumper fetch {entry['name']} "
                          f"err: {e}", flush=True)
        except Exception as e:
            print(f"  detect {uuid[:8]}: bumper-list err: {e}",
                  flush=True)

    cmd = [TVD, "--quiet", "--workers", str(TVD_WORKERS),
           "--nn-backbone", str(backbone_path),
           "--nn-head",     str(head_path),
           "--nn-weight",   "0.3",
           "--logo-smooth", str(cfg.get("logo_smooth_s") or 0),
           # Letterbox-snap with 90s window catches the RTL "Werbung"-
           # promo period that precedes the actual logo-loss: during
           # 16:9-letterboxed RTL self-promos the small RTL badge keeps
           # the logo signal "present" so the state machine opens the
           # block 30-50 s late. Snapping to the EARLIEST letterbox
           # onset within ±90 s rewinds the START to the true ad cut
           # (validated on GZSZ: 1146→1099, user truth 1099.82).
           # No-op on broadcasters that always air same aspect.
           "--letterbox-snap", "90"]
    if cfg.get("start_extend_s", 0) > 0:
        cmd += ["--start-extend", str(cfg["start_extend_s"])]
    if cfg.get("end_extend_s", 0) > 0:
        cmd += ["--end-extend", str(cfg["end_extend_s"])]
    if cfg.get("min_block_s") and cfg.get("max_block_s"):
        cmd += ["--min-block-sec", str(cfg["min_block_s"]),
                "--max-block-sec", str(cfg["max_block_s"])]
    if slug:
        cmd += ["--channel-slug", slug]
    if logo_path and logo_path.exists() and logo_path.stat().st_size > 0:
        cmd += ["--logo", str(logo_path)]
        y_off = detect_letterbox_offset(src_url)
        if y_off > 0:
            cmd += ["--logo-y-offset", str(y_off)]
            print(f"  detect {uuid[:8]}: letterbox y-offset={y_off}",
                  flush=True)
    else:
        cmd += ["--auto-train", "5"]
    if bumper_paths:
        # Snap radius 90s — empirically the model under-predicts ad-end
        # by ~60-65s on RTL Spielfilm; 60 was JUST too tight to catch
        # 3 of 4 JC bumpers. 90 catches all four cleanly.
        cmd += ["--bumper-templates", ",".join(bumper_paths),
                "--bumper-snap", "90"]
        print(f"  detect {uuid[:8]}: {len(bumper_paths)} bumper "
              f"template(s) loaded for {slug}", flush=True)

    # Speaker-fingerprint orchestration (only if SPEAKER_ENABLE=1).
    # Three-step: extract embeddings → update show centroid → compute
    # per-recording speaker.csv. Returns None on cold-start (no
    # confirmed episodes yet) or any error — detect runs without speaker
    # in those cases.
    spk_csv = _ensure_speaker_artifacts(
        uuid, src_url, cfg.get("show_title") or "")
    if spk_csv:
        cmd += ["--speaker-csv", str(spk_csv),
                "--speaker-weight", str(SPEAKER_WEIGHT)]
        print(f"  detect {uuid[:8]}: speaker fingerprint engaged "
              f"(weight={SPEAKER_WEIGHT})", flush=True)

    cmd += ["--output", "cutlist", src_url]

    # tv-detect spawns ffprobe internally — give it a sane PATH that
    # launchd doesn't otherwise provide. Includes brew (Apple Silicon
    # default) and ~/.local/bin (where tv-detect itself lives).
    t0 = time.time()
    env = os.environ.copy()
    env["PATH"] = (f"{os.path.expanduser('~/.local/bin')}:"
                   f"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
    result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT_S, env=env)
    if result.returncode != 0:
        print(f"  detect {uuid[:8]} rc={result.returncode}: "
              f"{(result.stderr or '')[-300:]}", flush=True)
        _failed_until[uuid] = time.time() + FAIL_COOLDOWN_S
        return False
    cutlist = result.stdout
    try:
        http_post_stream(
            f"{GATEWAY}/api/internal/cutlist-uploaded/{uuid}",
            cutlist.encode())
    except Exception as e:
        print(f"  detect upload err: {e}", flush=True)
        return False
    n_blocks = sum(1 for ln in cutlist.splitlines() if ln.strip()
                    and "\t" in ln and not ln.startswith("FILE"))
    print(f"  detect {uuid[:8]}: {n_blocks} blocks, "
          f"{time.time()-t0:.0f}s", flush=True)
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
    detect_lock = threading.Lock()

    def _run_detect(uuid):
        try:
            process_detect(uuid)
        except Exception as e:
            print(f"  detect {uuid[:8]}: unhandled err: {e}", flush=True)
        finally:
            with detect_lock:
                detect_in_flight.discard(uuid)

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
    while True:
        cycle += 1
        _maybe_gc_orphans()
        thumbs = hls = detect = []
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
        # Per-show IoU snapshot — only when bulk-redetect is fully drained
        # (otherwise the snapshot would record stale-ads.json IoU=0 for
        # most rows). Marker file dropped by tv-train-head.sh.
        if not detect and in_flight_n == 0:
            _maybe_fire_snapshot()
        if thumbs or hls or detect or in_flight_n or cycle % 12 == 1:
            print(f"  [cycle {cycle}] thumbs={len(thumbs)} "
                  f"hls={len(hls)} detect={len(detect)} "
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

        # Submit detect jobs to the pool until either the pool is full or
        # the queue is exhausted. Each pool worker runs process_detect in
        # parallel; tv-detect inside scales itself to TVD_WORKERS cores.
        with detect_lock:
            available = DETECT_PARALLEL - len(detect_in_flight)
            already = set(detect_in_flight)
        for j in detect:
            if available <= 0:
                break
            uuid = j["uuid"]
            if uuid in already or uuid in cooled:
                continue
            with detect_lock:
                detect_in_flight.add(uuid)
            print(f"  → {uuid[:8]} detect=True (parallel slot)", flush=True)
            detect_executor.submit(_run_detect, uuid)
            available -= 1

        # HLS / thumbs stay sequential (cheap, doesn't benefit from parallel)
        for uuid in sorted((hls_uuids | thumb_uuids) - cooled):
            do_hls = uuid in hls_uuids
            do_thumbs = uuid in thumb_uuids
            with detect_lock:
                if uuid in detect_in_flight:
                    continue  # let detect-thread own this uuid this cycle
            print(f"  → {uuid[:8]} hls={do_hls} thumbs={do_thumbs}",
                  flush=True)
            if do_hls or do_thumbs:
                process_recording(uuid, do_hls, do_thumbs)
            if do_hls:
                break
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
