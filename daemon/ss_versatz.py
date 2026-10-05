"""Versatz zwischen Container- und Video-Beginn fuer ffmpeg-Eingangs-`-ss`.

ffmpeg misst ein Eingangs-`-ss` ab dem CONTAINER-Beginn (kleinster start_time
aller Stroeme), Labels und Erkennung zaehlen ab dem VIDEO-Beginn. Steckt ein
fremdes Programm mit frueherem PTS in der Datei (PID-Union-Zeit, Mai 2026),
landet jeder Sprung um diesen Versatz zu frueh — bei dvr-kabel-eins-1779893460
um 2516 s. Dieselbe Regel wie tv-detect internal/decode/probe.go
(SeekOffsetS, 5520247): erst ab 2 s, damit normale Dateien (Audio beginnt
Bruchteile frueher) bitgleich weiterlaufen.
"""
import json
import subprocess

MIN_S = 2.0
_cache = {}


def versatz(pfad, ffprobe="ffprobe"):
    """Sekunden, um die der Videostrom spaeter beginnt als der Container (0 unter MIN_S)."""
    key = str(pfad)
    if key in _cache:
        return _cache[key]
    v = 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=start_time",
             "-select_streams", "v:0", "-show_entries", "stream=start_time",
             "-of", "json", key],
            capture_output=True, text=True, timeout=30).stdout
        d = json.loads(out or "{}")
        fs = float(d.get("format", {}).get("start_time"))
        vs = float((d.get("streams") or [{}])[0].get("start_time"))
        if vs - fs >= MIN_S:
            v = vs - fs
    except (TypeError, ValueError, OSError, subprocess.SubprocessError):
        v = 0.0
    _cache[key] = v
    return v
