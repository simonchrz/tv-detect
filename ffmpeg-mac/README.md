# Mac ffmpeg build — copy for version control

The live copy is `~/ffmpeg-h3-mac/` (build.sh + patches/), which is not a git
repo. This mirror exists so the patch is not lost with that directory.

`patches/0001-ffmpeg_sched-fix-decoder-starvation.patch` fixes an upstream
regression in `03dfac56` ("fftools/ffmpeg_sched: allow throttling decoder
outputs"): `unchoke_downstream()` gates the traversal on the decoder's own
`choked_next`, which `schedule_update_locked()` resets at the top of every
pass, so whether it lets through depends on the order streams are visited in.
When it blocks, everything downstream of the decoder stays choked, its output
backs up and it stops draining its own overflow FIFO.

With two output streams that starves one decoder for most of the run: ffmpeg
writes ~42% of the video frames, exits 0, prints no warning and shows no
`drop=`, and the count differs on every run. Measured on a 4742 s capture,
video parked 234387 packets and was served 103314 while audio drained fully.

Verified: 237062 frames on five consecutive runs (previously 96314-110064),
an unaffected recording unchanged at 207740, video-only unchanged.

Submitted upstream as a git-format-patch. When it lands, delete the patch here
and in `~/ffmpeg-h3-mac/patches/` — build.sh fails loudly rather than silently
building an unpatched binary, so a stale patch cannot go unnoticed.
