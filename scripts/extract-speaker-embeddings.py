#!/usr/bin/env python3
"""Extract per-window speaker embeddings from a recording.

Output: .npz with keys
    times:      (N,) float32 — window start time in seconds
    embeddings: (N, 192) float32 — ECAPA-TDNN speaker embeddings
    has_speech: (N,) bool — true if window contained speech (else embedding
                            is unreliable/silence; downstream should treat
                            as "no signal" rather than "different speaker")

Designed to run once per recording and cached. Re-detects then read the .npz
without re-running ML inference.

Usage:
    extract-speaker-embeddings.py <input.ts|http-url> <output.npz>
                                  [--window-s 2.0] [--hop-s 1.0]
                                  [--vad-energy 0.005]
"""

import argparse
import io
import os
import subprocess
import sys
import time

import numpy as np

SR = 16000  # ECAPA-TDNN trained at 16kHz mono
ENERGY_VAD_FRAME_S = 0.025  # 25ms frame energy for VAD

# Default ONNX path; if file exists we use the CoreML-accelerated path
# (~5-10× faster than PyTorch CPU). Otherwise fall back to PyTorch via
# speechbrain — slower but always available.
DEFAULT_ONNX_PATH = os.environ.get(
    "ECAPA_ONNX",
    str(os.path.expanduser("~/.cache/tv-detect-daemon/models/ecapa.onnx")))


def extract_audio(input_path: str) -> np.ndarray:
    """ffmpeg → float32 mono 16kHz PCM as numpy."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", input_path,
        "-vn", "-ac", "1", "-ar", str(SR),
        "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def energy_vad(audio: np.ndarray, threshold: float) -> np.ndarray:
    """Per-sample boolean: is this sample inside a speech-energetic frame?

    Cheap RMS-energy VAD — silero-vad would be more accurate but adds a
    PyTorch model dependency. For ad/show separation the energy floor is
    enough: real speech is well above the music-bed and silence floors that
    surround silent or fade-to-black moments.
    """
    frame = int(SR * ENERGY_VAD_FRAME_S)
    if len(audio) < frame:
        return np.zeros(len(audio), dtype=bool)
    n_frames = len(audio) // frame
    framed = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(framed ** 2, axis=1))
    is_speech_frame = rms > threshold
    out = np.repeat(is_speech_frame, frame)
    if len(out) < len(audio):
        out = np.concatenate([out, np.zeros(len(audio) - len(out), dtype=bool)])
    return out


class _Encoder:
    """Wraps either onnxruntime+CoreML or PyTorch+speechbrain. Same call:
    encode_batch(np.ndarray (B, T)) → np.ndarray (B, 192)."""

    def __init__(self, mode: str, onnx_path: str = ""):
        self.mode = mode
        if mode == "onnx":
            import onnxruntime as ort
            avail = ort.get_available_providers()
            providers = []
            # CoreMLExecutionProvider with the legacy NeuralNetwork model
            # format crashes on STFT shape inference. The newer MLProgram
            # format (opt-in here) gets 170/256 ops onto Apple Neural
            # Engine + GPU, ~2× faster than CPU-only ONNX. Default ON
            # since validation showed parity stays at cosine 1.0.
            use_coreml = (
                "CoreMLExecutionProvider" in avail
                and os.environ.get("ECAPA_USE_COREML", "1") == "1"
            )
            if use_coreml:
                providers.append((
                    "CoreMLExecutionProvider",
                    {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
                ))
            providers.append("CPUExecutionProvider")
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(
                os.environ.get("ECAPA_INTRA_OP", "2"))
            self.sess = ort.InferenceSession(
                onnx_path, sess_options=so, providers=providers)
            print(f"  ECAPA ONNX providers: {self.sess.get_providers()}",
                  file=sys.stderr, flush=True)
        else:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            self._torch = torch
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )

    def encode_batch(self, batch: np.ndarray) -> np.ndarray:
        if self.mode == "onnx":
            out = self.sess.run(["embedding"], {"wavs": batch.astype(np.float32)})[0]
            return out.squeeze(1)  # (B, 192)
        with self._torch.no_grad():
            t = self._torch.from_numpy(batch)
            e = self.classifier.encode_batch(t)
        return e.squeeze(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=1.0,
                    help="window stride; smaller = more embeddings, more compute")
    ap.add_argument("--vad-energy", type=float, default=0.005,
                    help="per-frame RMS energy threshold for speech "
                         "(0.005 ≈ -40 dBFS). Windows with <50%% speech "
                         "frames get has_speech=False.")
    ap.add_argument("--onnx", default=DEFAULT_ONNX_PATH,
                    help="ECAPA ONNX model path (CoreML-accelerated). "
                         "If file exists, used instead of speechbrain PyTorch.")
    ap.add_argument("--force-pytorch", action="store_true",
                    help="ignore --onnx even if present and use PyTorch")
    args = ap.parse_args()

    print(f"loading audio from {args.input}", file=sys.stderr, flush=True)
    t0 = time.time()
    audio = extract_audio(args.input)
    audio_dur = len(audio) / SR
    print(f"  audio: {audio_dur:.1f}s ({len(audio)} samples) in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)

    use_onnx = (not args.force_pytorch) and os.path.exists(args.onnx)
    mode = "onnx" if use_onnx else "pytorch"
    print(f"loading ECAPA-TDNN ({mode})", file=sys.stderr, flush=True)
    t0 = time.time()
    encoder = _Encoder(mode, args.onnx if use_onnx else "")
    print(f"  loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    print(f"running VAD (energy={args.vad_energy})", file=sys.stderr, flush=True)
    speech_mask = energy_vad(audio, args.vad_energy)

    win = int(SR * args.window_s)
    hop = int(SR * args.hop_s)
    n_win = max(0, (len(audio) - win) // hop + 1)
    print(f"window={args.window_s}s hop={args.hop_s}s → {n_win} windows",
          file=sys.stderr, flush=True)

    times = np.zeros(n_win, dtype=np.float32)
    embeddings = np.zeros((n_win, 192), dtype=np.float32)
    has_speech = np.zeros(n_win, dtype=bool)

    BATCH = 32
    t0 = time.time()
    last_print = t0
    for i_start in range(0, n_win, BATCH):
        i_end = min(i_start + BATCH, n_win)
        chunks = []
        for i in range(i_start, i_end):
            s = i * hop
            chunks.append(audio[s : s + win])
            times[i] = s / SR
            speech_in_window = speech_mask[s : s + win]
            has_speech[i] = float(speech_in_window.mean()) >= 0.5
        # Pad partial last batch to fixed BATCH so the ONNX graph (which
        # has a baked-in batch dim of 32 — required for CoreML MLProgram /
        # MPS to compile static graphs) accepts it. Extra rows discarded.
        n_real = i_end - i_start
        if n_real < BATCH:
            pad = np.zeros((BATCH - n_real, win), dtype=np.float32)
            chunks.extend([pad[k] for k in range(BATCH - n_real)])
        batch_arr = np.stack(chunks).astype(np.float32)
        emb = encoder.encode_batch(batch_arr)  # (BATCH, 192)
        embeddings[i_start:i_end] = emb[:n_real]
        # L2-normalize so downstream cosine == dot product
        norms = np.linalg.norm(embeddings[i_start:i_end], axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings[i_start:i_end] /= norms
        if time.time() - last_print > 5:
            elapsed = time.time() - t0
            done_audio = (i_end * hop) / SR
            rt = done_audio / elapsed if elapsed > 0 else 0
            print(f"  {i_end}/{n_win} windows ({done_audio:.0f}s audio, "
                  f"{elapsed:.1f}s wall, {rt:.1f}× realtime)",
                  file=sys.stderr, flush=True)
            last_print = time.time()

    elapsed = time.time() - t0
    rt = audio_dur / elapsed if elapsed > 0 else 0
    print(f"done: {n_win} embeddings in {elapsed:.1f}s ({rt:.1f}× realtime), "
          f"speech-frames={has_speech.sum()}/{n_win} "
          f"({100*has_speech.mean():.0f}%)",
          file=sys.stderr, flush=True)

    np.savez_compressed(args.output,
                        times=times,
                        embeddings=embeddings,
                        has_speech=has_speech)
    print(f"wrote {args.output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
