"""Audio normalisation to the training contract: 16 kHz, mono, PCM_16.

Every utterance fed to the released model was normalised identically:
  1. read as float32
  2. downmix to mono (mean over channels) if multi-channel
  3. resample to 16 000 Hz when needed (polyphase, exact-ratio via gcd)
  4. write as 16-bit PCM

This is the contract the SFT data and the evaluation audio both follow, so a
user with their own Korean audio can build data that the model consumes the
same way it consumed the training data. Deterministic, no model in the loop.
"""
from __future__ import annotations
import io
from pathlib import Path

TARGET_SR = 16000


def _to_mono(x, sr):
    import numpy as np
    if x.ndim > 1:
        x = x.mean(1)
    return x, sr


def resample_to_16k(x, sr: int):
    """Return (float32 mono @ 16 000 Hz, 16000). No-op if already 16 kHz."""
    import numpy as np
    from scipy.signal import resample_poly
    x, sr = _to_mono(x, sr)
    if sr != TARGET_SR:
        g = int(np.gcd(sr, TARGET_SR))
        x = resample_poly(x, TARGET_SR // g, sr // g).astype("float32")
    else:
        x = np.asarray(x, dtype="float32")
    return x, TARGET_SR


def normalize_wav(in_path: str | Path, out_path: str | Path) -> dict:
    """Read any wav, normalise to 16 kHz mono PCM_16, write to out_path."""
    import soundfile as sf
    in_path, out_path = Path(in_path), Path(out_path)
    x, sr_in = sf.read(str(in_path), dtype="float32")
    x, sr_out = resample_to_16k(x, sr_in)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), x, sr_out, subtype="PCM_16")
    return {"in": str(in_path), "out": str(out_path),
            "sr_in": int(sr_in), "sr_out": int(sr_out), "duration_s": round(len(x) / sr_out, 3)}


def read_wav_bytes(raw: bytes):
    """In-memory variant (for data already in hand): -> (float32 mono @16k, sr)."""
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return resample_to_16k(x, sr)
