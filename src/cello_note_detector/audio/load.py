"""Load and normalize monophonic audio for the cello pipeline."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

# MERT-v1-95M expects 24 kHz input.
TARGET_SR = 24_000


def load_mono(
    path: str | Path,
    *,
    sr: int = TARGET_SR,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32 at ``sr`` (default 24 kHz).

    Returns
    -------
    audio : np.ndarray
        1-D waveform, float32 in [-1, 1].
    sample_rate : int
        Actual sample rate used (equals ``sr`` after resampling).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        duration=duration,
        dtype=np.float32,
    )
    return audio, int(sample_rate)
