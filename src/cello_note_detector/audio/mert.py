"""MERT-v1-95M frame embedding extraction (no trained pitch head)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .load import TARGET_SR

DEFAULT_MODEL_ID = "m-a-p/MERT-v1-95M"
# Keep chunks short enough for CPU / modest GPU memory.
DEFAULT_CHUNK_SECONDS = 5.0
DEFAULT_OVERLAP_SECONDS = 0.5


@dataclass
class MertBundle:
    model_id: str
    feature_extractor: Any
    model: Any
    device: torch.device
    hidden_size: int


def load_mert(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str | None = None,
) -> MertBundle:
    """Load MERT encoder + feature extractor from Hugging Face."""
    from transformers import AutoFeatureExtractor, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(torch_device)
    model.eval()

    hidden_size = int(getattr(model.config, "hidden_size", 768))
    return MertBundle(
        model_id=model_id,
        feature_extractor=feature_extractor,
        model=model,
        device=torch_device,
        hidden_size=hidden_size,
    )


def _chunk_bounds(
    n_samples: int,
    sr: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[int, int]]:
    """Return (start, end) sample indices for overlapping chunks."""
    chunk_len = max(1, int(chunk_seconds * sr))
    hop = max(1, int((chunk_seconds - overlap_seconds) * sr))
    if n_samples <= chunk_len:
        return [(0, n_samples)]

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(start + chunk_len, n_samples)
        bounds.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return bounds


def extract_mert_embeddings(
    audio: np.ndarray,
    bundle: MertBundle,
    *,
    sr: int = TARGET_SR,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> np.ndarray:
    """Extract frame embeddings shaped ``[T, hidden_size]`` (float32).

    Long audio is processed in overlapping chunks; overlapping frames from
    later chunks overwrite earlier ones (simple stitch).
    """
    if sr != TARGET_SR:
        raise ValueError(
            f"MERT expects {TARGET_SR} Hz audio; got sr={sr}. "
            "Resample with load_mono first."
        )

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    bounds = _chunk_bounds(len(audio), sr, chunk_seconds, overlap_seconds)

    # Approximate MERT frame rate (~75 Hz) for stitching by time.
    frame_hz = 75.0
    total_frames = max(1, int(round(len(audio) / sr * frame_hz)))
    out = np.zeros((total_frames, bundle.hidden_size), dtype=np.float32)
    filled = np.zeros(total_frames, dtype=bool)

    for start, end in bounds:
        chunk = audio[start:end]
        inputs = bundle.feature_extractor(
            chunk, sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bundle.model(**inputs)

        frames = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
        frames = np.asarray(frames, dtype=np.float32)

        # Map chunk frames onto the global timeline by chunk start time.
        t0 = start / float(sr)
        for local_i in range(frames.shape[0]):
            global_i = int(round((t0 + local_i / frame_hz) * frame_hz))
            if 0 <= global_i < total_frames:
                out[global_i] = frames[local_i]
                filled[global_i] = True

        # If this is a short file, prefer exact length from the model.
        if len(bounds) == 1:
            return frames

    # Trim trailing unfilled frames when estimate overshot.
    if filled.any():
        last = int(np.where(filled)[0][-1]) + 1
        out = out[:last]
        filled = filled[:last]
        # Forward-fill any internal holes from overlap math.
        if not filled.all():
            for i in range(1, len(out)):
                if not filled[i] and filled[i - 1]:
                    out[i] = out[i - 1]
                    filled[i] = True

    return out


def summarize_embeddings(embeddings: np.ndarray) -> dict:
    """Compact stats for console / JSON without dumping the full matrix."""
    arr = np.asarray(embeddings, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "l2_norm_mean": float(np.linalg.norm(arr, axis=-1).mean())
        if arr.ndim == 2 and arr.shape[0] > 0
        else 0.0,
    }
