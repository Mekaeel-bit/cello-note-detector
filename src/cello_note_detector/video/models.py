"""Download MediaPipe Tasks model bundles on demand."""

from __future__ import annotations

import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "video" / "models"

HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

HAND_NAME = "hand_landmarker.task"
POSE_NAME = "pose_landmarker_lite.task"


def ensure_models(model_dir: Path | None = None) -> dict[str, Path]:
    """Ensure hand + pose `.task` files exist; download if missing."""
    root = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, url in ((HAND_NAME, HAND_URL), (POSE_NAME, POSE_URL)):
        path = root / name
        if not path.is_file() or path.stat().st_size < 100_000:
            print(f"Downloading MediaPipe model → {path}")
            urllib.request.urlretrieve(url, path)
        out[name] = path
    return out
