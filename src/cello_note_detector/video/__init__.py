"""Video fingering / bowing sandbox (MediaPipe + OpenCV + optional AV fusion)."""

from cello_note_detector.video.detect import analyze_video
from cello_note_detector.video.fusion import fuse_timeline
from cello_note_detector.video.sheet import write_fused_musicxml
from cello_note_detector.video.strings import pitch_string_hypotheses

__all__ = [
    "analyze_video",
    "fuse_timeline",
    "pitch_string_hypotheses",
    "write_fused_musicxml",
]
