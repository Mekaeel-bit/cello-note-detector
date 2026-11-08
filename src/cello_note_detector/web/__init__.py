"""Local website: upload cello video → fused MusicXML with fingerings/positions."""

from cello_note_detector.web.pipeline import process_video_to_score

__all__ = ["process_video_to_score"]
