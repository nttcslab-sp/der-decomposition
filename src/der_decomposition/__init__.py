"""Pause-tolerant evaluation of speaker diarization."""

from .metric import (
    DER_CORE,
    DER_PAUSE,
    FALSE_ALARM_CORE,
    FALSE_ALARM_PAUSE,
    MISS_CORE,
    MISS_PAUSE,
    DecomposedDiarizationErrorRate,
)

__all__ = [
    "DER_CORE",
    "DER_PAUSE",
    "FALSE_ALARM_CORE",
    "FALSE_ALARM_PAUSE",
    "MISS_CORE",
    "MISS_PAUSE",
    "DecomposedDiarizationErrorRate",
]

__version__ = "0.1.0"
