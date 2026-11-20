"""Audio decoding via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")


def load_stereo(path: Path, sample_rate: int) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 samples of shape (n, 2). Mono input is duplicated."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sample_rate), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def has_distinct_channels(stereo: np.ndarray) -> bool:
    """True if left and right differ (Anarlog: mic vs call), False for mono or duplicated-mono files."""
    return not np.allclose(stereo[:, 0], stereo[:, 1], atol=1e-4)
