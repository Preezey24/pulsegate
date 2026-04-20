"""Engineered R-R temporal features (Chazal et al. 2004). See dataset.md §8."""

from __future__ import annotations

import numpy as np

# Number of preceding R-R intervals averaged into local_avg_rr (dataset.md §8).
LOCAL_AVG_WINDOW = 10


def rr_features(
    beat_samples: np.ndarray, idx: int, fs: int = 360
) -> dict[str, float | None]:
    """Four R-R temporal features for beat `idx` in an already-filtered R-peak array.

    Returns a dict with pre_rr, post_rr, local_avg_rr, rr_ratio (seconds or ratio).
    Features that can't be computed at record boundaries return None:
      - pre_rr / rr_ratio: None for the first beat (no previous R-peak)
      - post_rr: None for the last beat (no next R-peak)
      - local_avg_rr: None for the first beat only (needs at least one prior interval)
    See dataset.md §8 for formulas and rationale.
    """
    n = len(beat_samples)
    if idx < 0 or idx >= n:
        raise IndexError(f"idx {idx} out of range for {n} beats")

    pre_rr = (beat_samples[idx] - beat_samples[idx - 1]) / fs if idx > 0 else None
    post_rr = (beat_samples[idx + 1] - beat_samples[idx]) / fs if idx < n - 1 else None

    start = max(0, idx - LOCAL_AVG_WINDOW)
    preceding = beat_samples[start : idx + 1]
    local_avg_rr = (
        float(np.mean(np.diff(preceding)) / fs) if len(preceding) >= 2 else None
    )

    rr_ratio = (
        pre_rr / local_avg_rr
        if pre_rr is not None and local_avg_rr is not None and local_avg_rr > 0
        else None
    )

    return {
        "pre_rr": float(pre_rr) if pre_rr is not None else None,
        "post_rr": float(post_rr) if post_rr is not None else None,
        "local_avg_rr": local_avg_rr,
        "rr_ratio": float(rr_ratio) if rr_ratio is not None else None,
    }
