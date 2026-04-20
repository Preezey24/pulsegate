"""Flat feature-vector composition for sklearn training/eval. See dataset.md §8."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pulsegate_core.pipeline import BeatSample
from pulsegate_core.windowing import WINDOW_SIZE

# Per-beat feature vector: 252 z-scored window samples + 4 temporal scalars.
TEMPORAL_KEYS: tuple[str, ...] = ("pre_rr", "post_rr", "local_avg_rr", "rr_ratio")
FEATURE_LENGTH: int = WINDOW_SIZE + len(TEMPORAL_KEYS)  # 256


def beat_to_feature_vector(beat: BeatSample) -> np.ndarray:
    """Concatenate the z-scored window (252 floats) with the 4 temporal scalars.

    None temporal values (edge beats with no prev/next R-peak) are imputed as 0.0 —
    a baseline simplification. If premature-beat detection at record boundaries
    becomes a priority, revisit with a sentinel value or learned imputation.
    """
    temporal = np.array(
        [beat.temporal[k] if beat.temporal[k] is not None else 0.0 for k in TEMPORAL_KEYS],
        dtype=np.float32,
    )
    return np.concatenate([beat.window, temporal])


def beats_to_matrix(beats: Iterable[BeatSample]) -> tuple[np.ndarray, np.ndarray]:
    """Materialize an iterable of BeatSamples into (X, y) for sklearn.

    X: (N, FEATURE_LENGTH) float32 feature matrix.
    y: (N,) object array of AAMI class labels ('N', 'S', 'V', 'F', 'Q').
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for b in beats:
        rows.append(beat_to_feature_vector(b))
        labels.append(b.aami_class)
    X = np.stack(rows) if rows else np.empty((0, FEATURE_LENGTH), dtype=np.float32)
    y = np.array(labels, dtype=object)
    return X, y
