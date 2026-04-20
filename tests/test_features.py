"""Tests for pulsegate_core.features — flat feature-vector composition for sklearn."""

import numpy as np

from pulsegate_core.features import (
    FEATURE_LENGTH,
    TEMPORAL_KEYS,
    beat_to_feature_vector,
    beats_to_matrix,
)
from pulsegate_core.pipeline import BeatSample, iter_beats
from pulsegate_core.windowing import WINDOW_SIZE


def _fake_beat(temporal: dict, window: np.ndarray | None = None) -> BeatSample:
    """Synthetic BeatSample for tests that need exact control over temporal / window values."""
    return BeatSample(
        record_id="synthetic", beat_index=0, r_peak_sample=1000,
        symbol="N", aami_class="N",
        window=window if window is not None else np.zeros(WINDOW_SIZE, dtype=np.float32),
        temporal=temporal,
    )


def test_feature_length_constants_are_consistent():
    """FEATURE_LENGTH = WINDOW_SIZE + len(TEMPORAL_KEYS) — derived, not hard-coded."""
    assert FEATURE_LENGTH == WINDOW_SIZE + len(TEMPORAL_KEYS)


def test_concat_order_is_window_then_temporal_in_key_order():
    """First WINDOW_SIZE entries are the window; trailing 4 are temporal values in TEMPORAL_KEYS order."""
    win = (np.arange(WINDOW_SIZE, dtype=np.float32) / 100)
    b = _fake_beat({"pre_rr": 0.1, "post_rr": 0.2, "local_avg_rr": 0.3, "rr_ratio": 0.4}, win)
    v = beat_to_feature_vector(b)
    assert v.shape == (FEATURE_LENGTH,) and v.dtype == np.float32
    np.testing.assert_array_equal(v[:WINDOW_SIZE], win)
    np.testing.assert_allclose(v[-4:], [0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_none_temporal_values_are_imputed_as_zero():
    """Synthetic beat with Nones → imputed to 0.0 (code path not reached by real records)."""
    b = _fake_beat({"pre_rr": None, "post_rr": 0.8, "local_avg_rr": None, "rr_ratio": None})
    v = beat_to_feature_vector(b)
    np.testing.assert_allclose(v[-4:], [0.0, 0.8, 0.0, 0.0], atol=1e-6)


def test_beats_to_matrix_shapes_dtypes_labels():
    """(X, y) shapes match beat count; X is float32, y is object-dtype AAMI strings."""
    beats = [
        _fake_beat({"pre_rr": 0.8, "post_rr": 0.8, "local_avg_rr": 0.8, "rr_ratio": 1.0}),
        _fake_beat({"pre_rr": 0.7, "post_rr": 0.9, "local_avg_rr": 0.8, "rr_ratio": 0.9}),
    ]
    X, y = beats_to_matrix(beats)
    assert X.shape == (2, FEATURE_LENGTH) and X.dtype == np.float32
    assert y.shape == (2,) and y.dtype == object and y.tolist() == ["N", "N"]


def test_beats_to_matrix_empty_iterator_preserves_shape():
    """Empty input → (0, FEATURE_LENGTH) and (0,) — shape preserved so sklearn can still consume."""
    X, y = beats_to_matrix(iter([]))
    assert X.shape == (0, FEATURE_LENGTH) and y.shape == (0,)


def test_beats_to_matrix_integration_on_record_100():
    """Real record: row count matches yielded beats, all values numeric (no NaNs leaking through)."""
    X, y = beats_to_matrix(iter_beats("100"))
    assert X.shape == (2271, FEATURE_LENGTH) and y.shape == (2271,)
    assert not np.isnan(X).any()
