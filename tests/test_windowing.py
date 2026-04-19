"""Tests for pulsegate_core.windowing — R-peak window extraction + z-score normalization."""

import numpy as np

from pulsegate_core.io import mlii_channel_index
from pulsegate_core.labels import filter_beats
from pulsegate_core.windowing import (
    POST_PEAK_SAMPLES,
    PRE_PEAK_SAMPLES,
    WINDOW_SIZE,
    extract_window,
    zscore,
)


def test_window_geometry_is_252_samples():
    """Geometry invariant from dataset.md §7."""
    assert PRE_PEAK_SAMPLES + POST_PEAK_SAMPLES == WINDOW_SIZE == 252


def test_extract_window_places_r_peak_at_pre_peak_index():
    """Synthetic spike at sample 1000 — window's argmax should land at local index PRE_PEAK_SAMPLES."""
    signal = np.zeros(2000)
    signal[1000] = 1.0
    w = extract_window(signal, r_peak_idx=1000)
    assert w.shape == (WINDOW_SIZE,)
    assert int(np.argmax(w)) == PRE_PEAK_SAMPLES


def test_extract_window_returns_none_at_edges():
    """R-peak too close to either edge → window can't form, return None."""
    signal = np.zeros(2000)
    assert extract_window(signal, r_peak_idx=PRE_PEAK_SAMPLES - 1) is None
    assert extract_window(signal, r_peak_idx=2000 - POST_PEAK_SAMPLES + 1) is None


def test_extract_window_preserves_channel_dim_for_2d_signal():
    """(N, C) input → (WINDOW_SIZE, C) output (channel-agnostic slicing)."""
    w = extract_window(np.zeros((2000, 2)), r_peak_idx=1000)
    assert w.shape == (WINDOW_SIZE, 2)


def test_zscore_normalizes_to_unit_mean_std_float32():
    z = zscore(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert z.dtype == np.float32
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1.0) < 1e-6


def test_zscore_flat_window_returns_zeros_not_nan():
    """Zero-std input → zeros of float32, not NaN (rare but possible under signal saturation)."""
    z = zscore(np.full(WINDOW_SIZE, 0.5))
    assert z.dtype == np.float32
    assert np.all(z == 0)


def test_extract_window_integration_on_record_100(record_100):
    """Real record: nearly every beat yields a valid window, only edge beats drop."""
    mlii = record_100.signal[:, mlii_channel_index(record_100)]
    beat_samples, _ = filter_beats(record_100.ann_samples, record_100.ann_symbols)

    # Sanity: record 100 really has thousands of beats (guards against upstream failure).
    assert len(beat_samples) > 2000

    # Contract: extract_window only drops beats whose window would overrun the recording edges.
    valid = sum(1 for s in beat_samples if extract_window(mlii, int(s)) is not None)
    assert valid >= len(beat_samples) - 5
