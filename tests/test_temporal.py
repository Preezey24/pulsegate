"""Tests for pulsegate_core.temporal — Chazal R-R timing features."""

import numpy as np
import pytest

from pulsegate_core.labels import filter_beats
from pulsegate_core.temporal import rr_features


def test_steady_rhythm_yields_unit_features():
    """5 beats exactly 1 s apart at fs=360 → all features = 1.0, rr_ratio = 1.0."""
    beats = np.array([360, 720, 1080, 1440, 1800], dtype=np.int64)
    assert rr_features(beats, idx=2, fs=360) == {
        "pre_rr": 1.0, "post_rr": 1.0, "local_avg_rr": 1.0, "rr_ratio": 1.0,
    }


def test_first_beat_has_none_for_backward_looking_features():
    """No predecessor → pre_rr / local_avg_rr / rr_ratio None; post_rr still computable."""
    beats = np.array([360, 720, 1080], dtype=np.int64)
    f = rr_features(beats, idx=0, fs=360)
    assert f["pre_rr"] is None and f["local_avg_rr"] is None and f["rr_ratio"] is None
    assert f["post_rr"] == 1.0


def test_last_beat_has_none_for_post_rr():
    """No successor → post_rr None; pre_rr still computable."""
    beats = np.array([360, 720, 1080], dtype=np.int64)
    f = rr_features(beats, idx=2, fs=360)
    assert f["post_rr"] is None and f["pre_rr"] == 1.0


def test_premature_beat_yields_low_rr_ratio():
    """Simulated PVC: beat arrives 0.4s after prev on 1.0s baseline → rr_ratio ≈ 0.5, long compensatory pause follows."""
    beats = np.array([360, 720, 1080, 1224, 1800], dtype=np.int64)  # beat 3 is premature
    f = rr_features(beats, idx=3, fs=360)
    assert f["pre_rr"] == pytest.approx(0.4)
    assert f["post_rr"] == pytest.approx(1.6)
    assert f["rr_ratio"] == pytest.approx(0.5, rel=0.01)


def test_fs_scaling_divides_sample_gap_by_sampling_rate():
    """Same sample-gap at different fs yields different times (1000-sample gap / 1000 Hz = 1.0 s)."""
    beats = np.array([1000, 2000, 3000], dtype=np.int64)
    assert rr_features(beats, idx=1, fs=1000)["pre_rr"] == 1.0


def test_out_of_range_idx_raises_index_error():
    with pytest.raises(IndexError):
        rr_features(np.array([360, 720], dtype=np.int64), idx=99)


def test_local_avg_uses_at_most_window_preceding_intervals():
    """Switch rhythms mid-array; at a late index local_avg_rr reflects only the recent-fast-rhythm intervals."""
    slow = [360 * i for i in range(1, 11)]               # 10 beats at 1.0 s spacing
    fast = [slow[-1] + 180 * i for i in range(1, 11)]    # 10 beats at 0.5 s spacing
    beats = np.array(slow + fast, dtype=np.int64)
    f = rr_features(beats, idx=len(beats) - 1, fs=360)
    assert f["local_avg_rr"] == pytest.approx(0.5, rel=0.01)


def test_rr_features_on_real_record_in_physiological_range(record_100):
    """Sanity: record 100's features fall within plausible heart-rate ranges (30-150 bpm)."""
    beat_samples, _ = filter_beats(record_100.ann_samples, record_100.ann_symbols)
    f = rr_features(beat_samples, idx=100, fs=record_100.fs)
    for feat in ("pre_rr", "post_rr", "local_avg_rr"):
        assert 0.4 < f[feat] < 2.0
    assert 0.5 < f["rr_ratio"] < 2.0
