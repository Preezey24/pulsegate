"""Tests for pulsegate_core.io — record loading + MIT-BIH split constants."""

import dataclasses

import numpy as np
import pytest

from pulsegate_core.io import (
    DS1_RECORDS,
    DS2_RECORDS,
    EXCLUDED_RECORDS,
    load_record,
    mlii_channel_index,
)


def test_split_sizes_match_chazal_et_al_2004():
    """22/22/4 split = 48 total records, matching the canonical partition (dataset.md §6)."""
    assert len(DS1_RECORDS) == 22
    assert len(DS2_RECORDS) == 22
    assert len(EXCLUDED_RECORDS) == 4
    assert len(DS1_RECORDS) + len(DS2_RECORDS) + len(EXCLUDED_RECORDS) == 48


def test_splits_are_disjoint():
    """No record appears in multiple splits — required for inter-patient evaluation."""
    ds1, ds2, exc = set(DS1_RECORDS), set(DS2_RECORDS), set(EXCLUDED_RECORDS)
    assert ds1.isdisjoint(ds2) and ds1.isdisjoint(exc) and ds2.isdisjoint(exc)


def test_load_record_returns_record_with_expected_shape(record_100):
    """Record 100: 650,000 samples × 2 channels at 360 Hz; parallel annotation arrays."""
    assert record_100.record_id == "100"
    assert record_100.fs == 360
    assert record_100.signal.shape == (650000, 2)
    assert len(record_100.ann_samples) == len(record_100.ann_symbols)


def test_load_record_normalizes_wfdb_types(record_100):
    """Normalize wfdb's mutable lists and loose dtypes to immutable tuples + pinned int64."""
    assert record_100.ann_samples.dtype == np.int64
    assert isinstance(record_100.sig_name, tuple)
    assert isinstance(record_100.units, tuple)
    assert isinstance(record_100.ann_symbols, tuple)


def test_record_is_frozen(record_100):
    """Loaded records are immutable — prevents accidental cross-module mutation."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        record_100.fs = 999


def test_mlii_channel_index_on_record_100(record_100):
    """Record 100 has MLII on channel 0."""
    assert mlii_channel_index(record_100) == 0


def test_mlii_channel_index_raises_for_record_without_mlii():
    """Record 102 has V5/V2 instead — must raise, not silently return 0 (dataset.md §10)."""
    r102 = load_record("102")
    with pytest.raises(ValueError, match="no MLII lead"):
        mlii_channel_index(r102)
