"""Tests for pulsegate_core.pipeline — BeatSample iterator composing io + labels + windowing + temporal."""

import dataclasses

import numpy as np
import pytest

from pulsegate_core.labels import aami_class
from pulsegate_core.pipeline import BeatSample, iter_beats


def test_iter_beats_count_on_record_100():
    """Record 100: 2271 beats — total (2273) minus 2 edge beats that can't form a window."""
    assert len(list(iter_beats("100"))) == 2271


def test_beatsample_fields_populated_and_typed_correctly():
    """Comprehensive field check on a mid-record beat."""
    b = list(iter_beats("100"))[100]
    assert isinstance(b, BeatSample)
    assert b.record_id == "100"
    assert isinstance(b.beat_index, int) and isinstance(b.r_peak_sample, int)
    assert b.aami_class in {"N", "S", "V", "F", "Q"}
    assert b.window.shape == (252,) and b.window.dtype == np.float32
    assert set(b.temporal.keys()) == {"pre_rr", "post_rr", "local_avg_rr", "rr_ratio"}


def test_aami_class_field_matches_symbol_mapping():
    """Composition check: the AAMI field == aami_class(symbol) for every yielded beat."""
    for b in iter_beats("100"):
        assert b.aami_class == aami_class(b.symbol)


def test_iter_beats_is_lazy_until_consumed(monkeypatch):
    """Calling iter_beats returns a generator without running — body runs only on first next()."""
    from pulsegate_core import pipeline as pl

    def boom(record_id):
        raise RuntimeError(f"load_record({record_id!r}) should not run until iterated")

    monkeypatch.setattr(pl, "load_record", boom)
    gen = iter_beats("100")  # no-op; function body hasn't started
    with pytest.raises(RuntimeError):
        next(gen)  # now load_record runs, error propagates


def test_beatsample_is_frozen():
    """BeatSample instances are immutable — prevents accidental cross-module mutation."""
    b = next(iter_beats("100"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.record_id = "hacked"


def test_iter_beats_raises_for_record_without_mlii():
    """Record 102 (V5/V2 only) propagates the MLII error through the generator."""
    with pytest.raises(ValueError, match="no MLII lead"):
        next(iter_beats("102"))


def test_v_class_beats_show_prematurity_signature_on_record_203():
    """Full-pipeline clinical check: V-class beats on record 203 have mean rr_ratio < 1."""
    v_ratios = [
        b.temporal["rr_ratio"]
        for b in iter_beats("203")
        if b.aami_class == "V" and b.temporal["rr_ratio"] is not None
    ]
    assert len(v_ratios) > 100
    assert sum(v_ratios) / len(v_ratios) < 0.85
