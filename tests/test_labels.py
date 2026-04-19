"""Tests for pulsegate_core.labels — annotation filtering and AAMI EC57 mapping."""

import numpy as np
import pytest

from pulsegate_core.labels import (
    BEAT_SYMBOLS,
    NON_BEAT_SYMBOLS,
    aami_class,
    filter_beats,
)


@pytest.mark.parametrize("symbol,expected_class", [
    ("N", "N"), ("L", "N"), ("R", "N"), ("e", "N"), ("j", "N"),
    ("A", "S"), ("a", "S"), ("J", "S"), ("S", "S"),
    ("V", "V"), ("E", "V"),
    ("F", "F"),
    ("/", "Q"), ("f", "Q"), ("Q", "Q"), ("?", "Q"),
])
def test_aami_class_maps_all_beat_symbols(symbol, expected_class):
    """Every MIT-BIH beat symbol maps to its documented AAMI class (dataset.md §5)."""
    assert aami_class(symbol) == expected_class


def test_aami_class_raises_on_unknown_or_non_beat_symbol():
    """Non-beat or unknown symbols must fail loudly, not silently mis-label."""
    with pytest.raises(KeyError, match="has no AAMI mapping"):
        aami_class("+")


def test_beat_and_non_beat_symbol_sets_are_disjoint():
    """Structural invariant: no symbol is both a beat and a non-beat."""
    assert BEAT_SYMBOLS.isdisjoint(NON_BEAT_SYMBOLS)


def test_filter_beats_drops_non_beats_and_preserves_alignment():
    """Toy input with non-beats interspersed — output keeps beat positions only, parallel arrays stay aligned."""
    samples = np.array([18, 77, 370, 500, 662, 946], dtype=np.int64)
    symbols = ("+", "N", "N", "~", "N", "A")
    kept_samples, kept_symbols = filter_beats(samples, symbols)
    assert kept_symbols == ("N", "N", "N", "A")
    np.testing.assert_array_equal(kept_samples, [77, 370, 662, 946])


def test_filter_beats_all_non_beats_returns_empty_arrays():
    """All-non-beat input returns empty outputs, dtype preserved."""
    kept_samples, kept_symbols = filter_beats(
        np.array([10, 20], dtype=np.int64), ("+", "~")
    )
    assert kept_symbols == ()
    assert kept_samples.dtype == np.int64
    assert len(kept_samples) == 0
