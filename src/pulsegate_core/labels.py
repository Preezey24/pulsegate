"""Annotation symbol filtering and AAMI EC57 class mapping. See dataset.md §5."""

from __future__ import annotations

import numpy as np

# MIT-BIH annotation symbol → AAMI EC57 class.
# Source: dataset.md §5 label-space table.
_SYMBOL_TO_AAMI: dict[str, str] = {
    # N: normal sinus + bundle branch blocks + escape beats
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S: supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V: ventricular ectopic
    "V": "V", "E": "V",
    # F: fusion of ventricular + normal
    "F": "F",
    # Q: paced / unknown / unclassifiable
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}

BEAT_SYMBOLS: frozenset[str] = frozenset(_SYMBOL_TO_AAMI.keys())
NON_BEAT_SYMBOLS: frozenset[str] = frozenset({"+", "~", "|", "[", "]", '"'})


def aami_class(symbol: str) -> str:
    """Map an MIT-BIH annotation symbol to its AAMI EC57 class (N/S/V/F/Q)."""
    try:
        return _SYMBOL_TO_AAMI[symbol]
    except KeyError:
        raise KeyError(
            f"Symbol {symbol!r} has no AAMI mapping. "
            f"Known beat symbols: {sorted(BEAT_SYMBOLS)}. "
            f"Non-beat symbols should be dropped first via filter_beats()."
        ) from None


def filter_beats(
    ann_samples: np.ndarray, ann_symbols: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Drop non-beat annotations. Returns parallel arrays of beat events only."""
    mask = np.fromiter(
        (s in BEAT_SYMBOLS for s in ann_symbols), dtype=bool, count=len(ann_symbols)
    )
    kept_samples = ann_samples[mask]
    kept_symbols = tuple(s for s, keep in zip(ann_symbols, mask) if keep)
    return kept_samples, kept_symbols
