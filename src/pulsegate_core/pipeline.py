"""Per-beat preprocessing iterator. Composes io + labels + windowing + temporal."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pulsegate_core.io import load_record, mlii_channel_index
from pulsegate_core.labels import aami_class, filter_beats
from pulsegate_core.temporal import rr_features
from pulsegate_core.windowing import extract_window, zscore


@dataclass(frozen=True)
class BeatSample:
    record_id: str
    beat_index: int
    r_peak_sample: int
    symbol: str                         # raw MIT-BIH symbol
    aami_class: str                     # AAMI EC57: N / S / V / F / Q
    window: np.ndarray                  # (252,) float32, z-scored MLII
    temporal: dict[str, float | None]   # pre_rr, post_rr, local_avg_rr, rr_ratio


def iter_beats(record_id: str) -> Iterator[BeatSample]:
    """Yield one BeatSample per classifiable beat in a record.

    Pipeline per beat: filter non-beat annotations -> extract MLII window ->
    z-score + float32 cast -> rr_features -> assemble BeatSample. Skips edge beats
    where a full window doesn't fit. See design-notes.md §11 for memory lifecycle.
    """
    record = load_record(record_id)
    mlii = record.signal[:, mlii_channel_index(record)]
    beat_samples, beat_symbols = filter_beats(record.ann_samples, record.ann_symbols)

    for i, (r_peak, symbol) in enumerate(zip(beat_samples, beat_symbols)):
        raw = extract_window(mlii, int(r_peak))
        if raw is None:
            continue  # edge beat — window doesn't fit
        yield BeatSample(
            record_id=record_id,
            beat_index=i,
            r_peak_sample=int(r_peak),
            symbol=symbol,
            aami_class=aami_class(symbol),
            window=zscore(raw),
            temporal=rr_features(beat_samples, i, fs=record.fs),
        )
