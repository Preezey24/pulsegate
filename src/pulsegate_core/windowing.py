"""Beat-window extraction and per-beat z-score normalization. See dataset.md §7."""

from __future__ import annotations

import numpy as np

# Window geometry — see dataset.md §7. Asymmetric by ECG physiology:
# more diagnostic signal after R-peak (QRS tail + T-wave) than before it.
PRE_PEAK_SAMPLES: int = 90                                  # ~250 ms at 360 Hz
POST_PEAK_SAMPLES: int = 162                                # ~450 ms at 360 Hz
WINDOW_SIZE: int = PRE_PEAK_SAMPLES + POST_PEAK_SAMPLES     # 252; R-peak at local index 90


def extract_window(signal: np.ndarray, r_peak_idx: int) -> np.ndarray | None:
    """Slice a 252-sample window around an R-peak.

    R-peak sits at local index PRE_PEAK_SAMPLES (=90). Returns None if the
    window would extend past either edge (first/last few beats of a record).
    Works for 1D (single-channel) or 2D (N, C) signal arrays.
    """
    start = r_peak_idx - PRE_PEAK_SAMPLES
    end = r_peak_idx + POST_PEAK_SAMPLES
    if start < 0 or end > signal.shape[0]:
        return None
    return signal[start:end]


def zscore(window: np.ndarray) -> np.ndarray:
    """Per-beat z-score: (x - mean) / std, cast to float32 for transport.

    Removes the 2-3× per-patient voltage-scale variation (dataset.md §2) so
    the model learns morphology. Returns zeros if the window is flat (std=0,
    rare but possible under signal saturation).
    """
    mean = window.mean()
    std = window.std()
    if std == 0:
        return np.zeros(window.shape, dtype=np.float32)
    return ((window - mean) / std).astype(np.float32)
