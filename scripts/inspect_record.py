"""
Inspect a single MIT-BIH record. Prints schema-relevant facts and saves a plot.

Usage:
    uv run python scripts/inspect_record.py            # defaults to record 100
    uv run python scripts/inspect_record.py 203        # any record id
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

DATA_DIR = Path("data/mitdb")
PLOT_DIR = Path("data/plots")


def inspect(record_id: str) -> None:
    record_path = str(DATA_DIR / record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    print(f"\n{'=' * 60}")
    print(f"Record {record_id}")
    print(f"{'=' * 60}")

    # --- Signal metadata ---
    print("\n[Signal]")
    print(f"  Sampling rate (fs):     {record.fs} Hz")
    print(f"  Channels (n_sig):       {record.n_sig}")
    print(f"  Lead names:             {record.sig_name}")
    print(f"  Units:                  {record.units}")
    print(f"  Duration (samples):     {record.sig_len:,}")
    print(f"  Duration (seconds):     {record.sig_len / record.fs:,.1f}")
    print(f"  Duration (minutes):     {record.sig_len / record.fs / 60:.2f}")
    print(f"  Signal shape:           {record.p_signal.shape}")
    print(f"  Signal dtype:           {record.p_signal.dtype}")

    # --- Per-channel signal stats ---
    print("\n[Per-channel signal stats]")
    for i, lead in enumerate(record.sig_name):
        col = record.p_signal[:, i]
        print(
            f"  ch{i} ({lead}): "
            f"min={col.min():+.3f} max={col.max():+.3f} "
            f"mean={col.mean():+.4f} std={col.std():.4f}"
        )

    # --- Annotation summary ---
    print("\n[Annotations]")
    print(f"  Total annotations:      {len(annotation.sample):,}")
    print(f"  First annotation @ sample {annotation.sample[0]:,} "
          f"(t={annotation.sample[0] / record.fs:.3f}s)")
    print(f"  Last annotation @ sample {annotation.sample[-1]:,} "
          f"(t={annotation.sample[-1] / record.fs:.3f}s)")

    sym_counts = Counter(annotation.symbol)
    print(f"\n  Symbol counts (sorted by frequency):")
    total = sum(sym_counts.values())
    for sym, cnt in sym_counts.most_common():
        pct = 100 * cnt / total
        print(f"    {sym!r:6}  {cnt:>7,}  ({pct:5.2f}%)")

    # --- Inter-annotation interval stats (proxy for R-R intervals) ---
    diffs = np.diff(annotation.sample)
    diffs_sec = diffs / record.fs
    print(f"\n[Inter-annotation intervals (proxy for R-R)]")
    print(f"  count: {len(diffs):,}")
    print(f"  min: {diffs_sec.min():.3f}s   max: {diffs_sec.max():.3f}s")
    print(f"  mean: {diffs_sec.mean():.3f}s   median: {np.median(diffs_sec):.3f}s")
    print(f"  implied avg HR: {60 / diffs_sec.mean():.1f} bpm")

    # --- Plot first 10 seconds with annotations ---
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    n_samples = int(10 * record.fs)
    t = np.arange(n_samples) / record.fs

    fig, axes = plt.subplots(record.n_sig, 1, figsize=(14, 3 * record.n_sig), sharex=True)
    if record.n_sig == 1:
        axes = [axes]

    ann_in_window = annotation.sample[annotation.sample < n_samples]
    ann_syms_in_window = [s for i, s in enumerate(annotation.symbol) if annotation.sample[i] < n_samples]

    for i, lead in enumerate(record.sig_name):
        ax = axes[i]
        ax.plot(t, record.p_signal[:n_samples, i], linewidth=0.6)
        for samp, sym in zip(ann_in_window, ann_syms_in_window):
            ax.axvline(samp / record.fs, color="r", alpha=0.25, linewidth=0.5)
            ax.text(samp / record.fs, ax.get_ylim()[1] * 0.92, sym,
                    fontsize=8, ha="center", color="r")
        ax.set_ylabel(f"ch{i} {lead} ({record.units[i]})")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Record {record_id} — first 10 seconds with annotations")
    fig.tight_layout()

    plot_path = PLOT_DIR / f"record_{record_id}_10s.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\n[Plot] Saved {plot_path}")


if __name__ == "__main__":
    record_id = sys.argv[1] if len(sys.argv) > 1 else "100"
    inspect(record_id)
