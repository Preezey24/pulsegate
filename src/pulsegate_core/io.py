"""MIT-BIH record loading. Thin wrapper over wfdb + split constants from dataset.md §6."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

DS1_RECORDS = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2_RECORDS = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)
EXCLUDED_RECORDS = ("102", "104", "107", "217")

DEFAULT_DATA_DIR = Path("data/mitdb")


@dataclass(frozen=True)
class Record:
    record_id: str
    fs: int
    signal: np.ndarray            # shape (N, n_channels), float64, mV
    sig_name: tuple[str, ...]     # lead names, e.g. ("MLII", "V5")
    units: tuple[str, ...]
    ann_samples: np.ndarray       # int sample indices
    ann_symbols: tuple[str, ...]  # single-char annotation symbols


def load_record(record_id: str, data_dir: Path = DEFAULT_DATA_DIR) -> Record:
    """Load one MIT-BIH record + its .atr annotations."""
    path = str(data_dir / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    return Record(
        record_id=record_id,
        fs=int(rec.fs),
        signal=rec.p_signal,
        sig_name=tuple(rec.sig_name),
        units=tuple(rec.units),
        ann_samples=np.asarray(ann.sample, dtype=np.int64),
        ann_symbols=tuple(ann.symbol),
    )


def mlii_channel_index(record: Record) -> int:
    """Return channel index of MLII lead. Raises ValueError if record lacks MLII."""
    try:
        return record.sig_name.index("MLII")
    except ValueError:
        raise ValueError(
            f"Record {record.record_id} has no MLII lead (sig_name={record.sig_name})"
        ) from None
