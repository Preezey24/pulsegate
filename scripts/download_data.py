"""Download MIT-BIH Arrhythmia Database (48 records) from PhysioNet into data/mitdb/.

Usage:
    uv run python scripts/download_data.py

Idempotent — re-running verifies the full dataset is present and re-fetches any
missing records. Already-present files are not re-downloaded (wfdb.dl_database
skips existing files), so subsequent runs are fast. PhysioNet downloads occasionally
drop individual records silently, so the script always reports the final count.
"""

from __future__ import annotations

from pathlib import Path

import wfdb

DATA_DIR = Path("data/mitdb")
EXPECTED_RECORD_COUNT = 48  # MIT-BIH has 48 records; see dataset.md §2


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MIT-BIH into {DATA_DIR}/ ...")
    wfdb.dl_database("mitdb", dl_dir=str(DATA_DIR))

    dat_files = list(DATA_DIR.glob("*.dat"))
    print(f"Landed {len(dat_files)} .dat files in {DATA_DIR}/")
    if len(dat_files) < EXPECTED_RECORD_COUNT:
        print(
            f"WARNING: expected {EXPECTED_RECORD_COUNT} records, got {len(dat_files)}. "
            f"Re-run the script — PhysioNet occasionally drops records silently."
        )
    else:
        print(f"OK — all {EXPECTED_RECORD_COUNT} records present.")


if __name__ == "__main__":
    main()
