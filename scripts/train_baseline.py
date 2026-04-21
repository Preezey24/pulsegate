"""Train the RandomForest baseline on all of DS1 and save to models/baseline.joblib.

Usage:
    uv run python scripts/train_baseline.py

Trains on all 22 DS1 records (no internal val split). See docs/decisions.md
for the deliberation behind this choice. Evaluation happens separately in Step C
against DS2 (the canonical test set per Chazal et al. 2004).
"""

from __future__ import annotations

import itertools
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from pulsegate_core.features import beats_to_matrix
from pulsegate_core.io import DS1_RECORDS
from pulsegate_core.pipeline import iter_beats

MODEL_PATH = Path("models/baseline.joblib")


def main() -> None:
    print(f"Training on all {len(DS1_RECORDS)} DS1 records")

    t0 = time.perf_counter()
    beats = itertools.chain.from_iterable(iter_beats(r) for r in DS1_RECORDS)
    X_train, y_train = beats_to_matrix(beats)
    print(f"Built X_train={X_train.shape} in {time.perf_counter()-t0:.1f}s")

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    t1 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Fit complete in {time.perf_counter()-t1:.1f}s")
    print(f"Learned classes: {sorted(model.classes_)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "train_records": DS1_RECORDS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        MODEL_PATH,
    )
    print(f"Saved artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()
