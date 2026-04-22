"""Evaluate the baseline RandomForest on DS2 and write versioned metrics JSON.

Usage:
    uv run python scripts/eval_baseline.py

Loads models/baseline.joblib, runs predictions against all 22 DS2 records,
computes per-class metrics + confusion matrix + per-record macro-F1, and
serialises to eval/baseline_v0.json for regression tracking and CI gating.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from pulsegate_core.features import beats_to_matrix
from pulsegate_core.io import DS2_RECORDS
from pulsegate_core.pipeline import iter_beats

MODEL_PATH = Path("models/baseline.joblib")
OUTPUT_PATH = Path("eval/baseline_v0.json")
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    train = set(artifact["train_records"])
    overlap = train & set(DS2_RECORDS)
    assert not overlap, f"Data leakage: DS2 records appear in train set: {overlap}"

    print(f"Model: {MODEL_PATH} (trained {artifact['trained_at']})")
    print(f"Evaluating on {len(DS2_RECORDS)} DS2 records...")

    # Load DS2 record-by-record, tracking beat-index ranges per record for per-record F1.
    t0 = time.perf_counter()
    X_parts, y_parts, ranges = [], [], {}
    offset = 0
    for rec in DS2_RECORDS:
        X_rec, y_rec = beats_to_matrix(iter_beats(rec))
        if len(y_rec):
            X_parts.append(X_rec)
            y_parts.append(y_rec)
            ranges[rec] = (offset, offset + len(y_rec))
            offset += len(y_rec)

    X_test = np.vstack(X_parts)
    y_test = np.concatenate(y_parts)
    y_pred = model.predict(X_test)
    print(f"Predicted {len(y_pred):,} beats in {time.perf_counter()-t0:.1f}s")

    report = classification_report(
        y_test, y_pred, labels=AAMI_CLASSES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=AAMI_CLASSES).tolist()
    per_record = {
        rec: float(f1_score(
            y_test[s:e], y_pred[s:e], labels=AAMI_CLASSES, average="macro", zero_division=0
        ))
        for rec, (s, e) in ranges.items()
    }

    result = {
        "version": OUTPUT_PATH.stem,
        "model_path": str(MODEL_PATH),
        "model_trained_at": artifact["trained_at"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": f"MIT-BIH DS2 ({len(DS2_RECORDS)} records)",
        "records": list(DS2_RECORDS),
        "n_beats": int(len(y_test)),
        "aami_classes": AAMI_CLASSES,
        "per_class": {c: report[c] for c in AAMI_CLASSES},
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "confusion_matrix": cm,
        "per_record_macro_f1": per_record,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))

    print(f"\nmacro_f1    = {result['macro_f1']:.3f}")
    print(f"weighted_f1 = {result['weighted_f1']:.3f}")
    print(f"accuracy    = {result['accuracy']:.3f}")
    print("\nPer-class F1:")
    for c in AAMI_CLASSES:
        m = result["per_class"][c]
        print(f"  {c}: {m['f1-score']:.3f}  (support={m['support']})")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
