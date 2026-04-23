"""Run a single consumer that reads beats from ecg_beat:in, classifies, writes to ecg_beat:out.

Usage:
    uv run python scripts/run_consumer.py
    uv run python scripts/run_consumer.py --redis-url redis://localhost:6379

V1: single worker (consumer name 'worker-1'), model loaded from models/baseline.joblib.
Requires Redis running (docker compose up -d). See design-notes.md §8 for topology.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import redis

from pulsegate_core.streaming import BeatConsumer

MODEL_PATH = Path("models/baseline.joblib")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a pulsegate beat-classifier worker.")
    ap.add_argument("--redis-url", default="redis://localhost:6379", help="Redis connection URL")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Run: uv run python scripts/train_baseline.py"
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    client = redis.from_url(args.redis_url)
    client.ping()
    consumer = BeatConsumer(client=client, model=model)
    consumer.ensure_group()

    print(f"Consumer {consumer.consumer_name} listening on {consumer.in_stream} → {consumer.out_stream}")
    print(f"Model trained {artifact['trained_at']}, classes {list(model.classes_)}")

    n, t_start = 0, time.perf_counter()
    try:
        while True:
            result = consumer.consume_one()
            if result is None:
                continue
            n += 1
            if n % 50 == 0:
                rate = n / (time.perf_counter() - t_start)
                print(f"  [{n:>5} @ {rate:.1f} msg/sec] rec={result['record_id']}/{result['beat_index']} "
                      f"pred={result['predicted_class']} conf={result['confidence'][:5]} gt={result['ground_truth']}")
    except KeyboardInterrupt:
        elapsed = max(time.perf_counter() - t_start, 0.001)
        print(f"\nShutting down. Classified {n} beats in {elapsed:.1f}s ({n / elapsed:.1f} msg/sec).")


if __name__ == "__main__":
    main()
