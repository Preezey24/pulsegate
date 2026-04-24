"""Poll Redis stream depth + PEL size periodically, update Prometheus gauges.

Usage:
    uv run python scripts/run_stream_monitor.py
    uv run python scripts/run_stream_monitor.py --interval 2 --metrics-port 9093

Complements the producer/consumer per-beat metrics with stream-level gauges:
stream_length (XLEN) and stream_pending (XPENDING). See design-notes.md §8+§11.
Polling logic lives in pulsegate_core.streaming.monitor (importable + tested).
"""

from __future__ import annotations

import argparse
import time

import redis
from prometheus_client import start_http_server

from pulsegate_core.streaming.monitor import poll_and_update


def main() -> None:
    ap = argparse.ArgumentParser(description="Poll Redis stream gauges for Prometheus.")
    ap.add_argument("--redis-url", default="redis://localhost:6379")
    ap.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds (default 1)")
    ap.add_argument("--metrics-port", type=int, default=9093)
    args = ap.parse_args()

    client = redis.from_url(args.redis_url)
    client.ping()
    start_http_server(args.metrics_port)
    print(f"Stream monitor polling every {args.interval}s; "
          f"metrics: http://localhost:{args.metrics_port}/metrics")

    try:
        while True:
            poll_and_update(client)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStream monitor stopped.")


if __name__ == "__main__":
    main()
