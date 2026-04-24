"""Replay one MIT-BIH record onto Redis Stream ecg_beat:in at configurable speed.

Usage:
    uv run python scripts/run_producer.py 100
    uv run python scripts/run_producer.py 100 --speed 10
    uv run python scripts/run_producer.py 100 --speed 100 --max-beats 50

speed=1 replays at natural heart rate (~1 beat/sec). Higher speeds accelerate for
load testing. See design-notes.md §6 for replay strategy.

Requires Redis running locally (default: redis://localhost:6379).
"""

from __future__ import annotations

import argparse
import time

import redis
from prometheus_client import start_http_server

from pulsegate_core.pipeline import iter_beats
from pulsegate_core.streaming import BeatProducer


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a record onto Redis.")
    ap.add_argument("record_id", help="MIT-BIH record ID (e.g. '100')")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback multiplier (1.0 = real-time, 100.0 = accelerated)")
    ap.add_argument("--redis-url", default="redis://localhost:6379",
                    help="Redis connection URL")
    ap.add_argument("--max-beats", type=int, default=None,
                    help="Stop after N beats (default: whole record)")
    ap.add_argument("--metrics-port", type=int, default=9091,
                    help="Port to expose Prometheus /metrics on (default: 9091)")
    args = ap.parse_args()

    client = redis.from_url(args.redis_url)
    client.ping()  # fail fast if Redis is unreachable
    producer = BeatProducer(client=client)

    start_http_server(args.metrics_port)
    print(f"Prometheus metrics: http://localhost:{args.metrics_port}/metrics")
    print(f"Replaying record {args.record_id} at speed={args.speed}x → {args.redis_url}")
    t_start = time.perf_counter()
    n_emitted = 0

    for i, beat in enumerate(iter_beats(args.record_id)):
        if args.max_beats is not None and n_emitted >= args.max_beats:
            break
        # Pace by pre_rr / speed. Skip pacing on the very first beat for debug friendliness.
        pre_rr = beat.temporal["pre_rr"]
        if i > 0 and pre_rr is not None and args.speed > 0:
            time.sleep(pre_rr / args.speed)

        msg_id = producer.emit(beat)
        n_emitted += 1
        if n_emitted % 50 == 0:
            print(f"  emitted {n_emitted} beats  (last: idx={beat.beat_index}, "
                  f"class={beat.aami_class}, msg_id={msg_id.decode()})")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. Emitted {n_emitted} beats in {elapsed:.2f}s "
          f"({n_emitted / elapsed:.1f} beats/sec actual rate).")


if __name__ == "__main__":
    main()
