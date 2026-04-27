"""Replay one or more MIT-BIH records onto Redis Stream ecg_beat:in at configurable speed.

Usage:
    uv run python scripts/run_producer.py 100
    uv run python scripts/run_producer.py 100 --speed 10
    uv run python scripts/run_producer.py 100 --speed 100 --max-beats 50
    uv run python scripts/run_producer.py 100 103 105 --speed 50 --duration 60
    uv run python scripts/run_producer.py 100 --speed 100 --loop --duration 120

speed=1 replays at natural heart rate (~1 beat/sec). Higher speeds accelerate for
load testing. --loop cycles records indefinitely; each lap is tagged as a distinct
virtual_patient_id so the consumer sees a fresh per-patient stream per lap.
See design-notes.md §6 for replay strategy.

Requires Redis running locally (default: redis://localhost:6379).
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Iterator

import redis
from prometheus_client import start_http_server

from pulsegate_core.pipeline import BeatSample, iter_beats
from pulsegate_core.streaming import BeatProducer


def _beat_stream(record_ids: list[str], loop: bool) -> Iterator[tuple[BeatSample, int, str]]:
    """Yield (beat, lap, record_id) cycling through records; loops forever if loop=True."""
    laps: Iterator[int] = itertools.count() if loop else iter([0])
    for lap in laps:
        for rid in record_ids:
            for beat in iter_beats(rid):
                yield beat, lap, rid


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay one or more records onto Redis.")
    ap.add_argument("record_ids", nargs="+", help="MIT-BIH record ID(s), e.g. '100' or '100 103 105'")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback multiplier (1.0 = real-time, 100.0 = accelerated)")
    ap.add_argument("--loop", action="store_true",
                    help="Cycle through records indefinitely (until --duration or --max-beats)")
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after N seconds of wall time")
    ap.add_argument("--redis-url", default="redis://localhost:6379",
                    help="Redis connection URL")
    ap.add_argument("--max-beats", type=int, default=None,
                    help="Stop after N total beats (default: whole record list, one pass)")
    ap.add_argument("--metrics-port", type=int, default=9091,
                    help="Port to expose Prometheus /metrics on (default: 9091)")
    args = ap.parse_args()

    client = redis.from_url(args.redis_url)
    client.ping()  # fail fast if Redis is unreachable
    producer = BeatProducer(client=client)

    start_http_server(args.metrics_port)
    print(f"Prometheus metrics: http://localhost:{args.metrics_port}/metrics")
    print(f"Replaying records {args.record_ids} at speed={args.speed}x"
          f"{' --loop' if args.loop else ''} → {args.redis_url}")
    t_start = time.perf_counter()
    n_emitted = 0

    for i, (beat, lap, rid) in enumerate(_beat_stream(args.record_ids, args.loop)):
        if args.max_beats is not None and n_emitted >= args.max_beats:
            break
        if args.duration is not None and (time.perf_counter() - t_start) >= args.duration:
            break
        # Pace by pre_rr / speed. Skip pacing on the very first beat for debug friendliness.
        pre_rr = beat.temporal["pre_rr"]
        if i > 0 and pre_rr is not None and args.speed > 0:
            time.sleep(pre_rr / args.speed)

        msg_id = producer.emit(beat, virtual_patient_id=f"{rid}_lap{lap}")
        n_emitted += 1
        if n_emitted % 50 == 0:
            print(f"  emitted {n_emitted} beats  (last: rid={rid} lap={lap} "
                  f"idx={beat.beat_index}, class={beat.aami_class}, msg_id={msg_id.decode()})")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. Emitted {n_emitted} beats in {elapsed:.2f}s "
          f"({n_emitted / elapsed:.1f} beats/sec actual rate).")


if __name__ == "__main__":
    main()
