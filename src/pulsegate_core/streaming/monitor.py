"""Stream-depth polling logic: update stream_length + stream_pending gauges from Redis state.

Split from the CLI script (scripts/run_stream_monitor.py) so the polling body is importable
and testable. Mirrors the library/CLI separation used for BeatProducer and BeatConsumer.
"""

from __future__ import annotations

import redis

from pulsegate_core.streaming.consumer import DEFAULT_CONSUMER_GROUP, ECG_BEAT_OUT_STREAM
from pulsegate_core.streaming.metrics import stream_length, stream_pending
from pulsegate_core.streaming.producer import ECG_BEAT_STREAM


def _pending_count(client: redis.Redis, stream: str, group: str) -> int:
    """Return PEL count for (stream, group); treat missing stream/group as 0.

    Handles two client-specific error shapes: real Redis raises redis.ResponseError
    (NOGROUP) on a missing group; fakeredis raises IndexError because its response
    parser trips on the empty result. Both mean "no group yet" → 0.
    """
    try:
        info = client.xpending(stream, group)
        return int(info["pending"])
    except (redis.ResponseError, IndexError):
        return 0


def poll_and_update(client: redis.Redis) -> None:
    """Poll XLEN and XPENDING for the ecg_beat streams, update Prometheus gauges."""
    for s in (ECG_BEAT_STREAM, ECG_BEAT_OUT_STREAM):
        stream_length.labels(stream=s).set(client.xlen(s))
    stream_pending.labels(stream=ECG_BEAT_STREAM, group=DEFAULT_CONSUMER_GROUP).set(
        _pending_count(client, ECG_BEAT_STREAM, DEFAULT_CONSUMER_GROUP)
    )
