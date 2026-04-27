"""Stream-depth polling logic: update stream_length + stream_pending gauges from Redis state.

Split from the CLI script (scripts/run_stream_monitor.py) so the polling body is importable
and testable. Mirrors the library/CLI separation used for BeatProducer and BeatConsumer.
"""

from __future__ import annotations

import redis

from pulsegate_core.streaming.consumer import DEFAULT_CONSUMER_GROUP, ECG_BEAT_OUT_STREAM
from pulsegate_core.streaming.metrics import stream_lag, stream_length, stream_pending
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


def _group_lag(client: redis.Redis, stream: str, group: str) -> int:
    """Return undelivered-message count for (stream, group); 0 if missing.

    `lag` is the Redis 7+ field on XINFO GROUPS giving the count of messages written to
    the stream but not yet delivered to any consumer in this group. Distinct from PEL,
    which is delivered-but-not-acked. Together they characterise consumer health:
    rising lag = consumer can't keep up; rising PEL = consumer crashed mid-process.
    """
    try:
        groups = client.xinfo_groups(stream)
    except (redis.ResponseError, redis.exceptions.ResponseError):
        return 0
    for g in groups:
        name = g.get("name") if "name" in g else g.get(b"name")
        if isinstance(name, bytes):
            name = name.decode()
        if name == group:
            lag = g.get("lag") if "lag" in g else g.get(b"lag")
            return int(lag) if lag is not None else 0
    return 0


def poll_and_update(client: redis.Redis) -> None:
    """Poll XLEN, XPENDING, and XINFO GROUPS for the ecg_beat streams; update gauges."""
    for s in (ECG_BEAT_STREAM, ECG_BEAT_OUT_STREAM):
        stream_length.labels(stream=s).set(client.xlen(s))
    stream_pending.labels(stream=ECG_BEAT_STREAM, group=DEFAULT_CONSUMER_GROUP).set(
        _pending_count(client, ECG_BEAT_STREAM, DEFAULT_CONSUMER_GROUP)
    )
    stream_lag.labels(stream=ECG_BEAT_STREAM, group=DEFAULT_CONSUMER_GROUP).set(
        _group_lag(client, ECG_BEAT_STREAM, DEFAULT_CONSUMER_GROUP)
    )
