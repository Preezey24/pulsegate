"""Redis Streams producer — XADDs beat messages onto ecg_beat:in.

Role: producer side of the streaming pipeline (design-notes.md §8).
Wire-format per dataset.md §9. Async XADD, MAXLEN-bounded, fire-and-forget.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import redis

from pulsegate_core.pipeline import BeatSample
from pulsegate_core.streaming.metrics import (
    producer_beats_emitted_total,
    producer_emit_duration_seconds,
)

ECG_BEAT_STREAM = "ecg_beat:in"
DEFAULT_MAXLEN = 10000  # approximate cap on stream depth; design-notes.md §8.


@dataclass(frozen=True)
class BeatProducer:
    """Serialize BeatSamples to wire format and XADD them onto a Redis Stream."""

    client: redis.Redis
    stream_name: str = ECG_BEAT_STREAM
    maxlen: int = DEFAULT_MAXLEN

    def emit(self, beat: BeatSample, virtual_patient_id: str | None = None) -> bytes:
        """Serialize one beat and XADD to the stream. Returns the Redis-assigned message ID."""
        signal_type = "ecg_beat"
        with producer_emit_duration_seconds.labels(signal_type=signal_type).time():
            fields = {
                "signal_type": signal_type,
                "request_id": str(uuid.uuid4()),
                "virtual_patient_id": virtual_patient_id or beat.record_id,
                "record_id": beat.record_id,
                "beat_index": str(beat.beat_index),
                "r_peak_sample": str(beat.r_peak_sample),
                "samples": json.dumps(beat.window.tolist()),
                "pre_rr": _scalar(beat.temporal["pre_rr"]),
                "post_rr": _scalar(beat.temporal["post_rr"]),
                "local_avg_rr": _scalar(beat.temporal["local_avg_rr"]),
                "rr_ratio": _scalar(beat.temporal["rr_ratio"]),
                "ground_truth": beat.aami_class,
                "producer_ts": str(time.time()),
            }
            msg_id = self.client.xadd(
                self.stream_name, fields, maxlen=self.maxlen, approximate=True
            )
        producer_beats_emitted_total.labels(signal_type=signal_type).inc()
        return msg_id


def _scalar(value: float | None) -> str:
    """Redis stream fields are strings; None encodes as literal 'null' for consumer to decode."""
    return "null" if value is None else str(value)
