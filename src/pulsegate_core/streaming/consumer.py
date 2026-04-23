"""Redis Streams consumer — XREADGROUPs beats from ecg_beat:in, classifies, XADDs to ecg_beat:out.

Role: worker side of the streaming pipeline (design-notes.md §8).
V1: single worker, happy path only. Failure handling (retry, DLQ, XCLAIM) deferred;
see design-notes.md §8 Failure modes table for the planned behaviour.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis

from pulsegate_core.streaming.producer import ECG_BEAT_STREAM

ECG_BEAT_OUT_STREAM = "ecg_beat:out"
DEFAULT_CONSUMER_GROUP = "workers"
DEFAULT_CONSUMER_NAME = "worker-1"
_TEMPORAL_KEYS = ("pre_rr", "post_rr", "local_avg_rr", "rr_ratio")
_PASS_THROUGH_KEYS = ("request_id", "record_id", "beat_index", "r_peak_sample", "ground_truth")


@dataclass(frozen=True)
class BeatConsumer:
    """Consume beats from Redis, classify, emit results. Stateless between messages."""

    client: redis.Redis
    model: Any  # sklearn classifier with .predict + .predict_proba + .classes_
    in_stream: str = ECG_BEAT_STREAM
    out_stream: str = ECG_BEAT_OUT_STREAM
    group_name: str = DEFAULT_CONSUMER_GROUP
    consumer_name: str = DEFAULT_CONSUMER_NAME

    def ensure_group(self) -> None:
        """Create the consumer group if it doesn't exist. Idempotent."""
        try:
            self.client.xgroup_create(self.in_stream, self.group_name, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def consume_one(self, block_ms: int = 5000) -> dict | None:
        """Read one message, classify, emit result, ACK. Returns result dict or None on timeout."""
        response = self.client.xreadgroup(
            self.group_name, self.consumer_name,
            {self.in_stream: ">"}, count=1, block=block_ms,
        )
        if not response:
            return None
        _, messages = response[0]
        msg_id, fields = messages[0]

        msg = {k.decode(): v.decode() for k, v in fields.items()}
        samples = np.array(json.loads(msg["samples"]), dtype=np.float32)
        temporal = np.array(
            [0.0 if msg[k] == "null" else float(msg[k]) for k in _TEMPORAL_KEYS],
            dtype=np.float32,
        )
        features = np.concatenate([samples, temporal]).reshape(1, -1)
        pred = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]
        confidence = float(probs[list(self.model.classes_).index(pred)])

        result = {k: msg[k] for k in _PASS_THROUGH_KEYS}
        result.update({
            "signal_type": "ecg_beat",
            "predicted_class": str(pred),
            "confidence": str(confidence),
            "consumer_ts": str(time.time()),
        })
        self.client.xadd(self.out_stream, result)
        self.client.xack(self.in_stream, self.group_name, msg_id)
        return result
