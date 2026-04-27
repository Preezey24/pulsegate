"""Prometheus metric definitions for the pulsegate streaming pipeline.

Single source of truth for metric names, types, labels, and histogram buckets.
Modules record metrics by importing these objects. CLI scripts expose /metrics via
prometheus_client.start_http_server(port). See design-notes.md §5+§7 for the broader
observability design.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Latency/duration histogram buckets (seconds). Low-end fine-grained for predict (~26ms)
# and decode (<1ms); high-end extended to 60s to capture queue-wait under backpressure
# (load tests reach ~30s+ end-to-end when producer outruns consumer).
_LATENCY_BUCKETS = (
    0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500,
    1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0,
)


# --- Producer-side ---

producer_beats_emitted_total = Counter(
    "pulsegate_producer_beats_emitted_total",
    "Total beats XADDed to Redis by the producer.",
    ("signal_type",),
)
producer_emit_duration_seconds = Histogram(
    "pulsegate_producer_emit_duration_seconds",
    "Wall time in BeatProducer.emit() — serialisation + XADD round-trip.",
    ("signal_type",),
    buckets=_LATENCY_BUCKETS,
)


# --- Consumer-side ---

consumer_beats_consumed_total = Counter(
    "pulsegate_consumer_beats_consumed_total",
    "Total beats classified and ACKed by the consumer.",
    ("predicted_class",),
)
consumer_process_latency_seconds = Histogram(
    "pulsegate_consumer_process_latency_seconds",
    "Producer XADD → consumer classification emit. Pipeline sub-segment only; "
    "full caller-facing e2e lives on the Gateway (design-notes.md §8).",
    buckets=_LATENCY_BUCKETS,
)
consumer_predict_duration_seconds = Histogram(
    "pulsegate_consumer_predict_duration_seconds",
    "Wall time for model.predict + model.predict_proba on a 256-dim feature vector.",
    buckets=_LATENCY_BUCKETS,
)
consumer_decode_duration_seconds = Histogram(
    "pulsegate_consumer_decode_duration_seconds",
    "Wall time for JSON decode + numpy conversion of incoming beat message.",
    buckets=_LATENCY_BUCKETS,
)


# --- Stream-level (set by the stream monitor script via periodic polling) ---

stream_length = Gauge(
    "pulsegate_stream_length",
    "Current XLEN of a Redis Stream.",
    ("stream",),
)
stream_pending = Gauge(
    "pulsegate_stream_pending",
    "Number of messages in a consumer group's Pending Entries List (PEL).",
    ("stream", "group"),
)
stream_lag = Gauge(
    "pulsegate_stream_lag",
    "Messages written to the stream but not yet delivered to any consumer in the group "
    "(true backlog; PEL is in-flight, not waiting). Requires Redis 7+.",
    ("stream", "group"),
)
