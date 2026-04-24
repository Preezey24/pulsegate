"""Tests for streaming metrics instrumentation — verify Counters and Histograms record correctly.

Note: Prometheus metrics are module-level globals and state persists across tests within a
pytest run. Tests measure before/after deltas instead of absolute values.
"""

import fakeredis
import numpy as np
from prometheus_client import REGISTRY

from pulsegate_core.pipeline import BeatSample
from pulsegate_core.streaming import BeatConsumer, BeatProducer


class _StubModel:
    classes_ = np.array(["F", "N", "Q", "S", "V"])
    def predict(self, X):
        assert X.shape == (1, 256)
        return np.array(["N"])
    def predict_proba(self, X):
        assert X.shape == (1, 256)
        return np.array([[0.01, 0.95, 0.00, 0.01, 0.03]])


def _beat() -> BeatSample:
    return BeatSample(
        record_id="metric-test", beat_index=0, r_peak_sample=1000,
        symbol="N", aami_class="N",
        window=np.zeros(252, dtype=np.float32),
        temporal={"pre_rr": 0.8, "post_rr": 0.8, "local_avg_rr": 0.8, "rr_ratio": 1.0},
    )


def _get(name: str, labels: dict | None = None) -> float:
    """Get a metric sample value; returns 0.0 when the sample hasn't been recorded yet."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_producer_emit_increments_counter_and_histogram():
    """emit() increments the counter by 1 and records one histogram observation."""
    labels = {"signal_type": "ecg_beat"}
    counter_before = _get("pulsegate_producer_beats_emitted_total", labels)
    hist_before = _get("pulsegate_producer_emit_duration_seconds_count", labels)

    producer = BeatProducer(client=fakeredis.FakeStrictRedis())
    producer.emit(_beat())

    assert _get("pulsegate_producer_beats_emitted_total", labels) - counter_before == 1.0
    assert _get("pulsegate_producer_emit_duration_seconds_count", labels) - hist_before == 1.0


def test_consumer_consume_records_all_metrics_with_label():
    """consume_one() increments per-class counter + observes 3 histograms (process, predict, decode)."""
    client = fakeredis.FakeStrictRedis()
    producer = BeatProducer(client=client)
    consumer = BeatConsumer(client=client, model=_StubModel())
    consumer.ensure_group()

    counter_before = _get("pulsegate_consumer_beats_consumed_total", {"predicted_class": "N"})
    process_before = _get("pulsegate_consumer_process_latency_seconds_count")
    predict_before = _get("pulsegate_consumer_predict_duration_seconds_count")
    decode_before = _get("pulsegate_consumer_decode_duration_seconds_count")

    producer.emit(_beat())
    result = consumer.consume_one(block_ms=500)
    assert result is not None

    assert _get("pulsegate_consumer_beats_consumed_total", {"predicted_class": "N"}) - counter_before == 1.0
    assert _get("pulsegate_consumer_process_latency_seconds_count") - process_before == 1.0
    assert _get("pulsegate_consumer_predict_duration_seconds_count") - predict_before == 1.0
    assert _get("pulsegate_consumer_decode_duration_seconds_count") - decode_before == 1.0


def test_stream_monitor_poll_updates_gauges():
    """poll_and_update() reads XLEN + XPENDING and writes them to the Prometheus gauges."""
    from pulsegate_core.streaming.monitor import poll_and_update

    client = fakeredis.FakeStrictRedis()
    # Seed :in with 2 messages, :out with 1
    client.xadd("ecg_beat:in", {"k": "v1"})
    client.xadd("ecg_beat:in", {"k": "v2"})
    client.xadd("ecg_beat:out", {"k": "v3"})
    # Create the consumer group so XPENDING returns a well-formed response.
    client.xgroup_create("ecg_beat:in", "workers", id="0")

    poll_and_update(client)

    assert _get("pulsegate_stream_length", {"stream": "ecg_beat:in"}) == 2.0
    assert _get("pulsegate_stream_length", {"stream": "ecg_beat:out"}) == 1.0
    # Group exists, nothing read yet → PEL empty.
    assert _get("pulsegate_stream_pending", {"stream": "ecg_beat:in", "group": "workers"}) == 0.0


def test_stream_monitor_handles_missing_group_gracefully():
    """Monitor may start before consumers have created the group — must not crash."""
    from pulsegate_core.streaming.monitor import poll_and_update

    client = fakeredis.FakeStrictRedis()
    client.xadd("ecg_beat:in", {"k": "v"})
    # No group created → XPENDING would raise; poll_and_update must swallow it.

    poll_and_update(client)  # must not raise

    assert _get("pulsegate_stream_pending", {"stream": "ecg_beat:in", "group": "workers"}) == 0.0
