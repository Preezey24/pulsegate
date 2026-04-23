"""Tests for pulsegate_core.streaming.consumer — BeatConsumer decode + classify + XACK contract."""

import dataclasses

import fakeredis
import numpy as np
import pytest

from pulsegate_core.pipeline import BeatSample
from pulsegate_core.streaming.consumer import (
    BeatConsumer,
    DEFAULT_CONSUMER_GROUP,
    ECG_BEAT_OUT_STREAM,
)
from pulsegate_core.streaming.producer import BeatProducer, ECG_BEAT_STREAM


class _StubModel:
    """Minimal sklearn-compatible classifier. Always predicts 'N' with 0.95 confidence.
    Asserts the consumer built a proper 256-dim feature vector — doubles as a contract check."""
    classes_ = np.array(["F", "N", "Q", "S", "V"])
    def predict(self, X):
        assert X.shape == (1, 256), f"expected (1, 256) feature vector, got {X.shape}"
        return np.array(["N"])
    def predict_proba(self, X):
        assert X.shape == (1, 256), f"expected (1, 256) feature vector, got {X.shape}"
        return np.array([[0.01, 0.95, 0.00, 0.01, 0.03]])


@pytest.fixture
def fake_client():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def consumer(fake_client):
    return BeatConsumer(client=fake_client, model=_StubModel())


def _synthetic_beat(temporal: dict | None = None) -> BeatSample:
    return BeatSample(
        record_id="synth", beat_index=0, r_peak_sample=1000,
        symbol="N", aami_class="N",
        window=np.zeros(252, dtype=np.float32),
        temporal=temporal or {"pre_rr": 0.8, "post_rr": 0.8, "local_avg_rr": 0.8, "rr_ratio": 1.0},
    )


def test_beatconsumer_is_frozen(consumer):
    """Configuration is immutable after construction."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        consumer.group_name = "hacked"


def test_ensure_group_is_idempotent(consumer):
    """Calling ensure_group twice in a row doesn't raise (swallows BUSYGROUP)."""
    consumer.ensure_group()
    consumer.ensure_group()  # second call hits BUSYGROUP path, must not raise


def test_consume_one_returns_none_on_timeout(consumer):
    """No messages in stream → block expires → consume_one returns None."""
    consumer.ensure_group()
    assert consumer.consume_one(block_ms=50) is None


def test_null_temporal_decoded_without_crash(consumer, fake_client):
    """Consumer decodes 'null' temporal fields without crashing; prediction succeeds."""
    consumer.ensure_group()
    producer = BeatProducer(client=fake_client)
    beat = _synthetic_beat(temporal={"pre_rr": None, "post_rr": 0.8, "local_avg_rr": None, "rr_ratio": None})
    producer.emit(beat)

    result = consumer.consume_one(block_ms=500)
    assert result is not None
    assert result["predicted_class"] == "N"


def test_end_to_end_round_trip_produces_expected_result_and_acks(consumer, fake_client):
    """Producer emits → consumer classifies → result on :out has expected schema → input is ACKed."""
    consumer.ensure_group()
    producer = BeatProducer(client=fake_client)
    beat = _synthetic_beat()
    producer.emit(beat, virtual_patient_id="test_patient")

    result = consumer.consume_one(block_ms=500)
    assert result is not None

    # Schema: all expected fields present, no extras.
    assert set(result.keys()) == {
        "request_id", "signal_type", "record_id", "beat_index", "r_peak_sample",
        "predicted_class", "confidence", "ground_truth", "consumer_ts",
    }

    # Content matches _StubModel outputs and producer inputs.
    assert result["predicted_class"] == "N"
    assert result["confidence"] == "0.95"
    assert result["ground_truth"] == "N"
    assert result["record_id"] == "synth"
    assert result["signal_type"] == "ecg_beat"

    # Output actually emitted to :out.
    assert fake_client.xlen(ECG_BEAT_OUT_STREAM) == 1

    # Input XACKed — PEL should have 0 pending entries.
    pending_info = fake_client.xpending(ECG_BEAT_STREAM, DEFAULT_CONSUMER_GROUP)
    assert pending_info["pending"] == 0
