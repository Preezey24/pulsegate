"""Tests for pulsegate_core.streaming.producer — BeatProducer emit + serialization contract."""

import dataclasses
import itertools
import json

import fakeredis
import numpy as np
import pytest

from pulsegate_core.pipeline import BeatSample, iter_beats
from pulsegate_core.streaming.producer import BeatProducer, ECG_BEAT_STREAM


@pytest.fixture
def fake_client():
    """Fresh in-memory Redis client per test."""
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def producer(fake_client):
    return BeatProducer(client=fake_client)


def _synthetic_beat(temporal: dict | None = None) -> BeatSample:
    """Synthetic BeatSample for tests that don't need real MIT-BIH data."""
    return BeatSample(
        record_id="synth", beat_index=0, r_peak_sample=1000,
        symbol="N", aami_class="N",
        window=np.zeros(252, dtype=np.float32),
        temporal=temporal or {"pre_rr": 0.8, "post_rr": 0.8, "local_avg_rr": 0.8, "rr_ratio": 1.0},
    )


def test_beatproducer_is_frozen(producer):
    """Configuration is immutable after construction."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        producer.stream_name = "hacked"


def test_emit_respects_custom_stream_name(fake_client):
    """stream_name argument routes messages to the specified stream."""
    p = BeatProducer(client=fake_client, stream_name="custom:stream")
    p.emit(_synthetic_beat())
    assert fake_client.xlen("custom:stream") == 1
    assert fake_client.xlen(ECG_BEAT_STREAM) == 0


def test_emit_encodes_none_temporal_as_null_string(producer, fake_client):
    """Synthetic beat with None temporals → fields serialized as literal 'null' on the wire."""
    synthetic = _synthetic_beat(temporal={
        "pre_rr": None, "post_rr": 0.8, "local_avg_rr": None, "rr_ratio": None,
    })
    producer.emit(synthetic)
    _, fields = fake_client.xrange(ECG_BEAT_STREAM, "-", "+")[0]
    assert fields[b"pre_rr"] == b"null"
    assert fields[b"local_avg_rr"] == b"null"
    assert fields[b"rr_ratio"] == b"null"
    assert fields[b"post_rr"] == b"0.8"


def test_end_to_end_round_trip_on_record_100(producer, fake_client):
    """5-beat round-trip: emit via iter_beats, XRANGE back, byte-compare every field."""
    beats = list(itertools.islice(iter_beats("100"), 5))
    for b in beats:
        producer.emit(b)

    raw = fake_client.xrange(ECG_BEAT_STREAM, "-", "+")
    assert len(raw) == 5

    request_ids = set()
    for (msg_id, fields), source in zip(raw, beats):
        assert isinstance(msg_id, bytes) and b"-" in msg_id  # Redis ID format: "<ms>-<seq>"
        d = {k.decode(): v.decode() for k, v in fields.items()}
        assert d["signal_type"] == "ecg_beat"
        assert d["record_id"] == source.record_id
        assert int(d["beat_index"]) == source.beat_index
        assert int(d["r_peak_sample"]) == source.r_peak_sample
        assert d["ground_truth"] == source.aami_class
        assert d["virtual_patient_id"] == source.record_id  # V1 default

        # Samples: byte-exact after JSON round-trip through float32.
        decoded = np.array(json.loads(d["samples"]), dtype=np.float32)
        assert np.array_equal(decoded, source.window)

        # Temporal features: float equality within 1e-10.
        for key in ("pre_rr", "post_rr", "local_avg_rr", "rr_ratio"):
            assert abs(float(d[key]) - source.temporal[key]) < 1e-10

        request_ids.add(d["request_id"])
    assert len(request_ids) == 5  # auto-generated UUIDs unique across emissions
