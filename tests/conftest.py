"""Shared fixtures for the pulsegate test suite."""

import pytest

from pulsegate_core.io import load_record


@pytest.fixture(scope="session")
def record_100():
    """MIT-BIH record 100 — clean sinus rhythm, DS2 test record. Loaded once per test run."""
    return load_record("100")


@pytest.fixture(scope="session")
def record_203():
    """MIT-BIH record 203 — V-rich arrhythmia, DS1 training record. Loaded once per test run."""
    return load_record("203")
