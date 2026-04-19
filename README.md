# pulsegate

Real-time biosignal classification service. Streaming ECG arrhythmia detection on MIT-BIH (PhysioNet).

Work in progress.

## Setup

Requires **Python 3.12** and **[uv](https://docs.astral.sh/uv/)**.

```bash
# Install dependencies (runtime + dev) and the pulsegate package in editable mode
uv sync --group dev

# Download the MIT-BIH Arrhythmia Database (~100 MB) into data/mitdb/
uv run python scripts/download_data.py

# Run tests
uv run pytest
```

The data download is idempotent — re-running verifies the dataset is complete and re-fetches any missing records. PhysioNet occasionally drops records silently; the script reports how many landed.

Integration tests require the MIT-BIH data to be present at `data/mitdb/`. Unit tests run standalone.

## Docs

- [`docs/dataset.md`](docs/dataset.md) — dataset schema, labels, AAMI mapping, gotchas.
- [`docs/architecture.md`](docs/architecture.md) — system architecture.
- [`docs/design-notes.md`](docs/design-notes.md) — running decisions log.
