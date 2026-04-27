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

## Running Redis (Week 2+)

The streaming pipeline requires Redis 7+ running locally. Docker Compose manages it:

```bash
# Start Redis in the background (data persists across restarts via a named volume)
docker compose up -d

# Verify it's up
docker compose ps
redis-cli ping                # → PONG

# Tail logs
docker compose logs -f redis

# Stop Redis (data preserved)
docker compose down

# Stop Redis AND wipe all data (fresh state)
docker compose down -v
```

Redis binds to `127.0.0.1:6379` (localhost only) with AOF persistence enabled per `design-notes.md` §8. The producer script (`scripts/run_producer.py`) defaults to this URL — no config needed.

## Observability (Week 2+)

The streaming pipeline exposes Prometheus metrics:
- Producer `/metrics` on `localhost:9091`
- Consumer `/metrics` on `localhost:9092`
- Stream monitor `/metrics` on `localhost:9093`

`docker compose up -d` starts Prometheus (scrapes every 5s) and Grafana alongside Redis. Once the pipeline scripts are running, the pre-provisioned dashboard shows throughput, latency quantiles, per-phase timing breakdowns, and queue state:

```bash
# Start infra + producer/consumer/monitor (in separate terminals)
docker compose up -d
uv run python scripts/run_consumer.py
uv run python scripts/run_producer.py 100 --speed 100 --max-beats 500
uv run python scripts/run_stream_monitor.py

# Open the dashboard
open http://localhost:3000/d/pulsegate
```

Grafana is anonymous-access enabled (Viewer role) for local dev — no login needed.

## Docs

- [`docs/dataset.md`](docs/dataset.md) — dataset schema, labels, AAMI mapping, gotchas.
- [`docs/architecture.md`](docs/architecture.md) — system architecture.
- [`docs/design-notes.md`](docs/design-notes.md) — running decisions log.
- [`docs/decisions.md`](docs/decisions.md) — architectural decision log with rationale.
