# Cash-Out Hotspot Forecasting → Real-Time Cybercrime Intelligence Platform (IBM Z / LinuxONE target)

**IBM Z Datathon 2026 · Predict likely cash-withdrawal locations for cybercrime proceeds in advance — now as a streaming platform**

This repository contains three connected layers:

1. **Predictive Analytics Engine** – leakage-safe features, model training, evaluation, and daily hotspot forecasts.
2. **Prediction API** – a FastAPI service exposing the same production path as `src.predict`: **data → features → prediction → ranked hotspots**.
3. **Risk Heatmap Dashboard** – a Streamlit command-center UI with an interactive India map, filters for forecast date/location/fraud type/risk, and a live alert queue.

> **Data note:** real NCRP complaint and bank trace data is not public. `src/generate_data.py` creates synthetic NCRP-style data for demonstration. All metrics in this repo are synthetic and must not be presented as real-world performance.

## Quick start

```powershell
# 1) install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) reproduce the ML pipeline
python run_pipeline.py --regen

# 3) verify tests
python -m pytest -q tests

# 4) start the API (new terminal, same venv)
uvicorn api:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/predict `
  -ContentType 'application/json' `
  -Body '{"date":"2025-06-24","top_k":10}'
```

Start the dashboard in another terminal:

```powershell
streamlit run dashboard.py
```

The dashboard opens in the browser and scores all ATM clusters for the selected forecast day, then applies the filters in the UI.

## v2: real-time platform (what changed)

```text
complaint events → validation → online feature store → portable model scorer → alert engine → API / Live Operations UI
                                                                                   └→ hash-chained audit log
```

| New piece | Where | Why |
|---|---|---|
| Online feature store (same feature code as the trainer) | `src/stream/online_store.py`, `src/features.py::features_from_arrays` | Streaming features proven identical to batch (`tests/test_stream_parity.py`) |
| Portable numpy-only model + endian-explicit file | `src/portable_model.py`, `models/hgb_portable.npz` | s390x is big-endian; no scikit-learn needed on the serving node |
| Alert engine (NEW/ESCALATED/UNCHANGED, reason codes) | `src/stream/alerts.py` | Actionable, de-duplicated, explained alerts with a human-review flag |
| Tamper-evident audit chain | `src/stream/audit.py` | Every official forecast is hash-chained with the model fingerprint |
| Real-time API | `api.py` (`/api/v1/events`, `/live/*`, `/audit/*`, `/demo/replay/*`) | Existing `/predict` and `/forecast` endpoints are unchanged |
| Live Operations page | `pages/1_Live_Operations.py` | Live map, alert queue, latency, audit status |
| Benchmark harness | `bench/benchmark.py` | Run on x86 and on the IBM Z LPAR, compare honestly |
| Deployment | `Dockerfile`, `Dockerfile.dashboard`, `docker-compose.yml`, `requirements-runtime.txt` | `linux/s390x` build (not yet built — see docs) |

Full design, built-vs-designed table, measured numbers and demo script: **`docs/IBM_Z_ARCHITECTURE.md`**.

```powershell
python run_pipeline.py            # also exports models/hgb_portable.npz
uvicorn api:app                   # terminal 1
streamlit run dashboard.py        # terminal 2 → open "Live Operations" in the sidebar → ▶ Start
python bench/benchmark.py         # latency / throughput on this machine
python -m pytest -q tests         # 17 tests
```

## Architecture (v1 batch path, still available)

```text
NCRP / bank / FI records
          │
          ▼
   Data validation + cell mapping
          │
          ▼
Leakage-safe feature engineering
          │
          ▼
HistGradientBoosting model
          │
    ┌─────┴───────────┐
    ▼                 ▼
FastAPI             Streamlit
service             command dashboard
    │                 │
    ▼                 ▼
Ranked alerts     Risk heatmap + filters
    │                 │
    └───────┬─────────┘
            ▼
 Human investigator / LEA / FI action
```

The current code is designed so the API and dashboard call the same `forecast_from_frames()` path as the CLI. That keeps the demo consistent and makes it easier to replace the CSV source with a secure real-time IBM Z / LinuxONE data feed later.

## Problem formulation

The geographic space is divided into **cells** representing ATM clusters. For every `(cell, day)`, the model estimates:

**P(cash-out happens in this cell during the forecast day)**

using only complaint information known before the forecast day begins. Cells are then ranked so an agency can prioritise a limited number of locations.

### Leakage-safe design

* Label = cash withdrawal activity from `withdrawal_time`.
* Features = complaints received before `00:00` of the forecast date.
* Complaints can arrive after the underlying withdrawal, reproducing a real reporting delay.
* The chronology-aware train/validation/test split never shuffles time.
* `tests/test_no_leakage.py` checks that future complaints cannot change historical features.

## Dashboard features

The dashboard delivers the requested **~25% demo layer**:

* interactive India risk heatmap based on all scored ATM clusters;
* forecast-date filter;
* state and city filters;
* recent dominant fraud-type filter;
* minimum risk threshold;
* configurable alert-panel size;
* KPI cards for high-risk counts and risk statistics;
* ranked investigator table;
* live alert queue with city, risk, fraud type, and recent complaint volume.

The heatmap is deliberately paired with the alert table: the map shows geographic concentration while the queue creates an immediately actionable shortlist.

## API features

The API delivers the requested **~15% system layer**:

### `GET /health`
Service and data-source health check.

### `POST /api/v1/predict`
Runs the full pipeline for one date.

Request:

```json
{"date":"2025-06-24","top_k":10}
```

Response includes the forecast date, model name, and ranked hotspots with risk score, risk level, coordinates, city/state, recent fraud type, and complaint activity.

### `GET /api/v1/forecast?date=2025-06-24&top_k=10`
Convenient GET equivalent for browser/demo use.

FastAPI also exposes interactive Swagger documentation at `/docs`, which is useful in a judge demo because the prediction service can be called live rather than only shown as a static screenshot.

## Existing model outputs

* `outputs/plots/precision_coverage_at_k.png` – model comparison.
* `outputs/plots/feature_importance.png` – feature importance.
* `outputs/plots/lead_time.png` – advance warning vs reactive discovery.
* `outputs/plots/risk_map_example.png` – example model output.
* `outputs/forecast_<date>.csv` – ranked hotspot forecast.
* `outputs/alerts_<date>.json` – machine-readable alert payload.

## Using real data

Replace the synthetic files with validated records using this minimum schema:

`cells.csv`

```text
cell_id,city,state,lat,lon,tier,atm_count
```

`events.csv`

```text
event_id,withdrawal_time,complaint_time,cell_id,category,amount,traced
```

Then update the date range in `src/config.py` and retrain. In a real deployment, the CSV layer should be replaced by secure feeds / services and access controls appropriate for law-enforcement and financial data.

## IBM Z / LinuxONE positioning

See `docs/IBM_Z_ARCHITECTURE.md`. In short: the serving node is numpy + pandas + FastAPI with an endian-explicit model
artifact, and every claim in that document is labelled built / designed / not-yet-measured. **No benchmark in this
repository was run on IBM Z hardware** until you run `bench/benchmark.py` there and add the result.

## Responsible-use guardrails

The system is designed to prioritise locations for human review. A predicted hotspot is a risk signal, not evidence that a person, bank, city, or community is involved in crime. Real deployment should include access control, audit logging, data minimisation, model monitoring, human review, and documented escalation/override procedures.
