# 🚀 Real-Time Cashout Forecasting System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An enterprise-grade, real-time machine learning pipeline designed to predict and monitor cashout risks with sub-millisecond latency. 

Recently upgraded to a streaming architecture inspired by **IBM Z mainframes**, this system features a zero-dependency portable scoring engine, an in-memory feature store, and a tamper-proof cryptographic audit chain for strict compliance.

---

## ✨ Core Features

*   ⚡ **Real-Time Streaming API:** High-throughput, low-latency inference backend built on FastAPI.
*   🧠 **Portable ML Scoring:** Scikit-learn models (HistGradientBoosting) exported to zero-dependency `.npz` arrays for ultra-fast, environment-agnostic execution.
*   🏬 **Online Feature Store:** In-memory state management for real-time feature aggregations.
*   🚨 **Alert Engine:** Automated, threshold-based anomaly and risk alerting.
*   🔗 **Cryptographic Audit Chain:** Tamper-proof, cryptographically signed prediction logs for strict financial compliance.
*   📊 **Live Dashboard:** Interactive Streamlit UI for monitoring streaming operations, viewing risk maps, and investigating alerts.

---

## 🏗️ Architecture overview

The real-time streaming pipeline isolates the heavy data-science training environment from the lightweight production scoring environment.

1. **Ingestion:** Live transactional data flows into the API.
2. **Feature Store:** The `online_store.py` updates rolling aggregates in memory.
3. **Scoring:** `portable_model.py` executes matrix math instantly without loading heavy ML libraries.
4. **Alerts & Audit:** High-risk scores trigger the `alerts.py` engine, while *every* transaction is cryptographically hashed and chained in `audit.py`.

*(See [`docs/IBM_Z_ARCHITECTURE.md`](docs/IBM_Z_ARCHITECTURE.md) for a deep dive into the system design.)*

---

## 🚀 Quick Start (Local Run)

### 1. Setup Environment
Clone the repository and set up a Python virtual environment:

```powershell
git clone [https://github.com/rawat4113/cashout-forecasting-system.git](https://github.com/rawat4113/cashout-forecasting-system.git)
cd cashout-forecasting-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
