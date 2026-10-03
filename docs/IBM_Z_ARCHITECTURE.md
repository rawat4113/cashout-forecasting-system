# Real-time Cybercrime Intelligence Platform — IBM Z / LinuxONE design

## 1. What is built vs. what is only designed

| Capability | Status |
|---|---|
| Validated event ingestion (`POST /api/v1/events`), per-event rejection, late-event accounting | **Built & tested** |
| Online feature store, O(1) updates, **proven identical to the batch features** | **Built & tested** (`tests/test_stream_parity.py`) |
| Portable numpy-only model, endian-explicit file, parity with scikit-learn, simulated big-endian load | **Built & tested** (`tests/test_portable_model.py`) |
| Serving path runs with **no scikit-learn/joblib** | **Built & tested** (`tests/test_serving_has_no_sklearn.py`) |
| Official (00:00) vs provisional (rolling) forecasts, alert de-duplication (NEW/ESCALATED/UNCHANGED), reason codes | **Built & tested** |
| Hash-chained audit log, tamper + deletion detection, optional HMAC | **Built & tested** |
| Live Operations dashboard + replay control | **Built**, smoke-tested headless |
| Benchmark harness | **Built**; numbers below are from an x86 sandbox, **not IBM Z** |
| Docker / compose for `linux/s390x` | **Written, never built** (no Docker in the authoring sandbox) |
| Kafka / MQ consumer | **Not built** — `StreamPipeline.ingest(dict)` is the integration point |
| HSM-backed audit key (Crypto Express / ICSF) | **Not built** — design direction |
| Telum on-chip AI accelerator | **Not used** — see §4 |
| Anything measured on real IBM Z hardware | **None yet** — run `bench/benchmark.py` on the LPAR |

## 2. Data flow

```
 NCRP-style complaint events ─┐
 (later: partner-bank ATM     │   POST /api/v1/events   (or a Kafka/MQ consumer calling pipeline.ingest)
  withdrawal events)          ▼
                    ┌──────────────────────┐
                    │ schema validation    │  malformed → rejected individually, counted
                    └─────────┬────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ online feature store │  per-cluster/day counters, O(1) update
                    └─────────┬────────────┘   same code as the trainer: features_from_arrays()
                              ▼
                    ┌──────────────────────┐
                    │ portable GBM scorer  │  numpy only, endian-explicit artifact, sha256 fingerprint
                    └─────────┬────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ alert engine         │  NEW / ESCALATED / UNCHANGED, reason codes, human review flag
                    └───────┬──────┬───────┘
                            ▼      ▼
              hash-chained audit   API + Live Operations dashboard
```

Official forecast: issued at 00:00 from complaints received before midnight — exactly what the model was trained for.
Provisional forecast: rolling refresh using only the part of today that has arrived; labelled as a nowcast, never audit-logged as official.

## 3. Why IBM Z / LinuxONE — arguments you can defend

1. **Where the data lives.** The strongest honest argument: the *bank-side* signal (ATM withdrawal and transaction events)
   originates in core-banking systems, which at large banks commonly run on mainframes. Scoring next to that feed avoids
   copying sensitive transaction data to another environment. *Today the prototype consumes complaint events only; the
   bank feed is the extension this architecture is shaped for.*
2. **Trust and auditability.** Alerts steer law-enforcement attention, so every official forecast carries the model
   fingerprint and is hash-chained. On Z the signing key can live in a hardware security module — a deployment step, not built.
3. **Single binary artifact, tiny footprint.** The serving node needs numpy + pandas + FastAPI. Training can stay anywhere.
4. **Big-endian correctness.** s390x is big-endian; pickled scikit-learn models are not a safe interchange format.
   The portable artifact is endian-explicit and tested under simulated byte-swapping.

What *not* to say: that this model "needs" a mainframe to run (it doesn't — a 26-feature tree ensemble scores 145
clusters in ~1 ms on a laptop CPU), or that it uses the Telum accelerator (it doesn't).

## 4. Honest upgrade paths that would use Z-specific hardware

* A sequence/graph model over the complaint stream, exported to **ONNX**, is the route IBM documents for the on-chip AI
  accelerator on z16+. This model is a tree ensemble, so it does not use it today. Check IBM's current documentation
  for what Snap ML / ONNX-MLIR accelerate before making any claim.
* Bank-side stream joins at higher event rates.

## 5. Measured numbers (x86-64 sandbox VM, 1 vCPU, Python 3.12, synthetic data — NOT IBM Z)

| Measurement | Result |
|---|---|
| Ingest (validate + update state), single process | ~24,600 events/s (p99 0.07 ms/event) |
| Re-score all 145 clusters (features + inference) | p50 ≈ 16 ms, p99 ≈ 20 ms (inference alone ≈ 0.8 ms) |
| 1,000 synthetic clusters | p50 ≈ 99 ms |
| 3,000 synthetic clusters | p50 ≈ 269 ms |

Feature computation (≈ 80% of the time) is the bottleneck: it re-derives rolling windows from the full history on each
call. Incremental EWMA / days-since-last state would remove most of it. Real daily volume (45 events/day here) is tiny
next to these throughputs; the numbers show headroom, they do not show a need for Z. Run `python bench/benchmark.py` on
the Z system and report the comparison as-is.

## 6. 3-minute demo script

1. (20 s) Problem: cash-outs are discovered after the fact; we forecast where the next ones will be.
2. (60 s) Live Operations → ▶ Start. Events stream in, map re-scores, an alert appears with plain-language reasons.
3. (30 s) Open the alert: reasons, recommended *human-review* action, model fingerprint.
4. (30 s) Audit panel → verify chain. In a terminal edit one line of `outputs/audit/audit_log.jsonl` → verify → "chain broken".
5. (40 s) Architecture slide: why Z (§3), exactly what is built vs designed (§1), benchmark on the LPAR.

## 7. Known limitations (say these before the judges do)

* All data is synthetic; metrics are not real-world performance.
* The gradient-boosting model beats logistic regression by a hair (PR-AUC 0.467 vs 0.463) and the 7-day-recent-activity
  baseline by a modest margin; the value of this submission is the system, not model novelty.
* Reason codes are heuristic descriptions of feature values, not SHAP attributions.
* Forecast granularity is one day per ATM cluster.
