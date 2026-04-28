# Privacy Module — README

Privacy-aware routing layer for Palimpzest. No PZ source files are modified.

---

## Files

| File | Purpose |
|------|---------|
| `routing_stub.py` | Core routing logic: `PrivacyRouter`, `ModelConfig`, `RoutingGranularity`, `AnonymizationSensitivity`, detection backends, `execute_with_routing()` |
| `privacy_execution_strategy.py` | `PrivacyAwareExecutionStrategy` (subclass of PZ's sequential strategy) + `create_privacy_processor()` factory |
| `benchmark_granularity.py` | End-to-end Q1 benchmark — runs actual LLM calls with all three granularities |
| `OPERATOR_GRAPH_NOTES.md` | PZ architecture notes: logical/physical plan construction, model selection, hook point identification |
| `explore_pipeline.py` | Exploration script: prints logical + physical plans and field introspection for an Enron pipeline |
| `requirements.txt` | Extra dependencies for the privacy module (Presidio, spaCy) |

The primary benchmarks (dry-run, no LLM calls) live in `demos/`:

| File | Purpose |
|------|---------|
| `demos/benchmark_q1.py` | Q1 dry-run: routing granularity comparison on 1 000 records, no LLM calls |
| `demos/benchmark_q2.py` | Q2 dry-run: PII detector backend comparison (presidio / deberta / regex / ensemble) |

---

## Requirements

### What you can run right now (no Ollama, no DeBERTa)

The Q1 and Q2 dry-run benchmarks make **no LLM calls** and require only Presidio + spaCy:

```bash
pip install -r privacy/requirements.txt
python -m spacy download en_core_web_lg
```

This is sufficient for:
- `demos/benchmark_q1.py` — all granularities
- `demos/benchmark_q2.py --backends presidio regex` — presidio and regex backends
- `demos/benchmark_q2.py --backends presidio regex ensemble` — ensemble falls back to presidio + regex if DeBERTa is not installed

### DeBERTa backend (optional, Q2 only)

Required only if you want `--backends deberta` or want ensemble to include DeBERTa results.
The model (`iiiorg/piiranha-v1-detect-personal-information`, ~500 MB) downloads automatically on first use.

```bash
pip install transformers torch
```

Skip this if not installed — `routing_stub.py` detects the missing library and returns empty detections, so ensemble still runs via presidio + regex.

### End-to-end LLM runs (`benchmark_granularity.py` only)

Requires either:
- **Local path**: Ollama installed and running with `llama3.1:8b` pulled
  - Download: https://ollama.com/download
  - `ollama pull llama3.1:8b`
- **Cloud path**: `OPENAI_API_KEY` set in `.env` at the repo root (for records routed to cloud/cloud_anonymized)

The dry-run benchmarks (`benchmark_q1.py`, `benchmark_q2.py`) do **not** use Ollama or any API key.

---

## Running the benchmarks

All commands are run from the **repo root**.

### Q1 — Routing granularity (dry-run, recommended)

Compares OPERATOR / FIELD / DOCUMENT granularity on a two-operator pipeline.
No LLM calls — pure routing accuracy against ground-truth PII labels.

```bash
# Default: 25 records/group (100 total), multi-operator pipeline
.venv/bin/python demos/benchmark_q1.py --pipeline multi

# Full 1 000-record run (as in the midterm report)
.venv/bin/python demos/benchmark_q1.py --pipeline multi --sample 250 --out data/q1_multi_1000.json

# Single-operator variant
.venv/bin/python demos/benchmark_q1.py --pipeline single --sample 25
```

Results already available: `data/q1_results.json` (single, 100 records), `data/q1_multi_1000.json` (multi, 1 000 records).

### Q2 — PII detector backend comparison (dry-run)

Compares presidio / deberta / regex / ensemble at OPERATOR granularity.
No LLM calls — pure routing accuracy.

```bash
# All four backends, 25 records/group
.venv/bin/python demos/benchmark_q2.py

# Larger sample + save results
.venv/bin/python demos/benchmark_q2.py --sample 100 --out data/q2_results.json

# Subset of backends (e.g. skip DeBERTa if transformers not installed)
.venv/bin/python demos/benchmark_q2.py --backends presidio regex ensemble

# Tune routing threshold
.venv/bin/python demos/benchmark_q2.py --score-threshold 0.5 --out data/q2_threshold_0.5.json
```

### Q3 — Query-intent routing (dry-run)

Compares three-way routing (local / cloud_anonymized / cloud) against a naive
two-way baseline (any PII → local) across six operators with varying query sensitivity.
No LLM calls — pure routing accuracy.

```bash
# Default: presidio backend, 25 records/group
.venv/bin/python demos/benchmark_q3.py

# Larger sample + save results
.venv/bin/python demos/benchmark_q3.py --sample 100 --out data/q3_results.json

# Regex-only (no Presidio needed)
.venv/bin/python demos/benchmark_q3.py --backend regex --sample 100
```

### Q1 end-to-end (actual LLM calls)

Requires Ollama running with `llama3.1:8b`.

```bash
.venv/bin/python privacy/benchmark_granularity.py --sample 5

# Conservative anonymization (redact more aggressively on cloud_anonymized path)
.venv/bin/python privacy/benchmark_granularity.py --sample 5 --sensitivity conservative
```

---

## Key design decisions

### Three routing destinations

| Destination | When | Effect |
|-------------|------|--------|
| `local` | PII detected AND operator prompt needs the sensitive field | Send to local Llama via Ollama; data never leaves the machine |
| `cloud_anonymized` | PII detected BUT operator prompt does not need the sensitive field | Presidio redacts PII in place, then sends to cloud model |
| `cloud` | No PII detected | Send to cloud model as-is |

Query-intent detection uses `operator.get_input_fields()` (respects `depends_on`) and keyword matching on the operator's description/filter text.

### `AnonymizationSensitivity`

Controls how aggressively PII is redacted on the `cloud_anonymized` path. Independent of `score_threshold` (which controls routing decisions).

| Level | Presidio threshold | Effect |
|-------|--------------------|--------|
| `permissive` | ≥ 0.85 | Redact only high-confidence hits; preserve more content |
| `balanced` | ≥ 0.60 | Default |
| `conservative` | ≥ 0.30 | Redact even low-confidence detections; maximise privacy coverage |

```python
from routing_stub import ModelConfig, PrivacyRouter, AnonymizationSensitivity

router = PrivacyRouter(ModelConfig(
    anonymization_sensitivity=AnonymizationSensitivity.CONSERVATIVE,
))
```

### `RoutingGranularity`

| Level | What is scanned | Key property |
|-------|-----------------|--------------|
| `OPERATOR` | Only `get_input_fields()` (respects `depends_on`) | Correct — Op2 reading a PII-free derived field routes to cloud |
| `FIELD` | All fields in `input_schema` | Over-routes — scans original PII fields even if Op2 doesn't read them |
| `DOCUMENT` | All fields once per record; decision cached for all operators | Same over-routing as FIELD; ~2× faster due to caching |

OPERATOR is the Q1 winner (0% over-routing vs 53% for FIELD/DOCUMENT).

---

## Recent changes

**`routing_stub.py`**
- Added `AnonymizationSensitivity` enum (PERMISSIVE / BALANCED / CONSERVATIVE) and `_SENSITIVITY_TO_THRESHOLD` mapping.
- Added `anonymization_sensitivity` field to `ModelConfig` with a derived `anonymization_threshold` property.
- `_anonymize_text()` now uses `anonymization_threshold` for Presidio filtering, keeping it independent of the routing `score_threshold`.

**`privacy_execution_strategy.py`**
- Added `sys.path.insert(0, os.path.dirname(__file__))` so `from routing_stub import ...` resolves correctly when the module is imported from outside the `privacy/` directory.
- Updated `from routing_stub import` to include `AnonymizationSensitivity` and `ModelConfig`.

**`benchmark_granularity.py`**
- Added `--sensitivity` CLI flag (choices: permissive / balanced / conservative).
- `run_one()` now passes `AnonymizationSensitivity` through to `ModelConfig`.
- `print_table()` now prints the active sensitivity level above the results table.
- Fixed duplicate `RoutingGranularity` in import line.
- Fixed model string: `PrivacyRouter` now receives a `ModelConfig` with `local_model="openai/llama3.1:8b"` matching the `LOCAL_MODEL` passed to PZ's optimizer config.

**`demos/benchmark_q2.py`** *(new)*
- Q2 dry-run benchmark: compares presidio / deberta / regex / ensemble backends at OPERATOR granularity.
- Reports recall, specificity, precision, F1, %local, timing per backend.
- Reports top entity types fired per backend.
- FN overlap table shows which PII records each backend missed and whether failures are shared or independent.
- `cloud_anonymized` decisions are collapsed to `cloud` for accuracy scoring (no forced-local detection = treated as not detected).
- `--backends` flag lets you run a subset (e.g., skip DeBERTa if `transformers` is not installed).
