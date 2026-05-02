# Privacy-Aware Routing for Palimpzest

This document covers setup and usage for the privacy routing layer added on top of Palimpzest as part of MIT 6.5831. All implementation lives in `privacy/` and `demos/`; no PZ source files are modified.

## Overview

The system intercepts each Palimpzest operator call before the LLM is invoked, detects PII in the fields that operator actually reads, and routes to one of three paths:

- **local** — PII present and the query needs it → Ollama/Llama (on-device)
- **cloud\_anonymized** — PII present but the query doesn't need it → Presidio redacts, then cloud
- **cloud** — no PII detected → cloud model unchanged

## Repository layout

```
privacy/
  routing_stub.py               # PrivacyRouter, ModelConfig, execute_with_routing
  privacy_execution_strategy.py # PrivacyAwareExecutionStrategy, create_privacy_processor
  requirements.txt              # privacy-specific dependencies

demos/
  benchmark_q1.py               # Q1: routing granularity (OPERATOR vs FIELD vs DOCUMENT)
  benchmark_q2.py               # Q2: PII detector backend comparison
  benchmark_q3.py               # Q3: query-intent routing (two-way vs three-way)
  benchmark_granularity.py      # end-to-end LLM run for granularity comparison
  benchmark_sensitivity.py      # anonymization sensitivity knob benchmark
  gpt4o_baseline.py             # GPT-4o-mini cloud quality baseline
  resume-pii-demo.py            # end-to-end privacy-routed pipeline demo

data/
  resumes_with_pii.jsonl        # 14,566-record resume corpus (four PII tiers)
  pii_labels.jsonl              # ground-truth routing labels per record
  q1_multi_1000.json            # Q1 results (1,000-record multi-operator run)
  q2_results.json               # Q2 results (400-record backend comparison)
  q3_results.json               # Q3 results (400-record query-intent run)
  gpt4o_baseline_results.json   # GPT-4o-mini sem_filter + sem_map baseline
```

## Setup

**Requires Python ≥ 3.12.**

```bash
# 1. Clone and install Palimpzest
git clone https://github.com/mitdbg/palimpzest.git
cd palimpzest
uv venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -e .

# 2. Install privacy dependencies
uv pip install -r privacy/requirements.txt
python -m spacy download en_core_web_lg

# 3. (Optional) DeBERTa backend — downloads ~500 MB model on first use
uv pip install transformers torch
```

**Environment variables** — create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...          # required for cloud model runs
ANTHROPIC_API_KEY=...          # optional, if using Claude as cloud model
```

**Local model** — required only for end-to-end LLM runs (not the dry-run benchmarks):

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
```

## Running the benchmarks

All benchmark scripts are dry-run by default (no LLM calls, pure routing accuracy against ground-truth labels). Results print to stdout and are saved as JSON.

### Q1 — Routing granularity

Compares OPERATOR, FIELD, and DOCUMENT granularity on a two-operator pipeline.

```bash
# Single-operator baseline (all granularities identical)
.venv/bin/python demos/benchmark_q1.py --pipeline single --sample 100

# Multi-operator pipeline — granularities diverge here
.venv/bin/python demos/benchmark_q1.py --pipeline multi --sample 250 --out data/q1_multi_1000.json
```

### Q2 — PII detector backend comparison

Compares Presidio, DeBERTa, regex, and ensemble on routing accuracy and latency.

```bash
.venv/bin/python demos/benchmark_q2.py --sample 100 --out data/q2_results.json

# Run specific backends only
.venv/bin/python demos/benchmark_q2.py --backends presidio regex --sample 100
```

### Q3 — Query-intent routing

Tests three-way routing (local / cloud\_anonymized / cloud) against naive two-way routing.

```bash
.venv/bin/python demos/benchmark_q3.py --sample 100 --out data/q3_results.json

# Regex backend (faster)
.venv/bin/python demos/benchmark_q3.py --backend regex --sample 100
```

### GPT-4o-mini cloud baseline

Measures cloud model quality ceiling on sem\_filter and sem\_map. Requires `OPENAI_API_KEY`.

```bash
# Dry-run test first (8 records, ~$0.003)
.venv/bin/python demos/gpt4o_baseline.py --test

# Full baseline (100 records, ~$0.034)
.venv/bin/python demos/gpt4o_baseline.py --sample 25 --task both
```

## End-to-end demo

Runs the full privacy-routed pipeline on the resume dataset with live LLM calls. Requires `OPENAI_API_KEY` and Ollama running locally.

```bash
.venv/bin/python demos/resume-pii-demo.py
```

## Using the routing layer in your own pipeline

```python
from privacy.privacy_execution_strategy import create_privacy_processor
from privacy.routing_stub import AnonymizationSensitivity, RoutingGranularity
import palimpzest as pz

# Build your plan normally
dataset = pz.MemoryDataset(...)
plan = dataset.sem_filter("...", depends_on=["text", "ssn"])

config = pz.QueryProcessorConfig(policy=pz.MaxQuality())

# Swap in the privacy execution strategy — one line change
processor = create_privacy_processor(
    plan,
    config,
    granularity=RoutingGranularity.OPERATOR,          # recommended default
    sensitivity=AnonymizationSensitivity.CONSERVATIVE, # max PII redaction
)
result = processor.execute()

# Print routing summary
print(processor.execution_strategy.router.stats.summary())
```

`RoutingGranularity` options:
- `OPERATOR` — scan only the fields each operator actually reads (recommended)
- `FIELD` — scan all input schema fields regardless of `depends_on`
- `DOCUMENT` — scan all fields once per record, reuse decision for all operators

`AnonymizationSensitivity` options:
- `CONSERVATIVE` — redact detections ≥ 0.30 confidence (recommended; catches phones and SSNs)
- `BALANCED` — redact detections ≥ 0.60 (default; misses most phone numbers on resume data)
- `PERMISSIVE` — redact detections ≥ 0.85 (SSNs only; not recommended for most use cases)