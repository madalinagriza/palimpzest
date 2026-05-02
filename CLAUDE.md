# MIT 6.5831 Privacy Routing Project — Claude Context

This file lets Claude (or any AI assistant) quickly understand the project state without reading the full codebase.

## Project in one sentence

We added privacy-aware LLM routing to Palimpzest (PZ): before each operator call, we scan the record for PII and route to `local` (Ollama), `cloud_anonymized` (Presidio-redacted → cloud), or `cloud` (no PII → cloud directly).

## Working directory

```
/Users/simdenis/Desktop/6.5831/project/palimpzest/
```

All commands below assume this as CWD.

## Key files

```
privacy/
  routing_stub.py               # PrivacyRouter, ModelConfig, execute_with_routing — CORE LOGIC
  privacy_execution_strategy.py # PrivacyAwareExecutionStrategy + create_privacy_processor factory
  benchmark_granularity.py      # Q1 end-to-end LLM benchmark (all modes)
  benchmark_sensitivity.py      # anonymization sensitivity knob benchmark
  requirements.txt

demos/
  benchmark_q1.py               # Q1 dry-run (no LLM, fast, routing logic only)
  benchmark_q3.py               # Q3 dry-run, 14 operator configs
  gpt4o_baseline.py             # GPT-4o-mini cloud quality baseline
  resume-pii-demo.py            # end-to-end demo

data/
  resumes_with_pii.jsonl        # 14,566-record resume corpus (none/low/natural/high groups)
  q3_analysis.md                # full Q3 findings (keyword vs LLM intent, 14 operators)
```

## Local model

We use **qwen2.5:7b** via Ollama on the OpenAI-compatible endpoint:

```
model:    openai/qwen2.5:7b
api_base: http://localhost:11434/v1
```

Start Ollama before running any `--intent llm` benchmark:

```bash
ollama serve   # if not already running
ollama pull qwen2.5:7b
```

## How to run benchmarks

All commands from inside `palimpzest/`:

```bash
# Q1: granularity comparison — 3 modes

# Mode: single (one sem_filter, OPERATOR vs FIELD vs DOCUMENT granularity)
.venv/bin/python privacy/benchmark_granularity.py --sample 31 --intent keyword
.venv/bin/python privacy/benchmark_granularity.py --sample 31 --intent llm

# Mode: prompts (8 different filter queries, OPERATOR granularity, shows intent correctness)
.venv/bin/python privacy/benchmark_granularity.py --sample 10 --mode prompts --intent keyword
.venv/bin/python privacy/benchmark_granularity.py --sample 10 --mode prompts --intent llm

# Mode: multi (sem_map → sem_filter pipeline, OPERATOR vs DOCUMENT granularity)
# Key result: DOCUMENT over-routes Op2 (skills_summary, PII-free) to local
.venv/bin/python privacy/benchmark_granularity.py --sample 5 --mode multi --intent keyword

# Q3 dry-run (no LLM, fast, all 14 operator configs)
.venv/bin/python demos/benchmark_q3.py --sample 100
```

## Three routing paths

```
record has PII + query needs it    → local          (qwen2.5:7b via Ollama)
record has PII + query doesn't     → cloud_anonymized (Presidio redacts → cloud GPT-4o)
record has no PII                  → cloud           (GPT-4o directly)
```

## Routing granularities

- `OPERATOR` — scan only the fields in `depends_on` for each operator call (recommended)
- `FIELD` — scan all schema fields regardless of `depends_on`
- `DOCUMENT` — scan all fields once per record, reuse decision for all operators in the pipeline

Granularity only matters for **multi-operator pipelines**. For a single operator, all three give the same result.

## Intent detection methods

- `keyword` — string match of the operator query against a fixed sensitive-vocabulary list
  - Perfect on explicit queries ("Extract the Social Security Number")
  - Fails on implicit/paraphrased queries ("Find the best way to contact this applicant")
- `llm` — asks qwen2.5:7b yes/no per detected entity per record
  - Per-record (not cached across records) — important: earlier bug where same answer was reused for all records, now fixed
  - Small 7B model often fails even on explicit queries; not recommended for production

## Bugs fixed in this session (all in `privacy/routing_stub.py`)

1. **Query text extraction** — `operator.filter` returned the bound method object, not the filter string. Fixed to use `operator.filter_obj` which holds the actual `LLMFilter` object with `.filter_condition`.

2. **LLM intent cache keyed per (query, entity_type) only** — same answer reused for all records regardless of what PII was actually detected. Fixed: cache key now includes the detected value preview `(query_text, entity_type, detected_value_preview)`.

3. **LLM API endpoint** — `_ollama_yes_no()` was calling Ollama's native `/api/generate` endpoint. Switched to the OpenAI-compatible `/v1/chat/completions` endpoint at `local_api_base`, consistent with how PZ itself calls the model.

4. **Stats double-counting in DOCUMENT granularity** — `inspect()` called `stats.record()`, then `execute_with_routing()` also called `stats.record()` in the cached-decision branch, doubling all counts. Fixed: `inspect()` no longer records stats; only `execute_with_routing()` does (once, after all routing decisions).

## `benchmark_granularity.py` modes (as of this session)

### `--mode single` (default)
- Runs the original Q1 benchmark: one sem_filter ("does this resume contain PII?")
- Tries all 3 granularities (OPERATOR, FIELD, DOCUMENT)
- Reports: local%, cloud_anonymized%, cloud%, precision, recall, F1

### `--mode prompts`
- Loops over `FILTER_CONFIGS` — 8 filter queries spanning:
  - Sensitive + keyword visible: `extract_ssn`, `extract_contact`
  - Sensitive + implicit: `find_contact`, `attribute_authorship`, `fraud_check`
  - Non-sensitive: `summarize_skills`, `assess_seniority`, `score_relevance`
- Uses OPERATOR granularity (fixed)
- Reports routing accuracy per query (% PII records routed to correct destination)
- Key output: keyword method perfect on explicit queries, fails on implicit; LLM inconsistent

### `--mode multi`
- Two-operator pipeline: sem_map (summarize skills from `text`) → sem_filter (5yr+ experience?)
- Op1 reads `text` (may have embedded PII); Op2 reads `skills_summary` (PII-free derived field)
- Compares OPERATOR vs DOCUMENT granularity
- Expected result: OPERATOR routes Op2 to cloud (correct); DOCUMENT over-routes Op2 to local

## `FILTER_CONFIGS` in benchmark_granularity.py

```python
# Sensitive queries (ground truth: PII records → local)
extract_ssn          "Extract the Social Security Number from the resume text."
extract_contact      "Find the applicant's phone number and email address."
find_contact         "Find the best way to contact this applicant."          # keyword misses
attribute_authorship "Who wrote this resume? What is their background?"       # keyword misses
fraud_check          "Does anything about this application suggest..."        # keyword misses

# Non-sensitive queries (ground truth: PII records → cloud_anonymized)
summarize_skills     "Summarize the applicant's technical skills..."
assess_seniority     "Rate the applicant's seniority level..."
score_relevance      "Score this resume from 1 to 10..."
```

## Ground truth (for Q1 sem_filter quality metrics)

| pii_group | Expected by sem_filter |
|-----------|------------------------|
| `natural` | Accept (has real PII)  |
| `high`    | Accept (has real PII)  |
| `none`    | Reject                 |
| `low`     | Reject                 |

## Q3 key findings (see `data/q3_analysis.md` for full details)

- **Keyword** wins on explicit sensitive queries (100%) and all non-sensitive queries (100%)
- **Keyword** fails completely on paraphrased/implicit sensitive queries (17–25%)
- **LLM (qwen2.5:7b / llama3.2 3B)** fails even on explicit queries like "Extract the SSN" (routes 0 records to local for extract_ssn) — model-size/calibration failure
- **Verdict**: keyword method is strictly better than 3B LLM for this task; a larger model (70B+) would be needed for LLM method to add value

## What still needs to be done

- [ ] Re-run `--mode single` with all 3 granularities now that the stats double-count bug is fixed (document granularity showed inflated counts before)
- [ ] Run `--mode prompts` with both intent methods to populate Q1 multi-prompt results table
- [ ] Run `--mode multi` to get the OPERATOR vs DOCUMENT over-routing demonstration
- [ ] Write final report section: Q1 (granularity + multi-operator), Q3 analysis (see q3_analysis.md)

## Project structure context

- **No PZ source files are modified** — the privacy layer is entirely additive
- `create_privacy_processor()` in `privacy_execution_strategy.py` is the main entry point
- `PrivacyRouter` in `routing_stub.py` holds all routing logic, PII detection, and stats
- `RoutingStats.summary()` prints a one-line summary of routing decisions for a run
