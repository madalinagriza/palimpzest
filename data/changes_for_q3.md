# Changes For Q3

## Purpose

This note records the common Python files modified for the Q3 query-intent benchmark work and why they changed.

## Modified Python Files

### `privacy/routing_stub.py`

Why it changed:

- Added `RouteDecision.llm_intent_status` so each LLM intent decision can record whether Ollama returned `yes`, `no`, `invalid`, `error`, or `missing_query`.
- Changed the Ollama yes/no parser to distinguish valid `yes` and `no` answers from malformed responses.
- Made invalid/error LLM responses conservative: they are treated as `needs_sensitive=True`, which routes detected-PII records to `local`.
- Preserved the existing per-entity LLM prompt behavior, while making its outputs auditable.

Effect:

- Q3 can now tell whether LLM failures come from bad parsing or from valid but wrong semantic judgments.
- The sampled run showed `invalid=0` and `error=0`, so the observed LLM failures are not formatting failures.

### `demos/benchmark_q3.py`

Why it changed:

- Added LLM response bucket counters to `OperatorMetrics`.
- Wrote `llm_response_buckets` into the benchmark JSON output.
- Printed an LLM response bucket table in the console report for `--intent llm` and `--intent both`.
- Included per-record `llm_intent_status` in the in-memory `OperatorResult` objects.

Effect:

- New Q3 runs can report how often Ollama says `yes`, `no`, returns malformed output, or fails.
- `data/q3_results_sample_both.json` includes the new bucket fields used by the updated analysis.

### `demos/plot_q3.py`

Why it changed:

- Replaced the older busy bar-chart set with sample-aware figures.
- Removed hardcoded full-dataset labels such as `14,566 records` and `n = 12,566`.
- Split sensitive-query outcomes into three categories: `local`, `cloud_anonymized`, and `cloud`.
- Added a dedicated LLM response bucket figure.
- Added a detector-recall context figure for explicit sensitive operators.
- Set Matplotlib to the non-interactive `Agg` backend so figure generation works without a local Tk GUI install.

Effect:

- Figures now match the sampled `--sample 100` Q3 run.
- Diagrams distinguish intent-routing risk (`cloud_anonymized`) from detector misses/raw-cloud risk (`cloud`).
- The default plotting input is now `data/q3_results_sample_both.json`, with output to `data/figures/q3_sample_both`.

## Modified Report Files

### `data/q3_analysis.md`

Why it changed:

- Rebuilt the report around the current sampled Q3 run.
- Added the main thesis: keyword is the safer default, while the current LLM path is an experimental supplement.
- Added parse-bucket interpretation showing that Ollama gave valid yes/no answers.
- Clarified the difference between benchmark routing accuracy and end-to-end detector-plus-router privacy behavior.
- Added diagram descriptions and limitations.

### `data/changes_for_q3.md`

Why it exists:

- Provides a compact audit trail for the Q3-related code/report changes.

## Generated Result/Artifact Files

- `data/q3_results_sample_both.json`: sampled Q3 result with keyword and LLM methods plus LLM response buckets.
- `data/figures/q3_sample_both/`: updated Q3 diagrams generated from the sampled result.
