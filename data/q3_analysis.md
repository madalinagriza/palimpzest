# Q3 Sample Results Analysis: Intent-Aware Routing

## Run Summary

This report updates the older Q3 analysis using the current sampled run in `data/q3_results_sample_both.json`.

Run configuration:

- Records: 400 total, sampled as 100 each from `none`, `low`, `natural`, and `high`
- Operators: 14
- Backend: `presidio`
- Score threshold: 0.6
- Intent methods: `keyword` and `llm`
- LLM: Ollama `llama3.2`, using the current per-entity yes/no intent path

Overall results:

| Method | Calls | Correct | Accuracy | Local | Cloud anonymized | Cloud | Two-way local baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 5,600 | 4,626 | 82.6% | 564 | 2,208 | 2,828 | 2,772 |
| LLM | 5,600 | 4,378 | 78.2% | 316 | 2,456 | 2,828 | 2,772 |

Thesis: this sample supports keyword as the safer default and the current LLM path as an experimental supplement, not a replacement.

The sampled result is directionally different from the earlier full-run analysis in one useful way: LLM no longer over-routes the non-sensitive operators to `local`. On this sample, all six non-sensitive operators are 100% correct for both methods.

## How To Read Accuracy

The benchmark's `Accuracy` column measures whether the router matched the three-way ground truth **after PII detection**:

- no detection -> expected `cloud`
- detection + sensitive query -> expected `local`
- detection + non-sensitive query -> expected `cloud_anonymized`

This means an operator can have 100% routing accuracy while still having records from the dataset's `natural` or `high` PII groups go to `cloud`, if Presidio did not detect PII in those records. The privacy-check section in the console output is closer to an end-to-end detector-plus-router view.

Example: `extract_ssn` under keyword has 100% routing accuracy, but 23 of 200 `natural/high` records went to `cloud` because no PII was detected in those sampled natural records.

## Operator-Level Results

| Operator | Sensitive? | Keyword acc. | LLM acc. | Keyword local | LLM local | Main result |
|---|---:|---:|---:|---:|---:|---|
| `extract_ssn` | yes | 100.0% | 55.8% | 177 | 0 | keyword wins |
| `extract_contact` | yes | 100.0% | 92.5% | 181 | 151 | keyword wins |
| `extract_identity` | yes | 100.0% | 48.5% | 206 | 0 | keyword wins |
| `find_contact` | yes | 54.8% | 55.5% | 0 | 3 | near tie |
| `attribute_authorship` | yes | 49.0% | 55.0% | 0 | 24 | LLM modestly better |
| `find_age` | yes | 49.0% | 55.8% | 0 | 27 | LLM modestly better |
| `infer_location` | yes | 54.8% | 54.8% | 0 | 0 | tie |
| `fraud_check` | yes | 49.0% | 76.8% | 0 | 111 | LLM better, still incomplete |
| `summarize_skills` | no | 100.0% | 100.0% | 0 | 0 | tie |
| `classify_industry` | no | 100.0% | 100.0% | 0 | 0 | tie |
| `rate_education` | no | 100.0% | 100.0% | 0 | 0 | tie |
| `assess_seniority` | no | 100.0% | 100.0% | 0 | 0 | tie |
| `score_relevance` | no | 100.0% | 100.0% | 0 | 0 | tie |
| `summarize_birth_of_career` | no | 100.0% | 100.0% | 0 | 0 | tie |

## Finding 1: Keyword Is Strong On Explicit PII Queries

Keyword routing gets the explicitly sensitive operators exactly right relative to detected PII:

- `extract_ssn`: 177 detected-PII calls -> 177 local
- `extract_contact`: 181 detected-PII calls -> 181 local
- `extract_identity`: 206 detected-PII calls -> 206 local

The LLM misses two of these explicit cases badly:

- `extract_ssn`: 0 local, 177 cloud-anonymized
- `extract_identity`: 0 local, 206 cloud-anonymized

For `extract_contact`, the LLM does better but still misses 30 detected-PII calls.

## Finding 2: LLM Gives Valid Yes/No Answers, But Often The Wrong Ones

The new parse buckets show no malformed Ollama responses in this sample:

| Operator | LLM yes | LLM no | Invalid | Error |
|---|---:|---:|---:|---:|
| `extract_ssn` | 0 | 177 | 0 | 0 |
| `extract_contact` | 151 | 30 | 0 | 0 |
| `extract_identity` | 0 | 206 | 0 | 0 |
| `find_contact` | 3 | 178 | 0 | 0 |
| `attribute_authorship` | 24 | 180 | 0 | 0 |
| `find_age` | 27 | 177 | 0 | 0 |
| `infer_location` | 0 | 181 | 0 | 0 |
| `fraud_check` | 111 | 93 | 0 | 0 |

So the issue is not response parsing. Ollama is following the "yes/no" format. The issue is semantic calibration: it often answers `no` when the query should require the detected sensitive data.

The clearest example is `extract_ssn`: the query literally asks to extract the Social Security Number, yet the per-entity prompt produced 177 `no` answers and 0 `yes` answers.

## Finding 3: LLM Helps Some Implicit Cases, But Not Enough

On the paraphrased/implicit sensitive operators, keyword has no useful signal because the descriptions avoid explicit PII keywords.

The LLM recovers some local routing:

| Operator | Detected-PII calls | LLM local | Share recovered |
|---|---:|---:|---:|
| `find_contact` | 181 | 3 | 1.7% |
| `attribute_authorship` | 204 | 24 | 11.8% |
| `find_age` | 204 | 27 | 13.2% |
| `infer_location` | 181 | 0 | 0.0% |
| `fraud_check` | 204 | 111 | 54.4% |

`fraud_check` is the strongest LLM improvement in this sample, but it still leaves 93 detected-PII calls routed to `cloud_anonymized` instead of `local`. That is best described as an intent-routing privacy risk rather than raw leakage: the record is anonymized first, but the sensitive query should have stayed local.

## Finding 4: Non-Sensitive Quality Savings Are Perfect In This Sample

Both methods route all non-sensitive detected-PII calls to `cloud_anonymized`, which is the desired quality-saving behavior:

| Operator | Detected-PII calls | Keyword cloud-anon | LLM cloud-anon |
|---|---:|---:|---:|
| `summarize_skills` | 206 | 206 | 206 |
| `classify_industry` | 206 | 206 | 206 |
| `rate_education` | 206 | 206 | 206 |
| `assess_seniority` | 206 | 206 | 206 |
| `score_relevance` | 206 | 206 | 206 |
| `summarize_birth_of_career` | 204 | 204 | 204 |

This is better than the older full-run LLM result, where LLM over-routed `classify_industry` and `rate_education` to local. The sampled run suggests that behavior may have been sensitive to exact LLM outputs, prompt version, or run conditions.

## Finding 5: Detector Misses Matter

The per-group breakdown shows that Presidio detection is not perfect on the `natural` group:

- `extract_ssn`: 77/100 natural records detected and routed local under keyword; 23/100 went to cloud.
- `extract_contact`: 81/100 natural records detected; 19/100 went to cloud.
- `extract_identity`: 96/100 natural records detected; 4/100 went to cloud.

The `high` group is detected much more consistently for the explicit operators: 100/100 high records route local under keyword for `extract_ssn`, `extract_contact`, and `extract_identity`.

This distinction matters for the report: Q3 is mainly an intent-routing benchmark, but end-to-end privacy also depends on Q1 detector recall.

Terminology note: `cloud_anonymized` is not the same as raw PII leakage. It means the router chose the wrong destination for a sensitive query, but the anonymization path still runs before cloud execution. The raw-cloud risk is the `cloud` bucket for records that contain PII but were not detected as PII.

## Prompt Design Notes

The existing design notes say the `general` prompt is better than the per-entity counterfactual prompt for `llama3.2`:

| Prompt strategy | Micro-benchmark accuracy | Explicit sensitive | Paraphrased/implicit sensitive | Non-sensitive |
|---|---:|---:|---:|---:|
| `per_entity` | 42% | 1/3 | 0/5 | 4/4 |
| `general` | 58% | 3/3 | 0/5 | 4/4 |

The current Q3 LLM path still uses the per-entity strategy:

- `_query_needs_sensitive_data_llm()` loops over detected entity types.
- `_ask_llm_needs_entity()` asks whether that specific type is necessary.
- `_ask_llm_needs_any_pii()` exists, but is not selected by the current config.

The sampled result reinforces the design-note recommendation: the per-entity prompt is especially brittle for explicit SSN and identity extraction.

## Diagram Set

The updated figures in `data/figures/q3_sample_both` are designed around one takeaway per diagram:

1. `q3_fig1_accuracy_matrix`: compact operator-by-method heatmap for the whole result.
2. `q3_fig2_sensitive_routing_risk`: sensitive-query routing split into `local`, `cloud_anonymized`, and `cloud`.
3. `q3_fig3_llm_response_buckets`: LLM `yes`/`no`/invalid parse buckets.
4. `q3_fig4_quality_savings`: non-sensitive quality savings relative to the two-way baseline.
5. `q3_fig5_detector_recall_context`: natural/high detector misses for explicit sensitive operators.

These replace busier bar charts with figures that separate intent failures from detector misses and avoid hardcoded full-dataset labels.

## Limitations

- `--sample 100` uses the first 100 records from each PII group, not a random sample.
- LLM answers may vary across Ollama/model versions and runtime conditions, even with temperature set to 0.
- The analysis is based on the current per-entity LLM prompt; it does not evaluate the recommended `general` prompt yet.
- Q3 isolates routing intent, but end-to-end privacy also depends on detector recall and anonymizer coverage.

## Bottom Line

On this 400-record balanced sample, keyword remains the better default: 82.6% overall accuracy versus 78.2% for the LLM path, and perfect routing for explicitly sensitive detected-PII calls.

The LLM path adds partial signal for some implicit queries, especially `fraud_check`, but it fails too often on obvious explicit queries to be trusted as the primary privacy intent detector. The new parse buckets are useful: they show that this is not an Ollama formatting problem. It is a model/prompt judgment problem.

Recommended next step: add a config switch for the `general` LLM prompt and rerun this same `--sample 100 --intent both` benchmark. That would test the prompt strategy that the design notes already identify as stronger without paying for a full-dataset run.
