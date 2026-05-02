# Q3 Results Analysis: Intent-Aware Routing

## Inputs Reviewed

This analysis reflects the current benchmark code in `demos/benchmark_q3.py`, the router code in `privacy/routing_stub.py`, and two result files:

- `data/q3_results.json`: 100 records per PII group, 6 original operators, keyword intent only.
- `data/q3_results_both.json`: full dataset, 14 operators, Presidio backend, keyword and LLM intent compared.

It also incorporates the prompt notes in:

- `data/llm_prompt_comparison.md`
- `data/llm_intent_design_notes.md`

## What Q3 Measures

Q3 tests whether routing should use query intent in addition to PII detection.

The naive two-way baseline is:

- PII detected -> `local`
- no PII detected -> `cloud`

The intended three-way policy is:

- PII detected and query needs the sensitive data -> `local`
- PII detected and query does not need the sensitive data -> `cloud_anonymized`
- no PII detected -> `cloud`

The benchmark uses fake operators whose `desc` is the query text and whose `depends_on` fields determine which record fields are scanned for PII. Ground truth is encoded manually via each operator's `sensitive_query` flag.

## Result File Roles

`q3_results.json` is a small keyword-only sanity run:

| File | Scope | Intent method | Operators | Calls | Correct | Accuracy |
|---|---:|---|---:|---:|---:|---:|
| `q3_results.json` | 100/group sample | keyword | 6 | 2,400 | 2,400 | 100.0% |

That file covers only the original easy cases: three explicit sensitive queries and three non-sensitive queries. It should not be used to judge paraphrased or implicit intent handling.

`q3_results_both.json` is the main Q3 result:

| Method | Operators | Calls | Correct | Accuracy | Local | Cloud anonymized | Cloud |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 14 | 203,924 | 145,706 | 71.5% | 33,717 | 132,921 | 37,286 |
| LLM | 14 | 203,924 | 111,989 | 54.9% | 9,224 | 157,414 | 37,286 |

## Operator-Level Summary

| Operator | Sensitive? | Keyword acc. | LLM acc. | Main result |
|---|---:|---:|---:|---|
| `extract_ssn` | yes | 100.0% | 29.0% | keyword wins |
| `extract_contact` | yes | 100.0% | 25.4% | keyword wins |
| `extract_identity` | yes | 100.0% | 30.0% | keyword wins |
| `find_contact` | yes | 25.4% | 25.4% | both fail |
| `attribute_authorship` | yes | 16.5% | 32.4% | LLM has partial signal |
| `find_age` | yes | 16.5% | 16.5% | both fail |
| `infer_location` | yes | 25.4% | 25.4% | both fail |
| `fraud_check` | yes | 16.5% | 16.5% | both fail |
| `summarize_skills` | no | 100.0% | 100.0% | tie |
| `classify_industry` | no | 100.0% | 84.2% | keyword wins |
| `rate_education` | no | 100.0% | 84.2% | keyword wins |
| `assess_seniority` | no | 100.0% | 100.0% | tie |
| `score_relevance` | no | 100.0% | 100.0% | tie |
| `summarize_birth_of_career` | no | 100.0% | 100.0% | tie |

## Key Findings

Keyword intent is reliable only when the query contains explicit sensitive terms.

It gets all explicit sensitive operators right: `extract_ssn`, `extract_contact`, and `extract_identity`. It also gets all non-sensitive operators right in the full run, preserving maximum cloud-quality savings.

Keyword fails on paraphrased or implicit sensitive intent:

- `find_contact`: "best way to contact" needs phone/email, but the keyword list does not include this phrase.
- `attribute_authorship`: "Who wrote this resume?" implies identity/name, but no explicit name keyword fires.
- `find_age`: asks for age, but the benchmark depends on fields that do not encode DOB directly.
- `infer_location`: infers location from phone/SSN-like data, but no location keyword is treated as sensitive.
- `fraud_check`: likely needs name/SSN verification, but the dependency is implicit.

The LLM path in `q3_results_both.json` is weaker than keyword overall. It routes zero detected-PII records to `local` for several obvious sensitive operators, including `extract_ssn` and `extract_contact`. It only provides partial local routing for `extract_identity` and `attribute_authorship` with 2,306 local calls each.

The LLM also over-routes two non-sensitive operators to `local`:

| Operator | Keyword cloud-anon savings | LLM cloud-anon savings | LLM unnecessary local calls |
|---|---:|---:|---:|
| `classify_industry` | 12,509 | 10,203 | 2,306 |
| `rate_education` | 12,509 | 10,203 | 2,306 |

## Prompt Notes vs Current Router Code

The prompt notes show that the `general` LLM prompt is better than the `per_entity` counterfactual prompt on the 12-query micro-benchmark:

| Prompt strategy | Accuracy | Explicit sensitive | Paraphrased/implicit sensitive | Non-sensitive |
|---|---:|---:|---:|---:|
| `per_entity` | 42% | 1/3 | 0/5 | 4/4 |
| `general` | 58% | 3/3 | 0/5 | 4/4 |

However, the current router still uses the `per_entity` path for `intent_method="llm"`:

- `_ask_llm_needs_entity(query_text, entity_type)` is called from `_query_needs_sensitive_data_llm`.
- `_ask_llm_needs_any_pii(query_text)` exists, but is not currently selected by `ModelConfig`.
- The code comment mentions `ModelConfig.intent_llm_prompt`, but that config option is not currently present.

So the full `q3_results_both.json` LLM numbers should be interpreted as the older `per_entity` Ollama strategy, not the recommended `general` prompt from the notes.

## Ollama Yes/No Parse Bucket

Before this update, the benchmark did not account for non-yes/no Ollama responses as a separate bucket.

The previous parser did this:

- take the first word of the Ollama response
- return `True` only when the first word is `yes`
- treat every other response as `False`

That meant a refusal, explanation, empty answer, or malformed response was counted as a clean `no`, which could route sensitive queries to `cloud_anonymized`.

The current code now tracks LLM parse status on each routing decision:

- `yes`
- `no`
- `invalid`
- `error`
- `missing_query`

Future benchmark JSON output includes these counts under `llm_response_buckets`. Invalid and error responses are treated conservatively as `needs_sensitive=True`, sending detected-PII records to `local`.

Important caveat: the existing `q3_results_both.json` file was generated before this bucket existed, so it cannot report how many LLM answers were invalid. Rerun Q3 to populate the new bucket fields.

## Bottom Line

For the current full benchmark result, keyword routing is better than the small local Ollama path: 71.5% vs 54.9% overall accuracy, perfect handling of explicit sensitive queries, and no quality regression on non-sensitive operators.

The LLM experiment is still useful, but mostly as a prompt-design lesson. The `per_entity` counterfactual prompt is brittle for `llama3.2` 3B. The notes support switching the router to the `general` prompt before rerunning Q3, then comparing against both keyword and the existing `per_entity` result.
