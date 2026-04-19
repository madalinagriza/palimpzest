# Mid-Term Project Report
## Privacy-Aware Operator Routing in Palimpzest
### April 2026

---

## 1. Project Overview

This project adds privacy-aware model routing to Palimpzest (PZ), a semantic query processing framework that uses LLMs as first-class query operators. PZ currently sends every operator call to whichever cloud model the optimizer selects, regardless of whether the fields being processed contain PII. Our system inserts a lightweight routing layer that intercepts each operator call before the LLM is invoked, detects PII in the fields that operator actually reads, and redirects to an appropriate execution path.

We implement three routing conditions:

1. **Sensitive data required by the query** → local Llama via Ollama (full privacy)
2. **Sensitive data present but not required** → Presidio anonymizes in place, then cloud (privacy + quality)
3. **No sensitive data** → cloud as-is (full quality)

The primary benchmark dataset is a controlled resume corpus (14,566 records) with four PII tiers — no PII, name+email only, natural PII, and injected SSN+DOB — giving ground-truth labels for every routing decision. Three research questions drive the evaluation:

- **Q1** — At what routing granularity is the quality-privacy tradeoff best: document, field, or operator level?
- **Q2** — Which PII detection backend (Presidio, DeBERTa-v3, regex) produces the best routing decisions?
- **Q3** — Does query-intent awareness — routing based on whether the operator actually needs the sensitive field — improve quality without sacrificing privacy?

---

## 2. Project Status

### 2.1 Completed Tasks

- **PZ architecture mapping.** Documented the full operator graph: how logical plans are constructed, how the Cascades-style optimizer converts them to physical plans, and where model selection is baked in at optimization time. All relevant files and line numbers recorded in `OPERATOR_GRAPH_NOTES.md`.
- **Execution hook identification.** Best hook point is `SequentialSingleThreadExecutionStrategy._execute_plan()` line 88, where both the operator state and the full input record are accessible before any LLM call.
- **`explore_pipeline.py`.** Confirmed that `operator.get_input_fields()` respects `depends_on` (returns only the fields an operator actually reads), and that input/output schemas and generated fields are introspectable at runtime without modifying PZ source.
- **`routing_stub.py`.** Implements `ModelConfig`, two-layer `PrivacyRouter` (Presidio primary + field-name heuristic/regex fallback), `Detection`/`RouteDecision` dataclasses, `RoutingStats` (aggregates local/cloud counts and entity-type breakdown per run), module-level Presidio singleton, fixed model swap for real PZ `Model` instances via the vLLM constructor path, and `execute_with_routing()` wrapper with structured logging.
- **`privacy_execution_strategy.py`.** `PrivacyAwareExecutionStrategy` subclasses `SequentialSingleThreadExecutionStrategy` and replaces the bare `operator(input_record)` call with `execute_with_routing()` for LLM operators. `create_privacy_processor(dataset, config, router)` is a drop-in factory — no PZ source files modified.
- **Resume PII dataset pipeline.** `build_resumes_clean.py` → `reshape_pii.py` → `format_resumes.py` merges 13,389 HuggingFace resumes and 2,483 GitHub PDF resumes into 14,566 deduplicated records. Records are stratified into four PII groups (none: 1,000; low: 1,000; natural: 10,000; high: 2,566 with injected SSN and DOB via Faker) and formatted with six templates (classic, modern, minimal, academic, compact, creative). Documented in `data/phase1_report.md`.
- **Local model baseline (llama3.2, 20 records).** First end-to-end `sem_filter` experiment on 20 stratified records (5 per group) using llama3.2 3B on CPU. Results in `data/sem_filter_report.md`.
- **Cloud baseline (GPT-4o-mini, 100 records).** `sem_filter` run on 100 stratified records (25 per group) using GPT-4o-mini via the OpenAI API. This establishes the reference quality ceiling for the routing benchmark. Results reported in Section 3.

### 2.2 Open Tasks

- **Full routing benchmark (Q1).** Run the three routing conditions end-to-end using `create_privacy_processor` on the resume dataset and compare document-level, field-level, and operator-level routing granularities.
- **PII detector comparison (Q2).** Evaluate Presidio, DeBERTa-v3, and regex-only on the resume dataset's ground-truth PII labels.
- **Query-intent routing (Q3).** Add a `depends_on`-driven check that skips PII detection for fields the operator does not read; test whether routing to cloud with anonymization is safe when the query is insensitive to the PII field.
- **Final report.** Three-section structure mapping to Q1/Q2/Q3, results tables, and discussion of the quality-privacy tradeoff.

---

## 3. Results to Date

### 3.1 Local Model Baseline — llama3.2 3B (CPU, 20 records)

`sem_filter` on a 20-record stratified sample (5 per PII group) asking the model to identify resumes containing PII (SSN, phone number, or a real person's name). Inference ran locally via Ollama.

| PII Group | Records | Kept | Expected | Recall / Spec |
|-----------|---------|------|----------|---------------|
| none | 5 | 0 | 0 | 100% spec |
| low | 5 | 0 | 0 | 100% spec |
| natural | 5 | 1 | 5 | 20% recall |
| high | 5 | 0 | 5 | 0% recall |
| **Total** | **20** | **1** | **10** | |

**Cost:** $0.00 (local). **Latency:** ~33s/record.

The model correctly rejected all 10 records with no PII (100% specificity) but detected PII in only 1 of 10 records that contained phone numbers, addresses, and injected SSNs (10% recall). This is attributable to the model's small size (3B parameters) and CPU inference degrading generation quality.

### 3.2 Cloud Baseline — GPT-4o-mini (100 records)

Same `sem_filter` task on a 100-record stratified sample (25 per group) using GPT-4o-mini via the OpenAI API.

| PII Group | Records | Kept | Expected | Recall / Spec |
|-----------|---------|------|----------|---------------|
| none | 25 | 2 | 0 | 92% spec |
| low | 25 | 1 | 0 | 96% spec |
| natural | 25 | 20 | 25 | 80% recall |
| high | 25 | 25 | 25 | 100% recall |
| **Total** | **100** | **48** | **50** | |

**Cost:** $0.019. **Latency:** ~1.5s/record.

GPT-4o-mini detected PII in all 25 `high`-group records (injected SSNs and DOBs: 100% recall) and 20 of 25 `natural`-group records (phone numbers and addresses: 80% recall). Specificity dropped slightly relative to the local model: 3 false positives appeared across the `none` and `low` groups.

**False positive analysis.** We inspected all three flagged records:

- `hf_002008` (none, minimal template): the `minimal` template left a garbled fragment — `robert smith lead mechanical engineer phone 123 456 78` — that resembles a real name and phone number. This is a data pipeline artifact; PII stripping missed the partially-rendered field.
- `hf_013203` (low, modern template): the resume text ends with `jessica claire montgomery street san francisco ca 000 resumesampleexamp`, a watermark from a third-party resume template site that looks exactly like a name and address. GPT correctly flagged it; it is a genuine data quality issue, not a model error.
- `hf_010326` (none, modern template): no obvious PII artifact was found. This is a true false positive — the model hallucinated a PII signal in noisy truncated text.

Only one of the three false positives is a genuine model error. The other two reflect imperfections in the data pipeline that are useful to surface.

### 3.3 Comparison Summary

| Metric | llama3.2 (local, 3B) | GPT-4o-mini (cloud) |
|--------|----------------------|---------------------|
| Recall — natural group | 20% | 80% |
| Recall — high group | 0% | 100% |
| Specificity — none group | 100% | 92% |
| Specificity — low group | 100% | 96% |
| Latency per record | ~33s | ~1.5s |
| Cost per 100 records | $0.00 | $0.019 |

GPT-4o-mini provides the reference quality ceiling for the routing benchmark. The routing system will aim to match this on non-PII records (by routing them to cloud) while keeping sensitive records local, accepting the quality gap on locally-routed records as the privacy cost.

---

## 4. Potential Problems and Mitigations

- **Local model quality gap is empirically confirmed.** llama3.2 recall on `sem_filter` is 10% vs. GPT-4o-mini's 80–100%. This does not block the routing system (routing decisions are made by Presidio/regex on known labels, not by the LLM), but it means downstream extraction quality for locally-routed records will degrade measurably. The benchmark will report this degradation explicitly across PII groups.
- **Presidio false-positive rate on resumes.** Resume text contains names and dates that are not PII in context (section headers misidentified as names, year ranges matching phone patterns). Mitigation: tune `score_threshold`; consider disabling the `PERSON` detector in favor of structural PII types (SSN, phone, email) for routing decisions.
- **Data pipeline artifacts.** Two of three GPT-4o-mini false positives trace to the data pipeline (garbled `minimal` template rendering, third-party resume watermarks) rather than the model. We will flag these records in the ground-truth labels and exclude them from precision/recall computation, or fix the pipeline and re-run.
- **Routing condition 2 semantic preservation.** Anonymize-then-cloud requires Presidio's anonymizer to preserve enough content for GPT-4o-mini to answer the query. We will pilot on 10 records from the `high` group before full benchmark.

---

## 5. Updated Timeline

| Period | Status | Work |
|--------|--------|------|
| Weeks 1–7 | Complete | Architecture mapping, PZ execution stack documentation, `explore_pipeline.py`, initial routing stub |
| Week 8 | Complete | Presidio integration, `Detection`/`RouteDecision` dataclasses, regex/heuristic fallback, resume dataset pipeline, llama3.2 `sem_filter` experiment |
| Week 9 | Complete | `RoutingStats`, Presidio singleton, Model enum swap fix, `PrivacyAwareExecutionStrategy` + `create_privacy_processor`, project plan finalized |
| Week 10 | Complete | GPT-4o-mini cloud baseline (100 records); false positive analysis |
| Week 11 | In progress | Full routing benchmark (Q1): three conditions × three granularities; PII detector comparison (Q2); query-intent routing pilot (Q3) |
| Week 12 | Upcoming | Final report: results tables, figures, Q1/Q2/Q3 discussion |

---

## 6. Individual Contributions

**Person A (Denis)**
- PZ architecture reverse-engineering and documentation (`OPERATOR_GRAPH_NOTES.md`); `explore_pipeline.py`; initial `routing_stub.py` design.
- Iterative improvements to `routing_stub.py`: `RoutingStats`, Presidio singleton, fixed Model enum swap, `local_api_base`.
- `privacy_execution_strategy.py`: `PrivacyAwareExecutionStrategy` and `create_privacy_processor` factory.
- GPT-4o-mini cloud baseline run and false positive analysis.
- Will lead Q1 benchmarks (routing granularity comparison) in week 11.

**Person B (Onopre)**
- Full Presidio integration into `PrivacyRouter` (two-layer detector: Presidio + heuristic/regex fallback), `Detection`/`RouteDecision` dataclasses, detection-aware logging, extended smoke test.
- Will lead Q2 (PII detector comparison: Presidio vs. DeBERTa-v3 vs. regex) and Q3 track 1 (query-intent routing implementation) in week 11.

**Person C (Madalina)**
- Resume PII dataset pipeline (`build_resumes_clean.py` → `reshape_pii.py` → `format_resumes.py`): 14,566 deduplicated records across four PII groups with SSN/DOB injection and six formatting templates. Documented in `data/phase1_report.md`.
- First end-to-end experiment (`demos/resume-pii-demo.py`): `sem_filter` on 20 stratified records with llama3.2; results in `data/sem_filter_report.md`.
- Project plan: defined research questions, routing logic, agreed metrics, failure-mode threshold, and all design decisions.
- Will lead full benchmark execution in week 11.

---

## 7. Questions for Feedback

- **Low local-model recall (10%).** Our results show llama3.2 fails to detect PII in 9 of 10 records containing phone numbers and SSNs. Should we treat the LLM-based `sem_filter` purely as the task under evaluation (not the routing decision maker), and always use Presidio/regex for the routing decision? Or is this finding itself a reportable contribution — that naive local-LLM routing is insufficient and a dedicated detector is necessary?
- **Routing condition 2 scope.** Anonymize-then-cloud requires Presidio's anonymizer to preserve enough semantic content for the cloud model to answer the query. Is it acceptable to scope this condition to only the `high`-PII group, where SSN/DOB fields are injected and clearly separable from query-relevant content?
- **Q3 feasibility.** Query-intent routing is architecturally simple given our implementation — `get_input_fields()` already respects `depends_on`. But defining ground-truth labels for "does this query need this field" requires manual annotation. Is it acceptable to use the `depends_on` values set by the pipeline author as a proxy for query intent, without additional annotation?
- **Data pipeline artifacts in evaluation.** Two of three GPT-4o-mini false positives trace to the data pipeline rather than the model. Should we fix the pipeline and re-run the baseline, or exclude these records from evaluation metrics and report the artifact rate separately?
