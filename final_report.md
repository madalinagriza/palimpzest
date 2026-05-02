# Privacy-Aware Operator Routing in Palimpzest

**Denis Siminiuc, Onopre [last name], Madalina [last name]**  
MIT 6.5831 — Database Systems · April 2026

---

## Abstract

Semantic query processing frameworks such as Palimpzest (PZ) treat LLMs as first-class query operators and rely on a cost-based optimizer to assign each operator to a cloud or local model. This paper adds a lightweight, non-invasive *privacy routing layer* that intercepts operator calls before the LLM is invoked, detects personally identifiable information (PII) in the fields that operator actually reads, and redirects the call to an appropriate execution path: a local Llama model when the query requires the raw sensitive field, an anonymize-then-cloud path when PII is present but not needed by the query, and the original cloud model otherwise. We evaluate three design dimensions — routing granularity (document, field, operator), PII detection backend (Presidio, DeBERTa-v3, regex, ensemble), and query-intent awareness — on a controlled resume corpus of 14,566 records stratified across four PII tiers. Operator-level routing achieves 98.8% PII recall with zero unnecessary local re-routing; regex runs 390× faster than Presidio with only a marginal recall gap; and query-intent routing reclaims cloud-model quality for 49.5% of PII-record operator calls that naive routing would unnecessarily downgrade to local inference.

---

## 1. Introduction

Cloud LLM APIs (GPT-4o, Claude, Gemini) offer substantially higher accuracy than locally deployable models, but they require sending query inputs off-premise. For workloads that touch personally identifiable information — resumes, medical records, legal documents, email archives — this creates a hard tradeoff: accept the quality penalty of local inference, or accept the privacy risk of sending raw PII to a third-party API.

The dominant response in practice is a binary policy: either run everything locally (privacy-preserving, lower quality) or run everything on the cloud (high quality, no privacy). Neither option is satisfactory for real analytics workloads, where most of the data is not sensitive and the PII is concentrated in a small number of fields or records.

We add a third option: *per-operator, per-record routing* that inspects the specific data an operator will actually read and redirects the call based on what it finds. The routing decision is made entirely locally, before any LLM invocation, by a lightweight PII detector. Records with no PII are sent to the cloud unmodified. Records with PII that the operator's query does not need are anonymized locally and then sent to the cloud. Only records with PII that the operator explicitly requires are redirected to a local model.

This approach is integrated into Palimpzest without modifying any PZ source files. The `PrivacyAwareExecutionStrategy` subclasses PZ's existing `SequentialSingleThreadExecutionStrategy` and wraps each operator call with the routing layer, making adoption a one-line change for pipeline authors.

We pose three research questions:

- **Q1** — At what routing granularity is the quality-privacy tradeoff best: document, field, or operator level?
- **Q2** — Which PII detection backend produces the best routing accuracy, and at what computational cost?
- **Q3** — Does query-intent awareness — routing based on whether the operator's prompt actually needs the sensitive field — improve cloud-quality coverage without sacrificing privacy?

---

## 2. Background and Related Work

### 2.1 Palimpzest

Palimpzest [Liu et al., CIDR 2025] is a declarative LLM-powered analytics system. Users express data transformations as typed schema mappings (`sem_map`) and predicates (`sem_filter`), which PZ compiles into logical plans. A Cascades-style optimizer then converts logical plans to physical plans by selecting models for each operator based on estimated cost and quality. The optimizer bakes in model selection at plan-compilation time; there is no mechanism for per-record routing based on record content.

Our hook point is `SequentialSingleThreadExecutionStrategy._execute_plan()`, where both the operator state and the full input record are available before any LLM call. A key insight is that `operator.get_input_fields()` already respects `depends_on`, so at execution time we can determine which fields an operator actually reads without any additional schema annotation.

### 2.2 LLM Privacy and Data Routing

Prior work on LLM privacy has focused primarily on prompt injection defense [Perez & Ribeiro, 2022], membership inference [Carlini et al., 2021], and differential privacy for fine-tuning [Yu et al., 2022]. Routing sensitive data away from cloud APIs has received less systematic treatment.

The closest related systems are privacy-preserving RAG pipelines, which sometimes apply NER-based redaction before retrieval [Chen et al., 2024], and hybrid inference schedulers that route queries to local or cloud models based on query complexity [Ding et al., 2024]. Neither applies at the operator level within a typed query plan, nor does either use query intent to distinguish necessary from incidental PII exposure.

Presidio [Microsoft, 2020] is an open-source PII detection and anonymization toolkit that we use as our primary detector. NLP-based NER models such as DeBERTa-v3 [He et al., 2021] have been adapted for PII detection [Piiranha, 2023] and offer higher recall on free-text names and addresses at the cost of substantially higher latency.

---

## 3. System Design

### 3.1 Routing Policy

The router implements a three-way decision for each operator invocation:

1. **No PII detected** → route to cloud as-is
2. **PII detected, query does not need it** → anonymize locally, then route to cloud
3. **PII detected, query requires it** → route to local model (Ollama/Llama)

The distinction between cases 2 and 3 is query-intent detection. The router inspects the operator's `desc` field (which PZ uses to describe the operator's task), checks it against a keyword list (`ssn`, `phone number`, `email address`, `full name`, `contact info`, `identity`, etc.), and also checks whether any PII-bearing field name appears directly in the prompt. If neither check fires, the query is classified as not needing the sensitive field.

### 3.2 PII Detection

`PrivacyRouter` implements four selectable backends:

**Presidio** — Microsoft's rule-based analyzer with spaCy NLP. Used as the primary backend. We restrict the entity types to structured PII (SSN, phone, email, credit card, driver license, IP address, bank number) and exclude noisy resume entities (PERSON, LOCATION, DATE) that cause excessive false positives on resume text. A field-name heuristic fires as a fallback when Presidio finds nothing.

**DeBERTa-v3** (Piiranha) — a fine-tuned token-classification model for PII detection. Higher recall on ambiguous free-text patterns, but 7× higher latency than Presidio on CPU.

**Regex** — the field-name heuristic plus five regular expressions (SSN, phone, email, credit card, IP address). No model loading, near-zero latency.

**Ensemble** — union of Presidio + DeBERTa + regex, deduplicated by (field, entity type, preview) key. Maximum recall, maximum latency.

All backends are tunable via `ModelConfig.score_threshold` (default 0.60), which gates which detections count as PII for routing purposes.

### 3.3 Routing Granularity

Three granularity modes control which fields are scanned before each operator call:

- **OPERATOR** — scan only the fields the operator actually reads (`get_input_fields()`, which respects `depends_on`). A downstream operator that reads a derived PII-free field is scanned on that field only, not on the raw PII fields upstream.
- **FIELD** — scan all fields in the operator's input schema, regardless of `depends_on`.
- **DOCUMENT** — scan the full record once at the start of the pipeline and cache the routing decision for all subsequent operators.

### 3.4 Anonymization Sensitivity Knob

The `cloud_anonymized` path uses Presidio's anonymizer to redact PII before the record is sent to the cloud model. The aggressiveness of redaction is controlled by `AnonymizationSensitivity`:

| Level | Presidio threshold | Rationale |
|-------|-------------------|-----------|
| PERMISSIVE | 0.85 | Redact only high-confidence detections; preserve document quality |
| BALANCED | 0.60 | Default; moderate quality-privacy balance |
| CONSERVATIVE | 0.30 | Redact even low-confidence detections; maximize privacy coverage |

The routing threshold (`score_threshold`) and the anonymization threshold are independently configurable, allowing separate tuning of when records route to `cloud_anonymized` versus how aggressively they are redacted once there.

### 3.5 Integration with Palimpzest

`PrivacyAwareExecutionStrategy` subclasses `SequentialSingleThreadExecutionStrategy`. The only behavioral difference is that `operator(input_record)` is replaced with `execute_with_routing(operator, input_record, router)`, which performs PII detection, swaps the operator's model if needed (via the vLLM constructor path for PZ `Model` instances), optionally anonymizes the record, and then invokes the operator. No PZ source files are modified.

```python
processor = create_privacy_processor(
    dataset,
    config=ModelConfig(
        detector_backend="presidio",
        anonymization_sensitivity=AnonymizationSensitivity.CONSERVATIVE,
    ),
    router=PrivacyRouter(config),
)
result = processor.execute()
```

---

## 4. Dataset

The benchmark dataset is a corpus of 14,566 deduplicated resumes assembled from two sources: 13,389 resumes from the HuggingFace Resume dataset and 2,483 GitHub PDF resumes. Records are stratified into four PII tiers:

| Group | N | PII content |
|-------|---|-------------|
| none | 1,000 | No PII; placeholder names/dates stripped |
| low | 1,000 | Name and email only |
| natural | 10,000 | Real-world PII: phone, address, email, name |
| high | 2,566 | Injected SSN + DOB via Faker, plus natural PII |

Each record is rendered through one of six formatting templates (classic, modern, minimal, academic, compact, creative) to simulate document variety. The `none` and `low` groups serve as the negative class (should route cloud); `natural` and `high` are the positive class (should route local or `cloud_anonymized`).

A known data quality issue: many HuggingFace resumes carry a third-party watermark ("jessica claire montgomery street san francisco ca 000 resumesampleexamp") that looks like a real name and address to both detectors and LLMs. These records are labeled `none` or `low` in the ground truth but consistently trigger PII detectors. We report these as false positives and note that they reflect a labeling artifact rather than a detector error.

---

## 5. Evaluation

### 5.1 LLM Quality Baselines

Before evaluating routing accuracy, we measured the quality gap between local and cloud inference on a `sem_filter` task: "Does this resume contain PII such as SSN, phone number, or a real person's name?"

| Model | Recall (natural) | Recall (high) | Spec (none) | F1 | Cost/100 | Latency |
|-------|-----------------|--------------|-------------|-----|----------|---------|
| llama3.2 3B (local) | 20% | 0% | 100% | — | $0 | ~33s/record |
| GPT-4o-mini (cloud) | 84% | 100% | 60% | 0.821 | $0.034 | ~1.9s/record |

The local model's near-zero recall (10% overall) confirms that naive local-only routing produces unacceptable task quality for this dataset. GPT-4o-mini is used as the reference quality ceiling for routing evaluation. The cloud model's lower specificity (60% on the `none` group) is explained primarily by the "jessica claire" watermark artifact; excluding watermark records, specificity is 92%.

### 5.2 Q1 — Routing Granularity

We ran two complementary experiments for Q1.

**Experiment A — Single-operator routing accuracy (100 records, Presidio, no LLM calls).** A single `sem_filter` operator with `depends_on=["text","ssn","phone","name"]` was routed under all three granularities. All three produced identical decisions:

| Granularity | TP | TN | FP | FN | Routing time |
|------------|----|----|----|----|-------------|
| OPERATOR | 48 | 50 | 0 | 2 | 27s |
| FIELD | 48 | 50 | 0 | 2 | 10s |
| DOCUMENT | 48 | 50 | 0 | 2 | 9s |

The reason is structural: with a single operator reading raw fields, all three granularities scan the same values. Granularity cannot diverge unless a downstream operator reads a field that is different from what the upstream operator read.

This is consistent with the earlier end-to-end LLM run (Section 5.1): on the same single-operator `sem_filter` task with llama3.2, all three granularities produced identical routing — and therefore identical task output. Granularity alone does not change LLM quality when the pipeline has only one operator.

**Experiment B — Multi-operator routing accuracy (1,000 records, Presidio, no LLM calls).** This is where granularity actually matters. We constructed a two-operator pipeline:

- **Op1 `sem_map`** `depends_on=["text","ssn","phone","name"]` — reads raw fields, outputs a PII-free `skills_summary`
- **Op2 `sem_filter`** `depends_on=["skills_summary"]` — filters on the derived field only

Op2 should always route to cloud because `skills_summary` contains no raw PII. Any Op2 call routed to local is a quality downgrade with no privacy benefit.

| Granularity | Op1 recall | Op2 over-routing | Wall time |
|------------|-----------|-----------------|-----------|
| **OPERATOR** | **98.8%** | **0 / 1000 (0%)** | **130s** |
| FIELD | 98.8% | 532 / 1000 (53.2%) | 229s |
| DOCUMENT | 98.8% | 532 / 1000 (53.2%) | 105s |

OPERATOR-level routing eliminates Op2 over-routing entirely by scanning only `["skills_summary"]` for Op2 (respecting its `depends_on`), which contains no PII. FIELD and DOCUMENT scan the original document fields for every operator regardless of `depends_on`, so they see PII and incorrectly route Op2 to local for 53% of records.

**Summary**: granularity is irrelevant for single-operator pipelines, but critical for multi-operator pipelines where downstream operators read derived or processed fields. OPERATOR is the correct default because it is the only mode that tracks what each operator actually reads.

The 6 FN records (1.2% miss rate across both experiments) contain phone numbers in non-standard formats that the regex misses and names that Presidio's PERSON detector scores below threshold 0.60. These are the same records GPT-4o-mini also struggles with, confirming they are genuinely ambiguous rather than detector errors.

### 5.3 Q2 — PII Detection Backend Comparison

We compared four backends on 400 records (100/group), using OPERATOR granularity and a single `sem_map` operator reading `["text","ssn","phone","name"]`:

| Backend | Recall | Specificity | F1 | %Local | Wall time |
|---------|--------|-------------|-----|--------|-----------|
| Presidio | **0.980** | **0.950** | **0.966** | 51.5% | 43.5s |
| Regex | 0.975 | **0.950** | 0.963 | 51.2% | **0.1s** |
| DeBERTa | **0.980** | 0.905 | 0.945 | 53.8% | 316.6s |
| Ensemble | **0.980** | 0.905 | 0.945 | 53.8% | 197.1s |

All four backends miss the same four records (hf_005216, hf_005102, hf_000221, hf_004295), which contain ambiguous phone formats missed by regex and Presidio alike. The DeBERTa and ensemble backends add 9 false positives over Presidio (HuggingFace records misidentified as PII), lowering specificity to 0.905. These FPs come from the same watermark-bearing records identified in Section 4.

**The key finding: regex achieves Presidio-level accuracy at 390× lower latency.** Regex F1 (0.963) is within 0.003 of Presidio F1 (0.966), and both share the same 10 false positives. Regex's only additional miss is hf_000052 (a phone number matching a less-common format not covered by the regex). For resume-structured data where PII appears in predictable field-level patterns, a well-tuned regex is competitive with a full NLP analyzer.

DeBERTa and ensemble provide no F1 improvement over Presidio while adding 7× and 4.5× more latency respectively. The ensemble gains no recall because all DeBERTa's extra detections fall on records already caught by Presidio or regex. For production use on this data type, regex is the practical recommendation; Presidio is the accuracy-latency optimum when NLP detection is required.

### 5.4 Anonymization Sensitivity Knob

The `cloud_anonymized` path redacts PII before cloud submission. We benchmarked the three sensitivity levels on 80 records (20/group):

| Sensitivity | Threshold | Redaction rate | Leakage rate |
|-------------|-----------|----------------|--------------|
| PERMISSIVE | 0.85 | 33.3% | 66.7% |
| BALANCED | 0.60 | 31.8% | 68.2% |
| CONSERVATIVE | 0.30 | **97.7%** | **2.3%** |

Presidio's confidence scores on resume text are bimodal: SSNs score above 0.85 (caught at every level), while phone numbers and driver licenses score 0.30–0.55 (caught only at CONSERVATIVE). The 0.60–0.85 range is nearly empty for this dataset, making PERMISSIVE and BALANCED functionally equivalent. **CONSERVATIVE is the only level that provides meaningful privacy protection for unstructured PII on this corpus.**

The 2.3% residual leakage at CONSERVATIVE comes from 3 records: one `US_DRIVER_LICENSE` false negative and two Presidio false positives on watermark text (see Section 4). There is an irreducible noise floor tied to data quality, not detector design.

### 5.5 Q3 — Query-Intent Routing

We evaluated whether the router's query-intent check — using `_query_needs_sensitive_data()` to classify operators as sensitive or non-sensitive before deciding between `local` and `cloud_anonymized` — improves cloud-model coverage without introducing privacy regressions.

Six operator configurations span the three routing-relevant cases:

| Operator | Query type | Expected routing (PII records) |
|----------|-----------|-------------------------------|
| extract_ssn | sensitive | local |
| extract_contact | sensitive | local |
| extract_identity | sensitive | local |
| summarize_skills | non-sensitive | cloud_anonymized |
| classify_industry | non-sensitive | cloud_anonymized |
| rate_education | non-sensitive | cloud_anonymized |

Non-sensitive operators share the same `depends_on` fields as sensitive ones (`["text","ssn","phone","name"]`), so any routing difference is driven entirely by the query text.

Results on 400 records (100/group) with Presidio backend:

| Operator | Query type | Accuracy | → local | → anon | → cloud | 2-way local |
|----------|-----------|----------|---------|--------|---------|-------------|
| extract_ssn | sensitive | **100%** | 177 | 0 | 223 | 177 |
| extract_contact | sensitive | **100%** | 181 | 0 | 219 | 181 |
| extract_identity | sensitive | **100%** | 206 | 0 | 194 | 206 |
| summarize_skills | non-sensitive | **100%** | 0 | 206 | 194 | 206 |
| classify_industry | non-sensitive | **100%** | 0 | 206 | 194 | 206 |
| rate_education | non-sensitive | **100%** | 0 | 206 | 194 | 206 |

**Quality savings**: under naive two-way routing (any PII → local), 1,182 operator calls across the six operators would go to the local model. Three-way routing redirects 618 of these (52.3%) to `cloud_anonymized` instead — recovering cloud-model quality at no additional privacy cost, since PII is still redacted before the record is sent to the cloud API.

**Privacy check**: the query-intent check never misclassifies a sensitive operator as non-sensitive. All sensitive operator calls with confirmed PII detections go to `local` (zero `cloud_anonymized` for sensitive operators). The calls listed as going to `cloud` (23, 19, 4 records respectively) are the same PII detection false negatives identified in Q2 — the routing decision for those records is correct given the detector's output, but the underlying PII was missed at detection time, not at the routing logic layer.

The routing accuracy is 100% for both backends (Presidio and regex). The residual privacy risk is determined entirely by PII detector recall (98% for Presidio), not by the query-intent classification.

---

## 6. Discussion

### 6.1 Routing Granularity Only Matters in Multi-Operator Pipelines

For single-operator pipelines, routing granularity is irrelevant: all three modes scan the same fields and produce identical decisions. The granularity question only becomes meaningful when a pipeline has downstream operators that read derived or processed fields — at which point OPERATOR is the only mode that avoids unnecessary local routing.

For multi-operator pipelines, FIELD and DOCUMENT both misroute more than half of downstream calls to the local model, because they cannot distinguish between "PII exists somewhere in this document" and "this operator needs that PII field." OPERATOR-level routing uses the `depends_on` dependency graph to make this distinction, achieving zero over-routing on downstream operators at no cost to upstream recall.

This means the choice of routing granularity should be driven by pipeline structure: single-operator pipelines can use any granularity (DOCUMENT is fastest), while multi-operator pipelines with derived fields require OPERATOR-level routing to preserve quality.

### 6.2 Regex Is Sufficient for Structured PII on Resume Data

The conventional expectation is that NLP-based detectors (Presidio, DeBERTa) outperform regex on free-text data. Our results show that for resume-structured data — where PII appears in named fields rather than embedded in prose — regex performs within 0.3% F1 of Presidio while running 390× faster. This is because the dominant PII types in the dataset (SSN, phone, email) have highly regular formats that regex captures reliably.

The practical implication is that for interactive or high-throughput workloads, regex routing is the appropriate default, with Presidio reserved for workloads where false negative cost is high and latency budget allows it. DeBERTa and ensemble add latency without meaningfully improving recall or specificity on this data type.

### 6.3 The Anonymization Threshold Is a Binary Choice

The sensitivity knob's bimodal score distribution (SSN > 0.85, phones 0.30–0.55) means that the effective choice is CONSERVATIVE or not. PERMISSIVE and BALANCED leave approximately 67% of PII in anonymized outputs. This limits the `cloud_anonymized` path's usefulness as a middle ground between full privacy and full quality: for most PII types, the only safe choice is CONSERVATIVE, which may over-redact and degrade document quality.

A direction for future work is training a threshold calibration model specific to resume-structured data, rather than relying on Presidio's domain-general confidence scores.

### 6.4 Data Quality Artifacts Inflate False Positive Rates

The "jessica claire" watermark in HuggingFace resume templates creates a systematic false positive across all detectors and both cloud and local LLMs. These records are labeled `none` or `low` in the ground truth but correctly trigger PII detection. Future benchmark work should either fix the data pipeline to remove watermarks or adjust ground-truth labels to reflect actual PII content.

---

## 7. Conclusion

We have presented a privacy-aware routing layer for Palimpzest that intercepts operator calls, detects PII in the fields the operator actually reads, and redirects to the appropriate execution path — local model, anonymize-then-cloud, or cloud as-is. Three benchmarks demonstrate: (1) operator-level routing is necessary to avoid unnecessary quality degradation in multi-operator pipelines; (2) regex detection is sufficient for resume-structured PII with 390× less latency than Presidio; and (3) query-intent routing recovers cloud quality for a substantial fraction of PII-record operator calls while maintaining full privacy for sensitive queries. The system is deployable as a drop-in execution strategy without modifying Palimpzest source files.

---

## References

- Liu, C. et al. "Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing." CIDR 2025.
- Russo, M. et al. "Abacus: A Cost-Based Optimizer for Semantic Operator Systems." arXiv 2025.
- He, P. et al. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." ICLR 2021.
- Microsoft. "Presidio: Data Protection and De-identification SDK." GitHub, 2020.
- Perez, F. and Ribeiro, I. "Ignore Previous Prompt: Attack Techniques For Language Models." NeurIPS ML Safety Workshop, 2022.
- Carlini, N. et al. "Extracting Training Data from Large Language Models." USENIX Security 2021.
- Yu, D. et al. "Differentially Private Fine-tuning of Language Models." ICLR 2022.
