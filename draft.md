Mid-Term Project Report

Privacy-Aware Operator Routing in Palimpzest

April 2026

1\. Project Overview

This project adds privacy-aware model routing to Palimpzest (PZ), a semantic query processing framework that uses LLMs as first-class query operators. PZ currently sends every operator call to whichever cloud model the optimizer selects (typically GPT-4o), regardless of whether the fields being processed contain PII. Our system inserts a lightweight routing layer that intercepts each operator call before the LLM is invoked, detects PII in the fields that operator actually reads, and redirects to an appropriate execution path. We implement three routing conditions: (1) sensitive data required by the query → local Llama via Ollama; (2) sensitive data present but not required → Presidio anonymizes in place, then cloud; (3) no sensitive data → cloud as-is.

The primary benchmark dataset is a controlled resume corpus (14,566 records) with four PII tiers — no PII, name+email only, natural PII, injected SSN+DOB — giving ground-truth labels for every routing decision. Three research questions drive the evaluation: (Q1) At what routing granularity is the quality-privacy tradeoff best — document, field, or operator level? (Q2) Which PII detection backend (Presidio, DeBERTa-v3, regex) produces the best routing decisions? (Q3) Does query-intent awareness — routing based on whether the operator actually needs the sensitive field — improve quality without sacrificing privacy?

2\. Project Status

2.1 Completed Tasks

* Mapped the full PZ operator graph architecture: how logical plans (Dataset chains) are constructed, how the Cascades-style optimizer converts them to physical plans, and where model selection is baked in at optimization time. Documented every relevant file and line number in OPERATOR\_GRAPH\_NOTES.md.
* Identified the best execution hook point: SequentialSingleThreadExecutionStrategy.\_execute\_plan() line 88, where both the operator state and the full input record are accessible before any LLM call.
* Built explore\_pipeline.py on the Enron dataset to confirm that operator.get\_input\_fields() respects depends\_on (returns only the fields an operator actually reads), and that input/output schemas and generated fields are all introspectable at runtime without modifying PZ source.
* Implemented routing\_stub.py: ModelConfig, two-layer PrivacyRouter (Presidio primary + field-name heuristic/regex fallback), Detection/RouteDecision dataclasses, RoutingStats (aggregates local/cloud counts and entity-type breakdown per run for the benchmark table), module-level Presidio singleton, fixed model swap for real PZ Model instances via the vLLM constructor path, and execute\_with\_routing() wrapper with structured logging.
* Built privacy\_execution\_strategy.py: PrivacyAwareExecutionStrategy subclasses SequentialSingleThreadExecutionStrategy and replaces the bare operator(input\_record) call with execute\_with\_routing() for LLM operators; create\_privacy\_processor(dataset, config, router) is a drop-in factory — no PZ source files modified.
* Built the resume PII dataset pipeline (build\_resumes\_clean.py → reshape\_pii.py → format\_resumes.py): merged 13,389 HuggingFace resumes and 2,483 GitHub PDF resumes into 14,566 deduplicated records; stratified into four PII groups (none: 1,000; low: 1,000; natural: 10,000; high: 2,566 with injected SSN and DOB via Faker); applied six formatting templates (classic, modern, minimal, academic, compact, creative) to vary surface presentation.
* Ran the first end-to-end experiment (demos/resume-pii-demo.py): 20-record stratified sample (5 per PII group) through a sem\_filter operator asking llama3.2 (Ollama, CPU) to identify records containing PII. Collected precision, recall, and latency.

2.2 Open Tasks

* Collect cloud baseline: run the same sem\_filter and sem\_map tasks on the resume dataset with GPT-4o to establish reference F1 scores for each PII group and template. This is the denominator for measuring quality loss from local routing.
* Implement and benchmark the three routing conditions end-to-end using create\_privacy\_processor and the resume dataset: (1) local routing, (2) anonymize-then-cloud, (3) cloud-as-is. Compare across document-level, field-level, and operator-level granularities.
* Evaluate PII detection backends for Q2: compare Presidio, DeBERTa-v3, and the regex-only fallback on the resume dataset's ground-truth PII labels.
* Implement and test query-intent awareness for Q3: add a depends\_on–driven check that skips PII detection for fields the operator does not read, and test whether routing to cloud with anonymization is safe when the query is insensitive to the PII field.
* Write the final report: three-section structure mapping to Q1/Q2/Q3, results tables, and discussion of the quality-privacy tradeoff.

3\. Results to Date

The routing infrastructure is complete and end-to-end runnable. The first experimental result comes from Person C's sem\_filter run on 20 stratified resume records using llama3.2 (3B params, CPU inference):

| PII Group | Records | Kept (detected PII) | Expected |
|-----------|---------|---------------------|----------|
| none | 5 | 0 | 0 |
| low | 5 | 0 | 0 |
| natural | 5 | 1 | 5 |
| high | 5 | 0 | 5 |
| **Total** | **20** | **1** | **10** |

True negative rate: 10/10 (100%). True positive rate: 1/10 (10%). The local model correctly rejected all records with no PII but missed 9 of 10 containing phone numbers, addresses, and injected SSNs. Per-record latency was ~33 seconds on CPU. This motivates using a dedicated detector (Presidio/regex) for routing decisions rather than the LLM itself, and motivates GPU inference for the full benchmark.

After any pipeline run, router.stats.summary() provides a one-line benchmark-ready string (e.g., "total=10 local=4 (40.0%) cloud=6 top\_entities=[('PHONE\_NUMBER', 3), ('US\_SSN', 2)]").

4\. Potential Problems and Mitigations

* Local model quality gap is now empirically confirmed: llama3.2 recall on the sem\_filter PII detection task is 10%. This does not block the routing system (routing decisions are made by Presidio/regex on known labels, not by the LLM), but it means the downstream extraction quality for locally-routed records may degrade. Mitigation: the benchmark will measure this degradation explicitly across PII groups and report it alongside the routing breakdown.
* Presidio false-positive rate on resumes: resume text contains many names and dates that are not PII in context (section headers misidentified as names, year ranges matching phone patterns). Mitigation: tune score\_threshold, and consider disabling the PERSON detector in favor of structural PII types (SSN, phone, email) for the routing decision.
* CPU inference latency: ~33 seconds per record for llama3.2 on CPU makes full-dataset benchmarking impractical. Mitigation: use a machine with GPU access for the benchmark runs, or reduce the sample size to a representative subset.
* Routing condition 2 (anonymize-then-cloud) depends on Presidio's anonymizer preserving enough semantic content for GPT-4o to answer the query. Mitigation: pilot on 10 records before full benchmark.

5\. Updated Timeline

* Weeks 1–7 (complete): Architecture mapping, PZ execution stack documentation, explore pipeline, initial routing stub.
* Week 8 (complete): Presidio integration, Detection/RouteDecision dataclasses, regex/heuristic fallback, resume dataset pipeline, first sem\_filter experiment.
* Week 9 (complete): RoutingStats, Presidio singleton, Model enum swap fix, PrivacyAwareExecutionStrategy + create\_privacy\_processor, project plan finalized across all three research questions.
* Week 10: Cloud baseline collection (GPT-4o on resume sem\_filter + sem\_map tasks); benchmark of three routing conditions × three granularities; PII detector comparison (Q2); query-intent routing pilot (Q3).
* Week 11: Final report writing, results tables, figures, discussion.

The main change from the original timeline is scope expansion from one research question to three, and from Enron emails to a 14,566-record controlled corpus with ground-truth PII labels — a deliberate choice to make results more defensible.

6\. Individual Contributions

Person A (Denis):

* PZ architecture reverse-engineering and documentation (OPERATOR\_GRAPH\_NOTES.md); explore\_pipeline.py; initial routing\_stub.py design.
* Iterative improvements to routing\_stub.py: RoutingStats, Presidio singleton, fixed Model enum swap, local\_api\_base.
* privacy\_execution\_strategy.py: PrivacyAwareExecutionStrategy and create\_privacy\_processor factory (wiring routing into PZ's execution loop without modifying PZ source).
* Will lead Q1 benchmarks (routing granularity comparison) in week 10.

Person B (Onopre):

* Full Presidio integration into PrivacyRouter (two-layer detector: Presidio + heuristic/regex fallback), Detection/RouteDecision dataclasses, detection-aware logging, extended smoke test.
* Will lead Q2 (PII detector comparison: Presidio vs DeBERTa-v3 vs regex) and Q3 track 1 (query-intent routing implementation) in week 10.

Person C (Madalina):

* Resume PII dataset pipeline (build\_resumes\_clean.py → reshape\_pii.py → format\_resumes.py): 14,566 deduplicated records across four PII groups with SSN/DOB injection and six formatting templates. Documented in data/phase1\_report.md.
* First end-to-end experiment (demos/resume-pii-demo.py): sem\_filter on 20 stratified records with llama3.2; results in data/sem\_filter\_report.md.
* Project plan: defined research questions, routing logic, agreed metrics (accuracy, precision, recall, F1, LLM judge), failure-mode threshold (10% F1 drop), and all design decisions.
* Will lead cloud baseline and full benchmark execution in week 10.

7\. Questions for Feedback

* Low local-model recall (10%): Our first result shows llama3.2 fails to detect PII in 9 of 10 records containing phone numbers and SSNs. Should we treat the LLM-based sem\_filter purely as the task being evaluated (not the routing decision maker), and always use Presidio/regex for the routing decision? Or is this finding itself a contribution worth reporting — that naive local-LLM routing is insufficient and a dedicated detector is needed?
* Routing condition 2 scope: Anonymize-then-cloud requires Presidio's anonymizer to produce text that preserves enough semantic content for GPT-4o to answer the query correctly. Is it acceptable to scope this condition to only the high-PII group, where SSN/DOB fields are injected and clearly separable from query-relevant content, rather than applying it across all groups?
* Q3 feasibility: Query-intent routing (routing based on whether the operator's depends\_on annotation includes a PII field) is architecturally simple given our implementation — get\_input\_fields() already respects depends\_on. But defining ground-truth labels for "does this query need this field" requires manual annotation. Is it acceptable to use the depends\_on values set by the pipeline author as a proxy for query intent, without additional annotation?
