Mid-Term Project Report

Privacy-Aware Operator Routing in Palimpzest

April 2026

1\. Project Overview

This project implements privacy-aware model routing inside Palimpzest (PZ), a semantic query processing framework that uses LLMs as first-class query operators. The core insight is that not all data processed by a PZ pipeline is equally sensitive: an email's subject line and scheduling details carry very different privacy risks than a sender's name, phone number, or Social Security Number. The current PZ architecture routes every operator call to whichever cloud model the optimizer selects (typically GPT-4o), regardless of whether the fields being processed contain PII. Our system adds a lightweight classification and routing layer that intercepts each operator call before the LLM is invoked, detects PII in the relevant fields, and redirects sensitive operators to a locally-served model (Llama via Ollama) so that private data never leaves the machine.

2\. Project Status

2.1 Completed Tasks

* Mapped the full Palimpzest operator graph architecture, including how logical plans (Dataset chains) are constructed, how the Cascades-style optimizer converts them to physical plans, and where model selection is currently baked in at optimization time.
* Identified the best hook point for privacy routing: the execution loop inside SequentialSingleThreadExecutionStrategy.\_execute\_plan() (line 88), where both the operator state and input record are simultaneously accessible before any LLM call is made.
* Built a working exploration pipeline (explore\_pipeline.py) that loads Enron email testdata, runs a sem\_map to extract subject and sender fields followed by a sem\_filter for scheduling-related content, and prints both the logical and physical plan with full field introspection. This pipeline confirmed that operator.get\_input\_fields(), operator.generated\_fields, and operator.input\_schema.model\_fields are all accessible at execution time. Notably, get\_input\_fields() already respects the depends\_on annotation — it returns only the fields a given operator actually reads, not all fields in the input schema.
* Authored the initial routing stub (routing\_stub.py) containing the ModelConfig dataclass (mapping local/cloud model identifiers), the PrivacyRouter class with a documented route() interface, and the execute\_with\_routing() wrapper that logs routing decisions and calls the operator through PZ's standard \_\_call\_\_ path.
* Documented the planned Presidio integration point inside PrivacyRouter.route(), including the exact API call pattern (AnalyzerEngine.analyze()), install instructions, and the rationale for field-value scanning over field-name heuristics.
* Integrated Microsoft Presidio into PrivacyRouter: replaced the hardcoded 'always cloud' placeholder with a real two-layer detector. The primary layer runs Presidio's AnalyzerEngine on each field value; the fallback layer uses field-name heuristics (a set of 14 sensitive token patterns such as 'email', 'ssn', 'credit\_card') and regex patterns (email addresses, US phone numbers, SSNs, credit card numbers, IP addresses). Detection results are captured in structured Detection and RouteDecision dataclasses.
* Moved Presidio's AnalyzerEngine construction to a module-level singleton (\_get\_shared\_analyzer()). The engine takes ~1–2 seconds to initialize due to spaCy model loading; constructing it once per process rather than once per PrivacyRouter instance eliminates redundant startup cost across pipeline runs.
* Added RoutingStats dataclass to PrivacyRouter (router.stats). Every call to inspect() records whether the operator was routed local or cloud, and which entity types triggered the decision. After processor.execute() completes, router.stats.summary() returns a one-line string with counts and top entity types — the primary data source for the benchmark results table.
* Fixed the model swap for real PZ operators. PZ stores operator.model as a Model instance (not a plain string), so the previous \_set\_operator\_model\_if\_possible() silently no-oped for all LLM operators. The fix constructs a new Model(chosen\_model, api\_base=local\_api\_base) via PZ's vLLM constructor path, which handles locally-served models regardless of whether the model ID appears in PZ's curated metrics registry. ModelConfig now exposes a local\_api\_base field (default: http://localhost:11434) alongside the model identifiers.
* Wired privacy routing into the PZ execution loop via PrivacyAwareExecutionStrategy (privacy\_execution\_strategy.py). This subclasses SequentialSingleThreadExecutionStrategy and overrides \_execute\_plan(), replacing the bare operator(input\_record) call with execute\_with\_routing() for any operator that makes LLM calls. Non-LLM operators (scans, limits) are passed through unchanged. A convenience factory function create\_privacy\_processor(dataset, config, router) wraps QueryProcessorFactory.create\_processor() and swaps in the privacy strategy without modifying any PZ source file.

2.2 Open Tasks

* Run the benchmark suite (owned by Person C): execute the end-to-end pipeline across three routing granularities (document-level, field-level, operator-level) on the Enron dataset, using Person C's PII-injected test data and query labels, and measure extraction accuracy and filter precision against the cloud baseline reference scores.
* Write the final report, including research question framing, methodology description, quantitative results table, and discussion of the quality-privacy tradeoff at different routing granularities.

3\. Results to Date

The project has progressed from a purely architectural phase through detection implementation and into a fully wired end-to-end integration. The system is now runnable: a caller can replace the standard QueryProcessorFactory pipeline with create\_privacy\_processor(dataset, config) and get privacy-routed execution with no PZ source modifications.

The key architectural result — confirmed by explore\_pipeline.py — remains that all information needed for routing is accessible at a single point in the PZ execution stack before any LLM call: operator type, input schema field names and types, depends\_on annotations, generated fields, and the full input record. The implementation exploits this by intercepting at exactly that point in PrivacyAwareExecutionStrategy.\_execute\_plan().

The routing stub correctly identifies PII in real input records: given an Enron email body containing a name, email address, and phone number, the router returns 'local' with structured detection metadata. The fallback path (field-name heuristics and regex) ensures graceful degradation when Presidio is unavailable.

The execute\_with\_routing() wrapper now produces structured logs per operator call and updates router.stats. After a pipeline run, router.stats.summary() gives the routing breakdown (e.g., "total=10 local=4 (40.0%) cloud=6 top\_entities=[('EMAIL\_ADDRESS', 3), ('PHONE\_NUMBER', 2)]") directly usable in the results table. The model swap now works for real PZ operators via the vLLM constructor path.

4\. Potential Problems and Mitigations

* Presidio false-positive rate: Presidio's NLP-based detectors sometimes flag non-PII tokens (e.g., common names in business text). A high false-positive rate would route too many operators locally, degrading output quality. The mitigation is to tune the score\_threshold parameter in AnalyzerEngine.analyze() and evaluate on the Enron dataset before finalizing the threshold.
* Local model quality gap: Llama 3.2 produces lower-quality extractions than GPT-4o on structured fields like sender email addresses. The benchmark across three routing granularities is specifically designed to quantify this tradeoff and will be informed by Person C's cloud baseline reference scores.
* Ollama availability during benchmarks: The model swap now correctly constructs a PZ Model pointing at the local Ollama server (http://localhost:11434). If Ollama is not running when an operator is routed locally, the operator call will fail. The mitigation is to confirm Ollama is serving llama3.2 before each benchmark run.

5\. Updated Timeline

The execution strategy wiring and model enum fix originally targeted for week 9 are now complete (April 16). The remaining work is Person C's benchmark runs and the final report.

* Weeks 1–7 (complete): Architecture mapping, logical/physical plan introspection, explore pipeline, initial routing stub with documented integration points.
* Week 8 (complete): Presidio integration into PrivacyRouter; Detection/RouteDecision dataclasses; regex/heuristic fallback; best-effort model swap stub; extended smoke test.
* Week 9 (complete): Module-level Presidio singleton; RoutingStats; fixed model swap for PZ Model instances; PrivacyAwareExecutionStrategy wired into PZ execution loop; create\_privacy\_processor convenience factory.
* Week 10: Person C runs benchmark across document-level, field-level, and operator-level routing granularities using PII-injected data and query labels; cloud baseline reference scores collected; Presidio score\_threshold tuned on Enron false-positive rate.
* Week 11: Final report writing; figures; results table; discussion of quality-privacy tradeoff.

6\. Individual Contributions

Person A (Denis):

* Traced the full PZ execution stack from Dataset.sem\_map() and sem\_filter() calls through the Cascades optimizer to the physical operator \_\_call\_\_ invocation, and documented every relevant file and line number in OPERATOR\_GRAPH\_NOTES.md.
* Built explore\_pipeline.py: EnronTinyDataset loader, logical plan introspection printer, physical plan field-level introspector, end-to-end execution with output record pretty-printing. Confirmed that get\_input\_fields() respects depends\_on at the physical operator level.
* Designed the initial routing\_stub.py: ModelConfig, PrivacyRouter scaffold with documented Presidio integration skeleton, execute\_with\_routing() wrapper with logging.
* Identified and documented the four candidate hook points (Options A–D in OPERATOR\_GRAPH\_NOTES.md §5) and made the case for Option A as the recommended approach.
* Added RoutingStats dataclass to routing\_stub.py for benchmark data collection; moved Presidio initialization to a module-level singleton to eliminate redundant startup cost; fixed \_set\_operator\_model\_if\_possible() to handle real PZ Model instances via the vLLM constructor path; added local\_api\_base to ModelConfig.
* Built privacy\_execution\_strategy.py: PrivacyAwareExecutionStrategy subclass that wires execute\_with\_routing() into PZ's execution loop, and create\_privacy\_processor() convenience factory for drop-in use.

Person B (Onopre):

* Replaced the PrivacyRouter placeholder with a full two-layer detection implementation: Presidio AnalyzerEngine as the primary detector, plus field-name heuristic and regex fallback covering five entity types (email, phone, SSN, credit card, IP address).
* Added Detection and RouteDecision dataclasses for structured logging of every routing decision, including entity type, detection source, confidence score, and field preview.
* Implemented initial \_set\_operator\_model\_if\_possible() and extended execute\_with\_routing() with full detection-aware logging.
* Extended the smoke test to verify the full detection path with a realistic PII-containing record.

Person C:

* Responsible for PII injection into test data, query labeling, cloud baseline collection (reference scores across all tasks), and routing × data × query-intent benchmarking.
* These tasks consume the infrastructure built by Persons A and B: the benchmark runner will call create\_privacy\_processor() to get routed execution and read router.stats.summary() for routing breakdown metrics.

7\. Questions for Feedback

* Routing granularity benchmark design: We plan to compare document-level routing (route the entire pipeline to local if any PII is detected anywhere in the document), field-level routing (route an operator to local if any of its input fields contains PII), and operator-level routing (route each operator independently based on the fields it actually reads via depends\_on). Is this a fair comparison, or should we add a fourth condition—anonymize-then-cloud—where Presidio anonymizes PII in place before sending the record to GPT-4o?
* Evaluation metric for filter quality: For the sem\_filter step (keep emails discussing scheduling/meetings/travel), precision and recall against a human-labeled ground truth would be ideal, but labeling is expensive. Is it acceptable to use GPT-4o's decisions as the gold standard and measure local-model agreement rate instead?
* Presidio threshold tuning: Given that Enron emails contain many real personal names that are not PII in context (e.g., executive names used in a business capacity), should we disable Presidio's PERSON detector and rely only on structural PII types (email, phone, SSN, credit card) to reduce false-positive routing to local?
