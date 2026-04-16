# Project Plan: Privacy-Aware AI Routing

## Core Idea

- Check if a doc has sensitive data; call local AI (Llama) if so

---

## Other Ideas

- If a doc has sensitive data but we don't need those columns, filter them away before calling AI
- Check if we need sensitive data for the query. If not, have local AI anonymize the data and send it to cloud

---

## Suggested Project Scope

Focus on **operator-level routing with a sensitivity propagation model**. Concretely:

1. Add a PII classifier that runs on extracted fields (after initial schema extraction)
2. Model sensitivity as a property that propagates through PZ operators
3. Implement routing: operators with sensitive inputs → local, others → cloud
4. Benchmark quality loss across a few tasks (extraction accuracy, QA, summarization)

**Research question:** At what granularity of routing do you get the best quality-privacy tradeoff? Document-level vs field-level vs operator-level routing gives you three concrete conditions to compare.

---

## Tasks

- Gather datasets
- Build PII classifier
- Set up infrastructure for local AI

---

## Recommended Stack

Use **Presidio** as the framework (provides routing hooks and anonymization pipeline out of the box), backed by `ab-ai/pii_model` or DeBERTa-v3 as the NER engine. This also allows swapping classifiers in/out as an experimental variable, adding an interesting dimension to the paper.

### Datasets

- https://github.com/Sbhawal/resumeScraper
- https://huggingface.co/datasets/datasetmaster/resumes

---

## Key Design Decisions

### 1. Sensitivity Propagation Rule

If an operator takes a sensitive field as input but outputs something non-sensitive (e.g., "is patient over 18?" derived from a birthdate), does the output stay tagged SENSITIVE?

There's no obviously right answer — it's a design choice that defines the entire privacy model.

### 2. Routing Logic

| Condition | Action |
| --- | --- |
| Input has sensitive fields AND query needs them | → Local (Llama), full data |
| Input has sensitive fields AND query doesn't need them | → Anonymize with Presidio, send to cloud |
| Input has no sensitive fields | → Cloud as-is |

At any AI entry point, ask if the prompt is focused on sensitive data. If yes → local Llama. If no → use the classifier algorithm.

### 3. Quality Benchmark Definition

Must be agreed upon before the benchmark harness is built. Options:

- **Extraction accuracy** — straightforward
- **QA and summarization quality** — fuzzy; options include LLM judge, human evals, or ROUGE scores

Agreed metrics: accuracy, precision, recall, F1-score.

Also measure: how much worse is Llama vs. cloud AI in our specific use case.

- Look into benchmarking approaches for databases
- Compare: real answer vs. Llama answer vs. cloud answer

### 4. Failure Mode Policy

When the PII detector is uncertain, does it default to SENSITIVE (conservative, more local routing, lower quality) or SAFE (aggressive, more cloud routing, privacy risk)?

This is a values question the whole team needs to own together.

- Measure how many uncertain cases actually occur
- Start conservative (SENSITIVE); if quality results are too poor, flip to SAFE

### 5. Palimpzest Operators in Scope

Existing semantic operators: `sem_filter`, `sem_map`, `sem_flat_map`, `sem_join`, `sem_agg`, `sem_topk`

- **`sem_map`** — core case; field extraction happens here, PII most likely to appear as input/output; propagation question is most interesting → **Operator #1**
- **`sem_filter`** — nearly as important; e.g., "filter patients where age > 65" touches a sensitive field but outputs only a boolean keep/drop; directly illustrates the derived-output propagation question

### 6. Routing Granularity Definitions

Precisely define each level before benchmarking. Ambiguous definitions will make results hard to interpret:

- Does "field-level" mean one API call per field?
- Does "document-level" mean the whole doc goes local if *any* field is sensitive?

---

## Research Questions (Paper Structure)

**Q1 — Routing granularity** *(Person A)*
At what level of routing do you get the best quality-privacy tradeoff — document, field, or operator level?

**Q2 — Best data classifier for routing** *(Person B, Track 1)*
Which PII detection backend produces the best routing decisions — `ab-ai/pii_model`, DeBERTa-v3, or a regex baseline via Presidio?

**Q3 — Does query intent improve routing?** *(Person B Track 2 + Person C)*
If the query doesn't need the sensitive field, can you strip PII and safely send to cloud without quality loss? This tests whether a two-classifier system (data side + query side) beats a data-only system.

---

## Implementation Decisions

### Decision 1 — Failure Mode Threshold

A **10% F1 drop** is a reasonable default:

- If cloud baseline F1 is 0.85 and local Llama scores below 0.75 on a task → flip to SAFE
- Measure cloud-only baseline in weeks 3–4 (Person C's run); use those reference scores before running the full routing system
- Run conservative first, log every flip, report that distribution in the paper
- If 90% of cases are clear-cut and only 10% hit the threshold, that's a useful result in itself

### Decision 2 — Summarization Quality Metric

| Metric | Pros | Cons |
| --- | --- | --- |
| ROUGE | Fast, free, reproducible | Weakly correlated with actual quality; reviewers know it's weak |
| LLM judge (GPT-4 scores 1–5) | Convincing, human-aligned | Costs money; introduces GPT-4 as a dependency |
| Both | Covers all bases | More work for Person C |

**Recommendation: use both, lead with LLM judge in the paper.** ROUGE as a sanity check in the appendix.

LLM judge prompt: *"Rate this summary 1–5 for accuracy and completeness relative to the source document."*

> **Important:** The judge (GPT-4) should see the original document *with* PII when scoring — otherwise you're scoring against an anonymized reference, which conflates anonymization quality with summarization quality. Keep the judge separate from the routing system.
