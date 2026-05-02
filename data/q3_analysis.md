# Q3 Results Analysis — Intent-Aware Routing
## Full dataset (14,566 records) · Presidio backend · both intent methods

---

## What was tested

14 operators across two query categories, each applied to all 14,566 records:

- **8 sensitive operators** — queries that need the PII field (ground truth: PII records → `local`)
- **6 non-sensitive operators** — queries that do not need the PII field (ground truth: PII records → `cloud_anonymized`)

Two intent-detection methods were compared:
- **Keyword** — fixed vocabulary scan of the operator description
- **LLM** — `llama3.2` (3B, via Ollama) asked per (query, entity-type) pair whether the query needs that data; answers cached across records

---

## Summary table

| Operator | Query type | Sensitive? | Keyword acc. | LLM acc. | Winner |
|---|---|---|---|---|---|
| `extract_ssn` | explicit | ✓ | **100%** | 29% | Keyword |
| `extract_contact` | explicit | ✓ | **100%** | 25% | Keyword |
| `extract_identity` | explicit | ✓ | **100%** | 30% | Keyword |
| `find_contact` | paraphrased | ✓ | 25% | 25% | Neither |
| `attribute_authorship` | paraphrased | ✓ | 17% | 32% | LLM (partial) |
| `find_age` | implicit | ✓ | 17% | 17% | Neither |
| `infer_location` | implicit | ✓ | 25% | 25% | Neither |
| `fraud_check` | implicit | ✓ | 17% | 17% | Neither |
| `summarize_skills` | — | ✗ | **100%** | **100%** | Tie |
| `classify_industry` | — | ✗ | **100%** | 84% | Keyword |
| `rate_education` | — | ✗ | **100%** | 84% | Keyword |
| `assess_seniority` | — | ✗ | **100%** | **100%** | Tie |
| `score_relevance` | — | ✗ | **100%** | **100%** | Tie |
| `birth_of_career` | — | ✗ | **100%** | **100%** | Tie |

Accuracy = % of routing decisions matching ground truth across all 14,566 records.  
Note: for non-sensitive operators, accuracy counts `cloud_anonymized` as correct (not `local`).

---

## Finding 1 — Keyword method is exact-match only

Keyword correctly handles all three **explicitly labelled** sensitive operators
(`extract_ssn`, `extract_contact`, `extract_identity`): their descriptions
contain terms like `"Social Security Number"`, `"phone number"`, `"full name"` that
appear directly in `_SENSITIVE_QUERY_KEYWORDS`.

It completely fails on the five **paraphrased or implicit** sensitive operators:

| Operator | Description | Why keyword misses |
|---|---|---|
| `find_contact` | "Find the best way to contact this applicant." | "contact" is not a keyword; neither is "reach out" |
| `attribute_authorship` | "Who wrote this resume? What is their background?" | "who" implies name — no keyword fires |
| `find_age` | "Find me applicants above age 30" | age inference requires DOB — no keyword |
| `infer_location` | "Is this candidate likely based in the United States?" | location inference via phone prefix — no keyword |
| `fraud_check` | "Does anything about this application suggest it may be fraudulent?" | fraud verification needs name/SSN — no keyword |

**Privacy consequence:** for these five operators, 75–83% of all PII records
(natural + high groups, n = 12,566) are routed to `cloud_anonymized` instead of
`local` — i.e. Presidio strips the PII before sending to cloud, but the routing
decision itself was wrong. If anonymization ever misses a detection, those records
are exposed.

Keyword achieves **100% accuracy on all six non-sensitive operators** — no
false positives, no unnecessary local routing.

---

## Finding 2 — LLM (llama3.2 3B) fails even on obvious sensitive queries

The most striking result: `llama3.2` routes **zero records to `local`** for
`extract_ssn` — a query that literally says "Extract the Social Security Number."

| Sensitive operator | LLM local count | Expected | Gap |
|---|---|---|---|
| `extract_ssn` | 0 | 10,336 | −10,336 |
| `extract_contact` | 0 | 10,872 | −10,872 |
| `find_contact` | 0 | 10,872 | −10,872 |
| `find_age` | 0 | 12,158 | −12,158 |
| `infer_location` | 0 | 10,872 | −10,872 |
| `fraud_check` | 0 | 12,158 | −12,158 |
| `extract_identity` | 2,306 | 12,509 | −10,203 |
| `attribute_authorship` | 2,306 | 12,158 | −9,852 |

This is a **model-size / calibration failure**: `llama3.2` 3B cannot reliably
answer meta-questions about whether a query needs a given data type. The LLM is
being asked "Does 'Extract the Social Security Number' need a Social Security
Number?" and saying no. A larger or better-calibrated model would likely behave
differently.

The LLM does provide **partial signal** for two operators:
- `extract_identity` — 2,306 records routed to `local` (18% of the PII records
  that should go local). These are likely the records where Presidio detected
  `PERSON`-type entities, and the LLM judged "Identify the full name and personal
  contact info" as needing them.
- `attribute_authorship` — same 2,306 records, same pattern. The identical count
  suggests the LLM cache hit on the same (entity_type) answers for these two operators.

---

## Finding 3 — LLM over-routes two non-sensitive operators

`classify_industry` and `rate_education` are non-sensitive (do not need PII), but
the LLM routes 2,306 records to `local` for each — reducing their quality savings
by 18.4% relative to keyword.

These two operators:
- Share the same `depends_on` fields as `extract_identity`
- Have descriptions starting with action verbs ("Identify...", "Rate...") that the
  LLM may associate with identity-related tasks

The identical count (2,306 for both) strongly suggests the LLM's cached answers
for the detected entity types are being reused — the model answered "yes, this
entity type matters" for a (description, entity_type) pair that these operators
share with a sensitive one.

`assess_seniority`, `score_relevance`, and `birth_of_career` are unaffected
(100% accuracy), suggesting the LLM correctly handles simpler non-sensitive phrasings.

---

## Finding 4 — Quality savings: keyword outperforms LLM for non-sensitive ops

| Operator | Two-way baseline | Keyword savings | LLM savings | LLM shortfall |
|---|---|---|---|---|
| `summarize_skills` | 12,509 | 12,509 (100%) | 12,509 (100%) | 0 |
| `classify_industry` | 12,509 | 12,509 (100%) | 10,203 (82%) | −2,306 |
| `rate_education` | 12,509 | 12,509 (100%) | 10,203 (82%) | −2,306 |
| `assess_seniority` | 12,509 | 12,509 (100%) | 12,509 (100%) | 0 |
| `score_relevance` | 12,509 | 12,509 (100%) | 12,509 (100%) | 0 |
| `birth_of_career` | 12,158 | 12,158 (100%) | 12,158 (100%) | 0 |

Keyword achieves the maximum possible quality savings on every non-sensitive operator.
LLM unnecessarily sends 4,612 calls to `local` across `classify_industry` and `rate_education`.

---

## Finding 5 — Four operators both methods fail identically

`find_contact`, `find_age`, `infer_location`, and `fraud_check` achieve the same
low accuracy under both methods (17–25%). These represent a class of queries where:

1. No PII keyword appears in the description
2. The LLM (3B) does not understand the implicit data dependency

These are the hardest cases for any lightweight intent detector. Correct routing
would require either: (a) a larger LLM with better reasoning, (b) schema-level
annotations in the pipeline definition, or (c) the `depends_on` field explicitly
set to exclude PII fields when the operator doesn't need them.

---

## Overall verdict

| Criterion | Keyword | LLM (llama3.2 3B) |
|---|---|---|
| Explicit sensitive queries | ✓ Perfect | ✗ Fails (sends SSN queries to cloud) |
| Paraphrased sensitive queries | ✗ Fails (100% leakage) | ✗ Fails (partial signal only) |
| Implicit sensitive queries | ✗ Fails | ✗ Fails equally |
| Non-sensitive accuracy | ✓ Perfect | ✗ 84% on 2/6 operators (over-routes) |
| Quality savings | ✓ Maximum | ✗ −18% on 2/6 operators |
| Privacy guarantee on explicit queries | ✓ Solid | ✗ None |
| Speed | Instant | ~1–2s (cached after first call) |
| Auditability | ✓ Readable keyword list | ✗ Black box |

**The keyword method is strictly better than LLM (llama3.2 3B) on this benchmark.**
The LLM provides no additional privacy protection — it fails even where keyword
succeeds — and introduces quality regression on non-sensitive operators.

The result does not rule out LLM intent detection in general. A larger model
(GPT-4o, llama3.1:70b) would likely score better on explicit queries. The finding
is specific to small local models used for cost-free intent detection.

---

## Diagrams needed

The current Fig 1 has text overlay issues (8 operators × 2 bars in one panel, annotations crowd).
Planned improvements:

1. **Fig 1** — switch to horizontal bars; remove per-bar accuracy labels; use color fill intensity instead
2. **Fig 2** — keep privacy leakage, reduce operator labels to short names
3. **Fig 3** — routing distribution is readable; increase figure height
4. **Fig 4** — disagreement 2×2 panel works well; just needs font tuning
5. **Fig 5** — quality savings: simplify to a single grouped bar without annotations overlay
6. **New: Fig 6** — a clean summary matrix table (operators × methods, cells = accuracy %) rendered as a heatmap — the single figure that tells the whole story at a glance
