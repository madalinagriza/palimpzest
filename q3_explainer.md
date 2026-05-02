# Q3 — What Is Query-Intent Routing?

## The core problem

When a record contains PII (say, a resume with an SSN), the naive approach is:
> "There is sensitive data in this record → send it to the local model."

This is safe, but it's blunt. It ignores *why* the operator is touching the record at all.

---

## The insight

An operator isn't touching every field in a record — it's answering a specific question.

Consider two operators that both receive the same resume (which happens to contain an SSN):

| Operator | Query | Does it need the SSN? |
|---|---|---|
| `extract_ssn` | "Extract the Social Security Number from the resume." | **Yes** — the SSN is the whole point |
| `summarize_skills` | "Summarize the applicant's technical skills and work experience." | **No** — the SSN is just noise; skills are what matter |

Naive routing sends **both** to the local model because PII was detected.

Query-intent routing asks: *does this specific query actually need the sensitive field?*
- If yes → local (data never leaves the machine)
- If no → strip the PII with Presidio first, then send to cloud (`cloud_anonymized`)

---

## What we are weighing against each other

| | Naive two-way | Intent-aware three-way |
|---|---|---|
| **Any PII detected** | → local | → check query intent first |
| **Query needs PII** | → local | → local (same) |
| **Query doesn't need PII** | → local | → cloud (after anonymization) |
| **No PII detected** | → cloud | → cloud (same) |

The only case that changes is the middle one: **PII present but irrelevant to the query**.

---

## What does "quality savings" mean?

The local model (llama3.2 3B) is significantly weaker than the cloud model (GPT-4o-mini).
From Section 3.1 vs 3.2 of the report:

- Local recall on PII-containing records: **10%**
- Cloud recall on the same records: **80–100%**

Every call that gets redirected from local → `cloud_anonymized` is a call that gets cloud-quality results instead of local-quality results, **at no additional privacy cost** — because Presidio has already stripped the PII before the record leaves the machine.

"Quality savings" in the benchmark = number of operator calls that escape local routing and get cloud quality instead.

---

## How intent detection works — two methods

This is what Q3 actually compares. Both methods take the **same input** (the operator's description text and the list of detected entity types) and answer the same question: *does this query need the sensitive field?*

---

### Method 1 — Keyword matching (`--intent keyword`)

`PrivacyRouter._query_needs_sensitive_data()` checks two things:

1. **Keyword scan on the operator description.**
   A fixed set of terms like `"ssn"`, `"social security"`, `"phone number"`, `"full name"`, `"contact info"`, `"identity"`, etc. If any appear in the description → sensitive.

2. **Field-name match.**
   If the detected PII lives in a field whose name appears verbatim in the query description → sensitive.

If neither fires → non-sensitive → `cloud_anonymized`.

**Conservative fallback:** if the description is empty, default to `local`.

**Advantage:** instant, deterministic, no dependencies.
**Weakness:** only fires on exact vocabulary. A query like *"Who wrote this resume?"* would be classified as non-sensitive even though it asks for a person's identity — because the word `"identity"` doesn't appear.

---

### Method 2 — LLM understanding (`--intent llm`)

`PrivacyRouter._query_needs_sensitive_data_llm()` asks the local Llama model (via Ollama) a direct yes/no question for each detected entity type:

> *"A data processing operator is running this query: 'Summarize the applicant's technical skills.'*
> *Does this query require access to a person's Social Security Number to produce a correct answer?*
> *Reply with only the word 'yes' or 'no'."*

The LLM answer is **cached per (query text, entity type) pair** — so across a full 14,566-record run with 6 operators, Ollama is called at most ~24 times (6 operators × ~4 entity types), not 87,000 times.

**Advantage:** can reason about implicit intent — *"Who wrote this?"* plausibly implies name/identity even without the exact keyword.
**Weakness:** non-deterministic across model versions; adds Ollama as a hard dependency; wrong answers are harder to audit.

---

### What we are comparing

| | Keyword | LLM |
|---|---|---|
| Speed | Instant | ~1–2s per unique (query, entity) pair, then cached |
| Requires Ollama | No | Yes |
| Sensitive query detection | Exact vocabulary match | Semantic understanding |
| Risk of false non-sensitive | High for paraphrased queries | Lower |
| Risk of false sensitive | Low (conservative keyword list) | Depends on model calibration |
| Auditable | Yes — fixed keyword list | No — black box |

The benchmark runs both on the same 6 operators and reports where they disagree, whether either causes privacy regressions on sensitive operators, and how many quality savings each method achieves.

---

## What the benchmark measures

Six operators are run against all 14,566 records. Three are sensitive queries, three are not. All six share the same raw PII fields in their `depends_on` lists — so the **only variable is the query text**.

| Operator | Query type | PII fields read | Expected routing (PII records) |
|---|---|---|---|
| `extract_ssn` | sensitive | text, ssn | → local |
| `extract_contact` | sensitive | text, phone, email | → local |
| `extract_identity` | sensitive | text, name, phone | → local |
| `summarize_skills` | non-sensitive | text, ssn, phone, name | → cloud_anonymized |
| `classify_industry` | non-sensitive | text, ssn, phone, name | → cloud_anonymized |
| `rate_education` | non-sensitive | text, ssn, phone, name | → cloud_anonymized |

The benchmark reports:
- **Routing accuracy** — did each call go to the right destination?
- **Quality savings** — how many calls moved from local → cloud_anonymized vs. naive routing?
- **Privacy check** — did any sensitive-query call leak past local routing?

---

## The answer Q3 is trying to give

> Yes, query-intent awareness improves quality without sacrificing privacy —
> but only when the routing system can reliably determine that the query
> doesn't need the sensitive field.

The risk: if the keyword matcher misclassifies a sensitive query as non-sensitive,
PII goes to the cloud (even if anonymized). The benchmark's privacy check section
verifies this doesn't happen on the test operators.
