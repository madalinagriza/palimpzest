# Design Notes: LLM-Based Intent Detection with Ollama

## Why we tried LLM intent detection

The keyword-based intent detector works perfectly for operator descriptions that
explicitly name a PII type ("Extract the Social Security Number", "Find phone number").
But keyword matching has no generalization: a description like "Find the best way to
contact this applicant" contains no keyword from the sensitive vocabulary, so the
router treats it as non-sensitive and allows PII to be sent to cloud after anonymization.

The hypothesis was: a language model could understand the *semantic* relationship
between a query and a data type — recognizing that "contact" implies phone/email,
or that "fraud check" implies name/SSN verification — without needing an explicit keyword.

To keep this cost-free and locally runnable, we used **Ollama** with **llama3.2 (3B)**,
a small open model that runs on a laptop. No cloud API calls, no latency budget impact
beyond the first call per (query, entity_type) pair (subsequent calls hit a local cache).

---

## First attempt: per_entity counterfactual prompt

The first prompt strategy framed intent detection as a counterfactual:

> "If [PII type] were completely redacted from the input, would this query produce
> a wrong or incomplete answer? yes or no"

The intuition: if redacting SSNs would break the query, the query needs SSNs.

**This failed immediately.** The model answered "no" even for
"Extract the Social Security Number from the resume text" — one of the most
obviously sensitive queries in the benchmark.

Investigation revealed a framing problem: the model appeared to reason that
"if the SSN were redacted, the query would return nothing — not a *wrong* answer,
just an empty one." The counterfactual did not trigger the expected reasoning.

We also found two implementation bugs that had been masking this problem:
1. `num_predict: 5` — the token budget was too low; the model could be cut off before
   generating "yes" or "no" if it started with any other token.
2. Missing `system` prompt — the Ollama API requires a separate `system` field; inline
   role instructions inside the `prompt` field were being ignored by the model.

Both were fixed, but accuracy remained low (42% on a 12-query micro-benchmark).

---

## Second attempt: general prompt

The second strategy asked a simpler, more direct question:

> "Does answering this query correctly require access to any personal data? yes or no"

This avoids the counterfactual and asks the model to reason about data *necessity*
from first principles. It also removes the need to ask one question per detected
entity type — a single call per unique query text is sufficient.

Results on the same 12-query benchmark:

| Strategy | Accuracy | Explicit sensitive | Paraphrased/implicit sensitive | Non-sensitive |
|---|---|---|---|---|
| per_entity | 42% (5/12) | 1/3 correct | 0/5 correct | 4/4 correct |
| general    | 58% (7/12) | 3/3 correct | 0/5 correct | 4/4 correct |

`general` recovers all three explicitly-phrased sensitive queries (including "Extract
the SSN"). Both strategies fail equally on the five paraphrased and implicit sensitive
queries, confirming this is a model-size limitation rather than a prompt-framing issue.

---

## What this means for the system

**Keyword is still better overall.** For the operators in our benchmark:
- Keyword: 100% on explicit sensitive, 100% on non-sensitive, 0% on paraphrased/implicit
- General LLM: 100% on explicit sensitive, 100% on non-sensitive, 0% on paraphrased/implicit

In aggregate over 14,566 records and 14 operators, keyword outperforms LLM because
it never misroutes an explicit sensitive query (SSN extraction goes local every time),
whereas `per_entity` LLM sends those same records to `cloud_anonymized`.

**The LLM path does not yet provide the generalization benefit we were after.**
The paraphrased and implicit operators — the cases where keyword falls short — are
also the cases where a 3B model cannot infer the data dependency from description
text alone. A larger model (e.g. llama3.1:70b or GPT-4o) would likely score better.

**The `general` prompt should be the default if LLM is used.** It matches keyword
on easy cases and avoids the counterfactual trap. The `per_entity` prompt should
be retired unless tested with a stronger model.

---

## Implementation detail

The router implements both strategies in `privacy/routing_stub.py`:

- `_ask_llm_needs_entity(query_text, entity_type)` — per_entity prompt, one call per
  (query, entity_type) pair, result cached in `_llm_intent_cache`
- `_ask_llm_needs_any_pii(query_text)` — general prompt, one call per unique query text
- `_ollama_yes_no(system, prompt)` — shared HTTP call to `localhost:11434/api/generate`,
  temperature=0, falls back to `True` (assume sensitive) if Ollama is unreachable

Selected via `ModelConfig(intent_method="keyword" | "llm")`.
