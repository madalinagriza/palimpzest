# LLM Prompt Strategy Comparison — llama3.2 (3B, Ollama)

## Background

When using LLM-based intent detection, the router must ask the model a yes/no question:
"does this operator's query need the personal data that was just detected?"
The framing of that question turns out to matter significantly.

Two strategies were evaluated on 12 labeled benchmark queries (8 sensitive, 4 non-sensitive):

| Strategy | Question asked |
|---|---|
| **per_entity** | "If [specific PII type] were redacted, would this query produce a wrong or incomplete answer?" |
| **general** | "Does answering this query correctly require access to any personal data?" |

---

## Results on 12-query micro-benchmark (llama3.2, temperature=0)

| Query | Type | Ground truth | per_entity | general |
|---|---|---|---|---|
| Extract the Social Security Number from the resume text. | explicit | sensitive | NO (wrong) | YES (correct) |
| Find the applicant's phone number and email address. | explicit | sensitive | YES (correct) | YES (correct) |
| Identify the full name and personal contact info of the applicant. | explicit | sensitive | NO (wrong) | YES (correct) |
| Find the best way to contact this applicant. | paraphrased | sensitive | NO (wrong) | NO (wrong) |
| Who wrote this resume? What is their background? | paraphrased | sensitive | NO (wrong) | NO (wrong) |
| Find me applicants above age 30 | implicit | sensitive | NO (wrong) | NO (wrong) |
| Is this candidate likely based in the United States? | implicit | sensitive | NO (wrong) | NO (wrong) |
| Does anything about this application suggest it may be fraudulent? | implicit | sensitive | NO (wrong) | NO (wrong) |
| Summarize the applicant's technical skills and work experience. | — | non-sensitive | NO (correct) | NO (correct) |
| Identify the primary industry category for this resume. | — | non-sensitive | NO (correct) | NO (correct) |
| Rate the quality and relevance of the applicant's education. | — | non-sensitive | NO (correct) | NO (correct) |
| Rate the applicant's seniority level based on years of experience. | — | non-sensitive | NO (correct) | NO (correct) |
| **Overall accuracy** | | | **5/12 (42%)** | **7/12 (58%)** |

---

## Key finding: per_entity fails on obvious sensitive queries

The `per_entity` counterfactual frame produces the most surprising failure:
for "Extract the Social Security Number from the resume text," the model answers **no**.

This is a framing problem, not a knowledge problem. The model appears to reason:
"If the SSN were redacted, the query would simply return nothing — not a *wrong* answer."
The counterfactual does not register as a failure in the model's interpretation.

The `general` prompt avoids this trap. Asking "does this query need any personal data?"
is a simpler, more direct question that the 3B model handles correctly for explicit cases.

---

## Where both strategies fail

Both strategies fail equally on **paraphrased and implicit** sensitive queries —
queries where the PII dependency is indirect:

- "Find the best way to contact this applicant." (needs phone/email, but the word "contact" is not a PII term)
- "Is this candidate likely based in the United States?" (location inferred from phone prefix)
- "Find me applicants above age 30" (age computed from date of birth — not stored directly)
- "Does anything about this application suggest it may be fraudulent?" (fraud check needs name/SSN for verification)

For these, a 3B model with no schema context cannot infer the data dependency from
the natural language description alone. Correct routing would require either a larger
model or explicit schema annotations on the operator.

---

## Recommendation

For LLM-based intent detection with small local models:

- **Use `general` over `per_entity`**: it achieves 58% vs 42% accuracy on this benchmark,
  and critically recovers the most obvious sensitive queries that `per_entity` misses.
- **Neither strategy replaces keyword matching for explicit PII queries**: keyword achieves
  100% on all three explicit sensitive operators vs 58% for the best LLM strategy.
- **LLM adds marginal value on paraphrased queries**: both methods fail on 5 of 8 sensitive
  operators, so LLM intent detection is not a reliable substitute for explicit labeling or
  a larger model.
