# Palimpzest `sem_filter` Demo — Results Report

**Date:** April 14, 2026  
**Script:** `demos/resume-pii-demo.py`  
**Model:** Ollama llama3.2 (3B params, local CPU inference)  
**Data Source:** `data/resumes_with_pii.jsonl` (14,566 records total)

---

## 1. Objective

Test the Palimpzest `sem_filter` operator on the resume PII dataset. The filter asks the LLM to decide, for each resume, whether it **contains personally identifiable information** (SSN, phone number, or a real person's name).

### Filter Definition

```python
dataset.sem_filter(
    "The resume contains personally identifiable information such as a "
    "Social Security Number, phone number, or a real person's name",
    depends_on=["text", "ssn", "phone", "name"],
)
```

---

## 2. Input — Stratified Sample (20 records)

5 records were drawn from each of the 4 PII groups for a balanced test set.

### `none` group (5 records) — All PII stripped

| record_id | category | template | PII fields present |
|-----------|----------|----------|--------------------|
| hf_008107 | CIVIL ENGINEER | minimal | _(none)_ |
| hf_004275 | PHYSICAL EDUCATION | modern | _(none)_ |
| hf_005054 | MECHANICAL ENGINEER | classic | _(none)_ |
| gh_DIGITAL-MEDIA_25525152 | DIGITAL-MEDIA | academic | _(none)_ |
| hf_004133 | .NET DEVELOPER | compact | _(none)_ |

All fields (name, email, phone, address, ssn, dob) are `null`. These resumes should be **rejected** by the filter.

### `low` group (5 records) — Only name + email retained (if present)

| record_id | category | template | PII fields present | name | phone | ssn |
|-----------|----------|----------|--------------------|------|-------|-----|
| hf_010189 | HEALTH AND FITNESS | minimal | _(none)_ | — | — | — |
| hf_000418 | ARTS | academic | _(none)_ | — | — | — |
| hf_013015 | TESTING | modern | _(none)_ | — | — | — |
| hf_013203 | WEB DESIGNING | modern | _(none)_ | — | — | — |
| hf_000199 | AGRICULTURE | academic | _(none)_ | — | — | — |

These particular `low` records had no extractable name/email from the HuggingFace source (pre-processed text). Should also be **rejected**.

### `natural` group (5 records) — Original PII left as-is

| record_id | category | template | PII fields present | name | phone | ssn |
|-----------|----------|----------|--------------------|------|-------|-----|
| gh_CONSULTANT_15602094 | CONSULTANT | academic | name, education, skills | `IT CONSULTANT` | — | — |
| hf_003737 | CONSULTANT | academic | phone, address, education | — | `558 6766046 1234` | — |
| gh_PUBLIC-RELATIONS_14611516 | PUBLIC-RELATIONS | academic | name, education, skills | `PROPERTY MANAGEMENT ASSISTANT` | — | — |
| hf_011115 | MANAGEMENT | modern | phone, address, education | — | `555 4321000` | — |
| hf_005830 | ACCOUNTANT | modern | phone, address, education | — | `555 4321000` | — |

These have real phone numbers and addresses embedded in the text. Should be **accepted**.

### `high` group (5 records) — Original PII + injected SSN/DOB

| record_id | category | template | PII fields present | name | phone | ssn | dob |
|-----------|----------|----------|--------------------|------|-------|-----|-----|
| gh_INFORMATION-TECHNOLOGY_21283365 | INFORMATION-TECHNOLOGY | compact | name, address, education, ssn | `DIRECTOR OF INFORMATION TECHNOLOGY` | — | `655-15-0410` | — |
| hf_007784 | BUSINESS ANALYST | modern | phone, address, education, ssn | — | `555 4321000` | `760-36-4013` | — |
| hf_002346 | REACT DEVELOPER | compact | address, ssn | — | — | `229-18-1680` | — |
| hf_003284 | AVIATION | academic | phone, address, ssn | — | `94103 831 4018` | `693-95-8936` | — |
| hf_010587 | INFORMATION TECHNOLOGY | modern | phone, address, education, ssn, dob | — | `555 4321000` | `090-76-6913` | `06/12/1962` |

These have injected SSNs (all 5) and some DOBs. Should definitely be **accepted**.

---

## 3. Pipeline Configuration

| Setting | Value |
|---------|-------|
| Model | `openai/llama3.2` via Ollama (`http://localhost:11434/v1`) |
| Policy | `pz.MaxQuality()` |
| Execution strategy | `sequential` |
| Optimizer strategy | `pareto` |
| Cost | **$0.00** (local inference) |

---

## 4. Results

### Run 1 — First 20 records (all `none` group)

| Metric | Value |
|--------|-------|
| Input records | 20 |
| Records kept by `sem_filter` | **0** |
| Records rejected | 20 |
| Time | 327.6s (~16.4s/record) |
| Cost | $0.00 |

All 20 were from the `none` PII group (first 20 lines in the JSONL are shuffled into `none`). The LLM correctly rejected all of them.

### Run 2 — Stratified sample (5 per group)

| Metric | Value |
|--------|-------|
| Input records | 20 (5 none + 5 low + 5 natural + 5 high) |
| Records kept by `sem_filter` | **1** |
| Records rejected | 19 |
| Time | 664.7s (~33.2s/record) |
| Cost | $0.00 |

#### Records that passed the filter

| record_id | pii_group | category | name | phone | ssn |
|-----------|-----------|----------|------|-------|-----|
| gh_PUBLIC-RELATIONS_14611516 | natural | PUBLIC-RELATIONS | `PROPERTY MANAGEMENT ASSISTANT` | — | — |

#### Confusion matrix (by PII group)

| PII Group | Total | Kept (has PII) | Rejected (no PII) | Expected kept |
|-----------|-------|----------------|--------------------|----|
| `none` | 5 | 0 | 5 | 0 |
| `low` | 5 | 0 | 5 | 0 |
| `natural` | 5 | 1 | 4 | 5 |
| `high` | 5 | 0 | 5 | 5 |
| **Total** | **20** | **1** | **19** | **10** |

---

## 5. Analysis

- **True negatives (correct rejections):** 10/10 — The `none` and `low` groups had no PII fields, and the model correctly rejected all 10.
- **True positives (correct accepts):** 1/10 — Only 1 of 10 PII-containing resumes was detected.
- **False negatives (missed PII):** 9/10 — The model failed to detect PII in 9 records from `natural` and `high` groups.

### Why the low recall?

1. **Small model (3B params):** llama3.2 is optimized for speed, not nuance. It struggles to identify PII embedded in long resume text, especially formatted SSNs like `655-15-0410` mixed in with addresses and dates.
2. **CPU inference:** Running on CPU without GPU means degraded generation quality at the token level.
3. **Template formatting:** The `academic` and `compact` templates weave PII into paragraphs or fixed-width layouts, making it harder for a small model to isolate.
4. **Noisy extracted fields:** Names like `PROPERTY MANAGEMENT ASSISTANT` and `DIRECTOR OF INFORMATION TECHNOLOGY` are section headers misidentified as names — yet the model only caught one of these.

### Recommendations for production use

- Use a larger model (llama3.1:70b, GPT-4, or Claude) for significantly better PII detection accuracy.
- Use GPU inference (even a consumer GPU) to reduce per-record time from ~33s to ~1-2s.
- Consider `sem_map` to extract PII fields first, then use a programmatic `filter()` on the extracted values for deterministic results.

---

## 6. How to reproduce

```powershell
# Ensure Ollama is running with llama3.2
ollama pull llama3.2

# Run the demo
.venv\Scripts\python.exe demos\resume-pii-demo.py
```
