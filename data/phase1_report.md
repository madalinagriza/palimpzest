# Resume PII Classifier — Data Pipeline Report

**Last Updated:** April 14, 2026  
**Purpose:** Unified resume dataset for PII classifier training  
**Pipeline:** `build_resumes_clean.py` → `reshape_pii.py` → `format_resumes.py`

---

## 1. Data Sources

| Source | Format | Raw Records | After Dedup | Label |
|--------|--------|-------------|-------------|-------|
| `data/resume_atlas.csv` | CSV (columns: `Category`, `Text`) | 13,389 | 12,085 | `huggingface` |
| `data/github-resume-scraper/` | 2,484 PDFs across 24 occupation dirs | 2,483 (1 empty) | 2,481 | `github` |
| **Total** | | **15,872** | **14,566** | |

- **1,306 duplicates removed** via exact text match (SHA-256 of whitespace-normalized text)
- **0 PDF parse errors**, 1 PDF skipped (empty text extraction)

---

## 2. Output Files

| File | Records | Description |
|------|---------|-------------|
| `data/resumes_clean.jsonl` | 14,566 | Raw merged records with regex-extracted PII |
| `data/resumes_with_pii.jsonl` | 14,566 | Final records: PII groups, templates, formatted text |
| `data/pii_labels.jsonl` | 14,566 | Label-only companion (pii_group, fields_present, template, field_status) |
| `data/documents/*.txt` | 14,566 | Individual formatted resume text files |

---

## 3. PII Group Distribution

Records are shuffled (seed=42) and split into four groups:

| Group | Count | PII Treatment |
|-------|-------|---------------|
| `none` | 1,000 | All PII fields stripped from text |
| `low` | 1,000 | Only name + email retained (if present); rest stripped |
| `natural` | 10,000 | Original PII left as-is from regex extraction |
| `high` | 2,566 | Original PII + injected SSN (100%) and DOB (60%) via Faker |

### PII Fields per Group

| Field | none | low | natural | high |
|-------|------|-----|---------|------|
| name | 0 | 176 | 1,722 | 408 |
| email | 0 | 0 | 13 | 4 |
| phone | 0 | 0 | 8,249 | 2,146 |
| address | 0 | 0 | 9,750 | 2,516 |
| linkedin | 0 | 0 | 7 | 2 |
| education | 0 | 0 | 9,062 | 2,328 |
| experience | 0 | 0 | 267 | 75 |
| skills | 0 | 0 | 1,617 | 385 |
| ssn | 0 | 0 | 0 | **2,566** |
| dob | 0 | 0 | 0 | **1,541** |

---

## 4. Template Distribution

Each record is assigned one of 6 resume formatting templates:

| Template | Count | Style |
|----------|-------|-------|
| `classic` | 2,454 | PII at top, `======` section headers, bullet points |
| `modern` | 2,501 | Name + title top, contact info at bottom after `──────` rule |
| `minimal` | 2,366 | All lowercase, no headers/labels, terse one-liners |
| `academic` | 2,429 | Third-person narrative prose, PII woven into text |
| `compact` | 2,424 | Fixed-width two-column layout with `=` borders |
| `creative` | 2,392 | First-person voice, PII scattered in conversational paragraphs |

---

## 5. Schema — `resumes_with_pii.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `string` | Unique ID: `hf_NNNNNN` or `gh_CATEGORY_NUMERICID` |
| `source` | `string` | `"huggingface"` or `"github"` |
| `category` | `string` | Occupation category (24 categories) |
| `pii_group` | `string` | `"none"`, `"low"`, `"natural"`, or `"high"` |
| `template_used` | `string` | One of: classic, modern, minimal, academic, compact, creative |
| `pii_fields_present` | `list[str]` | Which PII fields exist in this record |
| `field_status` | `dict` | Per-field status: `original`, `injected`, `stripped`, or `absent` |
| `text` | `string` | Formatted resume text (template-applied) |
| `name` | `string\|null` | Extracted or stripped name |
| `email` | `string\|null` | Extracted or stripped email |
| `phone` | `string\|null` | Extracted or stripped phone |
| `address` | `string\|null` | Extracted or stripped address |
| `linkedin` | `string\|null` | Extracted or stripped LinkedIn URL |
| `education` | `list[str]\|null` | Education entries |
| `experience` | `list[str]\|null` | Experience entries |
| `skills` | `list[str]\|null` | Skill entries |
| `ssn` | `string\|null` | Injected SSN (high group only) |
| `dob` | `string\|null` | Injected DOB (high group, ~60%) |

---

## 6. Schema — `pii_labels.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `string` | Matches `resumes_with_pii.jsonl` |
| `pii_group` | `string` | none / low / natural / high |
| `pii_fields_present` | `list[str]` | Which PII fields are present |
| `template_used` | `string` | Template applied |
| `field_status` | `dict` | Per-field: original / injected / stripped / absent |

---

## 7. Categories (24 total)

```
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE,
AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION,
CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS,
HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
```

---

## 8. Quality Notes

- **HuggingFace resumes** are pre-processed (lowercased, punctuation stripped, contact info removed), limiting regex PII extraction. These records are best used for training on *text patterns* rather than exact PII labels.
- **GitHub PDF resumes** are raw and yield much better structured-field extraction (names, emails, proper addresses).
- **Phone/address false positives** exist: year ranges like "20162018" match phone regex; university lines match address regex. These serve as useful negative examples in classifier training.
- **Name extraction** sometimes captures section headers (e.g., "Summary of Qualifications") rather than actual names.

---

## 9. How to Re-Run

```powershell
# Phase 1: Merge raw data → resumes_clean.jsonl
.venv\Scripts\python.exe scripts\build_resumes_clean.py

# Phase 2: Stratify into PII groups → resumes_with_pii.jsonl + pii_labels.jsonl + documents/
.venv\Scripts\python.exe scripts\reshape_pii.py

# Phase 3: Apply resume templates → updates resumes_with_pii.jsonl + documents/
.venv\Scripts\python.exe scripts\format_resumes.py
```

Dependencies: `pandas`, `pypdf` (in `pyproject.toml`), `faker` (pip install)
