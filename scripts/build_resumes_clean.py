"""
Build a unified resumes_clean.jsonl from two data sources:
  1. data/resume_atlas.csv  (HuggingFace dataset, source='huggingface')
  2. data/github-resume-scraper/<CATEGORY>/*.pdf  (source='github')

Extracts structured PII fields via regex, deduplicates by exact text match,
and writes one JSON object per line to data/resumes_clean.jsonl.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / "resume_atlas.csv"
GITHUB_DIR = DATA_DIR / "github-resume-scraper"
OUTPUT_PATH = DATA_DIR / "resumes_clean.jsonl"

# ---------------------------------------------------------------------------
# Regex helpers for PII extraction
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"           # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"         # optional area code
    r"\d{3,4}[\s\-.]?\d{3,4}"            # main number
)
LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s,;\"'<>]+", re.IGNORECASE)

# Address: look for lines with city/state/zip patterns
ADDRESS_RE = re.compile(
    r"(?:"
    r"[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}"  # City, ST 12345
    r"|"
    r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct)"
    r")",
    re.IGNORECASE,
)

DEGREE_KEYWORDS = re.compile(
    r"\b(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Ph\.?D\.?|Doctor|Bachelor|Master|Associate|Diploma"
    r"|University|College|Institute|School of|Academy)\b",
    re.IGNORECASE,
)

EXPERIENCE_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s*\d{4}\s*[-–—]\s*(?:Present|Current|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?\s*\d{0,4})"
    r"|\b\d{4}\s*[-–—]\s*(?:Present|Current|\d{4})\b",
    re.IGNORECASE,
)

SKILLS_HEADER_RE = re.compile(r"^\s*(?:skills|technical\s+skills|core\s+competencies|key\s+skills)\s*:?\s*$", re.IGNORECASE | re.MULTILINE)


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def extract_phone(text: str) -> str | None:
    # Find all candidate phone numbers, filter out likely non-phone matches
    for m in PHONE_RE.finditer(text):
        candidate = m.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 7 <= len(digits) <= 15:
            return candidate
    return None


def extract_linkedin(text: str) -> str | None:
    m = LINKEDIN_RE.search(text)
    return m.group(0) if m else None


def extract_address(text: str) -> str | None:
    m = ADDRESS_RE.search(text)
    return m.group(0).strip() if m else None


def extract_name(text: str) -> str | None:
    """Heuristic: first non-empty, non-email, non-phone, short line is likely the name."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are clearly not names
        if EMAIL_RE.search(line) and len(line) > 60:
            continue
        if line.startswith("http"):
            continue
        if len(line) > 60:
            continue
        # Skip lines that are section headers
        if re.match(r"^(objective|summary|experience|education|skills|profile|resume|curriculum vitae|cv)\s*:?\s*$", line, re.IGNORECASE):
            continue
        # Likely a name if it's short and mostly alphabetic
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
        if alpha_ratio > 0.7 and len(line) < 50:
            return line
    return None


def extract_education(text: str) -> list[str] | None:
    lines = text.splitlines()
    results = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and DEGREE_KEYWORDS.search(line_stripped):
            results.append(line_stripped)
    return results if results else None


def extract_experience(text: str) -> list[str] | None:
    lines = text.splitlines()
    results = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and EXPERIENCE_DATE_RE.search(line_stripped):
            results.append(line_stripped)
    return results if results else None


def extract_skills(text: str) -> list[str] | None:
    """Extract skills from lines following a 'Skills' header, or comma-separated skill lists."""
    m = SKILLS_HEADER_RE.search(text)
    if m:
        # Grab lines after the skills header until next section header or blank line gap
        start = m.end()
        remainder = text[start:]
        skills_lines = []
        blank_count = 0
        for line in remainder.splitlines():
            stripped = line.strip()
            if not stripped:
                blank_count += 1
                if blank_count >= 2:
                    break
                continue
            # Stop at next section header
            if re.match(r"^[A-Z][A-Z\s]{2,}:?\s*$", stripped):
                break
            blank_count = 0
            # Split by commas, pipes, or bullet points
            parts = re.split(r"[,|•·▪◦●]\s*", stripped)
            skills_lines.extend(p.strip() for p in parts if p.strip())
        if skills_lines:
            return skills_lines

    return None


def extract_pii(text: str) -> dict:
    """Run all PII extractors on a resume text."""
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "address": extract_address(text),
        "linkedin": extract_linkedin(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "skills": extract_skills(text),
    }


def normalize_text(text: str) -> str:
    """Collapse whitespace for dedup comparison."""
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1: Ingest CSV (HuggingFace)
# ---------------------------------------------------------------------------
def ingest_csv(path: Path) -> list[dict]:
    print(f"[CSV] Reading {path} ...")
    df = pd.read_csv(path, dtype=str)
    print(f"[CSV] Columns detected: {list(df.columns)}")
    print(f"[CSV] Raw rows: {len(df)}")

    # Dynamically find the text column and category column
    cols_lower = {c.lower().strip(): c for c in df.columns}
    text_col = None
    cat_col = None

    for candidate in ("resume_str", "resume_text", "resume", "text", "content", "body"):
        if candidate in cols_lower:
            text_col = cols_lower[candidate]
            break
    for candidate in ("category", "job_category", "label", "class", "occupation"):
        if candidate in cols_lower:
            cat_col = cols_lower[candidate]
            break

    # Fallback: if only 2 columns, assume first is category, second is text
    if text_col is None and len(df.columns) == 2:
        cat_col, text_col = df.columns[0], df.columns[1]
        print(f"[CSV] Fallback: using '{cat_col}' as category, '{text_col}' as text")
    elif text_col is None:
        # Last resort: pick the column with longest average string length
        avg_lens = {c: df[c].dropna().str.len().mean() for c in df.columns}
        text_col = max(avg_lens, key=avg_lens.get)
        remaining = [c for c in df.columns if c != text_col]
        cat_col = remaining[0] if remaining else None
        print(f"[CSV] Heuristic: using '{text_col}' as text (avg len {avg_lens[text_col]:.0f})")

    if text_col:
        print(f"[CSV] Text column: '{text_col}', Category column: '{cat_col}'")
    else:
        print("[CSV] ERROR: Could not identify a text column. Aborting CSV ingest.", file=sys.stderr)
        return []

    records = []
    for idx, row in df.iterrows():
        txt = str(row[text_col]) if pd.notna(row[text_col]) else ""
        txt = txt.strip()
        if not txt:
            continue
        category = str(row[cat_col]).strip().upper() if cat_col and pd.notna(row[cat_col]) else None
        record_id = f"hf_{idx:06d}"
        rec = {
            "record_id": record_id,
            "source": "huggingface",
            "category": category,
            "text": txt,
        }
        records.append(rec)

    print(f"[CSV] Valid records: {len(records)}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Phase 2: Ingest PDFs (GitHub scraper)
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def ingest_pdfs(github_dir: Path) -> list[dict]:
    if not github_dir.is_dir():
        print(f"[PDF] Directory not found: {github_dir}", file=sys.stderr)
        return []

    categories = sorted(
        d.name for d in github_dir.iterdir() if d.is_dir()
    )
    print(f"[PDF] Found {len(categories)} categories: {categories}")

    # Count total PDFs for progress
    all_pdfs = []
    for cat_name in categories:
        cat_dir = github_dir / cat_name
        for pdf_path in sorted(cat_dir.glob("*.pdf")):
            all_pdfs.append((cat_name, pdf_path))
    total = len(all_pdfs)
    print(f"[PDF] Total PDFs to process: {total}")

    records = []
    skipped = 0
    errors = 0

    for i, (cat_name, pdf_path) in enumerate(all_pdfs, 1):
        if i % 100 == 0 or i == total:
            print(f"[PDF] Progress: {i}/{total} ({100*i/total:.0f}%)", flush=True)
        stem = pdf_path.stem
        record_id = f"gh_{cat_name}_{stem}"
        try:
            txt = extract_pdf_text(pdf_path)
            txt = txt.strip()
            if not txt:
                skipped += 1
                continue
        except Exception as e:
            print(f"[PDF] Error reading {pdf_path.name}: {e}", file=sys.stderr)
            errors += 1
            continue

        rec = {
            "record_id": record_id,
            "source": "github",
            "category": cat_name,
            "text": txt,
        }
        records.append(rec)

    print(f"[PDF] Valid records: {len(records)}, skipped (empty): {skipped}, errors: {errors}")
    return records


# ---------------------------------------------------------------------------
# Phase 3 & 4: Extract PII, deduplicate, write JSONL
# ---------------------------------------------------------------------------
def main():
    all_records: list[dict] = []

    # Ingest both sources
    csv_records = ingest_csv(CSV_PATH)
    all_records.extend(csv_records)

    pdf_records = ingest_pdfs(GITHUB_DIR)
    all_records.extend(pdf_records)

    print(f"\n[MERGE] Total records before dedup: {len(all_records)}", flush=True)

    # Deduplicate by exact text (normalized whitespace)
    seen_hashes: set[str] = set()
    unique_records: list[dict] = []
    dupes = 0
    for rec in all_records:
        h = text_hash(rec["text"])
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)
        unique_records.append(rec)

    print(f"[MERGE] Duplicates removed: {dupes}")
    print(f"[MERGE] Unique records: {len(unique_records)}")

    # Extract PII fields
    print("[PII] Extracting structured fields via regex ...", flush=True)
    for rec in unique_records:
        pii = extract_pii(rec["text"])
        rec.update(pii)

    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in unique_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Wrote {len(unique_records)} records to {OUTPUT_PATH}")

    # Print sample record
    if unique_records:
        sample = unique_records[0]
        sample_display = {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in sample.items()}
        print(f"\n[SAMPLE] {json.dumps(sample_display, indent=2, ensure_ascii=False)}")

    # Summary stats
    sources = {}
    for rec in unique_records:
        sources[rec["source"]] = sources.get(rec["source"], 0) + 1
    print(f"\n[STATS] Records by source: {sources}")

    pii_fields = ["name", "email", "phone", "address", "linkedin", "education", "experience", "skills"]
    for field in pii_fields:
        count = sum(1 for rec in unique_records if rec.get(field))
        print(f"[STATS] {field}: {count}/{len(unique_records)} ({100*count/max(len(unique_records),1):.1f}%)")


if __name__ == "__main__":
    main()
