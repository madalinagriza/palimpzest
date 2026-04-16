"""
Reshape resumes_clean.jsonl into a PII-stratified dataset for PII classifier training.

Groups:
  - none   (~1,000): All PII stripped from text
  - low    (~1,000): Only name + email kept
  - natural(~10,000): Original PII preserved as-is
  - high   (~2,500): Original PII + injected SSN (100%) and DOB (60%)

Outputs:
  - data/resumes_with_pii.jsonl   (enriched JSONL)
  - data/documents/*.txt          (one .txt per record)
  - data/pii_labels.jsonl         (label metadata)
"""

import json
import os
import random
import re
import shutil
from pathlib import Path

from faker import Faker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_PATH = DATA_DIR / "resumes_clean.jsonl"
OUTPUT_JSONL = DATA_DIR / "resumes_with_pii.jsonl"
LABELS_JSONL = DATA_DIR / "pii_labels.jsonl"
DOCS_DIR = DATA_DIR / "documents"

GROUP_SIZES = {"none": 1000, "low": 1000, "natural": 10000}
# "high" gets the remainder

# ---------------------------------------------------------------------------
# Regex patterns (same as build_resumes_clean.py for consistency)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"
    r"\d{3,4}[\s\-.]?\d{3,4}"
)
LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?", re.IGNORECASE)
ADDRESS_RE = re.compile(
    r"(?:"
    r"[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}"
    r"|"
    r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct)"
    r")",
    re.IGNORECASE,
)
SSN_PATTERN = re.compile(r"\d{3}-\d{2}-\d{4}")
DOB_PATTERN = re.compile(
    r"\b(?:(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2})"
    r"|(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Name detection: first short mostly-alpha line (heuristic from original)
# ---------------------------------------------------------------------------
NAME_LINE_RE = re.compile(
    r"^(objective|summary|experience|education|skills|profile|resume|curriculum vitae|cv)\s*:?\s*$",
    re.IGNORECASE,
)


def _likely_name_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 60:
        return False
    if EMAIL_RE.search(line):
        return False
    if line.startswith("http"):
        return False
    if NAME_LINE_RE.match(line):
        return False
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
    return alpha_ratio > 0.7 and len(line) < 50


# ---------------------------------------------------------------------------
# Stripping helpers
# ---------------------------------------------------------------------------
def strip_all_pii(text: str) -> str:
    """Remove all PII from text: name, email, phone, address, linkedin, URLs."""
    # Remove emails
    text = EMAIL_RE.sub("[EMAIL]", text)
    # Remove linkedin
    text = LINKEDIN_RE.sub("[URL]", text)
    # Remove addresses
    text = ADDRESS_RE.sub("[ADDRESS]", text)
    # Remove phone numbers (only 7-15 digit matches)
    def _phone_repl(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 7 <= len(digits) <= 15:
            return "[PHONE]"
        return m.group(0)
    text = PHONE_RE.sub(_phone_repl, text)
    # Remove name (first likely name line)
    lines = text.splitlines()
    new_lines = []
    name_removed = False
    for line in lines:
        if not name_removed and _likely_name_line(line):
            new_lines.append("[NAME]")
            name_removed = True
        else:
            new_lines.append(line)
    text = "\n".join(new_lines)
    # Remove any SSNs or DOBs that might exist
    text = SSN_PATTERN.sub("[SSN]", text)
    text = DOB_PATTERN.sub("[DOB]", text)
    return text


def strip_except_name_email(text: str) -> str:
    """Strip everything except name and email."""
    # Remove addresses
    text = ADDRESS_RE.sub("[ADDRESS]", text)
    # Remove linkedin
    text = LINKEDIN_RE.sub("[URL]", text)
    # Remove phone numbers
    def _phone_repl(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 7 <= len(digits) <= 15:
            return "[PHONE]"
        return m.group(0)
    text = PHONE_RE.sub(_phone_repl, text)
    # Remove SSNs/DOBs
    text = SSN_PATTERN.sub("[SSN]", text)
    text = DOB_PATTERN.sub("[DOB]", text)
    # Keep name and email intact
    return text


# ---------------------------------------------------------------------------
# Resume templates
# ---------------------------------------------------------------------------
TEMPLATES = {
    "classic": """\
{name_block}
{contact_block}

{separator}
PROFESSIONAL SUMMARY
{separator}
{summary}

{separator}
EXPERIENCE
{separator}
{experience}

{separator}
EDUCATION
{separator}
{education}

{separator}
SKILLS
{separator}
{skills}
{extra_pii_block}""",

    "modern": """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{name_block}
{contact_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ SUMMARY
{summary}

▸ EXPERIENCE
{experience}

▸ EDUCATION
{education}

▸ SKILLS
{skills}
{extra_pii_block}""",

    "minimal": """\
{name_block}
{contact_block}

---

{summary}

Experience:
{experience}

Education:
{education}

Skills:
{skills}
{extra_pii_block}""",

    "academic": """\
CURRICULUM VITAE

{name_block}
{contact_block}

I. RESEARCH INTERESTS / SUMMARY
{summary}

II. PROFESSIONAL EXPERIENCE
{experience}

III. EDUCATION
{education}

IV. TECHNICAL SKILLS
{skills}
{extra_pii_block}""",

    "compact": """\
{name_block} | {contact_inline}

PROFILE: {summary}

EXPERIENCE
{experience}

EDUCATION
{education}

SKILLS: {skills}
{extra_pii_block}""",

    "creative": """\
╔══════════════════════════════════════╗
║  {name_block}
║  {contact_block}
╚══════════════════════════════════════╝

★ About Me
{summary}

★ Where I've Worked
{experience}

★ Education
{education}

★ What I Know
{skills}
{extra_pii_block}""",
}

TEMPLATE_NAMES = list(TEMPLATES.keys())


def _get_section(text: str, header_pattern: str, next_patterns: list[str] | None = None) -> str:
    """Try to extract a section from raw text by header keyword."""
    lines = text.splitlines()
    collecting = False
    result = []
    header_re = re.compile(header_pattern, re.IGNORECASE)
    stop_res = [re.compile(p, re.IGNORECASE) for p in (next_patterns or [])]

    for line in lines:
        if header_re.search(line) and not collecting:
            collecting = True
            continue
        if collecting:
            if any(r.search(line) for r in stop_res):
                break
            result.append(line)
    return "\n".join(result).strip() if result else ""


SECTION_HEADERS = [
    r"^\s*(professional\s+)?summary",
    r"^\s*(work\s+)?experience",
    r"^\s*education",
    r"^\s*(technical\s+)?skills",
    r"^\s*objective",
    r"^\s*certifications?",
    r"^\s*projects?",
]


def _split_into_chunks(text: str) -> dict:
    """Best-effort split of resume text into sections."""
    summary = _get_section(text, r"summary|objective|profile",
                           [r"experience", r"education", r"skills", r"projects?", r"certifications?"])
    experience = _get_section(text, r"experience|employment|work\s+history",
                              [r"education", r"skills", r"projects?", r"certifications?", r"summary"])
    education = _get_section(text, r"education|academic",
                             [r"experience", r"skills", r"projects?", r"certifications?", r"summary"])
    skills = _get_section(text, r"skills|competenc|technologies|tools",
                          [r"experience", r"education", r"projects?", r"certifications?", r"summary"])

    # Fallback: if no sections found, split text roughly into quarters
    if not any([summary, experience, education, skills]):
        lines = [l for l in text.splitlines() if l.strip()]
        n = len(lines)
        q = max(n // 4, 1)
        summary = "\n".join(lines[:q])
        experience = "\n".join(lines[q:2*q])
        education = "\n".join(lines[2*q:3*q])
        skills = "\n".join(lines[3*q:])

    return {
        "summary": summary or "N/A",
        "experience": experience or "N/A",
        "education": education or "N/A",
        "skills": skills or "N/A",
    }


def apply_template(rec: dict, template_name: str, extra_pii: dict | None = None) -> str:
    """Format a record using the given template."""
    template = TEMPLATES[template_name]
    text = rec["text"]
    sections = _split_into_chunks(text)

    name = rec.get("name") or ""
    email = rec.get("email") or ""
    phone = rec.get("phone") or ""
    address = rec.get("address") or ""
    linkedin = rec.get("linkedin") or ""

    # Build contact block
    contact_parts = []
    if email:
        contact_parts.append(email)
    if phone:
        contact_parts.append(phone)
    if address:
        contact_parts.append(address)
    if linkedin:
        contact_parts.append(linkedin)

    contact_block = " | ".join(contact_parts) if contact_parts else ""
    contact_inline = contact_block

    # Extra PII block for high group
    extra_lines = []
    if extra_pii:
        if extra_pii.get("ssn"):
            extra_lines.append(f"\nSSN: {extra_pii['ssn']}")
        if extra_pii.get("dob"):
            extra_lines.append(f"Date of Birth: {extra_pii['dob']}")
    extra_pii_block = "\n".join(extra_lines)

    separator = "=" * 40

    result = template.format(
        name_block=name or "CANDIDATE",
        contact_block=contact_block,
        contact_inline=contact_inline,
        summary=sections["summary"],
        experience=sections["experience"],
        education=sections["education"],
        skills=sections["skills"],
        extra_pii_block=extra_pii_block,
        separator=separator,
    )
    return result


# ---------------------------------------------------------------------------
# PII field tracking
# ---------------------------------------------------------------------------
PII_FIELD_NAMES = ["name", "email", "phone", "address", "linkedin",
                   "education", "experience", "skills", "ssn", "dob"]


def compute_pii_fields_present(rec: dict) -> list[str]:
    """Return list of PII field names that have a non-empty value."""
    present = []
    for f in PII_FIELD_NAMES:
        val = rec.get(f)
        if val:
            present.append(f)
    return present


def compute_field_status(rec: dict, group: str, extra_pii: dict | None = None) -> dict:
    """For each PII field, return 'original', 'injected', 'stripped', or 'absent'."""
    status = {}
    original_fields = ["name", "email", "phone", "address", "linkedin",
                       "education", "experience", "skills"]

    for f in original_fields:
        has_original = bool(rec.get(f"_orig_{f}") if f"_orig_{f}" in rec else rec.get(f))
        if group == "none":
            status[f] = "stripped" if has_original else "absent"
        elif group == "low":
            if f in ("name", "email"):
                status[f] = "original" if rec.get(f) else "absent"
            else:
                status[f] = "stripped" if has_original else "absent"
        else:
            status[f] = "original" if rec.get(f) else "absent"

    # SSN / DOB
    if extra_pii and extra_pii.get("ssn"):
        status["ssn"] = "injected"
    else:
        status["ssn"] = "absent"
    if extra_pii and extra_pii.get("dob"):
        status["dob"] = "injected"
    else:
        status["dob"] = "absent"

    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load records
    print(f"[LOAD] Reading {INPUT_PATH} ...", flush=True)
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    total = len(records)
    print(f"[LOAD] {total} records loaded", flush=True)

    # Shuffle and split into groups
    random.shuffle(records)

    g_none = records[:GROUP_SIZES["none"]]
    g_low = records[GROUP_SIZES["none"]:GROUP_SIZES["none"] + GROUP_SIZES["low"]]
    natural_end = GROUP_SIZES["none"] + GROUP_SIZES["low"] + GROUP_SIZES["natural"]
    g_natural = records[GROUP_SIZES["none"] + GROUP_SIZES["low"]:natural_end]
    g_high = records[natural_end:]

    print(f"[SPLIT] none={len(g_none)}, low={len(g_low)}, natural={len(g_natural)}, high={len(g_high)}", flush=True)

    # Prepare output dirs
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    output_records = []
    label_records = []

    # -----------------------------------------------------------------------
    # Group 1: ZERO PII
    # -----------------------------------------------------------------------
    print("[GROUP none] Stripping all PII ...", flush=True)
    for rec in g_none:
        template_name = random.choice(TEMPLATE_NAMES)
        # Store originals before stripping
        orig_text = rec["text"]

        # Strip PII from text
        stripped_text = strip_all_pii(orig_text)
        rec["text"] = stripped_text

        # Zero out extracted PII fields
        stripped_rec = dict(rec)
        for f in ["name", "email", "phone", "address", "linkedin"]:
            stripped_rec[f"_orig_{f}"] = stripped_rec.get(f)
            stripped_rec[f] = None

        # Apply template
        formatted = apply_template(stripped_rec, template_name)

        field_status = compute_field_status(stripped_rec, "none")
        pii_present = [f for f, s in field_status.items() if s in ("original", "injected")]

        out_rec = {
            "record_id": rec["record_id"],
            "source": rec["source"],
            "category": rec.get("category"),
            "pii_group": "none",
            "template_used": template_name,
            "pii_fields_present": pii_present,
            "field_status": field_status,
            "text": formatted,
            "name": None,
            "email": None,
            "phone": None,
            "address": None,
            "linkedin": None,
            "education": stripped_rec.get("education"),
            "experience": stripped_rec.get("experience"),
            "skills": stripped_rec.get("skills"),
            "ssn": None,
            "dob": None,
        }
        output_records.append(out_rec)
        label_records.append({
            "record_id": rec["record_id"],
            "pii_group": "none",
            "pii_fields_present": pii_present,
            "template_used": template_name,
            "field_status": field_status,
        })

    # -----------------------------------------------------------------------
    # Group 2: LOW PII (keep name + email only)
    # -----------------------------------------------------------------------
    print("[GROUP low] Stripping to name + email only ...", flush=True)
    for rec in g_low:
        template_name = random.choice(TEMPLATE_NAMES)
        orig_text = rec["text"]

        stripped_text = strip_except_name_email(orig_text)
        rec["text"] = stripped_text

        stripped_rec = dict(rec)
        for f in ["phone", "address", "linkedin"]:
            stripped_rec[f"_orig_{f}"] = stripped_rec.get(f)
            stripped_rec[f] = None

        formatted = apply_template(stripped_rec, template_name)

        field_status = compute_field_status(stripped_rec, "low")
        pii_present = [f for f, s in field_status.items() if s in ("original", "injected")]

        out_rec = {
            "record_id": rec["record_id"],
            "source": rec["source"],
            "category": rec.get("category"),
            "pii_group": "low",
            "template_used": template_name,
            "pii_fields_present": pii_present,
            "field_status": field_status,
            "text": formatted,
            "name": rec.get("name"),
            "email": rec.get("email"),
            "phone": None,
            "address": None,
            "linkedin": None,
            "education": rec.get("education"),
            "experience": rec.get("experience"),
            "skills": rec.get("skills"),
            "ssn": None,
            "dob": None,
        }
        output_records.append(out_rec)
        label_records.append({
            "record_id": rec["record_id"],
            "pii_group": "low",
            "pii_fields_present": pii_present,
            "template_used": template_name,
            "field_status": field_status,
        })

    # -----------------------------------------------------------------------
    # Group 3: NATURAL PII (as-is)
    # -----------------------------------------------------------------------
    print("[GROUP natural] Keeping original PII ...", flush=True)
    for rec in g_natural:
        template_name = random.choice(TEMPLATE_NAMES)
        formatted = apply_template(rec, template_name)

        field_status = compute_field_status(rec, "natural")
        pii_present = [f for f, s in field_status.items() if s in ("original", "injected")]

        out_rec = {
            "record_id": rec["record_id"],
            "source": rec["source"],
            "category": rec.get("category"),
            "pii_group": "natural",
            "template_used": template_name,
            "pii_fields_present": pii_present,
            "field_status": field_status,
            "text": formatted,
            "name": rec.get("name"),
            "email": rec.get("email"),
            "phone": rec.get("phone"),
            "address": rec.get("address"),
            "linkedin": rec.get("linkedin"),
            "education": rec.get("education"),
            "experience": rec.get("experience"),
            "skills": rec.get("skills"),
            "ssn": None,
            "dob": None,
        }
        output_records.append(out_rec)
        label_records.append({
            "record_id": rec["record_id"],
            "pii_group": "natural",
            "pii_fields_present": pii_present,
            "template_used": template_name,
            "field_status": field_status,
        })

    # -----------------------------------------------------------------------
    # Group 4: HIGH PII (original + injected SSN + DOB)
    # -----------------------------------------------------------------------
    print("[GROUP high] Injecting SSN + DOB ...", flush=True)
    for rec in g_high:
        template_name = random.choice(TEMPLATE_NAMES)

        extra_pii = {"ssn": fake.ssn()}
        if random.random() < 0.6:
            extra_pii["dob"] = fake.date_of_birth(minimum_age=22, maximum_age=65).strftime("%m/%d/%Y")
        else:
            extra_pii["dob"] = None

        formatted = apply_template(rec, template_name, extra_pii=extra_pii)

        field_status = compute_field_status(rec, "high", extra_pii=extra_pii)
        pii_present = [f for f, s in field_status.items() if s in ("original", "injected")]

        out_rec = {
            "record_id": rec["record_id"],
            "source": rec["source"],
            "category": rec.get("category"),
            "pii_group": "high",
            "template_used": template_name,
            "pii_fields_present": pii_present,
            "field_status": field_status,
            "text": formatted,
            "name": rec.get("name"),
            "email": rec.get("email"),
            "phone": rec.get("phone"),
            "address": rec.get("address"),
            "linkedin": rec.get("linkedin"),
            "education": rec.get("education"),
            "experience": rec.get("experience"),
            "skills": rec.get("skills"),
            "ssn": extra_pii["ssn"],
            "dob": extra_pii.get("dob"),
        }
        output_records.append(out_rec)
        label_records.append({
            "record_id": rec["record_id"],
            "pii_group": "high",
            "pii_fields_present": pii_present,
            "template_used": template_name,
            "field_status": field_status,
        })

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    print(f"\n[WRITE] Writing {len(output_records)} records ...", flush=True)

    # JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[WRITE] {OUTPUT_JSONL}", flush=True)

    # Labels
    with open(LABELS_JSONL, "w", encoding="utf-8") as f:
        for lab in label_records:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    print(f"[WRITE] {LABELS_JSONL}", flush=True)

    # TXT documents
    for rec in output_records:
        txt_path = DOCS_DIR / f"{rec['record_id']}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(rec["text"])
    print(f"[WRITE] {len(output_records)} .txt files → {DOCS_DIR}", flush=True)

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("FINAL DISTRIBUTION STATS", flush=True)
    print("=" * 60, flush=True)

    # Group counts
    group_counts = {}
    for rec in output_records:
        g = rec["pii_group"]
        group_counts[g] = group_counts.get(g, 0) + 1
    print(f"\n{'Group':<12} {'Count':>6} {'%':>7}")
    print("-" * 27)
    for g in ["none", "low", "natural", "high"]:
        c = group_counts.get(g, 0)
        print(f"{g:<12} {c:>6} {100*c/len(output_records):>6.1f}%")
    print(f"{'TOTAL':<12} {len(output_records):>6}")

    # Template distribution
    template_counts = {}
    for rec in output_records:
        t = rec["template_used"]
        template_counts[t] = template_counts.get(t, 0) + 1
    print(f"\n{'Template':<12} {'Count':>6} {'%':>7}")
    print("-" * 27)
    for t in TEMPLATE_NAMES:
        c = template_counts.get(t, 0)
        print(f"{t:<12} {c:>6} {100*c/len(output_records):>6.1f}%")

    # PII field presence across all records
    print(f"\n{'PII Field':<12} {'Present':>8} {'%':>7}")
    print("-" * 29)
    for f in PII_FIELD_NAMES:
        c = sum(1 for rec in output_records if rec.get(f))
        print(f"{f:<12} {c:>8} {100*c/len(output_records):>6.1f}%")

    # Field status distribution
    print(f"\n{'Field':<12} {'original':>9} {'injected':>9} {'stripped':>9} {'absent':>9}")
    print("-" * 52)
    for f in PII_FIELD_NAMES:
        statuses = {"original": 0, "injected": 0, "stripped": 0, "absent": 0}
        for rec in output_records:
            s = rec.get("field_status", {}).get(f, "absent")
            statuses[s] = statuses.get(s, 0) + 1
        print(f"{f:<12} {statuses['original']:>9} {statuses['injected']:>9} {statuses['stripped']:>9} {statuses['absent']:>9}")

    # Per-group PII field summary
    for g in ["none", "low", "natural", "high"]:
        g_recs = [r for r in output_records if r["pii_group"] == g]
        n = len(g_recs)
        print(f"\n[{g.upper()}] ({n} records)")
        for f in PII_FIELD_NAMES:
            c = sum(1 for r in g_recs if r.get(f))
            print(f"  {f:<12} {c:>6} ({100*c/max(n,1):.1f}%)")

    print(f"\n[DONE] All outputs written.", flush=True)


if __name__ == "__main__":
    main()
