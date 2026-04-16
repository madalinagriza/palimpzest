"""
Generate formatted resume .txt files from resumes_with_pii.jsonl.

Reads original raw text from resumes_clean.jsonl for section parsing,
applies the template_used assignment, and writes formatted .txt files.

Outputs:
  - data/documents/{record_id}.txt  (one per record)
  - data/resumes_with_pii.jsonl     (updated text field)
"""

import json
import re
import shutil
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_PII = DATA_DIR / "resumes_with_pii.jsonl"
INPUT_CLEAN = DATA_DIR / "resumes_clean.jsonl"
DOCS_DIR = DATA_DIR / "documents"

# ---------------------------------------------------------------------------
# PII stripping regexes (same as reshape_pii.py)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"
    r"\d{3,4}[\s\-.]?\d{3,4}"
)
LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?", re.I)
ADDRESS_RE = re.compile(
    r"(?:[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}"
    r"|\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct))",
    re.I,
)
SSN_RE = re.compile(r"\d{3}-\d{2}-\d{4}")
DOB_RE = re.compile(
    r"\b(?:(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2})"
    r"|(?:(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)
SECTION_HEADER_RE = re.compile(
    r"^(objective|summary|experience|education|skills|profile|resume|curriculum vitae|cv)\s*:?\s*$",
    re.I,
)


def _likely_name_line(line):
    line = line.strip()
    if not line or len(line) > 60:
        return False
    if EMAIL_RE.search(line) or line.startswith("http") or SECTION_HEADER_RE.match(line):
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
    return alpha > 0.7 and len(line) < 50


def strip_all_pii(text):
    """Remove all PII: name, email, phone, address, linkedin."""
    text = EMAIL_RE.sub("", text)
    text = LINKEDIN_RE.sub("", text)
    text = ADDRESS_RE.sub("", text)
    text = PHONE_RE.sub(
        lambda m: "" if 7 <= len(re.sub(r"\D", "", m.group(0))) <= 15 else m.group(0),
        text,
    )
    lines, out, done = text.splitlines(), [], False
    for ln in lines:
        if not done and _likely_name_line(ln):
            done = True
            continue
        out.append(ln)
    text = "\n".join(out)
    text = SSN_RE.sub("", text)
    text = DOB_RE.sub("", text)
    return text


def strip_except_name_email(text):
    """Remove everything except name and email."""
    text = ADDRESS_RE.sub("", text)
    text = LINKEDIN_RE.sub("", text)
    text = PHONE_RE.sub(
        lambda m: "" if 7 <= len(re.sub(r"\D", "", m.group(0))) <= 15 else m.group(0),
        text,
    )
    text = SSN_RE.sub("", text)
    text = DOB_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------
HEADER_PATTERNS = [
    ("summary", re.compile(
        r"^\s*(?:professional\s+)?(?:summary|profile|objective)\s*:?\s*$", re.I)),
    ("experience", re.compile(
        r"^\s*(?:work\s+|employment\s+|professional\s+)?(?:experience|history)\s*:?\s*$", re.I)),
    ("education", re.compile(
        r"^\s*education(?:al\s+background)?\s*:?\s*$", re.I)),
    ("skills", re.compile(
        r"^\s*(?:technical\s+|core\s+|key\s+)?(?:skills|competenc(?:ies|y)|technologies)\s*:?\s*$", re.I)),
    ("languages", re.compile(r"^\s*languages?\s*:?\s*$", re.I)),
    ("certifications", re.compile(r"^\s*(?:certifications?|training)\s*:?\s*$", re.I)),
    ("projects", re.compile(r"^\s*projects?\s*:?\s*$", re.I)),
]

INLINE_MARKERS = [
    ("summary", r"\b(?:summary|profile|objective)\b"),
    ("experience", r"\b(?:experience|employment)\b"),
    ("education", r"\beducation\b"),
    ("skills", r"\b(?:skills|competenc)\b"),
    ("certifications", r"\b(?:certif|training)\b"),
    ("languages", r"\blanguages?\b"),
]

SECTION_KEYS = ["summary", "experience", "education", "skills",
                "languages", "certifications", "projects", "other"]


def parse_sections(text):
    """Parse resume text into sections by header lines or inline keywords."""
    sections = {k: [] for k in SECTION_KEYS}
    lines = text.splitlines()
    current = "other"

    for line in lines:
        s = line.strip()
        matched = False
        for name, pat in HEADER_PATTERNS:
            if pat.match(s):
                current = name
                matched = True
                break
        if not matched:
            sections[current].append(line)

    result = {k: "\n".join(v).strip() for k, v in sections.items()}

    # If most content ended up in "other", try inline keyword splitting
    classified = sum(len(result[k]) for k in SECTION_KEYS if k != "other")
    if classified < len(text) * 0.15:
        result = _inline_split(text)

    return result


def _inline_split(text):
    """Split text by finding keyword positions inline."""
    lower = text.lower()
    positions = []
    seen = set()
    for name, pattern in INLINE_MARKERS:
        if name in seen:
            continue
        m = re.search(pattern, lower)
        if m:
            positions.append((m.start(), m.end(), name))
            seen.add(name)
    positions.sort()

    result = {k: "" for k in SECTION_KEYS}

    if not positions:
        lns = [l for l in text.splitlines() if l.strip()]
        n = len(lns)
        q = max(n // 4, 1)
        result["summary"] = "\n".join(lns[:q])
        result["experience"] = "\n".join(lns[q : 2 * q])
        result["education"] = "\n".join(lns[2 * q : 3 * q])
        result["skills"] = "\n".join(lns[3 * q :])
        return result

    # Content before the first marker
    if positions[0][0] > 0:
        result["other"] = text[: positions[0][0]].strip()

    for i, (start, kw_end, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[kw_end:end].strip()
        content = re.sub(r"^[\s:—\-]+", "", content)  # strip leading colon/dash
        result[name] = content

    return result


# ---------------------------------------------------------------------------
# Template utilities
# ---------------------------------------------------------------------------
def _bullet(text, marker="\u2022"):
    """Turn text lines into bullet points."""
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(f"  {marker} {l}" for l in lines)


def _wrap(text, width=76, indent=""):
    if not text:
        return ""
    words = " ".join(text.split())
    return textwrap.fill(words, width=width,
                         initial_indent=indent, subsequent_indent=indent)


def _contact_line(rec, fields):
    """Build a ' | '-joined contact string from present fields."""
    parts = []
    for key, label in fields:
        val = rec.get(key)
        if val:
            parts.append(f"{label}{val}" if label else val)
    return " | ".join(parts)


# =========================================================================
# TEMPLATE FORMATTERS
# =========================================================================

def fmt_classic(rec, sec):
    """
    All PII grouped at top under the name.
    Clear section headers with bullet points.
    """
    out = []

    # --- PII block at top ---
    if rec.get("name"):
        out.append(rec["name"].upper())
        out.append("")
    contact = _contact_line(rec, [
        ("email", ""), ("phone", ""), ("address", ""),
        ("linkedin", ""),
    ])
    if contact:
        out.append(contact)
    hi_pii = _contact_line(rec, [("ssn", "SSN: "), ("dob", "DOB: ")])
    if hi_pii:
        out.append(hi_pii)
    if contact or hi_pii:
        out.append("")

    # --- Sections ---
    sep = "=" * 54
    for key, header in [
        ("summary", "PROFESSIONAL SUMMARY"),
        ("experience", "EMPLOYMENT HISTORY"),
        ("skills", "SKILLS"),
        ("education", "EDUCATION"),
        ("certifications", "CERTIFICATIONS"),
        ("languages", "LANGUAGES"),
        ("projects", "PROJECTS"),
    ]:
        content = sec.get(key, "")
        if content:
            out.append(sep)
            out.append(header)
            out.append(sep)
            out.append(_bullet(content))
            out.append("")

    # Remaining content
    if sec.get("other"):
        out.append(_bullet(sec["other"]))

    return "\n".join(out).rstrip()


def fmt_modern(rec, sec):
    """
    Name + job title at top.  Sections for profile, experience, skills.
    All contact details in a CONTACT INFORMATION section at the very
    bottom after a horizontal rule.
    """
    out = []
    name = (rec.get("name") or "").upper()
    if name:
        out.append(name)
    cat = rec.get("category") or ""
    if cat:
        out.append(cat.replace("-", " ").title())
    out += ["", "\u2500" * 54, ""]

    for key, hdr in [
        ("summary", "PROFILE"),
        ("experience", "EXPERIENCE"),
        ("skills", "SKILLS"),
        ("education", "EDUCATION"),
        ("certifications", "CERTIFICATIONS"),
        ("languages", "LANGUAGES"),
        ("projects", "PROJECTS"),
    ]:
        c = sec.get(key, "")
        if c:
            out.append(hdr)
            out.append(c)
            out.append("")

    if sec.get("other"):
        out.append(sec["other"])
        out.append("")

    # --- Contact at bottom ---
    contact_lines = []
    for key, label in [
        ("phone", "Phone"),
        ("email", "Email"),
        ("address", "Address"),
        ("linkedin", "Web"),
        ("ssn", "SSN"),
        ("dob", "DOB"),
    ]:
        val = rec.get(key)
        if val:
            contact_lines.append(f"  {label + ':':<10} {val}")

    if contact_lines:
        out.append("\u2500" * 54)
        out.append("CONTACT INFORMATION")
        out.extend(contact_lines)

    return "\n".join(out).rstrip()


def fmt_minimal(rec, sec):
    """
    All lowercase. No section headers, no field labels.
    Terse one-liners separated by blank lines.
    """
    parts = []

    if rec.get("name"):
        parts.append(rec["name"].lower())

    bits = []
    if rec.get("email"):
        bits.append(rec["email"].lower())
    if rec.get("phone"):
        bits.append(rec["phone"])
    if rec.get("address"):
        bits.append(rec["address"].lower())
    if rec.get("linkedin"):
        bits.append(rec["linkedin"].lower())
    if rec.get("ssn"):
        bits.append(rec["ssn"])
    if rec.get("dob"):
        bits.append(rec["dob"])
    if bits:
        parts.append("\n".join(bits))

    for key in ["summary", "experience", "education", "skills",
                "certifications", "languages", "projects", "other"]:
        c = sec.get(key, "")
        if c:
            terse = "\n".join(
                l.strip().lower() for l in c.splitlines() if l.strip()
            )
            if terse:
                parts.append(terse)

    return "\n\n".join(parts).rstrip()


def fmt_academic(rec, sec):
    """
    Third-person narrative with no section headers.
    PII woven naturally into flowing prose.
    """
    name = rec.get("name") or "The candidate"
    first = name.split()[0] if rec.get("name") else "the candidate"
    S, P = "They", "Their"  # singular they

    paras = []

    # --- Opening paragraph: identity + contact PII ---
    intro = f"{name} is"
    cat = (rec.get("category") or "").replace("-", " ").lower()
    if cat:
        a = "an" if cat[0] in "aeiou" else "a"
        intro += f" {a} experienced {cat} professional"
    else:
        intro += " a seasoned professional"
    if rec.get("address"):
        intro += f" residing at {rec['address']}"
    intro += "."

    contact = []
    if rec.get("email"):
        contact.append(f"{S} can be reached at {rec['email']}.")
    if rec.get("phone"):
        contact.append(f"{P} phone number is {rec['phone']}.")
    if rec.get("linkedin"):
        contact.append(
            f"{P} professional profile is available at {rec['linkedin']}."
        )
    if rec.get("ssn"):
        contact.append(f"{P} Social Security Number is {rec['ssn']}.")
    if rec.get("dob"):
        contact.append(f"{S} were born on {rec['dob']}.")

    opening = intro + (" " + " ".join(contact) if contact else "")
    paras.append(_wrap(opening))

    # --- Professional content ---
    if sec.get("summary"):
        s = " ".join(sec["summary"].split())
        paras.append(_wrap(
            f"{P} professional profile can be summarized as follows: {s}"
        ))
    if sec.get("experience"):
        s = " ".join(sec["experience"].split())
        paras.append(_wrap(
            f"In terms of professional experience, {first} has worked in "
            f"the following capacities: {s}"
        ))
    if sec.get("education"):
        s = " ".join(sec["education"].split())
        paras.append(_wrap(f"{P} educational background includes: {s}"))
    if sec.get("skills"):
        s = " ".join(sec["skills"].split())
        paras.append(_wrap(
            f"{S} possess the following skills and competencies: {s}"
        ))
    if sec.get("certifications"):
        s = " ".join(sec["certifications"].split())
        paras.append(_wrap(
            f"Additionally, {first} holds these certifications: {s}"
        ))
    if sec.get("languages"):
        s = " ".join(sec["languages"].split())
        paras.append(_wrap(f"{S} are proficient in: {s}"))
    if sec.get("other"):
        s = " ".join(sec["other"].split())
        if s:
            paras.append(_wrap(f"Further details: {s}"))

    return "\n\n".join(p for p in paras if p).rstrip()


def fmt_compact(rec, sec):
    """
    Fixed-width two-column layout.
    PII stacked on the left, professional content on the right.
    '=' borders.
    """
    W = 78
    LW, RW = 34, 40

    # --- Left column: PII ---
    left = []
    for key, label in [
        ("name", "Name"),
        ("email", "Email"),
        ("phone", "Phone"),
        ("address", "Address"),
        ("linkedin", "Web"),
        ("ssn", "SSN"),
        ("dob", "DOB"),
    ]:
        val = rec.get(key)
        if val:
            left.append(f"{label + ':':<9} {val}")

    # --- Right column: professional content ---
    right = []
    for key, hdr in [
        ("summary", "SUMMARY"),
        ("experience", "EXPERIENCE"),
        ("skills", "SKILLS"),
        ("education", "EDUCATION"),
        ("certifications", "CERTIFICATIONS"),
        ("languages", "LANGUAGES"),
        ("projects", "PROJECTS"),
    ]:
        c = sec.get(key, "")
        if c:
            right.append(hdr)
            for ln in c.splitlines():
                s = ln.strip()
                if s:
                    for w in textwrap.wrap(s, width=RW):
                        right.append(w)
            right.append("")

    if sec.get("other"):
        for ln in sec["other"].splitlines():
            s = ln.strip()
            if s:
                for w in textwrap.wrap(s, width=RW):
                    right.append(w)

    mx = max(len(left), len(right), 1)
    left += [""] * (mx - len(left))
    right += [""] * (mx - len(right))

    border = "=" * W
    rows = [border]
    for l, r in zip(left, right):
        rows.append(f"{l[:LW]:<{LW}} | {r[:RW]:<{RW}}")
    rows.append(border)
    return "\n".join(rows).rstrip()


def fmt_creative(rec, sec):
    """
    First-person voice.  PII scattered across different paragraphs,
    embedded in sentences.  Never grouped together.
    """
    paras = []

    # --- Opening: name + location ---
    opening = []
    if rec.get("name"):
        opening.append(f"Hi, I'm {rec['name']}.")
    else:
        opening.append("Hello!")
    cat = (rec.get("category") or "").replace("-", " ").lower()
    if cat:
        opening.append(f"I work in {cat}")
    if rec.get("address"):
        opening.append(f"and I'm based in {rec['address']}.")
    elif cat:
        opening[-1] += "."
    paras.append(_wrap(" ".join(opening)))

    # --- Summary ---
    if sec.get("summary"):
        s = " ".join(sec["summary"].split())
        paras.append(_wrap(f"A bit about me: {s}"))

    # --- Experience (embed phone) ---
    if sec.get("experience"):
        s = " ".join(sec["experience"].split())
        tail = ""
        if rec.get("phone"):
            tail = (
                f" Feel free to call me at {rec['phone']} if you'd like "
                "to discuss my background."
            )
        paras.append(_wrap(f"Here's my professional journey: {s}{tail}"))

    # --- Education ---
    if sec.get("education"):
        s = " ".join(sec["education"].split())
        paras.append(_wrap(f"On the education front: {s}"))

    # --- Skills (embed email) ---
    if sec.get("skills"):
        s = " ".join(sec["skills"].split())
        tail = ""
        if rec.get("email"):
            tail = f" Want to talk tech? Reach me at {rec['email']}."
        paras.append(_wrap(f"My toolkit includes: {s}{tail}"))
    elif rec.get("email"):
        paras.append(_wrap(f"You can reach me at {rec['email']}."))

    # --- Certifications / Languages ---
    if sec.get("certifications"):
        s = " ".join(sec["certifications"].split())
        paras.append(_wrap(f"I also hold these certifications: {s}"))
    if sec.get("languages"):
        s = " ".join(sec["languages"].split())
        paras.append(_wrap(f"I speak: {s}"))

    # --- Closing: linkedin, SSN, DOB ---
    closing = []
    if rec.get("linkedin"):
        closing.append(f"Check out my full profile at {rec['linkedin']}.")
    if rec.get("ssn"):
        closing.append(f"For official records, my SSN is {rec['ssn']}.")
    if rec.get("dob"):
        closing.append(f"I was born on {rec['dob']}.")
    if closing:
        paras.append(_wrap(" ".join(closing)))

    # Phone fallback if not embedded in experience
    if rec.get("phone") and not sec.get("experience"):
        paras.append(
            _wrap(f"Best way to reach me is by phone: {rec['phone']}.")
        )

    if sec.get("other"):
        s = " ".join(sec["other"].split())
        if s:
            paras.append(_wrap(f"A few more things: {s}"))

    return "\n\n".join(p for p in paras if p).rstrip()


FORMATTERS = {
    "classic": fmt_classic,
    "modern": fmt_modern,
    "minimal": fmt_minimal,
    "academic": fmt_academic,
    "compact": fmt_compact,
    "creative": fmt_creative,
}


# =========================================================================
# Main
# =========================================================================
def main():
    # --- Load clean records (raw text) ---
    print("[LOAD] Reading resumes_clean.jsonl ...", flush=True)
    clean_by_id = {}
    with open(INPUT_CLEAN, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            clean_by_id[rec["record_id"]] = rec["text"]
    print(f"[LOAD] {len(clean_by_id)} clean records", flush=True)

    # --- Load PII records ---
    print("[LOAD] Reading resumes_with_pii.jsonl ...", flush=True)
    pii_records = []
    with open(INPUT_PII, encoding="utf-8") as f:
        for line in f:
            pii_records.append(json.loads(line))
    print(f"[LOAD] {len(pii_records)} PII records", flush=True)

    # --- Prepare output dir ---
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    samples = {}  # template_name -> formatted text (first occurrence)

    for i, rec in enumerate(pii_records):
        if (i + 1) % 2000 == 0 or i + 1 == len(pii_records):
            print(f"[FMT] {i + 1}/{len(pii_records)}", flush=True)

        rid = rec["record_id"]
        group = rec["pii_group"]
        template = rec["template_used"]

        # Get raw text from clean source
        raw_text = clean_by_id.get(rid, "")

        # Strip PII from text content for none/low groups
        if group == "none":
            raw_text = strip_all_pii(raw_text)
        elif group == "low":
            raw_text = strip_except_name_email(raw_text)

        # Parse sections from (possibly stripped) raw text
        sections = parse_sections(raw_text)

        # Apply template
        formatter = FORMATTERS.get(template, fmt_classic)
        formatted = formatter(rec, sections)

        # Update record text
        rec["text"] = formatted

        # Write .txt file
        txt_path = DOCS_DIR / f"{rid}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted)

        # Capture first sample per template
        if template not in samples:
            samples[template] = (rid, group, formatted)

    # --- Update JSONL ---
    print(f"[WRITE] Updating {INPUT_PII} ...", flush=True)
    with open(INPUT_PII, "w", encoding="utf-8") as f:
        for rec in pii_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[WRITE] {len(pii_records)} .txt files -> {DOCS_DIR}", flush=True)

    # --- Print samples ---
    for template in FORMATTERS:
        if template in samples:
            rid, group, text = samples[template]
            print(f"\n{'=' * 70}")
            print(f"SAMPLE: template={template}  pii_group={group}  id={rid}")
            print(f"{'=' * 70}")
            lines = text.splitlines()
            if len(lines) > 60:
                print("\n".join(lines[:60]))
                print(f"\n  ... ({len(lines) - 60} more lines)")
            else:
                print(text)

    # --- Template distribution ---
    print(f"\n{'=' * 70}")
    print("TEMPLATE DISTRIBUTION")
    print(f"{'=' * 70}")
    counts = {}
    for rec in pii_records:
        counts[rec["template_used"]] = counts.get(rec["template_used"], 0) + 1
    for t in FORMATTERS:
        c = counts.get(t, 0)
        pct = 100 * c / len(pii_records)
        print(f"  {t:<14} {c:>6} ({pct:.1f}%)")
    print(f"  {'TOTAL':<14} {len(pii_records):>6}")

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
