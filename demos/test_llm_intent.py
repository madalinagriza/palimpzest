#!/usr/bin/env python3
"""
Test what Ollama outputs for benchmark queries under both prompt strategies.

per_entity — counterfactual: "if this data were redacted, would the answer degrade?"
             one call per (query, entity_type)
general    — "does this query need any personal data at all?"
             one call per query, ignores entity type

Usage:
    .venv\Scripts\python demos\test_llm_intent.py
    .venv\Scripts\python demos\test_llm_intent.py --mode general
    .venv\Scripts\python demos\test_llm_intent.py --mode both
    .venv\Scripts\python demos\test_llm_intent.py --query "Extract the SSN" --entity US_SSN
"""
from __future__ import annotations

import argparse
import json
import urllib.request

ENTITY_LABELS = {
    "US_SSN":            "Social Security Number (SSN)",
    "EMAIL_ADDRESS":     "email address",
    "PHONE_NUMBER":      "phone number",
    "CREDIT_CARD":       "credit card number",
    "US_DRIVER_LICENSE": "driver's license number",
    "IP_ADDRESS":        "IP address",
    "US_BANK_NUMBER":    "bank account number",
    "US_PASSPORT":       "passport number",
}

# ── prompt strategies ─────────────────────────────────────────────────────────

def build_per_entity(query: str, entity_type: str) -> tuple[str, str]:
    label = ENTITY_LABELS.get(entity_type, entity_type)
    system = (
        "You are a data privacy classifier. "
        "You will be given a database query and one type of sensitive personal data. "
        "Your job: decide if that data is necessary for the query to work correctly. "
        "Answer with exactly one word — 'yes' or 'no' — nothing else."
    )
    prompt = (
        f'Query: "{query}"\n'
        f"Personal data type: {label}\n\n"
        f"If {label} were completely redacted from the input, "
        f"would this query produce a wrong or incomplete answer?\n"
        f"yes or no:"
    )
    return system, prompt


def build_general(query: str, entity_type: str = "") -> tuple[str, str]:
    system = (
        "You are a data privacy classifier. "
        "You will be given a database query. "
        "Your job: decide if the query requires access to any personally identifiable "
        "information (PII) — such as names, phone numbers, email addresses, SSNs, "
        "dates of birth, or other personal data — to produce a correct answer. "
        "Answer with exactly one word — 'yes' or 'no' — nothing else."
    )
    prompt = (
        f'Query: "{query}"\n\n'
        f"Does answering this query correctly require access to any personal data?\n"
        f"yes or no:"
    )
    return system, prompt


# ── benchmark pairs ───────────────────────────────────────────────────────────
# (query, entity_type, ground_truth_sensitive)
BENCHMARK_PAIRS = [
    # explicit sensitive
    ("Extract the Social Security Number from the resume text.",           "US_SSN",       True),
    ("Find the applicant's phone number and email address.",               "PHONE_NUMBER", True),
    ("Identify the full name and personal contact info of the applicant.", "PHONE_NUMBER", True),
    # paraphrased / implicit sensitive
    ("Find the best way to contact this applicant.",                       "PHONE_NUMBER", True),
    ("Who wrote this resume? What is their background?",                   "PHONE_NUMBER", True),
    ("Find me applicants above age 30",                                    "PHONE_NUMBER", True),
    ("Is this candidate likely based in the United States?",               "PHONE_NUMBER", True),
    ("Does anything about this application suggest it may be fraudulent?", "US_SSN",       True),
    # non-sensitive (should answer NO)
    ("Summarize the applicant's technical skills and work experience.",    "US_SSN",       False),
    ("Identify the primary industry category for this resume.",            "PHONE_NUMBER", False),
    ("Rate the quality and relevance of the applicant's education.",       "US_SSN",       False),
    ("Rate the applicant's seniority level based on years of experience.", "US_SSN",       False),
]


# ── Ollama call ───────────────────────────────────────────────────────────────
def ask_ollama(system: str, prompt: str, model: str) -> tuple[str, str]:
    """Return (raw_response, verdict) where verdict is 'yes', 'no', or 'error'."""
    try:
        payload = json.dumps({
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 16},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        raw = result.get("response", "").strip()
        first_word = raw.lower().split()[0].rstrip(".,!") if raw.split() else ""
        if first_word == "yes":
            return raw, "yes"
        elif first_word == "no":
            return raw, "no"
        else:
            return raw, "error"
    except Exception as e:
        return str(e), "error"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="llama3.2")
    parser.add_argument("--mode",   default="both", choices=["per_entity", "general", "both"])
    parser.add_argument("--query",  default=None, help="Single custom query")
    parser.add_argument("--entity", default="US_SSN")
    args = parser.parse_args()

    pairs = [(args.query, args.entity, None)] if args.query else BENCHMARK_PAIRS
    modes = {"per_entity": build_per_entity, "general": build_general}
    active = {k: v for k, v in modes.items() if args.mode in (k, "both")}

    print(f"\nModel: {args.model}   mode: {args.mode}")
    print("=" * 90)

    correct = {m: 0 for m in active}
    errors  = {m: 0 for m in active}
    total = 0

    for query, entity, ground_truth in pairs:
        label = ENTITY_LABELS.get(entity, entity)
        print(f"\nQuery  : {query}")
        if args.mode != "general":
            print(f"Entity : {label}")
        if ground_truth is not None:
            print(f"Truth  : {'sensitive -> local' if ground_truth else 'non-sensitive -> cloud_anon'}")

        for mode_name, builder in active.items():
            system, prompt = builder(query, entity)
            raw, verdict = ask_ollama(system, prompt, args.model)
            if verdict == "error":
                errors[mode_name] += 1
                print(f"  [{mode_name:10}]  raw={repr(raw):<30}  ERROR (not counted)")
            else:
                decision = verdict == "yes"
                label_str = "YES -> local" if decision else "NO  -> cloud_anon"
                correct_str = ""
                if ground_truth is not None:
                    ok = decision == ground_truth
                    correct_str = "  OK" if ok else "  XX"
                    if ok:
                        correct[mode_name] += 1
                print(f"  [{mode_name:10}]  raw={repr(raw):<30}  {label_str}{correct_str}")

        total += 1
        print("-" * 90)

    if total > 1:
        print(f"\nResults over {total} pairs:")
        for mode_name in active:
            answered = total - errors[mode_name]
            acc_str = f"{correct[mode_name]}/{answered}  ({correct[mode_name]/answered*100:.0f}%)" if answered else "n/a"
            print(f"  {mode_name:12}: accuracy={acc_str}  errors={errors[mode_name]}/{total}")


if __name__ == "__main__":
    main()
