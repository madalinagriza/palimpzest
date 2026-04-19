#!/usr/bin/env python3
r"""
Demo: Use Palimpzest sem_filter on the resume PII dataset.

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python demos/resume-pii-demo.py
    # Windows: set OPENAI_API_KEY=sk-... && .venv\Scripts\python.exe demos\resume-pii-demo.py
"""
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import palimpzest as pz

# ---------------------------------------------------------------------------
# 1. Define schema matching resumes_with_pii.jsonl fields
# ---------------------------------------------------------------------------
resume_schema = [
    {"name": "record_id", "type": str, "desc": "Unique resume identifier"},
    {"name": "category", "type": str, "desc": "Occupation category (e.g. ENGINEERING, HEALTHCARE)"},
    {"name": "pii_group", "type": str, "desc": "PII treatment group: none, low, natural, or high"},
    {"name": "text", "type": str, "desc": "Full formatted resume text"},
    {"name": "name", "type": str, "desc": "Name found in the resume, if any"},
    {"name": "phone", "type": str, "desc": "Phone number found in the resume, if any"},
    {"name": "email", "type": str, "desc": "Email address found in the resume, if any"},
    {"name": "ssn", "type": str, "desc": "Social Security Number if present"},
]

# ---------------------------------------------------------------------------
# 2. Custom IterDataset that loads a JSONL file
# ---------------------------------------------------------------------------
class ResumeDataset(pz.IterDataset):
    def __init__(self, jsonl_path: str, limit: int | None = None, sample_per_group: int | None = None):
        super().__init__(id="resumes-pii", schema=resume_schema)
        all_records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    all_records.append({
                        "record_id": raw["record_id"],
                        "category": raw["category"],
                        "pii_group": raw["pii_group"],
                        "text": raw["text"],
                        "name": raw.get("name") or "",
                        "phone": raw.get("phone") or "",
                        "email": raw.get("email") or "",
                        "ssn": raw.get("ssn") or "",
                    })

        if sample_per_group is not None:
            # Take N records from each pii_group for a balanced sample
            from collections import defaultdict
            groups = defaultdict(list)
            for rec in all_records:
                groups[rec["pii_group"]].append(rec)
            self.records = []
            for group in ["none", "low", "natural", "high"]:
                self.records.extend(groups[group][:sample_per_group])
        elif limit is not None:
            self.records = all_records[:limit]
        else:
            self.records = all_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]

# ---------------------------------------------------------------------------
# 3. Build pipeline with sem_filter
# ---------------------------------------------------------------------------
def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run the data pipeline first.")
        sys.exit(1)

    # Load a stratified sample: 25 from each PII group for a meaningful comparison
    print("Loading stratified sample from resumes_with_pii.jsonl ...")
    dataset = ResumeDataset(data_path, sample_per_group=25)
    print(f"Loaded {len(dataset)} records.\n")

    # sem_filter: keep only resumes that contain PII (SSN, phone, or real name)
    filtered = dataset.sem_filter(
        "The resume contains personally identifiable information such as a "
        "Social Security Number, phone number, or a real person's name",
        depends_on=["text", "ssn", "phone", "name"],
    )

    # Configure to use GPT-4o-mini via OpenAI API (key from OPENAI_API_KEY env var)
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        available_models=[pz.Model.GPT_4o_MINI],
        execution_strategy="sequential",
        optimizer_strategy="pareto",
        verbose=True,
    )

    print("Running sem_filter pipeline via GPT-4o-mini ...\n")
    start = time.time()
    result = filtered.run(config)
    elapsed = time.time() - start

    # ---------------------------------------------------------------------------
    # 4. Print results with per-group breakdown for comparison
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"sem_filter completed in {elapsed:.1f}s  (~{elapsed/len(dataset):.1f}s/record)")
    print(f"{'='*70}\n")

    from collections import defaultdict
    kept_by_group = defaultdict(int)
    total_by_group = defaultdict(int)
    for rec in dataset.records:
        total_by_group[rec["pii_group"]] += 1

    kept = 0
    for record in result:
        kept += 1
        rec = record.to_dict() if hasattr(record, "to_dict") else record
        kept_by_group[rec.get("pii_group", "?")] += 1
        print(f"  [{rec.get('record_id', '?')}]  group={rec.get('pii_group', '?')}  "
              f"category={rec.get('category', '?')}  "
              f"name={rec.get('name', '')!r}  phone={rec.get('phone', '')!r}  "
              f"ssn={rec.get('ssn', '')!r}")

    print(f"\n{'='*70}")
    print(f"{'PII Group':<12} {'Total':>7} {'Kept':>6} {'Expected':>10} {'Recall/Spec':>12}")
    print(f"{'-'*50}")
    for group in ["none", "low", "natural", "high"]:
        total = total_by_group[group]
        kept_n = kept_by_group[group]
        expected = 0 if group in ("none", "low") else total
        metric = f"{kept_n/total*100:.0f}% spec" if group in ("none", "low") else f"{kept_n/total*100:.0f}% recall"
        # spec = % correctly rejected (1 - FPR), recall = % correctly kept (TPR)
        if group in ("none", "low"):
            metric = f"{(total-kept_n)/total*100:.0f}% spec"
        print(f"{group:<12} {total:>7} {kept_n:>6} {expected:>10} {metric:>12}")
    print(f"\nTotal input: {len(dataset)}  →  kept: {kept}")
    print(f"Model: GPT-4o-mini  |  Cost: see OpenAI dashboard")


if __name__ == "__main__":
    main()
