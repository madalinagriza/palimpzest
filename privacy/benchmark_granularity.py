#!/usr/bin/env python3
"""
Q1 Benchmark: Routing granularity comparison
=============================================
Runs the same sem_filter task on a stratified sample of resumes under each of
the three routing granularities (OPERATOR / FIELD / DOCUMENT) and reports:

  - Routing breakdown: local % vs cloud %
  - Top PII entity types detected by Presidio
  - Task quality vs ground-truth pii_group labels:
      precision, recall, F1

Ground truth: records from the 'natural' and 'high' PII groups should be
accepted (they contain real PII); 'none' and 'low' should be rejected.

Usage:
    # from the project root
    .venv/bin/python privacy/benchmark_granularity.py

    # limit sample size (default: 5 per group = 20 records)
    .venv/bin/python privacy/benchmark_granularity.py --sample 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Add src/ to path so PZ imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import palimpzest as pz

from privacy_execution_strategy import create_privacy_processor
from routing_stub import AnonymizationSensitivity, ModelConfig, PrivacyRouter, RoutingGranularity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
)

# Ollama model (must be pulled: `ollama pull llama3.1:8b`)
LOCAL_MODEL = pz.Model("openai/qwen2.5:7b", api_base="http://localhost:11434/v1")

# Ground truth: which groups should sem_filter ACCEPT?
POSITIVE_GROUPS = {"natural", "high"}
NEGATIVE_GROUPS = {"none", "low"}

# ---------------------------------------------------------------------------
# Dataset helper (mirrors resume-pii-demo.py)
# ---------------------------------------------------------------------------

class ResumeDataset(pz.IterDataset):
    SCHEMA = [
        {"name": "record_id", "type": str, "desc": "Unique resume identifier"},
        {"name": "category",  "type": str, "desc": "Occupation category"},
        {"name": "pii_group", "type": str, "desc": "PII treatment group: none/low/natural/high"},
        {"name": "text",      "type": str, "desc": "Full formatted resume text"},
        {"name": "name",      "type": str, "desc": "Name in the resume, if any"},
        {"name": "phone",     "type": str, "desc": "Phone number in the resume, if any"},
        {"name": "email",     "type": str, "desc": "Email address in the resume, if any"},
        {"name": "ssn",       "type": str, "desc": "Social Security Number if present"},
    ]

    def __init__(self, jsonl_path: str, sample_per_group: int = 5):
        super().__init__(id="resumes-pii", schema=self.SCHEMA)
        groups: dict[str, list] = defaultdict(list)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                groups[raw["pii_group"]].append({
                    "record_id": raw["record_id"],
                    "category":  raw["category"],
                    "pii_group": raw["pii_group"],
                    "text":      raw["text"],
                    "name":      raw.get("name") or "",
                    "phone":     raw.get("phone") or "",
                    "email":     raw.get("email") or "",
                    "ssn":       raw.get("ssn") or "",
                })

        self.records = []
        for g in ["none", "low", "natural", "high"]:
            self.records.extend(groups[g][:sample_per_group])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(
    all_records: list[dict],
    kept_ids: set[str],
) -> dict:
    tp = fp = tn = fn = 0
    for r in all_records:
        accepted = r["record_id"] in kept_ids
        positive = r["pii_group"] in POSITIVE_GROUPS
        if accepted and positive:
            tp += 1
        elif accepted and not positive:
            fp += 1
        elif not accepted and not positive:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1)


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_one(
    granularity: RoutingGranularity,
    sensitivity: AnonymizationSensitivity,
    all_records: list[dict],
    dataset: ResumeDataset,
    config: pz.QueryProcessorConfig,
    verbose: bool,
    intent_method: str = "keyword",
) -> dict:
    plan = dataset.sem_filter(
        "The resume contains personally identifiable information such as a "
        "Social Security Number, phone number, or a real person's name",
        depends_on=["text", "ssn", "phone", "name"],
    )

    router = PrivacyRouter(ModelConfig(
        local_model="openai/qwen2.5:7b",
        local_api_base="http://localhost:11434/v1",
        anonymization_sensitivity=sensitivity,
        intent_method=intent_method,
        intent_llm_model="qwen2.5:7b",
    ))
    processor = create_privacy_processor(plan, config, router=router, granularity=granularity)

    t0 = time.time()
    result = processor.execute()
    elapsed = time.time() - t0

    # Collect IDs that passed the filter
    kept_ids: set[str] = set()
    for rec in result:
        rid = getattr(rec, "record_id", None)
        if rid is None and hasattr(rec, "to_dict"):
            rid = rec.to_dict().get("record_id")
        if rid is None and isinstance(rec, dict):
            rid = rec.get("record_id")
        if rid:
            kept_ids.add(rid)

    metrics = compute_metrics(all_records, kept_ids)
    stats   = router.stats

    return dict(
        granularity=granularity.value,
        elapsed=elapsed,
        kept=len(kept_ids),
        total=len(all_records),
        stats_summary=stats.summary(),
        local_pct=100 * stats.routed_local / stats.total if stats.total else 0,
        cloud_pct=100 * (stats.total - stats.routed_local) / stats.total if stats.total else 0,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], sensitivity: AnonymizationSensitivity):
    print(f"\nAnonymization sensitivity: {sensitivity.value.upper()}")
    header = (
        f"{'Granularity':<12}  {'Local%':>7}  {'Cloud%':>7}  "
        f"{'P':>6}  {'R':>6}  {'F1':>6}  "
        f"{'TP':>4}  {'FP':>4}  {'TN':>4}  {'FN':>4}  {'Time(s)':>8}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['granularity']:<12}  "
            f"{r['local_pct']:>6.1f}%  "
            f"{r['cloud_pct']:>6.1f}%  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['tp']:>4}  {r['fp']:>4}  {r['tn']:>4}  {r['fn']:>4}  "
            f"{r['elapsed']:>8.1f}"
        )
    print("=" * len(header))
    print()
    print("Routing detail:")
    for r in rows:
        print(f"  [{r['granularity']}]  {r['stats_summary']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SENSITIVITY_CHOICES = [s.value for s in AnonymizationSensitivity]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5,
                        help="Records per PII group (default: 5 → 20 total)")
    parser.add_argument(
        "--sensitivity",
        choices=_SENSITIVITY_CHOICES,
        default=AnonymizationSensitivity.BALANCED.value,
        help=(
            "Anonymization aggressiveness for the cloud_anonymized path. "
            "'permissive' redacts only high-confidence PII (≥0.85); "
            "'balanced' uses the default threshold (≥0.60); "
            "'conservative' redacts even low-confidence detections (≥0.30). "
            f"Default: {AnonymizationSensitivity.BALANCED.value}"
        ),
    )
    parser.add_argument(
        "--intent",
        choices=["keyword", "llm"],
        default="keyword",
        help="Intent-detection method: 'keyword' (default) or 'llm' (Ollama call per entity)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    sensitivity = AnonymizationSensitivity(args.sensitivity)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    print(f"Loading {args.sample} records per PII group from {DATA_PATH} ...")
    dataset   = ResumeDataset(DATA_PATH, sample_per_group=args.sample)
    all_records = [dataset[i] for i in range(len(dataset))]
    print(f"Loaded {len(all_records)} records "
          f"({args.sample} × none/low/natural/high)\n")

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        available_models=[LOCAL_MODEL],
        execution_strategy="sequential",
        optimizer_strategy="pareto",
        verbose=args.verbose,
        progress=False,
    )

    results = []
    for granularity in [
        RoutingGranularity.OPERATOR,
        RoutingGranularity.FIELD,
        RoutingGranularity.DOCUMENT,
    ]:
        print(f"--- Running granularity: {granularity.value}  sensitivity: {sensitivity.value}  intent: {args.intent} ---")
        row = run_one(granularity, sensitivity, all_records, dataset, config, args.verbose, intent_method=args.intent)
        results.append(row)
        print(f"  done in {row['elapsed']:.1f}s  "
              f"kept={row['kept']}/{row['total']}  "
              f"F1={row['f1']:.3f}  {row['stats_summary']}\n")

    print_table(results, sensitivity)


if __name__ == "__main__":
    main()
