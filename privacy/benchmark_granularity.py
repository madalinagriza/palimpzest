#!/usr/bin/env python3
"""
Q1 Benchmark: Routing granularity comparison
=============================================
Runs sem_filter (and optionally sem_map) tasks on a stratified sample of
resumes under each routing granularity and reports:

  - Routing breakdown: local % / cloud_anonymized % / cloud %
  - Task quality vs ground-truth pii_group labels: precision, recall, F1

Modes
-----
  single   (default) — one sem_filter across all 3 granularities
  prompts  — loop over FILTER_CONFIGS (sensitive + non-sensitive queries),
             run each under OPERATOR granularity, report per-prompt routing
  multi    — two-operator pipeline (sem_map → sem_filter) under OPERATOR vs
             DOCUMENT granularity to show over-routing at DOCUMENT level

Ground truth: records from 'natural' and 'high' PII groups should be
accepted (they contain real PII); 'none' and 'low' should be rejected.

Usage:
    .venv/bin/python privacy/benchmark_granularity.py
    .venv/bin/python privacy/benchmark_granularity.py --sample 10
    .venv/bin/python privacy/benchmark_granularity.py --mode prompts
    .venv/bin/python privacy/benchmark_granularity.py --mode multi --sample 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

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

LOCAL_MODEL = pz.Model("openai/qwen2.5:7b", api_base="http://localhost:11434/v1")

POSITIVE_GROUPS = {"natural", "high"}
NEGATIVE_GROUPS = {"none", "low"}

# ---------------------------------------------------------------------------
# Filter / operator configurations for --mode prompts
#
# Each entry is a sem_filter query.  sensitive_query=True means the query
# actually needs the raw PII field; ground-truth routing for PII records is:
#   sensitive_query=True  → local
#   sensitive_query=False → cloud_anonymized
# ---------------------------------------------------------------------------

FILTER_CONFIGS = [
    # ── Sensitive: keyword method fires ──────────────────────────────────────
    {
        "name": "extract_ssn",
        "desc": "Extract the Social Security Number from the resume text.",
        "depends_on": ["text", "ssn"],
        "sensitive_query": True,
    },
    {
        "name": "extract_contact",
        "desc": "Find the applicant's phone number and email address.",
        "depends_on": ["text", "phone", "email"],
        "sensitive_query": True,
    },
    # ── Sensitive: implicit / paraphrased — keyword method may miss ───────────
    {
        "name": "find_contact",
        "desc": "Find the best way to contact this applicant.",
        "depends_on": ["text", "phone", "email"],
        "sensitive_query": True,
    },
    {
        "name": "attribute_authorship",
        "desc": "Who wrote this resume? What is their background?",
        "depends_on": ["text", "name"],
        "sensitive_query": True,
    },
    {
        "name": "fraud_check",
        "desc": "Does anything about this application suggest it may be fraudulent?",
        "depends_on": ["text", "name", "ssn"],
        "sensitive_query": True,
    },
    # ── Non-sensitive: operator reads PII fields but query doesn't need them ──
    {
        "name": "summarize_skills",
        "desc": "Summarize the applicant's technical skills and work experience.",
        "depends_on": ["text", "ssn", "phone", "name"],
        "sensitive_query": False,
    },
    {
        "name": "assess_seniority",
        "desc": "Rate the applicant's seniority level based on years of experience.",
        "depends_on": ["text", "ssn", "phone", "name"],
        "sensitive_query": False,
    },
    {
        "name": "score_relevance",
        "desc": "Score this resume from 1 to 10 for relevance to a software engineering role.",
        "depends_on": ["text", "ssn", "phone", "name"],
        "sensitive_query": False,
    },
]

# ---------------------------------------------------------------------------
# Dataset helper
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

def compute_metrics(all_records: list[dict], kept_ids: set[str]) -> dict:
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


def collect_kept_ids(result) -> set[str]:
    kept: set[str] = set()
    for rec in result:
        rid = getattr(rec, "record_id", None)
        if rid is None and hasattr(rec, "to_dict"):
            rid = rec.to_dict().get("record_id")
        if rid is None and isinstance(rec, dict):
            rid = rec.get("record_id")
        if rid:
            kept.add(rid)
    return kept


def make_router(sensitivity, intent_method):
    return PrivacyRouter(ModelConfig(
        local_model="openai/qwen2.5:7b",
        local_api_base="http://localhost:11434/v1",
        anonymization_sensitivity=sensitivity,
        intent_method=intent_method,
        intent_llm_model="qwen2.5:7b",
    ))


# ---------------------------------------------------------------------------
# Mode: single — original Q1 benchmark (3 granularities, 1 filter query)
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

    router = make_router(sensitivity, intent_method)
    processor = create_privacy_processor(plan, config, router=router, granularity=granularity)

    t0 = time.time()
    result = processor.execute()
    elapsed = time.time() - t0

    kept_ids = collect_kept_ids(result)
    metrics  = compute_metrics(all_records, kept_ids)
    stats    = router.stats

    return dict(
        granularity=granularity.value,
        elapsed=elapsed,
        kept=len(kept_ids),
        total=len(all_records),
        stats_summary=stats.summary(),
        local_pct=100 * stats.routed_local / stats.total if stats.total else 0,
        cloud_anon_pct=100 * stats.routed_cloud_anon / stats.total if stats.total else 0,
        cloud_pct=100 * stats.routed_cloud / stats.total if stats.total else 0,
        **metrics,
    )


def print_single_table(rows: list[dict], sensitivity: AnonymizationSensitivity):
    print(f"\nAnonymization sensitivity: {sensitivity.value.upper()}")
    header = (
        f"{'Granularity':<12}  {'Local%':>7}  {'Anon%':>7}  {'Cloud%':>7}  "
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
            f"{r['cloud_anon_pct']:>6.1f}%  "
            f"{r['cloud_pct']:>6.1f}%  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['tp']:>4}  {r['fp']:>4}  {r['tn']:>4}  {r['fn']:>4}  "
            f"{r['elapsed']:>8.1f}"
        )
    print("=" * len(header))
    print()
    for r in rows:
        print(f"  [{r['granularity']}]  {r['stats_summary']}")
    print()


# ---------------------------------------------------------------------------
# Mode: prompts — one run per FILTER_CONFIGS entry, OPERATOR granularity
# ---------------------------------------------------------------------------

def run_prompts(
    sensitivity: AnonymizationSensitivity,
    all_records: list[dict],
    dataset: ResumeDataset,
    config: pz.QueryProcessorConfig,
    intent_method: str,
    verbose: bool,
) -> list[dict]:
    rows = []
    for cfg in FILTER_CONFIGS:
        print(f"  [{cfg['name']}]  intent={intent_method}  sensitive_query={cfg['sensitive_query']}")
        plan = dataset.sem_filter(
            cfg["desc"],
            depends_on=cfg["depends_on"],
        )

        router = make_router(sensitivity, intent_method)
        processor = create_privacy_processor(
            plan, config, router=router,
            granularity=RoutingGranularity.OPERATOR,
        )

        t0 = time.time()
        result = processor.execute()
        elapsed = time.time() - t0

        kept_ids = collect_kept_ids(result)
        stats    = router.stats

        # Routing correctness for PII records:
        #  sensitive_query=True  → ground truth: local
        #  sensitive_query=False → ground truth: cloud_anonymized
        pii_records = [r for r in all_records if r["pii_group"] in POSITIVE_GROUPS]
        n_pii = len(pii_records)
        if cfg["sensitive_query"]:
            n_correct = stats.routed_local
        else:
            n_correct = stats.routed_cloud_anon
        routing_acc = 100 * n_correct / n_pii if n_pii else 0.0

        rows.append(dict(
            name=cfg["name"],
            sensitive_query=cfg["sensitive_query"],
            elapsed=elapsed,
            kept=len(kept_ids),
            total=len(all_records),
            local_pct=100 * stats.routed_local / stats.total if stats.total else 0,
            cloud_anon_pct=100 * stats.routed_cloud_anon / stats.total if stats.total else 0,
            cloud_pct=100 * stats.routed_cloud / stats.total if stats.total else 0,
            routing_acc=routing_acc,
            stats_summary=stats.summary(),
        ))
        print(f"      → local={stats.routed_local} cloud_anon={stats.routed_cloud_anon} "
              f"cloud={stats.routed_cloud}  routing_acc={routing_acc:.1f}%  {elapsed:.1f}s\n")
    return rows


def print_prompts_table(rows: list[dict], intent_method: str):
    print(f"\nRouting per query (intent={intent_method}, granularity=OPERATOR)")
    header = (
        f"{'Operator':<22}  {'Sensitive':>9}  "
        f"{'Local%':>7}  {'Anon%':>7}  {'Cloud%':>7}  "
        f"{'RouteAcc%':>10}  {'Time(s)':>8}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<22}  "
            f"{'Yes' if r['sensitive_query'] else 'No':>9}  "
            f"{r['local_pct']:>6.1f}%  "
            f"{r['cloud_anon_pct']:>6.1f}%  "
            f"{r['cloud_pct']:>6.1f}%  "
            f"{r['routing_acc']:>9.1f}%  "
            f"{r['elapsed']:>8.1f}"
        )
    print("=" * len(header))
    print()
    print("RouteAcc% = % of PII records routed to the correct destination")
    print("  sensitive_query=Yes → correct destination is 'local'")
    print("  sensitive_query=No  → correct destination is 'cloud_anonymized'")
    print()


# ---------------------------------------------------------------------------
# Mode: multi — sem_map → sem_filter pipeline, OPERATOR vs DOCUMENT
# ---------------------------------------------------------------------------

def run_multi(
    sensitivity: AnonymizationSensitivity,
    all_records: list[dict],
    dataset: ResumeDataset,
    config: pz.QueryProcessorConfig,
    intent_method: str,
    verbose: bool,
) -> list[dict]:
    """
    Two-operator pipeline:
      Op1  sem_map   depends_on=["text"] — summarize skills (non-sensitive,
                     but text may contain embedded PII for natural/high groups)
      Op2  sem_filter depends_on=["skills_summary"] — check experience level
                     (derived field; Presidio should find no PII here)

    OPERATOR granularity: routes each operator independently.
      Op1 reads "text" — if PII embedded in text, routes local.
      Op2 reads "skills_summary" (PII-free derived field) — routes cloud.

    DOCUMENT granularity: scans all fields once per record, reuses for both ops.
      If document has PII → both Op1 AND Op2 go local (over-routing Op2).
    """
    rows = []
    for granularity in [RoutingGranularity.OPERATOR, RoutingGranularity.DOCUMENT]:
        print(f"  [multi / {granularity.value}]  intent={intent_method}")

        plan = (
            dataset
            .sem_map(
                [{"name": "skills_summary",
                  "desc": "Summarize the applicant's technical skills and years of experience.",
                  "type": str}],
                desc="Extract skills from resume",
                depends_on=["text"],
            )
            .sem_filter(
                "Does this applicant have 5 or more years of relevant work experience?",
                depends_on=["skills_summary"],
            )
        )

        router = make_router(sensitivity, intent_method)
        processor = create_privacy_processor(
            plan, config, router=router, granularity=granularity,
        )

        t0 = time.time()
        result = processor.execute()
        elapsed = time.time() - t0

        kept_ids = collect_kept_ids(result)
        stats    = router.stats
        metrics  = compute_metrics(all_records, kept_ids)

        rows.append(dict(
            granularity=granularity.value,
            elapsed=elapsed,
            kept=len(kept_ids),
            total=len(all_records),
            local_pct=100 * stats.routed_local / stats.total if stats.total else 0,
            cloud_anon_pct=100 * stats.routed_cloud_anon / stats.total if stats.total else 0,
            cloud_pct=100 * stats.routed_cloud / stats.total if stats.total else 0,
            stats_summary=stats.summary(),
            **metrics,
        ))
        print(f"      → local={stats.routed_local} cloud_anon={stats.routed_cloud_anon} "
              f"cloud={stats.routed_cloud}  kept={len(kept_ids)}/{len(all_records)}  {elapsed:.1f}s\n")
    return rows


def print_multi_table(rows: list[dict]):
    print("\nMulti-operator pipeline: sem_map (skills) → sem_filter (5yr+ experience)")
    print("Op1 reads 'text' (may contain PII); Op2 reads 'skills_summary' (PII-free)")
    header = (
        f"{'Granularity':<12}  {'Local%':>7}  {'Anon%':>7}  {'Cloud%':>7}  "
        f"{'P':>6}  {'R':>6}  {'F1':>6}  {'Time(s)':>8}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['granularity']:<12}  "
            f"{r['local_pct']:>6.1f}%  "
            f"{r['cloud_anon_pct']:>6.1f}%  "
            f"{r['cloud_pct']:>6.1f}%  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['elapsed']:>8.1f}"
        )
    print("=" * len(header))
    print()
    print("Expected: OPERATOR routes Op2 (skills_summary) to cloud;")
    print("          DOCUMENT over-routes Op2 to local (same decision as Op1).")
    print()
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
    )
    parser.add_argument(
        "--intent",
        choices=["keyword", "llm"],
        default="keyword",
        help="Intent-detection method (default: keyword)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "prompts", "multi"],
        default="single",
        help=(
            "Benchmark mode: "
            "'single' — one sem_filter across 3 granularities (default); "
            "'prompts' — loop over 8 filter queries under OPERATOR granularity; "
            "'multi' — two-operator pipeline (sem_map → sem_filter) comparing "
            "OPERATOR vs DOCUMENT granularity"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    sensitivity = AnonymizationSensitivity(args.sensitivity)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    print(f"Loading {args.sample} records per PII group from {DATA_PATH} ...")
    dataset     = ResumeDataset(DATA_PATH, sample_per_group=args.sample)
    all_records = [dataset[i] for i in range(len(dataset))]
    print(f"Loaded {len(all_records)} records "
          f"({args.sample} × none/low/natural/high)  "
          f"mode={args.mode}  intent={args.intent}\n")

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        available_models=[LOCAL_MODEL],
        execution_strategy="sequential",
        optimizer_strategy="pareto",
        verbose=args.verbose,
        progress=False,
    )

    if args.mode == "single":
        results = []
        for granularity in [
            RoutingGranularity.OPERATOR,
            RoutingGranularity.FIELD,
            RoutingGranularity.DOCUMENT,
        ]:
            print(f"--- Granularity: {granularity.value}  "
                  f"sensitivity: {sensitivity.value}  intent: {args.intent} ---")
            row = run_one(
                granularity, sensitivity, all_records, dataset,
                config, args.verbose, intent_method=args.intent,
            )
            results.append(row)
            print(f"  done in {row['elapsed']:.1f}s  "
                  f"kept={row['kept']}/{row['total']}  "
                  f"F1={row['f1']:.3f}  {row['stats_summary']}\n")
        print_single_table(results, sensitivity)

    elif args.mode == "prompts":
        print(f"Running {len(FILTER_CONFIGS)} filter queries  "
              f"sensitivity={sensitivity.value}  intent={args.intent}\n")
        rows = run_prompts(
            sensitivity, all_records, dataset, config, args.intent, args.verbose,
        )
        print_prompts_table(rows, args.intent)

    else:  # multi
        print(f"Running multi-operator pipeline  "
              f"sensitivity={sensitivity.value}  intent={args.intent}\n")
        rows = run_multi(
            sensitivity, all_records, dataset, config, args.intent, args.verbose,
        )
        print_multi_table(rows)


if __name__ == "__main__":
    main()
