#!/usr/bin/env python3
"""
GPT-4o cloud baseline: sem_filter + sem_map on the resume PII dataset.

Produces reference precision/recall/F1 scores for quality-loss comparison
against the privacy-routing system.

Usage (always run --test first):
    set OPENAI_API_KEY=sk-...
    python demos/gpt4o_baseline.py --test --task filter
    python demos/gpt4o_baseline.py --test
    python demos/gpt4o_baseline.py --sample 25 --task both

Cost estimates (GPT-4o at $2.50/M in, $10/M out):
    --test  (2/group =  8 records x 2 tasks):  ~$0.003
    --sample 5  (20 records x 2 tasks):        ~$0.008
    --sample 25 (100 records x 2 tasks):       ~$0.40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import palimpzest as pz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
)
LABELS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "pii_labels.jsonl")
)
RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "gpt4o_baseline_results.json")
)

POSITIVE_GROUPS = {"natural", "high"}
NEGATIVE_GROUPS = {"none", "low"}
POSITIVE_STATUS = {"original", "injected"}

PLACEHOLDER_VALUES = {"", "n/a", "none", "not provided", "not found",
                      "[placeholder]", "unknown", "absent", "no ssn", "no phone",
                      "no email", "no name"}

# ---------------------------------------------------------------------------
# Dataset
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
# Ground-truth helpers
# ---------------------------------------------------------------------------

def load_pii_labels(path: str) -> dict[str, dict]:
    """Returns {record_id: field_status_dict}."""
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            labels[obj["record_id"]] = obj.get("field_status", {})
    return labels


def is_extracted(value: str) -> bool:
    v = (value or "").strip().lower()
    return bool(v) and v not in PLACEHOLDER_VALUES


def _get_rid(rec) -> str | None:
    if hasattr(rec, "to_dict"):
        return rec.to_dict().get("record_id")
    if isinstance(rec, dict):
        return rec.get("record_id")
    return getattr(rec, "record_id", None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_filter_metrics(all_records: list[dict], kept_ids: set[str]) -> dict:
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4))


def compute_map_metrics(
    map_outputs: list[dict],
    pii_labels: dict[str, dict],
) -> dict:
    """Per-field precision/recall/F1 for sem_map extraction."""
    fields = ["name", "phone", "email", "ssn"]
    counts: dict[str, dict] = {f: dict(tp=0, fp=0, tn=0, fn=0) for f in fields}

    for rec in map_outputs:
        rid = rec.get("record_id")
        if rid not in pii_labels:
            continue
        fs = pii_labels[rid]
        for field in fields:
            extracted_key = f"extracted_{field}"
            extracted = is_extracted(rec.get(extracted_key, ""))
            positive = fs.get(field, "absent") in POSITIVE_STATUS
            if extracted and positive:
                counts[field]["tp"] += 1
            elif extracted and not positive:
                counts[field]["fp"] += 1
            elif not extracted and not positive:
                counts[field]["tn"] += 1
            else:
                counts[field]["fn"] += 1

    result = {}
    for field in fields:
        c = counts[field]
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        result[field] = dict(
            tp=c["tp"], fp=c["fp"], tn=c["tn"], fn=c["fn"],
            precision=round(p, 4), recall=round(r, 4), f1=round(f, 4)
        )
    return result


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_filter(dataset: ResumeDataset, config: pz.QueryProcessorConfig) -> tuple[set[str], float]:
    plan = dataset.sem_filter(
        "The resume contains personally identifiable information such as a "
        "Social Security Number, phone number, or a real person's name",
        depends_on=["text", "ssn", "phone", "name"],
    )
    t0 = time.time()
    result = plan.run(config)
    elapsed = time.time() - t0

    kept_ids: set[str] = set()
    for rec in result:
        rid = _get_rid(rec)
        if rid:
            kept_ids.add(rid)
    return kept_ids, elapsed


def run_map(dataset: ResumeDataset, config: pz.QueryProcessorConfig) -> tuple[list[dict], float]:
    plan = dataset.sem_map(
        cols=[
            {"name": "extracted_name",  "type": str,
             "desc": "Full name of the person as it appears in the resume. "
                     "Return empty string if no real name is present."},
            {"name": "extracted_phone", "type": str,
             "desc": "Phone number in the resume. "
                     "Return empty string if absent."},
            {"name": "extracted_email", "type": str,
             "desc": "Email address in the resume. "
                     "Return empty string if absent."},
            {"name": "extracted_ssn",   "type": str,
             "desc": "Social Security Number in XXX-XX-XXXX format. "
                     "Return empty string if absent."},
        ],
        desc="Extract PII fields from the resume text.",
        depends_on=["text"],
    )
    t0 = time.time()
    result = plan.run(config)
    elapsed = time.time() - t0

    outputs: list[dict] = []
    for rec in result:
        d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        outputs.append(d)
    return outputs, elapsed


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_filter_table(metrics: dict, by_group: dict, total: int, elapsed: float):
    print(f"\n{'='*65}")
    print("sem_filter  (GPT-4o)")
    print(f"{'='*65}")
    print(f"{'Group':<10} {'Total':>6} {'Kept':>6} {'Metric':>14}")
    print(f"{'-'*40}")
    for g in ["none", "low", "natural", "high"]:
        tot = by_group["total"][g]
        kept = by_group["kept"][g]
        if g in NEGATIVE_GROUPS:
            label = f"{(tot-kept)/tot*100:.0f}% spec" if tot else "—"
        else:
            label = f"{kept/tot*100:.0f}% recall" if tot else "—"
        print(f"{g:<10} {tot:>6} {kept:>6} {label:>14}")
    print(f"{'-'*40}")
    print(f"{'OVERALL':<10} {total:>6} {sum(by_group['kept'].values()):>6}")
    print(f"\n  Precision={metrics['precision']:.3f}  "
          f"Recall={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  "
          f"(TP={metrics['tp']} FP={metrics['fp']} "
          f"TN={metrics['tn']} FN={metrics['fn']})")
    print(f"  Time: {elapsed:.1f}s  (~{elapsed/total:.1f}s/record)")
    print(f"{'='*65}\n")


def print_map_table(metrics: dict, elapsed: float, total: int):
    print(f"\n{'='*65}")
    print("sem_map  (GPT-4o)  — per-field extraction")
    print(f"{'='*65}")
    print(f"{'Field':<10} {'P':>7} {'R':>7} {'F1':>7}  "
          f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print(f"{'-'*55}")
    for field in ["name", "phone", "email", "ssn"]:
        m = metrics[field]
        print(f"{field:<10} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}  "
              f"{m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4}")
    print(f"\n  Time: {elapsed:.1f}s  (~{elapsed/total:.1f}s/record)")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-4o cloud baseline (sem_filter + sem_map) on the resume PII dataset."
    )
    parser.add_argument("--sample", type=int, default=25,
                        help="Records per PII group (default 25 → 100 total)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 2 records per group (8 total, ~$0.003)")
    parser.add_argument("--task", choices=["filter", "map", "both"], default="both",
                        help="Which task to run (default: both)")
    args = parser.parse_args()

    n_per_group = 2 if args.test else args.sample
    n_total = n_per_group * 4

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    for path, name in [(DATA_PATH, "resumes_with_pii.jsonl"), (LABELS_PATH, "pii_labels.jsonl")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    mode_label = "TEST MODE" if args.test else f"{n_per_group}/group"
    print(f"\nGPT-4o baseline  |  task={args.task}  |  {mode_label}  |  {n_total} records")
    est_cost = n_total * 2 * 0.004  # rough: 2 tasks, ~$0.004/record
    if args.task != "both":
        est_cost /= 2
    print(f"Estimated cost: ~${est_cost:.3f}  (GPT-4o, both tasks)\n")

    print(f"Loading {n_per_group} records per PII group ...")
    dataset = ResumeDataset(DATA_PATH, sample_per_group=n_per_group)
    all_records = [dataset[i] for i in range(len(dataset))]
    print(f"Loaded {len(all_records)} records.\n")

    pii_labels = load_pii_labels(LABELS_PATH)

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        available_models=[pz.Model.GPT_4o],
        execution_strategy="sequential",
        optimizer_strategy="pareto",
        verbose=False,
        progress=True,
    )

    results: dict = {"sample_per_group": n_per_group, "total_records": n_total}

    # --- sem_filter ---
    if args.task in ("filter", "both"):
        print("Running sem_filter with GPT-4o ...")
        kept_ids, elapsed = run_filter(dataset, config)

        by_group: dict = {"total": defaultdict(int), "kept": defaultdict(int)}
        for r in all_records:
            by_group["total"][r["pii_group"]] += 1
        for r in all_records:
            if r["record_id"] in kept_ids:
                by_group["kept"][r["pii_group"]] += 1

        filter_metrics = compute_filter_metrics(all_records, kept_ids)
        print_filter_table(filter_metrics, by_group, n_total, elapsed)
        results["sem_filter"] = {
            "metrics": filter_metrics,
            "elapsed_s": round(elapsed, 2),
            "kept_ids": sorted(kept_ids),
            "by_group": {
                g: {"total": by_group["total"][g], "kept": by_group["kept"][g]}
                for g in ["none", "low", "natural", "high"]
            },
        }

    # --- sem_map ---
    if args.task in ("map", "both"):
        print("Running sem_map with GPT-4o ...")
        map_outputs, elapsed = run_map(dataset, config)
        map_metrics = compute_map_metrics(map_outputs, pii_labels)
        print_map_table(map_metrics, elapsed, n_total)

        results["sem_map"] = {
            "metrics": map_metrics,
            "elapsed_s": round(elapsed, 2),
            "outputs": [
                {k: v for k, v in o.items()
                 if k in ("record_id", "pii_group",
                          "extracted_name", "extracted_phone",
                          "extracted_email", "extracted_ssn")}
                for o in map_outputs
            ],
        }

    # --- save results ---
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
