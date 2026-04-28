#!/usr/bin/env python3
"""
Q2 Benchmark: PII Detector Backend Comparison
==============================================
Compares four detector backends — presidio, deberta, regex, ensemble — on the
same stratified resume corpus used in Q1.  Granularity is fixed at OPERATOR
(the Q1 winner) and the pipeline is single-operator (sem_filter reading raw
resume fields), so the only variable is the PII detection backend.

Metrics (routing accuracy, no LLM calls):
  Recall      = % of PII records (natural/high) correctly routed local
  Specificity = % of non-PII records (none/low) correctly routed cloud
  Precision   = % of local-routed records that truly had PII
  F1          = harmonic mean of recall and precision
  %Local      = fraction of all records sent to local model
  Time        = wall time for routing all records

Also reports:
  - Top entity types fired per backend
  - Per-PII-group breakdown
  - FN (missed PII) overlap across backends

Ground truth:
  pii_group in ("natural", "high") → should route LOCAL
  pii_group in ("none",    "low")  → should route CLOUD

Usage:
    .venv/bin/python demos/benchmark_q2.py
    .venv/bin/python demos/benchmark_q2.py --sample 100 --out data/q2_results.json
    .venv/bin/python demos/benchmark_q2.py --backends presidio regex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "privacy"))

from routing_stub import PrivacyRouter, RoutingGranularity, ModelConfig

ALL_BACKENDS = ["presidio", "deberta", "regex", "ensemble"]

# Fields the sem_filter operator reads (depends_on)
SCHEMA_FIELDS        = ["record_id", "category", "pii_group", "text", "name", "phone", "email", "ssn"]
SEM_FILTER_DEPENDS_ON = ["text", "ssn", "phone", "name"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sample(jsonl_path: str, sample_per_group: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            groups[rec["pii_group"]].append(
                {k: rec.get(k) or "" for k in SCHEMA_FIELDS}
            )
    records = []
    for group in ["none", "low", "natural", "high"]:
        records.extend(groups[group][:sample_per_group])
    return records


# ---------------------------------------------------------------------------
# Fake operator / record proxy (mirrors benchmark_q1.py)
# ---------------------------------------------------------------------------
class _FakeSemFilter:
    def __init__(self, depends_on: list[str]):
        self._depends_on = depends_on

    def get_input_fields(self) -> list[str]:
        return list(self._depends_on)

    def get_model_name(self) -> str:
        return "openai/gpt-4o-mini-2024-07-18"

    def op_name(self) -> str:
        return "LLMFilter"

    @property
    def input_schema(self):
        return None


class _RecordProxy:
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name, "")


# ---------------------------------------------------------------------------
# Per-backend metrics
# ---------------------------------------------------------------------------
@dataclass
class BackendMetrics:
    backend: str
    total: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    elapsed_s: float = 0.0
    entity_counts: dict = field(default_factory=dict)
    by_group: dict = field(default_factory=dict)
    fn_ids: list = field(default_factory=list)   # record_ids missed (PII → cloud)
    fp_ids: list = field(default_factory=list)   # record_ids over-routed (non-PII → local)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def local_rate(self) -> float:
        return (self.tp + self.fp) / self.total if self.total > 0 else 0.0


# ---------------------------------------------------------------------------
# Run one backend condition
# ---------------------------------------------------------------------------
def run_backend(
    records: list[dict],
    backend: str,
    score_threshold: float,
) -> BackendMetrics:
    config = ModelConfig(
        detector_backend=backend,
        score_threshold=score_threshold,
    )
    router = PrivacyRouter(config)
    operator = _FakeSemFilter(depends_on=SEM_FILTER_DEPENDS_ON)
    metrics = BackendMetrics(backend=backend)

    t0 = time.time()

    for rec in records:
        proxy = _RecordProxy(rec)
        pii_group = rec["pii_group"]
        ground_truth = "cloud" if pii_group in ("none", "low") else "local"

        decision = router.inspect(operator, SEM_FILTER_DEPENDS_ON, input_record=proxy)
        destination = decision.destination
        # cloud_anonymized counts as cloud for routing accuracy purposes —
        # it means no PII strong enough to force local was detected.
        if destination == "cloud_anonymized":
            destination = "cloud"

        if pii_group not in metrics.by_group:
            metrics.by_group[pii_group] = {"local": 0, "cloud": 0, "total": 0}
        metrics.by_group[pii_group][destination] += 1
        metrics.by_group[pii_group]["total"] += 1
        metrics.total += 1

        if ground_truth == "local" and destination == "local":
            metrics.tp += 1
        elif ground_truth == "cloud" and destination == "cloud":
            metrics.tn += 1
        elif ground_truth == "cloud" and destination == "local":
            metrics.fp += 1
            metrics.fp_ids.append(rec["record_id"])
        else:
            metrics.fn += 1
            metrics.fn_ids.append(rec["record_id"])

        for d in decision.detections:
            metrics.entity_counts[d.entity_type] = (
                metrics.entity_counts.get(d.entity_type, 0) + 1
            )

    metrics.elapsed_s = time.time() - t0
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(all_metrics: list[BackendMetrics], sample_per_group: int) -> None:
    total_records = sample_per_group * 4
    print(f"\n{'='*76}")
    print(f"Q2 PII DETECTOR BACKEND COMPARISON")
    print(f"{total_records} records ({sample_per_group}/group)  |  "
          f"granularity=OPERATOR  |  pipeline=sem_filter")
    print(f"{'='*76}\n")
    print(f"{'Backend':<12} {'Recall':>8} {'Spec':>8} {'Prec':>8} "
          f"{'F1':>8} {'%Local':>8} {'Time':>8}")
    print("-" * 64)
    for m in all_metrics:
        available = "(n/a)" if m.total == 0 else ""
        print(
            f"{m.backend:<12} "
            f"{m.recall*100:>7.1f}% "
            f"{m.specificity*100:>7.1f}% "
            f"{m.precision*100:>7.1f}% "
            f"{m.f1*100:>7.1f}% "
            f"{m.local_rate*100:>7.1f}% "
            f"{m.elapsed_s:>7.2f}s  {available}"
        )

    print(f"\n  Recall     = % of PII records correctly routed local  (privacy protection)")
    print(f"  Spec       = % of non-PII records correctly routed cloud  (no over-routing)")
    print(f"  Precision  = % of local-routed records that truly had PII")
    print(f"  F1         = harmonic mean of recall and precision")
    print(f"  %Local     = fraction of all records sent to local model\n")

    print(f"{'='*76}")
    print("CONFUSION MATRIX  (local = positive class)")
    print(f"{'='*76}")
    for m in all_metrics:
        print(f"\n  [{m.backend}]")
        print(f"    TP (PII → local)     : {m.tp:>4}  (privacy protected)")
        print(f"    TN (no-PII → cloud)  : {m.tn:>4}  (correctly unprotected)")
        print(f"    FP (no-PII → local)  : {m.fp:>4}  (over-routing, quality cost)")
        print(f"    FN (PII → cloud)     : {m.fn:>4}  *** privacy risk ***")

    print(f"\n{'='*76}")
    print("PER-GROUP BREAKDOWN")
    print(f"{'='*76}")
    groups = ["none", "low", "natural", "high"]
    expected = {"none": "cloud", "low": "cloud", "natural": "local", "high": "local"}
    for m in all_metrics:
        print(f"\n  [{m.backend}]")
        print(f"  {'Group':<12} {'Expected':>10} {'→ local':>10} {'→ cloud':>10}")
        print(f"  {'-'*44}")
        for g in groups:
            bg = m.by_group.get(g, {"local": 0, "cloud": 0, "total": 0})
            print(
                f"  {g:<12} {expected[g]:>10} "
                f"{bg['local']:>10} "
                f"{bg['cloud']:>10}"
            )

    print(f"\n{'='*76}")
    print("TOP ENTITY TYPES DETECTED PER BACKEND")
    print(f"{'='*76}")
    for m in all_metrics:
        top = sorted(m.entity_counts.items(), key=lambda x: -x[1])[:8]
        print(f"\n  [{m.backend}]")
        if top:
            for entity, count in top:
                print(f"    {entity:<40} {count:>5}")
        else:
            print(f"    (no detections)")

    # FN overlap: which PII records were missed by multiple backends?
    print(f"\n{'='*76}")
    print("FN OVERLAP — PII RECORDS MISSED BY MULTIPLE BACKENDS")
    print(f"{'='*76}")
    fn_sets = {m.backend: set(m.fn_ids) for m in all_metrics}
    all_fn_ids = set().union(*fn_sets.values())
    if all_fn_ids:
        print(f"\n  {'record_id':<20}", end="")
        for m in all_metrics:
            print(f"  {m.backend:<12}", end="")
        print()
        print(f"  {'-'*20}", end="")
        for m in all_metrics:
            print(f"  {'-'*12}", end="")
        print()
        missed_by_all = all_fn_ids
        for m in all_metrics:
            missed_by_all &= fn_sets[m.backend]
        for rid in sorted(all_fn_ids):
            miss_count = sum(1 for m in all_metrics if rid in fn_sets[m.backend])
            marker = " ← all backends" if rid in missed_by_all else ""
            print(f"  {rid:<20}", end="")
            for m in all_metrics:
                cell = "MISSED" if rid in fn_sets[m.backend] else "detected"
                print(f"  {cell:<12}", end="")
            print(marker)
    else:
        print("\n  No FNs — all backends detected PII in every PII record.")

    print(f"\n{'='*76}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Q2 PII detector backend comparison")
    parser.add_argument("--sample", type=int, default=25,
                        help="Records per PII group (default 25, total = 4×)")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to write JSON results (optional)")
    parser.add_argument("--score-threshold", type=float, default=0.6,
                        help="Routing confidence threshold (default 0.6)")
    parser.add_argument("--backends", nargs="+", choices=ALL_BACKENDS,
                        default=ALL_BACKENDS,
                        help="Backends to compare (default: all four)")
    args = parser.parse_args()

    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
    )
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    print(f"Loading {args.sample} records per PII group ...")
    records = load_sample(data_path, args.sample)
    print(f"Loaded {len(records)} records.\n")

    all_metrics: list[BackendMetrics] = []

    for backend in args.backends:
        print(f"Running backend: {backend} ...", flush=True)
        m = run_backend(records, backend, args.score_threshold)
        all_metrics.append(m)
        print(
            f"  done — recall={m.recall*100:.1f}%  "
            f"spec={m.specificity*100:.1f}%  "
            f"F1={m.f1*100:.1f}%  "
            f"FN={m.fn}  FP={m.fp}  "
            f"time={m.elapsed_s:.2f}s"
        )

    print_report(all_metrics, args.sample)

    if args.out:
        out_path = os.path.abspath(args.out)
        payload = {
            "sample_per_group": args.sample,
            "score_threshold": args.score_threshold,
            "backends": args.backends,
            "metrics": [
                {
                    "backend": m.backend,
                    "total": m.total,
                    "tp": m.tp, "tn": m.tn, "fp": m.fp, "fn": m.fn,
                    "recall": m.recall,
                    "specificity": m.specificity,
                    "precision": m.precision,
                    "f1": m.f1,
                    "local_rate": m.local_rate,
                    "elapsed_s": m.elapsed_s,
                    "top_entities": sorted(
                        m.entity_counts.items(), key=lambda x: -x[1]
                    )[:10],
                    "fn_ids": m.fn_ids,
                    "fp_ids": m.fp_ids,
                }
                for m in all_metrics
            ],
        }
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
