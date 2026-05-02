#!/usr/bin/env python3
"""
Q1 Benchmark: Routing Granularity Comparison

Two pipeline modes:

  --pipeline single  (default)
      One operator: sem_filter on raw resume text.
      All granularities see the same fields → results will be identical.
      Establishes the baseline routing accuracy of Presidio/regex.

  --pipeline multi
      Two operators chained:
        Op1  sem_map    depends_on=["text","ssn","phone","name"]
             → extracts skills_summary (PII-free derived field)
        Op2  sem_filter depends_on=["skills_summary"]
             → filters on the derived field only

      HERE the granularities diverge:
        OPERATOR  — Op2 scans only ["skills_summary"] → no PII → cloud  ✓
        FIELD     — Op2 scans all schema fields (incl. original PII fields) → local  ✗
        DOCUMENT  — cached Op1 decision (PII found) reused for Op2 → local  ✗

      OPERATOR granularity is the only one that correctly avoids over-routing Op2.

No LLM calls are made — pure routing accuracy against ground-truth pii_labels.jsonl.

Routing ground truth:
  - pii_group in ("natural", "high") → should route LOCAL  (has real PII)
  - pii_group in ("none", "low")     → should route CLOUD  (no PII)

Metrics:
  Recall      = % of PII records correctly sent local  (privacy protection)
  Specificity = % of non-PII records correctly sent cloud  (avoids over-routing)
  Precision   = % of local-routed records that truly had PII
  F1          = harmonic mean of recall and precision
  %Local      = fraction of operator calls sent to local model

Usage:
    .venv/bin/python demos/benchmark_q1.py
    .venv/bin/python demos/benchmark_q1.py --pipeline multi
    .venv/bin/python demos/benchmark_q1.py --pipeline multi --sample 50 --out data/q1_multi.json
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

from routing_stub import PrivacyRouter, RoutingGranularity, ModelConfig, RouteDecision


# ---------------------------------------------------------------------------
# Schema that matches resumes_with_pii.jsonl (same as the demo)
# ---------------------------------------------------------------------------
SCHEMA_FIELDS = [
    "record_id", "category", "pii_group",
    "text", "name", "phone", "email", "ssn",
]

# Fields the sem_filter operator actually reads (its depends_on)
SEM_FILTER_DEPENDS_ON = ["text", "ssn", "phone", "name"]

# Multi-pipeline operator definitions
#   Op1: sem_map reads raw PII-containing fields, extracts a skills summary
#   Op2: sem_filter reads only the derived (PII-free) skills_summary field
OP1_DEPENDS_ON = ["text", "ssn", "phone", "name"]   # sem_map input
OP2_DEPENDS_ON = ["skills_summary"]                  # sem_filter input (derived)

# Schema visible to Op2 at FIELD granularity: original fields + derived field
OP2_FIELD_SCHEMA = SCHEMA_FIELDS + ["skills_summary"]


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
            groups[rec["pii_group"]].append({
                k: rec.get(k) or "" for k in SCHEMA_FIELDS
            })

    records = []
    for group in ["none", "low", "natural", "high"]:
        records.extend(groups[group][:sample_per_group])
    return records


# ---------------------------------------------------------------------------
# Simulate what sem_map produces: a PII-free skills summary
# We strip structured PII from text to approximate what an LLM would extract
# when asked "summarize this person's skills and experience" (no raw PII needed).
# ---------------------------------------------------------------------------
import re as _re

_EMAIL_RE   = _re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", _re.I)
_PHONE_RE   = _re.compile(r"(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3,4}[\s\-.]?\d{3,4}")
_SSN_RE     = _re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DOB_RE     = _re.compile(r"\b(?:(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2})\b", _re.I)
_ADDR_RE    = _re.compile(
    r"(?:[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}|\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Blvd|Road|Rd|Drive|Dr|Lane|Ln))",
    _re.I,
)

def _simulate_sem_map_output(text: str) -> str:
    """Strip structured PII to approximate what sem_map would extract as skills_summary."""
    t = _SSN_RE.sub("", text)
    t = _DOB_RE.sub("", t)
    t = _EMAIL_RE.sub("", t)
    t = _ADDR_RE.sub("", t)
    t = _PHONE_RE.sub(
        lambda m: "" if 7 <= len(_re.sub(r"\D", "", m.group(0))) <= 15 else m.group(0),
        t,
    )
    return " ".join(t.split())[:500]  # truncate to ~summary length


# ---------------------------------------------------------------------------
# Fake operator that mimics a PZ sem_filter for routing purposes
# ---------------------------------------------------------------------------
class _FakeSemFilter:
    """Minimal stand-in for a PZ PhysicalOperator so routing_stub can inspect it."""

    def __init__(self, depends_on: list[str], input_schema_fields: list[str]):
        self._depends_on = depends_on
        self._input_schema_fields = input_schema_fields

    def get_input_fields(self) -> list[str]:
        return list(self._depends_on)

    def get_model_name(self) -> str:
        return "openai/gpt-4o-mini-2024-07-18"

    def op_name(self) -> str:
        return "LLMFilter"

    @property
    def input_schema(self):
        return None  # we handle field_level override explicitly below


class _RecordProxy:
    """Wraps a plain dict so routing_stub can do getattr(record, field_name)."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name, "")


# ---------------------------------------------------------------------------
# Per-record routing decision
# ---------------------------------------------------------------------------
@dataclass
class RoutingResult:
    record_id: str
    pii_group: str
    granularity: str
    fields_scanned: list[str]
    destination: str          # "local" or "cloud"
    detections: list[dict]
    ground_truth: str         # "local" or "cloud"
    correct: bool


# ---------------------------------------------------------------------------
# Routing accuracy metrics for one condition
# ---------------------------------------------------------------------------
@dataclass
class GranularityMetrics:
    granularity: str
    total: int = 0
    tp: int = 0   # PII record → local  (correct)
    tn: int = 0   # non-PII record → cloud  (correct)
    fp: int = 0   # non-PII record → local  (over-routing)
    fn: int = 0   # PII record → cloud  (missed PII = privacy risk)
    elapsed_s: float = 0.0
    by_group: dict = field(default_factory=dict)

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
# Run one granularity condition — multi-operator pipeline
# ---------------------------------------------------------------------------
def run_granularity_multi(
    records: list[dict],
    granularity: RoutingGranularity,
    router: PrivacyRouter,
) -> tuple[GranularityMetrics, list[RoutingResult]]:
    """
    Two-operator pipeline:
      Op1 (sem_map)    depends_on = OP1_DEPENDS_ON  (reads raw PII text)
      Op2 (sem_filter) depends_on = OP2_DEPENDS_ON  (reads derived skills_summary)

    At DOCUMENT granularity Op1's routing decision is cached and reused for Op2,
    causing Op2 to be over-routed to local even though skills_summary has no PII.
    At OPERATOR granularity Op2 correctly goes to cloud.
    At FIELD granularity Op2 scans the full schema (incl. original PII fields) → local.
    """
    op1 = _FakeSemFilter(depends_on=OP1_DEPENDS_ON, input_schema_fields=SCHEMA_FIELDS)
    op2 = _FakeSemFilter(depends_on=OP2_DEPENDS_ON, input_schema_fields=OP2_FIELD_SCHEMA)

    metrics = GranularityMetrics(granularity=granularity.value)
    results: list[RoutingResult] = []
    doc_cache: dict[str, object] = {}

    t0 = time.time()

    for rec in records:
        proxy1 = _RecordProxy(rec)
        pii_group = rec["pii_group"]
        ground_truth = "cloud" if pii_group in ("none", "low") else "local"

        # --- Op1: sem_map routing ---
        if granularity == RoutingGranularity.OPERATOR:
            op1_fields = OP1_DEPENDS_ON
        elif granularity == RoutingGranularity.FIELD:
            op1_fields = SCHEMA_FIELDS
        else:  # DOCUMENT
            op1_fields = SCHEMA_FIELDS

        if granularity == RoutingGranularity.DOCUMENT:
            rid = rec["record_id"]
            if rid not in doc_cache:
                doc_cache[rid] = router.inspect(op1, op1_fields, input_record=proxy1)
            op1_decision = doc_cache[rid]
        else:
            op1_decision = router.inspect(op1, op1_fields, input_record=proxy1)

        # --- Simulate sem_map output: PII-free skills_summary ---
        skills_summary = _simulate_sem_map_output(rec.get("text", ""))
        rec2 = {**rec, "skills_summary": skills_summary}
        proxy2 = _RecordProxy(rec2)

        # --- Op2: sem_filter routing ---
        if granularity == RoutingGranularity.OPERATOR:
            op2_fields = OP2_DEPENDS_ON
            op2_decision = router.inspect(op2, op2_fields, input_record=proxy2)
        elif granularity == RoutingGranularity.FIELD:
            op2_fields = OP2_FIELD_SCHEMA  # includes original PII fields
            op2_decision = router.inspect(op2, op2_fields, input_record=proxy2)
        else:  # DOCUMENT — reuse cached Op1 decision
            op2_decision = RouteDecision(
                destination=op1_decision.destination,
                detections=op1_decision.detections,
                inspected_fields=op1_decision.inspected_fields,
                reason=f"document-level cached: {op1_decision.reason}",
            )
            router.stats.record(op2_decision)

        # We evaluate routing at the Op2 level: the interesting question is
        # whether Op2 is correctly sent to cloud (it reads no PII).
        # Ground truth for Op2: should ALWAYS be cloud (derived field is PII-free).
        op2_ground_truth = "cloud"
        op2_correct = op2_decision.destination == op2_ground_truth

        # Also track Op1 separately (Op1 ground truth same as record ground truth)
        op1_correct = op1_decision.destination == ground_truth

        # Confusion matrix uses Op2 decisions (the interesting divergence point)
        if pii_group not in metrics.by_group:
            metrics.by_group[pii_group] = {"local": 0, "cloud": 0, "cloud_anonymized": 0, "total": 0}
        metrics.by_group[pii_group][op2_decision.destination] += 1
        metrics.by_group[pii_group]["total"] += 1
        metrics.total += 1

        # For multi-pipeline: FP = op2 routed local (always wrong, it has no PII)
        #                     TN = op2 routed cloud (always correct)
        # We still track op1 TP/FN for privacy completeness
        if op1_decision.destination == "local" and ground_truth == "local":
            metrics.tp += 1
        elif op1_decision.destination == "cloud" and ground_truth == "local":
            metrics.fn += 1

        if op2_decision.destination == "local":
            metrics.fp += 1  # Op2 should never go local
        else:
            metrics.tn += 1

        results.append(RoutingResult(
            record_id=rec["record_id"],
            pii_group=pii_group,
            granularity=granularity.value,
            fields_scanned=op2_decision.inspected_fields,
            destination=op2_decision.destination,
            detections=[
                {"field": d.field_name, "entity": d.entity_type,
                 "source": d.source, "preview": d.preview}
                for d in op2_decision.detections
            ],
            ground_truth=op2_ground_truth,
            correct=op2_correct,
        ))

    metrics.elapsed_s = time.time() - t0
    return metrics, results


# ---------------------------------------------------------------------------
# Run one granularity condition — single operator
# ---------------------------------------------------------------------------
def run_granularity(
    records: list[dict],
    granularity: RoutingGranularity,
    router: PrivacyRouter,
) -> tuple[GranularityMetrics, list[RoutingResult]]:

    operator = _FakeSemFilter(
        depends_on=SEM_FILTER_DEPENDS_ON,
        input_schema_fields=SCHEMA_FIELDS,
    )

    metrics = GranularityMetrics(granularity=granularity.value)
    results: list[RoutingResult] = []

    # For DOCUMENT granularity we cache routing decisions per record
    doc_cache: dict[str, object] = {}

    t0 = time.time()

    for rec in records:
        proxy = _RecordProxy(rec)
        pii_group = rec["pii_group"]
        ground_truth = "cloud" if pii_group in ("none", "low") else "local"

        # Determine which fields to scan based on granularity
        if granularity == RoutingGranularity.OPERATOR:
            fields_to_scan = SEM_FILTER_DEPENDS_ON

        elif granularity == RoutingGranularity.FIELD:
            fields_to_scan = SCHEMA_FIELDS

        else:  # DOCUMENT
            fields_to_scan = SCHEMA_FIELDS

        # For DOCUMENT: use cached decision if available
        if granularity == RoutingGranularity.DOCUMENT:
            rid = rec["record_id"]
            if rid not in doc_cache:
                doc_cache[rid] = router.inspect(operator, fields_to_scan, input_record=proxy)
            decision = doc_cache[rid]
        else:
            decision = router.inspect(operator, fields_to_scan, input_record=proxy)

        destination = decision.destination
        correct = destination == ground_truth

        # Tally per-group counts
        if pii_group not in metrics.by_group:
            metrics.by_group[pii_group] = {"local": 0, "cloud": 0, "cloud_anonymized": 0, "total": 0}
        metrics.by_group[pii_group][destination] += 1
        metrics.by_group[pii_group]["total"] += 1

        # Confusion matrix
        metrics.total += 1
        if ground_truth == "local" and destination == "local":
            metrics.tp += 1
        elif ground_truth == "cloud" and destination == "cloud":
            metrics.tn += 1
        elif ground_truth == "cloud" and destination == "local":
            metrics.fp += 1
        else:
            metrics.fn += 1

        results.append(RoutingResult(
            record_id=rec["record_id"],
            pii_group=pii_group,
            granularity=granularity.value,
            fields_scanned=fields_to_scan,
            destination=destination,
            detections=[
                {"field": d.field_name, "entity": d.entity_type,
                 "source": d.source, "preview": d.preview}
                for d in decision.detections
            ],
            ground_truth=ground_truth,
            correct=correct,
        ))

    metrics.elapsed_s = time.time() - t0
    return metrics, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(all_metrics: list[GranularityMetrics], sample_per_group: int, pipeline: str) -> None:
    total_records = sample_per_group * 4
    mode_label = "MULTI-OPERATOR (sem_map → sem_filter)" if pipeline == "multi" else "SINGLE-OPERATOR (sem_filter)"
    print(f"\n{'='*72}")
    print(f"Q1 ROUTING GRANULARITY BENCHMARK  —  {mode_label}")
    print(f"{total_records} records ({sample_per_group}/group)")
    print(f"{'='*72}\n")
    if pipeline == "multi":
        print("  Op1 sem_map    depends_on=[text, ssn, phone, name]  →  extracts skills_summary")
        print("  Op2 sem_filter depends_on=[skills_summary]          →  filters on derived field")
        print()
        print("  Op2 should ALWAYS route to cloud (skills_summary contains no raw PII).")
        print("  Over-routing Op2 to local = wasted quality with no privacy benefit.\n")

    if pipeline == "multi":
        header = f"{'Granularity':<12} {'Op1 Recall':>11} {'Op2 Over-Route':>15} {'Op2 Correct':>12} {'%Local Op2':>11} {'Time':>8}"
        print(header)
        print("-" * 72)
        for m in all_metrics:
            op2_over_route = m.fp / m.total * 100 if m.total > 0 else 0
            op2_correct = m.tn / m.total * 100 if m.total > 0 else 0
            print(
                f"{m.granularity:<12} "
                f"{m.recall*100:>10.1f}% "
                f"{op2_over_route:>14.1f}% "
                f"{op2_correct:>11.1f}% "
                f"{m.local_rate*100:>10.1f}% "
                f"{m.elapsed_s:>7.2f}s"
            )
        print(f"\n  Op1 Recall     = % of PII records correctly sent local at Op1 (sem_map)")
        print(f"  Op2 Over-Route = % of records unnecessarily sent local at Op2 (sem_filter)")
        print(f"  Op2 Correct    = % of records correctly sent cloud at Op2")
        print(f"  %Local Op2     = fraction of Op2 calls routed to local model")
    else:
        header = f"{'Granularity':<12} {'Recall':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'%Local':>8} {'Time':>8}"
        print(header)
        print("-" * 64)
        for m in all_metrics:
            print(
                f"{m.granularity:<12} "
                f"{m.recall*100:>7.1f}% "
                f"{m.specificity*100:>7.1f}% "
                f"{m.precision*100:>7.1f}% "
                f"{m.f1*100:>7.1f}% "
                f"{m.local_rate*100:>7.1f}% "
                f"{m.elapsed_s:>7.2f}s"
            )
        print(f"\n  Recall     = % of PII records correctly routed local  (privacy protection)")
        print(f"  Spec       = % of non-PII records correctly routed cloud  (avoids over-routing)")
        print(f"  Precision  = % of local-routed records that truly had PII")
        print(f"  F1         = harmonic mean of recall and precision")
        print(f"  %Local     = fraction of all records sent to local model")

    print(f"\n{'='*72}")
    print("PER-GROUP BREAKDOWN")
    print(f"{'='*72}")
    groups = ["none", "low", "natural", "high"]
    expected = {"none": "cloud", "low": "cloud", "natural": "local", "high": "local"}
    for m in all_metrics:
        print(f"\n  [{m.granularity}]")
        print(f"  {'Group':<12} {'Expected':>10} {'Routed Local':>14} {'Routed Cloud':>14}")
        print(f"  {'-'*52}")
        for g in groups:
            bg = m.by_group.get(g, {"local": 0, "cloud": 0, "total": 0})
            print(
                f"  {g:<12} {expected[g]:>10} "
                f"{bg['local']:>14} "
                f"{bg['cloud']:>14}"
            )

    print(f"\n{'='*72}")
    print("CONFUSION MATRIX (local = positive class)")
    print(f"{'='*72}")
    for m in all_metrics:
        print(f"\n  [{m.granularity}]")
        print(f"    TP (PII → local)      : {m.tp:>4}  (privacy protected)")
        print(f"    TN (no-PII → cloud)   : {m.tn:>4}  (correctly unprotected)")
        print(f"    FP (no-PII → local)   : {m.fp:>4}  (over-routing, quality cost)")
        print(f"    FN (PII → cloud)      : {m.fn:>4}  *** privacy risk ***")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Q1 routing granularity benchmark")
    parser.add_argument("--sample", type=int, default=25,
                        help="Records per PII group (default 25, total = 4x)")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to write JSON results (optional)")
    parser.add_argument("--score-threshold", type=float, default=0.6,
                        help="Presidio confidence threshold (default 0.6)")
    parser.add_argument("--pipeline", choices=["single", "multi"], default="single",
                        help="single = sem_filter only; multi = sem_map → sem_filter chain")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    print(f"Loading {args.sample} records per PII group ...")
    records = load_sample(data_path, args.sample)
    print(f"Loaded {len(records)} records.\n")

    config = ModelConfig(score_threshold=args.score_threshold)
    granularities = [
        RoutingGranularity.OPERATOR,
        RoutingGranularity.FIELD,
        RoutingGranularity.DOCUMENT,
    ]

    all_metrics: list[GranularityMetrics] = []
    all_results: list[RoutingResult] = []

    runner = run_granularity_multi if args.pipeline == "multi" else run_granularity

    for gran in granularities:
        print(f"Running granularity: {gran.value} ...", flush=True)
        router = PrivacyRouter(config)  # fresh router per condition
        metrics, results = runner(records, gran, router)
        all_metrics.append(metrics)
        all_results.extend(results)
        if args.pipeline == "multi":
            over = metrics.fp / metrics.total * 100 if metrics.total > 0 else 0
            print(f"  done — op1_recall={metrics.recall*100:.1f}%  op2_over_route={over:.1f}%")
        else:
            print(f"  done — recall={metrics.recall*100:.1f}%  spec={metrics.specificity*100:.1f}%  F1={metrics.f1*100:.1f}%")

    print_report(all_metrics, args.sample, args.pipeline)

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        output = {
            "sample_per_group": args.sample,
            "score_threshold": args.score_threshold,
            "metrics": [asdict(m) for m in all_metrics],
            "records": [
                {
                    "record_id": r.record_id,
                    "pii_group": r.pii_group,
                    "granularity": r.granularity,
                    "destination": r.destination,
                    "ground_truth": r.ground_truth,
                    "correct": r.correct,
                    "detections": r.detections,
                }
                for r in all_results
            ],
        }
        output["pipeline"] = args.pipeline
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
