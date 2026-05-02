#!/usr/bin/env python3
"""
Q3 Benchmark: Query-Intent Routing
===================================
Tests whether three-way routing (local / cloud_anonymized / cloud) correctly
splits operator calls based on whether the operator's query actually needs the
sensitive field, compared to a naive two-way baseline (any PII → local).

The core question: on records that contain PII, does routing to cloud_anonymized
(anonymize first, then send to cloud) correctly apply when the operator prompt
does NOT need the raw sensitive field — preserving cloud-quality results while
still stripping PII before it leaves the machine?

Two routing modes are compared:
  two-way   — any PII detection → local (ignores query intent)
  three-way — PII detected + query needs it → local
               PII detected + query doesn't need it → cloud_anonymized
               no PII detected → cloud

Six operator configurations span the interesting cases:
  Sensitive queries   (need PII field) → ground truth: local on PII records
  Non-sensitive queries (skip PII field) → ground truth: cloud_anonymized on PII records
  Derived-field query (reads PII-free field) → ground truth: cloud on all records

Key outputs:
  - Routing correctness per operator type (% correct three-way decisions)
  - Quality savings: % of PII-record calls routed cloud_anonymized (cloud quality)
    vs. how many would go local under naive two-way routing
  - Privacy check: sensitive queries still go local with three-way (no regression)
  - Per-operator, per-group routing breakdown

No LLM calls are made.

Usage:
    .venv/bin/python demos/benchmark_q3.py
    .venv/bin/python demos/benchmark_q3.py --sample 100 --out data/q3_results.json
    .venv/bin/python demos/benchmark_q3.py --backend regex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "privacy"))

from routing_stub import PrivacyRouter, ModelConfig

# ---------------------------------------------------------------------------
# Operator configurations
#
# Each entry describes one fake sem_map / sem_filter.  The router's
# _extract_query_text() reads operator.desc, so query-intent is fully driven
# by that string against _SENSITIVE_QUERY_KEYWORDS and field-name matching.
#
# depends_on is what gets scanned for PII.  Sensitive and non-sensitive
# operators intentionally share the same raw PII fields so the only variable
# is the query text — isolating the query-intent signal.
# ---------------------------------------------------------------------------

OPERATOR_CONFIGS = [
    # ── Sensitive queries — query explicitly needs the PII field ─────────────
    # These contain keywords from _SENSITIVE_QUERY_KEYWORDS; expected → local
    {
        "name": "extract_ssn",
        "desc": "Extract the Social Security Number from the resume text.",
        "depends_on": ["text", "ssn"],
        "sensitive_query": True,
        "op_type": "sem_map",
    },
    {
        "name": "extract_contact",
        "desc": "Find the applicant's phone number and email address.",
        "depends_on": ["text", "phone", "email"],
        "sensitive_query": True,
        "op_type": "sem_map",
    },
    {
        "name": "extract_identity",
        "desc": "Identify the full name and personal contact info of the applicant.",
        "depends_on": ["text", "name", "phone"],
        "sensitive_query": True,
        "op_type": "sem_map",
    },
    # ── Non-sensitive queries — operator reads PII fields but doesn't need them
    # No keywords from _SENSITIVE_QUERY_KEYWORDS; expected → cloud_anonymized
    {
        "name": "summarize_skills",
        "desc": "Summarize the applicant's technical skills and work experience.",
        "depends_on": ["text", "ssn", "phone", "name"],   # same raw fields as above
        "sensitive_query": False,
        "op_type": "sem_map",
    },
    {
        "name": "classify_industry",
        "desc": "Identify the primary industry category for this resume.",
        "depends_on": ["text", "ssn", "phone", "name"],
        "sensitive_query": False,
        "op_type": "sem_filter",
    },
    {
        "name": "rate_education",
        "desc": "Rate the quality and relevance of the applicant's education background.",
        "depends_on": ["text", "ssn", "phone", "name"],
        "sensitive_query": False,
        "op_type": "sem_filter",
    },
    # Ambiguous — asks about a "contact" but doesn't say what kind
    {
        "name": "find_contact",
        "desc": "Find the best way to contact this applicant.",
        "depends_on": ["text", "phone", "email"],
        "sensitive_query": True,   # phone/email are PII the query needs
        "op_type": "sem_filter",
    },
    # Paraphrased — needs identity but never says "name" or "identity"
    {
        "name": "attribute_authorship",
        "desc": "Who wrote this resume? What is their background?",
        "depends_on": ["text", "name"],
        "sensitive_query": True,   # keyword method may miss this — good LLM test case
        "op_type": "sem_map",
    },
    # Non-sensitive but reads everything
    {
        "name": "assess_seniority",
        "desc": "Rate the applicant's seniority level based on years of experience.",
        "depends_on": ["text", "ssn", "phone", "name"],
        "sensitive_query": False,
        "op_type": "sem_filter",
    },
    {
    "name": "find_age",          # unique, used in output tables
    "desc": "Find me applicants above age 30",    # THIS is what intent detection reads
    "depends_on": ["text", "name"],   # fields scanned for PII — must be from SCHEMA_FIELDS
    "sensitive_query": True,         # ground truth: does this query actually need the PII?
    "op_type": "sem_filter",            # "sem_map" or "sem_filter" (cosmetic only)
    },
    {
    "name": "score_relevance",
    "desc": "Score this resume from 1 to 10 for relevance to a software engineering role.",
    "depends_on": ["text", "ssn", "phone", "name"],
    "sensitive_query": False,
    "op_type": "sem_map",
    },
    {
    "name": "infer_location",
    "desc": "Is this candidate likely based in the United States?",
    "depends_on": ["text", "phone", "ssn"],
    "sensitive_query": True,
    "op_type": "sem_filter",
    },
    {
    "name": "fraud_check",
    "desc": "Does anything about this application suggest it may be fraudulent?",
    "depends_on": ["text", "name", "ssn"],
    "sensitive_query": True,   # fraud detection likely needs name/SSN to verify
    "op_type": "sem_filter",
    },
    {
    "name": "summarize_birth_of_career",
    "desc": "Summarize where and how this person's career began.",
    "depends_on": ["text", "name"],
    "sensitive_query": False,  # "birth" appears but means career start, not DOB
    "op_type": "sem_map",      # "birth" IS a keyword → keyword method routes to local incorrectly
    }

]

SCHEMA_FIELDS = ["record_id", "category", "pii_group", "text", "name", "phone", "email", "ssn"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sample(jsonl_path: str, sample_per_group: int | None) -> list[dict]:
    """Load records. Pass None to load every record in each group."""
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
        subset = groups[group] if sample_per_group is None else groups[group][:sample_per_group]
        records.extend(subset)
    return records


# ---------------------------------------------------------------------------
# Fake operator / record proxy
# ---------------------------------------------------------------------------
class _FakeOperator:
    def __init__(self, cfg: dict):
        self.desc = cfg["desc"]
        self._depends_on = cfg["depends_on"]
        self._name = cfg["name"]

    def get_input_fields(self) -> list[str]:
        return list(self._depends_on)

    def get_model_name(self) -> str:
        return "openai/gpt-4o-mini-2024-07-18"

    def op_name(self) -> str:
        return self._name

    @property
    def generated_fields(self) -> list[str]:
        return []

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
# Per-operator result
# ---------------------------------------------------------------------------
@dataclass
class OperatorResult:
    op_name: str
    sensitive_query: bool
    record_id: str
    pii_group: str
    destination: str          # actual three-way decision
    two_way_dest: str         # simulated naive baseline (any PII → local)
    expected_three_way: str   # ground truth for three-way routing
    correct_three_way: bool
    detections: list[dict]


# ---------------------------------------------------------------------------
# Summary metrics per operator
# ---------------------------------------------------------------------------
@dataclass
class OperatorMetrics:
    op_name: str
    sensitive_query: bool
    total: int = 0
    # three-way correctness
    correct: int = 0
    # destination distribution
    n_local: int = 0
    n_cloud_anon: int = 0
    n_cloud: int = 0
    # two-way baseline counts
    two_way_local: int = 0
    two_way_cloud: int = 0
    # per pii_group breakdown
    by_group: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def quality_savings(self) -> int:
        """Calls that get cloud quality (cloud_anonymized) instead of being forced local."""
        return self.two_way_local - self.n_local

    @property
    def quality_savings_pct(self) -> float:
        if self.two_way_local == 0:
            return 0.0
        return 100.0 * self.quality_savings / self.two_way_local


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
def run(
    records: list[dict],
    backend: str,
    score_threshold: float,
    intent_method: str = "keyword",
) -> tuple[list[OperatorMetrics], list[OperatorResult]]:

    config = ModelConfig(
        detector_backend=backend,
        score_threshold=score_threshold,
        intent_method=intent_method,
    )
    router = PrivacyRouter(config)

    all_metrics: list[OperatorMetrics] = []
    all_results: list[OperatorResult] = []

    total_ops = len(OPERATOR_CONFIGS)
    for op_idx, cfg in enumerate(OPERATOR_CONFIGS, 1):
        op = _FakeOperator(cfg)
        sensitive_query = cfg["sensitive_query"]
        metrics = OperatorMetrics(op_name=cfg["name"], sensitive_query=sensitive_query)
        qtype = "sensitive" if sensitive_query else "non-sensitive"
        print(
            f"  [{op_idx}/{total_ops}] {cfg['name']}  ({qtype})"
            f"  — {len(records)} records  [intent={intent_method}]"
        )

        t0 = time.time()
        _progress_step = max(1, len(records) // 20)  # print ~20 updates per operator
        for rec_idx, rec in enumerate(records, 1):
            if rec_idx % _progress_step == 0 or rec_idx == len(records):
                elapsed = time.time() - t0
                pct = 100.0 * rec_idx / len(records)
                eta = (elapsed / rec_idx) * (len(records) - rec_idx)
                print(
                    f"\r    {rec_idx:>6}/{len(records)}  ({pct:4.1f}%)"
                    f"  {elapsed:5.1f}s elapsed  ~{eta:4.0f}s left   ",
                    end="", flush=True,
                )
            proxy = _RecordProxy(rec)
            pii_group = rec["pii_group"]
            has_pii = pii_group in ("natural", "high")

            decision = router.inspect(op, cfg["depends_on"], input_record=proxy)
            dest = decision.destination          # local / cloud_anonymized / cloud
            pii_detected = bool(decision.detections)

            # Two-way baseline: any PII detection → local, otherwise cloud
            two_way = "local" if pii_detected else "cloud"

            # Ground truth for three-way routing:
            #   PII detected + sensitive query  → local
            #   PII detected + non-sensitive    → cloud_anonymized
            #   no PII detected                 → cloud (regardless of query)
            if not pii_detected:
                expected = "cloud"
            elif sensitive_query:
                expected = "local"
            else:
                expected = "cloud_anonymized"

            correct = (dest == expected)

            if pii_group not in metrics.by_group:
                metrics.by_group[pii_group] = {
                    "local": 0, "cloud_anonymized": 0, "cloud": 0, "total": 0
                }
            metrics.by_group[pii_group][dest] += 1
            metrics.by_group[pii_group]["total"] += 1
            metrics.total += 1
            if correct:
                metrics.correct += 1
            if dest == "local":
                metrics.n_local += 1
            elif dest == "cloud_anonymized":
                metrics.n_cloud_anon += 1
            else:
                metrics.n_cloud += 1
            if two_way == "local":
                metrics.two_way_local += 1
            else:
                metrics.two_way_cloud += 1

            all_results.append(OperatorResult(
                op_name=cfg["name"],
                sensitive_query=sensitive_query,
                record_id=rec["record_id"],
                pii_group=pii_group,
                destination=dest,
                two_way_dest=two_way,
                expected_three_way=expected,
                correct_three_way=correct,
                detections=[
                    {"field": d.field_name, "entity": d.entity_type,
                     "source": d.source, "preview": d.preview}
                    for d in decision.detections
                ],
            ))

        metrics_elapsed = time.time() - t0
        print()  # newline after progress bar
        all_metrics.append(metrics)

    return all_metrics, all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(all_metrics: list[OperatorMetrics], sample_per_group: int | None, backend: str) -> None:
    # Derive actual counts from the first operator's by_group data
    first = all_metrics[0]
    total_records = first.total
    pii_records = (
        first.by_group.get("natural", {}).get("total", 0) +
        first.by_group.get("high",    {}).get("total", 0)
    )

    print(f"\n{'='*80}")
    print(f"Q3 QUERY-INTENT ROUTING BENCHMARK")
    print(f"{total_records} records ({sample_per_group}/group)  |  "
          f"granularity=OPERATOR  |  backend={backend}")
    print(f"{'='*80}\n")

    # ── Main routing correctness table ───────────────────────────────────────
    print(f"{'Operator':<20} {'Query':>10}  {'Accuracy':>9}  "
          f"{'→ local':>8}  {'→ anon':>8}  {'→ cloud':>8}  {'2-way local':>12}")
    print("-" * 80)
    for m in all_metrics:
        qtype = "sensitive" if m.sensitive_query else "non-sensitive"
        print(
            f"{m.op_name:<20} {qtype:>10}  "
            f"{m.accuracy*100:>8.1f}%  "
            f"{m.n_local:>8}  "
            f"{m.n_cloud_anon:>8}  "
            f"{m.n_cloud:>8}  "
            f"{m.two_way_local:>12}"
        )
    print(f"\n  Accuracy    = % of routing decisions matching three-way ground truth")
    print(f"  → local     = sent to local model (full privacy, lower quality)")
    print(f"  → anon      = anonymized then sent to cloud (privacy + cloud quality)")
    print(f"  → cloud     = sent to cloud as-is (no PII detected)")
    print(f"  2-way local = how many would go local under naive any-PII→local routing\n")

    # ── Quality savings summary ───────────────────────────────────────────────
    sensitive_ops   = [m for m in all_metrics if m.sensitive_query]
    nonsensitive_ops = [m for m in all_metrics if not m.sensitive_query]

    total_pii_calls    = pii_records * len(all_metrics)
    two_way_local_all  = sum(m.two_way_local for m in all_metrics)
    three_way_local    = sum(m.n_local for m in all_metrics)
    three_way_anon     = sum(m.n_cloud_anon for m in all_metrics)
    savings            = two_way_local_all - three_way_local

    print(f"{'='*80}")
    print(f"QUALITY SAVINGS FROM QUERY-INTENT ROUTING")
    print(f"{'='*80}")
    print(f"  Two-way baseline   : {two_way_local_all} operator calls → local model")
    print(f"  Three-way routing  : {three_way_local} operator calls → local model")
    print(f"                       {three_way_anon} operator calls → cloud (anonymized)")
    if two_way_local_all > 0:
        pct = 100.0 * savings / two_way_local_all
        print(f"  Quality savings    : {savings} calls ({pct:.1f}%) get cloud quality")
        print(f"                       instead of local-model quality — at no privacy cost")
    print()

    # ── Privacy check ────────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"PRIVACY CHECK — sensitive queries still go local")
    print(f"{'='*80}")
    for m in sensitive_ops:
        pii_local = m.by_group.get("natural", {}).get("local", 0) + \
                    m.by_group.get("high",    {}).get("local", 0)
        pii_anon  = m.by_group.get("natural", {}).get("cloud_anonymized", 0) + \
                    m.by_group.get("high",    {}).get("cloud_anonymized", 0)
        pii_cloud = m.by_group.get("natural", {}).get("cloud", 0) + \
                    m.by_group.get("high",    {}).get("cloud", 0)
        total_pii = pii_local + pii_anon + pii_cloud
        risk = pii_cloud + pii_anon   # anything not local for sensitive query = privacy risk
        print(f"\n  [{m.op_name}]  (sensitive query — should always go local for PII records)")
        print(f"    PII records → local       : {pii_local}/{total_pii}")
        print(f"    PII records → cloud_anon  : {pii_anon}  ← privacy risk if non-zero")
        print(f"    PII records → cloud       : {pii_cloud}  ← privacy risk if non-zero")
        if risk == 0:
            print(f"    ✓ No privacy regression")
        else:
            print(f"    ✗ {risk} records leaked past local routing — check query keywords")

    # ── Per-operator per-group breakdown ─────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"PER-OPERATOR PER-GROUP BREAKDOWN")
    print(f"{'='*80}")
    groups   = ["none", "low", "natural", "high"]
    expected_group = {"none": "cloud", "low": "cloud", "natural": "?", "high": "?"}
    for m in all_metrics:
        qtype = "sensitive" if m.sensitive_query else "non-sensitive"
        exp_pii = "local" if m.sensitive_query else "cloud_anonymized"
        print(f"\n  [{m.op_name}]  ({qtype} query)")
        print(f"  {'Group':<10} {'Expected':>15}  {'→ local':>8}  {'→ anon':>8}  {'→ cloud':>8}")
        print(f"  {'-'*54}")
        for g in groups:
            bg = m.by_group.get(g, {"local": 0, "cloud_anonymized": 0, "cloud": 0, "total": 0})
            exp = "cloud" if g in ("none", "low") else exp_pii
            print(
                f"  {g:<10} {exp:>15}  "
                f"{bg['local']:>8}  "
                f"{bg.get('cloud_anonymized', 0):>8}  "
                f"{bg['cloud']:>8}"
            )

    print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# Comparison report (--intent both)
# ---------------------------------------------------------------------------
def print_comparison(kw_metrics: list[OperatorMetrics], llm_metrics: list[OperatorMetrics]) -> None:
    print(f"\n{'='*90}")
    print("INTENT METHOD COMPARISON — keyword vs. LLM")
    print(f"{'='*90}")
    print(
        f"{'Operator':<20} {'Query':>12}  "
        f"{'KW local':>9} {'KW anon':>8} {'KW cloud':>9}  "
        f"{'LLM local':>9} {'LLM anon':>9} {'LLM cloud':>9}  "
        f"{'Disagree':>9}"
    )
    print("-" * 90)
    total_disagree = 0
    for kw, llm in zip(kw_metrics, llm_metrics):
        qtype = "sensitive" if kw.sensitive_query else "non-sensitive"
        # Count records where the two methods chose different destinations
        # We don't have per-record data here, so approximate via distribution diff
        disagree = abs(kw.n_local - llm.n_local)
        total_disagree += disagree
        print(
            f"{kw.op_name:<20} {qtype:>12}  "
            f"{kw.n_local:>9} {kw.n_cloud_anon:>8} {kw.n_cloud:>9}  "
            f"{llm.n_local:>9} {llm.n_cloud_anon:>9} {llm.n_cloud:>9}  "
            f"{disagree:>9}"
        )
    print(f"\n  Total routing disagreements (|KW_local − LLM_local| per operator): {total_disagree}")

    # Privacy regression check for sensitive operators
    print(f"\n  Privacy check on sensitive operators:")
    for kw, llm in zip(kw_metrics, llm_metrics):
        if not kw.sensitive_query:
            continue
        kw_risk  = kw.n_cloud_anon + kw.n_cloud
        llm_risk = llm.n_cloud_anon + llm.n_cloud
        kw_str  = f"{'OK' if kw_risk  == 0 else f'LEAK {kw_risk}'}"
        llm_str = f"{'OK' if llm_risk == 0 else f'LEAK {llm_risk}'}"
        print(f"    {kw.op_name:<22} keyword={kw_str:<12} llm={llm_str}")
    print(f"\n  Quality savings comparison (calls rescued from local → cloud_anonymized):")
    for kw, llm in zip(kw_metrics, llm_metrics):
        if kw.sensitive_query:
            continue
        print(
            f"    {kw.op_name:<22} keyword={kw.quality_savings:<6}  "
            f"llm={llm.quality_savings:<6}"
        )
    print(f"{'='*90}\n")


def _serialize_metrics(metrics: list[OperatorMetrics]) -> list[dict]:
    return [
        {
            "op_name": m.op_name,
            "sensitive_query": m.sensitive_query,
            "total": m.total,
            "correct": m.correct,
            "accuracy": m.accuracy,
            "n_local": m.n_local,
            "n_cloud_anonymized": m.n_cloud_anon,
            "n_cloud": m.n_cloud,
            "two_way_local": m.two_way_local,
            "quality_savings": m.quality_savings,
            "quality_savings_pct": m.quality_savings_pct,
            "by_group": m.by_group,
        }
        for m in metrics
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Q3 query-intent routing benchmark")
    parser.add_argument("--sample", type=int, default=25,
                        help="Records per PII group (default 25, total = 4×). Ignored if --all is set.")
    parser.add_argument("--all", action="store_true",
                        help="Use every record in each group (overrides --sample)")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to write JSON results (optional)")
    parser.add_argument("--score-threshold", type=float, default=0.6,
                        help="Routing confidence threshold (default 0.6)")
    parser.add_argument("--backend", default="presidio",
                        choices=["presidio", "regex", "ensemble", "deberta"],
                        help="PII detection backend (default: presidio)")
    parser.add_argument("--intent", default="keyword",
                        choices=["keyword", "llm", "both"],
                        help="Intent detection method: keyword matching, LLM (Ollama), or both (default: keyword)")
    args = parser.parse_args()
    sample_per_group: int | None = None if args.all else args.sample

    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
    )
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    load_desc = "all" if sample_per_group is None else str(sample_per_group)
    print(f"Loading {load_desc} records per PII group ...")
    records = load_sample(data_path, sample_per_group)
    print(f"Loaded {len(records)} records.\n")

    intent_methods = ["keyword", "llm"] if args.intent == "both" else [args.intent]
    results_by_method: dict[str, list[OperatorMetrics]] = {}

    for method in intent_methods:
        print(f"\n{'='*60}")
        print(f"Running Q3  backend={args.backend}  intent={method}")
        print(f"{'='*60}")
        t0 = time.time()
        all_metrics, all_results = run(records, args.backend, args.score_threshold, method)
        elapsed = time.time() - t0
        results_by_method[method] = all_metrics
        print(f"  Completed in {elapsed:.1f}s")
        print_report(all_metrics, sample_per_group, f"{args.backend}/{method}")

    if args.intent == "both":
        print_comparison(results_by_method["keyword"], results_by_method["llm"])

    if args.out:
        out_path = os.path.abspath(args.out)
        if args.intent == "both":
            payload = {
                "sample_per_group": sample_per_group,
                "score_threshold": args.score_threshold,
                "backend": args.backend,
                "intent": "both",
                "keyword": {"operators": _serialize_metrics(results_by_method["keyword"])},
                "llm":     {"operators": _serialize_metrics(results_by_method["llm"])},
            }
        else:
            payload = {
                "sample_per_group": sample_per_group,
                "score_threshold": args.score_threshold,
                "backend": args.backend,
                "intent": args.intent,
                "operators": _serialize_metrics(results_by_method[args.intent]),
            }
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
