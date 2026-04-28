#!/usr/bin/env python3
"""
Sensitivity knob benchmark — Tests 1 & 2
=========================================

Test 1: Redaction rate
    For each sensitivity level, anonymize each record and count how many PII
    entities (by type) are redacted.  No LLM needed.

Test 2: Privacy leakage
    After anonymization, re-scan the output with Presidio at a fixed
    CONSERVATIVE threshold (0.30) and count residual PII.  Measures how much
    PII slips through each sensitivity setting.

Usage:
    .venv/Scripts/python privacy/benchmark_sensitivity.py
    .venv/Scripts/python privacy/benchmark_sensitivity.py --sample 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from routing_stub import (
    AnonymizationSensitivity,
    ModelConfig,
    PrivacyRouter,
    _SENSITIVITY_TO_THRESHOLD,
)

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "resumes_with_pii.jsonl")
)

# Fixed threshold used to scan anonymized output for residual PII (Test 2)
LEAKAGE_SCAN_THRESHOLD = 0.30

SENSITIVE_ENTITIES = frozenset({
    "US_SSN", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "US_DRIVER_LICENSE", "IP_ADDRESS", "US_BANK_NUMBER", "US_PASSPORT",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str, sample_per_group: int) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            groups[r["pii_group"]].append(r)

    records = []
    for g in ["none", "low", "natural", "high"]:
        records.extend(groups[g][:sample_per_group])
    return records


# ---------------------------------------------------------------------------
# Scan text for PII entities above a given threshold
# ---------------------------------------------------------------------------

def scan_for_pii(
    text: str,
    analyzer,
    threshold: float,
    entity_filter: frozenset[str],
) -> list[dict]:
    """Return list of {entity_type, score} dicts for detections above threshold."""
    if not text.strip() or analyzer is None:
        return []
    try:
        results = analyzer.analyze(text=text, language="en")
    except Exception:
        return []
    hits = []
    for r in results:
        etype = getattr(r, "entity_type", "")
        score = getattr(r, "score", 0.0) or 0.0
        if etype in entity_filter and score >= threshold:
            hits.append({"entity_type": etype, "score": score})
    return hits


# ---------------------------------------------------------------------------
# Per-record measurement
# ---------------------------------------------------------------------------

@dataclass
class RecordResult:
    record_id: str
    pii_group: str
    detections_before: list[dict] = field(default_factory=list)
    detections_after: list[dict] = field(default_factory=list)

    @property
    def redacted_count(self) -> int:
        return len(self.detections_before) - len(self.detections_after)

    @property
    def leakage_count(self) -> int:
        return len(self.detections_after)


# ---------------------------------------------------------------------------
# Run one sensitivity level
# ---------------------------------------------------------------------------

def run_sensitivity(
    sensitivity: AnonymizationSensitivity,
    records: list[dict],
    baseline_before: dict[str, list[dict]],
) -> list[RecordResult]:
    """
    Anonymize each record at *sensitivity* and measure what PII remains.

    baseline_before maps record_id → list of PII hits scanned at the fixed
    conservative threshold on the original text.  Using a shared baseline means
    redaction rates are comparable across sensitivity levels.
    """
    config = ModelConfig(anonymization_sensitivity=sensitivity)
    router = PrivacyRouter(config)

    from routing_stub import _get_shared_analyzer
    analyzer = _get_shared_analyzer()

    results = []
    for r in records:
        text = r.get("text") or ""

        # Anonymize at this sensitivity level
        anonymized_text = router._anonymize_text(text)

        # Scan the anonymized output at the same conservative threshold used
        # to build the baseline — this is the residual PII (leakage)
        after = scan_for_pii(anonymized_text, analyzer, LEAKAGE_SCAN_THRESHOLD, SENSITIVE_ENTITIES)

        results.append(RecordResult(
            record_id=r["record_id"],
            pii_group=r["pii_group"],
            detections_before=baseline_before[r["record_id"]],
            detections_after=after,
        ))

    return results


def build_baseline(records: list[dict]) -> dict[str, list[dict]]:
    """
    Scan every record's original text at the conservative threshold.
    This is the ground-truth PII count used as the denominator across all
    sensitivity levels so redaction rates are directly comparable.
    """
    from routing_stub import _get_shared_analyzer
    analyzer = _get_shared_analyzer()
    baseline: dict[str, list[dict]] = {}
    for r in records:
        text = r.get("text") or ""
        baseline[r["record_id"]] = scan_for_pii(
            text, analyzer, LEAKAGE_SCAN_THRESHOLD, SENSITIVE_ENTITIES
        )
    return baseline


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(results: list[RecordResult]) -> dict:
    total_before = sum(len(r.detections_before) for r in results)
    total_after  = sum(len(r.detections_after) for r in results)
    redacted     = sum(r.redacted_count for r in results)

    by_group: dict[str, dict] = defaultdict(lambda: {"before": 0, "after": 0, "records": 0})
    entity_before: dict[str, int] = defaultdict(int)
    entity_after:  dict[str, int] = defaultdict(int)

    for r in results:
        g = r.pii_group
        by_group[g]["records"] += 1
        by_group[g]["before"] += len(r.detections_before)
        by_group[g]["after"]  += len(r.detections_after)
        for d in r.detections_before:
            entity_before[d["entity_type"]] += 1
        for d in r.detections_after:
            entity_after[d["entity_type"]] += 1

    redaction_rate = redacted / total_before if total_before else 0.0
    leakage_rate   = total_after  / total_before if total_before else 0.0

    return dict(
        total_records=len(results),
        total_before=total_before,
        total_redacted=redacted,
        total_leakage=total_after,
        redaction_rate=redaction_rate,
        leakage_rate=leakage_rate,
        by_group=dict(by_group),
        entity_before=dict(entity_before),
        entity_after=dict(entity_after),
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_summary(sensitivity: AnonymizationSensitivity, stats: dict):
    threshold = _SENSITIVITY_TO_THRESHOLD[sensitivity]
    print(f"\n{'='*60}")
    print(f"  Sensitivity: {sensitivity.value.upper()}  (anon threshold = {threshold})")
    print(f"{'='*60}")
    print(f"  Records:        {stats['total_records']}")
    print(f"  PII hits (before anon):  {stats['total_before']}")
    print(f"  Redacted:       {stats['total_redacted']}  "
          f"({100*stats['redaction_rate']:.1f}% redaction rate)")
    print(f"  Residual PII:   {stats['total_leakage']}  "
          f"({100*stats['leakage_rate']:.1f}% leakage rate)")

    print(f"\n  By PII group:")
    for g in ["none", "low", "natural", "high"]:
        row = stats["by_group"].get(g, {"records": 0, "before": 0, "after": 0})
        print(f"    {g:<10}  before={row['before']:>3}  "
              f"after={row['after']:>3}  "
              f"redacted={row['before']-row['after']:>3}")

    print(f"\n  Entity types detected before anonymization:")
    for etype, cnt in sorted(stats["entity_before"].items(), key=lambda x: -x[1]):
        after = stats["entity_after"].get(etype, 0)
        print(f"    {etype:<25}  before={cnt:>3}  residual={after:>3}")


def print_comparison_table(all_stats: dict[str, dict]):
    levels = list(all_stats.keys())
    print(f"\n{'='*60}")
    print("  Comparison: redaction rate vs. privacy leakage")
    print(f"{'='*60}")
    header = f"  {'Sensitivity':<14}  {'Anon thresh':>11}  {'Redaction%':>10}  {'Leakage%':>9}"
    print(header)
    print(f"  {'-'*54}")
    for level_name, stats in all_stats.items():
        s = AnonymizationSensitivity(level_name)
        t = _SENSITIVITY_TO_THRESHOLD[s]
        print(
            f"  {level_name:<14}  {t:>11.2f}  "
            f"{100*stats['redaction_rate']:>9.1f}%  "
            f"{100*stats['leakage_rate']:>8.1f}%"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=20,
                        help="Records per PII group (default: 20 → 80 total)")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    print(f"Loading {args.sample} records per PII group ...")
    records = load_records(DATA_PATH, args.sample)
    print(f"Loaded {len(records)} records\n")

    print("Building PII baseline (conservative scan on originals) ...")
    baseline = build_baseline(records)
    total_baseline = sum(len(v) for v in baseline.values())
    print(f"Ground-truth PII hits across {len(records)} records: {total_baseline}\n")

    all_stats: dict[str, dict] = {}
    for sensitivity in AnonymizationSensitivity:
        print(f"Running {sensitivity.value} ...")
        results = run_sensitivity(sensitivity, records, baseline)
        stats = aggregate(results)
        all_stats[sensitivity.value] = stats
        print_summary(sensitivity, stats)

    print_comparison_table(all_stats)


if __name__ == "__main__":
    main()
