#!/usr/bin/env python3
"""
Q3 visualization: query-intent routing benchmark diagrams.

The figures are designed for both full and sampled Q3 comparison JSON files
created by:
    benchmark_q3.py --intent both --out data/q3_results_sample_both.json

Usage:
    .venv\\Scripts\\python demos\\plot_q3.py --results data\\q3_results_sample_both.json --out data\\figures\\q3_sample_both
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_KEYWORD = "#2ca02c"
C_LLM = "#7b51a8"
C_LOCAL = "#2ca02c"
C_ANON = "#f0a202"
C_CLOUD = "#2b6cb0"
C_GRID = "#e8e8e8"
C_BAD = "#c7362f"
C_NEUTRAL = "#6b7280"

GROUPS = ["none", "low", "natural", "high"]
EXPLICIT_OPS = ["extract_ssn", "extract_contact", "extract_identity"]

OP_LABELS = {
    "extract_ssn": "extract_ssn",
    "extract_contact": "extract_contact",
    "extract_identity": "extract_identity",
    "find_contact": "find_contact",
    "attribute_authorship": "attribute_authorship",
    "find_age": "find_age",
    "infer_location": "infer_location",
    "fraud_check": "fraud_check",
    "summarize_skills": "summarize_skills",
    "classify_industry": "classify_industry",
    "rate_education": "rate_education",
    "assess_seniority": "assess_seniority",
    "score_relevance": "score_relevance",
    "summarize_birth_of_career": "birth_of_career",
}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_operators(data: dict, method: str) -> list[dict]:
    if data.get("intent") == "both":
        return data[method]["operators"]
    return data["operators"]


def label(name: str) -> str:
    return OP_LABELS.get(name, name)


def save(fig, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=170, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


def run_label(data: dict) -> str:
    sample = data.get("sample_per_group")
    if sample is None:
        return "full dataset"
    return f"sample: {sample}/group ({sample * len(GROUPS)} records)"


def sorted_ops(ops: Iterable[dict]) -> list[dict]:
    ops_list = list(ops)
    return [o for o in ops_list if o["sensitive_query"]] + [
        o for o in ops_list if not o["sensitive_query"]
    ]


def pii_group_total(op: dict) -> int:
    return sum(op["by_group"].get(g, {}).get("total", 0) for g in ("natural", "high"))


def pii_counts(op: dict) -> tuple[int, int, int]:
    local = sum(op["by_group"].get(g, {}).get("local", 0) for g in ("natural", "high"))
    anon = sum(op["by_group"].get(g, {}).get("cloud_anonymized", 0) for g in ("natural", "high"))
    cloud = sum(op["by_group"].get(g, {}).get("cloud", 0) for g in ("natural", "high"))
    return local, anon, cloud


def method_totals(ops: list[dict]) -> dict[str, int | float]:
    total = sum(o["total"] for o in ops)
    correct = sum(o["correct"] for o in ops)
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "local": sum(o["n_local"] for o in ops),
        "anon": sum(o["n_cloud_anonymized"] for o in ops),
        "cloud": sum(o["n_cloud"] for o in ops),
    }


def fig_accuracy_heatmap(data: dict, kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    ops = sorted_ops(kw_ops)
    llm_by_name = {o["op_name"]: o for o in llm_ops}
    matrix = np.array([[op["accuracy"] * 100, llm_by_name[op["op_name"]]["accuracy"] * 100] for op in ops])

    fig_height = max(6, len(ops) * 0.42)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Keyword", "LLM"])
    ax.set_yticks(np.arange(len(ops)))
    ax.set_yticklabels(
        [f"{'S' if o['sensitive_query'] else 'N'}  {label(o['op_name'])}" for o in ops],
        fontsize=8,
    )

    for i, op in enumerate(ops):
        for j in range(2):
            val = matrix[i, j]
            text_color = "white" if val < 45 else "#222222"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_title(f"Q3 accuracy matrix ({run_label(data)})\nS = sensitive query, N = non-sensitive query", fontsize=11)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Routing accuracy (%)")
    fig.tight_layout()
    save(fig, out_dir, "q3_fig1_accuracy_matrix")


def fig_sensitive_routing_risk(data: dict, kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    sens_kw = [o for o in sorted_ops(kw_ops) if o["sensitive_query"]]
    llm_by_name = {o["op_name"]: o for o in llm_ops}
    rows: list[tuple[str, str, int, int, int]] = []
    for op in sens_kw:
        for method, source in (("Keyword", op), ("LLM", llm_by_name[op["op_name"]])):
            local, anon, cloud = pii_counts(source)
            rows.append((label(op["op_name"]), method, local, anon, cloud))

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.34)))
    local = np.array([r[2] for r in rows])
    anon = np.array([r[3] for r in rows])
    cloud = np.array([r[4] for r in rows])

    ax.barh(y, local, color=C_LOCAL, label="local: correct for sensitive PII")
    ax.barh(y, anon, left=local, color=C_ANON, label="cloud_anonymized: wrong intent, anonymized")
    ax.barh(y, cloud, left=local + anon, color=C_CLOUD, label="cloud: detector miss/raw-cloud risk")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} - {r[1]}" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Records in natural + high groups")
    ax.set_title(f"Sensitive-query routing risk ({run_label(data)})", fontsize=11)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "q3_fig2_sensitive_routing_risk")


def fig_llm_response_buckets(data: dict, llm_ops: list[dict], out_dir: str) -> None:
    ops = sorted_ops(llm_ops)
    names = [label(o["op_name"]) for o in ops]
    yes = np.array([o.get("llm_response_buckets", {}).get("yes", 0) for o in ops])
    no = np.array([o.get("llm_response_buckets", {}).get("no", 0) for o in ops])
    invalid = np.array([o.get("llm_response_buckets", {}).get("invalid", 0) for o in ops])
    error = np.array([o.get("llm_response_buckets", {}).get("error", 0) for o in ops])
    missing = np.array([o.get("llm_response_buckets", {}).get("missing_query", 0) for o in ops])
    other = invalid + error + missing

    y = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(10, max(6, len(ops) * 0.36)))
    ax.barh(y, yes, color=C_LOCAL, label="yes")
    ax.barh(y, no, left=yes, color=C_NEUTRAL, label="no")
    ax.barh(y, other, left=yes + no, color=C_BAD, label="invalid/error/missing")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("LLM intent calls on records with detected PII")
    ax.set_title(f"Ollama intent response buckets ({run_label(data)})\nNo invalid/error responses means failures are semantic, not parse failures", fontsize=11)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "q3_fig3_llm_response_buckets")


def fig_quality_savings(data: dict, kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    ns_kw = [o for o in sorted_ops(kw_ops) if not o["sensitive_query"]]
    llm_by_name = {o["op_name"]: o for o in llm_ops}
    names = [label(o["op_name"]) for o in ns_kw]
    baseline = np.array([o["two_way_local"] for o in ns_kw])
    kw = np.array([o["quality_savings"] for o in ns_kw])
    llm = np.array([llm_by_name[o["op_name"]]["quality_savings"] for o in ns_kw])

    y = np.arange(len(ns_kw))
    h = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(y - h, baseline, height=h, color="#b9d7f0", label="two-way baseline local calls")
    ax.barh(y, kw, height=h, color=C_KEYWORD, label="keyword cloud-anon savings")
    ax.barh(y + h, llm, height=h, color=C_LLM, label="LLM cloud-anon savings")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Detected-PII calls on non-sensitive operators")
    ax.set_title(f"Quality savings from cloud_anonymized routing ({run_label(data)})", fontsize=11)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "q3_fig4_quality_savings")


def fig_detector_recall_context(data: dict, kw_ops: list[dict], out_dir: str) -> None:
    by_name = {o["op_name"]: o for o in kw_ops}
    ops = [by_name[name] for name in EXPLICIT_OPS if name in by_name]
    labels = [label(o["op_name"]) for o in ops]
    natural_detected = np.array([o["by_group"]["natural"]["local"] for o in ops])
    natural_cloud = np.array([o["by_group"]["natural"]["cloud"] for o in ops])
    high_detected = np.array([o["by_group"]["high"]["local"] for o in ops])
    high_cloud = np.array([o["by_group"]["high"]["cloud"] for o in ops])

    x = np.arange(len(ops))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, natural_detected, width=width, color=C_LOCAL, label="natural detected -> local")
    ax.bar(x - width / 2, natural_cloud, width=width, bottom=natural_detected, color=C_CLOUD, label="natural missed -> cloud")
    ax.bar(x + width / 2, high_detected, width=width, color="#72b76a", label="high detected -> local")
    ax.bar(x + width / 2, high_cloud, width=width, bottom=high_detected, color="#7ea6d8", label="high missed -> cloud")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Records per PII group")
    ax.set_title(f"Detector recall context for explicit sensitive queries ({run_label(data)})\nKeyword intent is correct when PII is detected; remaining cloud records are detector misses", fontsize=11)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "q3_fig5_detector_recall_context")


def write_summary(data: dict, kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    kw = method_totals(kw_ops)
    llm = method_totals(llm_ops)
    path = os.path.join(out_dir, "q3_figure_summary.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Q3 figures generated from {run_label(data)}\n")
        f.write(f"Keyword accuracy: {kw['accuracy']:.1f}% ({kw['correct']}/{kw['total']})\n")
        f.write(f"LLM accuracy: {llm['accuracy']:.1f}% ({llm['correct']}/{llm['total']})\n")
        f.write("\nFigures:\n")
        f.write("1. q3_fig1_accuracy_matrix: method accuracy by operator\n")
        f.write("2. q3_fig2_sensitive_routing_risk: local vs anonymized-risk vs cloud detector misses\n")
        f.write("3. q3_fig3_llm_response_buckets: yes/no/invalid parsing buckets\n")
        f.write("4. q3_fig4_quality_savings: cloud_anonymized savings on non-sensitive operators\n")
        f.write("5. q3_fig5_detector_recall_context: natural/high detector misses for explicit sensitive ops\n")
    print(f"  saved -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q3 result figures")
    parser.add_argument(
        "--results",
        default="data\\q3_results_sample_both.json",
        help="Path to Q3 results JSON",
    )
    parser.add_argument(
        "--out",
        default="data\\figures\\q3_sample_both",
        help="Output directory",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_path = args.results if os.path.isabs(args.results) else os.path.join(root, args.results)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    print(f"Loading {results_path} ...")
    data = load(results_path)
    if data.get("intent") != "both":
        raise ValueError("plot_q3.py expects a comparison result generated with --intent both")

    kw_ops = get_operators(data, "keyword")
    llm_ops = get_operators(data, "llm")
    print(f"  backend={data['backend']}  {run_label(data)}\n")

    print("Fig 1 - accuracy matrix")
    fig_accuracy_heatmap(data, kw_ops, llm_ops, out_dir)
    print("Fig 2 - sensitive routing risk")
    fig_sensitive_routing_risk(data, kw_ops, llm_ops, out_dir)
    print("Fig 3 - LLM response buckets")
    fig_llm_response_buckets(data, llm_ops, out_dir)
    print("Fig 4 - quality savings")
    fig_quality_savings(data, kw_ops, llm_ops, out_dir)
    print("Fig 5 - detector recall context")
    fig_detector_recall_context(data, kw_ops, out_dir)
    write_summary(data, kw_ops, llm_ops, out_dir)

    print(f"\nAll figures written to {out_dir}\\")


if __name__ == "__main__":
    main()
