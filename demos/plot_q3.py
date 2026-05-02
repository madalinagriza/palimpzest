#!/usr/bin/env python3
"""
Q3 visualization: query-intent routing benchmark diagrams.

Handles both single-method JSON (intent=keyword/llm) and comparison
JSON (intent=both) produced by:
    benchmark_q3.py --intent both --out data/q3_results_both.json

Usage:
    .venv\Scripts\python demos\plot_q3.py
    .venv\Scripts\python demos\plot_q3.py --results data\q3_results_both.json --out data\figures\q3_both
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
C_LOCAL  = "#d62728"   # red    – local model
C_ANON   = "#ff7f0e"   # amber  – cloud + anonymized
C_CLOUD  = "#1f77b4"   # blue   – cloud as-is
C_KW     = "#2ca02c"   # green  – keyword method
C_LLM    = "#9467bd"   # purple – LLM method
C_SAFE   = "#2ca02c"   # green  – safe routing
C_LEAK   = "#d62728"   # red    – privacy leakage

GROUPS = ["none", "low", "natural", "high"]

# Readable operator labels (name → display label)
OP_LABELS = {
    "extract_ssn":               "extract_ssn\n[explicit: SSN]",
    "extract_contact":           "extract_contact\n[explicit: phone/email]",
    "extract_identity":          "extract_identity\n[explicit: name/phone]",
    "find_contact":              "find_contact\n[paraphrased: reach out]",
    "attribute_authorship":      "attribute_authorship\n[paraphrased: who wrote?]",
    "find_age":                  "find_age\n[implicit: age filter]",
    "infer_location":            "infer_location\n[implicit: US-based?]",
    "fraud_check":               "fraud_check\n[implicit: fraudulent?]",
    "summarize_skills":          "summarize_skills",
    "classify_industry":         "classify_industry",
    "rate_education":            "rate_education",
    "assess_seniority":          "assess_seniority",
    "score_relevance":           "score_relevance",
    "summarize_birth_of_career": "birth_of_career\n[birth ≠ DOB]",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_operators(data: dict, method: str) -> list[dict]:
    if data.get("intent") == "both":
        return data[method]["operators"]
    return data["operators"]


def _label(name: str) -> str:
    return OP_LABELS.get(name, name)


def _save(fig, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
    plt.close(fig)


def _sort_ops(ops: list[dict]) -> list[dict]:
    """Sensitive operators first, then non-sensitive."""
    return sorted(ops, key=lambda o: (not o["sensitive_query"], o["op_name"]))


def _pii_leaked(op: dict) -> int:
    """PII records (natural+high) that bypassed local routing."""
    pii_groups = ["natural", "high"]
    return sum(
        op["by_group"].get(g, {}).get("cloud_anonymized", 0) +
        op["by_group"].get(g, {}).get("cloud", 0)
        for g in pii_groups
    )


def _pii_local(op: dict) -> int:
    return sum(op["by_group"].get(g, {}).get("local", 0) for g in ["natural", "high"])


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Accuracy per operator: keyword vs LLM
# Split into sensitive / non-sensitive panels so the pattern is obvious.
# ─────────────────────────────────────────────────────────────────────────────
def fig_accuracy_comparison(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    llm_by_name = {o["op_name"]: o for o in llm_ops}

    sens_ops = [o for o in kw_ops if o["sensitive_query"]]
    ns_ops   = [o for o in kw_ops if not o["sensitive_query"]]

    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(15, 5))

    for ax, ops, panel_title, correct_dest in [
        (ax_s, sens_ops, "Sensitive queries  (PII records should → local)",      "local"),
        (ax_n, ns_ops,   "Non-sensitive queries  (PII records should → cloud_anon)", "cloud_anon"),
    ]:
        names   = [_label(o["op_name"]) for o in ops]
        kw_acc  = [o["accuracy"] * 100 for o in ops]
        llm_acc = [llm_by_name[o["op_name"]]["accuracy"] * 100 for o in ops]

        x = np.arange(len(ops))
        w = 0.32

        bars_kw  = ax.bar(x - w/2, kw_acc,  width=w, color=C_KW,  label="Keyword",        alpha=0.87)
        bars_llm = ax.bar(x + w/2, llm_acc, width=w, color=C_LLM, label="LLM (llama3.2)", alpha=0.87)

        ax.axhline(100, color="gray", lw=0.8, linestyle="--", alpha=0.5, label="Perfect (100%)")

        for bar, acc in zip(list(bars_kw) + list(bars_llm),
                            kw_acc + llm_acc):
            h = bar.get_height()
            color = bar.get_facecolor()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0, 120)
        ax.set_ylabel("Routing accuracy (%)")
        ax.set_title(panel_title, fontsize=9.5, pad=10)
        ax.legend(fontsize=8.5)

        # Shade operators where both methods fail (acc < 50 for both)
        for i, (ka, la) in enumerate(zip(kw_acc, llm_acc)):
            if ka < 50 and la < 50:
                ax.axvspan(i - 0.5, i + 0.5, color="#ffdddd", alpha=0.35, zorder=0)
            elif ka < 50 or la < 50:
                ax.axvspan(i - 0.5, i + 0.5, color="#fff3cc", alpha=0.35, zorder=0)

    fig.suptitle(
        "Q3 — Intent Detection Accuracy: Keyword vs LLM  (14,566 records, Presidio backend)\n"
        "Red shading = both methods fail  |  Yellow shading = one method fails",
        fontsize=10, y=1.04,
    )
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig1_accuracy_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Privacy leakage on sensitive operators
# For each sensitive op × method: safe (→ local) vs leaked (→ cloud/anon).
# This is the most important safety figure.
# ─────────────────────────────────────────────────────────────────────────────
def fig_privacy_leakage(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    sens_kw = [o for o in kw_ops if o["sensitive_query"]]
    llm_by_name = {o["op_name"]: o for o in llm_ops}

    labels     = [_label(o["op_name"]) for o in sens_kw]
    kw_local   = [_pii_local(o) for o in sens_kw]
    kw_leaked  = [_pii_leaked(o) for o in sens_kw]
    llm_ops_s  = [llm_by_name[o["op_name"]] for o in sens_kw]
    llm_local  = [_pii_local(o) for o in llm_ops_s]
    llm_leaked = [_pii_leaked(o) for o in llm_ops_s]
    totals     = [l + lk for l, lk in zip(kw_local, kw_leaked)]

    x = np.arange(len(sens_kw))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Keyword bars
    ax.bar(x - w/2, kw_local,  width=w, color="#2ca02c", label="Keyword → local (safe)",         alpha=0.87)
    ax.bar(x - w/2, kw_leaked, width=w, color=C_LEAK,    label="Keyword → cloud/anon (leakage)", alpha=0.87,
           bottom=kw_local)

    # LLM bars
    ax.bar(x + w/2, llm_local,  width=w, color="#98df8a",  label="LLM → local (safe)",          alpha=0.87)
    ax.bar(x + w/2, llm_leaked, width=w, color="#ff9896",  label="LLM → cloud/anon (leakage)",  alpha=0.87,
           bottom=llm_local)

    # Leakage rate annotations
    for i, (kl, ll_val, tot) in enumerate(zip(kw_leaked, llm_leaked, totals)):
        offset = tot * 0.02
        if kl == 0:
            ax.text(i - w/2, kw_local[i] + offset, "✓ 0%", ha="center", fontsize=8,
                    color="#2ca02c", fontweight="bold")
        else:
            ax.text(i - w/2, tot + offset, f"{kl/tot*100:.0f}%\nleak",
                    ha="center", fontsize=7.5, color=C_LEAK, fontweight="bold")
        if ll_val == 0:
            ax.text(i + w/2, llm_local[i] + offset, "✓ 0%", ha="center", fontsize=8,
                    color="#2ca02c", fontweight="bold")
        else:
            ax.text(i + w/2, tot + offset, f"{ll_val/tot*100:.0f}%\nleak",
                    ha="center", fontsize=7.5, color="#c5352b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("PII records (natural + high groups,  n = 12,566 total)")
    ax.set_title(
        "Q3 — Privacy Leakage on Sensitive Operators\n"
        "PII records NOT routed to local model (should be 0 for correct routing)",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    _save(fig, out_dir, "q3_fig2_privacy_leakage")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Full routing distribution: keyword vs LLM, horizontal stacked bars
# Sensitive operators on top, non-sensitive below, separated by dashed line.
# ─────────────────────────────────────────────────────────────────────────────
def fig_routing_distribution(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    ops_sorted = _sort_ops(kw_ops)
    llm_by_name = {o["op_name"]: o for o in llm_ops}

    n_ops = len(ops_sorted)
    n_sensitive = sum(1 for o in ops_sorted if o["sensitive_query"])

    fig, (ax_kw, ax_llm) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, method_label, ops_source, alpha in [
        (ax_kw,  "Keyword method",       ops_sorted,                               0.87),
        (ax_llm, "LLM method (llama3.2)", [llm_by_name[o["op_name"]] for o in ops_sorted], 0.87),
    ]:
        labels = [_label(o["op_name"]) for o in ops_sorted]
        local  = np.array([o["n_local"]             for o in ops_source])
        anon   = np.array([o["n_cloud_anonymized"]   for o in ops_source])
        cloud  = np.array([o["n_cloud"]              for o in ops_source])

        y = np.arange(n_ops)
        ax.barh(y, local, color=C_LOCAL, alpha=alpha, label="→ local")
        ax.barh(y, anon,  left=local,        color=C_ANON,  alpha=alpha, label="→ cloud + anonymized")
        ax.barh(y, cloud, left=local + anon, color=C_CLOUD, alpha=alpha, label="→ cloud as-is")

        # Background shading: red tint for sensitive, blue tint for non-sensitive
        for i, op in enumerate(ops_sorted):
            color = "#fff0f0" if op["sensitive_query"] else "#f0f4ff"
            ax.axhspan(i - 0.5, i + 0.5, color=color, alpha=0.5, zorder=0)

        # Divider between sensitive and non-sensitive
        ax.axhline(n_sensitive - 0.5, color="gray", lw=1.2, linestyle="--")
        ax.text(500, n_sensitive - 0.45, "sensitive ↑  |  ↓ non-sensitive",
                fontsize=7.5, color="gray", va="bottom")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Operator calls  (14,566 per operator)")
        ax.set_title(method_label, fontsize=10, pad=8)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Q3 — Routing Destination per Operator: Keyword vs LLM", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig3_routing_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Where the methods disagree and why
# Four-panel breakdown: (sensitive / non-sensitive) × (keyword wins / LLM wins / both fail)
# ─────────────────────────────────────────────────────────────────────────────
def fig_disagreement_summary(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    llm_by_name = {o["op_name"]: o for o in llm_ops}

    categories = {
        "Both correct\n(100% accuracy)":     [],
        "Keyword only correct\n(LLM fails)": [],
        "LLM only correct\n(Keyword fails)": [],
        "Both fail\n(< 50% accuracy)":       [],
    }

    for op in kw_ops:
        name = op["op_name"]
        kw_acc  = op["accuracy"]
        llm_acc = llm_by_name[name]["accuracy"]
        kw_ok   = kw_acc >= 0.95
        llm_ok  = llm_acc >= 0.95

        if kw_ok and llm_ok:
            categories["Both correct\n(100% accuracy)"].append((name, kw_acc, llm_acc, op["sensitive_query"]))
        elif kw_ok and not llm_ok:
            categories["Keyword only correct\n(LLM fails)"].append((name, kw_acc, llm_acc, op["sensitive_query"]))
        elif llm_ok and not kw_ok:
            categories["LLM only correct\n(Keyword fails)"].append((name, kw_acc, llm_acc, op["sensitive_query"]))
        else:
            categories["Both fail\n(< 50% accuracy)"].append((name, kw_acc, llm_acc, op["sensitive_query"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    cat_colors = {
        "Both correct\n(100% accuracy)":     "#2ca02c",
        "Keyword only correct\n(LLM fails)": "#2196F3",
        "LLM only correct\n(Keyword fails)": "#9467bd",
        "Both fail\n(< 50% accuracy)":       "#d62728",
    }

    for ax, (cat_label, entries) in zip(axes.flat, categories.items()):
        if not entries:
            ax.text(0.5, 0.5, "No operators", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(f"{cat_label}  (n=0)", fontsize=9)
            ax.axis("off")
            continue

        names   = [_label(e[0]) for e in entries]
        kw_accs = [e[1] * 100 for e in entries]
        ll_accs = [e[2] * 100 for e in entries]
        sens    = [e[3] for e in entries]

        y = np.arange(len(entries))
        w = 0.3
        col = cat_colors[cat_label]

        bars_kw = ax.barh(y - w/2, kw_accs, height=w, color=C_KW,  alpha=0.85, label="Keyword")
        bars_ll = ax.barh(y + w/2, ll_accs, height=w, color=C_LLM, alpha=0.85, label="LLM")

        for i, s in enumerate(sens):
            marker = "S" if s else "N"
            mcolor = "#8c1a11" if s else "#1a5276"
            ax.text(-3, i, marker, ha="right", va="center", fontsize=8,
                    color=mcolor, fontweight="bold")

        ax.axvline(100, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_xlim(0, 115)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"{cat_label}  (n={len(entries)})", fontsize=9, color=col, pad=6)
        ax.legend(fontsize=7.5)

        # S = sensitive, N = non-sensitive legend
        ax.text(0.98, 0.02, "S = sensitive query   N = non-sensitive",
                transform=ax.transAxes, fontsize=6.5, ha="right", color="gray")

    fig.suptitle(
        "Q3 — Method Disagreement: Where Keyword and LLM Agree or Diverge\n"
        "Threshold: ≥ 95% accuracy = correct",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig4_disagreement")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Quality savings on non-sensitive operators
# ─────────────────────────────────────────────────────────────────────────────
def fig_quality_savings(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    ns_kw = [o for o in kw_ops if not o["sensitive_query"]]
    llm_by_name = {o["op_name"]: o for o in llm_ops}

    labels      = [_label(o["op_name"]) for o in ns_kw]
    two_way     = [o["two_way_local"]  for o in ns_kw]
    kw_savings  = [o["quality_savings"] for o in ns_kw]
    llm_savings = [llm_by_name[o["op_name"]]["quality_savings"] for o in ns_kw]
    kw_lost     = [tw - ks for tw, ks in zip(two_way, kw_savings)]
    llm_lost    = [tw - ls for tw, ls in zip(two_way, llm_savings)]

    x = np.arange(len(ns_kw))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w,   two_way,     width=w, color="#aec7e8", alpha=0.7,  label="Naive two-way (baseline: all PII → local)")
    ax.bar(x,       kw_savings,  width=w, color=C_KW,      alpha=0.87, label="Keyword: calls saved to cloud_anon")
    ax.bar(x + w,   llm_savings, width=w, color=C_LLM,     alpha=0.87, label="LLM: calls saved to cloud_anon")

    # Annotate how many each method keeps unnecessarily local
    for i, (kl, ll_val) in enumerate(zip(kw_lost, llm_lost)):
        if kl > 0:
            ax.annotate(f"{kl} still local\n(quality lost)",
                        xy=(i, kw_savings[i]), xytext=(i - 0.35, kw_savings[i] + two_way[i] * 0.06),
                        fontsize=6.5, color=C_KW, ha="center")
        if ll_val > 0:
            ax.annotate(f"{ll_val} still local\n(quality lost)",
                        xy=(i + w, llm_savings[i]), xytext=(i + w + 0.2, llm_savings[i] + two_way[i] * 0.06),
                        fontsize=6.5, color=C_LLM, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Calls rescued from local → cloud_anonymized\n(cloud quality at no extra privacy cost)")
    ax.set_title(
        "Q3 — Quality Savings on Non-Sensitive Operators\n"
        "Higher = more calls get cloud quality instead of degraded local quality",
        fontsize=10,
    )
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig5_quality_savings")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q3 result figures")
    parser.add_argument("--results", default="data\\q3_results_both.json",
                        help="Path to results JSON (default: data/q3_results_both.json)")
    parser.add_argument("--out", default="data\\figures\\q3_both",
                        help="Output directory (default: data/figures/q3_both)")
    args = parser.parse_args()

    results_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", args.results)
        if not os.path.isabs(args.results) else args.results
    )
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", args.out)
        if not os.path.isabs(args.out) else args.out
    )

    print(f"Loading {results_path} ...")
    data = load(results_path)
    intent = data.get("intent", "keyword")
    n_total = data.get("sample_per_group") or "all"
    print(f"  intent={intent}  backend={data['backend']}  records={n_total}\n")

    kw_ops  = get_operators(data, "keyword")
    llm_ops = get_operators(data, "llm") if intent == "both" else kw_ops

    print("Fig 1 — accuracy comparison (keyword vs LLM, split by query type)")
    fig_accuracy_comparison(kw_ops, llm_ops, out_dir)

    print("Fig 2 — privacy leakage on sensitive operators")
    fig_privacy_leakage(kw_ops, llm_ops, out_dir)

    print("Fig 3 — full routing distribution (horizontal stacked bars)")
    fig_routing_distribution(kw_ops, llm_ops, out_dir)

    print("Fig 4 — where methods agree / disagree (2×2 panel)")
    fig_disagreement_summary(kw_ops, llm_ops, out_dir)

    print("Fig 5 — quality savings on non-sensitive operators")
    fig_quality_savings(kw_ops, llm_ops, out_dir)

    print(f"\nAll figures written to {out_dir}\\")


if __name__ == "__main__":
    main()
