#!/usr/bin/env python3
"""
Q3 visualization: query-intent routing benchmark diagrams.

Handles both single-method JSON (intent=keyword/llm) and comparison
JSON (intent=both) produced by:
    benchmark_q3.py --intent both --out data/q3_results_both.json

Usage:
    .venv/Scripts/python demos/plot_q3.py
    .venv/Scripts/python demos/plot_q3.py --results data/q3_results_both.json --out data/figures/q3_both
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

# Hard-coded 12-query benchmark from data/llm_prompt_comparison.md
# Tuple: (query_text, query_type, ground_truth_sensitive, per_entity_correct, general_correct)
PROMPT_BENCHMARK = [
    ("Extract the SSN from the resume text.",   "explicit",      True,  False, True),
    ("Find the applicant's phone & email.",     "explicit",      True,  True,  True),
    ("Identify full name & personal contact.",  "explicit",      True,  False, True),
    ("Find the best way to contact applicant.", "paraphrased",   True,  False, False),
    ("Who wrote this resume? Background?",      "paraphrased",   True,  False, False),
    ("Find me applicants above age 30.",        "implicit",      True,  False, False),
    ("Is candidate likely US-based?",           "implicit",      True,  False, False),
    ("Does anything suggest fraud?",            "implicit",      True,  False, False),
    ("Summarize applicant's skills.",           "non-sensitive", False, True,  True),
    ("Identify primary industry category.",     "non-sensitive", False, True,  True),
    ("Rate quality of applicant's education.",  "non-sensitive", False, True,  True),
    ("Rate the seniority level.",               "non-sensitive", False, True,  True),
]


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
        print(f"  saved -> {path}")
    plt.close(fig)


def _pii_leaked(op: dict) -> int:
    """PII records (natural+high groups) routed to cloud or cloud_anonymized."""
    return sum(
        op["by_group"].get(g, {}).get("cloud_anonymized", 0) +
        op["by_group"].get(g, {}).get("cloud", 0)
        for g in ("natural", "high")
    )


def _pii_total(op: dict) -> int:
    return sum(
        op["by_group"].get(g, {}).get("total", 0)
        for g in ("natural", "high")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Accuracy: only operators where keyword and LLM diverge
# Shows the meaningful signal; drops all 100%/100% and identical-failure pairs.
# ─────────────────────────────────────────────────────────────────────────────
def fig_accuracy_variance(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    llm_by_name = {o["op_name"]: o for o in llm_ops}

    divergent = [
        o for o in kw_ops
        if abs(o["accuracy"] - llm_by_name[o["op_name"]]["accuracy"]) > 0.02
    ]
    sens_ops = [o for o in divergent if o["sensitive_query"]]
    ns_ops   = [o for o in divergent if not o["sensitive_query"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panel_specs = [
        (axes[0], sens_ops, "Sensitive operators  (should route PII → local)"),
        (axes[1], ns_ops,   "Non-sensitive operators  (PII ok → cloud_anonymized)"),
    ]

    for ax, ops, panel_title in panel_specs:
        if not ops:
            ax.text(0.5, 0.5, "No divergent operators in this group",
                    ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
            ax.set_title(panel_title, fontsize=10)
            ax.axis("off")
            continue

        names   = [_label(o["op_name"]) for o in ops]
        kw_acc  = np.array([o["accuracy"] * 100 for o in ops])
        llm_acc = np.array([llm_by_name[o["op_name"]]["accuracy"] * 100 for o in ops])

        y = np.arange(len(ops))
        h = 0.33

        bars_kw  = ax.barh(y + h / 2, kw_acc,  height=h, color=C_KW,  alpha=0.87, label="Keyword")
        bars_llm = ax.barh(y - h / 2, llm_acc, height=h, color=C_LLM, alpha=0.87, label="LLM (llama3.2)")

        # Value labels right-aligned inside bars (skip if bar too short)
        for bar, val in zip(list(bars_kw) + list(bars_llm), list(kw_acc) + list(llm_acc)):
            bw = bar.get_width()
            by = bar.get_y() + bar.get_height() / 2
            if bw > 8:
                ax.text(bw - 1.5, by, f"{val:.0f}%",
                        ha="right", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
            else:
                ax.text(bw + 1.5, by, f"{val:.0f}%",
                        ha="left", va="center", fontsize=8.5,
                        color=bar.get_facecolor())

        ax.axvline(100, color="gray", lw=0.8, linestyle="--", alpha=0.5, label="100%")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(0, 118)
        ax.set_xlabel("Routing accuracy (%)", fontsize=9)
        ax.set_title(panel_title, fontsize=9.5, pad=8)
        ax.legend(fontsize=8.5, loc="lower right")

        # Faint row background
        for i, op in enumerate(ops):
            c = "#fff0f0" if op["sensitive_query"] else "#f0f4ff"
            ax.axhspan(i - 0.5, i + 0.5, color=c, alpha=0.4, zorder=0)

    fig.suptitle(
        "Q3 — Intent Detection Accuracy: Keyword vs LLM\n"
        "Showing only operators where accuracy differs by > 2%  (14,566 records, Presidio backend)",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig1_accuracy_variance")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Case study: extract_ssn routing by PII-density group
# The clearest example of keyword vs LLM divergence on an explicit PII operator.
# ─────────────────────────────────────────────────────────────────────────────
def fig_extract_ssn_case_study(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    kw_op  = next(o for o in kw_ops  if o["op_name"] == "extract_ssn")
    llm_op = next(o for o in llm_ops if o["op_name"] == "extract_ssn")

    groups       = ["none", "low", "natural", "high"]
    group_labels = ["none\n(no PII)", "low\n(sparse)", "natural\n(implicit PII)", "high\n(dense PII)"]

    def _stack(op):
        local = np.array([op["by_group"][g]["local"]            for g in groups])
        anon  = np.array([op["by_group"][g]["cloud_anonymized"] for g in groups])
        cloud = np.array([op["by_group"][g]["cloud"]            for g in groups])
        return local, anon, cloud

    kw_local,  kw_anon,  kw_cloud  = _stack(kw_op)
    llm_local, llm_anon, llm_cloud = _stack(llm_op)

    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Keyword (full opacity)
    ax.bar(x - w / 2, kw_local,  width=w, color=C_LOCAL, alpha=0.87,
           label="→ local (safe)")
    ax.bar(x - w / 2, kw_anon,   width=w, color=C_ANON,  alpha=0.87,
           label="→ cloud + anonymized", bottom=kw_local)
    ax.bar(x - w / 2, kw_cloud,  width=w, color=C_CLOUD, alpha=0.87,
           label="→ cloud as-is", bottom=kw_local + kw_anon)

    # LLM (lighter — same legend entries, no duplicate labels)
    ax.bar(x + w / 2, llm_local, width=w, color=C_LOCAL, alpha=0.45)
    ax.bar(x + w / 2, llm_anon,  width=w, color=C_ANON,  alpha=0.45, bottom=llm_local)
    ax.bar(x + w / 2, llm_cloud, width=w, color=C_CLOUD, alpha=0.45,
           bottom=llm_local + llm_anon)

    # Method labels above bar clusters
    totals_kw  = kw_local  + kw_anon  + kw_cloud
    totals_llm = llm_local + llm_anon + llm_cloud
    top = max(totals_kw.max(), totals_llm.max())
    for i in range(len(groups)):
        ax.text(i - w / 2, totals_kw[i]  + top * 0.015, "Keyword",
                ha="center", fontsize=8, color=C_KW,  fontweight="bold")
        ax.text(i + w / 2, totals_llm[i] + top * 0.015, "LLM",
                ha="center", fontsize=8, color=C_LLM, fontweight="bold")

    # Shade PII-present groups
    ax.axvspan(1.5, 3.5, color="#fff0d0", alpha=0.35, zorder=0)
    ax.text(2.5, top * 0.97,
            "PII-present groups\n(natural + high records\nshould route → local)",
            ha="center", va="top", fontsize=8, color="#8c4000", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8ee", ec="#e0a040", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Records routed", fontsize=9)
    ax.set_title(
        "Q3 — Case Study: extract_ssn  (explicit PII operator)\n"
        "Keyword correctly routes natural+high PII records to local;  "
        "LLM sends them to cloud_anonymized  [privacy risk]",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper left",
              title="Destination (Keyword = solid, LLM = faded)", title_fontsize=7.5)

    fig.tight_layout()
    _save(fig, out_dir, "q3_fig2_ssn_case_study")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Prompt strategy comparison (12-query benchmark sample)
# Heatmap showing per_entity vs general prompt correctness per query.
# Data hard-coded from data/llm_prompt_comparison.md (llama3.2 3B, temp=0).
# ─────────────────────────────────────────────────────────────────────────────
def fig_prompt_comparison(out_dir: str) -> None:
    type_order = ["explicit", "paraphrased", "implicit", "non-sensitive"]
    type_bg = {
        "explicit":      "#ffeaea",
        "paraphrased":   "#fff4e0",
        "implicit":      "#fffbe0",
        "non-sensitive": "#eafbea",
    }
    type_labels_right = {
        "explicit":      "Explicit\nsensitive",
        "paraphrased":   "Paraphrased\nsensitive",
        "implicit":      "Implicit\nsensitive",
        "non-sensitive": "Non-\nsensitive",
    }

    rows = sorted(PROMPT_BENCHMARK, key=lambda r: type_order.index(r[1]))
    n    = len(rows)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    C_CORRECT = "#2ca02c"
    C_WRONG   = "#d62728"

    for row_i, (query, qtype, _sensitive, pe_ok, gen_ok) in enumerate(rows):
        y = n - 1 - row_i

        ax.axhspan(y - 0.5, y + 0.5, color=type_bg[qtype], alpha=0.5, zorder=0)

        for col, ok in enumerate([pe_ok, gen_ok]):
            color = C_CORRECT if ok else C_WRONG
            ax.add_patch(plt.Rectangle((col, y - 0.42), 1, 0.84,
                                       color=color, alpha=0.82, zorder=1))
            ax.text(col + 0.5, y, "✓" if ok else "✗",
                    ha="center", va="center", fontsize=14,
                    color="white", fontweight="bold", zorder=2)

        ax.text(-0.12, y, query, ha="right", va="center", fontsize=7.5, style="italic")

    # Group dividers
    prev = rows[0][1]
    for row_i in range(1, n):
        if rows[row_i][1] != prev:
            y_div = n - row_i - 0.5
            ax.hlines(y_div, 0, 2, color="#666", lw=1.2, linestyle="--", zorder=3)
            prev = rows[row_i][1]

    # Type labels on right
    for qtype in type_order:
        idxs = [i for i, r in enumerate(rows) if r[1] == qtype]
        if idxs:
            y_mid = n - 1 - (idxs[0] + idxs[-1]) / 2
            ax.text(2.12, y_mid, type_labels_right[qtype],
                    ha="left", va="center", fontsize=7.5, color="#444")

    # Summary row
    pe_total  = sum(r[3] for r in rows)
    gen_total = sum(r[4] for r in rows)
    for col, total in enumerate([pe_total, gen_total]):
        ax.add_patch(plt.Rectangle((col, -0.92), 1, 0.74,
                                   color="#bbbbbb", alpha=0.85, zorder=1))
        ax.text(col + 0.5, -0.55, f"{total} / {n}",
                ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=2)
    ax.text(-0.12, -0.55, "Total correct:", ha="right", va="center", fontsize=8.5)

    # Column headers
    for col, label in enumerate(["per_entity\nprompt", "general\nprompt"]):
        ax.text(col + 0.5, n + 0.15, label,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xlim(-4.0, 3.2)
    ax.set_ylim(-1.35, n + 0.9)
    ax.axis("off")

    fig.suptitle(
        "Q3 — LLM Prompt Strategy Comparison  (12-query benchmark sample)\n"
        "llama3.2 3B · Ollama · temperature=0   |   ✓ correctly classified   ✗ wrong",
        fontsize=9.5, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig3_prompt_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Privacy leakage: lollipop chart, no inline text
# Each sensitive operator gets two lollipops (keyword / LLM) showing how many
# PII records (natural + high groups) were NOT routed to the local model.
# ─────────────────────────────────────────────────────────────────────────────
def fig_privacy_leakage_clean(kw_ops: list[dict], llm_ops: list[dict], out_dir: str) -> None:
    llm_by_name = {o["op_name"]: o for o in llm_ops}
    sens_ops    = [o for o in kw_ops if o["sensitive_query"]]

    labels     = [_label(o["op_name"]) for o in sens_ops]
    kw_leaked  = [_pii_leaked(o)                              for o in sens_ops]
    llm_leaked = [_pii_leaked(llm_by_name[o["op_name"]])      for o in sens_ops]
    totals     = [_pii_total(o)                               for o in sens_ops]

    n   = len(sens_ops)
    gap = 0.22
    y   = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i in range(n):
        kl = kw_leaked[i]
        ll = llm_leaked[i]
        tot = totals[i]

        yi_kw  = y[i] + gap
        yi_llm = y[i] - gap

        # Faint total-records reference bar
        ax.barh(y[i], tot, height=0.55, color="#e0e0e0", alpha=0.4, zorder=0)

        # Keyword lollipop
        if kl > 0:
            ax.hlines(yi_kw, 0, kl, color=C_SAFE, lw=2.5, alpha=0.9)
        ax.plot(kl, yi_kw, "o", color=C_SAFE, ms=9, zorder=3)

        # LLM lollipop
        if ll > 0:
            ax.hlines(yi_llm, 0, ll, color=C_LEAK, lw=2.5, alpha=0.9)
        ax.plot(ll, yi_llm, "o", color=C_LEAK, ms=9, zorder=3)

    # Row backgrounds
    for i in range(n):
        ax.axhspan(i - 0.5, i + 0.5, color="#fff0f0", alpha=0.25, zorder=0)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(
        "PII records (natural + high groups) routed to cloud or cloud_anonymized\n"
        "[target = 0 for perfect privacy;  grey bar = total PII records in those groups]",
        fontsize=8.5,
    )

    legend_elements = [
        mpatches.Patch(color=C_SAFE, alpha=0.9, label="Keyword — leaked PII records"),
        mpatches.Patch(color=C_LEAK, alpha=0.9, label="LLM — leaked PII records"),
        mpatches.Patch(color="#e0e0e0", alpha=0.7, label="Total PII records per operator"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc="lower right")

    ax.set_title(
        "Q3 — Privacy Leakage on Sensitive Operators\n"
        "PII records NOT routed to local model  (0 = perfect privacy)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "q3_fig4_privacy_leakage")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q3 result figures")
    parser.add_argument("--results", default="data\\q3_results_both.json",
                        help="Path to results JSON")
    parser.add_argument("--out", default="data\\figures\\q3_both",
                        help="Output directory")
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
    data   = load(results_path)
    intent = data.get("intent", "keyword")
    print(f"  intent={intent}  backend={data['backend']}\n")

    kw_ops  = get_operators(data, "keyword")
    llm_ops = get_operators(data, "llm") if intent == "both" else kw_ops

    print("Fig 1 — accuracy comparison (variance-only operators)")
    fig_accuracy_variance(kw_ops, llm_ops, out_dir)

    print("Fig 2 — extract_ssn case study (routing by PII group)")
    fig_extract_ssn_case_study(kw_ops, llm_ops, out_dir)

    print("Fig 3 — prompt strategy comparison (12-query benchmark sample)")
    fig_prompt_comparison(out_dir)

    print("Fig 4 — privacy leakage lollipop chart")
    fig_privacy_leakage_clean(kw_ops, llm_ops, out_dir)

    print(f"\nAll figures written to {out_dir}\\")


if __name__ == "__main__":
    main()
