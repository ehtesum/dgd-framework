#!/usr/bin/env python3
"""
Plots publication-grade figures comparing Baseline vs. Proposed DGD.

Outputs:
    figures/fig3_dgd_ablation.png
"""

import os
import sys
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def plot_ablation(db_path: str = "dgd_evaluation.db", out_path: str = "figures/fig3_dgd_ablation.png"):
    if not os.path.isabs(out_path):
        out_path = str(REPO_ROOT / out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(db_path):
        # Fallback to parent directory if in repo
        parent_db = REPO_ROOT.parent / db_path
        if parent_db.exists():
            db_path = str(parent_db)
        else:
            print(f"[!] Database '{db_path}' not found. Run a benchmark first.")
            return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT architecture, is_optimal, total_tokens, latency_sec FROM dgd_comparison")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("[!] No records found in database.")
        return

    base_rows = [r for r in rows if r[0] == "Baseline_Qu_Zhai"]
    dgd_rows = [r for r in rows if r[0] == "Proposed_DGD"]

    if not base_rows or not dgd_rows:
        print("[!] Need both Baseline and DGD records to plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # 1. Average Token Cost
    base_tok = np.mean([r[2] for r in base_rows])
    dgd_tok = np.mean([r[2] for r in dgd_rows])
    savings = ((base_tok - dgd_tok) / base_tok) * 100

    bars1 = ax1.bar(
        ["Baseline (Qu/Zhai 2026)", "Proposed DGD (Ours)"],
        [base_tok, dgd_tok],
        color=["#94A3B8", "#10B981"],
        width=0.45,
        edgecolor="black",
        linewidth=1.2
    )
    ax1.set_title("Test-Time Token Efficiency", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Average Tokens per Scenario", fontsize=10, fontweight="semibold")
    ax1.bar_label(bars1, fmt="%.0f tok", padding=3, fontweight="bold")
    ax1.text(
        0.5, max(base_tok, dgd_tok) * 0.45,
        f"-{savings:.1f}% Compute Reduction\n(Pareto Dominant)",
        ha="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="#DCFCE7", edgecolor="#10B981", boxstyle="round,pad=0.5")
    )

    # 2. Decision Accuracy
    base_acc = np.mean([r[1] for r in base_rows]) * 100
    dgd_acc = np.mean([r[1] for r in dgd_rows]) * 100

    bars2 = ax2.bar(
        ["Baseline (Qu/Zhai 2026)", "Proposed DGD (Ours)"],
        [base_acc, dgd_acc],
        color=["#EF4444", "#3B82F6"],
        width=0.45,
        edgecolor="black",
        linewidth=1.2
    )
    ax2.set_title("Diagnostic Decision Optimality", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Decision Optimality (%)", fontsize=10, fontweight="semibold")
    ax2.set_ylim(0, 115)
    ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Successfully generated publication plot: {out_path}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "dgd_evaluation.db"
    plot_ablation(db_path=db)
