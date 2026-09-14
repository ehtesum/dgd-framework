"""
DGD Benchmark Evaluator & Statistical Analytics Engine.

Calculates paired token savings, latency speedup, decision optimality,
and generates summary tables for academic publication.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class DGDEvaluator:
    """Computes empirical statistics and comparative tables from evaluation records."""

    def __init__(self, db_path: str = "dgd_evaluation.db"):
        self.db_path = db_path

    def load_dataframe(self) -> pd.DataFrame:
        """Loads evaluation records from SQLite."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM dgd_comparison", conn)
        conn.close()
        return df

    def compute_summary(self) -> Dict[str, Any]:
        """Calculates global paired evaluation statistics."""
        df = self.load_dataframe()
        if df.empty:
            return {"error": "Database is empty."}

        piv = df.pivot_table(
            index=["task_id", "model", "domain"],
            columns="architecture",
            values=["total_tokens", "is_optimal", "latency_sec", "r_conf"]
        ).dropna()
        piv.columns = ["_".join(c) for c in piv.columns]

        base_tok = float(piv["total_tokens_Baseline_Qu_Zhai"].sum())
        dgd_tok = float(piv["total_tokens_Proposed_DGD"].sum())
        net_saved = base_tok - dgd_tok
        savings_pct = (net_saved / max(1.0, base_tok)) * 100

        base_acc = float(piv["is_optimal_Baseline_Qu_Zhai"].mean()) * 100
        dgd_acc = float(piv["is_optimal_Proposed_DGD"].mean()) * 100

        base_lat = float(piv["latency_sec_Baseline_Qu_Zhai"].mean())
        dgd_lat = float(piv["latency_sec_Proposed_DGD"].mean())
        speedup_pct = ((base_lat - dgd_lat) / max(0.001, base_lat)) * 100

        return {
            "total_records": len(df),
            "paired_trials": len(piv),
            "total_baseline_tokens": int(base_tok),
            "total_dgd_tokens": int(dgd_tok),
            "net_tokens_saved": int(net_saved),
            "token_savings_pct": round(savings_pct, 2),
            "base_accuracy_pct": round(base_acc, 2),
            "dgd_accuracy_pct": round(dgd_acc, 2),
            "base_avg_latency_sec": round(base_lat, 2),
            "dgd_avg_latency_sec": round(dgd_lat, 2),
            "latency_speedup_pct": round(speedup_pct, 1),
        }

    def print_summary_table(self):
        """Prints formatted terminal table of benchmark results."""
        stats = self.compute_summary()
        if "error" in stats:
            print(f"[!] {stats['error']}")
            return

        print("=" * 75)
        print("          DIAGNOSTIC-GUIDED DELIBERATION (DGD) BENCHMARK SUMMARY")
        print("=" * 75)
        print(f"[*] Total Database Records:      {stats['total_records']:,}")
        print(f"[*] Paired Comparative Trials:   {stats['paired_trials']:,}")
        print("-" * 75)
        print(f"[*] Baseline Total Tokens:       {stats['total_baseline_tokens']:,} tokens")
        print(f"[*] Proposed DGD Total Tokens:   {stats['total_dgd_tokens']:,} tokens")
        print(f"[*] Net Compute Saved:           {stats['net_tokens_saved']:,} tokens (-{stats['token_savings_pct']}%)")
        print(f"[*] Decision Optimality:         {stats['dgd_accuracy_pct']}% (DGD) vs {stats['base_accuracy_pct']}% (Baseline)")
        print(f"[*] Mean Inference Latency:      {stats['dgd_avg_latency_sec']}s vs {stats['base_avg_latency_sec']}s (+{stats['latency_speedup_pct']}% faster)")
        print("=" * 75)
