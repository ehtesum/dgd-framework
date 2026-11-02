#!/usr/bin/env python3
"""
CLI entrypoint to run comparative benchmark evaluations.

Usage:
    python scripts/run_benchmark.py --sample
    python scripts/run_benchmark.py --data data/sample_scenarios.jsonl --model open-mistral-nemo
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dgd import ComparativeRunner, DGDEvaluator
from dotenv import load_dotenv

# Load .env if present
load_dotenv(REPO_ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="Run DGD vs Baseline Comparative Evaluation")
    parser.add_argument("--domain", choices=["sample", "clinical", "software", "logic", "abductive"], default="sample", help="Benchmark domain preset")
    parser.add_argument("--data", type=str, default=None, help="Custom path to evaluation JSONL dataset")
    parser.add_argument("--model", type=str, default="open-mistral-nemo", help="Model to evaluate (e.g., open-mistral-nemo, qwen/qwen3.8-27b, openai/gpt-oss-120b)")
    parser.add_argument("--db", type=str, default="dgd_benchmark.db", help="SQLite database output path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of scenarios to evaluate")
    parser.add_argument("--sample", action="store_true", help="Run a quick 3-sample demonstration test")
    args = parser.parse_args()

    domain_files = {
        "sample": str(REPO_ROOT / "data" / "sample_scenarios.jsonl"),
        "clinical": str(REPO_ROOT / "data" / "dc_clinical_4k.jsonl"),
        "software": str(REPO_ROOT / "data" / "dc_software_4k.jsonl"),
        "logic": str(REPO_ROOT / "data" / "dc_logic_4k.jsonl"),
        "abductive": str(REPO_ROOT / "data" / "dc_abductive_4k.jsonl"),
    }

    data_file = args.data if args.data else domain_files.get(args.domain, domain_files["sample"])

    print("=" * 75)
    print("      DIAGNOSTIC-GUIDED DELIBERATION (DGD) - COMPARATIVE BENCHMARK")
    print("=" * 75)
    print(f"[*] Benchmark Domain: {args.domain.upper()}")
    print(f"[*] Target Model:     {args.model}")
    print(f"[*] Dataset File:     {data_file}")
    print(f"[*] Database Output:  {args.db}")
    print("=" * 75 + "\n")

    if not os.path.exists(data_file):
        print(f"[!] Error: Dataset file '{data_file}' not found.")
        sys.exit(1)

    tasks = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if args.sample:
        tasks = tasks[:3]
        print(f"[*] Running 3 sample demonstration scenarios...\n")
    elif args.limit:
        tasks = tasks[:args.limit]
        print(f"[*] Evaluating up to {len(tasks)} scenarios...\n")

    runner = ComparativeRunner(db_path=args.db)

    for i, task in enumerate(tasks):
        s_id = task.get("task_id", f"TASK-{i+1:04d}")
        domain = task.get("domain", "general")
        print(f"[{i+1:02d}/{len(tasks):02d}] Evaluating {s_id} ({domain})...", end="", flush=True)

        try:
            res = runner.evaluate_task(task, model=args.model)
            opt_tag = "PASS" if res["dgd_optimal"] == 1 else "FAIL"
            print(f" | Base: {res['base_tokens']}t | DGD: {res['dgd_tokens']}t (-{res['token_savings_pct']}%) | Decision: {opt_tag}")
        except Exception as e:
            print(f" | [Notice]: {e} (Please verify your API key in .env)")

    print("\n" + "=" * 75)
    print("[*] Benchmark run completed successfully!")
    print("=" * 75)

    evaluator = DGDEvaluator(db_path=args.db)
    evaluator.print_summary_table()


if __name__ == "__main__":
    main()
