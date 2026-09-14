#!/usr/bin/env python3
"""
Generates counterfactual evaluation tasks for Diagnostic-Guided Deliberation (DC-Bench).
"""

import os
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_sample_scenarios(n: int = 10):
    scenarios = []

    # 1. Clinical Medicine
    scenarios.append({
        "task_id": "CLIN-0001_varA",
        "domain": "clinical_medicine",
        "variant": "varA_canonical",
        "context": "A 48-year-old male with acute pleuritic chest pain and dyspnea. Vital signs show tachycardia (HR 115 bpm) and tachypnea.",
        "first_hyp": "H1: Acute Pulmonary Embolism",
        "second_hyp": "H2: Acute Pericarditis",
        "prompt": "Patient presents with acute pleuritic chest pain and tachycardia.\nH1: Acute Pulmonary Embolism\nH2: Acute Pericarditis\nWhich diagnostic test has higher Expected Diagnostic Value (EDV)?\nOption 1: CT Pulmonary Angiogram\nOption 2: Transthoracic Echocardiogram with PR-segment analysis",
        "optimal_choice": "Option 1"
    })

    # 2. Software Systems
    scenarios.append({
        "task_id": "SOFT-0001_varA",
        "domain": "software_systems",
        "variant": "varA_canonical",
        "context": "Microservice [PaymentGateway] interacting with [PostgreSQL Replica] is exhibiting elevated p99 latency exceeding 4500ms.",
        "first_hyp": "H1: Thread pool exhaustion under connection burst in [PaymentGateway]",
        "second_hyp": "H2: TCP window stall due to socket buffer starvation connecting to [PostgreSQL Replica]",
        "prompt": "PaymentGateway latency spiked to 4500ms.\nH1: Thread pool exhaustion\nH2: TCP socket buffer starvation\nWhich diagnostic test has higher Expected Diagnostic Value (EDV)?\nOption 1: Capture thread dumps and OS thread state locks\nOption 2: Capture PCAP network packet traces on egress interface",
        "optimal_choice": "Option 1"
    })

    # 3. Formal Logic
    scenarios.append({
        "task_id": "LOGIC-0001_varA",
        "domain": "formal_logic",
        "variant": "varA_canonical",
        "context": "Given premises: P -> Q, Q -> R, ~R. Agent contemplates validity of P under falsification.",
        "first_hyp": "H1: P is True",
        "second_hyp": "H2: P is False",
        "prompt": "Premises: P -> Q, Q -> R, ~R.\nH1: P is True\nH2: P is False\nIdentify the decisive logical deduction to evaluate the hypothesis:\nOption 1: Modus tollens yields ~Q and consequently ~P\nOption 2: Assume P and infer Q",
        "optimal_choice": "Option 1"
    })

    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate DC-Bench Evaluation Tasks")
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "sample_scenarios.jsonl"), help="Output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scenarios = generate_sample_scenarios()

    with open(args.out, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[*] Generated {len(scenarios)} sample scenarios to: {args.out}")


if __name__ == "__main__":
    main()
