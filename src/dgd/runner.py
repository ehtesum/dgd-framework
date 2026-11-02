"""
Multi-Model Comparative Runner.

Executes head-to-head empirical evaluations comparing:
1. Baseline Unconstrained Deliberation (Qu / Zhai 2026 paradigm)
2. Proposed Diagnostic-Guided Deliberation (DGD)
"""

import os
import time
import sqlite3
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from dgd.controller import DGDController


class ComparativeRunner:
    """Orchestrates API calls, comparative rollouts, and metrics recording."""

    def __init__(
        self,
        db_path: str = "dgd_evaluation.db",
        groq_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.db_path = db_path
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.mistral_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

        self._init_database()

    def _init_database(self):
        """Ensures the SQLite table schema exists."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dgd_comparison (
            run_id TEXT PRIMARY KEY,
            task_id TEXT,
            domain TEXT,
            variant TEXT,
            model TEXT,
            architecture TEXT,
            is_optimal INTEGER,
            chosen_option TEXT,
            optimal_choice TEXT,
            total_tokens INTEGER,
            r_conf REAL,
            tau_star INTEGER,
            intervention_triggered INTEGER,
            latency_sec REAL,
            reasoning_excerpt TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def call_llm(
        self,
        prompt: str,
        model: str = "open-mistral-nemo",
        max_tokens: int = 140,
        stop: Optional[list] = None
    ) -> Tuple[str, int, float]:
        """Calls the appropriate model endpoint."""
        t0 = time.time()

        # Engine selection
        if "mistral" in model.lower():
            client = OpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key=self.mistral_key
            )
        elif "groq" in model.lower() or any(m in model.lower() for m in ["qwen", "gpt-oss"]):
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_key
            )
        else:
            client = OpenAI(api_key=self.openai_key)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop
        )

        content = resp.choices[0].message.content or ""
        total_tokens = resp.usage.total_tokens if resp.usage else max_tokens
        latency = time.time() - t0
        return content, total_tokens, latency

    def evaluate_task(
        self,
        task: Dict[str, Any],
        model: str = "open-mistral-nemo"
    ) -> Dict[str, Any]:
        """
        Runs both Baseline and Proposed DGD on a given task scenario.
        """
        hypotheses = [task.get("first_hyp", "H1"), task.get("second_hyp", "H2")]
        domain = task.get("domain", "general")
        controller = DGDController(hypotheses, domain=domain)

        # 1. Baseline Run (Unconstrained Rollout)
        base_prompt = (
            f"{task['prompt']}\n\n"
            f"Concisely evaluate both candidate hypotheses. State Option 1 or Option 2 on the first line, "
            f"followed by a 2-sentence rationale."
        )
        base_text, base_tokens, base_lat = self.call_llm(base_prompt, model=model, max_tokens=140)
        base_choice = controller.parse_decision(base_text)
        base_optimal = 1 if base_choice == task["optimal_choice"] else 0
        eval_base = controller.evaluate_trajectory(base_text)

        # 2. Proposed DGD Run (Epistemic Steering + Sentinels)
        dgd_cfg = controller.construct_dgd_prompt(task["prompt"])
        dgd_text, dgd_tokens, dgd_lat = self.call_llm(
            dgd_cfg["prompt"],
            model=model,
            max_tokens=dgd_cfg["max_tokens"],
            stop=dgd_cfg["stop"]
        )
        clean_dgd_text = dgd_text.replace("[STOP]", "").strip()
        dgd_choice = controller.parse_decision(clean_dgd_text)
        dgd_optimal = 1 if dgd_choice == task["optimal_choice"] else 0
        eval_dgd = controller.evaluate_trajectory(clean_dgd_text)

        # Compute Comparative Metrics
        net_saved = base_tokens - dgd_tokens
        savings_pct = (net_saved / max(1, base_tokens)) * 100

        # Persist to SQLite
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO dgd_comparison (
            run_id, task_id, domain, variant, model, architecture, is_optimal,
            chosen_option, optimal_choice, total_tokens, r_conf, tau_star,
            intervention_triggered, latency_sec, reasoning_excerpt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{task['task_id']}_{model}_baseline", task["task_id"], domain, task.get("variant", "base"),
            model, "Baseline_Qu_Zhai", base_optimal, base_choice, task["optimal_choice"],
            base_tokens, eval_base["r_conf"], eval_base["tau_star"] or 0, 0, base_lat, base_text[:200]
        ))
        cur.execute("""
        INSERT OR REPLACE INTO dgd_comparison (
            run_id, task_id, domain, variant, model, architecture, is_optimal,
            chosen_option, optimal_choice, total_tokens, r_conf, tau_star,
            intervention_triggered, latency_sec, reasoning_excerpt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{task['task_id']}_{model}_dgd", task["task_id"], domain, task.get("variant", "base"),
            model, "Proposed_DGD", dgd_optimal, dgd_choice, task["optimal_choice"],
            dgd_tokens, eval_dgd["r_conf"], eval_dgd["tau_star"] or 0, 1, dgd_lat, clean_dgd_text[:200]
        ))
        conn.commit()
        conn.close()

        return {
            "task_id": task["task_id"],
            "model": model,
            "base_tokens": base_tokens,
            "dgd_tokens": dgd_tokens,
            "net_tokens_saved": net_saved,
            "token_savings_pct": round(savings_pct, 2),
            "base_optimal": base_optimal,
            "dgd_optimal": dgd_optimal,
            "base_lat": round(base_lat, 2),
            "dgd_lat": round(dgd_lat, 2)
        }
