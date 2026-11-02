# Diagnostic-Guided Deliberation (DGD)

Epistemic entropy steering and diagnostic controllers for language model reasoning under competing hypotheses.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)](tests/)

---

## Overview

When language models reason through tasks involving competing hypotheses (such as differential medical diagnosis, distributed system troubleshooting, or abductive reasoning), unconstrained test-time compute scaling frequently triggers an early confirmation bias. Models anchor on a single candidate early in the generation chain and spend subsequent tokens rationalizing that initial anchor, rather than seeking observations that actively discriminate between competing alternatives.

**Diagnostic-Guided Deliberation (DGD)** introduces an information-theoretic controller that monitors **Expected Diagnostic Value (EDV)** during rollout. When the controller detects a low-entropy confirmatory loop, it injects an epistemic falsification sentinel to force evaluation of rival hypotheses before committing to a final conclusion.

---

## Method

### Expected Diagnostic Value (EDV)

At reasoning step $t$, the Expected Diagnostic Value of an exploratory action $a_t$ is defined as the conditional mutual information between hypothesis set $H$ and prospective evidence $Y$:

$$\text{EDV}(a_t) = I(H; Y \mid C, a_t) = H(H \mid C) - \mathbb{E}_{y \sim P(Y \mid C, a_t)} \big[ H(H \mid C, a_t, y) \big]$$

* **Discriminative action ($\text{EDV} > 0$):** Partitions candidate space by testing a specific feature that confirms one hypothesis while ruling out alternatives.
* **Confirmatory action ($\text{EDV} \approx 0$):** Elaborates narrative details for an already favored hypothesis without altering posterior candidate probabilities.

### Controller Flow

1. **Rationalization Ratio ($R_{\text{conf}}$):** Tracks the proportion of confirmatory versus discriminative steps across the rollout.
2. **Commitment Sentinel:** Detects the token boundary where self-generated rationale creates an attractor basin, preventing hypothesis switching.
3. **Falsification Intervention:** Injects lightweight steering prompts requiring an explicit discriminator (a concrete condition that would falsify the leading candidate) and applies early stopping sentinels to curtail redundant tokens.

---

## Installation

```bash
git clone https://github.com/ehtesum/dgd-framework.git
cd dgd-framework

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

```python
from dgd import DGDController, EDVTracker

hypotheses = [
    "H1: Acute Pulmonary Embolism",
    "H2: Acute Pericarditis"
]

# 1. Initialize controller
controller = DGDController(hypotheses, domain="clinical_medicine")

# 2. Construct steering prompt with early-stopping sentinels
config = controller.construct_dgd_prompt(
    "Patient presents with acute pleuritic chest pain and tachycardia."
)

# 3. Calculate Expected Diagnostic Value for prospective clinical tests
tracker = EDVTracker(hypotheses)
edv, post_pos, post_neg = tracker.calculate_edv([0.95, 0.05])

print(f"EDV: {edv:.3f} bits")
print(f"Is Discriminative Action: {tracker.is_discriminative_action(edv)}")
```

---

## Repository Structure

```
dgd-framework/
├── src/
│   └── dgd/
│       ├── __init__.py
│       ├── controller.py        # Prompt construction and stopping sentinels
│       ├── edv_tracker.py       # Expected Diagnostic Value & entropy math
│       ├── commitment.py        # Commitment horizon & rationalization ratio tracking
│       ├── runner.py            # Model API execution with fallback handling
│       └── evaluator.py         # Empirical metrics and latency analyzers
├── scripts/
│   ├── run_benchmark.py         # Benchmark CLI runner
│   ├── generate_dataset.py      # Benchmark scenario generator
│   └── plot_results.py          # Summary visualizer
├── data/
│   └── dcbench_balanced.jsonl   # Curated multi-domain evaluation instances
├── results/
│   ├── fig_accuracy_vs_tokens.png
│   └── fig_pareto_frontier.png
├── tests/
│   └── test_dgd.py              # Unit tests for controller, EDV, and runner
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

## Evaluation Results

Tested on counterfactually permuted diagnostic scenarios across three domains (13,546 total trials):

### Domain Summary

| Domain | Trials ($N$) | Baseline Tokens | DGD Tokens | Token Reduction | Accuracy | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clinical Medicine | 5,027 | 1,544,008 | 1,379,818 | -10.6% | 90.7% | 1.12s vs 1.75s |
| Formal Logic | 2,242 | 624,100 | 511,767 | -18.0% | 96.3% | 0.80s vs 1.93s |
| Software Systems | 6,277 | 2,036,609 | 1,837,234 | -9.8% | 60.0% | 0.98s vs 1.49s |
| **Total** | **13,546** | **4,204,717** | **3,728,819** | **-11.3%** | **77.4%** | **1.00s vs 1.66s** |

### Multi-Model Efficiency

| Model | Parameter Scale | Provider | Token Delta | Accuracy Delta |
| :--- | :---: | :---: | :---: | :---: |
| Mistral-NeMo | 12B | Mistral AI | -10.9% | +4.9% (89.0% vs 93.9%) |
| Qwen-2.5 | 27B | Groq | -9.2% | Parity (75.5% vs 75.8%) |
| GPT-OSS | 120B | Groq | -14.1% | Parity |

---

## Running Tests

Run the test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
