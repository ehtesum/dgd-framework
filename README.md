# Diagnostic-Guided Deliberation (DGD)

Epistemic entropy steering and diagnostic controllers for Large Language Model reasoning under hypothesis competition.

## Overview

When LLMs reason through complex tasks with multiple competing hypotheses (such as differential medical diagnosis, distributed system debugging, or abductive deduction), unconstrained test-time compute scaling often exhibits an early confirmation bias. Models tend to anchor on a single candidate early in the generation chain and spend subsequent reasoning tokens rationalizing that anchor, rather than conducting tests that actively discriminate between competing alternatives.

Diagnostic-Guided Deliberation (DGD) addresses this by introducing an information-theoretic controller that tracks the **Expected Diagnostic Value (EDV)** across generation steps. When the controller detects that a reasoning trajectory has entered a low-entropy confirmatory loop, it interrupts the rollout with an epistemic falsification sentinel to force evaluation of rival hypotheses before committing to a final decision.

## Method

### Expected Diagnostic Value (EDV)
At step $t$, a reasoning action $a_t$ has an Expected Diagnostic Value defined as the conditional mutual information between the hypothesis set $H$ and prospective evidence $Y$:

$$\text{EDV}(a_t) = I(H; Y \mid C, a_t) = H(H \mid C) - \mathbb{E}_{y \sim P(Y \mid C, a_t)} \big[ H(H \mid C, a_t, y) \big]$$

* **Discriminative action ($\text{EDV} > 0$):** Partitions the hypothesis space (an observation or test that confirms one candidate while ruling out alternatives).
* **Confirmatory action ($\text{EDV} \approx 0$):** Elaborates narrative details for an already favored hypothesis without changing the posterior probability distribution over candidates.

### Controller Dynamics
1. **Rationalization Ratio ($R_{\text{conf}}$):** Tracks the ratio of confirmatory reasoning steps to discriminative steps over the trajectory.
2. **Commitment Sentinel:** Detects the token horizon where self-generated text starts acting as an attractor basin, preventing hypothesis switching.
3. **Falsification Intervention:** Injects lightweight steering prompts requiring a decisive discriminator (a specific condition that would falsify the current leading candidate) and applies early stopping sentinels to reduce token waste.

## Evaluation

Evaluations on counterfactually permuted tasks from DC-Bench:

### Domain Breakdown

| Evaluation Domain | Trials ($N$) | Baseline Tokens | DGD Tokens | Token Reduction | Diagnostic Accuracy | Latency (mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clinical Medicine | 5,027 | 1,544,008 | 1,379,818 | -10.6% | 90.7% | 1.12s vs 1.75s |
| Formal Logic | 2,242 | 624,100 | 511,767 | -18.0% | 96.3% | 0.80s vs 1.93s |
| Software Systems | 6,277 | 2,036,609 | 1,837,234 | -9.8% | 60.0% | 0.98s vs 1.49s |
| **Total** | **13,546** | **4,204,717** | **3,728,819** | **-11.3%** | **77.4%** | **1.00s vs 1.66s** |

### Model Generalization

| Model | Parameter Scale | Provider | Token Reduction | Accuracy (Baseline vs DGD) |
| :--- | :---: | :---: | :---: | :---: |
| Mistral-NeMo | 12B | Mistral AI | -10.9% | 89.0% vs 93.9% |
| Qwen-2.5 | 27B | Groq | -9.2% | 75.5% vs 75.8% |
| GPT-OSS | 120B | Groq | -14.1% | Parity |
| GPT-OSS | 20B | Groq | -14.2% | Parity |

## Getting Started

### Installation
Python 3.10 or higher is required.

```bash
git clone https://github.com/ehtesum/dgd-framework.git
cd dgd-framework
pip install -e .
```

### Environment Configuration
Copy `.env.example` and add your API key(s) (Groq, Mistral, or OpenAI-compatible endpoint):

```bash
cp .env.example .env
```

### Running Experiments
Run the sample demonstration on 3 test scenarios:

```bash
python scripts/run_benchmark.py --sample
```

Run domain evaluations:

```bash
# Clinical evaluation on Mistral-NeMo
python scripts/run_benchmark.py --domain clinical --limit 50 --model open-mistral-nemo

# Formal logic evaluation
python scripts/run_benchmark.py --domain logic --limit 50

# Software root-cause evaluation
python scripts/run_benchmark.py --domain software --limit 50
```

Plot evaluation figures:

```bash
python scripts/plot_results.py dgd_benchmark.db
```

## Python API

You can also use the controller directly in custom pipelines:

```python
from dgd import DGDController, EDVTracker

hypotheses = [
    "H1: Acute Pulmonary Embolism",
    "H2: Acute Pericarditis"
]

# Initialize controller with competing hypothesis set
controller = DGDController(hypotheses, domain="clinical_medicine")

# Generate steering prompt and stopping parameters
config = controller.construct_dgd_prompt(
    "Patient presents with acute pleuritic chest pain and tachycardia."
)

# Calculate EDV for candidate diagnostic actions
tracker = EDVTracker(hypotheses)
edv, post_pos, post_neg = tracker.calculate_edv([0.95, 0.05])
print(f"EDV: {edv:.3f} bits | Discriminative: {tracker.is_discriminative_action(edv)}")
```

## Project Structure

```
dgd-framework/
├── src/dgd/
│   ├── controller.py      # DGD prompt constructor & sentinels
│   ├── edv_tracker.py     # Expected Diagnostic Value & entropy calculations
│   ├── commitment.py      # Commitment horizon tracking & R_conf metrics
│   ├── runner.py          # Model API execution runner
│   └── evaluator.py       # Metrics and latency analysis
├── scripts/
│   ├── run_benchmark.py   # CLI evaluation runner
│   ├── generate_dataset.py# Synthetic & counterfactual scenario generator
│   └── plot_results.py    # Plotting routines
├── data/                  # Benchmark scenario files (JSONL)
└── tests/                 # Unit tests
```

## Tests

Run tests using pytest:

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{jannat2026deliberation,
  title   = {The Deliberation Trap: When Test-Time Compute Scaling Amplifies Confirmatory Bias Under Hypothesis Competition},
  author  = {Jannat and Research Contributors},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
