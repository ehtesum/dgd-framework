"""
Unit tests for Diagnostic-Guided Deliberation (DGD) Core Components.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dgd import DGDController, EDVTracker, CommitmentHorizonDetector


def test_edv_tracker_entropy():
    tracker = EDVTracker(["H1", "H2"])
    # Maximum entropy for 2 uniform hypotheses is log2(2) = 1.0 bit
    h = tracker.shannon_entropy(tracker.priors)
    assert abs(h - 1.0) < 1e-5


def test_edv_tracker_discriminative_value():
    tracker = EDVTracker(["H1", "H2"])
    # Perfect discriminator: P(Y=+ | H1) = 1.0, P(Y=+ | H2) = 0.0
    edv, p_pos, p_neg = tracker.calculate_edv([1.0, 0.0])
    assert abs(edv - 1.0) < 1e-5
    assert tracker.is_discriminative_action(edv) is True


def test_commitment_detector():
    detector = CommitmentHorizonDetector(["H1", "H2"])
    conf_text = "Clearly H1 is right. This proves that H1 is true. Therefore H1 naturally holds."
    res = detector.evaluate_trajectory(conf_text)
    assert res["r_conf"] > 0.70
    assert res["is_captured"] is True


def test_controller_prompt_construction():
    controller = DGDController(["H1", "H2"], domain="clinical_medicine")
    dgd_cfg = controller.construct_dgd_prompt("Patient with chest pain.")
    assert "[DGD Epistemic Directive]" in dgd_cfg["prompt"]
    assert "[STOP]" in dgd_cfg["stop"]
    assert dgd_cfg["max_tokens"] == 45


def test_controller_choice_parsing():
    controller = DGDController(["H1", "H2"])
    assert controller.parse_decision("Option 1\nRationale here.") == "Option 1"
    assert controller.parse_decision("I recommend Option 2.") == "Option 2"
    assert controller.parse_decision("No option mentioned") == "Unknown"
