"""
Diagnostic-Guided Deliberation (DGD) Controller.

Main steering engine replacing unconstrained token scaling with epistemic
entropy-maximizing intervention and early-stopping sentinels.
"""

import re
from typing import List, Dict, Any, Optional
from dgd.edv_tracker import EDVTracker
from dgd.commitment import CommitmentHorizonDetector


class DGDController:
    """Controller orchestrating epistemic steering during deliberation."""

    def __init__(self, hypotheses: List[str], domain: str = "general"):
        self.hypotheses = hypotheses
        self.domain = domain.lower()
        self.edv_tracker = EDVTracker(hypotheses)
        self.commitment_detector = CommitmentHorizonDetector(hypotheses)

    def is_advanced_domain(self) -> bool:
        """Checks if the scenario falls into formal logic or abductive forensics."""
        return any(k in self.domain for k in ["logic", "abductive"])

    def construct_dgd_prompt(self, base_prompt: str) -> Dict[str, Any]:
        """
        Transforms an open-ended reasoning query into an Epistemic Falsification Anchor.
        """
        is_adv = self.is_advanced_domain()

        if is_adv:
            # Option A Tight Falsification Anchor (<= 8 words) for Logic/Abductive
            directive = (
                f"{base_prompt}\n\n"
                f"[DGD Epistemic Directive]: Prioritize identifying the decisive operation with maximal "
                f"Expected Diagnostic Value (EDV) to decisively falsify one hypothesis and verify the rival.\n"
                f"Line 1: State Option 1 or Option 2\n"
                f"Line 2: Decisive observation ruling out rival hypothesis (<= 8 words). End with [STOP]."
            )
            max_tokens = 30
            stop_sentinels = ["[STOP]", "\n\n", "Option 1\n", "Option 2\n", "\nLine"]
        else:
            # Standard DGD Directive (<= 15 words) for Clinical / Software Systems
            directive = (
                f"{base_prompt}\n\n"
                f"[DGD Epistemic Directive]: Prioritize identifying the decisive operation with maximal "
                f"Expected Diagnostic Value (EDV) to decisively falsify one hypothesis and verify the rival.\n"
                f"Line 1: State Option 1 or Option 2\n"
                f"Line 2: Decisive falsification discriminator (<= 15 words). End with [STOP]."
            )
            max_tokens = 45
            stop_sentinels = ["[STOP]", "\n\n\n"]

        return {
            "prompt": directive,
            "max_tokens": max_tokens,
            "stop": stop_sentinels
        }

    def parse_decision(self, response_text: str) -> str:
        """Extracts the chosen option ('Option 1' or 'Option 2') robustly."""
        t = response_text.lower()
        # Direct line-start or strict match
        m1 = "option 1" in t
        m2 = "option 2" in t
        if m1 and not m2:
            return "Option 1"
        if m2 and not m1:
            return "Option 2"

        # Regex fallback for embedded phrases
        match = re.search(r'\b(choose|select|recommend|opt for|support|pick)?\s*option\s*([12])\b', t)
        if match:
            return f"Option {match.group(2)}"

        return "Unknown"

    def evaluate_trajectory(self, text: str) -> Dict[str, Any]:
        """Analyzes trajectory for confirmatory capture and commitment horizon."""
        return self.commitment_detector.evaluate_trajectory(text)
