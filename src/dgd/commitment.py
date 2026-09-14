"""
Commitment Horizon (tau*) & Confirmatory Rationalization Detector.

Implements Law 1 (Compute-Dogmatism Law) and Law 2 (Commitment Horizon Phase Transition).
"""

import re
from typing import Dict, Any, List, Optional


class CommitmentHorizonDetector:
    """Detects premature commitment phase transitions and measures confirmatory bias."""

    # Lexical patterns signaling confirmation rationalization vs discriminative falsification
    CONFIRMATORY_PATTERNS = [
        r'\b(clearly|obviously|as established|evidently|undoubtedly)\b',
        r'\b(this supports|proves that|reinforces|validates|confirms)\b',
        r'\b(therefore|hence|naturally|it must be)\b',
    ]

    DISCRIMINATIVE_PATTERNS = [
        r'\b(conversely|on the other hand|however|alternatively|in contrast)\b',
        r'\b(falsif|rule out|discriminat|counter-evidence|incompatible with)\b',
        r'\b(test between|crucial distinction|decisive difference)\b',
    ]

    def __init__(self, hypotheses: List[str]):
        self.hypotheses = hypotheses

    def count_matches(self, text: str, patterns: List[str]) -> int:
        count = 0
        t_lower = text.lower()
        for pat in patterns:
            count += len(re.findall(pat, t_lower))
        return count

    def evaluate_trajectory(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a reasoning trajectory segment to compute R_conf and detect tau*.
        """
        conf_count = self.count_matches(text, self.CONFIRMATORY_PATTERNS)
        disc_count = self.count_matches(text, self.DISCRIMINATIVE_PATTERNS)

        # Rationalization Index R_conf = Confirmatory / (Confirmatory + Discriminative)
        total_signals = conf_count + disc_count
        if total_signals == 0:
            r_conf = 0.5
        else:
            r_conf = conf_count / total_signals

        # Commitment Horizon Tau* heuristic
        # If model exhibits >2 confirmatory signals before any discriminative test
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        cumulative_tokens = 0
        tau_star: Optional[int] = None

        c_t = 0
        d_t = 0
        for sent in sentences:
            words = sent.split()
            cumulative_tokens += len(words)
            c_t += self.count_matches(sent, self.CONFIRMATORY_PATTERNS)
            d_t += self.count_matches(sent, self.DISCRIMINATIVE_PATTERNS)

            if c_t >= 2 and d_t == 0 and tau_star is None:
                tau_star = cumulative_tokens

        return {
            "r_conf": round(r_conf, 3),
            "confirmatory_count": conf_count,
            "discriminative_count": disc_count,
            "tau_star": tau_star,
            "is_captured": r_conf > 0.70 or (tau_star is not None and tau_star < 80)
        }
