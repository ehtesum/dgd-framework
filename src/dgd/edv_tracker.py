"""
Expected Diagnostic Value (EDV) & Epistemic Entropy Tracker.

Formalizes:
    EDV(a_t) = I(H; Y | C, a_t) = H(H | C) - E_y [ H(H | C, a_t, y) ]
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple


class EDVTracker:
    """Tracks and calculates the Expected Diagnostic Value (EDV) of candidate reasoning steps."""

    def __init__(self, hypotheses: List[str]):
        """
        Initialize the tracker with a set of competing hypotheses {H_1, ..., H_k}.
        """
        self.hypotheses = hypotheses
        self.k = len(hypotheses)
        # Uniform prior P(H_i) = 1/k
        self.priors = np.ones(self.k) / self.k

    def shannon_entropy(self, dist: np.ndarray) -> float:
        """Computes Shannon entropy H(P) in bits."""
        dist = dist[dist > 0]
        return -float(np.sum(dist * np.log2(dist)))

    def calculate_edv(
        self,
        p_outcome_pos_given_h: List[float],
        prior: np.ndarray = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate Expected Diagnostic Value for a binary test action a_t.

        Parameters
        ----------
        p_outcome_pos_given_h : List[float]
            Likelihood P(Y=+ | H_i) for each hypothesis H_i in {H_1, ..., H_k}.
        prior : np.ndarray, optional
            Current prior belief vector P(H). Defaults to self.priors.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            - edv: Mutual information I(H; Y | C, a_t) in bits.
            - post_pos: Posterior distribution P(H | Y=+).
            - post_neg: Posterior distribution P(H | Y=-).
        """
        if prior is None:
            prior = self.priors

        h_prior = self.shannon_entropy(prior)
        likelihood_pos = np.array(p_outcome_pos_given_h)
        likelihood_neg = 1.0 - likelihood_pos

        # Total marginal probability of outcomes
        p_pos = np.sum(prior * likelihood_pos)
        p_neg = np.sum(prior * likelihood_neg)

        # Posterior distributions via Bayes' theorem
        post_pos = (prior * likelihood_pos) / max(1e-12, p_pos)
        post_neg = (prior * likelihood_neg) / max(1e-12, p_neg)

        # Expected posterior entropy
        e_post_entropy = (
            p_pos * self.shannon_entropy(post_pos) +
            p_neg * self.shannon_entropy(post_neg)
        )

        edv = max(0.0, h_prior - e_post_entropy)
        return edv, post_pos, post_neg

    def is_discriminative_action(self, edv: float, threshold: float = 0.3) -> bool:
        """Determines if a reasoning step provides discriminative information gain."""
        return bool(edv >= threshold)
