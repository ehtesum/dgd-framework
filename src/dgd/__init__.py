"""
Diagnostic-Guided Deliberation (DGD) Framework
==============================================
A principled epistemic entropy-steered deliberation framework for 
Large Language Model reasoning under hypothesis competition.
"""

from dgd.controller import DGDController
from dgd.edv_tracker import EDVTracker
from dgd.commitment import CommitmentHorizonDetector
from dgd.runner import ComparativeRunner
from dgd.evaluator import DGDEvaluator

__version__ = "0.1.0"
__all__ = [
    "DGDController",
    "EDVTracker",
    "CommitmentHorizonDetector",
    "ComparativeRunner",
    "DGDEvaluator",
]
