from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LinUCB baseline requires numpy. Install it with: pip install numpy"
    ) from e


@dataclass
class _ArmStats:
    A: np.ndarray
    b: np.ndarray


class LinUCBPreferencePredictor:
    """Independent LinUCB bandit per preference dimension (arms = allowed options)."""

    def __init__(self, dimension_options: Dict[str, List[str]], context_dim: int, alpha: float) -> None:
        self.dimension_options = dimension_options
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)

        self._bandits: Dict[str, Dict[str, _ArmStats]] = {}
        for field, options in self.dimension_options.items():
            self._bandits[field] = {}
            for opt in options:
                self._bandits[field][opt] = _ArmStats(
                    A=np.eye(self.context_dim, dtype=np.float64),
                    b=np.zeros(self.context_dim, dtype=np.float64),
                )

    def _ucb(self, stats: _ArmStats, x: np.ndarray) -> float:
        # Solve A^{-1} x and A^{-1} b using linear solves (stable vs explicit inverse).
        A_inv_x = np.linalg.solve(stats.A, x)
        theta = np.linalg.solve(stats.A, stats.b)
        mean = float(x @ theta)
        bonus = self.alpha * float(np.sqrt(x @ A_inv_x))
        return mean + bonus

    def predict_bundle(self, x: np.ndarray) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for field, options in self.dimension_options.items():
            best_opt = options[0]
            best_score = float("-inf")
            for opt in options:
                score = self._ucb(self._bandits[field][opt], x)
                if score > best_score:
                    best_score = score
                    best_opt = opt
            out[field] = best_opt
        return out

    def update(self, field: str, arm: str, x: np.ndarray, reward: float) -> None:
        stats = self._bandits[field][arm]
        stats.A += np.outer(x, x)
        stats.b += float(reward) * x

    def update_from_truth(
        self,
        x: np.ndarray,
        pred_bundle: Dict[str, str],
        truth_bundle: Dict[str, str],
    ) -> List[Tuple[str, str, float]]:
        """Semi-bandit update: reward 1 for correct, else (pred arm->0, truth arm->1)."""
        updates: List[Tuple[str, str, float]] = []
        for field, truth in truth_bundle.items():
            pred = pred_bundle.get(field, "")
            if pred == truth and pred in self.dimension_options[field]:
                self.update(field, pred, x, 1.0)
                updates.append((field, pred, 1.0))
            else:
                if pred in self.dimension_options[field]:
                    self.update(field, pred, x, 0.0)
                    updates.append((field, pred, 0.0))
                if truth in self.dimension_options[field]:
                    self.update(field, truth, x, 1.0)
                    updates.append((field, truth, 1.0))
        return updates

