"""Base training strategy interface for model training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.pipeline import Pipeline


class TrainingStrategy(ABC):
    """Abstract base class for model-specific training strategies."""

    @abstractmethod
    def train_with_epochs(
        self,
        model: Pipeline,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_epochs: int,
    ) -> tuple[Pipeline, list[float], list[float]]:
        """Train model with epoch tracking.

        Returns:
            Tuple of (trained_model, train_accuracies, test_accuracies)
        """
        pass
