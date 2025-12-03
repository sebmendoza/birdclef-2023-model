"""Model definitions and generic training loop for BirdCLEF baselines."""

from __future__ import annotations

import json
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FeatureSpec, compute_or_load_features
from logistic_classifier import (
    LogisticRegressionStrategy,
    build_logistic_regression_pipeline,
)
from random_forest import RandomForestStrategy, build_random_forest_pipeline
from training_strategy import TrainingStrategy


# ---------------------------------------------------------------------------
# Training strategies (extensible pattern for different model types)
# ---------------------------------------------------------------------------

# TrainingStrategy base class is now in training_strategy.py
# Model builders and strategies are in separate files:
# - logistic_classifier.py: build_logistic_regression_pipeline, LogisticRegressionStrategy
# - random_forest.py: build_random_forest_pipeline, RandomForestStrategy


class DefaultStrategy(TrainingStrategy):
    """Default training strategy for models without epoch tracking."""

    def train_with_epochs(
        self,
        model: Pipeline,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_epochs: int,
    ) -> tuple[Pipeline, list[float], list[float]]:
        """Train model normally without epoch tracking."""
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        return model, [train_acc], [test_acc]


# Strategy registry - easy to extend with new model types!
TRAINING_STRATEGIES: dict[str, type[TrainingStrategy]] = {
    "logistic_regression": LogisticRegressionStrategy,
    "random_forest": RandomForestStrategy,
    "default": DefaultStrategy,
}


def get_training_strategy(model: Pipeline) -> TrainingStrategy:
    """Get the appropriate training strategy for a model."""
    model_type = _get_model_type(model)
    strategy_class = TRAINING_STRATEGIES.get(
        model_type, TRAINING_STRATEGIES["default"])
    return strategy_class()


# ---------------------------------------------------------------------------
# Experiment configuration and training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single baseline experiment."""

    name: str
    feature_spec: FeatureSpec
    model_builder: Callable[[], Pipeline]


def train_and_evaluate_experiment(
    cfg: ExperimentConfig,
    base_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    chunk_root: Path | str | None = None,
    include_augmented: bool = False,
    n_epochs: int = 50,
) -> None:
    """Train a model specified by `cfg` with epoch tracking and save results.

    Args:
        cfg: Experiment configuration
        base_dir: Base directory for the project
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        chunk_root: Directory containing audio chunks
        include_augmented: Whether to include augmented data in training
        n_epochs: Number of training epochs
    """

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    X, y = compute_or_load_features(
        cfg.feature_spec, base_dir=base_dir, chunk_root=chunk_root, include_augmented=include_augmented)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Determine model type from the builder
    model = cfg.model_builder()

    print(f"=== {cfg.name} ===")
    print(f"Training for {n_epochs} epochs...")

    # Get appropriate training strategy and train
    strategy = get_training_strategy(model)
    model, train_accuracies, test_accuracies = strategy.train_with_epochs(
        model, X_train, X_test, y_train, y_test, n_epochs
    )

    # Final evaluation
    y_pred = model.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred)

    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    print("\nClassification report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Save results
    _save_results(
        cfg.name,
        model,
        train_accuracies,
        test_accuracies,
        report,
        final_acc,
        base_dir,
    )


def _get_model_type(model: Pipeline) -> str:
    """Determine the type of model from the pipeline."""
    clf = model.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return "logistic_regression"
    elif isinstance(clf, RandomForestClassifier):
        return "random_forest"
    else:
        return "unknown"


def _save_results(
    experiment_name: str,
    model: Pipeline,
    train_accuracies: list[float],
    test_accuracies: list[float],
    classification_report: dict,
    final_accuracy: float,
    base_dir: Path,
) -> None:
    """Save model, results, and plots to results directory."""
    # Create results directory structure
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Create experiment-specific directory (sanitize name)
    exp_dir_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
    exp_dir = results_dir / exp_dir_name
    exp_dir.mkdir(exist_ok=True)

    # Save model
    model_path = exp_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    # Save metrics
    metrics = {
        "final_accuracy": final_accuracy,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "classification_report": classification_report,
        "n_epochs": len(train_accuracies),
    }

    metrics_path = exp_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save classification report as text
    report_path = exp_dir / "classification_report.txt"
    with report_path.open("w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Final Test Accuracy: {final_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(classification_report))
    print(f"Saved classification report to {report_path}")

    # Plot and save accuracy over epochs
    if len(train_accuracies) > 1:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o")
        plt.plot(epochs, test_accuracies, label="Test Accuracy", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Training Progress: {experiment_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = exp_dir / "accuracy_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved accuracy plot to {plot_path}")
