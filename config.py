"""Configuration file for BirdCLEF experiments.

This module defines all experiment configurations, making it easy to add,
modify, or remove experiments without changing the main training code.

To add a new model type:
1. Add a builder function in models.py (e.g., build_svm_pipeline)
2. Register it in MODEL_BUILDERS below
3. Add experiments using the new model_type
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from features import FeatureSpec
from logistic_classifier import build_logistic_regression_pipeline
from models import ExperimentConfig
from random_forest import build_random_forest_pipeline

# Model builder registry - add new models here!
MODEL_BUILDERS: dict[str, Callable] = {
    "lr": build_logistic_regression_pipeline,
    "rf": build_random_forest_pipeline,
    # Add new models here:
    # "svm": build_svm_pipeline,
    # "xgboost": build_xgboost_pipeline,
}


@dataclass
class ExperimentDefinition:
    """Definition of a single experiment with all its parameters."""

    name: str
    feature_kind: str  # "rms" or "logmel"
    model_type: str  # "lr" for logistic regression, "rf" for random forest
    feature_n_mels: int | None = None  # Only for logmel features
    model_params: dict = None  # Additional model parameters
    n_epochs: int = 50  # Number of training epochs

    def to_experiment_config(self) -> ExperimentConfig:
        """Convert to ExperimentConfig for training."""
        # Create feature spec
        feature_spec = FeatureSpec(
            kind=self.feature_kind,
            n_mels=self.feature_n_mels,
        )

        # Get model builder from registry
        model_type_lower = self.model_type.lower()
        if model_type_lower not in MODEL_BUILDERS:
            available = ", ".join(MODEL_BUILDERS.keys())
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Available types: {available}"
            )

        base_builder = MODEL_BUILDERS[model_type_lower]

        # Create model builder with parameters
        def model_builder():
            if model_type_lower in ["lr", "logistic", "logistic_regression"]:
                return base_builder(n_epochs=self.n_epochs)
            elif model_type_lower in ["rf", "random_forest", "randomforest"]:
                return base_builder(
                    random_state=42,
                    n_epochs=self.n_epochs,
                    **(self.model_params or {})
                )
            else:
                # For other model types, pass n_epochs and model_params
                kwargs = {"n_epochs": self.n_epochs}
                if self.model_params:
                    kwargs.update(self.model_params)
                return base_builder(**kwargs)

        return ExperimentConfig(
            name=self.name,
            feature_spec=feature_spec,
            model_builder=model_builder,
        )


# Define all experiments
ALL_EXPERIMENTS: list[ExperimentDefinition] = [
    ExperimentDefinition(
        name="Logistic Regression with RMS",
        feature_kind="rms",
        model_type="lr",
    ),
    ExperimentDefinition(
        name="Random Forest with RMS",
        feature_kind="rms",
        model_type="rf",
    ),
    ExperimentDefinition(
        name="Logistic Regression with Log-Mel",
        feature_kind="logmel",
        feature_n_mels=64,
        model_type="lr",
    ),
    ExperimentDefinition(
        name="Random Forest with Log-Mel",
        feature_kind="logmel",
        feature_n_mels=64,
        model_type="rf",
    ),
]


def get_experiments_by_model_type(model_type: str | None = None) -> list[ExperimentDefinition]:
    """Get experiments filtered by model type.

    Args:
        model_type: Model type to filter by (e.g., "lr", "rf", or None for all)

    Returns:
        List of matching experiment definitions
    """
    if model_type is None:
        return ALL_EXPERIMENTS

    model_type_lower = model_type.lower()

    # Normalize model type to match experiment definitions
    # Map aliases to canonical form
    if model_type_lower in ["lr", "logistic", "logistic_regression"]:
        canonical_type = "lr"
    elif model_type_lower in ["rf", "random_forest", "randomforest"]:
        canonical_type = "rf"
    else:
        # Check if it's a registered model type
        if model_type_lower in MODEL_BUILDERS:
            # Try to find experiments with this model type
            return [exp for exp in ALL_EXPERIMENTS if exp.model_type.lower() == model_type_lower]
        available = ", ".join(set(exp.model_type for exp in ALL_EXPERIMENTS))
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types in experiments: {available}"
        )

    return [exp for exp in ALL_EXPERIMENTS if exp.model_type == canonical_type]


def get_experiment_by_name(name: str) -> ExperimentDefinition | None:
    """Get an experiment by its name."""
    for exp in ALL_EXPERIMENTS:
        if exp.name == name:
            return exp
    return None
