"""Entry point for running simple BirdCLEF baseline models.

This script trains four models on features computed from 15-second chunks:

1) Logistic Regression with duration + global RMS
2) Random Forest with duration + global RMS
3) Logistic Regression with duration + log-mel features
4) Random Forest with duration + log-mel features
"""

from __future__ import annotations

from pathlib import Path

from features import FeatureSpec
from data import TRIMMED_CHUNK_ROOT
from models import (
    ExperimentConfig,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_and_evaluate_experiment,
)


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="Logistic Regression with RMS",
        feature_spec=FeatureSpec(kind="rms"),
        model_builder=build_logistic_regression_pipeline,
    ),
    ExperimentConfig(
        name="Random Forest with RMS",
        feature_spec=FeatureSpec(kind="rms"),
        model_builder=lambda: build_random_forest_pipeline(random_state=42),
    ),
    ExperimentConfig(
        name="Logistic Regression with Log-Mel",
        feature_spec=FeatureSpec(kind="logmel", n_mels=64),
        model_builder=build_logistic_regression_pipeline,
    ),
    ExperimentConfig(
        name="Random Forest with Log-Mel",
        feature_spec=FeatureSpec(kind="logmel", n_mels=64),
        model_builder=lambda: build_random_forest_pipeline(random_state=42),
    ),
]


def main() -> None:
    base_dir = Path.cwd()
    for cfg in EXPERIMENTS:
        train_and_evaluate_experiment(
            cfg, base_dir=base_dir, chunk_root=TRIMMED_CHUNK_ROOT)
        print("\n")


if __name__ == "__main__":
    main()
