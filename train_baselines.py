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


def main(include_augmented: bool = False) -> None:
    """Run baseline experiments with optional augmented data.
    
    Args:
        include_augmented: Whether to include augmented data in training
    """
    base_dir = Path.cwd()
    print(f"🎵 Running experiments with augmented data: {include_augmented}")
    print("=" * 50)
    
    for cfg in EXPERIMENTS:
        train_and_evaluate_experiment(
            cfg, base_dir=base_dir, chunk_root=TRIMMED_CHUNK_ROOT, include_augmented=include_augmented)
        print("\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BirdCLEF baseline models")
    parser.add_argument(
        "--include-augmented", 
        action="store_true", 
        help="Include augmented data in training"
    )
    
    args = parser.parse_args()
    main(include_augmented=args.include_augmented)
