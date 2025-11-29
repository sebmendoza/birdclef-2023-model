"""Model definitions and generic training loop for BirdCLEF baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FeatureSpec, compute_or_load_features


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_logistic_regression_pipeline() -> Pipeline:
    """Numeric-feature pipeline with standardization + multinomial logistic regression."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model


def build_random_forest_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Pipeline with optional scaling + RandomForest classifier."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return model


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
) -> None:
    """Train a model specified by `cfg` and print simple metrics."""

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    X, y = compute_or_load_features(
        cfg.feature_spec, base_dir=base_dir, chunk_root=chunk_root)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = cfg.model_builder()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"=== {cfg.name} ===")
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
