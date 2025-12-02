"""Random Forest model implementation for BirdCLEF classification."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training_strategy import TrainingStrategy


def build_random_forest_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
    n_epochs: int = 30,
) -> Pipeline:
    """Pipeline with optional scaling + RandomForest classifier.

    Args:
        n_estimators: Total number of trees to train
        max_depth: Maximum depth of trees
        random_state: Random seed
        n_epochs: Number of training epochs (will train n_estimators/n_epochs trees per epoch)
    """
    # For random forest, we'll train incrementally by adding trees
    trees_per_epoch = max(1, n_estimators // n_epochs)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=trees_per_epoch,  # Start with trees_per_epoch
                    max_depth=max_depth,
                    n_jobs=-1,
                    random_state=random_state,
                    warm_start=False,  # We'll manually add trees
                ),
            ),
        ]
    )
    return model


class RandomForestStrategy(TrainingStrategy):
    """Training strategy for Random Forest models."""

    def train_with_epochs(
        self,
        model: Pipeline,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_epochs: int,
    ) -> tuple[Pipeline, list[float], list[float]]:
        """Train random forest with epoch tracking by incrementally adding trees."""
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]

        # Fit scaler once
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Get total n_estimators from the model (default 300)
        total_trees = clf.n_estimators if clf.n_estimators > 0 else 300
        trees_per_epoch = max(1, total_trees // n_epochs)

        # Create new classifier that we'll incrementally train
        clf_new = RandomForestClassifier(
            n_estimators=trees_per_epoch,  # Start with trees_per_epoch
            max_depth=clf.max_depth,
            n_jobs=clf.n_jobs,
            random_state=clf.random_state,
            warm_start=True,  # Allow incremental training
        )

        train_accuracies = []
        test_accuracies = []

        # Train incrementally
        for epoch in range(1, n_epochs + 1):
            clf_new.n_estimators = epoch * trees_per_epoch
            clf_new.fit(X_train_scaled, y_train)

            # Evaluate
            train_pred = clf_new.predict(X_train_scaled)
            test_pred = clf_new.predict(X_test_scaled)
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{n_epochs} - Trees: {clf_new.n_estimators} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Create a new pipeline with the fitted classifier and fit it properly
        fitted_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf_new),
        ])
        # Fit the pipeline (scaler will refit, but clf is already fitted)
        fitted_model.fit(X_train, y_train)

        return fitted_model, train_accuracies, test_accuracies
