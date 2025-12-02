"""Logistic Regression model implementation for BirdCLEF classification."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from training_strategy import TrainingStrategy


def build_logistic_regression_pipeline(n_epochs: int = 50) -> Pipeline:
    """Numeric-feature pipeline with standardization + multinomial logistic regression.

    Args:
        n_epochs: Number of training epochs (iterations)
    """
    # Use max_iter to control epochs, warm_start for incremental training
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1,  # Will train one iteration at a time
                    warm_start=True,  # Allows incremental training
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model


class LogisticRegressionStrategy(TrainingStrategy):
    """Training strategy for Logistic Regression models."""

    def train_with_epochs(
        self,
        model: Pipeline,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_epochs: int,
    ) -> tuple[Pipeline, list[float], list[float]]:
        """Train logistic regression with epoch tracking using SGDClassifier."""
        scaler = model.named_steps["scaler"]

        # Fit scaler once
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use SGDClassifier for true epoch-by-epoch training
        # Encode labels for SGDClassifier
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # Create SGD classifier (similar to logistic regression)
        clf_sgd = SGDClassifier(
            loss='log_loss',  # Logistic regression loss
            learning_rate='constant',
            eta0=0.01,
            max_iter=1,  # One iteration per fit call
            warm_start=True,
            random_state=42,
            n_jobs=-1,
        )

        train_accuracies = []
        test_accuracies = []

        # Train incrementally epoch by epoch
        for epoch in range(1, n_epochs + 1):
            clf_sgd.partial_fit(X_train_scaled, y_train_encoded,
                                classes=np.unique(y_train_encoded))

            # Evaluate
            train_pred_encoded = clf_sgd.predict(X_train_scaled)
            test_pred_encoded = clf_sgd.predict(X_test_scaled)

            # Convert back to original labels for accuracy calculation
            train_pred = label_encoder.inverse_transform(train_pred_encoded)
            test_pred = label_encoder.inverse_transform(test_pred_encoded)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{n_epochs} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        # For final model, train a full LogisticRegression for consistency
        # Create a new pipeline and fit it properly
        fitted_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=n_epochs * 10,  # Approximate iterations
                n_jobs=-1,
            )),
        ])
        # Fit the pipeline (this will fit both scaler and classifier)
        fitted_model.fit(X_train, y_train)

        return fitted_model, train_accuracies, test_accuracies
