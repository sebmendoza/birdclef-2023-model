# Extensibility Guide

This document explains how the codebase has been refactored to make it easy to add new models and features.

## Architecture Overview

The codebase now uses **registry patterns** and **strategy patterns** to eliminate code duplication and make extension trivial.

## Adding a New Model Type

### Step 1: Create Model Builder (in `models.py`)

```python
def build_svm_pipeline(n_epochs: int = 50) -> Pipeline:
    """SVM classifier pipeline."""
    from sklearn.svm import SVC
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel='rbf', probability=True)),
    ])
```

### Step 2: Register Model Builder (in `config.py`)

```python
MODEL_BUILDERS = {
    # ... existing models ...
    "svm": build_svm_pipeline,
    "support_vector": build_svm_pipeline,  # Add aliases if desired
}
```

### Step 3: Create Training Strategy (in `models.py`) - Optional

Only needed if you want epoch tracking. Otherwise, `DefaultStrategy` will be used.

```python
class SVMStrategy(TrainingStrategy):
    """Training strategy for SVM models."""

    def train_with_epochs(self, model, X_train, X_test, y_train, y_test, n_epochs):
        # Implement epoch-by-epoch training if possible
        # Or just train once and return single accuracy
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        return model, [train_acc], [test_acc]

# Register strategy
TRAINING_STRATEGIES["svm"] = SVMStrategy
```

### Step 4: Add Experiments (in `config.py`)

```python
ALL_EXPERIMENTS = [
    # ... existing experiments ...
    ExperimentDefinition(
        name="SVM with Log-Mel",
        feature_kind="logmel",
        feature_n_mels=64,
        model_type="svm",  # Uses registered builder!
        n_epochs=50,
    ),
]
```

**That's it!** No need to modify `main.py` or the training loop.

## Code Reduction

### Before Refactoring

- **models.py**: ~360 lines with duplicated training logic
- **config.py**: if/elif chains for each model type
- Adding a model: Modify 3+ places

### After Refactoring

- **models.py**: ~450 lines, but **no duplication** - each model type has its own strategy class
- **config.py**: Registry-based, no if/elif chains
- Adding a model: **Just register and add experiments** (2-3 lines)

## Key Design Patterns Used

### 1. Registry Pattern (`config.py`)

```python
MODEL_BUILDERS: dict[str, Callable] = {
    "lr": build_logistic_regression_pipeline,
    "rf": build_random_forest_pipeline,
    # Add new models here - no if/elif needed!
}
```

**Benefits:**

- No if/elif chains
- Easy to add new models
- Supports aliases (multiple names for same model)

### 2. Strategy Pattern (`models.py`)

```python
class TrainingStrategy(ABC):
    @abstractmethod
    def train_with_epochs(...):
        pass

TRAINING_STRATEGIES: dict[str, type[TrainingStrategy]] = {
    "logistic_regression": LogisticRegressionStrategy,
    "random_forest": RandomForestStrategy,
    # Add new strategies here
}
```

**Benefits:**

- Each model type has isolated training logic
- No code duplication
- Easy to add new training strategies

### 3. Common Logic Extraction

All common operations (scaling, evaluation, saving) are now in shared functions:

- `_get_model_type()` - Detects model type
- `get_training_strategy()` - Gets appropriate strategy
- `_save_results()` - Saves all results

## Example: Adding XGBoost

```python
# 1. In models.py
def build_xgboost_pipeline(n_estimators: int = 100, n_epochs: int = 50) -> Pipeline:
    import xgboost as xgb
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(n_estimators=n_estimators)),
    ])

class XGBoostStrategy(TrainingStrategy):
    def train_with_epochs(self, model, X_train, X_test, y_train, y_test, n_epochs):
        # XGBoost-specific epoch tracking
        clf = model.named_steps["clf"]
        train_accuracies, test_accuracies = [], []

        for epoch in range(1, n_epochs + 1):
            clf.n_estimators = epoch * (clf.n_estimators // n_epochs)
            model.fit(X_train, y_train)
            # ... evaluate and track ...

        return model, train_accuracies, test_accuracies

TRAINING_STRATEGIES["xgboost"] = XGBoostStrategy

# 2. In config.py
MODEL_BUILDERS["xgboost"] = build_xgboost_pipeline

# 3. Add experiment
ALL_EXPERIMENTS.append(
    ExperimentDefinition(
        name="XGBoost with Log-Mel",
        feature_kind="logmel",
        feature_n_mels=64,
        model_type="xgboost",
        model_params={"n_estimators": 200},
    )
)
```

## Benefits Summary

✅ **No code duplication** - Common logic extracted  
✅ **Easy to extend** - Just register new models  
✅ **Type-safe** - Uses dataclasses and type hints  
✅ **Maintainable** - Each model type isolated  
✅ **No breaking changes** - Existing code still works

## File Structure

```
models.py
├── Model builders (build_*_pipeline)
├── Training strategies (TrainingStrategy classes)
├── Strategy registry (TRAINING_STRATEGIES)
└── Training function (train_and_evaluate_experiment)

config.py
├── Model builder registry (MODEL_BUILDERS)
├── Experiment definitions (ALL_EXPERIMENTS)
└── Helper functions (get_experiments_by_model_type)
```

The architecture is now **scalable** - adding 10 new models won't make the code unmaintainable!
