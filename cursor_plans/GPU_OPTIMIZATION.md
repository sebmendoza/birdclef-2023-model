# GPU Optimization Guide

## Current Status

The current implementation uses scikit-learn models (LogisticRegression, RandomForest) which **do not use GPU** by default. They run on CPU.

## When GPU Helps

GPU acceleration is most beneficial for:

- **Deep learning models** (neural networks)
- **Large datasets** (millions of samples)
- **Complex feature engineering** (e.g., spectrogram processing)

For simple models like Logistic Regression and Random Forest on moderate datasets, **GPU may not provide speedup** and can even be slower due to data transfer overhead.

## Options for GPU Acceleration

### 1. RAPIDS cuML (Recommended for sklearn-like models)

RAPIDS provides GPU-accelerated versions of sklearn models:

```python
# Install: pip install cuml-cu11  # or cuml-cu12 for CUDA 12

from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR

# These run on GPU automatically
```

**Pros:**

- Drop-in replacement for sklearn models
- Easy to integrate
- Good for large datasets

**Cons:**

- Requires NVIDIA GPU with CUDA
- Larger memory footprint
- May not help for small datasets

### 2. PyTorch/TensorFlow (For neural networks)

If you want to add neural network models:

```python
import torch
import torch.nn as nn

class BirdClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdClassifier(input_dim=65, num_classes=num_species).to(device)
```

### 3. XGBoost with GPU

XGBoost supports GPU:

```python
import xgboost as xgb

# Enable GPU
model = xgb.XGBClassifier(
    tree_method='gpu_hist',  # Use GPU
    gpu_id=0,
)
```

## Adding GPU Models to This Project

To add a GPU-accelerated model:

1. **Add model builder in `models.py`:**

   ```python
   def build_gpu_xgboost_pipeline(n_epochs: int = 50) -> Pipeline:
       import xgboost as xgb
       return Pipeline([
           ("scaler", StandardScaler()),
           ("clf", xgb.XGBClassifier(
               tree_method='gpu_hist',
               n_estimators=100,
           )),
       ])
   ```

2. **Register in `config.py`:**

   ```python
   MODEL_BUILDERS = {
       # ... existing models ...
       "xgboost": build_gpu_xgboost_pipeline,
   }
   ```

3. **Add training strategy in `models.py`** (if needed for epoch tracking):
   ```python
   class XGBoostStrategy(TrainingStrategy):
       def train_with_epochs(self, ...):
           # Implementation
   ```

## Performance Tips (CPU)

Even without GPU, you can optimize:

1. **Use `n_jobs=-1`** (already done) - uses all CPU cores
2. **Reduce feature dimensions** - fewer features = faster training
3. **Use feature caching** (already implemented)
4. **Batch processing** - process data in chunks
5. **Reduce `n_epochs`** for faster iteration during development

## Recommendation

For this project:

- **Current models (LR, RF)**: Keep CPU-based, they're fast enough
- **If adding neural networks**: Use PyTorch with GPU support
- **If dataset grows large**: Consider RAPIDS cuML
- **For ensemble methods**: XGBoost with GPU can help

The current architecture makes it easy to add GPU models - just create a new builder function and register it!
