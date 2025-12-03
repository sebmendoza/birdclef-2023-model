"""
Create time-series pickle file for GRU training.
Run this locally to generate the pickle file, then upload to Google Drive.
"""

from pathlib import Path
from features import FeatureSpec, compute_or_load_features

BASE_DIR = Path.cwd()
INCLUDE_AUGMENTED = True  
# Number of mel bands
N_MELS = 64

print("\n📋 Configuration:")
print(f"   Base directory: {BASE_DIR}")
print(f"   Include augmented data: {INCLUDE_AUGMENTED}")
print(f"   Number of mel bands: {N_MELS}")


spec = FeatureSpec(
    kind="logmel_rms_timeseries",
    n_mels=N_MELS
)


X, y = compute_or_load_features(
    spec=spec,
    base_dir=BASE_DIR,
    chunk_root=None,  # Use default CHUNK_AUDIO_ROOT
    include_augmented=INCLUDE_AUGMENTED
)

# ============================================================
# VERIFY RESULTS
# ============================================================


import numpy as np

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Number of samples: {X.shape[0]}")
print(f"   Number of features: {X.shape[1]} ({N_MELS} mel + 1 RMS)")
print(f"   Time frames: {X.shape[2]}")
print(f"   Number of species: {len(np.unique(y))}")

print("\ Species distribution:")
unique_species = np.unique(y)
for i, species in enumerate(unique_species):
    count = np.sum(y == species)
    percentage = (count / len(y)) * 100
    print(f"   {i+1}. {species}: {count} samples ({percentage:.1f}%)")

# ============================================================
# FIND PICKLE FILE LOCATION
# ============================================================

from features import _get_feature_cache_path

cache_path = _get_feature_cache_path(
    spec, 
    base_dir=BASE_DIR, 
    chunk_root=None
)

if INCLUDE_AUGMENTED:
    cache_path = cache_path.with_name(f"combined_{cache_path.name}")

print(f"\n Pickle file saved to:")
print(f"   {cache_path}")
print(f"   Size: {cache_path.stat().st_size / (1024**2):.2f} MB")
