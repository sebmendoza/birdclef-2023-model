"""Data loading and path configuration for BirdCLEF baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Project-level paths (relative to repo root)
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "birdclef-2023"
AUGMENTED_DATA_ROOT = DATA_ROOT / "augmented"
TRAIN_AUDIO_ROOT = DATA_ROOT / "train_audio"
CHUNK_AUDIO_ROOT = DATA_ROOT / "15secondchunks"
TRIMMED_CHUNK_ROOT = DATA_ROOT / "15secondchunkstrimmed"
FEATURES_ROOT = DATA_ROOT / "features"
FEATURES_TRIMMED_ROOT = DATA_ROOT / "features_trimmed"
METADATA_CLEANED = DATA_ROOT / "train_metadata_cleaned.csv"
AUGMENTED_METADATA = AUGMENTED_DATA_ROOT / "augmented_metadata.csv"


def load_cleaned_metadata(base_dir: Path | str | None = None, include_augmented: bool = False) -> pd.DataFrame:
    """Load the cleaned metadata CSV produced by `preprocessing.ipynb`."""

    if base_dir is None:
        base_dir = PROJECT_ROOT
    else:
        base_dir = Path(base_dir)

    csv_path = base_dir / METADATA_CLEANED.relative_to(PROJECT_ROOT)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned metadata not found at {csv_path}. "
            "Make sure you've run `preprocessing.ipynb` to generate it."
        )

    df_original = pd.read_csv(csv_path)
    
    if include_augmented:
        augmented_path = base_dir / AUGMENTED_METADATA.relative_to(PROJECT_ROOT)
        if augmented_path.exists():
            print(f"Including augmented data from {augmented_path}")
            df_augmented = pd.read_csv(augmented_path)
            
            # Ensure both dataframes have the same columns
            common_columns = list(set(df_original.columns) & set(df_augmented.columns))
            df_original = df_original[common_columns]
            df_augmented = df_augmented[common_columns]
            
            # Concatenate the dataframes
            df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
            print(f"Combined dataset: {len(df_original)} original + {len(df_augmented)} augmented = {len(df_combined)} total samples")
            return df_combined
        else:
            print(f"Augmented metadata not found at {augmented_path}. Using original data only.")
    
    return df_original


def load_combined_metadata(base_dir: Path | str | None = None) -> pd.DataFrame:
    """Load combined metadata including both original and augmented data."""
    return load_cleaned_metadata(base_dir=base_dir, include_augmented=True)
