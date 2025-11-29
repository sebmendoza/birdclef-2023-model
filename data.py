"""Data loading and path configuration for BirdCLEF baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Project-level paths (relative to repo root)
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "birdclef-2023"
TRAIN_AUDIO_ROOT = DATA_ROOT / "train_audio"
CHUNK_AUDIO_ROOT = DATA_ROOT / "15secondchunks"
TRIMMED_CHUNK_ROOT = DATA_ROOT / "15secondchunkstrimmed"
FEATURES_ROOT = DATA_ROOT / "features"
FEATURES_TRIMMED_ROOT = DATA_ROOT / "features_trimmed"
METADATA_CLEANED = DATA_ROOT / "train_metadata_cleaned.csv"


def load_cleaned_metadata(base_dir: Path | str | None = None) -> pd.DataFrame:
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

    return pd.read_csv(csv_path)
