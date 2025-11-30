"""Audio feature extraction and caching utilities for BirdCLEF baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Tuple

import librosa
import numpy as np

from data import (
    CHUNK_AUDIO_ROOT,
    TRIMMED_CHUNK_ROOT,
    FEATURES_ROOT,
    FEATURES_TRIMMED_ROOT,
    AUGMENTED_DATA_ROOT,
    load_cleaned_metadata,
    load_combined_metadata,
)


# ---------------------------------------------------------------------------
# Feature specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for a particular feature representation."""

    kind: str  # "rms" or "logmel"
    n_mels: int | None = None  # only used for logmel


def get_feature_and_target_columns(spec: FeatureSpec) -> Tuple[list[str], str]:
    """Return feature column names (for metadata) and target column name.

    For RMS features we use:
        - duration
        - audio_rms (computed from chunks)

    For log-mel we treat *all* features as derived and only reuse duration
    as an additional numeric input.
    """

    if spec.kind == "rms":
        feature_cols = ["duration", "audio_rms"]
    elif spec.kind == "logmel":
        # Duration is included as the first feature; the rest are log-mel stats
        feature_cols = ["duration"]
    else:
        raise ValueError(f"Unknown feature kind: {spec.kind!r}")

    target_col = "primary_label"
    return feature_cols, target_col


# ---------------------------------------------------------------------------
# Low-level feature computation
# ---------------------------------------------------------------------------


def compute_global_rms(
    filename: str,
    base_dir: Path | str | None = None,
    sr: int | None = None,
    max_duration: float | None = 10.0,
    chunk_root: Path | str | None = None,
) -> float:
    """Compute a single RMS value for one 15-second audio chunk."""

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    if chunk_root is None:
        chunk_root = CHUNK_AUDIO_ROOT
    else:
        chunk_root = Path(chunk_root)

    audio_path = base_dir / chunk_root.relative_to(Path.cwd()) / filename

    # Check if this is an augmented file (look in augmented directory)
    if not audio_path.exists():
        augmented_audio_path = base_dir / AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio" / filename
        if augmented_audio_path.exists():
            audio_path = augmented_audio_path
        else:
            # Check subdirectories in augmented data
            for subdir in (base_dir / AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio").iterdir():
                if subdir.is_dir():
                    potential_path = subdir / filename
                    if potential_path.exists():
                        audio_path = potential_path
                        break

    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration)
        if y.size == 0:
            return float("nan")
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        return rms
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to compute RMS for {audio_path}: {exc}")
        return float("nan")


def compute_logmel_mean_features(
    filename: str,
    base_dir: Path | str | None = None,
    sr: int = 32000,
    n_mels: int = 64,
    max_duration: float | None = 15.0,
    chunk_root: Path | str | None = None,
) -> np.ndarray:
    """Compute per-mel-band mean log-mel features for one audio file."""

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    if chunk_root is None:
        chunk_root = CHUNK_AUDIO_ROOT
    else:
        chunk_root = Path(chunk_root)

    audio_path = base_dir / chunk_root.relative_to(Path.cwd()) / filename

    # Check if this is an augmented file (look in augmented directory)
    if not audio_path.exists():
        augmented_audio_path = base_dir / AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio" / filename
        if augmented_audio_path.exists():
            audio_path = augmented_audio_path
        else:
            # Check subdirectories in augmented data
            for subdir in (base_dir / AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio").iterdir():
                if subdir.is_dir():
                    potential_path = subdir / filename
                    if potential_path.exists():
                        audio_path = potential_path
                        break

    try:
        y, _sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        if y.size == 0:
            return np.full(n_mels, np.nan, dtype=np.float32)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)
        features = log_S.mean(axis=1).astype(np.float32)
        return features
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to compute log-mel for {audio_path}: {exc}")
        return np.full(n_mels, np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _get_feature_cache_path(
    spec: FeatureSpec,
    base_dir: Path | str | None = None,
    chunk_root: Path | str | None = None,
) -> Path:
    """Return the cache path for a given feature spec.

    If chunk_root is TRIMMED_CHUNK_ROOT, uses FEATURES_TRIMMED_ROOT.
    Otherwise defaults to FEATURES_ROOT (for regular chunks).
    """

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Determine which features directory to use based on chunk_root
    if chunk_root is None:
        features_dir = FEATURES_ROOT
    else:
        chunk_root = Path(chunk_root)
        # Check if this is the trimmed chunks directory
        if chunk_root.name == TRIMMED_CHUNK_ROOT.name or chunk_root == TRIMMED_CHUNK_ROOT:
            features_dir = FEATURES_TRIMMED_ROOT
        else:
            features_dir = FEATURES_ROOT

    cache_dir = base_dir / features_dir.relative_to(Path.cwd())
    cache_dir.mkdir(parents=True, exist_ok=True)

    name_parts = [spec.kind]
    if spec.kind == "logmel" and spec.n_mels is not None:
        name_parts.append(f"n_mels{spec.n_mels}")

    filename = "features_" + "_".join(name_parts) + ".pkl"
    return cache_dir / filename


def compute_or_load_features(
    spec: FeatureSpec,
    base_dir: Path | str | None = None,
    chunk_root: Path | str | None = None,
    include_augmented: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute features for all samples or load them from a cache file.
    
    Args:
        spec: Feature specification defining what features to compute
        base_dir: Base directory for the project 
        chunk_root: Directory containing audio chunks
        include_augmented: Whether to include augmented data in training
        
    Returns:
        Tuple of (X, y) arrays for features and targets
    """

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Modify cache path to account for augmented data
    cache_path = _get_feature_cache_path(
        spec, base_dir=base_dir, chunk_root=chunk_root)
    
    if include_augmented:
        # Use different cache file for augmented data
        cache_path = cache_path.with_name(f"augmented_{cache_path.name}")
    
    if cache_path.exists():
        print(f"Loading cached {spec.kind} features from {cache_path} ...")
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        return cache["X"], cache["y"]

    # No cache yet; compute from audio + metadata
    if include_augmented:
        print("🎵 Loading combined metadata (original + augmented)...")
        df = load_combined_metadata(base_dir=base_dir)
    else:
        df = load_cleaned_metadata(base_dir=base_dir)
    feature_cols, target_col = get_feature_and_target_columns(spec)

    required_columns = ["duration", target_col, "filename"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} to exist in cleaned metadata but they were missing."
        )

    filenames = df["filename"].tolist()
    durations = df["duration"].to_numpy(dtype=np.float32)
    targets = df[target_col].astype(str)

    if spec.kind == "rms":
        print("Computing RMS features from 15-second chunks (this may take a while)...")
        rms_values = [
            compute_global_rms(fn, base_dir=base_dir, chunk_root=chunk_root) for fn in filenames
        ]
        rms_values = np.asarray(rms_values, dtype=np.float32)

        X = np.stack([durations, rms_values], axis=1)

        feature_names = ["duration", "audio_rms"]

    elif spec.kind == "logmel":
        if spec.n_mels is None:
            raise ValueError("n_mels must be set for logmel features.")

        print("Computing log-mel features from 15-second chunks (this may take a while)...")
        logmel_features = []
        for fn in filenames:
            feats = compute_logmel_mean_features(
                filename=fn,
                base_dir=base_dir,
                chunk_root=chunk_root,
                n_mels=spec.n_mels,
            )
            logmel_features.append(feats)

        logmel_features = np.stack(logmel_features, axis=0)  # (N, n_mels)
        durations_col = durations.reshape(-1, 1)
        X = np.concatenate([durations_col, logmel_features], axis=1)

        feature_names = ["duration"] + \
            [f"logmel_{i}" for i in range(spec.n_mels)]

    else:
        raise ValueError(f"Unknown feature kind: {spec.kind!r}")

    # Drop any rows with NaNs in X
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = targets[mask].to_numpy()

    cache = {
        "feature_type": spec.kind,
        "version": 1,
        "feature_names": feature_names,
        "target_name": target_col,
        "X": X,
        "y": y,
    }
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    print(f"Saved {spec.kind} features to {cache_path}")

    return X, y
