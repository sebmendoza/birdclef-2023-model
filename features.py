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

    kind: str  # "rms", "logmel", or "logmel_rms_timeseries"
    n_mels: int | None = None  # only used for logmel and logmel_rms_timeseries


def get_feature_and_target_columns(spec: FeatureSpec) -> Tuple[list[str], str]:
    """Return feature column names (for metadata) and target column name."""

    if spec.kind == "rms":
        feature_cols = ["duration", "audio_rms"]
    elif spec.kind == "logmel":
        feature_cols = ["duration"]
    elif spec.kind == "logmel_rms_timeseries":
        feature_cols = ["duration"]
    else:
        raise ValueError(f"Unknown feature kind: {spec.kind!r}")

    target_col = "primary_label"
    return feature_cols, target_col


# ---------------------------------------------------------------------------
# Common audio loading utilities
# ---------------------------------------------------------------------------


def _resolve_audio_path(
    filename: str,
    base_dir: Path,
    chunk_root: Path,
) -> Path:
    """Resolve the path to an audio file, checking both original and augmented locations.

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    if chunk_root is None:
        chunk_root = CHUNK_AUDIO_ROOT
    else:
        chunk_root = Path(chunk_root)

    # Parse filename
    filename_path = Path(filename)
    if len(filename_path.parts) > 1:
        species = filename_path.parts[0]
        audio_filename = filename_path.name
    else:
        audio_filename = filename_path.name
        species = None
    
    # Try original location
    audio_path = base_dir / chunk_root.relative_to(Path.cwd()) / filename
    
    # If not found, try augmented directory
    if not audio_path.exists():
        augmented_base = base_dir / \
            AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio"
        if augmented_base.exists():
            if species:
                augmented_audio_path = augmented_base / species / audio_filename
                if augmented_audio_path.exists():
                    audio_path = augmented_audio_path
            
            if not audio_path.exists():
                for subdir in augmented_base.iterdir():
                    if subdir.is_dir():
                        potential_path = subdir / audio_filename
                        if potential_path.exists():
                            audio_path = potential_path
                            break

    return audio_path


def _load_audio(
    filename: str,
    base_dir: Path,
    chunk_root: Path,
    sr: int | None = None,
    max_duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio file with error handling.

    Args:
        filename: Audio filename
        base_dir: Base directory for the project
        chunk_root: Directory containing audio chunks
        sr: Sample rate (None to use file's native rate)
        max_duration: Maximum duration to load in seconds

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        FileNotFoundError: If audio file cannot be found
        Exception: If audio loading fails
    """
    audio_path = _resolve_audio_path(filename, base_dir, chunk_root)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, duration=max_duration)
        return y, sr_loaded
    except Exception as exc:
        raise Exception(
            f"Failed to load audio from {audio_path}: {exc}") from exc


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

    try:
        y, _ = _load_audio(filename, base_dir, chunk_root,
                           sr=sr, max_duration=max_duration)
        if y.size == 0:
            return float("nan")
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        return rms
    except Exception as exc:
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

    # Parse filename
    filename_path = Path(filename)
    if len(filename_path.parts) > 1:
        species = filename_path.parts[0]
        audio_filename = filename_path.name
    else:
        audio_filename = filename_path.name
        species = None
    
    # Try original location
    audio_path = base_dir / chunk_root.relative_to(Path.cwd()) / filename
    
    # If not found, try augmented directory
    if not audio_path.exists():
        augmented_base = base_dir / AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio"
        if augmented_base.exists():
            if species:
                augmented_audio_path = augmented_base / species / audio_filename
                if augmented_audio_path.exists():
                    audio_path = augmented_audio_path
            
            if not audio_path.exists():
                for subdir in augmented_base.iterdir():
                    if subdir.is_dir():
                        potential_path = subdir / audio_filename
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
    except Exception as exc:
        print(f"[WARN] Failed to compute log-mel for {audio_path}: {exc}")
        return np.full(n_mels, np.nan, dtype=np.float32)


def compute_logmel_rms_timeseries(
    filename: str,
    base_dir: Path | str | None = None,
    sr: int = 32000,
    n_mels: int = 64,
    max_duration: float | None = 15.0,
    chunk_root: Path | str | None = None,
) -> np.ndarray | None:
    """
    Compute FULL mel spectrogram + RMS features over time.
    
    Returns:
        np.ndarray of shape (n_mels + 1, time_frames) or None if failed
        - First n_mels rows: log-mel spectrogram
        - Last row: RMS energy
    
    This preserves time information
    """

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    if chunk_root is None:
        chunk_root = CHUNK_AUDIO_ROOT
    else:
        chunk_root = Path(chunk_root)

    # Parse filename
    filename_path = Path(filename)
    if len(filename_path.parts) > 1:
        species = filename_path.parts[0]
        audio_filename = filename_path.name
    else:
        audio_filename = filename_path.name
        species = None
        
    audio_path = base_dir / chunk_root.relative_to(Path.cwd()) / filename

    if not audio_path.exists():
        augmented_base = base_dir / AUGMENTED_DATA_ROOT.relative_to(Path.cwd()) / "train_audio"
        if augmented_base.exists():
            if species:
                augmented_audio_path = augmented_base / species / audio_filename
                if augmented_audio_path.exists():
                    audio_path = augmented_audio_path
            
            if not audio_path.exists():
                for subdir in augmented_base.iterdir():
                    if subdir.is_dir():
                        potential_path = subdir / audio_filename
                        if potential_path.exists():
                            audio_path = potential_path
                            break

    try:
        # Load audio
        y, _sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        if y.size == 0:
            return None

        # Extract mel spectrogram 
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract RMS
        rms = librosa.feature.rms(y=y)
        
        # Combine: Stack mel + rms vertically
        combined = np.vstack([log_mel, rms])
        
        return combined.astype(np.float32)
        
    except Exception as exc:
        print(f"[WARN] Failed to compute features for {audio_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _get_feature_cache_path(
    spec: FeatureSpec,
    base_dir: Path | str | None = None,
    chunk_root: Path | str | None = None,
) -> Path:
    """Return the cache path for a given feature spec."""

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Determine which features directory to use
    if chunk_root is None:
        features_dir = FEATURES_ROOT
    else:
        chunk_root = Path(chunk_root)
        if chunk_root.name == TRIMMED_CHUNK_ROOT.name or chunk_root == TRIMMED_CHUNK_ROOT:
            features_dir = FEATURES_TRIMMED_ROOT
        else:
            features_dir = FEATURES_ROOT

    cache_dir = base_dir / features_dir.relative_to(Path.cwd())
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    name_parts = [spec.kind]
    if spec.n_mels is not None and spec.kind in ["logmel", "logmel_rms_timeseries"]:
        name_parts.append(f"mels_{spec.n_mels}")

    filename = "features_" + "_".join(name_parts) + ".pkl"
    return cache_dir / filename


def compute_or_load_features(
    spec: FeatureSpec,
    base_dir: Path | str | None = None,
    chunk_root: Path | str | None = None,
    include_augmented: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute features for all samples or load them from cache."""

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Load metadata
    if include_augmented:
        print("🎵 Loading combined metadata (original + augmented)...")
        df = load_combined_metadata(base_dir=base_dir)
        cache_path = _get_feature_cache_path(spec, base_dir=base_dir, chunk_root=chunk_root)
        cache_path = cache_path.with_name(f"combined_{cache_path.name}")
    else:
        df = load_cleaned_metadata(base_dir=base_dir)
        cache_path = _get_feature_cache_path(spec, base_dir=base_dir, chunk_root=chunk_root)
    
    if cache_path.exists():
        print(f"Loading cached {spec.kind} features from {cache_path} ...")
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        return cache["X"], cache["y"]

    feature_cols, target_col = get_feature_and_target_columns(spec)

    required_columns = ["duration", target_col, "filename"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns {missing} in metadata.")

    filenames = df["filename"].tolist()
    durations = df["duration"].to_numpy(dtype=np.float32)
    targets = df[target_col].astype(str)

    if spec.kind == "logmel_rms_timeseries":
        if spec.n_mels is None:
            raise ValueError("n_mels must be set for logmel_rms_timeseries features.")
        
        timeseries_features = []
        valid_indices = []
        
        for idx, fn in enumerate(filenames):
            feats = compute_logmel_rms_timeseries(
                filename=fn,
                base_dir=base_dir,
                chunk_root=chunk_root,
                n_mels=spec.n_mels,
            )
            
            if feats is not None:
                timeseries_features.append(feats)
                valid_indices.append(idx)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(filenames)} files...")
                
        # Pad all to same length
        max_time_frames = max([f.shape[1] for f in timeseries_features])
        print(f"   Max time frames: {max_time_frames}")
        
        padded_features = []
        for feats in timeseries_features:
            if feats.shape[1] < max_time_frames:
                pad_width = ((0, 0), (0, max_time_frames - feats.shape[1]))
                feats_padded = np.pad(feats, pad_width, mode='constant')
            else:
                feats_padded = feats
            padded_features.append(feats_padded)
        
        X = np.array(padded_features, dtype=np.float32)
        y = targets.iloc[valid_indices].to_numpy()
        
        feature_names = [f"mel_{i}" for i in range(spec.n_mels)] + ["rms"]
        
        print(f"Final shape: {X.shape}")
        print(f"Format: (samples, {spec.n_mels + 1} features, time_frames)")

    elif spec.kind == "rms":
        print("Computing RMS features...")
        rms_values = [
            compute_global_rms(fn, base_dir=base_dir, chunk_root=chunk_root) 
            for fn in filenames
        ]
        rms_values = np.asarray(rms_values, dtype=np.float32)
        X = np.stack([durations, rms_values], axis=1)
        y = targets.to_numpy()
        feature_names = ["duration", "audio_rms"]
    elif spec.kind == "logmel":
        if spec.n_mels is None:
            raise ValueError("n_mels must be set for logmel features.")

        print("Computing log-mel features (AVERAGED)...")
        logmel_features = []
        for fn in filenames:
            feats = compute_logmel_mean_features(
                filename=fn,
                base_dir=base_dir,
                chunk_root=chunk_root,
                n_mels=spec.n_mels,
            )
            logmel_features.append(feats)

        logmel_features = np.stack(logmel_features, axis=0)
        durations_col = durations.reshape(-1, 1)
        X = np.concatenate([durations_col, logmel_features], axis=1)
        y = targets.to_numpy()
        feature_names = ["duration"] + [f"logmel_{i}" for i in range(spec.n_mels)]

    else:
        raise ValueError(f"Unknown feature kind: {spec.kind!r}")

    # Drop rows with NaNs
    if spec.kind != "logmel_rms_timeseries":
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

    # Save to cache
    cache = {
        "feature_type": spec.kind,
        "version": 1,
        "feature_names": feature_names,
        "target_name": target_col,
        "X": X,
        "y": y,
    }
    
    if spec.kind == "logmel_rms_timeseries":
        cache["n_mels"] = spec.n_mels
        cache["max_time_frames"] = max_time_frames
        cache["sample_rate"] = 32000
    
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    print(f"Saved {spec.kind} features to {cache_path}")

    return X, y