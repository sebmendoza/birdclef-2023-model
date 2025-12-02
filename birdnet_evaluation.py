"""BirdNET baseline evaluation on BirdCLEF dataset.

This script evaluates the pre-trained BirdNET model's accuracy on the dataset
by running inference on audio files and comparing predictions to ground truth labels.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data import DATA_ROOT, PROJECT_ROOT
from features import _resolve_audio_path


def load_metadata_with_scientific_names(
    base_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load metadata that includes scientific names for species mapping.

    Args:
        base_dir: Base directory for the project

    Returns:
        DataFrame with primary_label, scientific_name, filename, etc.
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT
    else:
        base_dir = Path(base_dir)

    # Load metadata with scientific names
    metadata_path = base_dir / \
        DATA_ROOT.relative_to(PROJECT_ROOT) / \
        "train_metadata_with_duration.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata with scientific names not found at {metadata_path}. "
            "This file is needed for species mapping."
        )

    df = pd.read_csv(metadata_path)

    # Ensure required columns exist
    required_cols = ["primary_label", "scientific_name", "filename"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns {missing} not found in metadata. "
            "Need scientific_name for BirdNET species mapping."
        )

    return df


def create_species_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Create mapping from scientific names to primary_label codes.

    Args:
        df: DataFrame with primary_label and scientific_name columns

    Returns:
        Dictionary mapping scientific_name -> primary_label
    """
    # Create mapping: scientific_name -> primary_label
    # Handle cases where multiple primary_labels might map to same scientific name
    mapping = {}
    for _, row in df.iterrows():
        sci_name = str(row["scientific_name"]).strip()
        primary_label = str(row["primary_label"]).strip()

        # If multiple primary_labels for same scientific name, keep the first one
        # (This shouldn't happen in practice, but handle it gracefully)
        if sci_name not in mapping:
            mapping[sci_name] = primary_label
        elif mapping[sci_name] != primary_label:
            # Log warning if there's a conflict
            print(f"Warning: Scientific name '{sci_name}' maps to multiple primary_labels: "
                  f"'{mapping[sci_name]}' and '{primary_label}'. Using '{mapping[sci_name]}'.")

    # Also create reverse mapping for quick lookup
    return mapping


def predict_with_birdnet(
    audio_path: Path,
    analyzer: Analyzer,
    min_conf: float = 0.1,
) -> List[Dict]:
    """Run BirdNET inference on an audio file.

    Args:
        audio_path: Path to audio file
        analyzer: BirdNET Analyzer instance
        min_conf: Minimum confidence threshold for detections

    Returns:
        List of detections, each with 'common_name', 'scientific_name', 'confidence'

    Raises:
        Exception: If analysis fails (caller should handle)
    """
    # Create recording instance
    # Note: We don't have location/date info, so omit those parameters
    recording = Recording(
        analyzer,
        str(audio_path),
        min_conf=min_conf,
    )

    # Analyze the recording
    recording.analyze()

    # Return detections sorted by confidence (highest first)
    detections = sorted(
        recording.detections,
        key=lambda x: x.get("confidence", 0.0),
        reverse=True,
    )

    return detections


def map_birdnet_prediction_to_primary_label(
    detections: List[Dict],
    species_mapping: Dict[str, str],
    top_k: int = 1,
) -> str | None:
    """Map BirdNET predictions to dataset primary_label.

    Args:
        detections: List of BirdNET detections (sorted by confidence)
        species_mapping: Dictionary mapping scientific_name -> primary_label
        top_k: Use top-k predictions (default: 1 for top-1)

    Returns:
        Primary label if found in top-k, None otherwise
    """
    if not detections:
        return None

    # Check top-k predictions
    for i in range(min(top_k, len(detections))):
        detection = detections[i]
        sci_name = detection.get("scientific_name", "").strip()

        # Try exact match first
        if sci_name in species_mapping:
            return species_mapping[sci_name]

        # Try case-insensitive match
        for mapped_sci_name, primary_label in species_mapping.items():
            if sci_name.lower() == mapped_sci_name.lower():
                return primary_label

    return None


def evaluate_birdnet(
    base_dir: Path | str | None = None,
    chunk_root: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    min_conf: float = 0.1,
    subset: int | None = None,
    top_k: int = 1,
) -> None:
    """Evaluate BirdNET on the dataset.

    Args:
        base_dir: Base directory for the project
        chunk_root: Directory containing audio chunks
        test_size: Fraction of data to use for testing
        random_state: Random seed for train/test split
        min_conf: Minimum confidence threshold for BirdNET detections
        subset: If provided, only evaluate on this many samples (for testing)
        top_k: Evaluate top-k accuracy (default: 1 for top-1)
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT
    else:
        base_dir = Path(base_dir)

    if chunk_root is None:
        # Default to 15secondchunkstrimmed as specified by user
        chunk_root = base_dir / \
            DATA_ROOT.relative_to(PROJECT_ROOT) / "15secondchunkstrimmed"
    else:
        chunk_root = Path(chunk_root)

    print("=" * 60)
    print("BirdNET Baseline Evaluation")
    print("=" * 60)

    # Load metadata with scientific names
    print("\n📊 Loading metadata...")
    df = load_metadata_with_scientific_names(base_dir=base_dir)

    # Create species mapping
    print("🔗 Creating species mapping...")
    species_mapping = create_species_mapping(df)
    print(f"   Mapped {len(species_mapping)} species")

    # Filter to only samples that exist
    print("\n🔍 Checking audio file availability...")
    valid_indices = []
    for idx, row in df.iterrows():
        filename = row["filename"]
        audio_path = _resolve_audio_path(filename, base_dir, chunk_root)
        if audio_path.exists():
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].copy()
    print(f"   Found {len(df_valid)} valid audio files out of {len(df)} total")

    # Optionally subset for faster testing
    if subset is not None and subset < len(df_valid):
        print(f"\n⚠️  Using subset of {subset} samples for evaluation")
        df_valid = df_valid.sample(
            n=subset, random_state=random_state).reset_index(drop=True)

    # Split into train/test (for consistency with other models, though we don't train)
    print(f"\n📂 Splitting data (test_size={test_size})...")
    train_df, test_df = train_test_split(
        df_valid,
        test_size=test_size,
        random_state=random_state,
        stratify=df_valid["primary_label"],
    )

    print(f"   Train: {len(train_df)} samples")
    print(f"   Test: {len(test_df)} samples")

    # Initialize BirdNET analyzer
    print("\n🦅 Initializing BirdNET analyzer...")
    analyzer = Analyzer()

    # Run inference on test set
    print(f"\n🎵 Running BirdNET inference on {len(test_df)} test samples...")
    print(f"   (This may take a while...)")

    predictions = []
    ground_truth = []
    confidences = []
    failed_files = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        filename = row["filename"]
        primary_label = row["primary_label"]

        # Resolve audio path
        audio_path = _resolve_audio_path(filename, base_dir, chunk_root)

        if not audio_path.exists():
            print(f"\nWarning: Audio file not found: {audio_path}")
            failed_files.append(str(audio_path))
            continue

        # Run BirdNET inference
        try:
            detections = predict_with_birdnet(
                audio_path, analyzer, min_conf=min_conf)
        except Exception as e:
            print(f"\nWarning: Failed to analyze {audio_path}: {e}")
            failed_files.append(str(audio_path))
            continue

        # Map prediction to primary_label
        predicted_label = map_birdnet_prediction_to_primary_label(
            detections, species_mapping, top_k=top_k
        )

        # Store results
        ground_truth.append(primary_label)
        if predicted_label is not None:
            predictions.append(predicted_label)
            # Get confidence of top prediction
            top_conf = detections[0].get(
                "confidence", 0.0) if detections else 0.0
            confidences.append(top_conf)
        else:
            # If no prediction found, use a placeholder or skip
            # For evaluation, we'll use a special "unknown" label
            predictions.append("unknown")
            confidences.append(0.0)

    if failed_files:
        print(
            f"\n⚠️  Failed to process {len(failed_files)} files (out of {len(test_df)} total)")
        if len(failed_files) <= 10:
            print("Failed files:")
            for f in failed_files:
                print(f"  - {f}")

    # Convert to numpy arrays
    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)

    # Calculate accuracy
    # For "unknown" predictions, count as incorrect
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Top-{top_k} Accuracy: {accuracy:.4f}")

    # Calculate per-class metrics
    print("\n📈 Classification Report:")
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    # Save results
    _save_birdnet_results(
        experiment_name="BirdNET Baseline",
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        classification_report=report,
        accuracy=accuracy,
        base_dir=base_dir,
        top_k=top_k,
        min_conf=min_conf,
        n_samples=len(test_df),
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


def _save_birdnet_results(
    experiment_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: List[float],
    classification_report: dict,
    accuracy: float,
    base_dir: Path,
    top_k: int,
    min_conf: float,
    n_samples: int,
) -> None:
    """Save BirdNET evaluation results.

    Args:
        experiment_name: Name of the experiment
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidences: Confidence scores for predictions
        classification_report: Classification report dictionary
        accuracy: Overall accuracy
        base_dir: Base directory for saving results
        top_k: Top-k value used
        min_conf: Minimum confidence threshold used
        n_samples: Number of samples evaluated
    """
    # Create results directory
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    exp_dir_name = "birdnet_baseline"
    exp_dir = results_dir / exp_dir_name
    exp_dir.mkdir(exist_ok=True)

    # Save metrics
    metrics = {
        "experiment_name": experiment_name,
        "final_accuracy": float(accuracy),
        "top_k": top_k,
        "min_confidence_threshold": min_conf,
        "n_samples": n_samples,
        "classification_report": classification_report,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "median_confidence": float(np.median(confidences)) if confidences else 0.0,
    }

    metrics_path = exp_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics to {metrics_path}")

    # Save classification report as text
    report_path = exp_dir / "classification_report.txt"
    with report_path.open("w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Top-{top_k} Accuracy: {accuracy:.4f}\n")
        f.write(f"Minimum Confidence Threshold: {min_conf}\n")
        f.write(f"Number of Samples: {n_samples}\n")
        f.write(f"Mean Confidence: {metrics['mean_confidence']:.4f}\n")
        f.write(f"Median Confidence: {metrics['median_confidence']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(classification_report))
    print(f"💾 Saved classification report to {report_path}")

    # Create a simple accuracy visualization
    # (Since we don't have training epochs, just show per-class accuracy)
    if len(np.unique(y_true)) <= 20:  # Only plot if reasonable number of classes
        plt.figure(figsize=(12, 8))

        # Get per-class accuracies
        unique_labels = sorted(set(y_true) | set(y_pred))
        class_accuracies = []
        class_names = []

        for label in unique_labels:
            if label == "unknown":
                continue
            mask = y_true == label
            if mask.sum() > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                class_accuracies.append(class_acc)
                class_names.append(label)

        if class_accuracies:
            plt.barh(range(len(class_names)), class_accuracies)
            plt.yticks(range(len(class_names)), class_names)
            plt.xlabel("Accuracy")
            plt.ylabel("Species")
            plt.title(f"BirdNET Per-Class Accuracy (Top-{top_k})")
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()

            plot_path = exp_dir / "per_class_accuracy.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"💾 Saved per-class accuracy plot to {plot_path}")


def main():
    """Main entry point for BirdNET evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained BirdNET on BirdCLEF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python birdnet_evaluation.py                    # Full evaluation
  python birdnet_evaluation.py --subset 100        # Test on 100 samples
  python birdnet_evaluation.py --top-k 3          # Evaluate top-3 accuracy
  python birdnet_evaluation.py --min-conf 0.25    # Use higher confidence threshold
        """,
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Evaluate on a subset of samples (for faster testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Evaluate top-k accuracy (default: 1)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.1,
        help="Minimum confidence threshold for BirdNET detections (default: 0.1)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    args = parser.parse_args()

    evaluate_birdnet(
        test_size=args.test_size,
        random_state=args.random_state,
        min_conf=args.min_conf,
        subset=args.subset,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
