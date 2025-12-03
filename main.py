"""Entry point for running simple BirdCLEF baseline models.

This script trains models on features computed from 15-second chunks.
Experiments are defined in config.py.

Usage:
    python main.py                    # Run all experiments
    python main.py lr                 # Run only logistic regression models
    python main.py rf                 # Run only random forest models
    python main.py --include-augmented # Include augmented data
"""

from __future__ import annotations

from pathlib import Path

from config import get_experiments_by_model_type
from data import TRIMMED_CHUNK_ROOT
from models import train_and_evaluate_experiment


def main(model_type: str | None = None, include_augmented: bool = False, n_epochs: int = 50) -> None:
    """Run baseline experiments with optional model filtering.

    Args:
        model_type: Model type to run ("lr", "rf", or None for all)
        include_augmented: Whether to include augmented data in training
        n_epochs: Number of training epochs
    """
    base_dir = Path.cwd()

    # Get experiments based on model type filter
    experiment_defs = get_experiments_by_model_type(model_type)

    if not experiment_defs:
        print(f"No experiments found for model type: {model_type}")
        return

    print(f"🎵 Running {len(experiment_defs)} experiment(s)")
    print(f"   Model filter: {model_type if model_type else 'all'}")
    print(f"   Augmented data: {include_augmented}")
    print(f"   Epochs: {n_epochs}")
    print("=" * 50)

    for exp_def in experiment_defs:
        cfg = exp_def.to_experiment_config()
        train_and_evaluate_experiment(
            cfg,
            base_dir=base_dir,
            chunk_root=TRIMMED_CHUNK_ROOT,
            include_augmented=include_augmented,
            n_epochs=n_epochs,
        )
        print("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train BirdCLEF baseline models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run all experiments
  python main.py lr                 # Run only logistic regression
  python main.py rf                 # Run only random forest
  python main.py --include-augmented # Include augmented data
  python main.py lr --n-epochs 100   # Run LR with 100 epochs
        """
    )
    parser.add_argument(
        "model_type",
        nargs="?",
        default=None,
        help="Model type to run: 'lr' for logistic regression, 'rf' for random forest, or omit for all"
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include augmented data in training"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )

    args = parser.parse_args()
    main(
        model_type=args.model_type,
        include_augmented=args.include_augmented,
        n_epochs=args.n_epochs,
    )
