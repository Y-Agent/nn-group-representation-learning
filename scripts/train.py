#!/usr/bin/env python
"""Main training script for spherical neural network learning."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.trainer import SphericalTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train wide neural network with spherical parameterization"
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)"
    )

    # Task arguments
    parser.add_argument("--task", type=str, help="Task name (e.g., modular_addition)")
    parser.add_argument("--p", type=int, help="Prime modulus for modular arithmetic")

    # Basis arguments
    parser.add_argument("--basis", type=str, help="Basis type (e.g., fourier)")

    # Model arguments
    parser.add_argument("--width", "-M", type=int, help="Number of neurons (width)")
    parser.add_argument("--act", type=str, help="Activation type (ReLU, Quad, Abs, GeLU)")
    parser.add_argument("--init-scale", type=float, help="Initial scale value")
    parser.add_argument("--no-share-embed", action="store_true",
                       help="Use separate embeddings per position (for non-commutative ops like subtraction)")

    # Stage 1 arguments
    parser.add_argument("--stage1-lr", type=float, help="Stage 1 learning rate")
    parser.add_argument("--stage1-epochs", type=int, help="Stage 1 number of epochs")
    parser.add_argument("--stage1-optimizer", type=str, help="Stage 1 optimizer (Adam, SGD)")
    parser.add_argument("--no-stage1", action="store_true", help="Disable Stage 1")

    # Stage 2 arguments
    parser.add_argument("--stage2-lr", type=float, help="Stage 2 learning rate")
    parser.add_argument("--stage2-epochs", type=int, help="Stage 2 number of epochs")
    parser.add_argument("--stage2-optimizer", type=str, help="Stage 2 optimizer (Adam, SGD)")
    parser.add_argument("--no-stage2", action="store_true", help="Disable Stage 2")
    parser.add_argument("--stage2-train-scales", action="store_true", help="Train scales in Stage 2")
    parser.add_argument("--stage2-no-directions", action="store_true", help="Don't train directions in Stage 2")

    # Logging arguments
    parser.add_argument("--print-every", type=int, help="Print progress every N epochs")
    parser.add_argument("--save-every", type=int, help="Save checkpoint every N epochs")

    # System arguments
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Device (auto, cuda, cpu)")

    # Output
    parser.add_argument("--save-path", type=str, help="Path to save final checkpoint")
    parser.add_argument("--analyze", action="store_true", help="Run phase alignment analysis after training")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    # Build overrides from command line
    overrides = {}

    if args.task:
        overrides["task_name"] = args.task
    if args.p:
        overrides["p"] = args.p
    if args.basis:
        overrides["basis_name"] = args.basis
    if args.width:
        overrides["width"] = args.width
    if args.act:
        overrides["act_type"] = args.act
    if args.init_scale:
        overrides["init_scale"] = args.init_scale
    if args.no_share_embed:
        overrides["share_embed"] = False

    # Stage 1
    if args.stage1_lr:
        overrides["stage1_lr"] = args.stage1_lr
    if args.stage1_epochs:
        overrides["stage1_num_epochs"] = args.stage1_epochs
    if args.stage1_optimizer:
        overrides["stage1_optimizer"] = args.stage1_optimizer
    if args.no_stage1:
        overrides["stage1_enabled"] = False

    # Stage 2
    if args.stage2_lr:
        overrides["stage2_lr"] = args.stage2_lr
    if args.stage2_epochs:
        overrides["stage2_num_epochs"] = args.stage2_epochs
    if args.stage2_optimizer:
        overrides["stage2_optimizer"] = args.stage2_optimizer
    if args.no_stage2:
        overrides["stage2_enabled"] = False
    if args.stage2_train_scales:
        overrides["stage2_train_scales"] = True
    if args.stage2_no_directions:
        overrides["stage2_train_directions"] = False

    # Logging
    if args.print_every:
        overrides["print_every"] = args.print_every
    if args.save_every:
        overrides["save_every"] = args.save_every

    # System
    if args.seed:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device

    # Load config
    config = Config.from_yaml(str(config_path), **overrides)

    print("=" * 60)
    print("Spherical Neural Network Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Task: {config.task_name} (p={config.p})")
    print(f"Model: width={config.width}, act={config.act_type}, init_scale={config.init_scale}, share_embed={config.share_embed}")
    print(f"Stage 1: enabled={config.stage1_enabled}, lr={config.stage1_lr}, epochs={config.stage1_num_epochs}")
    print(f"Stage 2: enabled={config.stage2_enabled}, lr={config.stage2_lr}, epochs={config.stage2_num_epochs}")
    print("=" * 60)

    # Create trainer
    trainer = SphericalTrainer(config)

    # Train
    stage1_history, stage2_history = trainer.train()

    # Analyze if requested
    if args.analyze:
        trainer.analyze_phase_alignment(verbose=True)

    # Save checkpoint if requested
    if args.save_path:
        trainer.save_checkpoint(args.save_path, stage="final")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Summary
    if stage1_history.losses:
        print(f"Stage 1: Final loss={stage1_history.losses[-1]:.6f}, "
              f"Final acc={stage1_history.accuracies[-1]:.4f}")
    if stage2_history.losses:
        print(f"Stage 2: Final loss={stage2_history.losses[-1]:.6f}, "
              f"Final acc={stage2_history.accuracies[-1]:.4f}")


if __name__ == "__main__":
    main()
