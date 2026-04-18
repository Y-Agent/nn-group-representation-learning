"""Configuration management for spherical neural network training."""

import yaml
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Union, List
from functools import reduce
from operator import mul
from pathlib import Path


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    """Configuration container for training experiments."""

    # Task
    task_name: str = "modular_addition"
    p: int = 23  # Single modulus (1-d case, backward compatible)
    n: Optional[List[int]] = None  # Moduli list for d-dimensional case (overrides p if set)

    # Basis
    basis_name: str = "fourier"

    # Model
    width: int = 128
    act_type: str = "Quad"
    init_scale: float = 0.5
    share_embed: bool = True  # If False, use separate embeddings per position (for non-commutative ops)
    # Initialization type: "iid" or "matched_scale"
    # Note: "matched_scale" only works for modular tasks with Fourier basis
    init_type: str = "iid"

    # Stage 1
    stage1_enabled: bool = True
    stage1_lr: float = 0.001
    stage1_num_epochs: int = 6000
    stage1_optimizer: str = "Adam"
    stage1_freeze_scales: bool = True

    # Stage 2
    stage2_enabled: bool = True
    stage2_lr: float = 5.0
    stage2_num_epochs: int = 50000
    stage2_optimizer: str = "SGD"
    stage2_train_directions: bool = True
    stage2_train_scales: bool = False

    # Logging
    print_every: int = 2000
    save_every: int = 1000
    track_phases: bool = True
    track_scales: bool = True
    phase_save_every: int = 200
    snapshot_every: int = 0  # Save model parameter snapshots every N epochs (0 = disabled)

    # System
    seed: int = 42
    device: str = "auto"

    # Computed fields
    _device: Optional[torch.device] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize computed fields and set seeds."""
        set_all_seeds(self.seed)

        if self.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device)

    @property
    def torch_device(self) -> torch.device:
        """Get the torch device."""
        return self._device

    @property
    def moduli(self) -> List[int]:
        """Get moduli list (normalized from p or n)."""
        if self.n is not None:
            return self.n
        return [self.p]

    @property
    def d_vocab(self) -> int:
        """Vocabulary size (product of moduli for modular arithmetic tasks)."""
        return reduce(mul, self.moduli, 1)

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "Config":
        """Load configuration from YAML file with optional overrides."""
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)

        # Flatten nested structure
        flat_dict = {}

        # Task
        if "task" in yaml_dict:
            flat_dict["task_name"] = yaml_dict["task"].get("name", "modular_addition")
            flat_dict["p"] = yaml_dict["task"].get("p", 23)
            if "n" in yaml_dict["task"]:
                flat_dict["n"] = yaml_dict["task"]["n"]

        # Basis
        if "basis" in yaml_dict:
            flat_dict["basis_name"] = yaml_dict["basis"].get("name", "fourier")

        # Model
        if "model" in yaml_dict:
            flat_dict["width"] = yaml_dict["model"].get("width", 128)
            flat_dict["act_type"] = yaml_dict["model"].get("act_type", "Quad")
            flat_dict["init_scale"] = yaml_dict["model"].get("init_scale", 0.5)
            flat_dict["share_embed"] = yaml_dict["model"].get("share_embed", True)
            flat_dict["init_type"] = yaml_dict["model"].get("init_type", "iid")

        # Stage 1
        if "stage1" in yaml_dict:
            flat_dict["stage1_enabled"] = yaml_dict["stage1"].get("enabled", True)
            flat_dict["stage1_lr"] = yaml_dict["stage1"].get("lr", 0.001)
            flat_dict["stage1_num_epochs"] = yaml_dict["stage1"].get("num_epochs", 6000)
            flat_dict["stage1_optimizer"] = yaml_dict["stage1"].get("optimizer", "Adam")
            flat_dict["stage1_freeze_scales"] = yaml_dict["stage1"].get("freeze_scales", True)

        # Stage 2
        if "stage2" in yaml_dict:
            flat_dict["stage2_enabled"] = yaml_dict["stage2"].get("enabled", True)
            flat_dict["stage2_lr"] = yaml_dict["stage2"].get("lr", 5.0)
            flat_dict["stage2_num_epochs"] = yaml_dict["stage2"].get("num_epochs", 50000)
            flat_dict["stage2_optimizer"] = yaml_dict["stage2"].get("optimizer", "SGD")
            flat_dict["stage2_train_directions"] = yaml_dict["stage2"].get("train_directions", True)
            flat_dict["stage2_train_scales"] = yaml_dict["stage2"].get("train_scales", False)

        # Logging
        if "logging" in yaml_dict:
            flat_dict["print_every"] = yaml_dict["logging"].get("print_every", 2000)
            flat_dict["save_every"] = yaml_dict["logging"].get("save_every", 1000)
            flat_dict["track_phases"] = yaml_dict["logging"].get("track_phases", True)
            flat_dict["track_scales"] = yaml_dict["logging"].get("track_scales", True)
            flat_dict["phase_save_every"] = yaml_dict["logging"].get("phase_save_every", 200)
            flat_dict["snapshot_every"] = yaml_dict["logging"].get("snapshot_every", 0)

        # System
        flat_dict["seed"] = yaml_dict.get("seed", 42)
        flat_dict["device"] = yaml_dict.get("device", "auto")

        # Apply overrides
        flat_dict.update(overrides)

        return cls(**flat_dict)

    @classmethod
    def from_dict(cls, config_dict: dict, **overrides) -> "Config":
        """Create configuration from dictionary with optional overrides."""
        config_dict.update(overrides)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "task_name": self.task_name,
            "p": self.p,
            "n": self.n,
            "basis_name": self.basis_name,
            "width": self.width,
            "act_type": self.act_type,
            "init_scale": self.init_scale,
            "share_embed": self.share_embed,
            "init_type": self.init_type,
            "stage1_enabled": self.stage1_enabled,
            "stage1_lr": self.stage1_lr,
            "stage1_num_epochs": self.stage1_num_epochs,
            "stage1_optimizer": self.stage1_optimizer,
            "stage1_freeze_scales": self.stage1_freeze_scales,
            "stage2_enabled": self.stage2_enabled,
            "stage2_lr": self.stage2_lr,
            "stage2_num_epochs": self.stage2_num_epochs,
            "stage2_optimizer": self.stage2_optimizer,
            "stage2_train_directions": self.stage2_train_directions,
            "stage2_train_scales": self.stage2_train_scales,
            "print_every": self.print_every,
            "save_every": self.save_every,
            "track_phases": self.track_phases,
            "track_scales": self.track_scales,
            "phase_save_every": self.phase_save_every,
            "snapshot_every": self.snapshot_every,
            "seed": self.seed,
            "device": self.device,
        }
