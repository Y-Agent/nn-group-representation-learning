"""Spherical gradient descent trainer for wide neural networks."""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json

from .config import Config
from .model import WideNetworkScaleSphere
from .tasks import get_task, Task
from .basis import get_basis, Basis
from .utils import (
    cross_entropy_high_precision,
    accuracy,
    project_gradient_to_tangent_space,
    compute_gradient_norms,
    compute_param_norms,
    normalize_to_pi,
)


@dataclass
class TrainingHistory:
    """Container for training history."""
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    grad_norms: List[Tuple[float, float]] = field(default_factory=list)
    param_norms: List[Tuple[float, float, float, float]] = field(default_factory=list)
    phase_history: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    scale_history: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    epochs_recorded: List[int] = field(default_factory=list)
    # Model parameter snapshots: list of (epoch, W_in, W_out) tuples
    # W_in: (width, d_input), W_out: (d_vocab, width)
    param_snapshots: List[Tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)


class SphericalTrainer:
    """Trainer implementing spherical (projected) gradient descent.

    Stage 1: Projected GD on unit sphere
        - Directions (W_in, W_out) are trained with gradients projected to tangent space
        - Scales (scale_in, scale_out) are frozen
        - After each step, directions are re-normalized to unit sphere

    Stage 2: Standard GD (optional)
        - Can train directions, scales, or both
        - No projection or normalization constraints
    """

    def __init__(self, config: Config):
        """Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.torch_device

        # Initialize task and basis
        # Use n if specified, otherwise fall back to p
        if config.n is not None:
            self.task: Task = get_task(config.task_name, n=config.n)
            self.basis: Basis = get_basis(config.basis_name, n=config.n)
        else:
            self.task: Task = get_task(config.task_name, p=config.p)
            self.basis: Basis = get_basis(config.basis_name, n=config.p)

        # Generate data
        self.data, self.labels = self.task.generate_all_data(self.device)

        # Initialize model
        self.model = WideNetworkScaleSphere(
            d_vocab=config.d_vocab,
            width=config.width,
            act_type=config.act_type,
            init_scale=config.init_scale,
            share_embed=config.share_embed,
        ).to(self.device)

        # Apply custom initialization (only if not default 'iid', which model already does)
        init_type = getattr(config, 'init_type', 'iid')
        if init_type != 'iid':
            self._initialize_weights(init_type)

        # Training history
        self.stage1_history = TrainingHistory()
        self.stage2_history = TrainingHistory()

        # Checkpoints
        self.initial_state: Optional[Dict[str, torch.Tensor]] = None
        self.stage1_checkpoint: Optional[Dict[str, torch.Tensor]] = None

        print(f"Initialized SphericalTrainer:")
        print(f"  Task: {self.task}")
        print(f"  Basis: {self.basis}")
        print(f"  Device: {self.device}")
        print(f"  Data size: {len(self.data)}")
        print(f"  Model:\n{self.model}")

    def _initialize_weights(self, init_type: str) -> None:
        """Initialize model weights based on specified initialization type.

        Args:
            init_type: Initialization type, one of:
                - "iid": Both θ (W_in) and ξ (W_out) are IID on unit sphere
                - "matched_scale": θ is IID on sphere, ξ has same Fourier
                  magnitudes as θ but with random phases.
                  NOTE: Only works for modular tasks with Fourier basis.
        """
        if init_type == "iid":
            self._init_iid_sphere()
        elif init_type == "matched_scale":
            # Check that we have a Fourier basis (required for matched_scale)
            from .basis.fourier import FourierBasis
            if not isinstance(self.basis, FourierBasis):
                raise ValueError(
                    f"init_type='matched_scale' requires FourierBasis, "
                    f"but got {type(self.basis).__name__}. "
                    f"This initialization only works for modular tasks."
                )
            self._init_matched_scale()
        else:
            raise ValueError(f"Unknown init_type: {init_type}. "
                           f"Must be 'iid' or 'matched_scale'.")

    def _init_iid_sphere(self) -> None:
        """IID initialization: both θ and ξ are uniformly random on unit sphere."""
        with torch.no_grad():
            # W_in: each row is IID on unit sphere
            w_in = torch.randn(self.model.width, self.model.d_input, device=self.device)
            w_in = w_in / w_in.norm(dim=1, keepdim=True)
            self.model.W_in.data = w_in

            # W_out: each column is IID on unit sphere
            w_out = torch.randn(self.model.d_vocab, self.model.width, device=self.device)
            w_out = w_out / w_out.norm(dim=0, keepdim=True)
            self.model.W_out.data = w_out

    def _init_matched_scale(self) -> None:
        """Matched-scale initialization for modular tasks with Fourier basis.

        θ is IID on sphere, ξ has same Fourier magnitudes as θ but random phases.

        For each neuron m:
        1. θ_m is IID on unit sphere (standard initialization)
        2. ξ_m has |ξ̂_m[k]| = |θ̂_m[k]| for all frequencies k,
           but with independently random phases (preserving conjugate symmetry)

        This ensures ξ has the same "energy" at each frequency as θ.

        NOTE: This initialization only works for modular tasks (modular_addition,
        modular_subtraction, etc.) that use FourierBasis.
        """
        with torch.no_grad():
            # W_in: each row is IID on unit sphere
            w_in = torch.randn(self.model.width, self.model.d_input, device=self.device)
            w_in = w_in / w_in.norm(dim=1, keepdim=True)
            self.model.W_in.data = w_in

            # Compute Fourier transform of W_in (move to CPU for basis transform)
            w_in_cpu = w_in.cpu()
            W_in_dft = self.basis.transform(w_in_cpu)  # (width, d_vocab), complex

            # Get magnitudes from W_in
            magnitudes = torch.abs(W_in_dft)  # (width, d_vocab)

            # Generate random phases for W_out with conjugate symmetry for real output
            # For real signals: X[-k] = conj(X[k]), so phase[-k] = -phase[k]
            d_vocab = self.model.d_vocab
            width = self.model.width
            random_phases = torch.zeros(width, d_vocab)

            # Get conjugate frequency mapping from basis
            for k in range(d_vocab):
                k_tuple = self.basis.get_frequency_tuple(k)
                # Compute -k mod n for each dimension
                neg_k_tuple = tuple((-kj) % nj for kj, nj in zip(k_tuple, self.basis.n))
                neg_k_idx = self.basis.tuple_to_index(neg_k_tuple)

                if neg_k_idx == k:
                    # Self-conjugate frequency (k = -k): phase must be 0 or pi for real output
                    # Use 0 or pi randomly
                    random_phases[:, k] = np.pi * torch.randint(0, 2, (width,)).float()
                elif neg_k_idx > k:
                    # Only set phase for k, then set -k = -phase[k]
                    random_phases[:, k] = 2 * np.pi * torch.rand(width)
                    random_phases[:, neg_k_idx] = -random_phases[:, k]
                # else: neg_k_idx < k, already set

            # Construct W_out in Fourier domain with same magnitudes but random phases
            W_out_dft = magnitudes * torch.exp(1j * random_phases)  # (width, d_vocab), complex

            # Inverse transform to get W_out in spatial domain
            w_out = self.basis.inverse_transform(W_out_dft)  # (width, d_vocab), complex

            # Take real part (should be nearly real due to conjugate symmetry)
            w_out = w_out.real  # (width, d_vocab)

            # Transpose to get (d_vocab, width) - no normalization needed if conjugate symmetry is correct
            w_out = w_out.T  # (d_vocab, width)

            # Normalize to unit norm (should already be close to 1 due to Parseval)
            w_out = w_out / w_out.norm(dim=0, keepdim=True)

            self.model.W_out.data = w_out.float().to(self.device)

    def _create_optimizer(
        self,
        params: List[torch.nn.Parameter],
        optimizer_name: str,
        lr: float
    ) -> torch.optim.Optimizer:
        """Create optimizer.

        Args:
            params: Parameters to optimize
            optimizer_name: "Adam" or "SGD"
            lr: Learning rate

        Returns:
            Optimizer instance
        """
        if optimizer_name == "Adam":
            return optim.Adam(params, lr=lr)
        elif optimizer_name == "SGD":
            return optim.SGD(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _compute_loss_and_acc(self) -> Tuple[torch.Tensor, float]:
        """Compute loss and accuracy on full dataset.

        Returns:
            Tuple of (loss_tensor, accuracy_float)
        """
        logits = self.model(self.data)
        loss = cross_entropy_high_precision(logits, self.labels)
        acc = accuracy(logits, self.labels)
        return loss, acc

    def _record_phase_and_scale(
        self,
        history: TrainingHistory,
        epoch: int
    ) -> None:
        """Record phase and scale information for analysis.

        Args:
            history: Training history to update
            epoch: Current epoch
        """
        with torch.no_grad():
            W_in = self.model.W_in.detach().cpu()
            W_out = self.model.W_out.detach().cpu()

            # Compute scales (norms)
            scales_in = W_in.norm(dim=1).numpy()
            scales_out = W_out.norm(dim=0).numpy()

            if self.config.track_scales:
                history.scale_history.append((scales_in.copy(), scales_out.copy()))

            # Compute phases via DFT
            if self.config.track_phases:
                # Normalize for DFT analysis
                W_in_norm = W_in / W_in.norm(dim=1, keepdim=True)
                W_out_norm = W_out / W_out.norm(dim=0, keepdim=True)

                # Apply Fourier transform (handles complex basis)
                W_in_dft = self.basis.transform(W_in_norm)
                W_out_dft = self.basis.transform(W_out_norm.T)

                phases_in = []
                phases_out = []
                for m in range(self.config.width):
                    # Get dominant frequency
                    dom_freq_in = self.basis.get_dominant_frequency(W_in_dft[m])
                    dom_freq_out = self.basis.get_dominant_frequency(W_out_dft[m])

                    # Get phase
                    phase_in = self.basis.get_phase(W_in_dft[m], dom_freq_in)
                    phase_out = self.basis.get_phase(W_out_dft[m], dom_freq_out)

                    phases_in.append(phase_in)
                    phases_out.append(phase_out)

                history.phase_history.append((np.array(phases_in), np.array(phases_out)))

            history.epochs_recorded.append(epoch)

    def _save_snapshot(
        self,
        history: TrainingHistory,
        epoch: int
    ) -> None:
        """Save a snapshot of model parameters.

        Args:
            history: Training history to update
            epoch: Current epoch
        """
        with torch.no_grad():
            W_in = self.model.W_in.detach().cpu().numpy().copy()
            W_out = self.model.W_out.detach().cpu().numpy().copy()
            history.param_snapshots.append((epoch, W_in, W_out))

    def train_stage1(self) -> TrainingHistory:
        """Run Stage 1: Projected gradient descent on unit sphere.

        Trains only directions (W_in, W_out) with spherical constraint.
        Scales (scale_in, scale_out) are frozen.

        Returns:
            Training history for stage 1
        """
        if not self.config.stage1_enabled:
            print("Stage 1 disabled, skipping...")
            return self.stage1_history

        print("\n" + "=" * 60)
        print("Stage 1: Projected GD (Directions only, Scale fixed)")
        print("=" * 60)
        print(f"Fixed scale_in = {self.model.scale_in.item():.4f}, "
              f"scale_out = {self.model.scale_out.item():.4f}")

        # Save initial state
        self.initial_state = self.model.get_state()

        # Freeze scales
        if self.config.stage1_freeze_scales:
            self.model.scale_in.requires_grad = False
            self.model.scale_out.requires_grad = False

        # Create optimizer for directions only
        # When share_embed=True, use half lr for W_in (theta) since gradients are doubled
        if self.config.share_embed:
            param_groups = [
                {'params': [self.model.W_in], 'lr': self.config.stage1_lr / 2},
                {'params': [self.model.W_out], 'lr': self.config.stage1_lr},
            ]
            if self.config.stage1_optimizer == "Adam":
                optimizer = optim.Adam(param_groups)
            elif self.config.stage1_optimizer == "SGD":
                optimizer = optim.SGD(param_groups)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.stage1_optimizer}")
            print(f"Learning rates: W_in={self.config.stage1_lr/2:.6f}, W_out={self.config.stage1_lr:.6f}")
        else:
            direction_params = self.model.get_direction_params()
            optimizer = self._create_optimizer(
                direction_params,
                self.config.stage1_optimizer,
                self.config.stage1_lr
            )

        for epoch in range(self.config.stage1_num_epochs):
            optimizer.zero_grad()

            # Forward and backward
            loss, acc = self._compute_loss_and_acc()
            loss.backward()

            # Project gradients onto tangent space of sphere
            with torch.no_grad():
                # W_in: rows are unit vectors
                self.model.W_in.grad = project_gradient_to_tangent_space(
                    self.model.W_in.data,
                    self.model.W_in.grad,
                    dim=1
                )

                # W_out: columns are unit vectors
                self.model.W_out.grad = project_gradient_to_tangent_space(
                    self.model.W_out.data,
                    self.model.W_out.grad,
                    dim=0
                )

            # Optimizer step
            optimizer.step()

            # Project back onto sphere
            self.model.normalize_directions()

            # Record history
            self.stage1_history.losses.append(loss.item())
            self.stage1_history.accuracies.append(acc)

            grad_norms = compute_gradient_norms(self.model)
            self.stage1_history.grad_norms.append(grad_norms)

            param_norms = compute_param_norms(self.model)
            self.stage1_history.param_norms.append(param_norms)

            # Record phase/scale periodically
            if epoch % self.config.phase_save_every == 0:
                self._record_phase_and_scale(self.stage1_history, epoch)

            # Save parameter snapshots periodically
            snapshot_every = getattr(self.config, 'snapshot_every', 0)
            if snapshot_every > 0 and epoch % snapshot_every == 0:
                self._save_snapshot(self.stage1_history, epoch)

            # Print progress
            if epoch % self.config.print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | Acc: {acc:.4f}")

        print(f"\nStage 1 Final | Loss: {self.stage1_history.losses[-1]:.6f} | "
              f"Acc: {self.stage1_history.accuracies[-1]:.4f}")

        # Save checkpoint
        self.stage1_checkpoint = self.model.get_state()

        return self.stage1_history

    def train_stage2(self) -> TrainingHistory:
        """Run Stage 2: Standard gradient descent (no spherical constraint).

        Can train directions, scales, or both based on configuration.

        Returns:
            Training history for stage 2
        """
        if not self.config.stage2_enabled:
            print("Stage 2 disabled, skipping...")
            return self.stage2_history

        print("\n" + "=" * 60)
        print("Stage 2: Standard GD")
        print("=" * 60)
        print(f"Training directions: {self.config.stage2_train_directions}")
        print(f"Training scales: {self.config.stage2_train_scales}")

        # Set requires_grad based on config
        self.model.W_in.requires_grad = self.config.stage2_train_directions
        self.model.W_out.requires_grad = self.config.stage2_train_directions
        self.model.scale_in.requires_grad = self.config.stage2_train_scales
        self.model.scale_out.requires_grad = self.config.stage2_train_scales

        # Collect trainable parameters with appropriate learning rates
        # When share_embed=True, use half lr for W_in (theta) since gradients are doubled
        if self.config.stage2_train_directions and self.config.share_embed:
            param_groups = [
                {'params': [self.model.W_in], 'lr': self.config.stage2_lr / 2},
                {'params': [self.model.W_out], 'lr': self.config.stage2_lr},
            ]
            if self.config.stage2_train_scales:
                param_groups.append({'params': self.model.get_scale_params(), 'lr': self.config.stage2_lr})

            if self.config.stage2_optimizer == "Adam":
                optimizer = optim.Adam(param_groups)
            elif self.config.stage2_optimizer == "SGD":
                optimizer = optim.SGD(param_groups)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.stage2_optimizer}")
            print(f"Learning rates: W_in={self.config.stage2_lr/2:.6f}, W_out={self.config.stage2_lr:.6f}")
        else:
            params = []
            if self.config.stage2_train_directions:
                params.extend(self.model.get_direction_params())
            if self.config.stage2_train_scales:
                params.extend(self.model.get_scale_params())

            if not params:
                print("No parameters to train in Stage 2!")
                return self.stage2_history

            optimizer = self._create_optimizer(
                params,
                self.config.stage2_optimizer,
                self.config.stage2_lr
            )

        for epoch in range(self.config.stage2_num_epochs):
            optimizer.zero_grad()

            # Forward and backward
            loss, acc = self._compute_loss_and_acc()
            loss.backward()

            # Standard optimizer step (no projection)
            optimizer.step()

            # Record history
            self.stage2_history.losses.append(loss.item())
            self.stage2_history.accuracies.append(acc)

            grad_norms = compute_gradient_norms(self.model)
            self.stage2_history.grad_norms.append(grad_norms)

            param_norms = compute_param_norms(self.model)
            self.stage2_history.param_norms.append(param_norms)

            # Record phase/scale periodically
            if epoch % self.config.phase_save_every == 0:
                self._record_phase_and_scale(self.stage2_history, epoch)

            # Save parameter snapshots periodically
            snapshot_every = getattr(self.config, 'snapshot_every', 0)
            if snapshot_every > 0 and epoch % snapshot_every == 0:
                self._save_snapshot(self.stage2_history, epoch)

            # Print progress
            if epoch % self.config.print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | Acc: {acc:.4f}")

        print(f"\nStage 2 Final | Loss: {self.stage2_history.losses[-1]:.6f} | "
              f"Acc: {self.stage2_history.accuracies[-1]:.4f}")

        return self.stage2_history

    def train(self) -> Tuple[TrainingHistory, TrainingHistory]:
        """Run full training (Stage 1 + Stage 2).

        Returns:
            Tuple of (stage1_history, stage2_history)
        """
        self.train_stage1()
        self.train_stage2()
        return self.stage1_history, self.stage2_history

    def save_checkpoint(self, path: str, stage: str = "final") -> None:
        """Save a checkpoint.

        Args:
            path: Path to save checkpoint
            stage: Stage identifier ("init", "stage1", "stage2", "final")
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state": self.model.get_state(),
            "stage": stage,
            "stage1_history": {
                "losses": self.stage1_history.losses,
                "accuracies": self.stage1_history.accuracies,
            },
            "stage2_history": {
                "losses": self.stage2_history.losses,
                "accuracies": self.stage2_history.accuracies,
            },
        }

        # Save tensors
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state(checkpoint["model_state"], self.device)
        print(f"Loaded checkpoint from {path}")

    def analyze_phase_alignment(self, verbose: bool = True) -> Dict[str, Any]:
        """Analyze phase alignment between W_in and W_out.

        For quadratic activation, we expect: 2 * phi_in ≈ psi_out

        Args:
            verbose: Whether to print detailed results

        Returns:
            Dictionary with analysis results
        """
        with torch.no_grad():
            W_in = self.model.W_in.detach().cpu()
            W_out = self.model.W_out.detach().cpu()

            # Normalize for DFT
            W_in_norm = W_in / W_in.norm(dim=1, keepdim=True)
            W_out_norm = W_out / W_out.norm(dim=0, keepdim=True)

            # Apply Fourier transform (handles complex basis)
            W_in_dft = self.basis.transform(W_in_norm)
            W_out_dft = self.basis.transform(W_out_norm.T)

            freq_match_count = 0
            phase_match_count = 0
            results = []

            for m in range(self.config.width):
                dom_freq_in = self.basis.get_dominant_frequency(W_in_dft[m])
                dom_freq_out = self.basis.get_dominant_frequency(W_out_dft[m])

                # Get frequency tuple for d-dimensional case
                freq_in_tuple = self.basis.get_frequency_tuple(dom_freq_in)
                freq_out_tuple = self.basis.get_frequency_tuple(dom_freq_out)

                phi_in = self.basis.get_phase(W_in_dft[m], dom_freq_in)
                psi_out = self.basis.get_phase(W_out_dft[m], dom_freq_out)

                two_phi = normalize_to_pi(2 * phi_in)
                psi = normalize_to_pi(psi_out)
                diff = abs(normalize_to_pi(two_phi - psi))

                freq_match = freq_in_tuple == freq_out_tuple
                phase_match = diff < 0.3

                if freq_match:
                    freq_match_count += 1
                if phase_match:
                    phase_match_count += 1

                results.append({
                    "neuron": m,
                    "freq_in": freq_in_tuple,
                    "freq_out": freq_out_tuple,
                    "phi_in": phi_in,
                    "psi_out": psi_out,
                    "two_phi": two_phi,
                    "diff": diff,
                    "freq_match": freq_match,
                    "phase_match": phase_match,
                })

            if verbose:
                print("\n" + "=" * 85)
                print("Phase Alignment Analysis")
                print("=" * 85)
                print(f"Frequency match: {freq_match_count}/{self.config.width} "
                      f"({100*freq_match_count/self.config.width:.1f}%)")
                print(f"Phase alignment (|2*phi - psi| < 0.3): {phase_match_count}/{self.config.width} "
                      f"({100*phase_match_count/self.config.width:.1f}%)")

            return {
                "freq_match_count": freq_match_count,
                "phase_match_count": phase_match_count,
                "total_neurons": self.config.width,
                "results": results,
            }
