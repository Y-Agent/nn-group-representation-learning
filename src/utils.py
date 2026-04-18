"""Utility functions for training and analysis."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss with high precision.

    Uses float32 log_softmax to avoid underflow with confident predictions.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)

    Returns:
        Scalar loss value
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)
    return -logprobs[torch.arange(len(labels), device=logits.device), labels].sum()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)

    Returns:
        Accuracy as a float in [0, 1]
    """
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def normalize_to_pi(x: float) -> float:
    """Normalize angle to [-pi, pi] range.

    Args:
        x: Angle in radians

    Returns:
        Angle normalized to [-pi, pi]
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def project_gradient_to_tangent_space(
    weight: torch.Tensor,
    grad: torch.Tensor,
    dim: int
) -> torch.Tensor:
    """Project gradient onto tangent space of unit sphere.

    For a point w on the unit sphere, the tangent space consists of vectors
    orthogonal to w. The projection is: g_proj = g - (g . w) * w

    Args:
        weight: Weight tensor with unit vectors along specified dimension
        grad: Gradient tensor (same shape as weight)
        dim: Dimension along which vectors are normalized (0 or 1)

    Returns:
        Projected gradient
    """
    projected_grad = grad.clone()

    if dim == 1:  # Rows are unit vectors (like W_in)
        for i in range(weight.shape[0]):
            w = weight[i]
            g = grad[i]
            projected_grad[i] = g - torch.dot(g, w) * w
    elif dim == 0:  # Columns are unit vectors (like W_out)
        for j in range(weight.shape[1]):
            w = weight[:, j]
            g = grad[:, j]
            projected_grad[:, j] = g - torch.dot(g, w) * w
    else:
        raise ValueError(f"dim must be 0 or 1, got {dim}")

    return projected_grad


def compute_gradient_norms(model) -> Tuple[float, float]:
    """Compute gradient norms for direction and scale parameters.

    Args:
        model: WideNetworkScaleSphere model

    Returns:
        Tuple of (direction_grad_norm, scale_grad_norm)
    """
    direction_grads = []
    scale_grads = []

    if model.W_in.grad is not None:
        direction_grads.append(model.W_in.grad.flatten())
    if model.W_out.grad is not None:
        direction_grads.append(model.W_out.grad.flatten())
    if model.scale_in.grad is not None:
        scale_grads.append(model.scale_in.grad.flatten())
    if model.scale_out.grad is not None:
        scale_grads.append(model.scale_out.grad.flatten())

    direction_norm = torch.cat(direction_grads).norm().item() if direction_grads else 0.0
    scale_norm = torch.cat(scale_grads).norm().item() if scale_grads else 0.0

    return direction_norm, scale_norm


def compute_param_norms(model) -> Tuple[float, float, float, float]:
    """Compute parameter norms.

    Args:
        model: WideNetworkScaleSphere model

    Returns:
        Tuple of (W_in_norm, W_out_norm, scale_in, scale_out)
    """
    return (
        model.W_in.data.norm().item(),
        model.W_out.data.norm().item(),
        model.scale_in.item(),
        model.scale_out.item(),
    )
