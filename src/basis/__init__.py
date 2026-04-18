"""Basis definitions for spectral analysis."""

from typing import Union, List
from .base import Basis
from .fourier import FourierBasis
from .symmetric import SymmetricGroupBasis
from .alternating import AlternatingGroupBasis
from .frobenius21 import Frobenius21Basis

# Basis registry
BASIS_REGISTRY = {
    "fourier": FourierBasis,
    "symmetric": SymmetricGroupBasis,
    "alternating": AlternatingGroupBasis,
    "frobenius21": Frobenius21Basis,
}


def get_basis(name: str, **kwargs) -> Basis:
    """Get a basis by name.

    Args:
        name: Basis name (e.g., "fourier")
        **kwargs: Basis-specific arguments:
            - n: int or List[int] for moduli
            - p: int (backward compatible, converted to n)

    Returns:
        Basis instance
    """
    if name not in BASIS_REGISTRY:
        raise ValueError(f"Unknown basis: {name}. Available: {list(BASIS_REGISTRY.keys())}")

    # Handle backward compatibility: convert p to n
    if "n" not in kwargs and "p" in kwargs:
        kwargs["n"] = kwargs.pop("p")
    elif "n" in kwargs and "p" in kwargs:
        kwargs.pop("p")  # n takes precedence

    return BASIS_REGISTRY[name](**kwargs)


__all__ = ["Basis", "get_basis", "FourierBasis", "SymmetricGroupBasis", "AlternatingGroupBasis", "Frobenius21Basis"]
