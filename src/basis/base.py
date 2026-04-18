"""Abstract base class for spectral bases."""

from abc import ABC, abstractmethod
from typing import Tuple, List
import torch


class Basis(ABC):
    """Abstract base class for spectral bases.

    A basis defines an orthonormal set of functions for decomposing
    signals defined over a finite domain.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the basis."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of basis functions (equals domain size)."""
        pass

    @property
    @abstractmethod
    def num_frequencies(self) -> int:
        """Number of distinct frequencies (excluding DC component)."""
        pass

    @abstractmethod
    def get_basis_matrix(self, device: torch.device) -> torch.Tensor:
        """Get the basis matrix.

        Args:
            device: Device to place tensor on

        Returns:
            Tensor of shape (size, size) where each row is a basis function
        """
        pass

    @abstractmethod
    def get_basis_names(self) -> List[str]:
        """Get human-readable names for each basis function.

        Returns:
            List of names for each basis function
        """
        pass

    def transform(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Transform a signal to the spectral domain.

        Args:
            x: Input tensor of shape (..., size)
            device: Device for computation

        Returns:
            Spectral coefficients of shape (..., size)
        """
        basis = self.get_basis_matrix(device)
        return x @ basis.T

    def inverse_transform(self, coeffs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Transform spectral coefficients back to the original domain.

        Args:
            coeffs: Spectral coefficients of shape (..., size)
            device: Device for computation

        Returns:
            Reconstructed signal of shape (..., size)
        """
        basis = self.get_basis_matrix(device)
        return coeffs @ basis

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
