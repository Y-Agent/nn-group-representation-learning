"""Complex Fourier basis for product of cyclic groups.

For G = Z_{n_1} × ... × Z_{n_d}, the dual group characters are:

    χ_k(x) = ∏_{j=1}^d ψ_{k_j}^{(j)}(x_j)

where ψ_{k_j}^{(j)}(z) = exp(i 2π k_j z / n_j).

DFT convention:
    ν̂[χ_k] = (1/|G|) Σ_x ν(x) χ_k(x^{-1}) = (1/|G|) Σ_x ν(x) χ_{-k}(x)
    ν(x) = Σ_k ν̂[χ_k] χ_k(x)

For abelian groups, all irreps are 1D characters, so no (i,j) indexing needed.
"""

from typing import List, Union, Tuple
import torch
import numpy as np
from functools import reduce
from operator import mul

from .base import Basis


class FourierBasis(Basis):
    """Complex Fourier basis for product of cyclic groups.

    For the group G = ⊕_{j=1}^d Z_{n_j}, the dual group Ĝ consists of characters:

        χ_k(x) = ∏_{j=1}^d exp(i 2π k_j x_j / n_j)

    where k = (k_1, ..., k_d) ∈ G indexes the character.

    The basis matrix B has entries:
        B[k, x] = χ_k(x) / sqrt(|G|)  (normalized)

    This is the unitary DFT matrix for the product group.
    """

    def __init__(self, n: Union[int, List[int]]):
        """Initialize Fourier basis.

        Args:
            n: Moduli for each component. Can be:
               - int: Single modulus (1-d case, equivalent to n=[p])
               - List[int]: List of moduli [n_1, ..., n_d] for d-dimensional case
        """
        # Normalize to list
        if isinstance(n, int):
            self.n = [n]
        else:
            self.n = list(n)

        self.d = len(self.n)
        self._size = reduce(mul, self.n, 1)
        self._strides = self._compute_strides(self.n)
        self._basis_cache = {}

    @staticmethod
    def _compute_strides(n: List[int]) -> List[int]:
        """Compute strides for mixed-radix encoding."""
        strides = []
        stride = 1
        for n_j in reversed(n):
            strides.append(stride)
            stride *= n_j
        return list(reversed(strides))

    def index_to_tuple(self, idx: int) -> Tuple[int, ...]:
        """Convert flat index to d-tuple."""
        result = []
        for s, n_j in zip(self._strides, self.n):
            result.append(idx // s)
            idx = idx % s
        return tuple(result)

    def tuple_to_index(self, tup: Tuple[int, ...]) -> int:
        """Convert d-tuple to flat index."""
        return sum(x * s for x, s in zip(tup, self._strides))

    @property
    def name(self) -> str:
        if self.d == 1:
            return f"fourier_p{self.n[0]}"
        else:
            return f"fourier_n{'x'.join(map(str, self.n))}"

    @property
    def size(self) -> int:
        """Number of basis functions (= |G|)."""
        return self._size

    @property
    def moduli(self) -> List[int]:
        """Get the list of moduli."""
        return self.n

    @property
    def num_frequencies(self) -> int:
        """Number of distinct non-DC frequencies (= |G| - 1)."""
        return self._size - 1

    def get_basis_names(self) -> List[str]:
        """Get names for each basis function (character).

        Returns:
            List of names like "χ_(0,0)", "χ_(1,0)", "χ_(0,1)", etc.
        """
        names = []
        for k_idx in range(self._size):
            k_tuple = self.index_to_tuple(k_idx)
            if self.d == 1:
                names.append(f"χ_{k_tuple[0]}")
            else:
                names.append(f"χ_{k_tuple}")
        return names

    def _compute_character(self, k_tuple: Tuple[int, ...], x_tuple: Tuple[int, ...]) -> complex:
        """Compute χ_k(x) = ∏_j exp(i 2π k_j x_j / n_j)."""
        phase = sum(2 * np.pi * k_j * x_j / n_j
                   for k_j, x_j, n_j in zip(k_tuple, x_tuple, self.n))
        return np.exp(1j * phase)

    def get_basis_matrix(self, device: torch.device = None) -> torch.Tensor:
        """Get the normalized complex Fourier basis matrix.

        Returns:
            Complex tensor of shape (|G|, |G|) where:
            B[k, x] = χ_k(x) / sqrt(|G|)

            This is a unitary matrix: B @ B^H = I
        """
        if device is None:
            device = torch.device('cpu')

        device_key = str(device)
        if device_key in self._basis_cache:
            return self._basis_cache[device_key]

        N = self._size
        basis = torch.zeros(N, N, dtype=torch.complex64, device=device)

        # Compute all characters
        for k_idx in range(N):
            k_tuple = self.index_to_tuple(k_idx)
            for x_idx in range(N):
                x_tuple = self.index_to_tuple(x_idx)
                basis[k_idx, x_idx] = self._compute_character(k_tuple, x_tuple)

        # Normalize to make unitary
        basis = basis / np.sqrt(N)

        self._basis_cache[device_key] = basis
        return basis

    def transform(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform: c = ν @ B^H.

        Returns Euclidean coefficients C that preserve norm: ||c||² = ||ν||².

        Args:
            signal: Real or complex tensor of shape (..., |G|)

        Returns:
            Complex tensor of shape (..., |G|) containing Euclidean coefficients.
            To get paper's ν̂[k] = (1/|G|) Σ_x ν(x) χ_k(x^{-1}), use transform_paper().
        """
        basis = self.get_basis_matrix(signal.device)
        # Handle real input
        if not signal.is_complex():
            signal = signal.to(torch.complex64)
        return signal @ basis.T.conj()

    def transform_paper(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform with paper's L²(G) scaling.

        Returns coefficients ν̂[k] matching paper's DFT definition:
            ν̂[k] = (1/|G|) Σ_x ν(x) χ_k(x^{-1})

        For abelian groups, all irreps are 1D (d_ρ = 1), so:
            ν̂[k] = C[k] / √|G|

        These satisfy Plancherel: Σ_k |ν̂[k]|² = ||ν||²_{L²(G)} = (1/|G|)||ν||²

        Args:
            signal: Real or complex tensor of shape (..., |G|)

        Returns:
            Complex tensor of shape (..., |G|) containing paper-notation coefficients.
        """
        # Get Euclidean coefficients
        C = self.transform(signal)

        # Convert to paper notation: ν̂[k] = C[k] / √|G|
        # For abelian groups, d_ρ = 1 for all irreps
        return C / np.sqrt(self._size)

    def inverse_transform(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply inverse Fourier transform: s = ŝ @ B.

        For unitary basis B with forward transform ŝ = s @ B^H,
        the inverse is s = ŝ @ B.

        Args:
            coeffs: Complex tensor of shape (..., |G|)

        Returns:
            Complex tensor of shape (..., |G|)
        """
        basis = self.get_basis_matrix(coeffs.device)
        return coeffs @ basis

    def get_frequency_tuple(self, k_idx: int) -> Tuple[int, ...]:
        """Get the frequency tuple (k_1, ..., k_d) for a given flat index."""
        return self.index_to_tuple(k_idx)

    def get_dominant_frequency(self, coeffs: torch.Tensor) -> int:
        """Find the dominant non-DC frequency in Fourier coefficients.

        Args:
            coeffs: Complex Fourier coefficients of shape (|G|,)

        Returns:
            Flat index of dominant frequency (excluding DC at index 0)
        """
        # Exclude DC component (index 0)
        magnitudes = torch.abs(coeffs[1:])
        return int(torch.argmax(magnitudes).item()) + 1

    def get_magnitude(self, coeffs: torch.Tensor, k_idx: int) -> float:
        """Get magnitude of a frequency component.

        Args:
            coeffs: Complex Fourier coefficients of shape (|G|,)
            k_idx: Flat index of the frequency

        Returns:
            Magnitude |ŝ_k|
        """
        return torch.abs(coeffs[k_idx]).item()

    def get_phase(self, coeffs: torch.Tensor, k_idx: int) -> float:
        """Get phase of a frequency component.

        Args:
            coeffs: Complex Fourier coefficients of shape (|G|,)
            k_idx: Flat index of the frequency

        Returns:
            Phase angle in radians [-π, π]
        """
        return torch.angle(coeffs[k_idx]).item()

    def __repr__(self) -> str:
        if self.d == 1:
            return f"FourierBasis(n={self.n[0]})"
        else:
            return f"FourierBasis(n={self.n}, size={self._size})"
