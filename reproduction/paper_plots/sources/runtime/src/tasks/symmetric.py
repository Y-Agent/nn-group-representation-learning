"""Symmetric group tasks.

Group multiplication on the symmetric group S_n (permutations of n elements).

The group S_n has n! elements. The operation is composition of permutations.
"""

from typing import Tuple, List
import torch
from itertools import permutations
from math import factorial

from .base import Task


class SymmetricGroupMultiplication(Task):
    """Permutation composition on the symmetric group S_n.

    Task: Given σ, τ ∈ S_n, compute σ ∘ τ (σ applied after τ)

    Convention: (σ ∘ τ)(i) = σ(τ(i))

    Permutations are represented as tuples where perm[i] is the image of i.
    For example, (1, 2, 0) means: 0→1, 1→2, 2→0

    Examples:
        SymmetricGroupMultiplication(3)  # S_3, vocab_size=6
        SymmetricGroupMultiplication(4)  # S_4, vocab_size=24
    """

    def __init__(self, n: int):
        """Initialize symmetric group multiplication task.

        Args:
            n: Degree of symmetric group S_n (permutations of n elements)
        """
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if n > 4:
            raise ValueError(f"n > 4 not yet supported (S_5 has 120 elements, basis implementation is complex)")

        self.n = n
        self._vocab_size = factorial(n)

        # Generate all permutations in lexicographic order
        self._perms: List[Tuple[int, ...]] = list(permutations(range(n)))

        # Create lookup table: perm tuple -> index
        self._perm_to_idx = {p: i for i, p in enumerate(self._perms)}

    @property
    def name(self) -> str:
        return f"symmetric_group_S{self.n}"

    @property
    def input_dim(self) -> int:
        return 2  # Two permutations

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def degree(self) -> int:
        """Degree of the symmetric group."""
        return self.n

    @property
    def elements(self) -> List[Tuple[int, ...]]:
        """List of all group elements (permutations)."""
        return self._perms

    def perm_to_index(self, perm: Tuple[int, ...]) -> int:
        """Convert permutation tuple to index."""
        return self._perm_to_idx[perm]

    def index_to_perm(self, idx: int) -> Tuple[int, ...]:
        """Convert index to permutation tuple."""
        return self._perms[idx]

    def compose(self, sigma: Tuple[int, ...], tau: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compose two permutations: (σ ∘ τ)(i) = σ(τ(i))."""
        return tuple(sigma[tau[i]] for i in range(self.n))

    def inverse(self, sigma: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the inverse permutation."""
        inv = [0] * self.n
        for i, j in enumerate(sigma):
            inv[j] = i
        return tuple(inv)

    def sign(self, sigma: Tuple[int, ...]) -> int:
        """Compute the sign (parity) of a permutation.

        Returns +1 for even permutations, -1 for odd permutations.
        """
        # Count inversions
        n_inv = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if sigma[i] > sigma[j]:
                    n_inv += 1
        return 1 if n_inv % 2 == 0 else -1

    def compute_label(self, sigma_idx: int, tau_idx: int) -> int:
        """Compute σ ∘ τ given indices."""
        sigma = self.index_to_perm(sigma_idx)
        tau = self.index_to_perm(tau_idx)
        result = self.compose(sigma, tau)
        return self.perm_to_index(result)

    def generate_all_data(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all (σ, τ) pairs and their composition labels."""
        N = self._vocab_size
        data = torch.tensor(
            [(i, j) for i in range(N) for j in range(N)],
            dtype=torch.long,
            device=device
        )
        labels = torch.tensor(
            [self.compute_label(i, j) for i in range(N) for j in range(N)],
            dtype=torch.long,
            device=device
        )
        return data, labels

    def __repr__(self) -> str:
        return f"SymmetricGroupMultiplication(n={self.n}, vocab_size={self._vocab_size})"
