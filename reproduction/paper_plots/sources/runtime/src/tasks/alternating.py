"""Alternating group tasks.

Group multiplication on the alternating group A_n (even permutations of n elements).

The group A_n has n!/2 elements. The operation is composition of permutations.
A_n is a normal subgroup of S_n consisting of all permutations with sign +1.
"""

from typing import Tuple, List
import torch
from itertools import permutations
from math import factorial

from .base import Task


def permutation_sign(perm: Tuple[int, ...]) -> int:
    """Compute the sign of a permutation.

    Returns +1 for even permutations, -1 for odd permutations.
    """
    n = len(perm)
    n_inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                n_inv += 1
    return 1 if n_inv % 2 == 0 else -1


class AlternatingGroupMultiplication(Task):
    """Permutation composition on the alternating group A_n.

    Task: Given σ, τ ∈ A_n, compute σ ∘ τ (σ applied after τ)

    Convention: (σ ∘ τ)(i) = σ(τ(i))

    Permutations are represented as tuples where perm[i] is the image of i.
    Only even permutations (sign = +1) are included.

    Examples:
        AlternatingGroupMultiplication(3)  # A_3, vocab_size=3
        AlternatingGroupMultiplication(4)  # A_4, vocab_size=12
        AlternatingGroupMultiplication(5)  # A_5, vocab_size=60
    """

    def __init__(self, n: int):
        """Initialize alternating group multiplication task.

        Args:
            n: Degree of alternating group A_n (even permutations of n elements)
        """
        if n < 3:
            raise ValueError(f"n must be at least 3 for non-trivial A_n, got {n}")
        if n > 5:
            raise ValueError(f"n > 5 not yet supported (A_6 has 360 elements)")

        self.n = n
        self._vocab_size = factorial(n) // 2

        # Generate all even permutations in lexicographic order
        self._perms: List[Tuple[int, ...]] = [
            p for p in permutations(range(n)) if permutation_sign(p) == 1
        ]

        # Verify count
        assert len(self._perms) == self._vocab_size, \
            f"Expected {self._vocab_size} even permutations, got {len(self._perms)}"

        # Create lookup table: perm tuple -> index
        self._perm_to_idx = {p: i for i, p in enumerate(self._perms)}

    @property
    def name(self) -> str:
        return f"alternating_group_A{self.n}"

    @property
    def input_dim(self) -> int:
        return 2  # Two permutations

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def degree(self) -> int:
        """Degree of the alternating group."""
        return self.n

    @property
    def elements(self) -> List[Tuple[int, ...]]:
        """List of all group elements (even permutations)."""
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
        return f"AlternatingGroupMultiplication(n={self.n}, vocab_size={self._vocab_size})"
