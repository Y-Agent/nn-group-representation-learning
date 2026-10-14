"""Frobenius group of order 21 task.

The Frobenius group G = C_7 ⋊ C_3 is the unique non-abelian group of order 21.

Group structure:
- Elements are pairs (a, b) where a ∈ {1, 2, 4} ⊂ Z_7* and b ∈ Z_7
- Multiplication: (a, b) · (a', b') = (a·a' mod 7, b + a·b' mod 7)
- Identity: (1, 0)
- Inverse: (a, b)^(-1) = (a^(-1), -a^(-1)·b mod 7)

This is isomorphic to the affine group Aff(F_7) restricted to {1, 2, 4}.

Presentation: G = ⟨x, y | x^7 = y^3 = 1, yxy^(-1) = x^2⟩
where x = (1, 1) and y = (2, 0).
"""

from typing import Tuple, List, Dict
import torch

from .base import Task


class Frobenius21Multiplication(Task):
    """Group multiplication on the Frobenius group of order 21.

    Task: Given g, h ∈ G, compute g · h

    Elements are represented as indices 0-20, with a bijection to pairs (a, b).

    This group is interesting because it has:
    - 5 irreps with dimensions 1, 1, 1, 3, 3
    - The two 3D irreps are a complex conjugate pair (non-self-conjugate)
    """

    def __init__(self):
        """Initialize Frobenius group multiplication task."""
        self._vocab_size = 21

        # Valid values for a (the multiplicative part)
        # These are the elements of the unique subgroup of order 3 in Z_7*
        # {1, 2, 4} since 2^1=2, 2^2=4, 2^3=1 (mod 7)
        self._a_values = [1, 2, 4]

        # Generate all 21 elements as (a, b) pairs
        self._elements: List[Tuple[int, int]] = []
        for a in self._a_values:
            for b in range(7):
                self._elements.append((a, b))

        # Create lookup tables
        self._elem_to_idx: Dict[Tuple[int, int], int] = {
            elem: i for i, elem in enumerate(self._elements)
        }

        # Precompute modular inverses for a values
        # a^(-1) mod 7: 1->1, 2->4, 4->2
        self._a_inv = {1: 1, 2: 4, 4: 2}

    @property
    def name(self) -> str:
        return "frobenius21"

    @property
    def input_dim(self) -> int:
        return 2  # Two group elements

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def elements(self) -> List[Tuple[int, int]]:
        """List of all group elements as (a, b) pairs."""
        return self._elements

    def elem_to_index(self, elem: Tuple[int, int]) -> int:
        """Convert (a, b) pair to index."""
        return self._elem_to_idx[elem]

    def index_to_elem(self, idx: int) -> Tuple[int, int]:
        """Convert index to (a, b) pair."""
        return self._elements[idx]

    def multiply(self, g: Tuple[int, int], h: Tuple[int, int]) -> Tuple[int, int]:
        """Multiply two group elements: (a, b) · (a', b') = (a·a' mod 7, b + a·b' mod 7)."""
        a, b = g
        a_prime, b_prime = h
        return ((a * a_prime) % 7, (b + a * b_prime) % 7)

    def inverse(self, g: Tuple[int, int]) -> Tuple[int, int]:
        """Compute inverse: (a, b)^(-1) = (a^(-1), -a^(-1)·b mod 7)."""
        a, b = g
        a_inv = self._a_inv[a]
        return (a_inv, (-a_inv * b) % 7)

    def identity(self) -> Tuple[int, int]:
        """Return the identity element (1, 0)."""
        return (1, 0)

    def compute_label(self, g_idx: int, h_idx: int) -> int:
        """Compute g · h given indices."""
        g = self.index_to_elem(g_idx)
        h = self.index_to_elem(h_idx)
        result = self.multiply(g, h)
        return self.elem_to_index(result)

    def generate_all_data(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all (g, h) pairs and their product labels."""
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
        return f"Frobenius21Multiplication(vocab_size={self._vocab_size})"
