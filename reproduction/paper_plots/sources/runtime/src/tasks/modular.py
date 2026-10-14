"""Modular arithmetic tasks.

Component-wise modular addition over product of cyclic groups:
    G = Z_{n_1} × Z_{n_2} × ... × Z_{n_d}

The operation is: (x ⊕ y)_j = (x_j + y_j) mod n_j for all j.
"""

from typing import Tuple, Union, List
import torch
from functools import reduce
from operator import mul

from .base import Task


class ModularAddition(Task):
    """Component-wise modular addition over product of cyclic groups.

    Task: (x ⊕_d y)_j = (x_j + y_j) mod n_j for all j ∈ {1, ..., d}

    The group is G = ⊕_{j=1}^d Z_{n_j} with |G| = ∏_j n_j elements.

    For the 1-d case (standard modular addition), just pass a single integer.

    Examples:
        ModularAddition(17)        # Z_17, vocab_size=17
        ModularAddition([5, 7])    # Z_5 × Z_7, vocab_size=35
        ModularAddition([3, 3, 3]) # Z_3 × Z_3 × Z_3, vocab_size=27
    """

    def __init__(self, n: Union[int, List[int]]):
        """Initialize component-wise modular addition task.

        Args:
            n: Moduli for each component. Can be:
               - int: Single modulus (1-d case)
               - List[int]: List of moduli [n_1, n_2, ..., n_d] for d-dimensional case
        """
        # Normalize to list
        if isinstance(n, int):
            self.n = [n]
        else:
            self.n = list(n)

        self.d = len(self.n)  # Number of components
        self._vocab_size = reduce(mul, self.n, 1)  # Product of all n_j

        # Precompute strides for mixed-radix conversion
        # stride[j] = prod_{k=j+1}^{d} n_k
        self._strides = self._compute_strides(self.n)

    @staticmethod
    def _compute_strides(n: List[int]) -> List[int]:
        """Compute strides for mixed-radix encoding."""
        strides = []
        stride = 1
        for n_j in reversed(n):
            strides.append(stride)
            stride *= n_j
        return list(reversed(strides))

    @property
    def name(self) -> str:
        if self.d == 1:
            return f"modular_addition_p{self.n[0]}"
        else:
            return f"modular_addition_n{'x'.join(map(str, self.n))}"

    @property
    def input_dim(self) -> int:
        return 2  # Two group elements

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tuple_to_index(self, tup: Tuple[int, ...]) -> int:
        """Convert d-tuple to flat index using mixed-radix encoding.

        idx = sum_j x_j * stride_j where stride_j = prod_{k>j} n_k
        """
        return sum(x * s for x, s in zip(tup, self._strides))

    def index_to_tuple(self, idx: int) -> Tuple[int, ...]:
        """Convert flat index back to d-tuple."""
        result = []
        for s, n_j in zip(self._strides, self.n):
            result.append(idx // s)
            idx = idx % s
        return tuple(result)

    def compute_label(self, x_idx: int, y_idx: int) -> int:
        """Compute (x ⊕ y) given flat indices."""
        x = self.index_to_tuple(x_idx)
        y = self.index_to_tuple(y_idx)
        # Component-wise addition mod n_j
        z = tuple((x_j + y_j) % n_j for x_j, y_j, n_j in zip(x, y, self.n))
        return self.tuple_to_index(z)

    def generate_all_data(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all (x, y) pairs and their labels."""
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
        if self.d == 1:
            return f"ModularAddition(n={self.n[0]})"
        else:
            return f"ModularAddition(n={self.n}, vocab_size={self._vocab_size})"
