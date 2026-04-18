"""Abstract base class for learning tasks."""

from abc import ABC, abstractmethod
from typing import Tuple, List
import torch


class Task(ABC):
    """Abstract base class for learning tasks.

    A task defines:
    1. The input space (what inputs are valid)
    2. The target function (what output should be produced for each input)
    3. The output space (vocabulary size / number of classes)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the task."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of input tokens (e.g., 2 for binary operations)."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the vocabulary / number of output classes."""
        pass

    @abstractmethod
    def generate_all_data(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all possible (input, label) pairs.

        Args:
            device: Device to place tensors on

        Returns:
            Tuple of (data, labels):
                - data: Tensor of shape (N, input_dim) containing input tokens
                - labels: Tensor of shape (N,) containing target labels
        """
        pass

    @abstractmethod
    def compute_label(self, *inputs: int) -> int:
        """Compute the target label for given inputs.

        Args:
            *inputs: Input values

        Returns:
            Target label
        """
        pass

    def generate_train_test_split(
        self,
        device: torch.device,
        frac_train: float = 0.8,
        shuffle: bool = True
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Generate train/test split of data.

        Args:
            device: Device to place tensors on
            frac_train: Fraction of data for training
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of ((train_data, train_labels), (test_data, test_labels))
        """
        data, labels = self.generate_all_data(device)
        n = len(data)

        if shuffle:
            perm = torch.randperm(n, device=device)
            data = data[perm]
            labels = labels[perm]

        n_train = int(n * frac_train)

        train_data = (data[:n_train], labels[:n_train])
        test_data = (data[n_train:], labels[n_train:])

        return train_data, test_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
