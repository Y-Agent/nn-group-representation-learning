"""Task definitions for neural network learning."""

from .alternating import AlternatingGroupMultiplication
from .base import Task
from .frobenius21 import Frobenius21Multiplication
from .modular import ModularAddition
from .symmetric import SymmetricGroupMultiplication

# Task registry
TASK_REGISTRY = {
    "modular_addition": ModularAddition,
    "symmetric_group": SymmetricGroupMultiplication,
    "alternating_group": AlternatingGroupMultiplication,
    "frobenius21": Frobenius21Multiplication,
}


def get_task(name: str, **kwargs) -> Task:
    """Get a task by name.

    Args:
        name: Task name (e.g., "modular_addition")
        **kwargs: Task-specific arguments:
            - n: int or List[int] for moduli
            - p: int (backward compatible, converted to n)

    Returns:
        Task instance
    """
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")

    # Handle backward compatibility: convert p to n
    if "n" not in kwargs and "p" in kwargs:
        kwargs["n"] = kwargs.pop("p")
    elif "n" in kwargs and "p" in kwargs:
        kwargs.pop("p")  # n takes precedence

    if name == "frobenius21":
        kwargs = {}
    return TASK_REGISTRY[name](**kwargs)


__all__ = [
    "Task",
    "get_task",
    "ModularAddition",
    "SymmetricGroupMultiplication",
    "AlternatingGroupMultiplication",
    "Frobenius21Multiplication",
]
