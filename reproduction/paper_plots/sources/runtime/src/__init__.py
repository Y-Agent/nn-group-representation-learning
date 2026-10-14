# Spherical Neural Network Learning
# A modular framework for training neural networks with spherical constraints

from .config import Config
from .model import WideNetworkScaleSphere
from .trainer import SphericalTrainer
from .tasks import get_task, Task
from .basis import get_basis, Basis

__version__ = "0.1.0"
__all__ = [
    "Config",
    "WideNetworkScaleSphere",
    "SphericalTrainer",
    "get_task",
    "Task",
    "get_basis",
    "Basis",
]
