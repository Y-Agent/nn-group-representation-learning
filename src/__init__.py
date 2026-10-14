# Spherical Neural Network Learning
# A modular framework for training neural networks with spherical constraints

from .basis import Basis, get_basis
from .config import Config
from .model import WideNetworkScaleSphere
from .tasks import Task, get_task
from .trainer import SphericalTrainer

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
