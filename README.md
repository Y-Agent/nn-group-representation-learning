# Spherical Neural Network Learning

A modular framework for training wide neural networks with spherical parameterization on discrete function learning tasks.

## Overview

This project implements a two-stage training procedure for wide neural networks where weight directions are constrained to lie on the unit sphere:

**Stage 1: Projected Gradient Descent (Spherical Constraint)**
- Train only directions (W_in, W_out) while keeping scales fixed
- Gradients are projected onto the tangent space of the unit sphere
- Directions are re-normalized after each step

**Stage 2: Standard Gradient Descent**
- Train directions, scales, or both
- No spherical constraint

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with default config
python scripts/train.py

# Run with custom parameters
python scripts/train.py --p 47 --width 256 --stage1-epochs 10000 --analyze

# Run only Stage 1
python scripts/train.py --no-stage2 --analyze

# Use different task
python scripts/train.py --task modular_subtraction --p 23
```

## Project Structure

```
spherical-nn-learning/
├── configs/
│   └── default.yaml          # Default configuration
├── scripts/
│   └── train.py              # Main training script
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── model.py              # WideNetworkScaleSphere model
│   ├── trainer.py            # SphericalTrainer class
│   ├── utils.py              # Utility functions
│   ├── tasks/                # Task definitions
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract Task class
│   │   └── modular.py        # Modular arithmetic tasks
│   └── basis/                # Spectral basis definitions
│       ├── __init__.py
│       ├── base.py           # Abstract Basis class
│       └── fourier.py        # Fourier basis
└── requirements.txt
```

## Configuration

Configuration can be specified via YAML file and/or command line overrides.

### YAML Config (configs/default.yaml)

```yaml
task:
  name: "modular_addition"
  p: 23

basis:
  name: "fourier"

model:
  width: 128
  act_type: "Quad"
  init_scale: 0.5

stage1:
  enabled: true
  lr: 0.001
  num_epochs: 6000
  optimizer: "Adam"
  freeze_scales: true

stage2:
  enabled: true
  lr: 5.0
  num_epochs: 50000
  optimizer: "SGD"
  train_directions: true
  train_scales: false
```

### Command Line Arguments

```
--task          Task name (modular_addition, modular_subtraction, etc.)
--p             Prime modulus
--width, -M     Number of neurons
--act           Activation (ReLU, Quad, Abs, GeLU)
--init-scale    Initial scale value

--stage1-lr     Stage 1 learning rate
--stage1-epochs Stage 1 epochs
--no-stage1     Disable Stage 1

--stage2-lr     Stage 2 learning rate
--stage2-epochs Stage 2 epochs
--no-stage2     Disable Stage 2

--analyze       Run phase alignment analysis
--save-path     Path to save checkpoint
```

## Adding New Tasks

Create a new task by implementing the `Task` interface:

```python
from src.tasks.base import Task

class MyTask(Task):
    def __init__(self, p: int):
        self.p = p

    @property
    def name(self) -> str:
        return "my_task"

    @property
    def input_dim(self) -> int:
        return 2

    @property
    def vocab_size(self) -> int:
        return self.p

    def compute_label(self, a: int, b: int) -> int:
        return (a * b) % self.p  # Example: modular multiplication

    def generate_all_data(self, device):
        # Generate all (input, label) pairs
        ...
```

Then register it in `src/tasks/__init__.py`:

```python
TASK_REGISTRY["my_task"] = MyTask
```

## Adding New Bases

Create a new basis by implementing the `Basis` interface:

```python
from src.basis.base import Basis

class WalshBasis(Basis):
    def __init__(self, n: int):
        self.n = n

    @property
    def name(self) -> str:
        return "walsh"

    @property
    def size(self) -> int:
        return 2 ** self.n

    def get_basis_matrix(self, device):
        # Return orthonormal Walsh-Hadamard basis
        ...
```

Then register it in `src/basis/__init__.py`:

```python
BASIS_REGISTRY["walsh"] = WalshBasis
```

## Python API

```python
from src.config import Config
from src.trainer import SphericalTrainer

# Create config
config = Config(
    task_name="modular_addition",
    p=23,
    width=128,
    act_type="Quad",
    stage1_num_epochs=6000,
    stage2_num_epochs=50000
)

# Create trainer
trainer = SphericalTrainer(config)

# Train
stage1_history, stage2_history = trainer.train()

# Analyze
analysis = trainer.analyze_phase_alignment()
print(f"Phase alignment: {analysis['phase_match_count']}/{analysis['total_neurons']}")

# Access model
model = trainer.model
```
