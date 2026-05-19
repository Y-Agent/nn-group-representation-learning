Public code repository for the paper **"Neural Networks Provably Learn Spectral Representations for Group Composition"** by Jianliang He\*, Leda Wang\*, Fengzhuo Zhang, Siyu Chen, Zhuoran Yang.

## Overview

This repository provides experiments supporting the theoretical analysis of how two-layer neural networks learn group composition $(g_1, g_2) \mapsto g_1 \star g_2$ for finite groups via projected gradient descent. We demonstrate that training drives each neuron to specialize in a single irreducible representation, with rank-one cross-layer Fourier alignment.

## Supported Groups & Tasks

The code supports group composition on any finite group registered in `src/tasks/` and `src/basis/`. Currently implemented:

| Group | Type | Basis |
|-------|------|-------|
| $\mathbb{Z}_{n_1} \oplus \cdots \oplus \mathbb{Z}_{n_d}$ (product of cyclic groups / generalized modular addition) | Abelian | Fourier (1D irreps) |
| Frobenius group $G = C_7 \rtimes C_3$, $\|G\| = 21$ | Non-Abelian | Peter–Weyl (1D + 3D irreps) |
| Symmetric group $S_n$, $n \leq 4$ ($\|S_n\| = n!$) | Non-Abelian | Young tableau irreps |
| Alternating group $A_n$, $n \leq 5$ ($\|A_n\| = n!/2$) | Non-Abelian | Restricted-from-$S_n$ irreps |

Both **shared** embedding (`share_embed=True`, suited for commutative/Abelian groups) and **unshared** embedding (`share_embed=False`, required for non-commutative groups) are supported. Dedicated notebooks are provided for the modular addition and Frobenius-21 examples; $S_n$ and $A_n$ can be used directly via the Python API.

## Training Recipe

Training follows the two-stage procedure from the paper:

**Stage 1 — Direction learning** (projected gradient descent on the unit sphere):
- Input/output weight directions $(\theta^1_m, \theta^2_m, \xi_m)$ are constrained to $\mathbb{S}^{|G|-1}$
- Gradients are projected onto the tangent space; directions are renormalized after each step
- Scaling factors $a_m$ are frozen

**Stage 2 — Scale optimization**:
- Directions are frozen from Stage 1
- Scaling factors $a_m$ are optimized (tied scalar or per-neuron)

Standard gradient descent (no sphere constraint) is also provided for comparison.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `frobenius21_group_unshared.ipynb` | Stage 1 on Frobenius-21: single-representation, rank-1 alignment, phase analysis |
| `frobenius21_group_standard.ipynb` | Standard GD on Frobenius-21 for comparison |
| `frobenius21_scale_comparison.ipynb` | Stage 1 + Stage 2, tied vs. per-neuron scale growth |
| `modular_addition_unshared.ipynb` | Stage 1 on $\mathbb{Z}_{n_1} \oplus \cdots \oplus \mathbb{Z}_{n_d}$ with unshared embedding |
| `modular_addition_unshared_running_example.ipynb` | Lightweight running example |
| `modular_addition_single_freq_phase_convergence.ipynb` | Phase convergence under specialized initialization |

## Project Structure

```
nn-group-representation-learning/
├── src/
│   ├── model.py              # WideNetworkScaleSphere
│   ├── trainer.py            # SphericalTrainer (Stage 1 & 2)
│   ├── config.py             # Config dataclass
│   ├── utils.py              # Training utilities
│   ├── frobenius_analysis.py # Spectral analysis: d_sim, Plancherel energy, rank-1 ratio
│   ├── plotting.py           # Generic DFT plots: plot_dft_bars, plot_dft_heatmap
│   ├── tasks/                # Group composition tasks
│   └── basis/                # Spectral bases (Fourier, Frobenius-21 Peter–Weyl)
├── notebooks/
└── figures/
```

## Python API

```python
from src import Config, SphericalTrainer

config = Config(
    task_name='modular_addition',
    n=[3, 5],           # G = Z_3 x Z_5
    width=512,
    act_type='Quad',
    share_embed=True,   # shared for Abelian; False for non-Abelian
    stage1_lr=1e-3,
    stage1_num_epochs=1000,
    stage2_enabled=False,
)

trainer = SphericalTrainer(config)
trainer.train()
```

## Installation

```bash
pip install -r requirements.txt
```
