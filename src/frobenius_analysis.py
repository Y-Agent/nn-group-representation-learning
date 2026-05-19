"""Shared analysis helpers for Frobenius-21 group notebooks."""

import numpy as np


def d_sim(C1, C2):
    """Frobenius-normalised similarity between two matrices.

    Returns |⟨C1, C2⟩_F| / (‖C1‖_F · ‖C2‖_F), or 0 if either matrix is ~zero.
    """
    vec1 = C1.flatten()
    vec2 = C2.flatten()
    inner = np.abs(np.vdot(vec1, vec2))
    norm1 = np.linalg.norm(C1, 'fro')
    norm2 = np.linalg.norm(C2, 'fro')
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return inner / (norm1 * norm2)


def extract_irrep_matrices(dft_coeffs, irrep_labels, irrep_dims):
    """Reshape flat DFT coefficient rows into per-irrep (d_ρ × d_ρ) matrices.

    Args:
        dft_coeffs:   (n_neurons, vocab_size) complex array of DFT coefficients
        irrep_labels: ordered list of irrep labels matching dft_coeffs columns
        irrep_dims:   dict {label: d_ρ}

    Returns:
        dict {label: (n_neurons, d_ρ, d_ρ) complex array}
    """
    n_neurons = dft_coeffs.shape[0]
    irrep_matrices = {}
    curr_idx = 0
    for label in irrep_labels:
        d = irrep_dims[label]
        block = dft_coeffs[:, curr_idx:curr_idx + d * d]
        irrep_matrices[label] = block.reshape(n_neurons, d, d)
        curr_idx += d * d
    return irrep_matrices


def compute_irrep_energy(dft_mag, irrep_labels, irrep_dims):
    """Compute L²(G) energy E_ρ = d_ρ ‖ν̂[ρ]‖²_F for each neuron and irrep.

    Args:
        dft_mag:      (n_neurons, vocab_size) float array of |DFT coefficients|
        irrep_labels: ordered list of irrep labels
        irrep_dims:   dict {label: d_ρ}

    Returns:
        energy: (n_neurons, n_irreps) float array
    """
    n_neurons = dft_mag.shape[0]
    energy = np.zeros((n_neurons, len(irrep_labels)))
    idx = 0
    for i, label in enumerate(irrep_labels):
        d = irrep_dims[label]
        energy[:, i] = d * np.sum(dft_mag[:, idx:idx + d * d] ** 2, axis=1)
        idx += d * d
    return energy


# Conjugate pairs for the Frobenius-21 group
_FROBENIUS21_CONJUGATE_PAIRS = {
    'omega': 'omega_conj', 'omega_conj': 'omega',
    'rho_3': 'rho_3_conj', 'rho_3_conj': 'rho_3',
}


def compute_grouped_energy(energy, irrep_labels):
    """Sum energies over conjugate irrep pairs.

    Args:
        energy:       (n_neurons, n_irreps) array from compute_irrep_energy
        irrep_labels: ordered list of irrep labels (same order as energy columns)

    Returns:
        grouped_energy: (n_neurons, n_groups) array
        grouped_labels: list of group label strings (e.g. 'omega+omega_conj')
    """
    n_neurons = energy.shape[0]
    grouped_labels = []
    seen = set()
    for label in irrep_labels:
        if label in seen:
            continue
        if label in _FROBENIUS21_CONJUGATE_PAIRS:
            conj = _FROBENIUS21_CONJUGATE_PAIRS[label]
            grouped_labels.append(f"{label}+{conj}")
            seen.add(label)
            seen.add(conj)
        else:
            grouped_labels.append(label)
            seen.add(label)

    grouped_energy = np.zeros((n_neurons, len(grouped_labels)))
    for gi, glabel in enumerate(grouped_labels):
        if '+' in glabel:
            for part in glabel.split('+'):
                idx = irrep_labels.index(part)
                grouped_energy[:, gi] += energy[:, idx]
        else:
            idx = irrep_labels.index(glabel)
            grouped_energy[:, gi] = energy[:, idx]

    return grouped_energy, grouped_labels


def compute_sv_ratio(mat):
    """Compute σ₂/σ₁ as a rank-1 diagnostic (0 = rank-1, 1 = full rank).

    Returns 0 for 1-D irreps (trivially rank-1) or near-zero matrices.
    """
    if mat.shape[0] == 1 or mat.shape[1] == 1:
        return 0.0
    sv = np.linalg.svd(mat, compute_uv=False)
    if sv[0] < 1e-10:
        return 0.0
    return sv[1] / sv[0] if len(sv) > 1 else 0.0
