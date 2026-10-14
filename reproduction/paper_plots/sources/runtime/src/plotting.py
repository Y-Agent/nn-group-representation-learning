"""Shared plotting helpers for spherical-NN notebooks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def setup_plot_style():
    """Apply default Palatino/CM font settings used across all notebooks."""
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'P052', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'


def make_blue_white_red_cmap():
    """Return the blue→white→red diverging colormap used in DFT heatmaps."""
    return LinearSegmentedColormap.from_list(
        'blue_white_red', ['#0D2758', 'white', '#A32015'], N=256
    )


# ── Frobenius-group display helpers ──────────────────────────────────────────

# LaTeX labels for Frobenius-21 irreps
FROBENIUS21_IRREP_LATEX = {
    'trivial':    r'$\rho_{\sf triv}$',
    'omega':      r'$\rho_1$',
    'rho_3':      r'$\rho_2$',
    'rho_3_conj': r'$\rho_2^\vee$',
    'omega_conj': r'$\rho_1^\vee$',
}


def get_frobenius_display_info(basis):
    """Build axis-label lists and boundary info for the Frobenius-21 basis.

    Returns:
        basis_labels:      list of '(i,j)' strings for each basis function
        irrep_centers:     list of center x-positions per irrep block
        irrep_boundaries:  list of (start, end) index pairs per irrep block
    """
    basis_labels = []
    irrep_boundaries = []
    irrep_centers = []
    curr_idx = 0
    for label in basis.irrep_labels:
        d = basis.irrep_dimensions[label]
        for i in range(d):
            for j in range(d):
                basis_labels.append(f'({i+1},{j+1})')
        irrep_boundaries.append((curr_idx, curr_idx + d * d))
        irrep_centers.append(curr_idx + (d * d - 1) / 2)
        curr_idx += d * d
    return basis_labels, irrep_centers, irrep_boundaries


def scale_by_sqrt_d(dft_coeffs, basis):
    """Scale each irrep block by √d_ρ (Plancherel weighting for heatmaps)."""
    scaled = dft_coeffs.clone()
    curr_idx = 0
    for label in basis.irrep_labels:
        d = basis.irrep_dimensions[label]
        scaled[:, curr_idx:curr_idx + d * d] *= np.sqrt(d)
        curr_idx += d * d
    return scaled


# ── Generic DFT plot functions ────────────────────────────────────────────────

def plot_dft_heatmap(
    data_columns,
    titles_re,
    titles_im,
    x_labels,
    vline_positions=None,
    bottom_annotations=None,
    cmap=None,
    num_components=None,
):
    """Generic 2×N DFT heatmap (Re row / Im row).

    Args:
        data_columns:        list of N complex arrays, each shape (M, K)
        titles_re:           list of N title strings for the Re row
        titles_im:           list of N title strings for the Im row
        x_labels:            list of K tick labels for x-axis
        vline_positions:     list of x positions for vertical dividers (optional)
        bottom_annotations:  list of (x_pos, label_str) pairs drawn below last row (optional)
        cmap:                colormap (defaults to blue_white_red)
        num_components:      rows to show; defaults to full M

    Returns:
        fig
    """
    if cmap is None:
        cmap = make_blue_white_red_cmap()

    n_cols = len(data_columns)
    if num_components is not None:
        data_columns = [d[:num_components] for d in data_columns]

    M = data_columns[0].shape[0]
    K = data_columns[0].shape[1]

    data_re = [np.real(d) for d in data_columns]
    data_im = [np.imag(d) for d in data_columns]

    vmaxes = [
        max(np.abs(data_re[c]).max(), np.abs(data_im[c]).max(), 0.1)
        for c in range(n_cols)
    ]

    x_locs = np.arange(K)
    y_locs = np.arange(M)

    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for row, (row_data, row_titles) in enumerate([(data_re, titles_re), (data_im, titles_im)]):
        for col in range(n_cols):
            ax = axes[row, col]
            im = ax.imshow(row_data[col], cmap=cmap,
                           vmin=-vmaxes[col], vmax=vmaxes[col], aspect='auto')
            ax.set_title(row_titles[col], fontsize=18)
            if col == 0:
                ax.set_ylabel('Neuron $m$', fontsize=18)
            ax.set_xticks(x_locs)
            ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
            ax.set_yticks(y_locs)
            ax.set_yticklabels(y_locs + 1)
            if vline_positions:
                for xv in vline_positions:
                    ax.axvline(x=xv, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
            if row == 1 and bottom_annotations:
                for x_pos, lbl in bottom_annotations:
                    ax.annotate(lbl,
                                xy=(x_pos, -0.15), xycoords=('data', 'axes fraction'),
                                ha='center', va='top', fontsize=13, fontweight='bold')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_dft_bars(
    data_columns_init,
    data_columns_trained,
    colors,
    titles,
    x_labels,
    vline_positions=None,
    bottom_annotations=None,
    row_labels=None,
    num_neurons=5,
    ylim=None,
):
    """Generic Init (gray) vs Trained (color) DFT magnitude bar chart.

    Args:
        data_columns_init:    list of N arrays, each shape (M, K) — init magnitudes
        data_columns_trained: list of N arrays, each shape (M, K) — trained magnitudes
        colors:               list of N color strings for trained bars
        titles:               list of N title strings
        x_labels:             list of K tick labels for x-axis
        vline_positions:      list of x positions for vertical dividers (optional)
        bottom_annotations:   list of (x_pos, label_str) pairs drawn below last row (optional)
        row_labels:           list of num_neurons strings for y-axis labels (default: 'Neuron k')
        num_neurons:          number of rows to plot
        ylim:                 y-axis upper limit (auto if None)

    Returns:
        fig
    """
    n_cols = len(data_columns_init)
    K = len(x_labels)
    freq_indices = np.arange(K)

    if row_labels is None:
        row_labels = [f'Neuron {i+1}' for i in range(num_neurons)]

    fig, axes = plt.subplots(num_neurons, n_cols, figsize=(6 * n_cols, 1.8 * num_neurons))
    if num_neurons == 1:
        axes = axes.reshape(1, n_cols)

    for row in range(num_neurons):
        is_last = (row == num_neurons - 1)
        for col in range(n_cols):
            ax = axes[row, col]
            ax.bar(freq_indices, data_columns_init[col][row], alpha=0.4, color='gray', label='Init')
            ax.bar(freq_indices, data_columns_trained[col][row], alpha=0.8, color=colors[col], label='Trained')
            if vline_positions:
                for xv in vline_positions:
                    ax.axvline(x=xv, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=18)
            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.grid(True, alpha=0.3, axis='y')
            if row == 0:
                ax.set_title(titles[col], fontsize=18, pad=15)
                ax.legend(loc='upper right', fontsize=15)
            if is_last:
                ax.set_xticks(freq_indices)
                ax.set_xticklabels(x_labels, fontsize=10, rotation=90)
                if bottom_annotations:
                    for x_pos, lbl in bottom_annotations:
                        ax.annotate(lbl,
                                    xy=(x_pos, -0.40), xycoords=('data', 'axes fraction'),
                                    ha='center', va='top', fontsize=13, fontweight='bold')
            else:
                ax.set_xticks([])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig
