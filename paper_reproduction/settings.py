"""Fixed presentation choices approved for the manuscript (October 2026)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "reproduction/paper_plots/sources"
RUNS = ROOT / "reproduction/runs"
OUTPUT = ROOT / "reproduction/paper"
DIAGNOSTIC_COLORS = ["#6C7785", "#477DA5", "#914F63"]
NEURON_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#B58B2A",
    "#C47A35",
    "#76B7B2",
    "#E15759",
    "#B07AA1",
    "#9C755F",
    "#6C7785",
    "#914F63",
]
LAYER_LABELS = [
    r"$\widehat{\theta^1_m}[\rho_m]$",
    r"$\widehat{\theta^2_m}[\rho_m]$",
    r"$\widehat{\xi_m}[\rho_m]$",
]
WIDTH, TOP_HEIGHT, BOTTOM_HEIGHT = 11.6, 6.5, 7.0
LEFT, PLOT_WIDTH, PLOT_HEIGHT = 1.6, 8.4, 4.6
LINE_WIDTH, AXIS_FONT, TICK_FONT = 3.2, 30, 24
STAGE_BOUNDARY, VISIBLE_END, SAMPLE_INTERVAL = 40000, 50000, 400
STAGE1_BREAK = (8000, 38000)
# Figure numbers after merging the former Figures 7 and 8.
ASSETS = {
    "1a": "modular_addition_xi_heatmap_example.jpg",
    "1b": "modular_addition_phase_alignment.jpg",
    "2a": "modular_addition_phase_circles.jpg",
    "2b": "modular_addition_freq_distribution_histogram.jpg",
    "5a": "modular_addition_freq_distribution.jpg",
    "5b": "symmetric_group_irrep_distribution.jpg",
    "6": "frobenius21_dft_heatmap.jpg",
    "7a": "frobenius21_diagnostics_compact.pdf",
    "7b": "frobenius21_scale_combined.pdf",
    "9": "modular_addition_dft_heatmap_example.jpg",
    "10": "modular_addition_dft_heatmap.jpg",
}
