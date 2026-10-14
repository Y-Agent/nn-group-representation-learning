# Reproducing the accepted paper figures

This is the authoritative workflow for the figures approved on **14 October 2026**.
It replaces the successive `refresh_*` and `render_local_*` drafts. Those drafts
are preserved as non-executable text in `scripts/legacy/`.

There are two different operations:

1. **Render archived results:** regenerate the accepted affected figures, without training.
2. **Rerun experiments:** generate new numerical results using the corrected implementation.
   This is substantially slower and is not a promise of byte-identical floating-point
   trajectories on a different CPU, BLAS library, PyTorch release, or thread count.

The original plotting cells and helpers from upstream commit
`c6fb3095a92e4c29eda0741933d547cb1b971eb5` remain frozen and hash checked in
`reproduction/paper_plots/sources/`. The numerical library in `src/` is maintained
separately. Importing a module does not start training or plotting.

## Quick start: archived-results replay

The portable archive includes the required checkpoints, initial/final arrays,
configuration and environment records, frozen notebook sources, and accepted
figure baselines. Extract it into an empty directory and run commands from that
repository root. A Git-only checkout must first acquire those archived inputs;
`.pt` checkpoints are intentionally ignored by Git. `render` reports missing files
and checks SHA-256 hashes before opening the checkpoint.

Two rendering environments are deliberate. Most notebook panels were accepted
using Matplotlib **3.10.9**; the later Figure 5 typography and Figure 7 composition
were accepted using **3.11.2**. Using one version for both changes text/layout pixels.
The CLI runs the old notebook adapter in an isolated subprocess and performs the
final layout in the newer environment.

```bash
python3.12 -m venv .venv-notebook
.venv-notebook/bin/python -m pip install -r requirements-notebook.txt

python3.14 -m venv .venv-paper
.venv-paper/bin/python -m pip install -r requirements-paper.txt

.venv-paper/bin/python -m paper_reproduction render \
  --notebook-python .venv-notebook/bin/python \
  --output reproduction/paper

.venv-paper/bin/python -m paper_reproduction verify \
  --output reproduction/paper
```

Validated on macOS ARM64 with Python 3.12.14 for the notebook adapter and 3.14.6
for final layout. Both environments use PyTorch 2.14.1. Exact resolved package
versions are recorded in the two requirements files. The rendering process uses
two PyTorch threads, matching the previous plotting workflow.

**Fonts are part of the setup.** Historical notebook panels and Figure 7 use
Matplotlib's bundled DejaVu Serif and Computer Modern math. The approved enlarged
Figure 5 uses macOS Palatino, with Computer Modern math. The manifest records the
resolved font paths and hashes. Palatino is not redistributed in this package.
On another operating system, install a properly licensed equivalent Palatino font
and make it visible to Matplotlib before rendering; a fallback font will be
reported by pixel verification as a mismatch. A scientific-data match is distinct
from a pixel match. No system font is silently copied into the archive.

`render` refuses nonempty output directories or any output path overlapping
checkpoints, frozen sources, or accepted baselines. To revise a plot, choose a new
output directory. It never writes to Dropbox, Overleaf, `figures/`, or archived runs.

## Outputs and verification

The output directory contains:

- `figures/`: the manuscript filenames, including the `.jpeg` alias used by Figure 1a.
- `panels/`: the four separate, aligned Figure 7 panels in PNG/PDF.
- `figure7.pdf`: the combined 2-by-2 view; `affected-paper-figures.pdf`: ordered review booklet.
- `gallery.html`: local preview, with the current manuscript numbering.
- `figure7-plotted-data.npz`: the exact statistics, scales, and observation grids plotted.
- `manifest.json`: input/source/output hashes, environment, fonts, and layout checks.
- `notebook-output/`: original notebook exports, independently regenerated diagnostics,
  distribution categories, and notebook provenance.
- `render.log`: notebook output; `baseline-verification.json`: verification results.

Verification checks all exported figure hashes, nine JPEG rasters, four Figure 7
panel rasters, and the seven diagnostic arrays. Pixel equality is exact. Numerical
array equality permits only absolute error **1e-12**, with zero relative tolerance,
and reports the actual maximum difference and whether equality was exact.
Cross-platform NumPy reductions in this review differed by at most 2.22e-16.
The accepted baseline itself is checked against the separate immutable contract;
verification does not bless newly rendered output by replacing the baseline.

## Figure map

Numbers refer to the current manuscript, after merging the former Figures 7 and 8.

| Figure | Asset | Experiment / code |
|---|---|---|
| 1a | `modular_addition_xi_heatmap_example.jpeg` | `z15_h2`; original running-example cell 11 |
| 1b | `modular_addition_phase_alignment.jpg` | `z15_h2`; cell 13 |
| 2a | `modular_addition_phase_circles.jpg` | `z15_h2`; cell 14 |
| 2b | `modular_addition_freq_distribution_histogram.jpg` | `z15_h2`; cell 15 |
| 5a | `modular_addition_freq_distribution.jpg` | `z30_h2`; original Abelian cell 15, enlarged typography |
| 5b | `symmetric_group_irrep_distribution.jpg` | `s4_h2`; same drawing cell, S4 data adapter |
| 6 | `frobenius21_dft_heatmap.jpg` | `f21_wide_h2`; original cell 15 |
| 7a | `frobenius21_diagnostics_compact.pdf` | `f21_wide_h2`; original metrics, direct compact renderer |
| 7b | `frobenius21_scale_combined.pdf` | refined Stage II from `f21_wide_h2` |
| 9 | `modular_addition_dft_heatmap_example.jpg` | `z15_h2`; original running-example cell 10 |
| 10 | `modular_addition_dft_heatmap.jpg` | `z30_h2`; first 20 neurons, original ordering |

Figures 3, 4, and 8 are unaffected conceptual or shared-input figures and are not
regenerated by the affected-only command. Their accepted assets are retained in
`reproduction/accepted/figures/`. Some historical conceptual illustrations have
no generating source; the package does not claim to reconstruct them from code.

## Numerical protocol and actual settings

All paper runs use seed 42 and the complete composition table. Each independent
input position and the output lie on separate Euclidean unit spheres. Gaussian
initialization follows the existing PyTorch float32 draw, followed by conversion
to float64 and sphere retraction; this detail is retained for reproducibility.

The quadratic predictor is `(1/M) sum_m a_m xi_m (theta1_m + theta2_m)^2`.
`scale_in=scale_out=0.5` implies effective **a=0.125**. Stage I uses summed CE,
separate tangent projections, Euler steps, and sphere retraction. It has no Adam,
momentum, weight decay, gradient clipping, or hidden width rescaling. The exact
neuron-chunk implementation is algebraic blocking, not stochastic minibatching.
For shared input the theta update has the required factor 1/2.

The actual refined runs used for the figures are:

| Run | Group | Width M | dt | Updates | Final physical time | CPU threads |
|---|---|---:|---:|---:|---:|---:|
| `z15_h2` | C3 × C5 | 1,024 | 10.24 | 8,000 | 81,920 | 2 |
| `z30_h2` | C2 × C3 × C5 | 50,000 | 500 | 8,000 | 4,000,000 | 6 |
| `s4_h2` | S4 | 50,000 | 500 | 4,800 | 2,400,000 | 6 |
| `f21_wide_h2` | C7 ⋊ C3 | 2,048 | 1 | 40,000 | 40,000 | 2 |
| `f21_scale_wide_h2` | C7 ⋊ C3, frozen directions | 2,048 | 0.5 | 80,000 | Stage-II time 40,000 | 2 |

`reproduction/configs.json` stores base settings. The runner divides dt and
multiplies updates by `--step-divisor`, preserving the physical horizon. Do not
mistake the stored base Stage-II dt/steps for the refined effective values above.
Coarse/reference and legacy-comparison runs remain in the original archive but
are not substituted for the main refined figures.

The independent S4 settings are an explicit corrected reproduction, since an
original independent-input S4 run configuration was unavailable. Its rendering
reuses the Abelian code, as confirmed by the author. Its horizon was shortened
from the initial computational budget after convergence, retaining all prior
observations. The Frobenius width was explicitly increased from 512 to 2,048 to
obtain perfect finite-network accuracy. These changes are not hidden claims of
original-setting replication.

Stage II starts from the refined wide checkpoint only after a positive correct-
class margin check. It optimizes direct effective coefficients, with frozen
copied directions and no positivity clipping. The tied variant has multiplier 1;
the displayed untied comparison has multiplier **M**. The Section-2 independent
flow with multiplier 1 is retained as a separate variant and is not relabeled as
the faster matched-clock comparison.

## What exactly is plotted

- Time means completed updates × dt, not optimizer iteration number. Figure 7
  losses are mean CE, i.e. summed risk divided by 441.
- Figure 7 diagnostics retain the **unsigned** normalized complex inner product
  from the original notebook. They support proportionality, not its positive-real
  sign. Signed-cosine diagnostics are separately available in raw experiment output.
  Earlier draft documentation incorrectly called the displayed metric signed.
- Rank-ratio summaries use higher-dimensional irreps; error bars are cross-neuron
  standard deviations, not standard errors or confidence intervals. The notebook's
  final-dominant-irrep selection and layer order are preserved.
- Stage-I observations use a 400-time-unit grid. The loss boundary observation is
  drawn as the initial Stage-II point at the same total time, with no interpolation.
- The Stage-I break omits **8,000 < t < 38,000**. Stage II displays total time
  **40,000–50,000**; full histories continue through total time 80,000.
- The ten per-neuron curves use indices 0–9 sorted by their **full saved endpoint**,
  exactly as the original cell; selection is not recomputed on the cropped window.
- Figure 5 counts every neuron. Its energy averages retain the original first-100
  assigned-neuron convention for nontrivial self-conjugate irreps; they are not
  falsely described as population averages over all 50,000 neurons.
- Figure 5a keeps blue labels above and red below; Figure 5b reverses those positions
  and uses horizontal category text. Numeric overlap checks run at render time.
- Figure 7 uses 3.2-point traces, 30-point axis/index text, 24-point ticks and major
  legends, aligned equal-sized plot regions, and inside legends. Figure 5 font
  sizes and offsets live in `paper_reproduction/distribution.py`.

## Rerun training on CPU

The archived training environment is Python 3.10.13 on Linux, PyTorch 2.14.1+cpu,
NumPy 2.2.6. `reproduction/requirements-lock-cpu.txt` is the original environment
capture. For that environment install the CPU wheel from the official PyTorch
CPU index and then the recorded requirements:

```bash
python3.10 -m venv .venv-train
.venv-train/bin/python -m pip install torch==2.14.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.venv-train/bin/python -m pip install -r reproduction/requirements-lock-cpu.txt

# Print the complete command sequence without starting expensive jobs.
.venv-train/bin/python -m paper_reproduction train --runs reproduction/new-runs

# Execute sequentially on a CPU host with sufficient memory.
.venv-train/bin/python -m paper_reproduction train \
  --runs reproduction/new-runs --threads 2 --execute
```

The default plan runs coarse/refined pairs for z15, z30, S4, and wide Frobenius,
then both Stage-II refinements from the same refined Stage-I endpoint. `--main-only`
runs only the refinements used in the paper. The archived job requested 14 CPUs
and 48 GB RAM and ran independent experiments concurrently with 2/6/6 threads;
the portable plan is sequential and records its actual thread count. For closer
historical scheduling use 6 threads for z30/S4 and 2 for the other runs. Thread
changes can change floating-point reduction order. No pod is provisioned by these
commands. For exact historical results, replay the archived arrays.

Training checkpoints can resume when the same command is rerun. Configurations
are checked before reuse. New Stage-II outputs bind their step settings and
frozen model fingerprint in a run contract. Historical outputs without that
contract remain valid plotting inputs; use a fresh directory to train with the
new code rather than adopting them silently.

To plot new results, use `render --runs reproduction/new-runs --new-results` and
a fresh output directory. The manifest records the new input hashes. It does not
claim those results equal the accepted baseline, and `verify` may rightly fail.

## Development and maintenance

```bash
.venv-paper/bin/python -m pip install -r requirements-dev.txt
.venv-paper/bin/python -m pytest -q
ruff check src scripts tests paper_reproduction
ruff format --check src scripts tests paper_reproduction
```

The tests cover sphere geometry, exact gradients, loss normalization, scale
continuity, all implemented representation multiplication laws, Fourier precision,
checkpoint state, resumed trajectories, zero-valued CLI arguments, and output
safeguards. Frozen historical helpers are intentionally excluded from formatting.

For a transferable archive with the required reference inputs:

```bash
.venv-paper/bin/python -m paper_reproduction package \
  --destination ../paper-reproduction.tar.gz
```

The package command verifies the archived inputs and baseline, excludes Python
caches, and refuses to overwrite an existing archive. It includes no credentials,
cloud configuration, system fonts, or manuscript source. It does not publish or
update Overleaf. Copy only reviewed files from the generated `figures/` directory
when updating the manuscript.
