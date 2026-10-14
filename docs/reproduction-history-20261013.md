# Historical reproduction notes (superseded)

These notes preserve the earlier experiment and plotting iterations. Figure
numbers, setup commands, and metric descriptions below may be obsolete. Use
[REPRODUCTION.md](../REPRODUCTION.md) for the accepted figures and current protocol,
including the correction that the displayed alignment metric is unsigned.

## Original notes

This isolated checkout starts at upstream commit c6fb3095a92e4c29eda0741933d547cb1b971eb5. No original manuscript or figure is overwritten. Original repository figures are historical inputs only; corrected outputs go to `reproduction/`.

## Protocol mapped to code

| Paper requirement | Implementation and verification |
|---|---|
| Section 2: independent uniform theta1, theta2, xi on Euclidean unit spheres | Independent Gaussian blocks, normalized separately in `src/model.py`; each input block has dimension N, not 2N, where N is the group order. |
| Quadratic predictor (1/M) sum a xi (theta1(g1)+theta2(g2))² | Existing factorization retained with effective a = scale_out scale_in². Notebook initialization 0.5,0.5 means a=0.125, NOT 0.5. |
| Complete composition table, summed CE | `cross_entropy_high_precision`; plotting mean CE divides by N² and labels it explicitly. Float64 is preserved. |
| Stage I: fixed a, separate tangent projections | `project_direction_gradients`, projected Euler followed by per-sphere retraction. No momentum, Adam, decay, gradient clipping, or extra width factor. |
| Section 5 shared Abelian flow | Single shared theta; half the theta step, full xi step. Kept distinct from independent-input warm-up and appendix experiments. |
| Continuous time | t = completed updates times dt for summed risk. Width-normalized time t/M may also be reported, explicitly labeled. |
| Stage II: frozen directions; direct a update | `FrozenScaleModel`, initialized from effective a with exact logit continuity. `scale_in` and `scale_out` are NOT independently optimized. |
| Section 2 independent coefficients | Untied learning-rate multiplier 1. |
| Section 4.5 tied scalar, Figure 8 untied comparison | Tied scalar sums gradients across neurons. Untied comparison multiplier M, explicitly a different time convention from Section 2. |
| Positive proportionality | Real Hermitian cosine and positive-ray residual, alongside original phase-insensitive cosine. Zero blocks are undefined and counts are recorded. |
| Spectral diagnostics | Transform raw trained directions, report norms separately; no post-training normalization to hide training violations. |

The upstream Frobenius notebooks use SGD, width 512, seed 42, and effective a=0.125. Stage-I trajectory: dt=0.5, 15,000 steps (T=7,500). Scale-comparison Stage I: dt=0.1, 65,000 steps (T=6,500); Stage II: dt=0.1, 40,000 steps (T=4,000). These are preserved and checked by step halving.

The upstream Z3 x Z5 running example uses width 1,024 and the Z2 x Z3 x Z5 distribution uses width 50,000, both seed 42. Their original Adam optimizer (dt=0.001, 1,000 steps) does not define the paper's gradient flow. Corrected SGD uses dt=0.02 M and T=80 M, with equal-horizon step-halving checks. These are documented replacement numerical settings, not recovered original settings. Finite-time results and convergence diagnostics must be reported rather than assumed.

The single-frequency and prealigned experiments retain shared-input width 16, effective a=0.125, SGD dt=0.01, horizons 70 and 210, respectively, with torch seed 42 and NumPy initialization seeds 0 and 1.

The repository contains S4 task/basis implementations and a deprecated shared-input S4 notebook, but no identified generator of the paper’s independent-input Figure-5b distribution. A new explicit configuration is required; no claim of bit-for-bit reproduction of the unavailable original run is possible.

## Figure inventory

| PDF figure | Source asset(s) | Generating notebook / corrected dependency |
|---|---|---|
| 1a | modular_addition_xi_heatmap_example.jpeg | modular_addition_unshared_running_example; source saves .jpg (extension mismatch) |
| 1b | modular_addition_phase_alignment.jpg | running example; larger modular notebook also writes same filename (collision removed in runner) |
| 2a | modular_addition_phase_circles.jpg | running example |
| 2b | modular_addition_freq_distribution_histogram.jpg | running example |
| 3a | cayley_Z12.jpg, Z12_spectral_basis.jpg | cayley_graphs_and_irreps; conceptual, independent of training |
| 3b | cayley_A4.jpg, A4_spectral_basis.jpg | cayley_graphs_and_irreps; conceptual, independent of training |
| 4a,b | center-stable-manifold.001.jpeg, .002.jpeg | conceptual illustrations; no experiment, generating source absent |
| 5a | modular_addition_freq_distribution.jpg | modular_addition_unshared; width 50,000 |
| 5b | symmetric_group_irrep_distribution.jpg | generator/settings absent; new S4 reproduction |
| 6 | frobenius21_dft_heatmap.jpg | frobenius21_group_unshared; heatmap not saved under this filename upstream |
| 7a,b,c | frobenius21_accuracy.jpg, frobenius21_alignment.jpg, frobenius21_rank1.jpg | frobenius21_group_unshared |
| 8a,b,c | frobenius21_scale_comparison_loss.jpg, frobenius21_scale_tied.jpg, frobenius21_scale_perneuron.jpg | frobenius21_scale_comparison |
| 9a,b | abelian_phase_alignment.jpg, abelian_scale_competition.jpg | modular_addition_single_freq_phase_convergence |
| 10 | modular_addition_dft_heatmap_example.jpg | running example |
| 11 | modular_addition_dft_heatmap.jpg | modular_addition_unshared; save missing upstream |

Additional dependent outputs include raw norm checks, summed/mean loss and accuracy curves, conjugate-orbit energy concentration, signed and unsigned alignment, rank trajectories, full-neuron representation counts and energy averages, checkpoint arrays, and step-size comparisons. Standard unconstrained Adam training is an explicitly separate comparison, not a paper-flow reproduction.

## Numerical interpretation

Retraction after a tangent Euler step has local error O(dt²); convergence to the flow is tested at equal physical horizons. Neither reaching high accuracy nor a phase-insensitive alignment score establishes the theorem's stronger claims. The small-logit approximation in the analysis is not substituted for the actual CE risk in numerical runs. Positivity is monitored without clipping, since clipping changes the prescribed ODE.

Status and exact commands will be recorded in `reproduction/manifest.json` and the final report after runs complete.

## Convergence and plotting horizons

Following the user's clarification, original step counts are reference horizons, not stopping guarantees. Inspect all-neuron concentration, signed alignment, high-dimensional singular-value ratios, accuracy, and Stage-II CE. Extend runs when these are still changing; preserve the reference checkpoint and record extensions separately. Plot axes can focus on a relevant recorded interval, but unabridged histories remain available. Never choose a plot window to omit an unfavorable endpoint or relabel optimization steps as physical time.

The main result is assessed using corrected experimental evidence; fixing implementation mismatches does not by itself refute it.

## Compute

Dedicated SkyPilot cluster `group-composition-cpu`, Nebius context `jianliang-nebius-credential`, namespace `agentic-rl`, 14 requested CPUs and 48 GB RAM, no GPUs. Persistent root `/workspace/data/group-composition-reproduction-20261013`; all environment and results reside under that task-specific directory. No other reservation is reused, modified, or released.

## Exact CPU optimization

The wide-run engine uses two passes over neuron chunks (1,024 neurons per chunk). The first computes the complete predictor; the second computes every neuron's exact gradient using the same full-table softmax residual. This is algebraic blocking, not stochastic or mini-batch training and not the small-logit surrogate. Tests compare both shared and independent updates against autograd across chunk sizes 1, 4, and larger than the width. Checkpointed runs may cross from the original autograd engine to this equivalent engine; earlier code hashes and logs are retained. CPU benchmarking, exact environment versions, and source hashes accompany the runs.

The shared-input phase and competition cases are intentionally width 16 with specialized initialization. Their purpose is to isolate neuronal dynamics, not demonstrate perfect finite-network accuracy. The wider independent-input runs address the composition classifier.

## User-authorized width increase

The 512-neuron reference is retained without selecting an earlier perfect-accuracy checkpoint: its final accuracy is 440/441 at T=7,500. To demonstrate finite-network perfect accuracy, the user authorized increasing width. The main Frobenius run uses M=2,048, seed 42, dt=2, and 20,000 updates (T=40,000, T/M=19.53125); its refinement uses dt=1 and 40,000 updates. Main Frobenius figures use the refined trajectory. This changes width and the numerical horizon, while preserving the predictor, initialization law, objective, and sphere geometry.

Stage II reuses that refined Stage-I endpoint only after verifying a strictly positive minimum correct-class margin over all 441 pairs. It uses direct coefficients initialized continuously at a=0.125, with T=40,000, dt=1 and dt=0.5 refinements. Both Stage-II step sizes start from identical frozen directions. The three variants are a tied scalar, independent coefficients with multiplier M (Figure 8), and independent coefficients with multiplier 1 (Section 2). The original 512-neuron Stage-II experiment remains a labeled reference and is not used to claim the positive-margin hypothesis.

Main Figure 7 alignment is the signed real Hermitian cosine, which can establish positive proportionality. The paper's unsigned cosine is also regenerated under `frobenius21_alignment_unsigned`; it cannot distinguish a negative or complex proportionality factor. The rank-one summary excludes 1D irreps, whose second singular value is identically zero. Error bars are cross-neuron standard deviations, not confidence intervals. Figures 1 and 10 deliberately show 15 and 20 neurons respectively, matching their captions; no neurons are selected for favorable appearance.

## Reproduction commands

Use Python 3.10+ with the exact package versions in `reproduction/requirements-lock-cpu.txt`. Create a separate environment, install the CPU PyTorch wheel, then run from the repository root:

```bash
python -m pytest -q
python scripts/reproduce.py --experiment phase --step-divisor 1 --threads 2
# Repeat for competition, f21, f21_scale, z15, z30, s4, f21_wide;
# repeat every configuration with --step-divisor 2.
python scripts/stage2_from_checkpoint.py --step-divisor 1 --threads 2
python scripts/stage2_from_checkpoint.py --step-divisor 2 --threads 2
python scripts/plot_reproduction.py
python scripts/plot_conceptual.py
python scripts/validate_reproduction.py
```

`run_cpu_suite.sh` and `run_wide_frobenius.sh` contain the exact CPU batch commands. The saved histories include every logged point from t=0 through the final time; numerical checks compare equal physical horizons. The common test tolerances and numerical-claim criteria are explicit in `scripts/validate_reproduction.py`. They are finite-run diagnostics, not proofs of asymptotic claims. Reference results that fail a criterion are reported rather than suppressed.

## S4 convergence-based horizon revision

S4 initially reserved T=80M as a conservative computational budget, with no published original setting to match. After the coarse trajectory attained perfect accuracy, mean concentration above 1-2e-9 and higher-dimensional mean rank ratios below 4.1e-6 at 1,520 updates, its final endpoint was set to 2,400 updates (T=2,400,000=48M). The independent refinement uses 4,800 half-sized updates at the identical physical horizon. All earlier samples and the original planned configuration are preserved. This is an explicit convergence-based numerical choice, not an omission of an unfavorable endpoint.

The saved S4 checkpoint was already at update 2,040 when the scheduling change completed, so the revised endpoint is 2,400 (not the initially proposed 2,000); no completed updates are discarded.
