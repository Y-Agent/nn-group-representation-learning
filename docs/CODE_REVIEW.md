# Code review and cleanup - 14 October 2026

The review covers the numerical library, group tasks and bases, training entry
points, checkpoint lifecycle, paper rendering, environment setup, and maintenance
tests. The accepted experiment archives and manuscript figures are the reference.
This review does not claim a fresh full-length reproduction of all training runs
or an independent verification of the paper's proofs.

## Findings and repairs

| Priority | Finding and consequence | Repair and evidence |
|---|---|---|
| High | S3 cycle matrices were assigned in reverse order relative to the task's permutation-composition convention. The resulting matrices did not satisfy the representation multiplication law. | Corrected the two assignments in `src/basis/symmetric.py`. Exhaustive multiplication checks cover S2, S3, S4, A3, A4, A5, and Frobenius-21. The affected paper runs use other groups; their accepted figures are unchanged. |
| High | Starting Stage I or loading a Stage-I-only checkpoint could leave a previous Stage-II predictor active. Predictions and histories could describe different stages. | Clear stale scale models at transitions, reset the appropriate histories, validate configuration before loading, and restore all recorded history fields. Regression tests exercise both transitions. |
| High | Stage-II completion files could be reused with a different frozen source model or step configuration. | Added model fingerprints and explicit run contracts before resuming or skipping any variant. Conflicting or untracked output directories fail. Interrupted and uninterrupted trajectories agree exactly for tied and both untied variants. |
| Medium | Fourier basis caching/conversion could discard float64 precision; complex64 non-Abelian input could fail matrix multiplication against complex128 bases. | Cache Fourier bases by dtype, preserve double precision, and promote complex inputs consistently. Double-precision and complex-input round-trip tests pass. |
| Medium | Loading model parameters replaced Parameter objects, potentially invalidating optimizer references, and could partially modify a model before reporting a shape error. | Validate shapes first, then copy values under `no_grad` while preserving parameter identity, device, and dtype. Regression tests verify identity and rejection. |
| Medium | Zero-valued numeric CLI overrides were ignored by truthiness checks; a zero-epoch Stage-I run accessed an empty history. | Use explicit `is not None` checks and support a zero-step run. CLI tests cover seed 0 and zero epochs. |
| Medium | A crash after saving the final Stage-I checkpoint but before writing the completion marker could duplicate the final observation on resume. | Replace an existing observation at the resumed horizon rather than appending a duplicate. A final-checkpoint recovery test covers this path. |
| Medium | The optimized full-table update assumed lexicographic inputs without checking them; unknown spectral groups fell through to S4. | Validate the table ordering and fail on unsupported group identifiers. Tests reject shuffled input and unknown groups. |
| Medium | Configuration parsing could modify its caller's dictionary; invalid horizons, dimensions, rates, or chunk sizes could fail late or give misleading results. | Copy configuration inputs and validate the relevant numerical bounds at entry points. Regression tests cover invalid settings. |
| Medium | Accepted plots depended on successive presentation scripts that executed earlier drafts, changed global plotting behavior, and relied on implicit local setup. | Replaced that chain with a small `paper_reproduction` package. Frozen original notebook cells run in an isolated adapter; final distributions and dynamics have explicit render functions and settings. All accepted raster comparisons pass. |
| Medium | Setup notes conflated notebook-rendering and final-layout environments, and described the displayed alignment metric as signed. | Recorded both exact environments, font requirements, input hashes, figure map, time conventions, width/horizon changes, and unsigned metric interpretation in `REPRODUCTION.md`. Historical notes are retained with a superseded warning. |

## Refactoring decisions

- `src/flow.py` remains the numerical flow implementation. Initialization and
  raw-weight diagnostics live in `src/experiment.py`; resumable scale training
  lives in `src/scale_training.py`; checkpoint guards live in `src/checkpointing.py`.
- Training scripts now orchestrate those functions rather than duplicating them.
  The Stage-II entry point is import safe, takes a portable runs path, and verifies
  a completed positive-margin Stage-I source.
- `paper_reproduction` has four commands: `render`, `verify`, `train`, and
  `package`. Rendering does not train. The training plan is printed unless
  `--execute` is supplied. New output cannot overlap archived inputs or overwrite
  an existing render directory.
- Twelve successive plotting/packaging drafts are preserved as text in
  `scripts/legacy/`. Two old command names remain as small compatibility wrappers.
  The original notebook drawing cells and runtime helpers remain frozen and
  hash checked, preserving their scientific definitions and plotting conventions.
- Active Python files use one formatter/linter configuration. Unused imports and
  intermediates were removed. Historical notebook helpers are deliberately excluded.
- The portable archive contains the required reference inputs and accepted
  baselines, so a recipient does not need a personal Dropbox path or a live pod.
  It is a reproduction subset, not a dump of all exploratory notebooks or runs.

## Validation

Validation was performed in fresh virtual environments, not only the previous
working environment. The exact package lists are in `requirements-paper.txt` and
`requirements-notebook.txt`.

| Check | Result |
|---|---|
| Unit and regression suite | 59 tests pass. Covers geometry, loss normalization, gradients, stage boundary, step refinement, representations, precision, checkpoint transitions, resumed trajectories, and CLI/output safeguards. |
| Lint and formatting | `ruff check` and `ruff format --check` pass for `src`, `scripts`, `tests`, and `paper_reproduction`. |
| Short before/after training | Four 12-update trajectories (`phase`, `f21`, `f21_wide`, `z15`) have identical saved final arrays, including diagnostics. |
| Accepted JPEG assets | All nine affected assets reproduce exactly at the decoded-pixel level. |
| Accepted Figure 7 panels | Alignment, accuracy, scale/loss, and per-neuron-scale panel PNGs reproduce exactly. |
| Diagnostic arrays | Seven arrays pass absolute tolerance 1e-12 with zero relative tolerance. Maximum observed difference is 2.22e-16; time and accuracy are exact. |
| Baseline integrity | Baseline hashes are checked against a separate saved contract before rendering, verification, or packaging. The renderer never updates the baseline. |
| Provenance | Each render records input/source/output hashes, package versions, actual font paths/hashes, and layout checks. |

The tests are targeted at mathematical identities, externally observable behavior,
and previously failing cases. Pixel comparison provides a separate check that code
cleanup did not silently change the approved presentation.

## Scientific interpretation preserved

The main runs retain separate Euclidean sphere constraints, summed CE for training,
projected Euler updates and retraction, frozen Stage-I scales, and direct effective
scale optimization in Stage II. The untied comparison with multiplier M remains
distinct from the independent flow with multiplier 1. Full saved histories are
preserved even where the plot displays only total time 40,000-50,000.

Displayed alignment is the original unsigned complex cosine; it supports
proportionality without by itself establishing positive-real proportionality.
Signed diagnostics remain separately available. Figure 5 retains the original
energy sampling convention. These are explicit interpretation limits, not changes
made to obtain a preferred visual result.

## Limits and follow-up boundaries

- Long CPU experiments were not rerun for this cleanup. Archived-result replay,
  short before/after trajectories, and numerical unit tests provide the evidence
  reported here. GPU execution and fresh full-length cross-platform trajectories
  are not certified by this review.
- Exact Figure 5 typography requires the recorded Palatino font. It is not
  redistributed; a different font can preserve scientific values while failing
  pixel equality. PDF metadata can differ even when the rendered image is identical.
- Some unaffected conceptual illustrations have no generating source in the
  repository. Their accepted files are retained; the affected-only workflow does
  not claim to regenerate them.
- A Git-only checkout lacks ignored checkpoint data. Use the portable archive or
  supply the required files listed by `reproduction/paper-contract.json`.
- New results are supported with `--new-results`, but their equality to the
  accepted baseline is neither assumed nor manufactured.
- The original pre-cleanup tree was already modified by earlier scientific repairs.
  Cleanup is applied to that state, with a file backup; no Git reset, commit, push,
  or manuscript update is part of this review.

For commands, experiment settings, manuscript filenames, and plotting conventions,
see [the reproduction guide](../REPRODUCTION.md).
