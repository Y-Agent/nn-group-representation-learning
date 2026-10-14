"""Bridge the frozen upstream notebook cells to portable experiment inputs."""


def main():
    """Isolated adapter for the hash-verified upstream plotting cells.

    Invoked by the public CLI, not imported by the numerical training code.
    """
    import argparse
    import ast
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    REVISED = True
    OUT = args.output.resolve()
    SOURCES = args.sources.resolve()
    if REVISED:
        from paper_reproduction.distribution import fix_distribution_labels
    revision_checks = {}
    RUNS = args.runs.resolve()
    sys.path.insert(0, str(SOURCES / "runtime"))
    import matplotlib
    import numpy as np
    import torch

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Matplotlib >=3.9 removed this alias. The colormap itself is unchanged.
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    from src.basis import FourierBasis, Frobenius21Basis
    from src.basis.symmetric import SymmetricGroupBasis

    torch.set_num_threads(2)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "notebooks").mkdir(exist_ok=True)
    os.chdir(OUT / "notebooks")
    PROVENANCE = json.loads((SOURCES / "provenance.json").read_text())
    for path, digest in PROVENANCE["helpers"].items():
        assert hashlib.sha256((SOURCES / "runtime" / path).read_bytes()).hexdigest() == digest
    CELLS = {n: json.loads((SOURCES / (n + ".json")).read_text()) for n in PROVENANCE["cells"]}
    for name, cells in CELLS.items():
        for i, source in cells.items():
            assert hashlib.sha256(source.encode()).hexdigest() == PROVENANCE["cells"][name][i]

    exports = []
    distributions = {}
    current = {}
    original_savefig = plt.savefig
    pdf = PdfPages(OUT / "affected-paper-plots.pdf")

    def savefig(path, *args, **kwargs):
        if current.get("export_as"):
            path = str(Path(path).with_name(current["export_as"] + Path(path).suffix))
        fig = plt.gcf()
        stem = Path(path).stem
        if REVISED and stem in [
            "modular_addition_freq_distribution",
            "symmetric_group_irrep_distribution",
        ]:
            revision_checks[stem] = fix_distribution_labels(fig)
        if REVISED and stem in [
            "frobenius21_diagnostics_compact",
            "modular_addition_freq_distribution",
            "symmetric_group_irrep_distribution",
        ]:
            kwargs["bbox_inches"] = None
        original_savefig(path, *args, **kwargs)
        fig.savefig(
            OUT / "figures" / (stem + ".pdf"),
            bbox_inches=None
            if REVISED
            and stem
            in [
                "frobenius21_diagnostics_compact",
                "modular_addition_freq_distribution",
                "symmetric_group_irrep_distribution",
            ]
            else "tight",
        )
        pdf.savefig(
            fig,
            bbox_inches=None
            if REVISED
            and stem
            in [
                "frobenius21_diagnostics_compact",
                "modular_addition_freq_distribution",
                "symmetric_group_irrep_distribution",
            ]
            else "tight",
        )
        exports.append(dict(filename=stem + ".jpg", **current))

    def show(*args, **kwargs):
        # Notebook cells 15 (F21) and 11 (Z30) display but do not save their figure.
        extra = current.get("external_export")
        if extra:
            savefig("../figures/" + extra + ".jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.close("all")

    plt.savefig = savefig
    plt.show = show

    def cell(name, i, env, run, external_export=None, source_start=None, export_as=None):
        nonlocal current
        current = dict(
            notebook="notebooks/" + name + ".ipynb",
            cell_index=i,
            run=run,
            cell_sha256=PROVENANCE["cells"][name][str(i)],
        )
        if external_export:
            current["external_export"] = external_export
        source = CELLS[name][str(i)]
        if source_start:
            source = source[source.index(source_start) :]
            current["source_section_start"] = source_start
            current["executed_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if export_as:
            current["export_as"] = export_as
        if REVISED and name == "modular_addition_unshared" and i == 11:
            assert source.count("num_components = 40") == 1
            source = source.replace("num_components = 40", "num_components = 20")
            current["authorized_presentation_edit"] = "First 20 neurons, in original order"
            current["executed_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        print(f"Executing original {name} cell {i}", flush=True)
        # Historical notebook assets were rendered on Linux using DejaVu.
        # The approved enlarged Figure 5 was rendered on macOS using Palatino.
        distribution = name == "modular_addition_unshared" and i == 15
        plt.rcParams["font.serif"] = (
            ["Palatino", "P052", "DejaVu Serif"] if distribution else ["DejaVu Serif"]
        )
        exec(compile(source, name + f":cell{i}", "exec"), env)
        if distribution:
            distributions[export_as or "modular_addition_freq_distribution"] = env[
                "freq_group_data"
            ]

    def tensors(run, endpoint):
        with np.load(RUNS / run / (endpoint + ".npz")) as data:
            w, v = data["W_in"].copy(), data["W_out"].copy()
        n = v.shape[0]
        norm_error = max(
            np.abs(np.linalg.norm(w[:, :n], axis=1) - 1).max(),
            np.abs(np.linalg.norm(w[:, n:], axis=1) - 1).max(),
            np.abs(np.linalg.norm(v, axis=0) - 1).max(),
        )
        assert norm_error < 1e-12, (run, norm_error)
        return torch.from_numpy(w), torch.from_numpy(v)

    # Original Z15 neuron selections (15, 128, 512, and 1024) are unchanged.
    for name, run, moduli, indices in [
        ("modular_addition_unshared_running_example", "z15_h2", [3, 5], [9, 10, 11, 13, 14, 15]),
        ("modular_addition_unshared", "z30_h2", [2, 3, 5], [9, 11, 15]),
    ]:
        env = {}
        cell(name, 1, env, run)
        wi, wo = tensors(run, "final")
        ii, io = tensors(run, "initial")
        env.update(
            config=SimpleNamespace(d_vocab=wo.shape[0]),
            trainer=SimpleNamespace(
                model=SimpleNamespace(W_in=wi, W_out=wo), basis=FourierBasis(moduli)
            ),
            W_in_init=ii,
            W_out_init=io,
            M=wi.shape[0],
        )
        for i in indices:
            cell(
                name,
                i,
                env,
                run,
                "modular_addition_dft_heatmap" if run == "z30_h2" and i == 11 else None,
            )

    # The user confirmed that Figure 5b reuses the Abelian distribution code.
    # Its sort + drawing + annotation + export section is executed verbatim.
    # Only the input categories and non-Abelian energy calculation differ.
    name = "modular_addition_unshared"
    run = "s4_h2"
    env = {}
    cell(name, 1, env, run)
    s4basis = SymmetricGroupBasis(4)
    partitions = s4basis.partitions
    dims = [s4basis.partition_dimensions[p] for p in partitions]
    with np.load(RUNS / run / "final.npz") as data:
        coefficients = data["coefficients"].copy()
        energy = data["energy"][0].copy()
        stored_omega = data["omega"].copy()
        assert np.max(np.abs(data["norms"] - 1)) < 1e-12
    winner = energy.argmax(axis=1)  # theta1's dominant irrep, as in the Abelian cell.
    M = len(winner)
    omega = np.zeros(M)
    offset = 0
    for i, d in enumerate(dims):
        A, B, C = [x[:, offset : offset + d * d].reshape(M, d, d) for x in coefficients]
        if i != 0:
            omega += d * np.einsum("mij,mij->m", (B @ A).conj(), C).real
        offset += d * d
    assert offset == 24
    assert np.allclose(omega, stored_omega, rtol=1e-12, atol=1e-15)
    s4_groups = []
    for i, partition in enumerate(partitions):
        neurons = np.flatnonzero(winner == i)
        # Original cell: first 100 neurons for each nontrivial self-conjugate irrep.
        sampled = neurons if i == 0 else neurons[:100]
        mean = float(omega[sampled].mean()) if len(sampled) else 0.0
        label = str(partition) + ("\n (triv)" if i == 0 else "")
        s4_groups.append((label, len(neurons) / M, mean, i == 0))
    env["freq_group_data"] = s4_groups.copy()
    cell(
        name,
        15,
        env,
        run,
        source_start="# Sort by proportion (ascending order)",
        export_as="symmetric_group_irrep_distribution",
    )

    # Recompute the original diagnostic (including original unsigned d_sim) from
    # every saved raw snapshot, with final-dominant-irrep selection in cell 17.
    name = "frobenius21_group_unshared"
    run = "f21_wide_h2"
    env = {}
    cell(name, 1, env, run)
    wi, wo = tensors(run, "final")
    ii, io = tensors(run, "initial")
    env.update(
        model=SimpleNamespace(W_in=wi, W_out=wo),
        basis=Frobenius21Basis(),
        vocab_size=21,
        WIDTH=wi.shape[0],
        W_in_init=ii,
        W_out_init=io,
    )
    for i in [10, 12]:
        cell(name, i, env, run)
    env["vline_positions"] = [start - 0.5 for start, _ in env["irrep_boundaries"] if start > 0]
    cell(name, 15, env, run, "frobenius21_dft_heatmap")
    source = CELLS[name]["8"]
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_all_metrics_at_epoch"
    )
    exec(
        compile(ast.get_source_segment(source, fn), name + ":original_metric_function", "exec"), env
    )
    checkpoint = torch.load(RUNS / run / "latest.pt", map_location="cpu", weights_only=False)
    snaps = checkpoint["snapshots"]
    records = checkpoint["records"]
    dt = checkpoint["config"]["dt"]
    times = np.array([r["time"] for r in records])
    stride = float(times[1] - times[0])
    assert times[0] == 0 and np.allclose(np.diff(times), stride)
    assert [s["step"] for s in snaps] == [r["step"] for r in records]
    alhist = {
        l: {c: [] for c in ["cond1", "cond2", "cond3"]}
        for l in env["basis"].irrep_labels
        if l != "trivial"
    }
    rhist = {
        l: {c: [] for c in ["theta1", "theta2", "xi"]}
        for l in env["basis"].irrep_labels
        if env["basis"].irrep_dimensions[l] > 1
    }
    ghist = [[], [], []]
    for idx, snap in enumerate(snaps):
        m = SimpleNamespace(
            W_in=torch.from_numpy(snap["W_in"]), W_out=torch.from_numpy(snap["W_out"])
        )
        al, g1, g2, go, labels, r1 = env["compute_all_metrics_at_epoch"](m, env["basis"], 21)
        for dest, g in zip(ghist, [g1, g2, go]):
            dest.append(g)
        for l, terms in alhist.items():
            for c in terms:
                terms[c].append(al[l][c])
        for l, terms in rhist.items():
            for c in terms:
                terms[c].append(r1[l][c])
        if idx % 10 == 0:
            print(f"Original F21 metrics: {idx + 1}/{len(snaps)} snapshots", flush=True)
    # These notebook input arrays contain recorded samples rather than every update.
    # LR is consequently the sample-time increment, NOT the optimizer step size.
    # Both original x-axis expressions reproduce the saved physical times exactly.
    env.update(
        LR=stride,
        epochs_tracked=(times / stride).astype(int).tolist(),
        accuracies=[r["accuracy"] for r in records],
        alignment_history_all=alhist,
        grouped_energy_history_theta1=ghist[0],
        grouped_energy_history_theta2=ghist[1],
        grouped_energy_history_xi=ghist[2],
        grouped_labels_cache=labels,
        rank1_history=rhist,
    )
    assert np.array_equal(np.asarray(env["epochs_tracked"]) * env["LR"], times)
    cell(name, 17, env, run)
    # Retain original metrics, selection, means and standard deviations for the
    # accepted Figure 7 renderer. No intermediate draft figures are generated.
    arrays = dict(
        time=env["time_arr"],
        accuracy_time=env["time_acc"],
        accuracy=env["acc_arr"],
        alignment_mean=np.asarray([env[k + "_mean"] for k in ["cond2", "cond3", "cond1"]]),
        alignment_std=np.asarray([env[k + "_std"] for k in ["cond2", "cond3", "cond1"]]),
        rank_mean=np.asarray([env[k + "_sv_mean"] for k in ["theta1", "theta2", "xi"]]),
        rank_std=np.asarray([env[k + "_sv_std"] for k in ["theta1", "theta2", "xi"]]),
    )
    np.savez_compressed(OUT / "frobenius_diagnostic_plot_data.npz", **arrays)

    # Use an identical 400-time-unit sample grid in both stages. All are actual
    # stored observations. Excluding the duplicate Stage-I endpoint makes the
    # unchanged original '+ S2_LR' offset start Stage II exactly at T=40000.
    name = "frobenius21_scale_comparison"
    run = "f21_scale_wide_h2"
    env = {}
    cell(name, 1, env, run)
    tied = np.load(RUNS / run / "tied.npz")
    pn = np.load(RUNS / run / "untied_matched.npz")
    assert np.array_equal(tied["time"], pn["time"])
    mask = np.flatnonzero(np.isclose(np.mod(tied["time"], stride), 0))
    t2 = tied["time"][mask]
    assert np.allclose(t2, np.arange(len(t2)) * stride)
    env.update(
        S1_LR=stride,
        S2_LR=stride,
        S2_EPOCHS=len(t2),
        WIDTH=wi.shape[0],
        s1_losses=[r["loss_sum"] for r in records[:-1]],
        s2_tied_losses=tied["loss_sum"][mask],
        s2_pn_losses=pn["loss_sum"][mask],
        s2_tied_a_hist=tied["a"][mask].reshape(-1),
        s2_pn_traj=pn["a"][mask],
    )
    cell(name, 11, env, run)
    assert env["s2_offset"] == times[-1]
    assert np.array_equal(env["s2_times"], times[-1] + t2)
    cell(name, 12, env, run)
    (OUT / "distribution-data.json").write_text(
        json.dumps(distributions, indent=2, default=lambda x: x.item())
    )
    pdf.close()
    assert len(exports) == 15, len(exports)
    manifest = dict(
        source_commit=PROVENANCE["commit"],
        exports=exports,
        original_plot_cells_modified=REVISED,
        original_helper_files_modified=False,
        presentation_revision=REVISED,
        presentation_checks=revision_checks,
        data_adapter=dict(
            f21_optimizer_dt=dt,
            plot_sample_time_increment=stride,
            stage2_optimizer_dt=0.5,
            stage2_sample_indices=mask.tolist(),
            interpolation=False,
            stage2_start_time=float(times[-1]),
            s4=dict(
                authorized_reuse="Abelian distribution sorting/drawing section, unchanged",
                dominant_irrep="theta1 energy argmax",
                energy="sum over nontrivial rho of d_rho Re Tr(theta2_hat theta1_hat xi_hat^H)",
                nontrivial_irrep_averaging="first 100 assigned neurons, original Abelian convention",
                categories=s4_groups,
            ),
            stage1_loss_endpoint="The boundary observation is drawn as Stage II at the same physical time.",
        ),
        excluded_figures={
            "3": "conceptual, unaffected",
            "4": "conceptual, unaffected",
            "9": "shared-input experiments, unaffected by separate-input repair",
        },
        missing_generator={},
        environment=dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            matplotlib=matplotlib.__version__,
        ),
        source_data_hashes={},
    )
    for run in ["z15_h2", "z30_h2", "s4_h2", "f21_wide_h2", "f21_scale_wide_h2"]:
        for file in ["config.json", "initial.npz", "final.npz", "history.json"] + (
            ["tied.npz", "untied_matched.npz"] if "scale" in run else []
        ):
            p = RUNS / run / file
            manifest["source_data_hashes"][run + "/" + file] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    manifest["figure_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((OUT / "figures").glob("*"))
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"DONE: {len(exports)} plot files; presentation_revision={REVISED}; no experiment data changed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
