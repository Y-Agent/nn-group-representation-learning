"""Render every training-dependent paper panel solely from corrected run arrays."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.spectral import Spectrum

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
P = Path(__file__).resolve().parents[1] / "reproduction"
F = P / "figures"
F.mkdir(parents=True, exist_ok=True)
RUNS = P / "runs"
provenance = {}
COL = ["#163e70", "#a32b26", "#23836e"]


def save(fig, name, runs):
    fig.tight_layout()
    for ext in ["pdf", "svg", "jpg"]:
        fig.savefig(F / (name + "." + ext), bbox_inches="tight")
    plt.close(fig)
    provenance[name] = dict(runs=runs, formats=["pdf", "svg", "jpg"])


def load(name):
    root = RUNS / name
    return (
        json.loads((root / "config.json").read_text()),
        json.loads((root / "history.json").read_text()),
        np.load(root / "final.npz"),
        torch.load(root / "latest.pt", map_location="cpu", weights_only=False),
    )


def lineplot(t, series, labels, ylabel, name, run, baseline=None, endpoint=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for j, (v, l) in enumerate(zip(series, labels)):
        ax.plot(t, v, color=COL[j % 3], label=l, lw=1.8)
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1)
    ax.set(xlabel="Time $t$", ylabel=ylabel)
    ax.grid(alpha=0.15)
    if endpoint is not None:
        ax.text(0.97, 0.06, endpoint, transform=ax.transAxes, ha="right", fontsize=8)
    if len(labels) > 1:
        ax.legend(frameon=False, fontsize=9)
    save(fig, name, [run])


def heatmap(spec, final, name, run, output_only=False):
    # Figure 1 requests 15; Figures 6, 10, and 11 request 20.
    count = 15 if output_only else 20
    ids = np.arange(min(count, len(final["winner"])))
    ids = ids[np.argsort(final["winner"][ids], kind="stable")]
    display = final["coefficients"][:, ids].copy()
    offset = 0
    for d in spec.dims:
        display[:, :, offset : offset + d * d] *= np.sqrt(d)
        offset += d * d
    if all(d == 1 for d in spec.dims):
        pairs = [o for o in spec.orbits if len(o) == 2]
        single = [o[0] for o in spec.orbits if len(o) == 1]
        order = [o[0] for o in pairs] + single + [o[1] for o in pairs][::-1]
        ticklabels = [spec.labels[i] for i in order]
        boundaries = [len(pairs) - 0.5, len(pairs) + len(single) - 0.5]
        display = display[:, :, order]
        centers = []
    else:
        ticklabels = []
        boundaries = []
        centers = []
        off = 0
        pretty = {
            "trivial": r"$1$",
            "omega": r"$\omega$",
            "omega_conj": r"$\bar\omega$",
            "rho_3": r"$\rho_2$",
            "rho_3_conj": r"$\rho_2^\vee$",
        }
        for label, d in zip(spec.labels, spec.dims):
            ticklabels += [f"({i + 1},{j + 1})" for i in range(d) for j in range(d)]
            centers.append((off + (d * d - 1) / 2, pretty.get(label, label)))
            off += d * d
            boundaries.append(off - 0.5)
        boundaries = boundaries[:-1]
    if output_only:
        display = display[2:]
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.7))
        axes = np.array([[axs[0]], [axs[1]]])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.5))
    vmax = max(abs(display.real).max(), abs(display.imag).max())
    for j, c in enumerate(display):
        for i, fn in enumerate([np.real, np.imag]):
            ax = axes[i, j]
            im = ax.imshow(
                fn(c), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
            )
            title = (["$\\xi$"] if output_only else ["$\\theta^1$", "$\\theta^2$", "$\\xi$"])[j]
            ax.set_title(title + (" real" if i == 0 else " imaginary"))
            ax.set_ylabel("Neuron")
            ax.set_xticks(np.arange(spec.n), ticklabels, rotation=90, fontsize=6.5)
            ticks = [k for k in [0, 4, 9, 14, 19] if k < len(ids)]
            ax.set_yticks(ticks, [str(ids[k]) for k in ticks])
            for line in boundaries:
                ax.axvline(line, color="black", ls="--", lw=0.65, alpha=0.65)
            if i == 1:
                for xx, label in centers:
                    ax.text(
                        xx,
                        -0.40,
                        label,
                        ha="center",
                        va="top",
                        transform=ax.get_xaxis_transform(),
                        fontsize=8,
                    )
    fig.subplots_adjust(right=0.9, wspace=0.32, hspace=0.48, bottom=0.25 if centers else 0.2)
    cb = fig.add_axes([0.93, 0.25, 0.015, 0.58])
    fig.colorbar(im, cax=cb, label=r"$\sqrt{d_\rho}\,\widehat\nu[\rho]$")
    for ext in ["pdf", "svg", "jpg"]:
        fig.savefig(F / (name + "." + ext), bbox_inches="tight")
    if output_only:
        fig.savefig(F / (name + ".jpeg"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    provenance[name] = dict(
        runs=[run],
        selection=f"first {count} neurons, sorted by dominant orbit",
        formats=["pdf", "svg", "jpg"],
    )


def distribution(spec, final, name, run, hist=False):
    counts = np.bincount(final["winner"], minlength=len(spec.orbits))
    omega = final["omega"]
    win = final["winner"]
    means = np.array(
        [omega[win == i].mean() if (win == i).any() else np.nan for i in range(len(counts))]
    )
    labels = [
        (r"$\pm$ " + spec.labels[o[0]]) if len(o) == 2 else spec.labels[o[0]] for o in spec.orbits
    ]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.58), 4.2))
    x = np.arange(len(counts))
    if hist:
        ax.bar(x, counts, color="#b5cce0", edgecolor=COL[0], lw=0.7)
    else:
        ax.plot(x, counts / len(win), "o-", color=COL[0], ms=4)
    ax.set(
        ylabel="Neuron count" if hist else "Fraction of neurons",
        xlabel="Irreducible representation (conjugate orbits)",
    )
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    if not hist:
        right = ax.twinx()
        right.plot(x, means, "o--", color=COL[1], ms=4)
        right.set_ylabel(r"Mean energy $\Omega_m$", color=COL[1])
        right.spines["right"].set_visible(True)
    save(fig, name, [run])
    np.savez_compressed(F / (name + "_data.npz"), counts=counts, energy_means=means, labels=labels)


def plot_group(run, heatname=None, prefix="frobenius21"):
    cfg, h, d, ck = load(run)
    spec = Spectrum(cfg["group"])
    t = np.array([r["time"] for r in h])
    if heatname:
        heatmap(spec, d, heatname, run)
    lineplot(
        t,
        [[r["accuracy"] for r in h]],
        ["Accuracy"],
        "Full-table accuracy",
        run + "_accuracy",
        run,
        1,
    )
    lineplot(
        t, [[r["loss_mean"] for r in h]], ["Mean CE"], "Mean cross-entropy", run + "_loss", run
    )
    lineplot(
        t,
        [[r["max_norm_error"] for r in h]],
        ["Norm error"],
        "Maximum unit-norm error",
        run + "_norm_error",
        run,
    )
    if cfg["group"] == "f21":
        snapshots = ck["snapshots"]
        final_orbits = d["winner"]
        ds = [spec.diagnostics(z["W_in"], z["W_out"], False, final_orbits) for z in snapshots]
        tt = np.array([z["step"] * cfg["dt"] for z in snapshots])
        active = d["selected_dimension"] > 1
        for metric, ylabel, name in [
            ("signed", "Positive alignment", "frobenius21_alignment"),
            ("rank", r"$\sigma_2/\sigma_1$", "frobenius21_rank1"),
            ("rank", r"$\sigma_2/\sigma_1$ (all irreps)", "frobenius21_rank1_all_irreps"),
            ("unsigned", "Phase-insensitive alignment", "frobenius21_alignment_unsigned"),
            ("ray", "Positive-ray relative residual", "frobenius21_positive_ray"),
        ]:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            mask = active if name == "frobenius21_rank1" else np.ones(len(active), bool)
            for j in range(3):
                vals = np.stack([z[metric][j, mask] for z in ds])
                mu = np.nanmean(vals, 1)
                sd = np.nanstd(vals, 1)
                ax.plot(
                    tt,
                    mu,
                    color=COL[j],
                    label=(
                        ["$\\theta^1$", "$\\theta^2$", "$\\xi$"]
                        if metric == "rank"
                        else ["Relation 1", "Relation 2", "Relation 3"]
                    )[j],
                )
                idx = np.arange(0, len(tt), max(1, len(tt) // 15))
                ax.errorbar(
                    tt[idx], mu[idx], yerr=sd[idx], fmt="none", ecolor=COL[j], alpha=0.35, capsize=2
                )
            ax.set(xlabel="Time $t$", ylabel=ylabel)
            ax.legend(frameon=False, fontsize=9)
            save(fig, name.replace("frobenius21", prefix), [run])
        lineplot(
            t,
            [[r["accuracy"] for r in h]],
            ["Accuracy"],
            "Full-table accuracy",
            prefix + "_accuracy",
            run,
            1,
            endpoint=f"M = {cfg['width']}; endpoint: {round(h[-1]['accuracy'] * 441)}/441",
        )
        lineplot(
            t,
            [[r["min_margin"] for r in h]],
            ["Minimum margin"],
            "Minimum correct-class margin",
            prefix + "_margin",
            run,
            0,
        )
        lineplot(
            t,
            np.array([r["concentration_mean"] for r in h]).T,
            ["$\\theta^1$", "$\\theta^2$", "$\\xi$"],
            "Dominant-orbit energy fraction",
            prefix + "_concentration",
            run,
            1,
        )
    if run == "z15":
        heatmap(spec, d, "modular_addition_xi_heatmap_example", run, True)
        c = d["coefficients"]
        k = d["selected_irrep"]
        idx = np.arange(len(k))
        a = c[0, idx, k]
        b = c[1, idx, k]
        o = c[2, idx, k]
        phases = np.mod(np.angle(a) + np.angle(b), 2 * np.pi)
        psi = np.mod(np.angle(o), 2 * np.pi)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.scatter(phases, psi, s=8, alpha=0.5, c=d["winner"], cmap="viridis")
        ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "--", color="gray")
        ax.set(xlabel=r"$\phi_1+\phi_2$ (mod $2\pi$)", ylabel=r"$\psi$ (mod $2\pi$)")
        save(fig, "modular_addition_phase_alignment", [run])
        fig = plt.figure(figsize=(11.5, 3.7))
        axes = [fig.add_subplot(1, 3, j, projection="polar") for j in [1, 2]]
        for ax, v, title in zip(axes, [a, b], [r"$\phi_1$", r"$\phi_2$"]):
            ax.scatter(np.angle(v), np.ones(len(v)), c=d["winner"], cmap="viridis", s=9, alpha=0.45)
            ax.set(ylim=(0, 1.1), title=title)
            ax.set_yticks([])
        joint = fig.add_subplot(1, 3, 3)
        joint.scatter(
            np.mod(np.angle(a), 2 * np.pi),
            np.mod(np.angle(b), 2 * np.pi),
            c=d["winner"],
            cmap="viridis",
            s=8,
            alpha=0.45,
        )
        joint.set(
            xlabel=r"$\phi_1$",
            ylabel=r"$\phi_2$",
            xlim=(0, 2 * np.pi),
            ylim=(0, 2 * np.pi),
            title="Joint phase distribution",
        )
        joint.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        joint.set_yticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        save(fig, "modular_addition_phase_circles", [run])
        distribution(spec, d, "modular_addition_freq_distribution_histogram", run, True)
    if run in ("z30", "s4"):
        distribution(
            spec,
            d,
            "modular_addition_freq_distribution"
            if run == "z30"
            else "symmetric_group_irrep_distribution",
            run,
        )


def specialized(run):
    cfg, h, d, ck = load(run)
    z = ck["snapshots"]
    t = np.array([v["step"] * cfg["dt"] for v in z])
    a = np.stack([v["theta_coeff"] for v in z])
    b = np.stack([v["output_coeff"] for v in z])
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    if run == "phase":
        k = np.arange(cfg["width"]) % 14 + 1
        ix = np.arange(cfg["width"])
        prod = a[:, ix, k] ** 2 * b[:, ix, k].conj()
        v = np.divide(prod, abs(prod), out=np.full_like(prod, np.nan), where=abs(prod) > 1e-15).real
        for m in range(cfg["width"]):
            ax.plot(t, v[:, m], color=plt.cm.viridis((v[0, m] + 1) / 2), lw=1.3)
        ax.set(ylabel=r"$\Re(\varphi_m)$", ylim=(-1.05, 1.05))
        name = "abelian_phase_alignment"
    else:
        spec = Spectrum("z15")
        m = 2
        for orb in spec.orbits:
            ax.plot(t, abs(a[:, m, orb[0]]), label=spec.labels[orb[0]], lw=1.5)
        ax.set_ylabel(r"$|\widehat\theta_2[\rho]|$")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        name = "abelian_scale_competition"
    ax.set_xlabel("Time $t$")
    save(fig, name, [run])


def scale_plots(run="f21_scale_wide_h2", prefix="frobenius21"):
    cfg, h, d, ck = load(run)
    root = RUNS / run
    variants = {
        k: np.load(root / (k + ".npz")) for k in ["tied", "untied_matched", "untied_section2"]
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    t = np.array([r["time"] for r in h])
    ax.plot(t, [r["loss_mean"] for r in h], "--", color="gray", label="Stage I")
    for j, k in enumerate(["tied", "untied_matched"]):
        v = variants[k]
        ax.plot(
            t[-1] + v["time"],
            v["loss_sum"] / 441,
            color=COL[j],
            label="Stage II: " + ("tied" if j == 0 else "untied, multiplier M"),
        )
    ax.axvline(t[-1], color="gray", ls=":")
    ax.set(xlabel="Time $t$", ylabel="Mean cross-entropy")
    ax.legend(frameon=False, fontsize=8)
    save(fig, prefix + "_scale_comparison_loss", [run])
    for k, name in [
        ("tied", "frobenius21_scale_tied"),
        ("untied_matched", "frobenius21_scale_perneuron"),
        ("untied_section2", "frobenius21_scale_section2"),
    ]:
        v = variants[k]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        a = v["a"]
        ids = np.arange(min(10, a.shape[1]))
        ids = ids[np.argsort(a[-1, ids])]
        for j in ids:
            ax.plot(v["time"], a[:, j], lw=1.2, alpha=0.8, label=f"$m={j}$")
        if len(ids) > 1:
            ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.set(xlabel="Stage-II time", ylabel="Effective coefficient $a$")
        save(fig, name.replace("frobenius21", prefix), [run])
    table = {
        k: dict(
            initial_a=float(v["a"][0, 0]),
            final_a_mean=float(v["a"][-1].mean()),
            minimum_a=float(v["a_min"].min()),
            final_mean_ce=float(v["loss_sum"][-1] / 441),
            final_accuracy=float(v["accuracy"][-1]),
            boundary_jump=float(v["boundary_max_logit_jump"]),
            lr_multiplier=float(v["lr_multiplier"]),
        )
        for k, v in variants.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for j, k in enumerate(["tied", "untied_matched"]):
        v = variants[k]
        tt = v["time"]
        mean_a = v["a"].mean(1)
        loss = v["loss_sum"] / 441
        label = "Tied" if j == 0 else "Untied, multiplier M"
        axes[0].plot(np.log1p(tt), mean_a, label=label, color=COL[j])
        axes[1].loglog(tt[1:], loss[1:], label=label, color=COL[j])
        late = tt >= tt[-1] / 2
        table[k]["late_loglog_loss_slope"] = float(
            np.polyfit(np.log(tt[late]), np.log(loss[late]), 1)[0]
        )
        table[k]["late_scale_logtime_slope"] = float(
            np.polyfit(np.log1p(tt[late]), mean_a[late], 1)[0]
        )
        table[k]["fit_window"] = [float(tt[late][0]), float(tt[-1])]
    axes[0].set(xlabel=r"$\log(1+t_{II})$", ylabel="Mean effective coefficient")
    axes[1].set(xlabel="Stage-II time (log scale)", ylabel="Mean CE (log scale)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    save(fig, prefix + "_scale_growth_diagnostics", [run])
    (P / (prefix + "_scale_summary.json")).write_text(
        json.dumps(dict(run=run, variants=table), indent=2)
    )


def comparisons():
    results = {}
    for base in ["f21", "f21_scale", "f21_wide", "z15", "z30", "s4", "phase", "competition"]:
        r1 = RUNS / base
        r2 = RUNS / (base + "_h2")
        if not (r1 / "final.npz").exists() or not (r2 / "final.npz").exists():
            continue
        a = np.load(r1 / "final.npz")
        b = np.load(r2 / "final.npz")
        h1 = json.loads((r1 / "history.json").read_text())
        h2 = json.loads((r2 / "history.json").read_text())
        dr = np.sqrt(
            (np.sum((a["W_in"] - b["W_in"]) ** 2) + np.sum((a["W_out"] - b["W_out"]) ** 2))
            / (a["W_in"].size + a["W_out"].size)
        )
        results[base] = dict(
            parameter_rms_difference=float(dr),
            max_direction_difference=float(
                max(
                    np.linalg.norm(
                        (a["W_in"] - b["W_in"]).reshape(len(a["W_in"]), -1, a["W_out"].shape[0]),
                        axis=-1,
                    ).max(),
                    np.linalg.norm(a["W_out"] - b["W_out"], axis=0).max(),
                )
            ),
            accuracy_difference=abs(h1[-1]["accuracy"] - h2[-1]["accuracy"]),
            mean_ce_difference=abs(h1[-1]["loss_mean"] - h2[-1]["loss_mean"]),
            max_representation_fraction_difference=float(
                np.max(
                    abs(
                        np.bincount(a["winner"], minlength=a["grouped_energy"].shape[-1])
                        - np.bincount(b["winner"], minlength=b["grouped_energy"].shape[-1])
                    )
                )
                / len(a["winner"])
            ),
            fraction_neurons_switching=float(np.mean(a["winner"] != b["winner"])),
        )
        ia = np.load(r1 / "initial.npz")
        ib = np.load(r2 / "initial.npz")
        results[base]["equal_physical_horizon"] = h1[-1]["time"] == h2[-1]["time"]
        results[base]["initial_max_difference"] = float(
            max(abs(ia["W_in"] - ib["W_in"]).max(), abs(ia["W_out"] - ib["W_out"]).max())
        )
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for hh, label in [(h1, "dt"), (h2, "dt/2")]:
            tt = [q["time"] for q in hh]
            axes[0].plot(tt, [q["accuracy"] for q in hh], label=label)
            axes[1].plot(tt, [q["signed_mean"][0] for q in hh], label=label)
        for ax in axes:
            ax.set_xlabel("Time $t$")
            ax.legend(frameon=False)
        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("Positive alignment")
        save(fig, base + "_step_check", [base, base + "_h2"])
    for base in ["f21_scale", "f21_scale_wide"]:
        stage_results = {}
        for variant in ["tied", "untied_matched", "untied_section2"]:
            p1 = RUNS / base / (variant + ".npz")
            p2 = RUNS / (base + "_h2") / (variant + ".npz")
            if not p1.exists() or not p2.exists():
                continue
            a = np.load(p1)
            b = np.load(p2)
            stage_results[variant] = dict(
                final_a_max_absolute_difference=float(abs(a["a"][-1] - b["a"][-1]).max()),
                final_a_relative_rms_difference=float(
                    np.linalg.norm(a["a"][-1] - b["a"][-1]) / np.linalg.norm(b["a"][-1])
                ),
                final_mean_ce_difference=float(abs(a["loss_sum"][-1] - b["loss_sum"][-1]) / 441),
                final_accuracy_difference=float(abs(a["accuracy"][-1] - b["accuracy"][-1])),
                equal_physical_horizon=bool(a["time"][-1] == b["time"][-1]),
                same_frozen_directions=base == "f21_scale_wide",
            )
        if stage_results:
            results[base + "_stage2"] = stage_results
    (P / "step_size_comparison.json").write_text(json.dumps(results, indent=2))
    for base in ["f21", "z15"]:
        if not (RUNS / (base + "_legacy") / "complete.json").exists():
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for name, label in [
            (base, "Separate spheres"),
            (base + "_legacy", "Original joint sphere"),
        ]:
            cfg, h, d, ck = load(name)
            t = [q["time"] for q in h]
            axes[0].plot(t, [q["accuracy"] for q in h], label=label)
            axes[1].plot(t, [q["signed_mean"][0] for q in h], label=label)
            axes[2].plot(t, [q["max_norm_error"] for q in h], label=label)
        for ax, title in zip(
            axes, ["Accuracy", "Positive alignment", "Individual sphere norm error"]
        ):
            ax.set(xlabel="Time $t$", ylabel=title)
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, base + "_before_after", [base, base + "_legacy"])


def main():
    missing = []
    for run, name in [
        ("f21_wide_h2", "frobenius21_dft_heatmap"),
        ("z15", "modular_addition_dft_heatmap_example"),
        ("z30", "modular_addition_dft_heatmap"),
        ("s4", None),
    ]:
        if (RUNS / run / "complete.json").exists():
            plot_group(run, name)
        else:
            missing.append(run)
    for run in ["phase", "competition"]:
        if (RUNS / run / "complete.json").exists():
            specialized(run)
        else:
            missing.append(run)
    if (RUNS / "f21" / "complete.json").exists():
        plot_group("f21", "frobenius21_reference512_dft_heatmap", "frobenius21_reference512")
    if (RUNS / "f21_scale_wide_h2" / "untied_section2.npz").exists():
        scale_plots()
    else:
        missing.append("f21_scale_wide_h2 Stage II")
    if (RUNS / "f21_scale" / "untied_section2.npz").exists():
        scale_plots("f21_scale", "frobenius21_reference512")
    comparisons()
    (P / "figure_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(generated=len(provenance), missing=missing)))


if __name__ == "__main__":
    main()
