"""Audit numerical evidence and completed outputs without silently passing gaps."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "reproduction"
FIGURES = {
    "1": ["modular_addition_xi_heatmap_example", "modular_addition_phase_alignment"],
    "2": ["modular_addition_phase_circles", "modular_addition_freq_distribution_histogram"],
    "3": ["cayley_Z12", "Z12_spectral_basis", "cayley_A4", "A4_spectral_basis"],
    "4": ["center-stable-manifold.001", "center-stable-manifold.002"],
    "5": ["modular_addition_freq_distribution", "symmetric_group_irrep_distribution"],
    "6": ["frobenius21_dft_heatmap"],
    "7": ["frobenius21_accuracy", "frobenius21_alignment", "frobenius21_rank1"],
    "8": [
        "frobenius21_scale_comparison_loss",
        "frobenius21_scale_tied",
        "frobenius21_scale_perneuron",
    ],
    "9": ["abelian_phase_alignment", "abelian_scale_competition"],
    "10": ["modular_addition_dft_heatmap_example"],
    "11": ["modular_addition_dft_heatmap"],
}


def main():
    checks = []
    claims = []
    inventory = []

    def check(name, passed, value, tolerance):
        checks.append(
            dict(check=name, status="PASS" if passed else "FAIL", value=value, tolerance=tolerance)
        )

    for fig, names in FIGURES.items():
        for name in names:
            formats = ["jpeg"] if fig == "4" else ["pdf", "svg", "jpg"]
            paths = [P / "figures" / f"{name}.{ext}" for ext in formats]
            inventory.append(
                dict(
                    figure=fig,
                    asset=name,
                    kind="unchanged conceptual artwork"
                    if fig == "4"
                    else "regenerated conceptual"
                    if fig == "3"
                    else "corrected experiment",
                    status="COMPLETE" if all(p.exists() for p in paths) else "MISSING",
                    files=[str(p.relative_to(ROOT)) for p in paths if p.exists()],
                    sha256={
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in paths
                        if p.exists()
                    },
                )
            )
    for run in [
        "phase",
        "competition",
        "f21_wide_h2",
        "f21_scale_wide_h2",
        "z15",
        "z30",
        "s4",
        "f21",
        "f21_scale",
    ]:
        path = P / "runs" / run
        hpath = path / "history.json"
        if not hpath.exists():
            check(run + " completed", False, "missing", "required")
            continue
        h = json.loads(hpath.read_text())
        check(
            run + " completed",
            (path / "complete.json").exists(),
            h[-1]["step"],
            "configured or documented extended horizon",
        )
        norm = max(x["max_norm_error"] for x in h)
        check(
            run + " separate unit norms", norm < 1e-12, norm, "max absolute error < 1e-12 (float64)"
        )
        size = (
            441
            if run.startswith("f21")
            else {"z15": 225, "z30": 900, "s4": 576, "phase": 225, "competition": 225}[run]
        )
        check(
            run + " loss/mean factor",
            max(abs(x["loss_sum"] / x["loss_mean"] - size) for x in h) < 1e-8,
            h[-1]["loss_sum"] / h[-1]["loss_mean"],
            "sum = |G|² × mean",
        )
        final = h[-1]
        if run not in ["phase", "competition"]:
            claims.append(
                dict(
                    claim=run + " finite-run perfect accuracy",
                    status="PASS"
                    if final["accuracy"] == 1 and final["min_margin"] > 0
                    else "NOT_ATTAINED",
                    computed=final["accuracy"],
                    minimum_margin=final["min_margin"],
                    role="reference width 512"
                    if run in ["f21", "f21_scale"]
                    else "main corrected run",
                    criterion="exactly 1 and positive minimum margin at endpoint; does not establish an infinite-width theorem",
                )
            )
        claims.append(
            dict(
                claim=run + " positive alignment (mean of each relation)",
                status="PASS" if min(final["signed_mean"]) > 0.999 else "NOT_YET_CONVERGED",
                computed=final["signed_mean"],
                criterion="> 0.999 for finite-time numerical evidence",
            )
        )
        claims.append(
            dict(
                claim=run + " dominant-orbit concentration (mean of each direction)",
                status="PASS" if min(final["concentration_mean"]) > 0.999 else "NOT_YET_CONVERGED",
                computed=final["concentration_mean"],
                criterion="> 0.999; report minima separately",
            )
        )
        if final["rank_highdim_mean"]:
            claims.append(
                dict(
                    claim=run + " rank-one compression (higher-dimensional active irreps)",
                    status="PASS"
                    if max(final["rank_highdim_mean"]) < 0.01
                    else "NOT_YET_CONVERGED",
                    computed=final["rank_highdim_mean"],
                    criterion="mean sigma2/sigma1 < 0.01; 1D irreps excluded",
                )
            )
        if run == "competition":
            a = np.load(path / "initial.npz")
            b = np.load(path / "final.npz")
            initial = a["grouped_energy"][0].copy()
            initial[:, 0] = -np.inf
            agree = float(np.mean(initial.argmax(-1) == b["winner"]))
            claims.append(
                dict(
                    claim="Scale competition: initially largest nontrivial orbit wins",
                    status="PASS" if agree == 1 else "NOT_ATTAINED",
                    computed=agree,
                    criterion="all 16 neurons",
                )
            )
    for variant in ["tied", "untied_matched", "untied_section2"]:
        path = P / "runs" / "f21_scale_wide_h2" / (variant + ".npz")
        if not path.exists():
            check("Stage II " + variant, False, "missing", "required")
            continue
        z = np.load(path)
        jump = float(z["boundary_max_logit_jump"])
        check(
            variant + " transition continuity",
            jump < 1e-12,
            jump,
            "max absolute logit difference < 1e-12",
        )
        check(
            variant + " initial effective coefficient",
            bool(np.all(z["a"][0] == 0.125)),
            [float(z["a"][0].min()), float(z["a"][0].max())],
            ".125 = .5 × .5²",
        )
        check(
            variant + " positive scales observed",
            float(z["a_min"].min()) > 0,
            float(z["a_min"].min()),
            "> 0 without clipping",
        )
        check(
            variant + " decreasing CE",
            bool(np.all(np.diff(z["loss_sum"]) <= 1e-9)),
            float(z["loss_sum"][-1]),
            "all saved increments <= 1e-9",
        )
    step_path = P / "step_size_comparison.json"
    steps = json.loads(step_path.read_text()) if step_path.exists() else {}
    for run in ["phase", "competition", "f21_wide", "z15", "z30", "s4"]:
        v = steps.get(run)
        check(
            run + " step halving available", v is not None, v, "same seed, equal physical horizon"
        )
        if v:
            check(
                run + " equal-horizon same-initialization comparison",
                v["equal_physical_horizon"] and v["initial_max_difference"] < 1e-14,
                [v["equal_physical_horizon"], v["initial_max_difference"]],
                "same physical horizon; max initial difference < 1e-14",
            )
            check(
                run + " step-halving accuracy stability",
                v["accuracy_difference"] == 0,
                v["accuracy_difference"],
                "identical endpoint full-table accuracy",
            )
            check(
                run + " step-halving representation stability",
                v["max_representation_fraction_difference"] < 0.005,
                v["max_representation_fraction_difference"],
                "absolute orbit-fraction difference < 0.005",
            )
            check(
                run + " step-halving direction stability",
                v["parameter_rms_difference"] < 0.005,
                v["parameter_rms_difference"],
                "parameter RMS difference < 0.005",
            )
    for variant in ["tied", "untied_matched", "untied_section2"]:
        v = steps.get("f21_scale_wide_stage2", {}).get(variant)
        check(
            "Stage II " + variant + " refinement available",
            v is not None,
            v,
            "same frozen directions; equal physical horizon",
        )
        if v:
            check(
                "Stage II " + variant + " step-halving scale stability",
                v["equal_physical_horizon"] and v["final_a_relative_rms_difference"] < 0.005,
                v["final_a_relative_rms_difference"],
                "relative RMS coefficient difference < 0.005",
            )
    check(
        "All paper panels available",
        all(r["status"] == "COMPLETE" for r in inventory),
        sum(r["status"] == "COMPLETE" for r in inventory),
        f"all {len(inventory)} assets",
    )
    report = dict(checks=checks, claims=claims, inventory=inventory)
    (P / "validation.json").write_text(json.dumps(report, indent=2))
    (P / "figure_inventory.json").write_text(json.dumps(inventory, indent=2))
    print(
        json.dumps(
            dict(
                implementation_checks_pass=sum(r["status"] == "PASS" for r in checks),
                implementation_checks_fail=sum(r["status"] == "FAIL" for r in checks),
                claim_statuses={
                    status: sum(r["status"] == status for r in claims)
                    for status in ["PASS", "NOT_ATTAINED", "NOT_YET_CONVERGED"]
                },
            )
        )
    )
    return 1 if any(r["status"] == "FAIL" for r in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
