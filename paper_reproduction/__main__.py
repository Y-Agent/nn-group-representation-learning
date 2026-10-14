"""CLI: python -m paper_reproduction {render,verify,train,package}."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from . import settings as S


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline():
    """Check the independent reference before comparing or packaging outputs."""
    contract = json.loads((S.ROOT / "reproduction/paper-contract.json").read_text())
    errors = []
    for relative, digest in contract["baseline_files"].items():
        path = S.ROOT / "reproduction/accepted" / relative
        if not path.is_file() or sha(path) != digest:
            errors.append(f"Accepted baseline is missing or changed: {relative}")
    if errors:
        raise ValueError("\n".join(errors))
    return contract


def verify_inputs(runs, reference=True):
    contract = verify_baseline()
    errors = []
    for relative, digest in contract["inputs"].items():
        path = runs / relative
        if not path.exists():
            errors.append(f"Missing {path}")
        elif reference and sha(path) != digest:
            errors.append(f"Changed reference input: {relative}")
    if errors:
        raise ValueError("\n".join(errors))
    return contract


def environment():
    from matplotlib.font_manager import FontProperties, findfont

    fonts = {}
    for name in ["Palatino", "P052", "DejaVu Serif"]:
        try:
            path = Path(findfont(FontProperties(family=[name]), fallback_to_default=False))
            fonts[name] = {"path": str(path), "sha256": sha(path)}
        except ValueError:
            pass
    return dict(
        python=sys.version,
        platform=platform.platform(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "matplotlib", "pypdf", "Pillow", "PyYAML"]
        },
        fonts=fonts,
    )


def render(args):
    contract = verify_inputs(args.runs, not args.new_results)
    output = args.output.resolve()
    # A render never writes into the original results, notebook sources, or baseline.
    for protected in [
        args.runs.resolve(),
        S.SOURCES.resolve(),
        (S.ROOT / "reproduction/accepted").resolve(),
    ]:
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"Unsafe output directory overlaps inputs: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    notebook_out = output / "notebook-output"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")
    env["MPLBACKEND"] = "Agg"
    with (output / "render.log").open("w") as log:
        subprocess.run(
            [
                str(args.notebook_python.absolute()),
                "-m",
                "paper_reproduction.notebook_adapter",
                "--runs",
                str(args.runs.resolve()),
                "--output",
                str(notebook_out),
                "--sources",
                str(S.SOURCES),
            ],
            cwd=S.ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    import matplotlib

    matplotlib.use("Agg")
    from .dynamics import render as render_dynamics

    figures = output / "figures"
    figures.mkdir()
    for asset in S.ASSETS.values():
        if asset.endswith(".jpg"):
            shutil.copy2(notebook_out / "figures" / asset, figures / asset)
    # The manuscript historically references this JPEG suffix.
    shutil.copy2(
        figures / "modular_addition_xi_heatmap_example.jpg",
        figures / "modular_addition_xi_heatmap_example.jpeg",
    )
    from .distribution import render as render_distributions

    distribution_checks = render_distributions(
        notebook_out / "distribution-data.json", S.SOURCES, figures
    )
    checks = render_dynamics(args.runs, output, notebook_out / "frobenius_diagnostic_plot_data.npz")
    checks["distributions"] = distribution_checks
    from pypdf import PdfReader, PdfWriter

    booklet = PdfWriter()
    html = [
        '<!doctype html><meta charset="utf-8"><title>Accepted paper figures</title>',
        "<style>body{max-width:1200px;margin:2em auto;font-family:serif}img,object{width:100%}object{height:680px}</style>",
        "<h1>Regenerated affected paper figures</h1>",
    ]
    for number, asset in S.ASSETS.items():
        pdf_name = Path(asset).with_suffix(".pdf").name
        pdf = figures / pdf_name
        if not pdf.exists():
            pdf = notebook_out / "figures" / pdf_name
        booklet.add_page(PdfReader(pdf).pages[0])
        html.append(f"<h2>Figure {number}</h2>")
        html.append(
            f'<object data="figures/{asset}" type="application/pdf"></object>'
            if asset.endswith(".pdf")
            else f'<img src="figures/{asset}" alt="Figure {number}">'
        )
    booklet.write(output / "affected-paper-figures.pdf")
    (output / "gallery.html").write_text("\n".join(html))
    manifest = dict(
        schema_version=1,
        reference_inputs=not args.new_results,
        assets=S.ASSETS,
        checks=checks,
        environment=environment(),
        notebook_environment=json.loads((notebook_out / "manifest.json").read_text())[
            "environment"
        ],
        inputs={p: sha(args.runs / p) for p in contract["inputs"]},
        sources={
            str(p.relative_to(S.ROOT)): sha(p)
            for p in sorted((S.ROOT / "paper_reproduction").glob("*.py"))
        },
        outputs={str(p.relative_to(output)): sha(p) for p in sorted(figures.iterdir())},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Rendered {len(S.ASSETS)} paper assets to {output}")


def verify(args):
    """Compare statistics within tolerance and raster pixels exactly."""
    import numpy as np
    from PIL import Image

    verify_baseline()
    out = args.output
    baseline = S.ROOT / "reproduction/accepted"
    report = {}
    manifest = json.loads((out / "manifest.json").read_text())
    for relative, digest in manifest["outputs"].items():
        report[f"hash:{relative}"] = {"exact": sha(out / relative) == digest}
    with (
        np.load(out / "notebook-output/frobenius_diagnostic_plot_data.npz") as new,
        np.load(baseline / "diagnostics.npz") as old,
    ):
        for key in old.files:
            report[f"data:{key}"] = dict(
                exact=bool(np.array_equal(new[key], old[key], equal_nan=True)),
                passed=bool(np.allclose(new[key], old[key], rtol=0, atol=1e-12, equal_nan=True)),
                max_abs_error=float(np.nanmax(np.abs(new[key] - old[key]))),
            )
    for name in ["alignment", "accuracy", "scale-loss", "per-neuron-scale"]:
        a = np.asarray(Image.open(out / "panels" / f"{name}.png").convert("RGB"))
        b = np.asarray(Image.open(baseline / "panels" / f"{name}.png").convert("RGB"))
        report[f"panel:{name}"] = {"exact": bool(np.array_equal(a, b)), "shape": list(a.shape)}
    # Compare decoded pixels; JPEG metadata is not scientific content.
    for asset in S.ASSETS.values():
        if asset.endswith(".jpg"):
            a = np.asarray(Image.open(out / "figures" / asset).convert("RGB"))
            b = np.asarray(Image.open(baseline / "figures" / asset).convert("RGB"))
            report[f"asset:{asset}"] = {
                "exact": bool(np.array_equal(a, b)),
                "shape_matches": a.shape == b.shape,
                "mean_abs_pixel_error": float(np.abs(a.astype(float) - b).mean())
                if a.shape == b.shape
                else None,
            }
    failed = [k for k, v in report.items() if not v.get("passed", v["exact"])]
    (out / "baseline-verification.json").write_text(
        json.dumps({"checks": report, "differences": failed}, indent=2)
    )
    print(json.dumps({"passed_checks": len(report) - len(failed), "differences": failed}, indent=2))
    if failed:
        raise SystemExit(1)


def training_commands(runs, threads, refinements=True):
    commands = []
    for name in ["z15", "z30", "s4", "f21_wide"]:
        for divisor in [1, 2] if refinements else [2]:
            commands.append(
                [
                    sys.executable,
                    str(S.ROOT / "scripts/reproduce.py"),
                    "--experiment",
                    name,
                    "--output",
                    str(runs),
                    "--step-divisor",
                    str(divisor),
                    "--threads",
                    str(threads),
                ]
            )
    for divisor in [1, 2] if refinements else [2]:
        commands.append(
            [
                sys.executable,
                str(S.ROOT / "scripts/stage2_from_checkpoint.py"),
                "--runs",
                str(runs),
                "--step-divisor",
                str(divisor),
                "--threads",
                str(threads),
            ]
        )
    return commands


def train(args):
    import shlex

    commands = training_commands(args.runs.resolve(), args.threads, not args.main_only)
    if not args.execute:
        print("\n".join(shlex.join(c) for c in commands))
        return
    if args.runs.resolve() == S.RUNS.resolve():
        raise ValueError("Use a fresh --runs directory; archived paper runs are read-only inputs.")
    for command in commands:
        subprocess.run(command, cwd=S.ROOT, check=True)


def package(args):
    contract = verify_inputs(args.runs)
    if args.destination.exists():
        raise ValueError("Refusing to overwrite an existing archive")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.destination, "w:gz") as archive:
        for relative in [
            "src",
            "paper_reproduction",
            "scripts",
            "tests",
            "configs",
            "docs",
            "README.md",
            "REPRODUCTION.md",
            "requirements.txt",
            "requirements-paper.txt",
            "requirements-notebook.txt",
            "requirements-dev.txt",
            "pyproject.toml",
            "reproduction/configs.json",
            "reproduction/paper-contract.json",
            "reproduction/requirements-lock-cpu.txt",
            "reproduction/paper_plots/sources",
            "reproduction/accepted",
        ]:
            for p in (
                [S.ROOT / relative]
                if (S.ROOT / relative).is_file()
                else sorted((S.ROOT / relative).rglob("*"))
            ):
                if p.is_file() and "__pycache__" not in p.parts:
                    archive.add(p, arcname=str(p.relative_to(S.ROOT)))
        for relative in contract["inputs"]:
            archive.add(args.runs / relative, arcname="reproduction/runs/" + relative)
    print(f"Portable code and reference-input archive: {args.destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render")
    p.add_argument("--runs", type=Path, default=S.RUNS)
    p.add_argument(
        "--notebook-python",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter with the recorded Matplotlib 3.10.9 notebook-rendering environment",
    )
    p.add_argument("--output", type=Path, default=S.OUTPUT)
    p.add_argument(
        "--new-results",
        action="store_true",
        help="Allow results from a new training run; record their hashes instead of requiring the archived hashes.",
    )
    p = sub.add_parser("verify")
    p.add_argument("--output", type=Path, default=S.OUTPUT)
    p = sub.add_parser("train")
    p.add_argument("--runs", type=Path, default=S.ROOT / "reproduction/new-runs")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--main-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("package")
    p.add_argument("--runs", type=Path, default=S.RUNS)
    p.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    # Use a writable local cache without changing the user's home directory.
    os.environ.setdefault("MPLCONFIGDIR", str(S.ROOT / ".cache/matplotlib"))
    globals()[args.command](args)


if __name__ == "__main__":
    main()
