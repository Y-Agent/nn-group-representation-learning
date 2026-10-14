"""Run exact full-table CE flows. Checkpoints permit interruption and resumption.

python scripts/reproduce.py --experiment f21 --output reproduction/runs --threads 4
Use --step-divisor 2 (or 4) for equal-horizon numerical convergence checks.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.checkpointing import save_atomic
from src.experiment import evaluate, init_special
from src.flow import direction_step, exact_chunked_table_step
from src.model import WideNetworkScaleSphere
from src.scale_training import stage2
from src.spectral import Spectrum
from src.utils import cross_entropy_high_precision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", default="reproduction/runs")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--step-divisor", type=int, default=1)
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--limit-steps", type=int)
    p.add_argument(
        "--extend-steps",
        type=int,
        help="Increase base-step horizon; refinement multiplier applied. Preserve reference output.",
    )
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    a = p.parse_args()
    if a.step_divisor < 1 or a.threads < 1:
        p.error("--step-divisor and --threads must be positive")
    if a.limit_steps is not None and a.limit_steps < 0:
        p.error("--limit-steps must be nonnegative")
    if a.extend_steps is not None and a.extend_steps <= 0:
        p.error("--extend-steps must be positive")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "reproduction/configs.json").read_text()
    )[a.experiment]
    cfg = dict(
        cfg,
        step_divisor=a.step_divisor,
        legacy=a.legacy,
        dtype=a.dtype,
        device=a.device,
        threads=a.threads,
    )
    cfg["dt"] /= a.step_divisor
    cfg["steps"] = (a.extend_steps or cfg["steps"]) * a.step_divisor
    if a.limit_steps is not None:
        cfg["steps"] = a.limit_steps
    name = (
        a.experiment
        + ("_legacy" if a.legacy else "")
        + (f"_h{a.step_divisor}" if a.step_divisor != 1 else "")
        + ("_pilot" if a.limit_steps is not None else "")
    )
    out = Path(a.output) / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    spec = Spectrum(cfg["group"])
    m = WideNetworkScaleSphere(spec.n, cfg["width"], share_embed=cfg.get("shared", False)).to(
        dtype=getattr(torch, a.dtype), device=a.device
    )
    m.normalize_directions()
    m.scale_in.requires_grad_(False)
    m.scale_out.requires_grad_(False)
    if "init" in cfg:
        init_special(m, spec, cfg)
    if a.legacy:
        # Reproduce original concatenated-sphere geometry from the same Gaussian directions.
        # Original Gaussian block lengths matter; reset using the same random stream.
        torch.manual_seed(cfg["seed"])
        w = torch.randn(m.width, m.d_input, dtype=torch.float32)
        v = torch.randn(spec.n, m.width, dtype=torch.float32)
        with torch.no_grad():
            m.W_in.copy_(w / w.norm(dim=1, keepdim=True))
            m.W_out.copy_(v / v.norm(dim=0, keepdim=True))
    x, y = spec.task.generate_all_data(torch.device(a.device))
    steps = cfg["steps"]
    dt = cfg["dt"]
    records = []
    snapshots = []
    start_step = 0
    checkpoint = out / "latest.pt"
    if (out / "complete.json").exists() and not checkpoint.exists():
        raise ValueError(
            "Completion marker has no checkpoint; refusing to reuse incomplete results"
        )
    if checkpoint.exists():
        old = torch.load(checkpoint, map_location=a.device, weights_only=False)
        if old["config"].get("threads") != cfg["threads"]:
            previous = dict(old["config"])
            old["config"]["threads"] = cfg["threads"]
            (out / f"runtime_change_at_step_{old['step']}.json").write_text(
                json.dumps(
                    dict(
                        previous_config=previous,
                        new_threads=cfg["threads"],
                        reason="CPU scheduling only; numerical settings unchanged",
                    ),
                    indent=2,
                )
            )
        if old["config"] != cfg:
            prior = dict(old["config"])
            prior["steps"] = cfg["steps"]
            # S4 has no published numerical horizon. Its explicit convergence-
            # based revision retains all data and is allowed only before completion.
            shorten = (
                a.experiment == "s4"
                and old["config"]["steps"] == 4000 * a.step_divisor
                and cfg["steps"] == 2400 * a.step_divisor
                and old["step"] <= cfg["steps"]
                and not (out / "complete.json").exists()
                and prior == cfg
            )
            if shorten:
                (out / "original_planned_config.json").write_text(
                    json.dumps(old["config"], indent=2)
                )
                (out / "horizon_revision.json").write_text(
                    json.dumps(
                        dict(
                            reason="User-authorized adaptive horizon: at 1520 coarse updates, perfect accuracy, mean concentration > 1-2e-9, and high-dimensional mean rank ratios < 4.1e-6. Use 2400 coarse / 4800 refined updates at equal physical time; the checkpoint had passed 2000 when cancellation completed.",
                            original_base_steps=4000,
                            final_base_steps=2400,
                        ),
                        indent=2,
                    )
                )
                old["config"]["steps"] = cfg["steps"]
            else:
                if not a.extend_steps or prior != cfg or cfg["steps"] <= old["config"]["steps"]:
                    raise ValueError("Resume config mismatch")
                for filename in ["complete.json", "final.npz", "history.json", "config.json"] + [
                    v + ext
                    for v in ["tied", "untied_matched", "untied_section2"]
                    for ext in [".npz", "_latest.pt"]
                ]:
                    source = out / filename
                    if source.exists():
                        source.rename(out / (f"reference_{old['config']['steps']}_" + filename))
        m.load_state_dict(old["model"])
        start_step = old["step"]
        records = old["records"]
        snapshots = old["snapshots"]
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    source_files = sorted((Path(__file__).resolve().parents[1] / "src").rglob("*.py")) + [
        Path(__file__).resolve()
    ]
    if (out / "source_hashes.json").exists():
        (out / "source_hashes.json").rename(out / f"source_hashes_before_step_{start_step}.json")
    (out / "source_hashes.json").write_text(
        json.dumps(
            {
                str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in source_files
            },
            indent=2,
        )
    )
    (out / "environment.json").write_text(
        json.dumps(
            dict(
                python=sys.version,
                torch=torch.__version__,
                numpy=np.__version__,
                platform=platform.platform(),
                cpu_count=os.cpu_count(),
            ),
            indent=2,
        )
    )
    every = max(1, steps // 100)
    start = time.time()
    if not (out / "complete.json").exists():
        for i in range(start_step, steps + 1):
            if i % every == 0 or i == steps:
                rec, d = evaluate(m, x, y, spec, i, dt)
                # A crash after the final checkpoint but before complete.json
                # must not duplicate the final observation on resume.
                if records and records[-1]["step"] == i:
                    records.pop()
                    snapshots.pop()
                records.append(rec)
                snapshots.append(
                    dict(
                        step=i,
                        **{
                            k: d[k][:, : min(1024, m.width)]
                            for k in ["signed", "rank", "concentration"]
                        },
                        theta_coeff=d["coefficients"][0, :16],
                        output_coeff=d["coefficients"][2, :16],
                        **(
                            dict(
                                W_in=m.W_in.detach().cpu().numpy().copy(),
                                W_out=m.W_out.detach().cpu().numpy().copy(),
                            )
                            if m.width <= 4096
                            else {}
                        ),
                    )
                )
                (out / "history.json").write_text(json.dumps(records, indent=2))
                print(
                    json.dumps(dict(experiment=name, **rec, elapsed=time.time() - start)),
                    flush=True,
                )
                if i == 0:
                    np.savez_compressed(
                        out / "initial.npz",
                        W_in=m.W_in.detach().cpu().numpy(),
                        W_out=m.W_out.detach().cpu().numpy(),
                        **d,
                    )
            if i == steps:
                break
            if not a.legacy:
                if m.width > 1024:
                    exact_chunked_table_step(m, y, dt, chunk_size=1024)
                else:
                    direction_step(m, x, y, dt, full_table=True)
            else:
                m.zero_grad(set_to_none=True)
                loss = cross_entropy_high_precision(m(x), y)
                loss.backward()
                with torch.no_grad():
                    for w, dim in [(m.W_in, 1), (m.W_out, 0)]:
                        g = w.grad - (w.grad * w).sum(dim=dim, keepdim=True) * w
                        w.add_(g, alpha=-dt)
                        w.div_(w.norm(dim=dim, keepdim=True))
            if (i + 1) % every == 0:
                save_atomic(
                    dict(
                        model=m.state_dict(),
                        step=i + 1,
                        config=cfg,
                        records=records,
                        snapshots=snapshots,
                    ),
                    checkpoint,
                )
        np.savez_compressed(
            out / "final.npz",
            W_in=m.W_in.detach().cpu().numpy(),
            W_out=m.W_out.detach().cpu().numpy(),
            **d,
        )
        save_atomic(
            dict(
                model=m.state_dict(), step=steps, config=cfg, records=records, snapshots=snapshots
            ),
            checkpoint,
        )
        (out / "complete.json").write_text(
            json.dumps(
                dict(
                    stage1="complete",
                    steps=steps,
                    physical_time=steps * dt,
                    elapsed=time.time() - start,
                    final=records[-1],
                ),
                indent=2,
            )
        )
    if "stage2_steps" in cfg and a.limit_steps is None and not a.legacy:
        stage2(m, x, y, cfg, out, a.step_divisor)


if __name__ == "__main__":
    main()
