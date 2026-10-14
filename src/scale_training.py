"""Resumable full-table SGD on frozen-direction effective scales."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpointing import model_fingerprint, require_run_contract, save_atomic
from .flow import FrozenScaleModel
from .utils import cross_entropy_high_precision


def stage2(m, x, y, cfg, out, divisor):
    if divisor < 1 or cfg["stage2_steps"] < 0 or cfg["stage2_dt"] <= 0:
        raise ValueError("Invalid Stage-II step settings")
    dt = cfg["stage2_dt"] / divisor
    steps = cfg["stage2_steps"] * divisor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(
        schema_version=1,
        source_model=model_fingerprint(m),
        dt=dt,
        steps=steps,
        variants={"tied": 1, "untied_matched": m.width, "untied_section2": 1},
    )
    require_run_contract(
        out / "stage2-contract.json",
        contract,
        [out / (v + suffix) for v in contract["variants"] for suffix in [".npz", "_latest.pt"]],
    )
    for variant in ["tied", "untied_matched", "untied_section2"]:
        target = out / (variant + ".npz")
        if target.exists():
            continue
        sm = FrozenScaleModel(m, variant == "tied")
        h = sm.features(x).detach()
        tied = variant == "tied"
        base = h @ sm.W_out.T / m.width if tied else None
        lr = dt * (m.width if variant == "untied_matched" else 1)
        with torch.no_grad():
            jump = (sm(x) - m(x)).abs().max().item()
        assert jump < (1e-12 if m.W_in.dtype == torch.float64 else 1e-6)
        losses = []
        accs = []
        scales = []
        times = []
        amin = []
        start = time.time()
        checkpoint = out / (variant + "_latest.pt")
        start_step = 0
        if checkpoint.exists():
            old = torch.load(checkpoint, weights_only=False)
            sm.a.data.copy_(old["a"])
            start_step = old["step"]
            losses, accs, scales, times, amin = old["records"]
        every = max(1, steps // 400)
        for i in range(start_step, steps + 1):
            z = base * sm.a if tied else (h * sm.a) @ sm.W_out.T / m.width
            loss = cross_entropy_high_precision(z, y)
            if i % every == 0 or i == steps:
                losses.append(loss.item())
                accs.append((z.argmax(-1) == y).double().mean().item())
                scales.append(sm.a.detach().cpu().numpy().copy())
                times.append(i * dt)
                amin.append(sm.a.min().item())
            if i == steps:
                break
            loss.backward()
            with torch.no_grad():
                sm.a.add_(sm.a.grad, alpha=-lr)
            sm.a.grad = None
            if (i + 1) % max(1, steps // 20) == 0:
                save_atomic(
                    dict(
                        a=sm.a.detach().cpu(),
                        step=i + 1,
                        records=(losses, accs, scales, times, amin),
                    ),
                    checkpoint,
                )
                print(
                    json.dumps(
                        dict(
                            stage2=variant,
                            step=i + 1,
                            loss=loss.item(),
                            elapsed=time.time() - start,
                        )
                    ),
                    flush=True,
                )
        np.savez_compressed(
            target,
            time=times,
            loss_sum=losses,
            accuracy=accs,
            a=np.array(scales),
            a_min=amin,
            boundary_max_logit_jump=jump,
            lr_multiplier=lr / dt,
        )
