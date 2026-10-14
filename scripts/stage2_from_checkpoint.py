"""Train Stage II from a positive-margin Stage-I checkpoint, on any CPU host."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from src.checkpointing import model_fingerprint, require_run_contract
from src.model import WideNetworkScaleSphere
from src.scale_training import stage2
from src.spectral import Spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "reproduction/runs")
    parser.add_argument("--source", default="f21_wide_h2")
    parser.add_argument("--step-divisor", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.step_divisor < 1 or args.threads < 1:
        parser.error("Divisor and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source = args.runs / args.source
    cfg = json.loads((source / "config.json").read_text())
    checkpoint = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
    if checkpoint["step"] != cfg["steps"]:
        raise ValueError("Stage I is incomplete")
    spec = Spectrum(cfg["group"])
    model = WideNetworkScaleSphere(spec.n, cfg["width"], share_embed=False).double()
    model.load_state_dict(checkpoint["model"])
    model.scale_in.requires_grad_(False)
    model.scale_out.requires_grad_(False)
    x, y = spec.task.generate_all_data(torch.device("cpu"))
    with torch.no_grad():
        logits = model(x)
        correct = logits.gather(1, y[:, None])
        other = logits.clone()
        other.scatter_(1, y[:, None], -torch.inf)
        margin = (correct[:, 0] - other.max(1).values).min().item()
    if margin <= 0:
        raise ValueError(f"Stage II requires positive margin; got {margin}")
    name = "f21_scale_wide" + ("" if args.step_divisor == 1 else f"_h{args.step_divisor}")
    out = args.runs / name
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(source=args.source, model=model_fingerprint(model), divisor=args.step_divisor)
    require_run_contract(out / "source-contract.json", contract, [out / "config.json"])
    newcfg = dict(
        cfg,
        stage2_dt=1.0,
        stage2_steps=40000,
        derived_from=args.source,
        stage2_step_divisor=args.step_divisor,
    )
    (out / "config.json").write_text(json.dumps(newcfg, indent=2))
    for filename in [
        "initial.npz",
        "final.npz",
        "history.json",
        "latest.pt",
        "environment.json",
        "source_hashes.json",
    ]:
        if not (out / filename).exists():
            shutil.copyfile(source / filename, out / filename)
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                stage1="reused completed trajectory", source_run=args.source, minimum_margin=margin
            ),
            indent=2,
        )
    )
    stage2(model, x, y, newcfg, out, args.step_divisor)


if __name__ == "__main__":
    main()
