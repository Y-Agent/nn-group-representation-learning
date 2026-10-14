"""Regression coverage for the October 2026 comprehensive review."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.checkpointing import require_run_contract
from src.config import Config
from src.flow import direction_step, exact_chunked_table_step
from src.model import WideNetworkScaleSphere
from src.spectral import Spectrum
from src.trainer import SphericalTrainer


def config(**kwargs):
    return Config(
        n=[3],
        width=8,
        stage1_num_epochs=1,
        stage2_num_epochs=1,
        track_phases=False,
        device="cpu",
        **kwargs,
    )


def test_config_does_not_mutate_source():
    source = {"width": 7}
    assert Config.from_dict(source, width=9).width == 9
    assert source == {"width": 7}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"print_every": 0},
        {"width": 0},
        {"stage1_lr": float("nan")},
        {"stage2_num_epochs": -1},
        {"phase_save_every": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_zero_epoch_and_stale_stage2_load(tmp_path):
    zero = SphericalTrainer(
        Config(n=[3], width=8, stage1_num_epochs=0, stage2_enabled=False, track_phases=False)
    )
    before = zero.model.get_state()
    zero.train_stage1()
    assert all(torch.equal(v, zero.model.get_state()[k]) for k, v in before.items())
    path = tmp_path / "stage1.pt"
    zero.save_checkpoint(str(path))
    trained = SphericalTrainer(config())
    trained.train_stage2()
    assert hasattr(trained, "scale_model")
    trained.load_checkpoint(str(path))
    assert not hasattr(trained, "scale_model")
    assert torch.equal(trained.predict(trained.data), zero.model(zero.data))


def test_load_rejects_different_task_before_mutation(tmp_path):
    trainer = SphericalTrainer(config())
    path = tmp_path / "model.pt"
    trainer.save_checkpoint(str(path))
    other = SphericalTrainer(Config(n=[5], width=8, track_phases=False))
    before = other.model.get_state()
    with pytest.raises(ValueError, match="configuration mismatch"):
        other.load_checkpoint(str(path))
    assert torch.equal(before["W_in"], other.model.W_in)


def test_checkpoint_restores_histories(tmp_path):
    trainer = SphericalTrainer(config())
    trainer.train()
    path = tmp_path / "model.pt"
    trainer.save_checkpoint(str(path))
    other = SphericalTrainer(config())
    other.load_checkpoint(str(path))
    assert other.stage1_history.losses == trainer.stage1_history.losses
    assert other.stage2_history.losses == trainer.stage2_history.losses
    assert np.array_equal(other.stage2_history.scale_history, trainer.stage2_history.scale_history)


def test_load_state_preserves_parameter_identity():
    m = WideNetworkScaleSphere(3, 8)
    parameter = m.W_in
    m.load_state(m.get_state())
    assert m.W_in is parameter
    bad = m.get_state()
    bad["W_out"] = bad["W_out"][:2]
    with pytest.raises(ValueError, match="shape mismatch"):
        m.load_state(bad)


def test_full_table_rejects_shuffled_inputs():
    model = WideNetworkScaleSphere(3, 8).double()
    model.normalize_directions()
    x = torch.cartesian_prod(torch.arange(3), torch.arange(3))
    y = x.sum(1) % 3
    with pytest.raises(ValueError, match="lexicographic"):
        direction_step(model, x.flip(0), y.flip(0), 0.1, full_table=True)
    with pytest.raises(ValueError):
        exact_chunked_table_step(model, y, 0.1, chunk_size=0)


def test_unknown_spectrum_group():
    with pytest.raises(ValueError):
        Spectrum("typo")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_fourier_roundtrip_precision(dtype):
    from src.basis import FourierBasis

    basis = FourierBasis([3, 5])
    x = torch.randn(4, 15, dtype=dtype)
    c = basis.transform(x)
    back = basis.inverse_transform(c)
    tol = 1e-12 if dtype in (torch.float64, torch.complex128) else 1e-6
    assert torch.allclose(back, x.to(back.dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "name,n",
    [
        ("symmetric", 2),
        ("symmetric", 3),
        ("symmetric", 4),
        ("alternating", 3),
        ("alternating", 4),
        ("alternating", 5),
    ],
)
def test_group_basis_orthogonality_and_product(name, n):
    from src.basis import get_basis
    from src.tasks import get_task

    basis = get_basis(name, n=n)
    task = get_task(name + "_group", n=n)
    B = basis.get_basis_matrix().numpy()
    N = task.vocab_size
    assert np.allclose(B @ B.conj().T, np.eye(N), atol=1e-10)
    _, labels = task.generate_all_data(torch.device("cpu"))
    table = labels.numpy().reshape(N, N)
    dims = (
        list(basis.partition_dimensions.values())
        if name == "symmetric"
        else list(basis.irrep_dimensions.values())
    )
    start = 0
    for d in dims:
        rho = (B[start : start + d * d].T * np.sqrt(N / d)).reshape(N, d, d).transpose(0, 2, 1)
        start += d * d
        assert np.allclose(rho[:, None] @ rho[None, :], rho[table], atol=1e-9)


def test_stage2_rejects_reused_output_with_other_settings(tmp_path):
    from scripts.reproduce import stage2

    m = WideNetworkScaleSphere(3, 8).double()
    m.normalize_directions()
    x = torch.cartesian_prod(torch.arange(3), torch.arange(3))
    y = x.sum(1) % 3
    cfg = {"stage2_dt": 0.1, "stage2_steps": 3}
    stage2(m, x, y, cfg, tmp_path, 1)
    initial = (tmp_path / "tied.npz").read_bytes()
    stage2(m, x, y, cfg, tmp_path, 1)
    assert initial == (tmp_path / "tied.npz").read_bytes()
    with pytest.raises(ValueError, match="contract mismatch"):
        stage2(m, x, y, {**cfg, "stage2_dt": 0.2}, tmp_path, 1)
    assert initial == (tmp_path / "tied.npz").read_bytes()


def test_contract_refuses_untracked_results(tmp_path):
    artifact = tmp_path / "old.npz"
    artifact.write_bytes(b"old")
    with pytest.raises(ValueError, match="lack a run contract"):
        require_run_contract(tmp_path / "contract.json", {"seed": 1}, [artifact])


def test_stage2_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import src.scale_training as runner

    m = WideNetworkScaleSphere(3, 8).double()
    m.normalize_directions()
    x = torch.cartesian_prod(torch.arange(3), torch.arange(3))
    y = x.sum(1) % 3
    cfg = {"stage2_dt": 0.1, "stage2_steps": 5}
    runner.stage2(m, x, y, cfg, tmp_path / "whole", 1)
    original = runner.save_atomic

    def interrupt(obj, path):
        original(obj, path)
        if obj["step"] == 3:
            raise InterruptedError("simulated interruption")

    monkeypatch.setattr(runner, "save_atomic", interrupt)
    with pytest.raises(InterruptedError):
        runner.stage2(m, x, y, cfg, tmp_path / "resume", 1)
    monkeypatch.setattr(runner, "save_atomic", original)
    runner.stage2(m, x, y, cfg, tmp_path / "resume", 1)
    for variant in ["tied", "untied_matched", "untied_section2"]:
        with (
            np.load(tmp_path / "whole" / f"{variant}.npz") as a,
            np.load(tmp_path / "resume" / f"{variant}.npz") as b,
        ):
            for key in a.files:
                assert np.array_equal(a[key], b[key]), key


def test_cli_accepts_seed_zero_and_zero_epochs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/train.py"
    path = tmp_path / "zero.pt"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--p",
            "3",
            "--width",
            "8",
            "--seed",
            "0",
            "--stage1-epochs",
            "0",
            "--no-stage2",
            "--save-path",
            str(path),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
    )
    saved = torch.load(path, weights_only=False)
    assert saved["config"]["seed"] == 0
    assert saved["config"]["stage1_num_epochs"] == 0


def test_stage2_module_has_no_import_side_effects():
    module = importlib.import_module("scripts.stage2_from_checkpoint")
    assert callable(module.main)


@pytest.mark.parametrize(
    "name,kwargs", [("frobenius21", {}), ("symmetric", {"n": 3}), ("alternating", {"n": 4})]
)
def test_nonabelian_complex64_input(name, kwargs):
    from src.basis import get_basis

    basis = get_basis(name, **kwargs)
    x = torch.randn(2, basis.size, dtype=torch.complex64)
    recovered = basis.inverse_transform(basis.transform(x))
    assert torch.allclose(recovered, x.to(recovered.dtype), atol=1e-12, rtol=1e-12)


def test_stage1_resume_after_final_checkpoint_has_unique_times(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/reproduce.py"),
        "--experiment",
        "phase",
        "--limit-steps",
        "4",
        "--threads",
        "1",
        "--output",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    run = tmp_path / "phase_pilot"
    with np.load(run / "final.npz") as data:
        before = data["W_in"].copy()
    (run / "complete.json").unlink()  # emulate interruption between checkpoint and marker
    subprocess.run(command, check=True, capture_output=True)
    records = json.loads((run / "history.json").read_text())
    assert len({r["time"] for r in records}) == len(records) == 5
    with np.load(run / "final.npz") as data:
        assert np.array_equal(before, data["W_in"])
