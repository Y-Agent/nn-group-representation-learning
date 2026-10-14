import copy

import numpy as np
import pytest
import torch

from src.flow import FrozenScaleModel, direction_step
from src.frobenius_analysis import d_sim, positive_alignment, positive_ray_residual
from src.model import WideNetworkScaleSphere
from src.utils import cross_entropy_high_precision


def model(shared=False):
    torch.manual_seed(42)
    m = WideNetworkScaleSphere(5, 11, share_embed=shared).double()
    m.normalize_directions()
    m.scale_in.requires_grad_(False)
    m.scale_out.requires_grad_(False)
    return m


def data():
    x = torch.cartesian_prod(torch.arange(5), torch.arange(5))
    return x, x.sum(1) % 5


@pytest.mark.parametrize("shared", [True, False])
def test_geometry_and_tangency(shared):
    m = model(shared)
    x, y = data()
    assert torch.allclose(m.input_blocks().norm(dim=-1), torch.ones_like(m.input_blocks()[..., 0]))
    if not shared:
        assert not torch.equal(m.input_blocks()[:, 0], m.input_blocks()[:, 1])
    cross_entropy_high_precision(m(x), y).backward()
    m.project_direction_gradients()
    assert (m.input_blocks() * m.W_in.grad.reshape_as(m.input_blocks())).sum(-1).abs().max() < 1e-14
    assert (m.W_out * m.W_out.grad).sum(0).abs().max() < 1e-14
    for _ in range(10):
        direction_step(m, x, y, 0.2)
    assert (m.input_blocks().norm(dim=-1) - 1).abs().max() < 1e-14
    assert (m.W_out.norm(dim=0) - 1).abs().max() < 1e-14
    assert m.effective_scale().item() == 0.125


@pytest.mark.parametrize("shared", [True, False])
def test_clock_and_loss(shared):
    m = model(shared)
    x, y = data()
    ref = copy.deepcopy(m)
    z = m(x)
    loss = cross_entropy_high_precision(z, y)
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(z, y, reduction="mean") * len(y))
    theta = m.input_blocks()
    h = theta[:, 0][:, x[:, 0]].T + theta[:, -1][:, x[:, 1]].T
    manual = (
        0.125 * sum(h[:, i : i + 1] ** 2 * m.W_out[:, i][None, :] for i in range(m.width)) / m.width
    )
    assert torch.allclose(z, manual, atol=1e-15)
    loss.backward()
    m.project_direction_gradients()
    dt = 0.3
    with torch.no_grad():
        m.W_in.add_(m.W_in.grad, alpha=-dt * (0.5 if shared else 1))
        m.W_out.add_(m.W_out.grad, alpha=-dt)
    m.normalize_directions()
    direction_step(ref, x, y, dt)
    assert torch.equal(m.W_in, ref.W_in) and torch.equal(m.W_out, ref.W_out)


@pytest.mark.parametrize("tied", [True, False])
def test_stage_boundary(tied):
    m = model()
    x, y = data()
    z = m(x).detach()
    s = FrozenScaleModel(m, tied)
    assert torch.allclose(z, s(x), atol=1e-15)
    before = s.W_in.clone()
    cross_entropy_high_precision(s(x), y).backward()
    assert s.W_in.grad is None and s.W_out.grad is None
    assert torch.equal(before, s.W_in)
    st = FrozenScaleModel(m, True)
    cross_entropy_high_precision(st(x), y).backward()
    assert torch.allclose(s.a.grad.sum(), st.a.grad.sum())


def test_signed_alignment():
    A = np.array([[1 + 2j, 3], [4j, 2]])
    for phase, value in [(1, 1), (-1, -1), (1j, 0)]:
        assert d_sim(A, phase * A) == pytest.approx(1)
        assert positive_alignment(A, phase * A) == pytest.approx(value)
    assert positive_ray_residual(A, 2 * A) == pytest.approx(0)
    assert positive_ray_residual(A, -A) == pytest.approx(1)
    assert np.isnan(positive_alignment(A, np.zeros_like(A)))


def test_step_halving():
    x, y = data()
    results = []
    for dt in [0.4, 0.2, 0.1]:
        m = model()
        for _ in range(round(4 / dt)):
            direction_step(m, x, y, dt)
        results.append(torch.cat([m.W_in.flatten(), m.W_out.flatten()]).detach())
    e1 = (results[0] - results[1]).norm()
    e2 = (results[1] - results[2]).norm()
    assert e2 < 0.6 * e1


@pytest.mark.parametrize("group", ["f21", "s4", "z15", "z30"])
def test_fourier_parseval_and_group_order(group):
    from src.spectral import Spectrum

    s = Spectrum(group)
    x, y = s.task.generate_all_data(torch.device("cpu"))
    # Finite group rows/columns are permutations (Latin-square composition table).
    table = y.reshape(s.n, s.n)
    assert torch.equal(table.sort(0).values, torch.arange(s.n)[:, None].expand_as(table))
    assert torch.equal(table.sort(1).values, torch.arange(s.n)[None, :].expand_as(table))
    rng = np.random.default_rng(4)
    w = rng.normal(size=(3, s.n))
    blocks = s.coefficients(w)
    assert np.allclose(
        sum(d * np.sum(abs(c) ** 2, axis=(1, 2)) for d, c in zip(s.dims, blocks)),
        np.sum(w * w, axis=1) / s.n,
        atol=1e-10,
    )
    # Reconstruct each representation matrix from the basis, checking multiplication order.
    for sl, d in zip(s.slices, s.dims):
        rho = (s.B[sl].T * np.sqrt(s.n / d)).reshape(s.n, d, d).transpose(0, 2, 1)
        for g in range(s.n):
            for h in range(s.n):
                assert np.allclose(rho[g] @ rho[h], rho[table[g, h]], atol=1e-8)


@pytest.mark.parametrize("shared", [True, False])
def test_trainer_two_stage_and_checkpoint(tmp_path, shared):
    from src.config import Config
    from src.trainer import SphericalTrainer

    cfg = Config(
        n=[3],
        width=8,
        share_embed=shared,
        stage1_num_epochs=2,
        stage2_num_epochs=2,
        stage1_lr=0.1,
        stage2_lr=0.1,
        track_phases=False,
    )
    t = SphericalTrainer(cfg)
    t.train_stage1()
    before = t.model.W_in.clone()
    t.train_stage2()
    assert torch.equal(before, t.model.W_in)
    assert t.stage2_history.losses[-1] <= t.stage2_history.losses[0]
    path = tmp_path / "model.pt"
    t.save_checkpoint(str(path))
    u = SphericalTrainer(cfg)
    u.load_checkpoint(str(path))
    assert torch.equal(t.predict(t.data), u.predict(u.data))


@pytest.mark.parametrize("shared", [True, False])
def test_optimized_full_table_logits_and_gradients(shared):
    from src.flow import full_table_logits

    m = model(shared)
    ref = copy.deepcopy(m)
    x, y = data()
    assert torch.allclose(m(x), full_table_logits(m), atol=1e-15)
    cross_entropy_high_precision(m(x), y).backward()
    cross_entropy_high_precision(full_table_logits(ref), y).backward()
    assert torch.allclose(m.W_in.grad, ref.W_in.grad, atol=1e-14)
    assert torch.allclose(m.W_out.grad, ref.W_out.grad, atol=1e-14)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("chunk", [1, 4, 100])
def test_chunked_exact_gradient_update(shared, chunk):
    from src.flow import exact_chunked_table_step

    m = model(shared)
    ref = copy.deepcopy(m)
    x, y = data()
    for _ in range(3):
        loss1 = direction_step(m, x, y, 0.3)
        loss2 = exact_chunked_table_step(ref, y, 0.3, chunk_size=chunk)
        assert torch.allclose(loss1, loss2, atol=1e-12)
        assert torch.allclose(m.W_in, ref.W_in, atol=1e-14)
        assert torch.allclose(m.W_out, ref.W_out, atol=1e-14)
