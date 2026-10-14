"""Initialization and raw-weight diagnostics shared by the paper runner."""

import numpy as np
import torch

from .utils import cross_entropy_high_precision


def init_special(m, spec, cfg):
    rng = np.random.default_rng(cfg["numpy_seed"])
    M = m.width
    n = spec.n
    if cfg["init"] == "phase":
        freq = np.arange(M) % (n - 1) + 1
        delta = np.linspace(-np.pi + 0.05, np.pi - 0.05, M, endpoint=False)[rng.permutation(M)]
        c = np.zeros((M, n), complex)
        o = c.copy()
        for j, k in enumerate(freq):
            ph = rng.uniform(0, 2 * np.pi)
            q = spec.conjugates[k]
            c[j, k] = np.exp(1j * ph) / np.sqrt(2)
            c[j, q] = c[j, k].conjugate()
            o[j, k] = np.exp(1j * (2 * ph - delta[j])) / np.sqrt(2)
            o[j, q] = o[j, k].conjugate()
        w = (c @ spec.B).real
        v = (o @ spec.B).real
    else:
        # Original NumPy seed and Gaussian draw; higher precision retained.
        w = rng.standard_normal((M, n))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        c = w @ spec.B.conj().T
        o = abs(c) * np.exp(2j * np.angle(c))
        v = (o @ spec.B).real
    with torch.no_grad():
        m.W_in.copy_(torch.as_tensor(w, dtype=m.W_in.dtype, device=m.W_in.device))
        m.W_out.copy_(torch.as_tensor(v.T, dtype=m.W_out.dtype, device=m.W_out.device))
    m.normalize_directions()


def summary(diag):
    out = {}
    for key in ["signed", "unsigned", "ray", "rank", "concentration"]:
        z = diag[key]
        out[key + "_mean"] = np.nanmean(z, axis=-1).tolist()
        out[key + "_std"] = np.nanstd(z, axis=-1).tolist()
        out[key + "_min"] = np.nanmin(z, axis=-1).tolist()
        out[key + "_valid"] = np.isfinite(z).sum(-1).tolist()
    out["rank_highdim_mean"] = (
        np.nanmean(diag["rank"][:, diag["selected_dimension"] > 1], axis=-1).tolist()
        if (diag["selected_dimension"] > 1).any()
        else []
    )
    out["rank_highdim_count"] = int((diag["selected_dimension"] > 1).sum())
    out["max_norm_error"] = float(abs(diag["norms"] - 1).max())
    out["representation_counts"] = np.bincount(
        diag["winner"], minlength=diag["grouped_energy"].shape[-1]
    ).tolist()
    return out


def evaluate(m, x, y, spec, step, dt):
    with torch.no_grad():
        z = m(x)
        loss = cross_entropy_high_precision(z, y).item()
        acc = (z.argmax(-1) == y).double().mean().item()
        correct = z.gather(1, y[:, None])
        other = z.clone()
        other.scatter_(1, y[:, None], -torch.inf)
        margin = (correct[:, 0] - other.max(1).values).min().item()
    d = spec.diagnostics(
        m.W_in.detach().cpu().numpy(), m.W_out.detach().cpu().numpy(), m.share_embed
    )
    rec = dict(
        step=step,
        time=step * dt,
        loss_sum=loss,
        loss_mean=loss / len(y),
        accuracy=acc,
        min_margin=margin,
        **summary(d),
    )
    return rec, d
