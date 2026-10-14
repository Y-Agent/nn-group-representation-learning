"""Paper flows: summed CE, 1/M predictor, projected Euler with sphere retraction."""

import math

import torch
from torch import nn

from .utils import cross_entropy_high_precision


class FrozenScaleModel(nn.Module):
    """Stage II with direct (tied or independent) effective quadratic coefficients.

    Buffers are cloned to make direction freezing unambiguous. No positivity
    clipping or exponential parametrization is added: those change the stated ODE.
    """

    def __init__(self, model, tied=True):
        super().__init__()
        a0 = model.effective_scale().detach()
        self.register_buffer("W_in", model.W_in.detach().clone())
        self.register_buffer("W_out", model.W_out.detach().clone())
        self.d_vocab, self.width, self.share_embed = model.d_vocab, model.width, model.share_embed
        self.a = nn.Parameter(a0.clone() if tied else a0.expand(model.width).clone())

    def features(self, x):
        t1 = self.W_in[:, : self.d_vocab]
        t2 = t1 if self.share_embed else self.W_in[:, self.d_vocab :]
        return (t1[:, x[:, 0]].T + t2[:, x[:, 1]].T).square()

    def forward(self, x):
        return (self.features(x) * self.a) @ self.W_out.T / self.width


def full_table_logits(model):
    """Exact quadratic logits for lexicographically ordered complete input pairs.

    Broadcasting removes multiplications by sparse one-hot matrices; it does not
    approximate CE or change its derivatives. Equivalent to model(cartesian_prod).
    """
    if model.act_type != "Quad":
        raise ValueError("Full-table optimized path is quadratic")
    left = model.W_in[:, : model.d_vocab].T
    right = left if model.share_embed else model.W_in[:, model.d_vocab :].T
    h = (left[:, None, :] + right[None, :, :]).reshape(-1, model.width)
    return (h.square() @ model.W_out.T) * model.effective_scale() / model.width


def direction_step(model, x, y, dt, full_table=False):
    """One first-order retracted Euler step. dt is time for the summed risk."""
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    if full_table:
        expected = torch.cartesian_prod(
            torch.arange(model.d_vocab, device=x.device),
            torch.arange(model.d_vocab, device=x.device),
        )
        if not torch.equal(x, expected):
            raise ValueError("Full-table optimization requires lexicographic input pairs")
    model.zero_grad(set_to_none=True)
    loss = cross_entropy_high_precision(full_table_logits(model) if full_table else model(x), y)
    loss.backward()
    model.project_direction_gradients()
    with torch.no_grad():
        model.W_in.add_(model.W_in.grad, alpha=-dt * (0.5 if model.share_embed else 1))
        model.W_out.add_(model.W_out.grad, alpha=-dt)
    model.normalize_directions()
    return loss.detach()


@torch.no_grad()
def exact_chunked_table_step(model, labels, dt, chunk_size=1024):
    """Memory-bounded exact full-table CE projected Euler update.

    Pass 1 accumulates the full predictor, then forms its CE residual once.
    Pass 2 evaluates each neuron's exact gradient against that fixed residual.
    Neuron chunks change memory layout only: no data minibatches, stochastic
    gradients, or small-logit approximations are used.
    """
    if model.act_type != "Quad":
        raise ValueError("Exact table step requires quadratic activation")
    if not math.isfinite(dt) or dt < 0 or chunk_size <= 0:
        raise ValueError("dt must be finite and nonnegative; chunk_size must be positive")
    n, M = model.d_vocab, model.width
    if len(labels) != n * n:
        raise ValueError("Complete lexicographic composition table required")
    a = model.effective_scale().detach() / M
    left = model.W_in[:, :n]
    right = left if model.share_embed else model.W_in[:, n:]
    z = torch.zeros((n * n, n), dtype=left.dtype, device=left.device)
    for lo in range(0, M, chunk_size):
        hi = min(M, lo + chunk_size)
        h = (left[lo:hi].T[:, None, :] + right[lo:hi].T[None, :, :]).reshape(n * n, hi - lo)
        z.add_(h.square() @ model.W_out[:, lo:hi].T)
    z.mul_(a)
    loss = cross_entropy_high_precision(z, labels)
    residual = z.softmax(-1)
    residual[torch.arange(n * n, device=z.device), labels] -= 1
    for lo in range(0, M, chunk_size):
        hi = min(M, lo + chunk_size)
        l = left[lo:hi]
        r = right[lo:hi]
        o = model.W_out[:, lo:hi]
        h = (l.T[:, None, :] + r.T[None, :, :]).reshape(n * n, hi - lo)
        go = (residual.T @ h.square()) * a
        dh = (residual @ o) * h * (2 * a)
        dh = dh.reshape(n, n, hi - lo)
        gl = dh.sum(1).T
        gr = dh.sum(0).T
        if model.share_embed:
            gl = (gl + gr) * 0.5
            gl -= (gl * l).sum(-1, keepdim=True) * l
            l.add_(gl, alpha=-dt)
            l.div_(l.norm(dim=-1, keepdim=True))
        else:
            gl -= (gl * l).sum(-1, keepdim=True) * l
            gr -= (gr * r).sum(-1, keepdim=True) * r
            l.add_(gl, alpha=-dt)
            r.add_(gr, alpha=-dt)
            l.div_(l.norm(dim=-1, keepdim=True))
            r.div_(r.norm(dim=-1, keepdim=True))
        go -= (go * o).sum(0, keepdim=True) * o
        o.add_(go, alpha=-dt)
        o.div_(o.norm(dim=0, keepdim=True))
    return loss
