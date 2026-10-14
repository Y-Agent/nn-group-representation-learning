"""Raw-weight spectral diagnostics, with explicit Euclidean/paper conventions."""

import numpy as np

from .basis import get_basis
from .tasks import get_task


class Spectrum:
    def __init__(self, group):
        if group not in {"z15", "z30", "f21", "s4"}:
            raise ValueError(f"Unsupported spectral group: {group}")
        if group in ("z15", "z30"):
            mods = [3, 5] if group == "z15" else [2, 3, 5]
            self.task = get_task("modular_addition", n=mods)
            basis = get_basis("fourier", n=mods)
            # Construct in double precision (legacy Fourier basis caches complex64).
            freq = np.array([basis.get_frequency_tuple(i) for i in range(basis.size)])
            self.B = np.exp(2j * np.pi * ((freq / np.array(mods)) @ freq.T)) / np.sqrt(basis.size)
            self.labels = [str(tuple(map(int, k))) for k in freq]
            self.dims = [1] * basis.size
            self.conjugates = [basis.tuple_to_index(tuple((-k) % np.array(mods))) for k in freq]
        else:
            name = "frobenius21" if group == "f21" else "symmetric"
            self.task = (
                get_task("frobenius21") if group == "f21" else get_task("symmetric_group", n=4)
            )
            basis = get_basis(name, **({} if group == "f21" else {"n": 4}))
            self.B = basis.get_basis_matrix().numpy()
            self.labels = list(basis.irrep_labels if group == "f21" else basis.partitions)
            self.dims = (
                [basis.irrep_dimensions[k] for k in self.labels]
                if group == "f21"
                else [basis.partition_dimensions[k] for k in self.labels]
            )
            self.labels = list(map(str, self.labels))
            conj = {
                "omega": "omega_conj",
                "omega_conj": "omega",
                "rho_3": "rho_3_conj",
                "rho_3_conj": "rho_3",
            }
            self.conjugates = (
                [self.labels.index(conj.get(k, k)) for k in self.labels]
                if group == "f21"
                else list(range(len(self.labels)))
            )
        self.n = self.task.vocab_size
        self.slices = []
        offset = 0
        for d in self.dims:
            self.slices.append(slice(offset, offset + d * d))
            offset += d * d
        self.orbits = []
        seen = set()
        for i, j in enumerate(self.conjugates):
            if i not in seen:
                orb = sorted({i, j})
                self.orbits.append(orb)
                seen.update(orb)
        assert np.max(np.abs(self.B @ self.B.conj().T - np.eye(self.n))) < 1e-6

    def coefficients(self, weights):
        raw = weights @ self.B.conj().T
        return [
            raw[:, s].reshape(-1, d, d) / np.sqrt(self.n * d)
            for s, d in zip(self.slices, self.dims)
        ]

    def diagnostics(self, W_in, W_out, shared=False, selected_orbits=None):
        w1 = W_in[:, : self.n]
        w2 = w1 if shared else W_in[:, self.n :]
        ws = [w1, w2, W_out.T]
        cs = [self.coefficients(w) for w in ws]
        energy = np.stack(
            [
                np.stack(
                    [self.n * d * np.sum(abs(c) ** 2, axis=(1, 2)) for c, d in zip(z, self.dims)], 1
                )
                for z in cs
            ]
        )
        grouped = np.stack([energy[:, :, o].sum(-1) for o in self.orbits], -1)
        winner = grouped.sum(0).argmax(-1) if selected_orbits is None else selected_orbits
        selected = np.array([self.orbits[i][0] for i in winner])
        M = len(w1)
        signed = np.full((3, M), np.nan)
        unsigned = signed.copy()
        rank = signed.copy()
        ray = signed.copy()
        omega = np.zeros(M)
        for i, d in enumerate(self.dims):
            A, B, C = [z[i] for z in cs]
            if i != 0:
                omega += d * np.einsum("mij,mij->m", (B @ A).conj(), C).real
            ix = np.where(selected == i)[0]
            if not len(ix):
                continue
            a, b, c = A[ix], B[ix], C[ix]
            for j, (x, y) in enumerate(
                [
                    (b @ a, c),
                    (b.conj().transpose(0, 2, 1) @ c, a),
                    (c @ a.conj().transpose(0, 2, 1), b),
                ]
            ):
                nx = np.linalg.norm(x, axis=(1, 2))
                ny = np.linalg.norm(y, axis=(1, 2))
                valid = (nx > 1e-12) & (ny > 1e-12)
                inner = np.einsum("mij,mij->m", x.conj(), y)
                s = np.divide(inner.real, nx * ny, out=np.full(len(ix), np.nan), where=valid)
                signed[j, ix] = np.clip(s, -1, 1)
                unsigned[j, ix] = np.divide(
                    abs(inner), nx * ny, out=np.full(len(ix), np.nan), where=valid
                )
                ray[j, ix] = np.sqrt(np.maximum(0, 1 - np.maximum(0, s) ** 2))
            for j, x in enumerate([a, b, c]):
                sv = np.linalg.svd(x, compute_uv=False)
                rank[j, ix] = np.where(
                    sv[:, 0] > 1e-12, sv[:, 1] / sv[:, 0] if d > 1 else 0, np.nan
                )
        norms = np.stack([np.linalg.norm(w, axis=1) for w in ws])
        return dict(
            coefficients=np.stack([np.concatenate([c.reshape(M, -1) for c in z], 1) for z in cs]),
            norms=norms,
            selected_irrep=selected,
            selected_dimension=np.array(self.dims)[selected],
            energy=energy,
            grouped_energy=grouped,
            winner=winner,
            signed=signed,
            unsigned=unsigned,
            ray=ray,
            rank=rank,
            omega=omega,
            concentration=grouped.max(-1) / np.maximum(grouped.sum(-1), 1e-30),
        )
