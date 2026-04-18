"""Symmetric group representation basis for S_n.

The Fourier transform on S_n uses irreducible representations (irreps).
Irreps of S_n are indexed by partitions (Young diagrams) of n.

DFT convention:
    ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g^{-1})
    ν(g) = Σ_{ρ,i,j} d_ρ · ν̂[ρ]_{ij} · ρ_{ji}(g)

The basis matrix B has rows indexed by (λ, i, j) where:
    - λ is a partition of n
    - i, j ∈ {1, ..., d_λ} index matrix entries

    B_{(λ,i,j), g} = √(d_λ/n!) · ρ_λ(g)_{ji}   (note: j,i indices)

This satisfies: B @ B^H = I (unitary/orthonormal basis)
"""

from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from math import factorial, sqrt
from itertools import permutations

from .base import Basis


def get_partitions(n: int) -> List[Tuple[int, ...]]:
    """Generate all partitions of n in decreasing order.

    A partition is a tuple (λ_1, λ_2, ...) where λ_1 ≥ λ_2 ≥ ... > 0 and sum = n.
    """
    def _partition(n, max_val):
        if n == 0:
            yield ()
            return
        for i in range(min(n, max_val), 0, -1):
            for p in _partition(n - i, i):
                yield (i,) + p

    return list(_partition(n, n))


def partition_dimension(partition: Tuple[int, ...]) -> int:
    """Compute the dimension of the irrep corresponding to a partition.

    Uses the hook length formula:
        d_λ = n! / ∏_{(i,j) ∈ λ} hook(i,j)

    where hook(i,j) = λ_i + λ'_j - i - j + 1 (λ' is the conjugate partition).
    """
    n = sum(partition)
    if n == 0:
        return 1

    # Compute conjugate partition
    conjugate = []
    for j in range(partition[0]):
        conjugate.append(sum(1 for part in partition if part > j))

    # Compute product of hook lengths
    hook_product = 1
    for i, part_i in enumerate(partition):
        for j in range(part_i):
            # hook(i,j) = arm + leg + 1
            # arm = λ_i - j - 1 (cells to the right)
            # leg = λ'_j - i - 1 (cells below)
            arm = part_i - j - 1
            leg = conjugate[j] - i - 1
            hook = arm + leg + 1
            hook_product *= hook

    return factorial(n) // hook_product


class SymmetricGroupBasis(Basis):
    """Fourier basis for the symmetric group S_n using irreducible representations.

    The basis functions are matrix entries of irreps:
        B_{(λ,i,j), g} = √(d_λ/n!) · ρ_λ(g)_{ij}

    where:
        - λ is a partition of n (Young diagram)
        - d_λ is the dimension of irrep λ
        - ρ_λ(g) is the representation matrix for element g
        - i, j ∈ {1, ..., d_λ}

    Currently supports S_3, S_4, and S_5.
    """

    def __init__(self, n: int):
        """Initialize symmetric group basis.

        Args:
            n: Degree of symmetric group S_n
        """
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if n > 4:
            raise ValueError(f"n > 4 not yet supported (S_5 requires more careful implementation)")

        self.n = n
        self._size = factorial(n)

        # Generate all permutations
        self._perms = list(permutations(range(n)))
        self._perm_to_idx = {p: i for i, p in enumerate(self._perms)}

        # Get partitions and their dimensions
        self._partitions = get_partitions(n)
        self._partition_dims = {p: partition_dimension(p) for p in self._partitions}

        # Verify: sum of d²_λ should equal n!
        dim_check = sum(d**2 for d in self._partition_dims.values())
        assert dim_check == self._size, f"Dimension check failed: {dim_check} != {self._size}"

        # Build representation matrices
        self._irreps = self._build_representations()

        # Cache for basis matrix
        self._basis_cache = {}

    def _build_representations(self) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], np.ndarray]]:
        """Build all irreducible representation matrices.

        Returns:
            Dict mapping partition -> (Dict mapping perm -> representation matrix)
        """
        if self.n == 2:
            return self._build_s2_irreps()
        elif self.n == 3:
            return self._build_s3_irreps()
        elif self.n == 4:
            return self._build_s4_irreps()
        else:
            raise NotImplementedError(f"S_{self.n} representations not implemented")

    def _build_s2_irreps(self) -> Dict:
        """Build irreps for S_2."""
        irreps = {}

        # S_2 elements: (0,1) = identity, (1,0) = swap
        e = (0, 1)
        s = (1, 0)

        # Trivial representation [2]
        irreps[(2,)] = {
            e: np.array([[1.0]]),
            s: np.array([[1.0]]),
        }

        # Sign representation [1,1]
        irreps[(1, 1)] = {
            e: np.array([[1.0]]),
            s: np.array([[-1.0]]),
        }

        return irreps

    def _build_s3_irreps(self) -> Dict:
        """Build irreps for S_3.

        S_3 has 3 irreps:
            - [3]: Trivial, dim=1
            - [2,1]: Standard, dim=2
            - [1,1,1]: Sign, dim=1
        """
        irreps = {}

        # S_3 elements
        e = (0, 1, 2)      # identity
        c1 = (1, 2, 0)     # (012) cycle
        c2 = (2, 0, 1)     # (021) cycle
        s1 = (0, 2, 1)     # (12) swap
        s2 = (2, 1, 0)     # (02) swap
        s3 = (1, 0, 2)     # (01) swap

        # Trivial representation [3]: ρ(g) = 1
        irreps[(3,)] = {
            e: np.array([[1.0]]),
            c1: np.array([[1.0]]),
            c2: np.array([[1.0]]),
            s1: np.array([[1.0]]),
            s2: np.array([[1.0]]),
            s3: np.array([[1.0]]),
        }

        # Sign representation [1,1,1]: ρ(g) = sign(g)
        irreps[(1, 1, 1)] = {
            e: np.array([[1.0]]),
            c1: np.array([[1.0]]),
            c2: np.array([[1.0]]),
            s1: np.array([[-1.0]]),
            s2: np.array([[-1.0]]),
            s3: np.array([[-1.0]]),
        }

        # Standard representation [2,1]: 2-dimensional
        # Use the orthogonal Young representation
        # Generators: c1 = (012), s3 = (01)
        # With orthonormal basis from Young tableaux

        sqrt3 = np.sqrt(3)

        # Standard rep matrices (Young's orthogonal form)
        irreps[(2, 1)] = {
            e: np.array([[1.0, 0.0], [0.0, 1.0]]),
            c1: np.array([[-0.5, -sqrt3/2], [sqrt3/2, -0.5]]),
            c2: np.array([[-0.5, sqrt3/2], [-sqrt3/2, -0.5]]),
            s1: np.array([[-0.5, sqrt3/2], [sqrt3/2, 0.5]]),
            s2: np.array([[-0.5, -sqrt3/2], [-sqrt3/2, 0.5]]),
            s3: np.array([[1.0, 0.0], [0.0, -1.0]]),
        }

        return irreps

    def _build_s4_irreps(self) -> Dict:
        """Build irreps for S_4.

        S_4 has 5 irreps:
            - [4]: Trivial, dim=1
            - [3,1]: Standard, dim=3
            - [2,2]: Two-dimensional
            - [2,1,1]: Sign ⊗ Standard, dim=3
            - [1,1,1,1]: Sign, dim=1
        """
        irreps = {}

        # Generate all 24 permutations
        perms = list(permutations(range(4)))

        def sign(p):
            """Compute sign of permutation."""
            n_inv = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    if p[i] > p[j]:
                        n_inv += 1
            return 1 if n_inv % 2 == 0 else -1

        def compose(s, t):
            """Compose permutations: (s ∘ t)(i) = s(t(i))"""
            return tuple(s[t[i]] for i in range(4))

        # Trivial representation [4]
        irreps[(4,)] = {p: np.array([[1.0]]) for p in perms}

        # Sign representation [1,1,1,1]
        irreps[(1, 1, 1, 1)] = {p: np.array([[float(sign(p))]]) for p in perms}

        # Standard representation [3,1] (3-dimensional)
        # Constructed from the natural permutation representation
        # Natural rep: 4x4 permutation matrices, subtract trivial = 3D irrep
        # Use orthonormal basis: v_1 = (1,-1,0,0)/√2, v_2 = (1,1,-2,0)/√6, v_3 = (1,1,1,-3)/√12

        sqrt2 = np.sqrt(2)
        sqrt6 = np.sqrt(6)
        sqrt12 = np.sqrt(12)

        # Change of basis matrix from natural rep to orthonormal complement of trivial
        # P: standard basis -> orthonormal basis for the 3D subspace orthogonal to (1,1,1,1)
        P = np.array([
            [1/sqrt2, -1/sqrt2, 0, 0],
            [1/sqrt6, 1/sqrt6, -2/sqrt6, 0],
            [1/sqrt12, 1/sqrt12, 1/sqrt12, -3/sqrt12]
        ])

        irreps[(3, 1)] = {}
        for p in perms:
            # Natural permutation matrix
            nat = np.zeros((4, 4))
            for i in range(4):
                nat[p[i], i] = 1.0
            # Project to 3D subspace: P @ nat @ P^T
            rep = P @ nat @ P.T
            irreps[(3, 1)][p] = rep

        # Sign ⊗ Standard [2,1,1] (3-dimensional)
        irreps[(2, 1, 1)] = {}
        for p in perms:
            irreps[(2, 1, 1)][p] = sign(p) * irreps[(3, 1)][p]

        # [2,2] representation (2-dimensional)
        # This can be constructed from the tensor square of the natural rep
        # Or explicitly using characters
        # Use explicit construction via Young's orthogonal representation

        # For [2,2], we use the fact that it appears in Alt^2(natural)
        # Basis: e_ij = e_i ∧ e_j for i < j, then project
        # Simpler: compute from character and explicit matrices

        # Identity
        e = (0, 1, 2, 3)

        # Build from generators: s = (01), t = (12), u = (23)
        s = (1, 0, 2, 3)  # (01)
        t = (0, 2, 1, 3)  # (12)
        u = (0, 1, 3, 2)  # (23)

        # Use Murphy's rational form for [2,2]
        # Jucys-Murphy elements give us the representation

        # For [2,2], the 2D rep can be constructed explicitly
        # Using the standard basis from Young tableaux:
        # T1: [[1,2],[3,4]], T2: [[1,3],[2,4]]

        # Explicit matrices for generators in the [2,2] representation
        sqrt2 = np.sqrt(2)

        # Define representation matrices for generators
        rho_22 = {e: np.eye(2)}

        # Compute for transpositions (adjacent)
        # (01) swaps T1 and T2 with specific coefficients
        # Using axial distance formula from Young's seminormal form

        # For (01): acts on {1,2} in row 1 of T1, different in T2
        # (01) on T1: [[2,1],[3,4]] = T1 (after sorting) -> coefficient related to axial distance
        # Axial distance for 1,2 in T1: same row, distance = 1 -> rho(s_{12}) = 1/1 = 1
        # But need off-diagonal terms for T2

        # I'll use the explicit form computed from the representation
        # [2,2] matrices for S_4 generators:

        # (01):
        rho_22[s] = np.array([[1.0, 0.0], [0.0, -1.0]])
        # (12):
        rho_22[t] = np.array([[-0.5, sqrt(3)/2], [sqrt(3)/2, 0.5]])
        # (23):
        rho_22[u] = np.array([[1.0, 0.0], [0.0, -1.0]])

        # Now generate all 24 elements by composition
        # S_4 is generated by (01), (12), (23)

        def get_rho(p):
            """Get rep matrix for permutation p by decomposing into generators."""
            if p in rho_22:
                return rho_22[p]

            # Try to build from known elements
            for known_p, known_mat in list(rho_22.items()):
                # Try s ∘ known = p, so s = p ∘ known^{-1}
                for gen, gen_mat in [(s, rho_22[s]), (t, rho_22[t]), (u, rho_22[u])]:
                    composed = compose(gen, known_p)
                    if composed == p:
                        rho_22[p] = gen_mat @ known_mat
                        return rho_22[p]
                    composed = compose(known_p, gen)
                    if composed == p:
                        rho_22[p] = known_mat @ gen_mat
                        return rho_22[p]
            return None

        # Iteratively build all representations
        for _ in range(10):  # Enough iterations to reach all elements
            for p in perms:
                if p not in rho_22:
                    get_rho(p)

        # Verify we got all 24
        if len(rho_22) < 24:
            # Fill in remaining using character orthogonality
            # For any missing, try all double compositions
            for p in perms:
                if p not in rho_22:
                    for p1, m1 in list(rho_22.items()):
                        for p2, m2 in list(rho_22.items()):
                            if compose(p1, p2) == p:
                                rho_22[p] = m1 @ m2
                                break
                        if p in rho_22:
                            break

        irreps[(2, 2)] = rho_22

        return irreps

    @property
    def name(self) -> str:
        return f"symmetric_S{self.n}"

    @property
    def size(self) -> int:
        """Number of basis functions (= n!)."""
        return self._size

    @property
    def num_frequencies(self) -> int:
        """Number of non-trivial basis functions."""
        return self._size - 1

    @property
    def partitions(self) -> List[Tuple[int, ...]]:
        """List of partitions indexing irreps."""
        return self._partitions

    @property
    def partition_dimensions(self) -> Dict[Tuple[int, ...], int]:
        """Dimensions of each irrep."""
        return self._partition_dims

    def get_basis_names(self) -> List[str]:
        """Get names for each basis function.

        Returns:
            List of names like "ρ_[3](1,1)", "ρ_[2,1](1,1)", "ρ_[2,1](1,2)", etc.
        """
        names = []
        for partition in self._partitions:
            dim = self._partition_dims[partition]
            part_str = ','.join(map(str, partition))
            for i in range(dim):
                for j in range(dim):
                    names.append(f"ρ_[{part_str}]({i+1},{j+1})")
        return names

    def get_basis_matrix(self, device: torch.device = None) -> torch.Tensor:
        """Get the Fourier basis matrix for S_n.

        Following the paper's DFT convention:
            ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g^{-1})
            ν(g) = Σ_{ρ,i,j} d_ρ · ν̂[ρ]_{ij} · ρ_{ji}(g)

        Returns:
            Complex tensor of shape (n!, n!) where:
            B_{(λ,i,j), g} = √(d_λ/n!) · ρ_λ(g)_{ji}   (note: j,i indices)

            This is a unitary matrix: B @ B^H = I
        """
        if device is None:
            device = torch.device('cpu')

        device_key = str(device)
        if device_key in self._basis_cache:
            return self._basis_cache[device_key]

        N = self._size
        basis = np.zeros((N, N), dtype=np.complex128)

        row_idx = 0
        for partition in self._partitions:
            dim = self._partition_dims[partition]
            norm = np.sqrt(dim / N)  # Normalization factor

            for i in range(dim):
                for j in range(dim):
                    for col_idx, perm in enumerate(self._perms):
                        rho_g = self._irreps[partition][perm]
                        # Use rho[j,i] per paper's convention: B_{(ρ,i,j),g} = √(d/|G|)·ρ_{ji}(g)
                        basis[row_idx, col_idx] = norm * rho_g[j, i]
                    row_idx += 1

        # Convert to torch tensor
        basis_tensor = torch.tensor(basis, dtype=torch.complex128, device=device)

        self._basis_cache[device_key] = basis_tensor
        return basis_tensor

    def transform(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform on S_n: c = ν @ B^H.

        Returns Euclidean coefficients C that preserve norm: ||c||² = ||ν||².

        Args:
            signal: Real or complex tensor of shape (..., n!)

        Returns:
            Complex tensor of shape (..., n!) containing Euclidean coefficients.
            To get paper's ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g^{-1}), use transform_paper().
        """
        basis = self.get_basis_matrix(signal.device)
        if not signal.is_complex():
            signal = signal.to(torch.complex128)
        return signal @ basis.T.conj()

    def transform_paper(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform with paper's L²(G) scaling.

        Returns coefficients ν̂[ρ] matching paper's DFT definition:
            ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g^{-1})

        These satisfy Plancherel: Σ_ρ d_ρ ||ν̂[ρ]||²_F = ||ν||²_{L²(G)} = (1/|G|)||ν||²

        Args:
            signal: Real or complex tensor of shape (..., n!)

        Returns:
            Complex tensor of shape (..., n!) containing paper-notation coefficients.
            Each ρ-block is ν̂[ρ] = C[ρ] / √(|G|·d_ρ).
        """
        # Get Euclidean coefficients
        C = self.transform(signal)

        # Convert to paper notation: ν̂[ρ] = C[ρ] / √(|G|·d_ρ)
        nu_hat = C.clone()
        N = self._size
        curr_idx = 0
        for partition in self._partitions:
            d = self._partition_dims[partition]
            block_size = d * d
            scale = np.sqrt(N * d)
            nu_hat[..., curr_idx:curr_idx + block_size] /= scale
            curr_idx += block_size

        return nu_hat

    def inverse_transform(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply inverse Fourier transform on S_n.

        For unitary basis B with forward transform c = ν·B^H,
        the inverse is ν = c·B.

        Args:
            coeffs: Complex tensor of shape (..., n!)

        Returns:
            Complex tensor of shape (..., n!)
        """
        basis = self.get_basis_matrix(coeffs.device)
        return coeffs @ basis

    def get_irrep_block_indices(self, partition: Tuple[int, ...]) -> Tuple[int, int]:
        """Get the start and end indices for an irrep block in the basis.

        Args:
            partition: Partition labeling the irrep

        Returns:
            (start_idx, end_idx) for the block in the basis
        """
        start = 0
        for p in self._partitions:
            if p == partition:
                dim = self._partition_dims[p]
                return start, start + dim * dim
            start += self._partition_dims[p] ** 2
        raise ValueError(f"Unknown partition: {partition}")

    def get_dominant_frequency(self, coeffs: torch.Tensor) -> int:
        """Find the dominant non-trivial frequency component.

        Args:
            coeffs: Complex Fourier coefficients of shape (n!,)

        Returns:
            Index of dominant coefficient (excluding trivial rep at index 0)
        """
        magnitudes = torch.abs(coeffs[1:])
        return int(torch.argmax(magnitudes).item()) + 1

    def get_phase(self, coeffs: torch.Tensor, idx: int) -> float:
        """Get phase of a frequency component.

        Args:
            coeffs: Complex Fourier coefficients of shape (n!,)
            idx: Index of the coefficient

        Returns:
            Phase angle in radians [-π, π]
        """
        return torch.angle(coeffs[idx]).item()

    def get_magnitude(self, coeffs: torch.Tensor, idx: int) -> float:
        """Get magnitude of a frequency component."""
        return torch.abs(coeffs[idx]).item()

    def get_frequency_label(self, idx: int) -> str:
        """Get the label for a basis function by index.

        Args:
            idx: Index in the flattened basis

        Returns:
            Label like "[2,1](1,2)" indicating partition and matrix position
        """
        curr_idx = 0
        for partition in self._partitions:
            dim = self._partition_dims[partition]
            block_size = dim * dim
            if idx < curr_idx + block_size:
                local_idx = idx - curr_idx
                i = local_idx // dim
                j = local_idx % dim
                part_str = ','.join(map(str, partition))
                return f"[{part_str}]({i+1},{j+1})"
            curr_idx += block_size
        return f"idx={idx}"

    def __repr__(self) -> str:
        dims = [f"[{','.join(map(str, p))}]:{d}" for p, d in self._partition_dims.items()]
        return f"SymmetricGroupBasis(n={self.n}, size={self._size}, irreps={{{', '.join(dims)}}})"
