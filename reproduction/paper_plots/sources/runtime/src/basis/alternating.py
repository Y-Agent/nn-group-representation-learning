"""Alternating group representation basis for A_n.

The Fourier transform on A_n uses irreducible representations (irreps).
A_n is the group of even permutations, a normal subgroup of S_n with |A_n| = n!/2.

DFT convention:
    ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g^{-1})
    ν(g) = Σ_{ρ,i,j} d_ρ · ν̂[ρ]_{ij} · ρ_{ji}(g)

The irreps of A_n come from restricting S_n irreps:
- If partition λ ≠ λ' (conjugate), then Res(ρ_λ) = Res(ρ_λ') is irreducible
- If partition λ = λ' (self-conjugate), then Res(ρ_λ) splits into two irreps

For A_3: 3 elements, 3 irreps of dimension 1 (isomorphic to Z_3)
For A_4: 12 elements, 4 irreps of dimensions 1, 1, 1, 3
For A_5: 60 elements, 5 irreps of dimensions 1, 3, 3, 4, 5
"""

from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from math import factorial, sqrt
from itertools import permutations
import cmath

from .base import Basis


def permutation_sign(perm: Tuple[int, ...]) -> int:
    """Compute sign of permutation."""
    n = len(perm)
    n_inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                n_inv += 1
    return 1 if n_inv % 2 == 0 else -1


class AlternatingGroupBasis(Basis):
    """Fourier basis for the alternating group A_n using irreducible representations.

    The basis functions are matrix entries of irreps:
        B_{(λ,i,j), g} = √(d_λ/|A_n|) · ρ_λ(g)_{ij}

    Currently supports A_3, A_4, and A_5.
    """

    def __init__(self, n: int):
        """Initialize alternating group basis.

        Args:
            n: Degree of alternating group A_n
        """
        if n < 3:
            raise ValueError(f"n must be at least 3, got {n}")
        if n > 5:
            raise ValueError(f"n > 5 not yet supported (A_6 has 360 elements)")

        self.n = n
        self._size = factorial(n) // 2

        # Generate all even permutations
        self._perms = [p for p in permutations(range(n)) if permutation_sign(p) == 1]
        self._perm_to_idx = {p: i for i, p in enumerate(self._perms)}

        # Build irrep information and representation matrices
        self._irrep_labels, self._irrep_dims, self._irreps = self._build_representations()

        # Verify: sum of d² should equal |A_n|
        dim_check = sum(d**2 for d in self._irrep_dims.values())
        assert dim_check == self._size, f"Dimension check failed: {dim_check} != {self._size}"

        # Cache for basis matrix
        self._basis_cache = {}

    def _build_representations(self):
        """Build all irreducible representation matrices."""
        if self.n == 3:
            return self._build_a3_irreps()
        elif self.n == 4:
            return self._build_a4_irreps()
        elif self.n == 5:
            return self._build_a5_irreps()
        else:
            raise NotImplementedError(f"A_{self.n} representations not implemented")

    def _build_a3_irreps(self):
        """Build irreps for A_3.

        A_3 ≅ Z_3 (cyclic group of order 3)
        3 irreps, all 1-dimensional:
            - Trivial: ρ(g) = 1
            - ω-rep: ρ((012)) = ω = e^(2πi/3)
            - ω²-rep: ρ((012)) = ω²
        """
        labels = ['trivial', 'omega', 'omega_conj']
        dims = {'trivial': 1, 'omega': 1, 'omega_conj': 1}
        irreps = {}

        # A_3 elements: e, (012), (021)
        e = (0, 1, 2)      # identity
        c1 = (1, 2, 0)     # (012) 3-cycle
        c2 = (2, 0, 1)     # (021) 3-cycle (= c1^2)

        omega = cmath.exp(2j * cmath.pi / 3)

        # Trivial representation
        irreps['trivial'] = {
            e: np.array([[1.0 + 0j]]),
            c1: np.array([[1.0 + 0j]]),
            c2: np.array([[1.0 + 0j]]),
        }

        # ω representation
        irreps['omega'] = {
            e: np.array([[1.0 + 0j]]),
            c1: np.array([[omega]]),
            c2: np.array([[omega**2]]),
        }

        # ω² representation (complex conjugate)
        irreps['omega_conj'] = {
            e: np.array([[1.0 + 0j]]),
            c1: np.array([[omega**2]]),
            c2: np.array([[omega]]),
        }

        return labels, dims, irreps

    def _build_a4_irreps(self):
        """Build irreps for A_4.

        A_4 has 12 elements and 4 conjugacy classes:
            - {e}: 1 element
            - {(12)(34), (13)(24), (14)(23)}: 3 double transpositions
            - 4 elements: 3-cycles conjugate to (012)
            - 4 elements: 3-cycles conjugate to (021)

        4 irreps with dimensions 1, 1, 1, 3:
            - Trivial (1D)
            - ω-rep (1D): one 3-cycle class → ω, other → ω², double-trans → 1
            - ω²-rep (1D): conjugate of ω-rep
            - Standard (3D): restriction of S_4 standard rep
        """
        labels = ['trivial', 'omega', 'omega_conj', 'standard']
        dims = {'trivial': 1, 'omega': 1, 'omega_conj': 1, 'standard': 3}
        irreps = {}

        # Generate all even permutations of {0,1,2,3}
        perms = [p for p in permutations(range(4)) if permutation_sign(p) == 1]

        omega = cmath.exp(2j * cmath.pi / 3)

        def compose(s, t):
            return tuple(s[t[i]] for i in range(4))

        def inverse(p):
            inv = [0]*4
            for i, j in enumerate(p):
                inv[j] = i
            return tuple(inv)

        # Build conjugacy classes properly by computing them explicitly
        # Reference 3-cycle: (012) = (1, 2, 0, 3) meaning 0->1, 1->2, 2->0, 3->3
        ref_3cycle = (1, 2, 0, 3)

        # Compute conjugacy class of reference 3-cycle in A_4
        class_of_ref = set()
        for a in perms:
            conj = compose(compose(a, ref_3cycle), inverse(a))
            class_of_ref.add(conj)

        def get_conjugacy_class(p):
            """Determine conjugacy class of permutation."""
            if p == (0, 1, 2, 3):
                return 'identity'
            # Check if double transposition
            if all(p[p[i]] == i for i in range(4)) and sum(1 for i in range(4) if p[i] == i) == 0:
                return 'double_trans'
            # Must be a 3-cycle - check which conjugacy class
            if p in class_of_ref:
                return '3cycle_class1'
            else:
                return '3cycle_class2'

        # Trivial representation
        irreps['trivial'] = {p: np.array([[1.0 + 0j]]) for p in perms}

        # ω and ω² representations
        # Double transpositions → 1
        # 3-cycle class 1 → ω (for omega rep)
        # 3-cycle class 2 → ω² (for omega rep)

        irreps['omega'] = {}
        irreps['omega_conj'] = {}

        for p in perms:
            cc = get_conjugacy_class(p)
            if cc == 'identity':
                irreps['omega'][p] = np.array([[1.0 + 0j]])
                irreps['omega_conj'][p] = np.array([[1.0 + 0j]])
            elif cc == 'double_trans':
                irreps['omega'][p] = np.array([[1.0 + 0j]])
                irreps['omega_conj'][p] = np.array([[1.0 + 0j]])
            elif cc == '3cycle_class1':
                irreps['omega'][p] = np.array([[omega]])
                irreps['omega_conj'][p] = np.array([[omega**2]])
            else:  # 3cycle_class2
                irreps['omega'][p] = np.array([[omega**2]])
                irreps['omega_conj'][p] = np.array([[omega]])

        # Standard 3D representation
        # Restriction of S_4 standard rep to A_4
        sqrt2 = np.sqrt(2)
        sqrt6 = np.sqrt(6)
        sqrt12 = np.sqrt(12)

        # Orthonormal basis for 3D subspace orthogonal to (1,1,1,1)
        P = np.array([
            [1/sqrt2, -1/sqrt2, 0, 0],
            [1/sqrt6, 1/sqrt6, -2/sqrt6, 0],
            [1/sqrt12, 1/sqrt12, 1/sqrt12, -3/sqrt12]
        ])

        irreps['standard'] = {}
        for p in perms:
            # Natural permutation matrix
            nat = np.zeros((4, 4))
            for i in range(4):
                nat[p[i], i] = 1.0
            # Project to 3D subspace
            rep = P @ nat @ P.T
            irreps['standard'][p] = rep.astype(complex)

        return labels, dims, irreps

    def _build_a5_irreps(self):
        """Build irreps for A_5.

        A_5 has 60 elements and 5 conjugacy classes:
            - {e}: 1 element
            - 15 double transpositions (ab)(cd)
            - 20 3-cycles
            - 12 5-cycles (type 1, conjugate to (01234))
            - 12 5-cycles (type 2, conjugate to (01243))

        5 irreps with dimensions 1, 3, 3, 4, 5:
            - Trivial (1D)
            - Two 3D irreps (from Alt^2 of 4D standard)
            - 4D irrep (standard, from S_5's [4,1] partition)
            - 5D irrep (from Sym^2 of 3D minus trivial)

        Character table:
            Class:    e    (ab)(cd)  3-cycle  5-cycle+  5-cycle-
            Size:     1       15       20        12        12
            trivial:  1        1        1         1         1
            rep_3a:   3       -1        0         φ       1-φ
            rep_3b:   3       -1        0       1-φ         φ
            rep_4:    4        0        1        -1        -1
            rep_5:    5        1       -1         0         0

        where φ = (1+√5)/2 is the golden ratio.
        """
        labels = ['trivial', 'rep_3a', 'rep_3b', 'rep_4', 'rep_5']
        dims = {'trivial': 1, 'rep_3a': 3, 'rep_3b': 3, 'rep_4': 4, 'rep_5': 5}
        irreps = {}

        perms = [p for p in permutations(range(5)) if permutation_sign(p) == 1]
        identity = (0, 1, 2, 3, 4)

        def compose(s, t):
            return tuple(s[t[i]] for i in range(5))

        def inverse(p):
            inv = [0]*5
            for i, j in enumerate(p):
                inv[j] = i
            return tuple(inv)

        def get_cycle_structure(p):
            """Get cycle structure of permutation."""
            visited = [False] * 5
            cycles = []
            for start in range(5):
                if visited[start]:
                    continue
                cycle = []
                i = start
                while not visited[i]:
                    visited[i] = True
                    cycle.append(i)
                    i = p[i]
                if len(cycle) > 1:
                    cycles.append(len(cycle))
            return tuple(sorted(cycles, reverse=True))

        # Build conjugacy classes explicitly
        # Reference elements for each class
        ref_5cycle_plus = (1, 2, 3, 4, 0)   # (01234)
        ref_5cycle_minus = (1, 2, 4, 0, 3)  # (01243) - different A_5 class

        # Compute A_5 conjugacy class of ref_5cycle_plus
        class_5plus = set()
        for a in perms:
            conj = compose(compose(a, ref_5cycle_plus), inverse(a))
            class_5plus.add(conj)

        def get_conjugacy_class(p):
            """Determine conjugacy class of permutation in A_5."""
            if p == identity:
                return 'identity'
            cs = get_cycle_structure(p)
            if cs == (2, 2):
                return 'double_trans'
            elif cs == (3,):
                return '3cycle'
            elif cs == (5,):
                if p in class_5plus:
                    return '5cycle_plus'
                else:
                    return '5cycle_minus'
            else:
                raise ValueError(f"Unexpected cycle structure: {cs}")

        # Character values (φ = golden ratio)
        phi = (1 + np.sqrt(5)) / 2
        chi = {
            'trivial': {'identity': 1, 'double_trans': 1, '3cycle': 1, '5cycle_plus': 1, '5cycle_minus': 1},
            'rep_3a': {'identity': 3, 'double_trans': -1, '3cycle': 0, '5cycle_plus': phi, '5cycle_minus': 1-phi},
            'rep_3b': {'identity': 3, 'double_trans': -1, '3cycle': 0, '5cycle_plus': 1-phi, '5cycle_minus': phi},
            'rep_4': {'identity': 4, 'double_trans': 0, '3cycle': 1, '5cycle_plus': -1, '5cycle_minus': -1},
            'rep_5': {'identity': 5, 'double_trans': 1, '3cycle': -1, '5cycle_plus': 0, '5cycle_minus': 0},
        }

        # ===== 1. Trivial representation (1D) =====
        irreps['trivial'] = {p: np.array([[1.0 + 0j]]) for p in perms}

        # ===== 2. Standard 4D representation =====
        # Project 5D natural permutation onto orthogonal complement of (1,1,1,1,1)
        sqrt2, sqrt6, sqrt12, sqrt20 = np.sqrt(2), np.sqrt(6), np.sqrt(12), np.sqrt(20)
        P4 = np.array([
            [1/sqrt2, -1/sqrt2, 0, 0, 0],
            [1/sqrt6, 1/sqrt6, -2/sqrt6, 0, 0],
            [1/sqrt12, 1/sqrt12, 1/sqrt12, -3/sqrt12, 0],
            [1/sqrt20, 1/sqrt20, 1/sqrt20, 1/sqrt20, -4/sqrt20]
        ])

        irreps['rep_4'] = {}
        for p in perms:
            nat = np.zeros((5, 5))
            for i in range(5):
                nat[p[i], i] = 1.0
            rep = P4 @ nat @ P4.T
            irreps['rep_4'][p] = rep.astype(complex)

        # ===== 3. Two 3D representations from Alt^2(4D) =====
        # Alt^2(4D) = 6D representation that decomposes as 3a ⊕ 3b

        def alt2_matrix(M):
            """Compute Alt^2(M) on antisymmetric tensors.

            Basis for Alt^2(R^4): e_0∧e_1, e_0∧e_2, e_0∧e_3, e_1∧e_2, e_1∧e_3, e_2∧e_3
            """
            basis = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            dim = len(basis)
            result = np.zeros((dim, dim), dtype=complex)
            for row, (i, j) in enumerate(basis):
                for col, (k, l) in enumerate(basis):
                    # M acts on e_k ∧ e_l → (Me_k) ∧ (Me_l)
                    # Coefficient of e_i ∧ e_j is det of 2x2 minor
                    result[row, col] = M[i, k] * M[j, l] - M[i, l] * M[j, k]
            return result

        # Build Alt^2(rep_4) for all elements
        alt2_rep = {p: alt2_matrix(irreps['rep_4'][p].real) for p in perms}

        # Use character-based projection to separate 3a and 3b
        # P_{3a} = (3/60) * Σ_g χ_{3a}(g)* · ρ_{Alt^2}(g)
        P_3a = np.zeros((6, 6), dtype=complex)
        for p in perms:
            cc = get_conjugacy_class(p)
            chi_val = chi['rep_3a'][cc]
            P_3a += chi_val.conjugate() * alt2_rep[p]
        P_3a *= 3.0 / 60.0

        # Find the 3D eigenspace of P_3a (eigenvalue ~1)
        eigvals, eigvecs = np.linalg.eigh(P_3a)
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top 3 eigenvectors (should have eigenvalue ~1)
        basis_3a = eigvecs[:, :3].T  # Shape (3, 6)

        # Project Alt^2 onto 3a subspace
        irreps['rep_3a'] = {}
        for p in perms:
            proj = basis_3a @ alt2_rep[p] @ basis_3a.T.conj()
            irreps['rep_3a'][p] = proj

        # Similarly for 3b (or use orthogonal complement)
        P_3b = np.zeros((6, 6), dtype=complex)
        for p in perms:
            cc = get_conjugacy_class(p)
            chi_val = chi['rep_3b'][cc]
            P_3b += chi_val.conjugate() * alt2_rep[p]
        P_3b *= 3.0 / 60.0

        eigvals_b, eigvecs_b = np.linalg.eigh(P_3b)
        idx_b = np.argsort(eigvals_b)[::-1]
        eigvecs_b = eigvecs_b[:, idx_b]
        basis_3b = eigvecs_b[:, :3].T

        irreps['rep_3b'] = {}
        for p in perms:
            proj = basis_3b @ alt2_rep[p] @ basis_3b.T.conj()
            irreps['rep_3b'][p] = proj

        # ===== 4. 5D representation from Sym^2(3a) minus trivial =====
        # Sym^2(3D) = 6D, decomposes as 1 ⊕ 5

        def sym2_from_3d(M):
            """Sym^2 of 3x3 matrix.

            Basis: e_0⊗e_0, e_1⊗e_1, e_2⊗e_2, (e_0⊗e_1+e_1⊗e_0)/√2, (e_0⊗e_2+e_2⊗e_0)/√2, (e_1⊗e_2+e_2⊗e_1)/√2
            """
            result = np.zeros((6, 6), dtype=complex)
            # Diagonal basis elements (i,i)
            for row in range(3):
                for col in range(3):
                    result[row, col] = M[row, col] ** 2
            # Off-diagonal basis elements
            off_diag = [(0, 1), (0, 2), (1, 2)]
            for row_idx, (i, j) in enumerate(off_diag):
                for col_idx, (k, l) in enumerate(off_diag):
                    # Coefficient of symmetric (i,j) from symmetric (k,l)
                    result[3 + row_idx, 3 + col_idx] = (
                        M[i, k] * M[j, l] + M[i, l] * M[j, k]
                    )
                # Mixed: symmetric (i,j) from diagonal (k,k)
                for col in range(3):
                    result[3 + row_idx, col] = np.sqrt(2) * M[i, col] * M[j, col]
                # Mixed: diagonal (i,i) from symmetric (k,l)
                for col_idx, (k, l) in enumerate(off_diag):
                    result[row_idx, 3 + col_idx] = np.sqrt(2) * M[row_idx, k] * M[row_idx, l]
            return result

        # Build Sym^2(rep_3a)
        sym2_rep = {p: sym2_from_3d(irreps['rep_3a'][p]) for p in perms}

        # Project out trivial component
        # Trivial subspace is spanned by identity in symmetric tensor = (1,1,1,0,0,0)/√3
        v_triv = np.array([1, 1, 1, 0, 0, 0], dtype=complex) / np.sqrt(3)
        P_triv = np.outer(v_triv, v_triv.conj())
        P_orth = np.eye(6, dtype=complex) - P_triv

        # Find orthonormal basis for 5D complement
        eigvals_5, eigvecs_5 = np.linalg.eigh(P_orth)
        idx_5 = np.argsort(eigvals_5)[::-1]
        eigvecs_5 = eigvecs_5[:, idx_5]
        basis_5d = eigvecs_5[:, :5].T  # Top 5 eigenvectors

        # Project Sym^2 onto 5D subspace
        irreps['rep_5'] = {}
        for p in perms:
            proj = basis_5d @ sym2_rep[p] @ basis_5d.T.conj()
            irreps['rep_5'][p] = proj

        return labels, dims, irreps

    @property
    def name(self) -> str:
        return f"alternating_A{self.n}"

    @property
    def size(self) -> int:
        """Number of basis functions (= n!/2)."""
        return self._size

    @property
    def num_frequencies(self) -> int:
        """Number of non-trivial basis functions."""
        return self._size - 1

    @property
    def irrep_labels(self) -> List[str]:
        """List of irrep labels."""
        return self._irrep_labels

    @property
    def irrep_dimensions(self) -> Dict[str, int]:
        """Dimensions of each irrep."""
        return self._irrep_dims

    def get_basis_names(self) -> List[str]:
        """Get names for each basis function."""
        names = []
        for label in self._irrep_labels:
            dim = self._irrep_dims[label]
            for i in range(dim):
                for j in range(dim):
                    names.append(f"ρ_{label}({i+1},{j+1})")
        return names

    def get_basis_matrix(self, device: torch.device = None) -> torch.Tensor:
        """Get the Fourier basis matrix for A_n.

        Following the paper's DFT convention:
            ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g^{-1})
            ν(g) = Σ_{ρ,i,j} d_ρ · ν̂[ρ]_{ij} · ρ_{ji}(g)

        Returns:
            Complex tensor of shape (|A_n|, |A_n|) where:
            B_{(λ,i,j), g} = √(d_λ/|A_n|) · ρ_λ(g)_{ji}   (note: j,i indices)
        """
        if device is None:
            device = torch.device('cpu')

        device_key = str(device)
        if device_key in self._basis_cache:
            return self._basis_cache[device_key]

        N = self._size
        basis = np.zeros((N, N), dtype=np.complex128)

        row_idx = 0
        for label in self._irrep_labels:
            dim = self._irrep_dims[label]
            norm = np.sqrt(dim / N)  # Normalization factor

            for i in range(dim):
                for j in range(dim):
                    for col_idx, perm in enumerate(self._perms):
                        rho_g = self._irreps[label][perm]
                        # Use rho[j,i] per paper's convention: B_{(ρ,i,j),g} = √(d/|G|)·ρ_{ji}(g)
                        basis[row_idx, col_idx] = norm * rho_g[j, i]
                    row_idx += 1

        basis_tensor = torch.tensor(basis, dtype=torch.complex128, device=device)
        self._basis_cache[device_key] = basis_tensor
        return basis_tensor

    def transform(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform on A_n."""
        basis = self.get_basis_matrix(signal.device)
        if not signal.is_complex():
            signal = signal.to(torch.complex128)
        return signal @ basis.T.conj()

    def inverse_transform(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply inverse Fourier transform on A_n.

        For unitary basis B with forward transform c = ν·B^H,
        the inverse is ν = c·B.
        """
        basis = self.get_basis_matrix(coeffs.device)
        return coeffs @ basis

    def get_irrep_block_indices(self, label: str) -> Tuple[int, int]:
        """Get the start and end indices for an irrep block in the basis."""
        start = 0
        for lbl in self._irrep_labels:
            if lbl == label:
                dim = self._irrep_dims[lbl]
                return start, start + dim * dim
            start += self._irrep_dims[lbl] ** 2
        raise ValueError(f"Unknown irrep label: {label}")

    def get_dominant_frequency(self, coeffs: torch.Tensor) -> int:
        """Find the dominant non-trivial frequency component."""
        magnitudes = torch.abs(coeffs[1:])
        return int(torch.argmax(magnitudes).item()) + 1

    def get_frequency_label(self, idx: int) -> str:
        """Get the label for a basis function by index."""
        curr_idx = 0
        for label in self._irrep_labels:
            dim = self._irrep_dims[label]
            block_size = dim * dim
            if idx < curr_idx + block_size:
                local_idx = idx - curr_idx
                i = local_idx // dim
                j = local_idx % dim
                return f"{label}({i+1},{j+1})"
            curr_idx += block_size
        return f"idx={idx}"

    def __repr__(self) -> str:
        dims = [f"{lbl}:{d}" for lbl, d in self._irrep_dims.items()]
        return f"AlternatingGroupBasis(n={self.n}, size={self._size}, irreps={{{', '.join(dims)}}})"
