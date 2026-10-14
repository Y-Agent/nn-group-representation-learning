"""Frobenius group of order 21 representation basis.

The Frobenius group G = C_7 ⋊ C_3 has 5 irreducible representations:
- Three 1D characters (from quotient G/C_7 ≅ C_3)
- Two 3D irreps forming a complex conjugate pair (non-self-conjugate)

This is a minimal example of a group with genuinely non-self-conjugate
irreps of dimension > 1.

Character table:
    Class:    e     y     y²    x     x²    x³    x⁴    x⁵    x⁶
    Size:     1     7     7     3     3     3     3     3     3

    χ_0:      1     1     1     1     1     1     1     1     1
    χ_1:      1     ω     ω²    1     1     1     1     1     1
    χ_2:      1     ω²    ω     1     1     1     1     1     1
    ρ:        3     0     0    ζ+ζ²+ζ⁴  ...  (sum of 7th roots)
    ρ̄:        3     0     0    ζ⁻¹+ζ⁻²+ζ⁻⁴  ...

where ω = e^(2πi/3) and ζ = e^(2πi/7).

Basis Convention (Euclidean-unitary)
------------------------------------
This module uses an ℓ²-orthonormal (Euclidean-unitary) basis:

    B_{(ρ,i,j), g} = √(d_ρ/|G|) · ρ_{ji}(g)

where:
- d_ρ is the dimension of irrep ρ
- |G| = 21 is the group order
- The index swap (ji instead of ij) aligns with DFT inversion formula

The basis matrix B is unitary: B @ B^H = I and B^H @ B = I.

Transforms:
    Forward:  c = ν @ B^H     (coefficients from signal)
    Inverse:  ν = c @ B       (signal from coefficients)

This preserves Euclidean norms: ||c||² = ||ν||².

Relation to Paper's L²(G) Convention
------------------------------------
The paper defines orthonormality w.r.t. L²(G) inner product:
    ⟨f,h⟩_{L²} = (1/|G|) Σ_g f(g) h̄(g)

Under this inner product, {√d_ρ · ρ_{ij}(·)} is orthonormal.

Our basis rows have L²(G) norm 1/√|G| (not 1), since we use √(d/|G|) scaling.

Coefficient Conversion
----------------------
If `coeffs = transform(nu)` and you reshape the ρ-block to C[ρ] ∈ ℂ^{d×d}:

    C[ρ]_{ij} = √(|G| · d_ρ) · ν̂[ρ]_{ij}

where ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g⁻¹) is the paper's DFT definition.

Equivalently: ν̂[ρ] = C[ρ] / √(|G| · d_ρ)
"""

from typing import List, Dict, Tuple
import torch
import numpy as np
import cmath
from math import sqrt

from .base import Basis


class Frobenius21Basis(Basis):
    """Fourier basis for the Frobenius group of order 21.

    This basis is orthonormal w.r.t. the Euclidean inner product (Σ_g),
    making the basis matrix B unitary: B @ B^H = B^H @ B = I.

    Basis definition:
        B_{(ρ,i,j), g} = √(d_ρ/|G|) · ρ_{ji}(g)

    Irreps:
    - trivial (1D): ρ(g) = 1
    - omega (1D): ρ(a,b) = ω^k where a = 2^k, ω = e^(2πi/3)
    - omega_conj (1D): conjugate of omega
    - rho_3 (3D): constructed from generators
    - rho_3_conj (3D): complex conjugate of rho_3

    To convert coefficients to paper's ν̂[ρ]:
        ν̂[ρ] = C[ρ] / √(|G| · d_ρ)
    where C[ρ] is the d×d matrix reshaped from the coefficient block.
    """

    def __init__(self):
        """Initialize Frobenius group basis."""
        self._size = 21

        # Valid a-values and their "log base 2" mod 7
        # 2^0 = 1, 2^1 = 2, 2^2 = 4 (mod 7)
        self._a_values = [1, 2, 4]
        self._a_to_k = {1: 0, 2: 1, 4: 2}  # a = 2^k mod 7
        self._a_inv = {1: 1, 2: 4, 4: 2}   # modular inverse

        # Generate all elements
        self._elements: List[Tuple[int, int]] = []
        for a in self._a_values:
            for b in range(7):
                self._elements.append((a, b))

        self._elem_to_idx = {elem: i for i, elem in enumerate(self._elements)}

        # Build irrep information
        # Order: triv, rho_1, rho_3, rho_3^vee, rho_1^vee
        self._irrep_labels = ['trivial', 'omega', 'rho_3', 'rho_3_conj', 'omega_conj']
        self._irrep_dims = {
            'trivial': 1,
            'omega': 1,
            'omega_conj': 1,
            'rho_3': 3,
            'rho_3_conj': 3
        }

        # Build all representations
        self._irreps = self._build_representations()

        # Verify dimension sum
        dim_check = sum(d**2 for d in self._irrep_dims.values())
        assert dim_check == self._size, f"Dimension check failed: {dim_check} != {self._size}"

        # Cache for basis matrix
        self._basis_cache = {}

    def _multiply(self, g: Tuple[int, int], h: Tuple[int, int]) -> Tuple[int, int]:
        """Multiply two group elements."""
        a, b = g
        a_prime, b_prime = h
        return ((a * a_prime) % 7, (b + a * b_prime) % 7)

    def _inverse(self, g: Tuple[int, int]) -> Tuple[int, int]:
        """Compute inverse of a group element."""
        a, b = g
        a_inv = self._a_inv[a]
        return (a_inv, (-a_inv * b) % 7)

    def _build_representations(self) -> Dict[str, Dict[Tuple[int, int], np.ndarray]]:
        """Build all irreducible representation matrices."""
        irreps = {}

        # Roots of unity
        omega = cmath.exp(2j * cmath.pi / 3)  # primitive 3rd root
        zeta = cmath.exp(2j * cmath.pi / 7)   # primitive 7th root

        # ===== 1D representations =====

        # Trivial representation
        irreps['trivial'] = {g: np.array([[1.0 + 0j]]) for g in self._elements}

        # Omega representation: (a, b) → ω^k where a = 2^k
        irreps['omega'] = {}
        irreps['omega_conj'] = {}
        for g in self._elements:
            a, b = g
            k = self._a_to_k[a]
            irreps['omega'][g] = np.array([[omega ** k]])
            irreps['omega_conj'][g] = np.array([[omega ** (-k)]])

        # ===== 3D representations =====
        # Generators: x = (1, 1), y = (2, 0)
        #
        # We need ρ(y)·ρ(x)·ρ(y)^(-1) = ρ(x²) (the conjugation relation)
        # With ρ(y) being the cycle permutation that maps diag(a,b,c) → diag(c,a,b),
        # we need diagonal entries (ζ^a, ζ^b, ζ^c) such that c=2a, a=2b, b=2c (mod 7).
        # Solution: (a,b,c) = (1, 4, 2) since 2=2·1, 1=2·4≡8≡1, 4=2·2.
        #
        # ρ(x) = diag(ζ, ζ⁴, ζ²)
        # ρ(y) = permutation matrix for cycle (1 2 3)

        # Generator matrices for rho_3
        rho_x = np.diag([zeta, zeta**4, zeta**2])
        rho_y = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=complex)

        # Build all 21 matrices by expressing each element in terms of generators
        # (a, b) = x^b · y^k where a = 2^k
        # Verification: x^b · y^k = (1, b) · (2^k, 0) = (2^k, b)
        # This decomposition gives a homomorphism: ρ(g·h) = ρ(g)·ρ(h)
        irreps['rho_3'] = {}
        irreps['rho_3_conj'] = {}

        for g in self._elements:
            a, b = g
            k = self._a_to_k[a]

            # ρ(x^b) = ρ(x)^b = diag(ζ^b, ζ^(4b), ζ^(2b))
            rho_xb = np.diag([zeta**b, zeta**(4*b), zeta**(2*b)])

            # ρ(y^k) = ρ(y)^k
            rho_yk = np.linalg.matrix_power(rho_y, k)

            # ρ(a, b) = ρ(x^b) · ρ(y^k)
            rho_g = rho_xb @ rho_yk
            irreps['rho_3'][g] = rho_g

            # Conjugate representation
            irreps['rho_3_conj'][g] = rho_g.conj()

        # Verify the group homomorphism property for a few elements
        self._verify_homomorphism(irreps)

        return irreps

    def _verify_homomorphism(self, irreps: Dict) -> None:
        """Verify that representations satisfy ρ(g)ρ(h) = ρ(gh)."""
        # Check a few random products
        test_pairs = [
            ((1, 1), (2, 0)),  # x · y
            ((2, 0), (1, 1)),  # y · x
            ((2, 3), (4, 5)),
            ((4, 2), (2, 6)),
        ]

        for label in ['rho_3', 'rho_3_conj']:
            for g, h in test_pairs:
                gh = self._multiply(g, h)
                product = irreps[label][g] @ irreps[label][h]
                expected = irreps[label][gh]
                err = np.max(np.abs(product - expected))
                assert err < 1e-10, f"Homomorphism failed for {label}: {g}·{h}, error={err}"

    @property
    def name(self) -> str:
        return "frobenius21"

    @property
    def size(self) -> int:
        return self._size

    @property
    def num_frequencies(self) -> int:
        return self._size - 1

    @property
    def irrep_labels(self) -> List[str]:
        return self._irrep_labels

    @property
    def irrep_dimensions(self) -> Dict[str, int]:
        return self._irrep_dims

    @property
    def elements(self) -> List[Tuple[int, int]]:
        return self._elements

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
        """Get the Euclidean-unitary Fourier basis matrix.

        Basis definition:
            B_{(ρ,i,j), g} = √(d_ρ/|G|) · ρ_{ji}(g)

        This basis is orthonormal under Euclidean inner product:
            B @ B^H = I  and  B^H @ B = I

        The (ji) index swap aligns with the Fourier inversion formula:
            ν(g) = Σ_{ρ,i,j} d_ρ · ν̂[ρ]_{ij} · ρ_{ji}(g)

        Returns:
            Complex tensor of shape (|G|, |G|) where rows are basis functions.
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
            norm = np.sqrt(dim / N)  # √(d_ρ/|G|) for Euclidean unitarity

            for i in range(dim):
                for j in range(dim):
                    for col_idx, g in enumerate(self._elements):
                        rho_g = self._irreps[label][g]
                        # Use ρ_{ji}(g) for alignment with inversion formula
                        basis[row_idx, col_idx] = norm * rho_g[j, i]
                    row_idx += 1

        basis_tensor = torch.tensor(basis, dtype=torch.complex128, device=device)
        self._basis_cache[device_key] = basis_tensor
        return basis_tensor

    def transform(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform: c = ν @ B^H.

        For unitary B, this preserves Euclidean norm: ||c||² = ||ν||².

        Args:
            signal: Real or complex tensor of shape (..., |G|)

        Returns:
            Complex tensor of shape (..., |G|) containing Euclidean coefficients C.
            Reshape ρ-block to d×d matrix C[ρ], then ν̂[ρ] = C[ρ] / √(|G|·d_ρ).
        """
        basis = self.get_basis_matrix(signal.device)
        if not signal.is_complex():
            signal = signal.to(torch.complex128)
        return signal @ basis.T.conj()

    def transform_paper(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Fourier transform with paper's L²(G) scaling.

        Returns coefficients ν̂[ρ] matching paper's DFT definition:
            ν̂[ρ] = (1/|G|) Σ_g ν(g) ρ(g⁻¹)

        These satisfy Plancherel: Σ_ρ d_ρ ||ν̂[ρ]||²_F = ||ν||²_{L²(G)} = (1/|G|)||ν||²

        Args:
            signal: Real or complex tensor of shape (..., |G|)

        Returns:
            Complex tensor of shape (..., |G|) containing paper-notation coefficients.
            Each ρ-block is ν̂[ρ] = C[ρ] / √(|G|·d_ρ).
        """
        # Get Euclidean coefficients
        C = self.transform(signal)

        # Convert to paper notation: ν̂[ρ] = C[ρ] / √(|G|·d_ρ)
        nu_hat = C.clone()
        N = self._size
        curr_idx = 0
        for label in self._irrep_labels:
            d = self._irrep_dims[label]
            block_size = d * d
            scale = np.sqrt(N * d)
            nu_hat[..., curr_idx:curr_idx + block_size] /= scale
            curr_idx += block_size

        return nu_hat

    def inverse_transform(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply inverse Fourier transform: ν = c @ B.

        For unitary B, this is the exact inverse of transform().

        Args:
            coeffs: Complex tensor of shape (..., |G|)

        Returns:
            Complex tensor of shape (..., |G|)
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

    def check_basis(self, verbose: bool = True) -> bool:
        """Verify basis correctness.

        Checks:
        (a) Euclidean unitarity: B @ B^H ≈ I and B^H @ B ≈ I
        (b) Schur block orthogonality: cross-irrep inner products ≈ 0
        (c) Delta function roundtrip: transform → inverse_transform exact

        Args:
            verbose: If True, print detailed results.

        Returns:
            True if all checks pass, False otherwise.
        """
        B = self.get_basis_matrix().numpy()
        N = self._size
        all_passed = True

        # (a) Euclidean unitarity
        BBH = B @ B.conj().T
        BHB = B.conj().T @ B
        unitarity_err_1 = np.max(np.abs(BBH - np.eye(N)))
        unitarity_err_2 = np.max(np.abs(BHB - np.eye(N)))
        unitarity_ok = unitarity_err_1 < 1e-12 and unitarity_err_2 < 1e-12

        if verbose:
            print("(a) Euclidean unitarity:")
            print(f"    ||B @ B^H - I||_max = {unitarity_err_1:.2e}")
            print(f"    ||B^H @ B - I||_max = {unitarity_err_2:.2e}")
            print(f"    PASS: {unitarity_ok}")

        all_passed = all_passed and unitarity_ok

        # (b) Schur block orthogonality
        # Check that different irreps are orthogonal
        schur_ok = True
        max_cross_err = 0.0
        for label1 in self._irrep_labels:
            start1, end1 = self.get_irrep_block_indices(label1)
            for label2 in self._irrep_labels:
                if label1 >= label2:
                    continue
                start2, end2 = self.get_irrep_block_indices(label2)
                # Cross inner product (should be 0)
                cross = B[start1:end1, :] @ B[start2:end2, :].conj().T
                err = np.max(np.abs(cross))
                max_cross_err = max(max_cross_err, err)
                if err > 1e-12:
                    schur_ok = False

        # Check within-block norms
        max_norm_err = 0.0
        for label in self._irrep_labels:
            start, end = self.get_irrep_block_indices(label)
            for k in range(start, end):
                norm_sq = np.sum(np.abs(B[k, :])**2)
                err = abs(norm_sq - 1.0)
                max_norm_err = max(max_norm_err, err)
                if err > 1e-12:
                    schur_ok = False

        if verbose:
            print("(b) Schur block orthogonality:")
            print(f"    Max cross-irrep inner product: {max_cross_err:.2e}")
            print(f"    Max |row norm² - 1|: {max_norm_err:.2e}")
            print(f"    PASS: {schur_ok}")

        all_passed = all_passed and schur_ok

        # (c) Delta function roundtrip
        delta_ok = True
        max_roundtrip_err = 0.0
        for g_idx in [0, 5, 10, 20]:  # Test a few delta functions
            nu = np.zeros(N)
            nu[g_idx] = 1.0
            nu_torch = torch.tensor(nu, dtype=torch.float64)
            coeffs = self.transform(nu_torch)
            nu_back = self.inverse_transform(coeffs).numpy().real
            err = np.max(np.abs(nu - nu_back))
            max_roundtrip_err = max(max_roundtrip_err, err)
            if err > 1e-12:
                delta_ok = False

        if verbose:
            print("(c) Delta function roundtrip:")
            print(f"    Max roundtrip error: {max_roundtrip_err:.2e}")
            print(f"    PASS: {delta_ok}")

        all_passed = all_passed and delta_ok

        if verbose:
            print(f"\nAll checks passed: {all_passed}")

        return all_passed

    def __repr__(self) -> str:
        dims = [f"{lbl}:{d}" for lbl, d in self._irrep_dims.items()]
        return f"Frobenius21Basis(size={self._size}, irreps={{{', '.join(dims)}}})"
