import numpy as np
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]


@dataclass
class BareExtendedHubbard:
    r"""
    Bare extended Hubbard interaction for a *spin-conserving* kagome model.

    This class is meant for the common situation where the noninteracting
    Hamiltonian may depend on spin (for example Kane-Mele-type intrinsic SOC,
    Zeeman splitting, or opposite flux for opposite spins), but **does not mix
    spins**. In that case one can build separate patch sets for spin-up and
    spin-down sectors, while the interaction remains the usual density-density
    form

        H_int = U \sum_{R,a} n_{Ra\uparrow} n_{Ra\downarrow}
              + V \sum_{<Ra,R'b>} n_{Ra} n_{R'b}.

    Convention for the projected vertex:

        V(k1,s1; k2,s2 -> k3,s3; k4,s4)

    where the nonzero matrix elements for this density-density interaction are
    spin-conserving along each fermion line,

        s3 = s1,   s4 = s2,

    with:
      - onsite U contributing only when s1 != s2,
      - nearest-neighbor V contributing for both same-spin and opposite-spin
        scattering.

    Parameters
    ----------
    U : float
        Onsite Hubbard repulsion.
    V : float
        Nearest-neighbor density-density repulsion.
    delta1, delta2, delta3 : array-like, shape (2,)
        Kagome nearest-neighbor bond vectors for AB, AC, BC pairs.
    """

    U: float
    V: float
    delta1: np.ndarray
    delta2: np.ndarray
    delta3: np.ndarray

    @classmethod
    def from_kagome_model(cls, model, U: float, V: float):
        for name in ("delta1", "delta2", "delta3"):
            if not hasattr(model, name):
                raise AttributeError(
                    f"Model does not have attribute {name}. "
                    "This class currently assumes kagome-style delta vectors."
                )
        return cls(
            U=float(U),
            V=float(V),
            delta1=np.asarray(model.delta1, dtype=float),
            delta2=np.asarray(model.delta2, dtype=float),
            delta3=np.asarray(model.delta3, dtype=float),
        )

    @property
    def Norb(self) -> int:
        return 3

    @staticmethod
    def normalize_spin(spin: SpinLike) -> str:
        if isinstance(spin, str):
            s = spin.strip().lower()
            if s in {"up", "u", "+", "+1", "spin_up", "↑"}:
                return "up"
            if s in {"down", "dn", "d", "-", "-1", "spin_down", "↓"}:
                return "dn"
        elif isinstance(spin, (int, np.integer)):
            if int(spin) > 0:
                return "up"
            if int(spin) < 0:
                return "dn"
        raise ValueError(f"Unsupported spin label: {spin!r}")

    def _nn_form_factors(self, q: np.ndarray) -> Tuple[float, float, float]:
        q = np.asarray(q, dtype=float)
        qab = 2.0 * np.cos(np.dot(q, self.delta1))
        qac = 2.0 * np.cos(np.dot(q, self.delta2))
        qbc = 2.0 * np.cos(np.dot(q, self.delta3))
        return qab, qac, qbc

    def orbital_interaction_matrix(
        self,
        q: np.ndarray,
        s1: SpinLike,
        s2: SpinLike,
        s3: SpinLike,
        s4: SpinLike,
    ) -> np.ndarray:
        r"""
        Return the 3x3 orbital-basis interaction kernel W_ab(q) for

            (k1,a,s1), (k2,b,s2) -> (k3,a,s3), (k4,b,s4),

        with q = k3 - k1.

        For the density-density interaction used here:

          1. Spin is conserved on each line, so nonzero only if
                 s3 = s1 and s4 = s2.

          2. Onsite U acts only for opposite spins, i.e. when s1 != s2.

          3. Nearest-neighbor V acts for both same-spin and opposite-spin
             scattering, because n_i n_j contains all spin combinations.
        """
        s1 = self.normalize_spin(s1)
        s2 = self.normalize_spin(s2)
        s3 = self.normalize_spin(s3)
        s4 = self.normalize_spin(s4)

        W = np.zeros((self.Norb, self.Norb), dtype=complex)
        if not (s3 == s1 and s4 == s2):
            return W

        qab, qac, qbc = self._nn_form_factors(q)

        # onsite U only for opposite spin
        if s1 != s2:
            np.fill_diagonal(W, self.U)

        # NN V for any spin combination
        W[0, 1] = W[1, 0] = self.V * qab
        W[0, 2] = W[2, 0] = self.V * qac
        W[1, 2] = W[2, 1] = self.V * qbc
        return W

    def band_vertex(
        self,
        k1: np.ndarray,
        u1: np.ndarray,
        s1: SpinLike,
        k2: np.ndarray,
        u2: np.ndarray,
        s2: SpinLike,
        k3: np.ndarray,
        u3: np.ndarray,
        s3: SpinLike,
        k4: np.ndarray,
        u4: np.ndarray,
        s4: SpinLike,
        *,
        check_momentum: bool = False,
        b1: Optional[np.ndarray] = None,
        b2: Optional[np.ndarray] = None,
        tol: float = 1e-7,
    ) -> complex:
        r"""
        Single-band projected bare vertex for a spin-conserving kagome model.

        The orbital-space kernel is first constructed as W_ab(q; s1,s2,s3,s4),
        then projected with the three-component Bloch eigenvectors of the
        corresponding spin sectors:

            V_band = \sum_{a,b} u3_a^* u4_b^* W_ab(q) u2_b u1_a.
        """
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k3 = np.asarray(k3, dtype=float)
        k4 = np.asarray(k4, dtype=float)
        u1 = np.asarray(u1, dtype=complex)
        u2 = np.asarray(u2, dtype=complex)
        u3 = np.asarray(u3, dtype=complex)
        u4 = np.asarray(u4, dtype=complex)

        for idx, u in enumerate((u1, u2, u3, u4), start=1):
            if u.ndim != 1 or u.shape[0] != self.Norb:
                raise ValueError(
                    f"u{idx} must be a length-{self.Norb} kagome eigenvector, got shape {u.shape}."
                )

        if check_momentum:
            if b1 is None or b2 is None:
                raise ValueError("b1 and b2 must be provided when check_momentum=True.")
            G = k1 + k2 - k3 - k4
            B = np.column_stack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)])
            coeff = np.linalg.solve(B, G)
            if not np.allclose(coeff, np.round(coeff), atol=tol):
                raise ValueError(
                    "Momentum conservation violated: k1+k2-k3-k4 is not a reciprocal lattice vector."
                )

        q = k3 - k1
        W = self.orbital_interaction_matrix(q, s1, s2, s3, s4)
        return np.einsum("a,b,ab,b,a->", np.conjugate(u3), np.conjugate(u4), W, u2, u1)

    def _patchset_for_spin(self, patchsets: PatchSetMap, spin: SpinLike):
        s = self.normalize_spin(spin)
        candidate_keys = [s]
        if s == "up":
            candidate_keys += ["u", +1]
        else:
            candidate_keys += ["dn", "down", "d", -1]
        for key in candidate_keys:
            if key in patchsets:
                return patchsets[key]
        raise KeyError(
            f"Could not find patch set for spin={spin!r}. Expected keys like 'up'/'dn' or +1/-1."
        )

    def patch_vertex(
        self,
        patchsets: PatchSetMap,
        p1: int,
        s1: SpinLike,
        p2: int,
        s2: SpinLike,
        p3: int,
        s3: SpinLike,
        p4: int,
        s4: SpinLike,
        *,
        check_momentum: bool = False,
    ) -> complex:
        r"""
        Evaluate the projected bare vertex on four patch representatives,
        where each external leg can belong to the spin-up or spin-down patch set.

        Parameters
        ----------
        patchsets : mapping
            Typically {"up": patchset_up, "dn": patchset_dn}.
        p1,s1,p2,s2,p3,s3,p4,s4 :
            Patch indices and corresponding spin labels for the four legs.
        """
        PS1 = self._patchset_for_spin(patchsets, s1)
        PS2 = self._patchset_for_spin(patchsets, s2)
        PS3 = self._patchset_for_spin(patchsets, s3)
        PS4 = self._patchset_for_spin(patchsets, s4)

        P1 = PS1.patches[p1]
        P2 = PS2.patches[p2]
        P3 = PS3.patches[p3]
        P4 = PS4.patches[p4]

        # For spin-conserving models considered here, the reciprocal lattice is
        # the same in both spin sectors. We read it from PS1.
        return self.band_vertex(
            P1.k_cart, P1.eigvec, s1,
            P2.k_cart, P2.eigvec, s2,
            P3.k_cart, P3.eigvec, s3,
            P4.k_cart, P4.eigvec, s4,
            check_momentum=check_momentum,
            b1=PS1.b1,
            b2=PS1.b2,
        )

    def patch_tensor(
        self,
        patchsets: PatchSetMap,
        s1: SpinLike,
        s2: SpinLike,
        s3: SpinLike,
        s4: SpinLike,
        *,
        enforce_momentum: bool = False,
    ) -> np.ndarray:
        r"""
        Build the full bare vertex tensor V[p1,p2,p3,p4] for a *fixed* choice of
        spin labels (s1,s2,s3,s4).

        Example:
            V_udud = interaction.patch_tensor(patchsets, 'up','dn','up','dn')
            V_uuuu = interaction.patch_tensor(patchsets, 'up','up','up','up')

        This scales as O(Npatch^4) and is intended only for debugging / small
        patch numbers.
        """
        PS1 = self._patchset_for_spin(patchsets, s1)
        PS2 = self._patchset_for_spin(patchsets, s2)
        PS3 = self._patchset_for_spin(patchsets, s3)
        PS4 = self._patchset_for_spin(patchsets, s4)
        N1, N2, N3, N4 = PS1.Npatch, PS2.Npatch, PS3.Npatch, PS4.Npatch
        out = np.zeros((N1, N2, N3, N4), dtype=complex)

        for p1 in range(N1):
            P1 = PS1.patches[p1]
            for p2 in range(N2):
                P2 = PS2.patches[p2]
                for p3 in range(N3):
                    P3 = PS3.patches[p3]
                    for p4 in range(N4):
                        P4 = PS4.patches[p4]
                        try:
                            out[p1, p2, p3, p4] = self.band_vertex(
                                P1.k_cart, P1.eigvec, s1,
                                P2.k_cart, P2.eigvec, s2,
                                P3.k_cart, P3.eigvec, s3,
                                P4.k_cart, P4.eigvec, s4,
                                check_momentum=enforce_momentum,
                                b1=PS1.b1,
                                b2=PS1.b2,
                            )
                        except ValueError:
                            out[p1, p2, p3, p4] = 0.0
        return out


def make_cooper_pairs(patchset) -> np.ndarray:
    r"""
    For each patch p, find the patch pbar whose representative reduced momentum
    is closest to -k_p (mod reciprocal lattice vectors).

    This is only a helper for quick bare-level pairing diagnostics.
    """
    kred = np.array([p.k_red for p in patchset.patches])
    pairs = np.zeros(patchset.Npatch, dtype=int)
    for p in range(patchset.Npatch):
        target = (-kred[p]) % 1.0
        d2 = np.sum((kred - target[None, :]) ** 2, axis=1)
        pairs[p] = int(np.argmin(d2))
    return pairs


def build_cooper_pairing_kernel(
    patchsets: PatchSetMap,
    interaction: BareExtendedHubbard,
    *,
    s1: SpinLike = "up",
    s2: SpinLike = "dn",
) -> np.ndarray:
    r"""
    Build a simple bare Cooper kernel

        K[p, p'] = V(p,s1; pbar,s2 -> p',s1; p'bar,s2),

    where pbar and p'bar approximate the opposite momenta in the corresponding
    spin sectors.

    For a spin-conserving Kane-Mele-type setup, the most common choice is the
    opposite-spin kernel (s1='up', s2='dn').
    """
    PS1 = interaction._patchset_for_spin(patchsets, s1)
    PS2 = interaction._patchset_for_spin(patchsets, s2)
    pairs1 = make_cooper_pairs(PS1)
    pairs2 = make_cooper_pairs(PS2)

    K = np.zeros((PS1.Npatch, PS1.Npatch), dtype=complex)
    for p in range(PS1.Npatch):
        pb = pairs2[pairs1[p] % PS2.Npatch] if PS1.Npatch != PS2.Npatch else pairs2[p]
        for pp in range(PS1.Npatch):
            ppb = pairs2[pairs1[pp] % PS2.Npatch] if PS1.Npatch != PS2.Npatch else pairs2[pp]
            K[p, pp] = interaction.patch_vertex(
                patchsets,
                p, s1,
                pb, s2,
                pp, s1,
                ppb, s2,
                check_momentum=False,
            )
    return K
