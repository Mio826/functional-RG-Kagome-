import numpy as np
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]


@dataclass
class BareExtendedHubbard:
    r"""
    Bare extended-Hubbard interaction for a spin-conserving kagome-like model.

    Physical setting
    ----------------
    The one-body Hamiltonian may depend on spin, but it must not mix spin-up and
    spin-down sectors. Typical examples are intrinsic SOC, opposite flux for
    opposite spins, or Zeeman-like splittings. In that situation one usually
    constructs separate patch sets for the two spin sectors.

    The interaction is the usual density-density form

        H_int = U \sum_{R,a} n_{R a \uparrow} n_{R a \downarrow}
              + V \sum_{<R a, R' b>} n_{R a} n_{R' b}.

    Important consequence for spin scattering
    -----------------------------------------
    Because the interaction is density-density, it does *not* flip spin. The
    allowed bare process is therefore spin-conserving along each fermion line,

        s3 = s1,   s4 = s2.

    However, that does *not* mean one should keep only opposite-spin scattering:

      - onsite U contributes only for opposite spins,
      - nearest-neighbor V contributes for both opposite-spin and same-spin
        scattering, because n_i n_j contains all spin combinations.

    Vertex conventions
    ------------------
    External labels are ordered as

        (k1, u1, s1), (k2, u2, s2) -> (k3, u3, s3), (k4, u4, s4),

    where each u_i is a three-component Bloch eigenvector in the kagome orbital
    basis.

    Two related vertices are exposed:

      1. direct_band_vertex(...)
         The raw density-density matrix element V_dir(1,2 -> 3,4).

      2. antisym_band_vertex(...)
         The fermionic antisymmetrized vertex

             Gamma(1,2 -> 3,4) = V_dir(1,2 -> 3,4) - V_dir(1,2 -> 4,3).

         This is the object that should be passed to channel decomposition and
         later FRG flow equations.
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
                    "This class assumes kagome nearest-neighbor bond vectors."
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
        Direct orbital-space kernel W_ab(q) for

            (k1,a,s1), (k2,b,s2) -> (k3,a,s3), (k4,b,s4),

        with q = k3 - k1.

        This is the *direct* density-density matrix element. It is intentionally
        not antisymmetrized under exchange of outgoing legs.
        """
        s1 = self.normalize_spin(s1)
        s2 = self.normalize_spin(s2)
        s3 = self.normalize_spin(s3)
        s4 = self.normalize_spin(s4)

        W = np.zeros((self.Norb, self.Norb), dtype=complex)
        if not (s3 == s1 and s4 == s2):
            return W

        qab, qac, qbc = self._nn_form_factors(q)

        if s1 != s2:
            np.fill_diagonal(W, self.U)

        W[0, 1] = W[1, 0] = self.V * qab
        W[0, 2] = W[2, 0] = self.V * qac
        W[1, 2] = W[2, 1] = self.V * qbc
        return W

    def _check_band_vectors(self, *vecs: np.ndarray) -> None:
        for idx, u in enumerate(vecs, start=1):
            u = np.asarray(u)
            if u.ndim != 1 or u.shape[0] != self.Norb:
                raise ValueError(
                    f"u{idx} must be a length-{self.Norb} kagome eigenvector, got shape {u.shape}."
                )

    @staticmethod
    def _check_momentum_conservation(
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        k4: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        tol: float = 1e-7,
    ) -> None:
        G = np.asarray(k1) + np.asarray(k2) - np.asarray(k3) - np.asarray(k4)
        B = np.column_stack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)])
        coeff = np.linalg.solve(B, G)
        if not np.allclose(coeff, np.round(coeff), atol=tol):
            raise ValueError(
                "Momentum conservation violated: k1+k2-k3-k4 is not a reciprocal lattice vector."
            )

    def direct_band_vertex(
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
        Direct projected bare matrix element

            V_dir(k1,s1; k2,s2 -> k3,s3; k4,s4)

        defined by

            V_dir = sum_{a,b} u3_a^* u4_b^* W_ab(k3-k1) u2_b u1_a.
        """
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k3 = np.asarray(k3, dtype=float)
        k4 = np.asarray(k4, dtype=float)
        u1 = np.asarray(u1, dtype=complex)
        u2 = np.asarray(u2, dtype=complex)
        u3 = np.asarray(u3, dtype=complex)
        u4 = np.asarray(u4, dtype=complex)

        self._check_band_vectors(u1, u2, u3, u4)

        if check_momentum:
            if b1 is None or b2 is None:
                raise ValueError("b1 and b2 must be provided when check_momentum=True.")
            self._check_momentum_conservation(k1, k2, k3, k4, b1, b2, tol=tol)

        q = k3 - k1
        W = self.orbital_interaction_matrix(q, s1, s2, s3, s4)
        return np.einsum("a,b,ab,b,a->", np.conjugate(u3), np.conjugate(u4), W, u2, u1)

    def antisym_band_vertex(
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
        Fermionic antisymmetrized vertex

            Gamma(1,2 -> 3,4) = V_dir(1,2 -> 3,4) - V_dir(1,2 -> 4,3).

        With the present density-density direct matrix element, this is the
        natural vertex to use in Cooper / particle-hole channel analysis.
        """
        if check_momentum:
            if b1 is None or b2 is None:
                raise ValueError("b1 and b2 must be provided when check_momentum=True.")
            self._check_momentum_conservation(k1, k2, k3, k4, b1, b2, tol=tol)

        direct = self.direct_band_vertex(
            k1, u1, s1,
            k2, u2, s2,
            k3, u3, s3,
            k4, u4, s4,
            check_momentum=False,
        )
        exchange = self.direct_band_vertex(
            k1, u1, s1,
            k2, u2, s2,
            k4, u4, s4,
            k3, u3, s3,
            check_momentum=False,
        )
        return direct - exchange

    def band_vertex(self, *args, antisym: bool = False, **kwargs) -> complex:
        r"""Backward-compatible wrapper.

        By default this returns the antisymmetrized fermionic vertex.
        Set ``antisym=False`` to obtain the raw direct matrix element instead.
        """
        if antisym:
            return self.antisym_band_vertex(*args, **kwargs)
        return self.direct_band_vertex(*args, **kwargs)

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
        antisym: bool = False,
        check_momentum: bool = False,
    ) -> complex:
        r"""
        Vertex evaluated on four patch representatives.

        Parameters
        ----------
        antisym : bool, default True
            Whether to return the fermionic antisymmetrized vertex Gamma or the
            raw direct matrix element V_dir.
        """
        PS1 = self._patchset_for_spin(patchsets, s1)
        PS2 = self._patchset_for_spin(patchsets, s2)
        PS3 = self._patchset_for_spin(patchsets, s3)
        PS4 = self._patchset_for_spin(patchsets, s4)

        P1 = PS1.patches[p1]
        P2 = PS2.patches[p2]
        P3 = PS3.patches[p3]
        P4 = PS4.patches[p4]

        return self.band_vertex(
            P1.k_cart, P1.eigvec, s1,
            P2.k_cart, P2.eigvec, s2,
            P3.k_cart, P3.eigvec, s3,
            P4.k_cart, P4.eigvec, s4,
            antisym=antisym,
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
        antisym: bool = False,
        enforce_momentum: bool = False,
    ) -> np.ndarray:
        r"""
        Build the full patch vertex tensor for a fixed spin sector.

        This returns either

            V[p1,p2,p3,p4]

        or

            Gamma[p1,p2,p3,p4],

        depending on ``antisym``.

        Notes
        -----
        This scales as O(Npatch^4) and is intended for debugging or small patch
        numbers only.
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
                                antisym=antisym,
                                check_momentum=enforce_momentum,
                                b1=PS1.b1,
                                b2=PS1.b2,
                            )
                        except ValueError:
                            out[p1, p2, p3, p4] = 0.0
        return out
