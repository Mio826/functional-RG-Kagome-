
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]


def _normalize_spin(spin: SpinLike) -> str:
    if isinstance(spin, str):
        s = spin.strip().lower()
        if s in {"up", "u", "+", "+1", "spin_up", "↑"}:
            return "up"
        if s in {"down", "dn", "d", "-", "-1", "spin_down", "↓"}:
            return "dn"
    elif isinstance(spin, (int, np.integer)):
        return "up" if int(spin) > 0 else "dn"
    raise ValueError(f"Unsupported spin label: {spin!r}")


def fermi(x: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("Temperature T must be positive.")
    z = np.clip(x / T, -700.0, 700.0)
    return 1.0 / (np.exp(z) + 1.0)


def _bubble_pp_general(x1: np.ndarray, x2: np.ndarray, T: float, tol: float = 1e-12) -> np.ndarray:
    """
    Local particle-particle Matsubara sum:
        B_pp = [1 - f(x1) - f(x2)] / (x1 + x2)

    with the equal-denominator limit handled analytically as
        d/dx [1 - f(x) - f(C-x)] at x=x1, with C=x1+x2.
    In the common symmetric case x2 = -x1 this reduces to
        tanh(x1/2T) / (2 x1), with the x1 -> 0 limit 1/(4T).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    den = x1 + x2
    num = 1.0 - fermi(x1, T) - fermi(x2, T)
    out = np.empty_like(den, dtype=float)

    mask = np.abs(den) > tol
    out[mask] = num[mask] / den[mask]

    if np.any(~mask):
        C = den[~mask]
        x = x1[~mask]
        h = 1e-7 * np.maximum(1.0, np.abs(x))
        def N(y):
            return 1.0 - fermi(y, T) - fermi(C - y, T)
        out[~mask] = (N(x + h) - N(x - h)) / (2.0 * h)
    return out


def _bubble_ph_general(x1: np.ndarray, x2: np.ndarray, T: float, tol: float = 1e-12) -> np.ndarray:
    """
    Local particle-hole Matsubara sum:
        B_ph = [f(x1) - f(x2)] / (x1 - x2)
    with equal-energy limit
        B_ph(x,x) = -f'(x) = 1/(4T) sech^2(x/2T).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    den = x1 - x2
    num = fermi(x1, T) - fermi(x2, T)
    out = np.empty_like(den, dtype=float)

    mask = np.abs(den) > tol
    out[mask] = num[mask] / den[mask]

    if np.any(~mask):
        x = x1[~mask]
        z = np.clip(x / (2.0 * T), -350.0, 350.0)
        out[~mask] = 0.25 / T / np.cosh(z) ** 2
    return out


@dataclass
class LoopBubble:
    """
    Patch-diagonal loop bubble for one channel at fixed transfer momentum Q.

    The weight vector `weights[p]` is the diagonal entry associated with the
    internal patch label p. In a patch FRG update one uses it schematically as

        dGamma_channel ~ K_left @ diag(weights) @ K_right

    where K_left/right are channel kernels built from the running vertex.
    """
    name: str
    Q: np.ndarray
    temperature: float
    weights: np.ndarray
    partner_patches: np.ndarray
    residuals: np.ndarray
    arc_lengths: np.ndarray
    vf_norms: np.ndarray
    radial_cutoff: float
    n_radial: int

    @property
    def Npatch(self) -> int:
        return int(self.weights.shape[0])

    def as_diagonal_matrix(self) -> np.ndarray:
        return np.diag(np.asarray(self.weights, dtype=float))

    def summary(self) -> Dict[str, float]:
        w = np.asarray(self.weights, dtype=float)
        return {
            "Npatch": float(self.Npatch),
            "temperature": float(self.temperature),
            "min_weight": float(np.min(w)),
            "max_weight": float(np.max(w)),
            "mean_weight": float(np.mean(w)),
            "max_partner_residual": float(np.max(self.residuals)),
            "mean_partner_residual": float(np.mean(self.residuals)),
        }


class TemperatureLoopKernel:
    r"""
    Build temperature-flow patch loop bubbles for pp / ph channels.

    Design goals
    ------------
    1) Keep this module independent of the interaction vertex itself.
    2) Make the temperature dependence explicit at the loop level.
    3) Return patch-diagonal loop objects that can be contracted with the
       channel kernels built by `channels.py`.

    Physical approximation
    ----------------------
    We use a simple Fermi-surface patch approximation:

      - each patch p carries a Fermi momentum k_p and an estimate of |v_F(p)|,
      - the tangential patch measure is estimated from neighboring patch spacing,
      - the energy dependence normal to the Fermi surface is linearized by a
        radial variable ell in [-Lambda_r, +Lambda_r].

    For each internal patch p and its shifted partner p_Q, we evaluate a local
    bubble B(ell; T) and integrate over ell numerically, then multiply by the
    patch measure ds_p / (2π)^2.

    This is intentionally a minimal kernel prototype for benchmark / plumbing.
    It is not yet the final production-quality FRG loop evaluator.
    """

    def __init__(
        self,
        patchsets: PatchSetMap,
        *,
        radial_cutoff: float = 1.0,
        n_radial: int = 801,
        dT_fraction: float = 1e-3,
        custom_patch_weights: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        self.patchsets = patchsets
        self.radial_cutoff = float(radial_cutoff)
        self.n_radial = int(n_radial)
        self.dT_fraction = float(dT_fraction)
        self.custom_patch_weights = custom_patch_weights or {}

        if self.radial_cutoff <= 0:
            raise ValueError("radial_cutoff must be positive.")
        if self.n_radial < 11 or self.n_radial % 2 == 0:
            raise ValueError("n_radial must be an odd integer >= 11.")
        if self.dT_fraction <= 0:
            raise ValueError("dT_fraction must be positive.")

    def _patchset_for_spin(self, spin: SpinLike):
        s = _normalize_spin(spin)
        if s not in self.patchsets:
            raise ValueError(f"spin '{s}' not present in patchsets.")
        ps = self.patchsets[s]
        if ps is None or not hasattr(ps, "patches"):
            raise ValueError(f"patchset for spin '{s}' is invalid.")
        if len(ps.patches) == 0:
            raise ValueError(f"patchset for spin '{s}' is empty.")
        return ps

    @staticmethod
    def _minimum_image_displacement(k_target, k_ref, b1, b2):
        k_target = np.asarray(k_target, dtype=float)
        k_ref = np.asarray(k_ref, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        b2 = np.asarray(b2, dtype=float)

        best = None
        best_norm = np.inf
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                disp = k_target - (k_ref + n1 * b1 + n2 * b2)
                norm = float(np.linalg.norm(disp))
                if norm < best_norm:
                    best_norm = norm
                    best = disp
        return best

    def find_shifted_patch_index(self, spin: SpinLike, k_target: np.ndarray):
        ps = self._patchset_for_spin(spin)
        ks = np.array([np.asarray(p.k_cart, dtype=float) for p in ps.patches], dtype=float)
        dists = []
        for k_ref in ks:
            disp = self._minimum_image_displacement(k_target, k_ref, ps.b1, ps.b2)
            dists.append(np.linalg.norm(disp))
        dists = np.asarray(dists, dtype=float)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def shifted_patch_map(self, spin: SpinLike, Q: Sequence[float], *, mode: str):
        Q = np.asarray(Q, dtype=float)
        ps = self._patchset_for_spin(spin)
        idxs = np.zeros(ps.Npatch, dtype=int)
        residuals = np.zeros(ps.Npatch, dtype=float)
        for p, patch in enumerate(ps.patches):
            k = np.asarray(patch.k_cart, dtype=float)
            if mode == "Q_minus_k":
                target = Q - k
            elif mode == "k_plus_Q":
                target = k + Q
            elif mode == "k_minus_Q":
                target = k - Q
            else:
                raise ValueError("mode must be one of {'Q_minus_k', 'k_plus_Q', 'k_minus_Q'}.")
            idx, dist = self.find_shifted_patch_index(spin, target)
            idxs[p] = idx
            residuals[p] = dist
        return idxs, residuals

    def _vf_norms(self, spin: SpinLike) -> np.ndarray:
        ps = self._patchset_for_spin(spin)
        out = np.zeros(ps.Npatch, dtype=float)
        for i, p in enumerate(ps.patches):
            candidates = []
            for name in ("vF", "vf", "vF_cart", "velocity", "vel"):
                if hasattr(p, name):
                    v = np.asarray(getattr(p, name), dtype=float).reshape(-1)
                    if v.size == 2:
                        candidates.append(float(np.linalg.norm(v)))
                    elif v.size == 1:
                        candidates.append(abs(float(v[0])))
            if not candidates:
                raise ValueError(
                    f"Patch {i} in spin sector '{_normalize_spin(spin)}' does not expose a usable Fermi-velocity estimate."
                )
            out[i] = max(candidates)
        out = np.maximum(out, 1e-12)
        return out

    def _arc_lengths(self, spin: SpinLike) -> np.ndarray:
        s = _normalize_spin(spin)
        if s in self.custom_patch_weights:
            w = np.asarray(self.custom_patch_weights[s], dtype=float)
            ps = self._patchset_for_spin(s)
            if w.shape != (ps.Npatch,):
                raise ValueError(f"custom_patch_weights['{s}'] has shape {w.shape}, expected {(ps.Npatch,)}.")
            return np.maximum(w, 1e-12)

        ps = self._patchset_for_spin(s)
        ks = np.array([np.asarray(p.k_cart, dtype=float) for p in ps.patches], dtype=float)
        N = ks.shape[0]
        out = np.zeros(N, dtype=float)
        for i in range(N):
            im = (i - 1) % N
            ip = (i + 1) % N
            d1 = self._minimum_image_displacement(ks[i], ks[im], ps.b1, ps.b2)
            d2 = self._minimum_image_displacement(ks[ip], ks[i], ps.b1, ps.b2)
            out[i] = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2))
        return np.maximum(out, 1e-12)

    def _measure_prefactor(self, spin: SpinLike) -> np.ndarray:
        arc = self._arc_lengths(spin)
        vf = self._vf_norms(spin)
        return arc / vf / (2.0 * np.pi) ** 2

    def _radial_grid(self) -> np.ndarray:
        return np.linspace(-self.radial_cutoff, self.radial_cutoff, self.n_radial, dtype=float)

    def _integrated_bubble(self, kind: str, v1: float, v2: float, T: float) -> float:
        ell = self._radial_grid()
        if kind == "pp":
            x1 = v1 * ell
            x2 = -v2 * ell
            vals = _bubble_pp_general(x1, x2, T)
        elif kind == "ph":
            x1 = v1 * ell
            x2 = v2 * ell
            vals = _bubble_ph_general(x1, x2, T)
        else:
            raise ValueError("kind must be 'pp' or 'ph'.")
        return float(np.trapezoid(vals, ell))

    def _dT_integrated_bubble(self, kind: str, v1: float, v2: float, T: float) -> float:
        dT = max(1e-8, self.dT_fraction * T)
        Tp = T + dT
        Tm = max(1e-8, T - dT)
        Bp = self._integrated_bubble(kind, v1, v2, Tp)
        Bm = self._integrated_bubble(kind, v1, v2, Tm)
        return float((Bp - Bm) / (Tp - Tm))

    def pp_loop(self, Q: Sequence[float], *, spin_pair: Tuple[SpinLike, SpinLike] = ("up", "dn"), T: float) -> LoopBubble:
        Q = np.asarray(Q, dtype=float)
        s1, s2 = map(_normalize_spin, spin_pair)
        ps1 = self._patchset_for_spin(s1)
        self._patchset_for_spin(s2)

        partner, residuals = self.shifted_patch_map(s2, Q, mode="Q_minus_k")
        vf1 = self._vf_norms(s1)
        vf2 = self._vf_norms(s2)
        measure = self._measure_prefactor(s1)

        weights = np.zeros(ps1.Npatch, dtype=float)
        for p in range(ps1.Npatch):
            p2 = int(partner[p])
            weights[p] = measure[p] * self._dT_integrated_bubble("pp", vf1[p], vf2[p2], T)

        return LoopBubble(
            name="pp",
            Q=Q,
            temperature=float(T),
            weights=weights,
            partner_patches=np.asarray(partner, dtype=int),
            residuals=np.asarray(residuals, dtype=float),
            arc_lengths=self._arc_lengths(s1),
            vf_norms=vf1,
            radial_cutoff=self.radial_cutoff,
            n_radial=self.n_radial,
        )

    def ph_direct_loop(self, Q: Sequence[float], *, spin_pair: Tuple[SpinLike, SpinLike] = ("up", "up"), T: float) -> LoopBubble:
        Q = np.asarray(Q, dtype=float)
        s1, s3 = map(_normalize_spin, spin_pair)
        ps1 = self._patchset_for_spin(s1)
        self._patchset_for_spin(s3)

        partner, residuals = self.shifted_patch_map(s3, Q, mode="k_plus_Q")
        vf1 = self._vf_norms(s1)
        vf3 = self._vf_norms(s3)
        measure = self._measure_prefactor(s1)

        weights = np.zeros(ps1.Npatch, dtype=float)
        for p in range(ps1.Npatch):
            p3 = int(partner[p])
            weights[p] = measure[p] * self._dT_integrated_bubble("ph", vf1[p], vf3[p3], T)

        return LoopBubble(
            name="ph_direct",
            Q=Q,
            temperature=float(T),
            weights=weights,
            partner_patches=np.asarray(partner, dtype=int),
            residuals=np.asarray(residuals, dtype=float),
            arc_lengths=self._arc_lengths(s1),
            vf_norms=vf1,
            radial_cutoff=self.radial_cutoff,
            n_radial=self.n_radial,
        )

    def ph_crossed_loop(self, Q: Sequence[float], *, spin_pair: Tuple[SpinLike, SpinLike] = ("up", "up"), T: float) -> LoopBubble:
        bubble = self.ph_direct_loop(Q, spin_pair=spin_pair, T=T)
        bubble.name = "ph_crossed"
        return bubble

    def build_basic_loops(self, Q: Sequence[float], *, T: float) -> Dict[str, LoopBubble]:
        out: Dict[str, LoopBubble] = {}
        for key, fn, spins in [
            ("pp_ud", self.pp_loop, ("up", "dn")),
            ("phd_uu", self.ph_direct_loop, ("up", "up")),
            ("phd_dd", self.ph_direct_loop, ("dn", "dn")),
            ("phc_uu", self.ph_crossed_loop, ("up", "up")),
            ("phc_dd", self.ph_crossed_loop, ("dn", "dn")),
        ]:
            try:
                out[key] = fn(Q, spin_pair=spins, T=T)
            except ValueError:
                pass
        if len(out) == 0:
            raise ValueError("No valid loop bubbles can be built from the current patchsets.")
        return out


__all__ = [
    "LoopBubble",
    "TemperatureLoopKernel",
    "fermi",
]
