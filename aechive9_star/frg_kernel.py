from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
ShiftMap = Tuple[np.ndarray, np.ndarray]


# ============================================================================
# Basic config / utilities
# ============================================================================


@dataclass(frozen=True)
class FlowConfig:
    temperature: float
    nfreq: int = 256
    include_explicit_T_prefactor: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.nfreq <= 0:
            raise ValueError("nfreq must be positive")


# Minimal-Sz0 internal-cache format used by the cleaned-up solver.
MinimalInternalCache = Dict[str, np.ndarray]


def normalize_spin(spin: SpinLike) -> str:
    if isinstance(spin, str):
        s = spin.strip().lower()
        if s in {"up", "u", "+", "+1", "spin_up", "↑"}:
            return "up"
        if s in {"dn", "down", "d", "-", "-1", "spin_down", "↓"}:
            return "dn"
    elif isinstance(spin, (int, np.integer)):
        if int(spin) > 0:
            return "up"
        if int(spin) < 0:
            return "dn"
    raise ValueError(f"Unsupported spin label: {spin!r}")


def _candidate_spin_keys(spin: str):
    if spin == "up":
        return ["up", "u", +1]
    return ["dn", "down", "d", -1]


def has_patchset(patchsets: PatchSetMap, spin: SpinLike) -> bool:
    s = normalize_spin(spin)
    for key in _candidate_spin_keys(s):
        if key in patchsets:
            ps = patchsets[key]
            return ps is not None and getattr(ps, "Npatch", len(getattr(ps, "patches", []))) > 0
    return False


def patchset_for_spin(patchsets: PatchSetMap, spin: SpinLike):
    s = normalize_spin(spin)
    for key in _candidate_spin_keys(s):
        if key in patchsets:
            ps = patchsets[key]
            if ps is None or getattr(ps, "Npatch", len(getattr(ps, "patches", []))) == 0:
                raise ValueError(f"Patch set for spin={s!r} exists but is empty")
            return ps
    raise KeyError(f"Could not find patch set for spin={spin!r}")


def available_physical_spins(patchsets: PatchSetMap) -> Tuple[str, ...]:
    out = []
    for s in ("up", "dn"):
        if has_patchset(patchsets, s):
            out.append(s)
    if not out:
        raise ValueError("No non-empty spin patch sets found.")
    return tuple(out)


# ============================================================================
# Q canonicalization / partner maps
# ============================================================================


def _wrap_reduced_coords_unit(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=float)
    uv = uv - np.floor(uv)
    uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
    uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
    return uv


def canonicalize_q_for_patchsets(patchsets: PatchSetMap, q: Sequence[float]) -> np.ndarray:
    ref_spin = "up" if has_patchset(patchsets, "up") else "dn"
    ps = patchset_for_spin(patchsets, ref_spin)
    B = np.column_stack([np.asarray(ps.b1, dtype=float), np.asarray(ps.b2, dtype=float)])
    uv = np.linalg.solve(B, np.asarray(q, dtype=float))
    uv = _wrap_reduced_coords_unit(uv)
    q_can = B @ uv
    q_can[np.isclose(q_can, 0.0, atol=1e-12)] = 0.0
    return q_can


def _minimum_image_displacement(
    k_target: Sequence[float], k_ref: Sequence[float], b1: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    k_target = np.asarray(k_target, dtype=float)
    k_ref = np.asarray(k_ref, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)

    best = None
    best_norm = np.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            disp = k_target - (k_ref + n1 * b1 + n2 * b2)
            nd = np.linalg.norm(disp)
            if nd < best_norm:
                best = disp
                best_norm = nd
    return np.asarray(best, dtype=float)


def _periodic_distance_to_target(
    target_k: Sequence[float], ref_k: Sequence[float], b1: np.ndarray, b2: np.ndarray
) -> float:
    return float(np.linalg.norm(_minimum_image_displacement(target_k, ref_k, b1, b2)))


def partner_map_from_q_index(
    patchsets: PatchSetMap,
    q_index_table: np.ndarray,
    *,
    source_spin: SpinLike,
    target_spin: SpinLike,
    iq_target: int,
    Q: Sequence[float],
    mode: str,
) -> ShiftMap:
    """
    mode:
      - 'Q_minus_k' : target ≈ Q - k
      - 'k_plus_Q'  : target ≈ k + Q
      - 'k_minus_Q' : target ≈ k - Q
    """
    src_spin = normalize_spin(source_spin)
    tgt_spin = normalize_spin(target_spin)
    ps_src = patchset_for_spin(patchsets, src_spin)
    ps_tgt = patchset_for_spin(patchsets, tgt_spin)
    Q = canonicalize_q_for_patchsets(patchsets, Q)

    q_index_table = np.asarray(q_index_table, dtype=int)
    if q_index_table.shape != (ps_src.Npatch, ps_tgt.Npatch):
        raise ValueError(
            f"q_index_table shape {q_index_table.shape} incompatible with patch counts "
            f"({ps_src.Npatch}, {ps_tgt.Npatch})"
        )

    ks_tgt = np.asarray([p.k_cart for p in ps_tgt.patches], dtype=float)
    idxs = np.full(ps_src.Npatch, -1, dtype=int)
    residuals = np.full(ps_src.Npatch, np.inf, dtype=float)

    for p_src, patch in enumerate(ps_src.patches):
        k_src = np.asarray(patch.k_cart, dtype=float)
        if mode == "Q_minus_k":
            target = Q - k_src
        elif mode == "k_plus_Q":
            target = k_src + Q
        elif mode == "k_minus_Q":
            target = k_src - Q
        else:
            raise ValueError("mode must be one of {'Q_minus_k', 'k_plus_Q', 'k_minus_Q'}")

        candidates = np.flatnonzero(q_index_table[p_src, :] == int(iq_target))
        if candidates.size == 0:
            continue

        best_idx = None
        best_dist = np.inf
        for cand in candidates.tolist():
            dist = _periodic_distance_to_target(target, ks_tgt[cand], ps_tgt.b1, ps_tgt.b2)
            if dist < best_dist - 1e-14 or (
                abs(dist - best_dist) <= 1e-14 and (best_idx is None or cand < best_idx)
            ):
                best_dist = dist
                best_idx = int(cand)

        idxs[p_src] = int(best_idx)
        residuals[p_src] = float(best_dist)

    return idxs, residuals


# ============================================================================
# Transfer grid / q catalog
# ============================================================================


@dataclass
class TransferGrid:
    patchsets: PatchSetMap
    q_list: Sequence[np.ndarray]
    decimals: int = 10
    merge_tol_red: float = 5e-2

    def __post_init__(self) -> None:
        ref_spin = available_physical_spins(self.patchsets)[0]
        ps = patchset_for_spin(self.patchsets, ref_spin)
        self.b1 = np.asarray(ps.b1, dtype=float)
        self.b2 = np.asarray(ps.b2, dtype=float)
        self.q_list = [canonicalize_q_for_patchsets(self.patchsets, q) for q in self.q_list]
        self.rep_uv_list: list[np.ndarray] = []
        self.key_to_index: Dict[Tuple[float, float], int] = {}
        unique_qs: list[np.ndarray] = []
        for q in self.q_list:
            uv = self._uv(q)
            idx = self._find_matching_rep(uv)
            if idx is None:
                idx = len(unique_qs)
                unique_qs.append(np.asarray(q, dtype=float))
                self.rep_uv_list.append(uv)
                self.key_to_index[self._exact_key_from_uv(uv)] = idx
        self.q_list = unique_qs

    def _reduced_coords(self, q: Sequence[float]) -> np.ndarray:
        B = np.column_stack([self.b1, self.b2])
        uv = np.linalg.solve(B, np.asarray(q, dtype=float))
        uv = uv - np.floor(uv)
        uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
        uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
        return uv

    def _uv(self, q: Sequence[float]) -> np.ndarray:
        return self._reduced_coords(canonicalize_q_for_patchsets(self.patchsets, q))

    def _exact_key_from_uv(self, uv: np.ndarray) -> Tuple[float, float]:
        return tuple(np.round(np.asarray(uv, dtype=float), decimals=self.decimals))

    def _reduced_distance(self, uv_a: np.ndarray, uv_b: np.ndarray) -> float:
        duv = np.asarray(uv_a, dtype=float) - np.asarray(uv_b, dtype=float)
        duv = duv - np.round(duv)
        return float(np.linalg.norm(duv))

    def _find_matching_rep(self, uv: np.ndarray) -> Optional[int]:
        if len(self.rep_uv_list) == 0:
            return None
        key = self._exact_key_from_uv(uv)
        if key in self.key_to_index:
            return int(self.key_to_index[key])
        best_idx = None
        best_dist = np.inf
        for irep, uv_rep in enumerate(self.rep_uv_list):
            dist = self._reduced_distance(uv, uv_rep)
            if dist <= self.merge_tol_red and (
                dist < best_dist - 1e-14
                or (abs(dist - best_dist) <= 1e-14 and (best_idx is None or irep < best_idx))
            ):
                best_idx = int(irep)
                best_dist = float(dist)
        return best_idx

    def canonicalize(self, q: Sequence[float]) -> np.ndarray:
        return canonicalize_q_for_patchsets(self.patchsets, q)

    def nearest_index(self, q: Sequence[float]) -> int:
        q_can = self.canonicalize(q)
        uv = self._uv(q_can)
        key = self._exact_key_from_uv(uv)
        if key in self.key_to_index:
            return int(self.key_to_index[key])
        idx = self._find_matching_rep(uv)
        if idx is not None:
            return int(idx)
        dists = [self._reduced_distance(uv, uv_rep) for uv_rep in self.rep_uv_list]
        return int(np.argmin(dists))


def _canonicalize_patch_k_array(ks: np.ndarray, patchsets: PatchSetMap) -> np.ndarray:
    out = np.zeros_like(ks, dtype=float)
    for i, k in enumerate(np.asarray(ks, dtype=float)):
        out[i] = canonicalize_q_for_patchsets(patchsets, k)
    return out


def build_unique_q_list(
    patchsets: PatchSetMap,
    *,
    mode: str,
    decimals: int = 10,
    merge_tol_red: float = 5e-2,
) -> list[np.ndarray]:
    ref_spin = available_physical_spins(patchsets)[0]
    ps = patchset_for_spin(patchsets, ref_spin)
    ks = _canonicalize_patch_k_array(np.asarray([p.k_cart for p in ps.patches], dtype=float), patchsets)

    raw_candidates: list[np.ndarray] = [np.zeros(2, dtype=float)]
    if mode == "pp":
        for k1 in ks:
            for k2 in ks:
                raw_candidates.append(np.asarray(k1 + k2, dtype=float))
    elif mode in {"ph", "phc"}:
        for k1 in ks:
            for k3 in ks:
                raw_candidates.append(np.asarray(k3 - k1, dtype=float))
    else:
        raise ValueError("mode must be one of {'pp', 'ph', 'phc'}")

    grid = TransferGrid(
        patchsets,
        raw_candidates,
        decimals=decimals,
        merge_tol_red=merge_tol_red,
    )
    return [np.asarray(q, dtype=float) for q in grid.q_list]


# ============================================================================
# Matsubara bubbles
# ============================================================================


@lru_cache(maxsize=128)
def _matsubara_grid_cached(temperature: float, nfreq: int):
    n = np.arange(-nfreq, nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * temperature
    dw_dT = (2.0 * n + 1.0) * np.pi
    return n, w, dw_dT


def physical_propagator(iw: np.ndarray, energy: float) -> np.ndarray:
    return 1.0 / (1j * iw - energy)


def d_physical_propagator_dT_fixed_sigma(
    n: np.ndarray,
    temperature: float,
    energy: float,
    *,
    sign: int = +1,
) -> np.ndarray:
    w = (2.0 * n + 1.0) * np.pi * temperature
    iw = sign * w
    dw_dT = (2.0 * n + 1.0) * np.pi
    g = physical_propagator(iw, energy)
    return -(1j * sign * dw_dT) * (g ** 2)


def bubble_dot_pp(energy_a: float, energy_b: float, config: FlowConfig) -> complex:
    T = config.temperature
    n = np.arange(-config.nfreq, config.nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * T
    g_a = physical_propagator(w, energy_a)
    g_b = physical_propagator(-w, energy_b)
    dg_a = d_physical_propagator_dT_fixed_sigma(n, T, energy_a, sign=+1)
    dg_b = d_physical_propagator_dT_fixed_sigma(n, T, energy_b, sign=-1)
    val = T * np.sum(dg_a * g_b + g_a * dg_b)
    if config.include_explicit_T_prefactor:
        val += np.sum(g_a * g_b)
    return complex(val)


def bubble_dot_ph(energy_a: float, energy_b: float, config: FlowConfig) -> complex:
    T = config.temperature
    n = np.arange(-config.nfreq, config.nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * T
    g_a = physical_propagator(w, energy_a)
    g_b = physical_propagator(w, energy_b)
    dg_a = d_physical_propagator_dT_fixed_sigma(n, T, energy_a, sign=+1)
    dg_b = d_physical_propagator_dT_fixed_sigma(n, T, energy_b, sign=+1)
    val = T * np.sum(dg_a * g_b + g_a * dg_b)
    if config.include_explicit_T_prefactor:
        val += np.sum(g_a * g_b)
    return complex(val)


def _patch_energies(ps) -> np.ndarray:
    return np.array([float(p.energy) for p in ps.patches], dtype=float)


def _bubble_dot_pp_vec(Ea: np.ndarray, Eb: np.ndarray, config: FlowConfig) -> np.ndarray:
    T = float(config.temperature)
    _, w, dw_dT = _matsubara_grid_cached(T, int(config.nfreq))
    Ea = np.asarray(Ea, dtype=float)[None, :]
    Eb = np.asarray(Eb, dtype=float)[None, :]
    wcol = w[:, None]
    dwcol = dw_dT[:, None]

    g_a = 1.0 / (1j * wcol - Ea)
    g_b = 1.0 / (-1j * wcol - Eb)
    dg_a = -(1j * dwcol) * (g_a ** 2)
    dg_b = +(1j * dwcol) * (g_b ** 2)

    val = T * np.sum(dg_a * g_b + g_a * dg_b, axis=0)
    if config.include_explicit_T_prefactor:
        val = val + np.sum(g_a * g_b, axis=0)
    return np.asarray(val, dtype=complex)


def _bubble_dot_ph_vec(Ea: np.ndarray, Eb: np.ndarray, config: FlowConfig) -> np.ndarray:
    T = float(config.temperature)
    _, w, dw_dT = _matsubara_grid_cached(T, int(config.nfreq))
    Ea = np.asarray(Ea, dtype=float)[None, :]
    Eb = np.asarray(Eb, dtype=float)[None, :]
    wcol = w[:, None]
    dwcol = dw_dT[:, None]

    g_a = 1.0 / (1j * wcol - Ea)
    g_b = 1.0 / (1j * wcol - Eb)
    dg_a = -(1j * dwcol) * (g_a ** 2)
    dg_b = -(1j * dwcol) * (g_b ** 2)

    val = T * np.sum(dg_a * g_b + g_a * dg_b, axis=0)
    if config.include_explicit_T_prefactor:
        val = val + np.sum(g_a * g_b, axis=0)
    return np.asarray(val, dtype=complex)


def _coerce_shift_map(shift_map: ShiftMap) -> ShiftMap:
    partner, residual = shift_map
    return np.asarray(partner, dtype=int), np.asarray(residual, dtype=float)


def build_pp_internal_cache_vec(
    patchsets: PatchSetMap,
    config: FlowConfig,
    *,
    shift_cache: Mapping[Tuple[str, str], ShiftMap],
) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for (sa, sb), raw_shift in shift_cache.items():
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        partner, residual = _coerce_shift_map(raw_shift)
        weights = np.zeros_like(eps_a, dtype=complex)
        valid = partner >= 0
        if np.any(valid):
            weights[valid] = _bubble_dot_pp_vec(eps_a[valid], eps_b_all[partner[valid]], config)
        cache[(sa, sb)] = {
            "partner": partner,
            "residual": residual,
            "weights": np.asarray(weights, dtype=complex),
        }
    return cache


def build_ph_internal_cache_vec(
    patchsets: PatchSetMap,
    config: FlowConfig,
    *,
    shift_cache: Mapping[Tuple[str, str], ShiftMap],
) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for (sa, sb), raw_shift in shift_cache.items():
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        partner, residual = _coerce_shift_map(raw_shift)
        weights = np.zeros_like(eps_a, dtype=complex)
        valid = partner >= 0
        if np.any(valid):
            weights[valid] = _bubble_dot_ph_vec(eps_a[valid], eps_b_all[partner[valid]], config)
        cache[(sa, sb)] = {
            "partner": partner,
            "residual": residual,
            "weights": np.asarray(weights, dtype=complex),
        }
    return cache


# ============================================================================
# Minimal S_z = 0 vertex accessor / cache conversion
# ============================================================================


SZ0VertexInput = Union[
    Callable[[int, int, int, int], complex],
    np.ndarray,
]


def build_sz0_vertex_accessor(v: SZ0VertexInput):
    """
    Build accessor for the PRL-compatible minimal vertex

        V(1,2;3,4) ≡ Γ_{up,dn -> dn,up}(1,2;3,4)

    Accepted forms
    --------------
    1. callable(p1,p2,p3,p4) -> complex
    2. ndarray shape (Np,Np,Np,Np)
    """
    if callable(v):
        return v

    arr = np.asarray(v, dtype=complex)
    if arr.ndim != 4:
        raise ValueError(f"ndarray-based S_z=0 vertex must have ndim=4, got {arr.ndim}")

    def accessor(p1: int, p2: int, p3: int, p4: int) -> complex:
        return complex(arr[p1, p2, p3, p4])

    return accessor


def extract_minimal_internal_cache(
    internal_cache: Union[MinimalInternalCache, Mapping[Tuple[str, str], Dict[str, np.ndarray]]]
) -> MinimalInternalCache:
    """
    Normalize the internal-cache format for the minimal PRL-compatible solver.

    The cleaned-up solver should pass a plain dict with keys
        {'partner', 'residual', 'weights'}.
    For backward compatibility we still accept the older keyed wrapper
        {('up','dn'): {...}}.
    """
    if "partner" in internal_cache and "weights" in internal_cache:
        return {
            "partner": np.asarray(internal_cache["partner"], dtype=int),
            "residual": np.asarray(internal_cache["residual"], dtype=float),
            "weights": np.asarray(internal_cache["weights"], dtype=complex),
        }

    if ("up", "dn") not in internal_cache:
        raise KeyError(
            "Minimal Sz=0 internal cache must either be a plain dict with keys "
            "'partner'/'residual'/'weights' or a legacy wrapper containing ('up','dn')."
        )

    info = internal_cache[("up", "dn")]
    return {
        "partner": np.asarray(info["partner"], dtype=int),
        "residual": np.asarray(info["residual"], dtype=float),
        "weights": np.asarray(info["weights"], dtype=complex),
    }


# ============================================================================
# One-loop RHS for minimal S_z = 0 vertex
# ============================================================================


def compute_pp_vertex_contribution_sz0(
    v: SZ0VertexInput,
    *,
    p1: int,
    p2: int,
    p3: int,
    p4: int,
    internal_cache: Union[MinimalInternalCache, Mapping[Tuple[str, str], Dict[str, np.ndarray]]],
) -> complex:
    r"""
    pp contribution for the minimal object

        V(1,2;3,4) \equiv \Gamma_{\uparrow\downarrow\to\downarrow\uparrow}(1,2;3,4).

    This routine is *not* a bare ``V*V`` legacy shell.  It is obtained by taking
    the full antisymmetrized pp topology for ``\Gamma`` and substituting

        \Gamma_{\sigma_1\sigma_2\sigma_3\sigma_4}(1,2;3,4)
        = V(1,2;3,4) \delta_{\sigma_1\sigma_4}\delta_{\sigma_2\sigma_3}
        - V(1,2;4,3) \delta_{\sigma_1\sigma_3}\delta_{\sigma_2\sigma_4}.

    For external ``(\uparrow,\downarrow\to\downarrow,\uparrow)`` the internal
    spin sum leaves two terms,

        \Phi_pp[V] = - \sum_{ab} L_pp(a,b)
            [ V(1,2;a,b) V(a,b;4,3)
            + V(1,2;b,a) V(a,b;3,4) ].

    The overall minus sign is a genuine consequence of reducing the full
    antisymmetrized vertex to the minimal ``S_z=0`` object.
    """
    vfn = build_sz0_vertex_accessor(v)
    info = extract_minimal_internal_cache(internal_cache)
    partner = np.asarray(info["partner"], dtype=int)
    weights = np.asarray(info["weights"], dtype=complex)

    acc = 0.0 + 0.0j
    for a in range(len(partner)):
        b = int(partner[a])
        if b < 0 or abs(weights[a]) == 0:
            continue
        term1 = vfn(p1, p2, a, b) * vfn(a, b, p4, p3)
        term2 = vfn(p1, p2, b, a) * vfn(a, b, p3, p4)
        acc -= weights[a] * (term1 + term2)
    return complex(acc)


def compute_phd_vertex_contribution_sz0(
    v: SZ0VertexInput,
    *,
    p1: int,
    p2: int,
    p3: int,
    p4: int,
    internal_cache: Union[MinimalInternalCache, Mapping[Tuple[str, str], Dict[str, np.ndarray]]],
) -> complex:
    r"""
    direct ph contribution for the minimal ``S_z=0`` object.

    After substituting the full-vertex reconstruction into the direct particle-
    hole topology, the spin sum for external

        (\uparrow,\downarrow\to\downarrow,\uparrow)

    collapses to the same two momentum permutations already familiar from the
    full antisymmetrized direct-ph channel,

        \Phi_phd[V] = - \sum_{ab} L_ph(a,b)
            [ V(1,a;3,b) V(b,2;a,4)
            + V(1,b;3,a) V(a,2;b,4) ].
    """
    vfn = build_sz0_vertex_accessor(v)
    info = extract_minimal_internal_cache(internal_cache)
    partner = np.asarray(info["partner"], dtype=int)
    weights = np.asarray(info["weights"], dtype=complex)

    acc = 0.0 + 0.0j
    for a in range(len(partner)):
        b = int(partner[a])
        if b < 0 or abs(weights[a]) == 0:
            continue
        term1 = vfn(p1, a, p3, b) * vfn(b, p2, a, p4)
        term2 = vfn(p1, b, p3, a) * vfn(a, p2, b, p4)
        acc -= weights[a] * (term1 + term2)
    return complex(acc)


def compute_phc_vertex_contribution_sz0(
    v: SZ0VertexInput,
    *,
    p1: int,
    p2: int,
    p3: int,
    p4: int,
    internal_cache: Union[MinimalInternalCache, Mapping[Tuple[str, str], Dict[str, np.ndarray]]],
) -> complex:
    r"""
    Correct crossed-ph contribution for the minimal S_z=0 object

        X(1,2;3,4) ≡ Γ_{up,dn -> dn,up}(1,2;3,4).

    Derived by explicit full-spin reconstruction and internal-spin summation
    for external (up, dn -> dn, up). The correct minimal formula is

        Φ_phc[X] = Σ_ab L_ph(a,b) [
            X(1,a;4,b) X(b,2;3,a)
          - 2 X(1,a;b,4) X(b,2;3,a)
          +   X(1,a;b,4) X(b,2;a,3)
        ]

    with NO extra "term_b". The previous implementation double counted.
    """
    vfn = build_sz0_vertex_accessor(v)
    info = extract_minimal_internal_cache(internal_cache)
    partner = np.asarray(info["partner"], dtype=int)
    weights = np.asarray(info["weights"], dtype=complex)

    acc = 0.0 + 0.0j
    for a in range(len(partner)):
        b = int(partner[a])
        if b < 0 or abs(weights[a]) == 0:
            continue

        term = (
            vfn(p1, a, p4, b) * vfn(b, p2, p3, a)
            - 2.0 * vfn(p1, a, b, p4) * vfn(b, p2, p3, a)
            +       vfn(p1, a, b, p4) * vfn(b, p2, a, p3)
        )
        acc += weights[a] * term

    return complex(acc)


__all__ = [
    "FlowConfig",
    "MinimalInternalCache",
    "PatchSetMap",
    "ShiftMap",
    "TransferGrid",
    "SZ0VertexInput",
    "available_physical_spins",
    "build_unique_q_list",
    "bubble_dot_ph",
    "bubble_dot_pp",
    "build_pp_internal_cache_vec",
    "build_ph_internal_cache_vec",
    "build_sz0_vertex_accessor",
    "canonicalize_q_for_patchsets",
    "compute_pp_vertex_contribution_sz0",
    "compute_phd_vertex_contribution_sz0",
    "compute_phc_vertex_contribution_sz0",
    "extract_minimal_internal_cache",
    "has_patchset",
    "normalize_spin",
    "partner_map_from_q_index",
    "patchset_for_spin",
]
