from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from channels import ChannelKernel
except Exception:
    @dataclass
    class ChannelKernel:
        name: str
        Q: np.ndarray
        matrix: np.ndarray
        row_patches: np.ndarray
        col_patches: np.ndarray
        row_partner_patches: np.ndarray
        col_partner_patches: np.ndarray
        row_spins: Tuple[str, str]
        col_spins: Tuple[str, str]
        residuals: np.ndarray

        @property
        def Npatch(self) -> int:
            return int(self.matrix.shape[0])


SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
SpinBlock = Tuple[str, str, str, str]
ShiftMap = Tuple[np.ndarray, np.ndarray]
InternalCache = Dict[Tuple[str, str], Dict[str, np.ndarray]]

GammaInput = Union[
    Callable[[int, str, int, str, int, str, int, str], complex],
    Mapping[Tuple[str, str, str, str], np.ndarray],
]


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


# ---------------------------------------------------------------------------
# Spin / patch utilities
# ---------------------------------------------------------------------------

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


def available_internal_spin_pairs(patchsets: PatchSetMap):
    out = []
    for sa in ("up", "dn"):
        if not has_patchset(patchsets, sa):
            continue
        for sb in ("up", "dn"):
            if not has_patchset(patchsets, sb):
                continue
            out.append((sa, sb))
    return out


# ---------------------------------------------------------------------------
# Q canonicalization and constrained partner maps
# ---------------------------------------------------------------------------

def _reduced_coords(k: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    B = np.column_stack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)])
    return np.linalg.solve(B, np.asarray(k, dtype=float))


def _wrap_reduced_coords_unit(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=float)
    uv = uv - np.floor(uv)
    uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
    uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
    return uv


def canonicalize_q_for_patchsets(patchsets: PatchSetMap, q: Sequence[float]) -> np.ndarray:
    """Canonicalize q modulo reciprocal lattice vectors using the unique [0,1) reduced-cell representative."""
    ref_spin = "up" if has_patchset(patchsets, "up") else "dn"
    ps = patchset_for_spin(patchsets, ref_spin)
    B = np.column_stack([np.asarray(ps.b1, dtype=float), np.asarray(ps.b2, dtype=float)])
    uv = np.linalg.solve(B, np.asarray(q, dtype=float))
    uv = _wrap_reduced_coords_unit(uv)
    q_can = B @ uv
    q_can[np.isclose(q_can, 0.0, atol=1e-12)] = 0.0
    return q_can


def reduced_q_key_for_patchsets(
    patchsets: PatchSetMap,
    q: Sequence[float],
    *,
    tol_red: float = 1e-8,
) -> Tuple[int, int]:
    """Quantized key in reduced coordinates, robust to floating-point patch noise."""
    ref_spin = "up" if has_patchset(patchsets, "up") else "dn"
    ps = patchset_for_spin(patchsets, ref_spin)
    q_can = canonicalize_q_for_patchsets(patchsets, q)
    uv = _reduced_coords(np.asarray(q_can, dtype=float), np.asarray(ps.b1, dtype=float), np.asarray(ps.b2, dtype=float))
    uv = _wrap_reduced_coords_unit(uv)
    return tuple(np.round(uv / float(tol_red)).astype(int).tolist())  # type: ignore[return-value]


def _minimum_image_displacement(k_target: Sequence[float], k_ref: Sequence[float], b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
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


def _periodic_distance_to_target(target_k: Sequence[float], ref_k: Sequence[float], b1: np.ndarray, b2: np.ndarray) -> float:
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
    Choose one target-leg patch for each source-leg patch, constrained to the requested transfer block `iq_target`.

    Parameters
    ----------
    mode:
      - 'Q_minus_k' : pair partner, target ≈ Q - k
      - 'k_plus_Q'  : particle-hole bilinear, target ≈ k + Q
      - 'k_minus_Q' : particle-hole bilinear, target ≈ k - Q
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
            if dist < best_dist - 1e-14 or (abs(dist - best_dist) <= 1e-14 and (best_idx is None or cand < best_idx)):
                best_dist = dist
                best_idx = int(cand)

        idxs[p_src] = int(best_idx)
        residuals[p_src] = float(best_dist)

    return idxs, residuals


def build_partner_maps_by_iq(
    patchsets: PatchSetMap,
    q_list: Sequence[Sequence[float]],
    q_index_table: np.ndarray,
    *,
    source_spin: SpinLike,
    target_spin: SpinLike,
    mode: str,
) -> Dict[int, ShiftMap]:
    """Precompute Q-constrained partner maps for every iq in a transfer grid."""
    out: Dict[int, ShiftMap] = {}
    for iq, Q in enumerate(q_list):
        out[int(iq)] = partner_map_from_q_index(
            patchsets,
            q_index_table,
            source_spin=source_spin,
            target_spin=target_spin,
            iq_target=int(iq),
            Q=Q,
            mode=mode,
        )
    return out


# ---------------------------------------------------------------------------
# Full-vertex accessor
# ---------------------------------------------------------------------------

def build_gamma_accessor(gamma: GammaInput) -> Callable[[int, str, int, str, int, str, int, str], complex]:
    """
    Build an accessor for the full two-particle vertex.

    Accepted forms
    --------------
    1. Callable(p1,s1,p2,s2,p3,s3,p4,s4) -> complex
    2. Mapping[(s1,s2,s3,s4)] -> ndarray with shape (Np1,Np2,Np3,Np4)

    Notes
    -----
    This accessor is deliberately full-vertex oriented. It does *not* assume a
    reduced channel representation.
    """
    if callable(gamma):
        def accessor(p1: int, s1: str, p2: int, s2: str, p3: int, s3: str, p4: int, s4: str) -> complex:
            return complex(gamma(p1, s1, p2, s2, p3, s3, p4, s4))
        return accessor

    tensors: Dict[Tuple[str, str, str, str], np.ndarray] = {
        tuple(map(normalize_spin, key)): np.asarray(val, dtype=complex)
        for key, val in gamma.items()
    }

    def accessor(p1: int, s1: str, p2: int, s2: str, p3: int, s3: str, p4: int, s4: str) -> complex:
        key = (normalize_spin(s1), normalize_spin(s2), normalize_spin(s3), normalize_spin(s4))
        block = tensors.get(key)
        if block is None:
            return 0.0 + 0.0j
        if block.ndim != 4:
            raise ValueError(
                "Mapping-based gamma blocks must be full 4-leg tensors of shape (Np1,Np2,Np3,Np4). "
                f"Got ndim={block.ndim} for block {key}."
            )
        return complex(block[p1, p2, p3, p4])

    return accessor


# ---------------------------------------------------------------------------
# Propagators and bubbles (temperature flow, static vertex)
# ---------------------------------------------------------------------------

def physical_propagator(iw: np.ndarray, energy: float, sigma: Optional[np.ndarray] = None) -> np.ndarray:
    if sigma is None:
        return 1.0 / (1j * iw - energy)
    return 1.0 / (1j * iw - energy - sigma)


def d_physical_propagator_dT_fixed_sigma(
    n: np.ndarray,
    temperature: float,
    energy: float,
    *,
    sign: int = +1,
    sigma: Optional[np.ndarray] = None,
) -> np.ndarray:
    w = (2.0 * n + 1.0) * np.pi * temperature
    iw = sign * w
    dw_dT = (2.0 * n + 1.0) * np.pi
    g = physical_propagator(iw, energy, sigma=sigma)
    return -(1j * sign * dw_dT) * (g ** 2)


@lru_cache(maxsize=128)
def _matsubara_grid_cached(temperature: float, nfreq: int):
    n = np.arange(-nfreq, nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * temperature
    dw_dT = (2.0 * n + 1.0) * np.pi
    return n, w, dw_dT


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
) -> InternalCache:
    """
    Build pp loop bubbles for a *fixed* Q sector.

    `shift_cache[(sa,sb)]` must encode the Q-constrained pair partner b ≈ Q - a for
    the internal spin pair (sa,sb).
    """
    cache: InternalCache = {}
    for sa, sb in available_internal_spin_pairs(patchsets):
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        if (sa, sb) not in shift_cache:
            raise KeyError(f"Missing shift_cache entry for internal spin pair {(sa, sb)!r}")
        partner, residual = _coerce_shift_map(shift_cache[(sa, sb)])
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
) -> InternalCache:
    """
    Build ph loop bubbles for a *fixed* Q sector.

    `shift_cache[(sa,sb)]` must encode the Q-constrained bilinear partner b ≈ a + Q
    for the internal spin pair (sa,sb).
    """
    cache: InternalCache = {}
    for sa, sb in available_internal_spin_pairs(patchsets):
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        if (sa, sb) not in shift_cache:
            raise KeyError(f"Missing shift_cache entry for internal spin pair {(sa, sb)!r}")
        partner, residual = _coerce_shift_map(shift_cache[(sa, sb)])
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


def build_internal_caches_by_iq(
    patchsets: PatchSetMap,
    config: FlowConfig,
    *,
    shift_caches_by_iq: Mapping[int, Mapping[Tuple[str, str], ShiftMap]],
    channel: str,
) -> Dict[int, InternalCache]:
    """Precompute internal loop bubbles for every transfer block iq."""
    out: Dict[int, InternalCache] = {}
    for iq, shift_cache in shift_caches_by_iq.items():
        if channel == "pp":
            out[int(iq)] = build_pp_internal_cache_vec(patchsets, config, shift_cache=shift_cache)
        elif channel in {"phd", "phc", "ph"}:
            out[int(iq)] = build_ph_internal_cache_vec(patchsets, config, shift_cache=shift_cache)
        else:
            raise ValueError("channel must be one of {'pp', 'phd', 'phc', 'ph'}")
    return out


# ---------------------------------------------------------------------------
# Exact one-loop contributions for the full vertex
# ---------------------------------------------------------------------------

def _normalize_allowed_spin_blocks(
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]]
) -> Optional[FrozenSet[SpinBlock]]:
    if allowed_spin_blocks is None:
        return None
    return frozenset(tuple(normalize_spin(x) for x in blk) for blk in allowed_spin_blocks)


def _allowed(block: SpinBlock, allowed_spin_blocks: Optional[FrozenSet[SpinBlock]]) -> bool:
    if allowed_spin_blocks is None:
        return True
    return block in allowed_spin_blocks


def compute_pp_vertex_contribution(
    gamma: GammaInput,
    *,
    p1: int,
    s1: SpinLike,
    p2: int,
    s2: SpinLike,
    p3: int,
    s3: SpinLike,
    p4: int,
    s4: SpinLike,
    internal_cache: InternalCache,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> complex:
    """
    Particle-particle contribution to the flow of the full vertex.

    Convention
    ----------
    Vertex is Γ[(p1,s1),(p2,s2) -> (p3,s3),(p4,s4)].
    This implements the first quadratic term in Metzner Eq. (52), i.e. the pp loop
    topology acting on the *full* vertex, not a reduced channel object. fileciteturn3file0
    """
    gamma_fn = build_gamma_accessor(gamma)
    s1 = normalize_spin(s1)
    s2 = normalize_spin(s2)
    s3 = normalize_spin(s3)
    s4 = normalize_spin(s4)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)

    acc = 0.0 + 0.0j
    for (sa, sb), info in internal_cache.items():
        left_blk = (s1, s2, sa, sb)
        right_blk = (sa, sb, s3, s4)
        if not (_allowed(left_blk, allowed) and _allowed(right_blk, allowed)):
            continue
        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)
        for a in range(len(partner)):
            b = int(partner[a])
            if b < 0 or abs(weights[a]) == 0:
                continue
            acc += weights[a] * gamma_fn(p1, s1, p2, s2, a, sa, b, sb) * gamma_fn(a, sa, b, sb, p3, s3, p4, s4)
    return complex(acc)


def compute_phd_vertex_contribution(
    gamma: GammaInput,
    *,
    p1: int,
    s1: SpinLike,
    p2: int,
    s2: SpinLike,
    p3: int,
    s3: SpinLike,
    p4: int,
    s4: SpinLike,
    internal_cache: InternalCache,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> complex:
    """
    Direct particle-hole contribution to the flow of the full vertex.

    Leg meaning
    -----------
    The transfer is Q = p3 - p1 = p2 - p4. Internal lines form a *bilinear* (a -> b=a+Q).
    This is not a pair partner map; its physical meaning is particle-hole. The two terms are
    the explicit permutations in Eq. (52). fileciteturn3file0
    """
    gamma_fn = build_gamma_accessor(gamma)
    s1 = normalize_spin(s1)
    s2 = normalize_spin(s2)
    s3 = normalize_spin(s3)
    s4 = normalize_spin(s4)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)

    acc = 0.0 + 0.0j
    for (sa, sb), info in internal_cache.items():
        keep1 = _allowed((s1, sa, s3, sb), allowed) and _allowed((sb, s2, sa, s4), allowed)
        keep2 = _allowed((s1, sb, s3, sa), allowed) and _allowed((sa, s2, sb, s4), allowed)
        if not (keep1 or keep2):
            continue
        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)
        for a in range(len(partner)):
            b = int(partner[a])
            if b < 0 or abs(weights[a]) == 0:
                continue
            term1 = 0.0 + 0.0j
            term2 = 0.0 + 0.0j
            if keep1:
                term1 = gamma_fn(p1, s1, a, sa, p3, s3, b, sb) * gamma_fn(b, sb, p2, s2, a, sa, p4, s4)
            if keep2:
                term2 = gamma_fn(p1, s1, b, sb, p3, s3, a, sa) * gamma_fn(a, sa, p2, s2, b, sb, p4, s4)
            acc -= weights[a] * (term1 + term2)
    return complex(acc)


def compute_phc_vertex_contribution(
    gamma: GammaInput,
    *,
    p1: int,
    s1: SpinLike,
    p2: int,
    s2: SpinLike,
    p3: int,
    s3: SpinLike,
    p4: int,
    s4: SpinLike,
    internal_cache: InternalCache,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> complex:
    """
    Crossed particle-hole contribution to the flow of the full vertex.

    Leg meaning
    -----------
    The transfer is Q = p3 - p2 = p1 - p4. Internal lines again form a particle-hole bilinear.
    In the sign convention used by the main flow solver, the direct ph channel enters with a minus sign while the crossed ph channel enters with a plus sign. fileciteturn3file0
    """
    gamma_fn = build_gamma_accessor(gamma)
    s1 = normalize_spin(s1)
    s2 = normalize_spin(s2)
    s3 = normalize_spin(s3)
    s4 = normalize_spin(s4)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)

    acc = 0.0 + 0.0j
    for (sa, sb), info in internal_cache.items():
        keep1 = _allowed((s1, sa, s4, sb), allowed) and _allowed((sb, s2, sa, s3), allowed)
        keep2 = _allowed((s1, sb, s4, sa), allowed) and _allowed((sa, s2, sb, s3), allowed)
        if not (keep1 or keep2):
            continue
        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)
        for a in range(len(partner)):
            b = int(partner[a])
            if b < 0 or abs(weights[a]) == 0:
                continue
            term1 = 0.0 + 0.0j
            term2 = 0.0 + 0.0j
            if keep1:
                term1 = gamma_fn(p1, s1, a, sa, p4, s4, b, sb) * gamma_fn(b, sb, p2, s2, a, sa, p3, s3)
            if keep2:
                term2 = gamma_fn(p1, s1, b, sb, p4, s4, a, sa) * gamma_fn(a, sa, p2, s2, b, sb, p3, s3)
            acc += weights[a] * (term1 + term2)
    return complex(acc)


def compute_rhs_block_fullvertex(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    *,
    spin_block: Tuple[SpinLike, SpinLike, SpinLike, SpinLike],
    p4_index: np.ndarray,
    pp_q_index: np.ndarray,
    phd_q_index: np.ndarray,
    phc_q_index: np.ndarray,
    pp_internal_caches_by_iq: Mapping[int, InternalCache],
    phd_internal_caches_by_iq: Mapping[int, InternalCache],
    phc_internal_caches_by_iq: Mapping[int, InternalCache],
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute the three one-loop contributions for one full-vertex spin block.

    Parameters
    ----------
    spin_block:
        (s1, s2, s3, s4) for Γ[(p1,s1),(p2,s2) -> (p3,s3),(p4,s4)].
    p4_index:
        Integer array of shape (N1,N2,N3). Entry [p1,p2,p3] gives the momentum-conserving
        fourth leg p4 for this block, or -1 if no valid completion exists in the discretized patch basis.
    pp_q_index:
        q-index table for the incoming pair geometry (p1,p2) with transfer Q = p1 + p2.
    phd_q_index:
        q-index table for the direct ph geometry (p3,p1) with transfer Q = p3 - p1.
    phc_q_index:
        q-index table for the crossed ph geometry (p3,p2) with transfer Q = p3 - p2.

    Returns
    -------
    dict with keys 'pp', 'phd', 'phc', 'total', each an ndarray with the same shape as p4_index.
    """
    s1, s2, s3, s4 = tuple(normalize_spin(s) for s in spin_block)
    N1 = patchset_for_spin(patchsets, s1).Npatch
    N2 = patchset_for_spin(patchsets, s2).Npatch
    N3 = patchset_for_spin(patchsets, s3).Npatch
    if p4_index.shape != (N1, N2, N3):
        raise ValueError(
            f"p4_index has shape {p4_index.shape}, expected {(N1, N2, N3)} for spin block {spin_block}."
        )

    rhs_pp = np.zeros((N1, N2, N3), dtype=complex)
    rhs_phd = np.zeros((N1, N2, N3), dtype=complex)
    rhs_phc = np.zeros((N1, N2, N3), dtype=complex)

    for p1 in range(N1):
        for p2 in range(N2):
            iq_pp = int(pp_q_index[p1, p2])
            pp_cache = pp_internal_caches_by_iq.get(iq_pp)
            if pp_cache is None:
                raise KeyError(f"Missing pp internal cache for iq={iq_pp}")
            for p3 in range(N3):
                p4 = int(p4_index[p1, p2, p3])
                if p4 < 0:
                    continue

                iq_phd = int(phd_q_index[p3, p1])
                iq_phc = int(phc_q_index[p3, p2])
                phd_cache = phd_internal_caches_by_iq.get(iq_phd)
                phc_cache = phc_internal_caches_by_iq.get(iq_phc)
                if phd_cache is None:
                    raise KeyError(f"Missing phd internal cache for iq={iq_phd}")
                if phc_cache is None:
                    raise KeyError(f"Missing phc internal cache for iq={iq_phc}")

                rhs_pp[p1, p2, p3] = compute_pp_vertex_contribution(
                    gamma,
                    p1=p1, s1=s1, p2=p2, s2=s2, p3=p3, s3=s3, p4=p4, s4=s4,
                    internal_cache=pp_cache,
                    allowed_spin_blocks=allowed_spin_blocks,
                )
                rhs_phd[p1, p2, p3] = compute_phd_vertex_contribution(
                    gamma,
                    p1=p1, s1=s1, p2=p2, s2=s2, p3=p3, s3=s3, p4=p4, s4=s4,
                    internal_cache=phd_cache,
                    allowed_spin_blocks=allowed_spin_blocks,
                )
                rhs_phc[p1, p2, p3] = compute_phc_vertex_contribution(
                    gamma,
                    p1=p1, s1=s1, p2=p2, s2=s2, p3=p3, s3=s3, p4=p4, s4=s4,
                    internal_cache=phc_cache,
                    allowed_spin_blocks=allowed_spin_blocks,
                )

    return {
        "pp": rhs_pp,
        "phd": rhs_phd,
        "phc": rhs_phc,
        "total": rhs_pp + rhs_phd + rhs_phc,
    }


# ---------------------------------------------------------------------------
# Compatibility stubs: diagnosis kernels move to channels.py in the redesign
# ---------------------------------------------------------------------------

def compute_pp_kernel(*args, **kwargs):
    raise NotImplementedError(
        "compute_pp_kernel has been removed from the physics core in the full-vertex redesign. "
        "Diagnosis kernels should be rebuilt from the flowed full vertex in channels.py."
    )


def compute_ph_kernel(*args, **kwargs):
    raise NotImplementedError(
        "compute_ph_kernel has been removed from the physics core in the full-vertex redesign. "
        "Diagnosis kernels should be rebuilt from the flowed full vertex in channels.py."
    )


def compute_phc_kernel(*args, **kwargs):
    raise NotImplementedError(
        "compute_phc_kernel has been removed from the physics core in the full-vertex redesign. "
        "Diagnosis kernels should be rebuilt from the flowed full vertex in channels.py."
    )


def compute_pp_kernel_fast2(*args, **kwargs):
    raise NotImplementedError("compute_pp_kernel_fast2 moved out of frg_kernel; use channels.py after the full-vertex migration.")


def compute_ph_kernel_fast2(*args, **kwargs):
    raise NotImplementedError("compute_ph_kernel_fast2 moved out of frg_kernel; use channels.py after the full-vertex migration.")


def compute_phc_kernel_fast2(*args, **kwargs):
    raise NotImplementedError("compute_phc_kernel_fast2 moved out of frg_kernel; use channels.py after the full-vertex migration.")


__all__ = [
    "ChannelKernel",
    "FlowConfig",
    "GammaInput",
    "InternalCache",
    "PatchSetMap",
    "ShiftMap",
    "SpinBlock",
    "SpinLike",
    "available_internal_spin_pairs",
    "build_gamma_accessor",
    "build_internal_caches_by_iq",
    "build_partner_maps_by_iq",
    "build_ph_internal_cache_vec",
    "build_pp_internal_cache_vec",
    "bubble_dot_ph",
    "bubble_dot_pp",
    "canonicalize_q_for_patchsets",
    "compute_phc_vertex_contribution",
    "compute_phd_vertex_contribution",
    "compute_pp_vertex_contribution",
    "compute_rhs_block_fullvertex",
    "has_patchset",
    "normalize_spin",
    "partner_map_from_q_index",
    "patchset_for_spin",
    "physical_propagator",
    "reduced_q_key_for_patchsets",
]
