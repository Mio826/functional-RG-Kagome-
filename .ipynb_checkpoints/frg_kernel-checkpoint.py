
from __future__ import annotations

from dataclasses import dataclass
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
GammaInput = Union[
    Callable[[int, str, int, str, int, str, int, str], complex],
    Mapping[Tuple[str, str, str, str], np.ndarray],
]
SpinBlock = Tuple[str, str, str, str]


@dataclass
class FlowConfig:
    temperature: float
    nfreq: int = 256
    include_explicit_T_prefactor: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.nfreq <= 0:
            raise ValueError("nfreq must be positive")


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


def find_shifted_patch_index(patchsets: PatchSetMap, spin: SpinLike, target_k: Sequence[float]) -> Tuple[int, float]:
    ps = patchset_for_spin(patchsets, spin)
    ks = np.asarray([p.k_cart for p in ps.patches], dtype=float)
    dists = np.array([
        np.linalg.norm(_minimum_image_displacement(target_k, k_ref, ps.b1, ps.b2))
        for k_ref in ks
    ], dtype=float)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def shifted_patch_map(patchsets: PatchSetMap, spin: SpinLike, Q: Sequence[float], mode: str) -> Tuple[np.ndarray, np.ndarray]:
    ps = patchset_for_spin(patchsets, spin)
    Q = np.asarray(Q, dtype=float)
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
            raise ValueError("mode must be one of {'Q_minus_k', 'k_plus_Q', 'k_minus_Q'}")
        idxs[p], residuals[p] = find_shifted_patch_index(patchsets, spin, target)
    return idxs, residuals


def _patch_energies(ps) -> np.ndarray:
    return np.array([float(p.energy) for p in ps.patches], dtype=float)


def build_gamma_accessor(gamma: GammaInput) -> Callable[[int, str, int, str, int, str, int, str], complex]:
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
        return complex(block[p1, p2, p3, p4])

    return accessor


def physical_propagator(iw: np.ndarray, energy: float, sigma: Optional[np.ndarray] = None) -> np.ndarray:
    if sigma is None:
        return 1.0 / (1j * iw - energy)
    return 1.0 / (1j * iw - energy - sigma)


def d_physical_propagator_dT_fixed_sigma(n: np.ndarray, temperature: float, energy: float, *, sign: int = +1, sigma: Optional[np.ndarray] = None) -> np.ndarray:
    w = (2.0 * n + 1.0) * np.pi * temperature
    iw = sign * w
    dw_dT = (2.0 * n + 1.0) * np.pi
    g = physical_propagator(iw, energy, sigma=sigma)
    return -(1j * sign * dw_dT) * (g ** 2)


def bubble_dot_pp(energy_a: float, energy_b: float, config: FlowConfig, *, sigma_a: Optional[np.ndarray] = None, sigma_b: Optional[np.ndarray] = None) -> complex:
    T = config.temperature
    n = np.arange(-config.nfreq, config.nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * T
    g_a = physical_propagator(w, energy_a, sigma=sigma_a)
    g_b = physical_propagator(-w, energy_b, sigma=sigma_b)
    dg_a = d_physical_propagator_dT_fixed_sigma(n, T, energy_a, sign=+1, sigma=sigma_a)
    dg_b = d_physical_propagator_dT_fixed_sigma(n, T, energy_b, sign=-1, sigma=sigma_b)
    val = T * np.sum(dg_a * g_b + g_a * dg_b)
    if config.include_explicit_T_prefactor:
        val += np.sum(g_a * g_b)
    return complex(val)


def bubble_dot_ph(energy_a: float, energy_b: float, config: FlowConfig, *, sigma_a: Optional[np.ndarray] = None, sigma_b: Optional[np.ndarray] = None) -> complex:
    T = config.temperature
    n = np.arange(-config.nfreq, config.nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * T
    g_a = physical_propagator(w, energy_a, sigma=sigma_a)
    g_b = physical_propagator(w, energy_b, sigma=sigma_b)
    dg_a = d_physical_propagator_dT_fixed_sigma(n, T, energy_a, sign=+1, sigma=sigma_a)
    dg_b = d_physical_propagator_dT_fixed_sigma(n, T, energy_b, sign=+1, sigma=sigma_b)
    val = T * np.sum(dg_a * g_b + g_a * dg_b)
    if config.include_explicit_T_prefactor:
        val += np.sum(g_a * g_b)
    return complex(val)


def _build_pp_internal_cache(patchsets: PatchSetMap, Q: Sequence[float], config: FlowConfig):
    cache = {}
    for sa, sb in available_internal_spin_pairs(patchsets):
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        partner, residual = shifted_patch_map(patchsets, sb, Q, mode="Q_minus_k")
        weights = np.array([bubble_dot_pp(eps_a[p], eps_b_all[partner[p]], config) for p in range(psa.Npatch)], dtype=complex)
        cache[(sa, sb)] = {
            "partner": partner,
            "residual": residual,
            "weights": weights,
        }
    return cache


def _build_ph_internal_cache(patchsets: PatchSetMap, Q: Sequence[float], config: FlowConfig):
    cache = {}
    for sa, sb in available_internal_spin_pairs(patchsets):
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        partner, residual = shifted_patch_map(patchsets, sb, Q, mode="k_plus_Q")
        weights = np.array([bubble_dot_ph(eps_a[p], eps_b_all[partner[p]], config) for p in range(psa.Npatch)], dtype=complex)
        cache[(sa, sb)] = {
            "partner": partner,
            "residual": residual,
            "weights": weights,
        }
    return cache


def _normalize_allowed_spin_blocks(allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]]) -> Optional[FrozenSet[SpinBlock]]:
    if allowed_spin_blocks is None:
        return None
    return frozenset(tuple(normalize_spin(x) for x in blk) for blk in allowed_spin_blocks)


def _allowed(block: SpinBlock, allowed_spin_blocks: Optional[FrozenSet[SpinBlock]]) -> bool:
    if allowed_spin_blocks is None:
        return True
    return block in allowed_spin_blocks


def compute_pp_kernel(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "dn"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s2 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s3, s4 = s1, s2
    else:
        s3, s4 = map(normalize_spin, outgoing_spins)

    ps_in = patchset_for_spin(patchsets, s1)
    ps_out = patchset_for_spin(patchsets, s3)
    partner_in, resid_in = shifted_patch_map(patchsets, s2, Q, mode="Q_minus_k")
    partner_out, resid_out = shifted_patch_map(patchsets, s4, Q, mode="Q_minus_k")
    internal = _build_pp_internal_cache(patchsets, Q, config)

    if ps_in.Npatch != ps_out.Npatch:
        raise ValueError("pp kernel requires matching patch counts for first incoming/outgoing legs")

    N = ps_in.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.zeros((N, N), dtype=float)

    active_pairs = []
    for (sa, sb), info in internal.items():
        blk_left = (s1, s2, sa, sb)
        blk_right = (sa, sb, s3, s4)
        if not (_allowed(blk_left, allowed) and _allowed(blk_right, allowed)):
            continue
        active_pairs.append(((sa, sb), info))

    for pout in range(N):
        p3 = pout
        p4 = int(partner_out[pout])
        for pin in range(N):
            p1 = pin
            p2 = int(partner_in[pin])
            acc = 0.0 + 0.0j
            max_resid = max(resid_in[pin], resid_out[pout])
            for (sa, sb), info in active_pairs:
                partner = info["partner"]
                weights = info["weights"]
                for a in range(patchset_for_spin(patchsets, sa).Npatch):
                    b = int(partner[a])
                    acc += weights[a] * gamma_fn(p1, s1, p2, s2, a, sa, b, sb) * gamma_fn(a, sa, b, sb, p3, s3, p4, s4)
                    max_resid = max(max_resid, info["residual"][a])
            K[pout, pin] = acc
            residuals[pout, pin] = max_resid

    return ChannelKernel(
        name="pp_kernel",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=np.asarray(partner_out, dtype=int),
        col_partner_patches=np.asarray(partner_in, dtype=int),
        row_spins=(s3, s4),
        col_spins=(s1, s2),
        residuals=residuals,
    )


def compute_ph_kernel(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "up"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s3 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s4, s2 = s1, s3
    else:
        s4, s2 = map(normalize_spin, outgoing_spins)

    ps1 = patchset_for_spin(patchsets, s1)
    ps4 = patchset_for_spin(patchsets, s4)
    kplus_in, resid_in = shifted_patch_map(patchsets, s3, Q, mode="k_plus_Q")
    kplus_out, resid_out = shifted_patch_map(patchsets, s2, Q, mode="k_plus_Q")
    internal = _build_ph_internal_cache(patchsets, Q, config)

    if ps1.Npatch != ps4.Npatch:
        raise ValueError("ph-direct kernel requires matching patch counts for legs 1 and 4")

    N = ps1.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.zeros((N, N), dtype=float)

    active_pairs = []
    for (sa, sb), info in internal.items():
        term1_left = (s1, sa, s3, sb)
        term1_right = (sb, s2, sa, s4)
        term2_left = (s1, sb, s3, sa)
        term2_right = (sa, s2, sb, s4)
        keep1 = _allowed(term1_left, allowed) and _allowed(term1_right, allowed)
        keep2 = _allowed(term2_left, allowed) and _allowed(term2_right, allowed)
        if keep1 or keep2:
            active_pairs.append(((sa, sb), info, keep1, keep2))

    for pout in range(N):
        p4 = pout
        p2 = int(kplus_out[pout])
        for pin in range(N):
            p1 = pin
            p3 = int(kplus_in[pin])
            acc = 0.0 + 0.0j
            max_resid = max(resid_in[pin], resid_out[pout])
            for (sa, sb), info, keep1, keep2 in active_pairs:
                partner = info["partner"]
                weights = info["weights"]
                for a in range(patchset_for_spin(patchsets, sa).Npatch):
                    b = int(partner[a])
                    term1 = 0.0 + 0.0j
                    term2 = 0.0 + 0.0j
                    if keep1:
                        term1 = gamma_fn(p1, s1, a, sa, p3, s3, b, sb) * gamma_fn(b, sb, p2, s2, a, sa, p4, s4)
                    if keep2:
                        term2 = gamma_fn(p1, s1, b, sb, p3, s3, a, sa) * gamma_fn(a, sa, p2, s2, b, sb, p4, s4)
                    acc -= weights[a] * (term1 + term2)
                    max_resid = max(max_resid, info["residual"][a])
            K[pout, pin] = acc
            residuals[pout, pin] = max_resid

    return ChannelKernel(
        name="ph_direct_kernel",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=np.asarray(kplus_out, dtype=int),
        col_partner_patches=np.asarray(kplus_in, dtype=int),
        row_spins=(s4, s2),
        col_spins=(s1, s3),
        residuals=residuals,
    )


def compute_phc_kernel(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "up"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s2 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s3, s4 = s2, s1
    else:
        s3, s4 = map(normalize_spin, outgoing_spins)

    ps1 = patchset_for_spin(patchsets, s1)
    ps2 = patchset_for_spin(patchsets, s2)
    kplus_out, resid_out = shifted_patch_map(patchsets, s3, Q, mode="k_plus_Q")
    kminus_in, resid_in = shifted_patch_map(patchsets, s4, Q, mode="k_minus_Q")
    internal = _build_ph_internal_cache(patchsets, Q, config)

    if ps1.Npatch != ps2.Npatch:
        raise ValueError("ph-crossed kernel requires matching patch counts for legs 1 and 2")

    N = ps1.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.zeros((N, N), dtype=float)

    active_pairs = []
    for (sa, sb), info in internal.items():
        term1_left = (s1, sa, s4, sb)
        term1_right = (sb, s2, sa, s3)
        term2_left = (s1, sb, s4, sa)
        term2_right = (sa, s2, sb, s3)
        keep1 = _allowed(term1_left, allowed) and _allowed(term1_right, allowed)
        keep2 = _allowed(term2_left, allowed) and _allowed(term2_right, allowed)
        if keep1 or keep2:
            active_pairs.append(((sa, sb), info, keep1, keep2))

    for pout in range(N):
        p2 = pout
        p3 = int(kplus_out[pout])
        for pin in range(N):
            p1 = pin
            p4 = int(kminus_in[pin])
            acc = 0.0 + 0.0j
            max_resid = max(resid_in[pin], resid_out[pout])
            for (sa, sb), info, keep1, keep2 in active_pairs:
                partner = info["partner"]
                weights = info["weights"]
                for a in range(patchset_for_spin(patchsets, sa).Npatch):
                    b = int(partner[a])
                    term1 = 0.0 + 0.0j
                    term2 = 0.0 + 0.0j
                    if keep1:
                        term1 = gamma_fn(p1, s1, a, sa, p4, s4, b, sb) * gamma_fn(b, sb, p2, s2, a, sa, p3, s3)
                    if keep2:
                        term2 = gamma_fn(p1, s1, b, sb, p4, s4, a, sa) * gamma_fn(a, sa, p2, s2, b, sb, p3, s3)
                    acc += weights[a] * (term1 + term2)
                    max_resid = max(max_resid, info["residual"][a])
            K[pout, pin] = acc
            residuals[pout, pin] = max_resid

    return ChannelKernel(
        name="ph_crossed_kernel",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=np.asarray(kplus_out, dtype=int),
        col_partner_patches=np.asarray(kminus_in, dtype=int),
        row_spins=(s2, s3),
        col_spins=(s1, s4),
        residuals=residuals,
    )


def compute_pp_kernel_fast(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "dn"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s2 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s3, s4 = s1, s2
    else:
        s3, s4 = map(normalize_spin, outgoing_spins)

    ps_in = patchset_for_spin(patchsets, s1)
    ps_out = patchset_for_spin(patchsets, s3)
    partner_in, resid_in = shifted_patch_map(patchsets, s2, Q, mode="Q_minus_k")
    partner_out, resid_out = shifted_patch_map(patchsets, s4, Q, mode="Q_minus_k")
    internal = _build_pp_internal_cache(patchsets, Q, config)

    if ps_in.Npatch != ps_out.Npatch:
        raise ValueError("pp kernel requires matching patch counts for first incoming/outgoing legs")

    N = ps_in.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.maximum.outer(resid_out, resid_in)

    for (sa, sb), info in internal.items():
        blk_left = (s1, s2, sa, sb)
        blk_right = (sa, sb, s3, s4)
        if not (_allowed(blk_left, allowed) and _allowed(blk_right, allowed)):
            continue

        Na = patchset_for_spin(patchsets, sa).Npatch
        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)

        L = np.empty((Na, N), dtype=complex)
        R = np.empty((Na, N), dtype=complex)
        for a in range(Na):
            b = int(partner[a])
            for pin in range(N):
                p1 = pin
                p2 = int(partner_in[pin])
                L[a, pin] = gamma_fn(p1, s1, p2, s2, a, sa, b, sb)
            for pout in range(N):
                p3 = pout
                p4 = int(partner_out[pout])
                R[a, pout] = gamma_fn(a, sa, b, sb, p3, s3, p4, s4)

        K += (weights[:, None] * R).T @ L
        residuals = np.maximum(residuals, np.max(np.asarray(info["residual"], dtype=float)))

    return ChannelKernel(
        name="pp_kernel_fast",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=np.asarray(partner_out, dtype=int),
        col_partner_patches=np.asarray(partner_in, dtype=int),
        row_spins=(s3, s4),
        col_spins=(s1, s2),
        residuals=residuals,
    )


def compute_ph_kernel_fast(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "up"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s3 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s4, s2 = s1, s3
    else:
        s4, s2 = map(normalize_spin, outgoing_spins)

    ps1 = patchset_for_spin(patchsets, s1)
    ps4 = patchset_for_spin(patchsets, s4)
    kplus_in, resid_in = shifted_patch_map(patchsets, s3, Q, mode="k_plus_Q")
    kplus_out, resid_out = shifted_patch_map(patchsets, s2, Q, mode="k_plus_Q")
    internal = _build_ph_internal_cache(patchsets, Q, config)

    if ps1.Npatch != ps4.Npatch:
        raise ValueError("ph-direct kernel requires matching patch counts for legs 1 and 4")

    N = ps1.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.maximum.outer(resid_out, resid_in)

    for (sa, sb), info in internal.items():
        term1_left = (s1, sa, s3, sb)
        term1_right = (sb, s2, sa, s4)
        term2_left = (s1, sb, s3, sa)
        term2_right = (sa, s2, sb, s4)
        keep1 = _allowed(term1_left, allowed) and _allowed(term1_right, allowed)
        keep2 = _allowed(term2_left, allowed) and _allowed(term2_right, allowed)
        if not (keep1 or keep2):
            continue

        Na = patchset_for_spin(patchsets, sa).Npatch
        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)

        if keep1:
            L1 = np.empty((Na, N), dtype=complex)
            R1 = np.empty((Na, N), dtype=complex)
        if keep2:
            L2 = np.empty((Na, N), dtype=complex)
            R2 = np.empty((Na, N), dtype=complex)

        for a in range(Na):
            b = int(partner[a])
            for pin in range(N):
                p1 = pin
                p3 = int(kplus_in[pin])
                if keep1:
                    L1[a, pin] = gamma_fn(p1, s1, a, sa, p3, s3, b, sb)
                if keep2:
                    L2[a, pin] = gamma_fn(p1, s1, b, sb, p3, s3, a, sa)
            for pout in range(N):
                p4 = pout
                p2 = int(kplus_out[pout])
                if keep1:
                    R1[a, pout] = gamma_fn(b, sb, p2, s2, a, sa, p4, s4)
                if keep2:
                    R2[a, pout] = gamma_fn(a, sa, p2, s2, b, sb, p4, s4)

        if keep1:
            K -= (weights[:, None] * R1).T @ L1
        if keep2:
            K -= (weights[:, None] * R2).T @ L2
        residuals = np.maximum(residuals, np.max(np.asarray(info["residual"], dtype=float)))

    return ChannelKernel(
        name="ph_direct_kernel_fast",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=np.asarray(kplus_out, dtype=int),
        col_partner_patches=np.asarray(kplus_in, dtype=int),
        row_spins=(s4, s2),
        col_spins=(s1, s3),
        residuals=residuals,
    )


def compute_phc_kernel_fast(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "up"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s2 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s3, s4 = s2, s1
    else:
        s3, s4 = map(normalize_spin, outgoing_spins)

    ps1 = patchset_for_spin(patchsets, s1)
    ps2 = patchset_for_spin(patchsets, s2)
    kplus_out, resid_out = shifted_patch_map(patchsets, s3, Q, mode="k_plus_Q")
    kminus_in, resid_in = shifted_patch_map(patchsets, s4, Q, mode="k_minus_Q")
    internal = _build_ph_internal_cache(patchsets, Q, config)

    if ps1.Npatch != ps2.Npatch:
        raise ValueError("ph-crossed kernel requires matching patch counts for legs 1 and 2")

    N = ps1.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.maximum.outer(resid_out, resid_in)

    for (sa, sb), info in internal.items():
        term1_left = (s1, sa, s4, sb)
        term1_right = (sb, s2, sa, s3)
        term2_left = (s1, sb, s4, sa)
        term2_right = (sa, s2, sb, s3)
        keep1 = _allowed(term1_left, allowed) and _allowed(term1_right, allowed)
        keep2 = _allowed(term2_left, allowed) and _allowed(term2_right, allowed)
        if not (keep1 or keep2):
            continue

        Na = patchset_for_spin(patchsets, sa).Npatch
        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)

        if keep1:
            L1 = np.empty((Na, N), dtype=complex)
            R1 = np.empty((Na, N), dtype=complex)
        if keep2:
            L2 = np.empty((Na, N), dtype=complex)
            R2 = np.empty((Na, N), dtype=complex)

        for a in range(Na):
            b = int(partner[a])
            for pin in range(N):
                p1 = pin
                p4 = int(kminus_in[pin])
                if keep1:
                    L1[a, pin] = gamma_fn(p1, s1, a, sa, p4, s4, b, sb)
                if keep2:
                    L2[a, pin] = gamma_fn(p1, s1, b, sb, p4, s4, a, sa)
            for pout in range(N):
                p2 = pout
                p3 = int(kplus_out[pout])
                if keep1:
                    R1[a, pout] = gamma_fn(b, sb, p2, s2, a, sa, p3, s3)
                if keep2:
                    R2[a, pout] = gamma_fn(a, sa, p2, s2, b, sb, p3, s3)

        if keep1:
            K += (weights[:, None] * R1).T @ L1
        if keep2:
            K += (weights[:, None] * R2).T @ L2
        residuals = np.maximum(residuals, np.max(np.asarray(info["residual"], dtype=float)))

    return ChannelKernel(
        name="ph_crossed_kernel_fast",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=np.asarray(kplus_out, dtype=int),
        col_partner_patches=np.asarray(kminus_in, dtype=int),
        row_spins=(s2, s3),
        col_spins=(s1, s4),
        residuals=residuals,
    )



# ===== Stage-2 vectorized bubble and Q-level cache helpers =====

def _patch_energies(ps) -> np.ndarray:
    return np.array([float(p.energy) for p in ps.patches], dtype=float)


@lru_cache(maxsize=128)
def _matsubara_grid_cached(temperature: float, nfreq: int):
    n = np.arange(-nfreq, nfreq + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.pi * temperature
    dw_dT = (2.0 * n + 1.0) * np.pi
    return n, w, dw_dT


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


InternalCache = Dict[Tuple[str, str], Dict[str, np.ndarray]]
ShiftMap = Tuple[np.ndarray, np.ndarray]


def build_pp_internal_cache_vec(
    patchsets: PatchSetMap,
    Q: Sequence[float],
    config: FlowConfig,
    *,
    shift_cache: Optional[Mapping[Tuple[str, str], ShiftMap]] = None,
) -> InternalCache:
    cache: InternalCache = {}
    for sa, sb in available_internal_spin_pairs(patchsets):
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        if shift_cache is not None and (sa, sb) in shift_cache:
            partner, residual = shift_cache[(sa, sb)]
        else:
            partner, residual = shifted_patch_map(patchsets, sb, Q, mode="Q_minus_k")
        partner = np.asarray(partner, dtype=int)
        residual = np.asarray(residual, dtype=float)
        weights = _bubble_dot_pp_vec(eps_a, eps_b_all[partner], config)
        cache[(sa, sb)] = {
            "partner": partner,
            "residual": residual,
            "weights": np.asarray(weights, dtype=complex),
        }
    return cache



def build_ph_internal_cache_vec(
    patchsets: PatchSetMap,
    Q: Sequence[float],
    config: FlowConfig,
    *,
    shift_cache: Optional[Mapping[Tuple[str, str], ShiftMap]] = None,
) -> InternalCache:
    cache: InternalCache = {}
    for sa, sb in available_internal_spin_pairs(patchsets):
        psa = patchset_for_spin(patchsets, sa)
        psb = patchset_for_spin(patchsets, sb)
        eps_a = _patch_energies(psa)
        eps_b_all = _patch_energies(psb)
        if shift_cache is not None and (sa, sb) in shift_cache:
            partner, residual = shift_cache[(sa, sb)]
        else:
            partner, residual = shifted_patch_map(patchsets, sb, Q, mode="k_plus_Q")
        partner = np.asarray(partner, dtype=int)
        residual = np.asarray(residual, dtype=float)
        weights = _bubble_dot_ph_vec(eps_a, eps_b_all[partner], config)
        cache[(sa, sb)] = {
            "partner": partner,
            "residual": residual,
            "weights": np.asarray(weights, dtype=complex),
        }
    return cache



def compute_pp_kernel_fast2(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "dn"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
    internal_cache: Optional[InternalCache] = None,
    partner_in_resid: Optional[ShiftMap] = None,
    partner_out_resid: Optional[ShiftMap] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s2 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s3, s4 = s1, s2
    else:
        s3, s4 = map(normalize_spin, outgoing_spins)

    ps_in = patchset_for_spin(patchsets, s1)
    ps_out = patchset_for_spin(patchsets, s3)
    if ps_in.Npatch != ps_out.Npatch:
        raise ValueError("pp kernel requires matching patch counts for first incoming/outgoing legs")

    if partner_in_resid is None:
        partner_in, resid_in = shifted_patch_map(patchsets, s2, Q, mode="Q_minus_k")
    else:
        partner_in, resid_in = partner_in_resid
    if partner_out_resid is None:
        partner_out, resid_out = shifted_patch_map(patchsets, s4, Q, mode="Q_minus_k")
    else:
        partner_out, resid_out = partner_out_resid
    if internal_cache is None:
        internal_cache = build_pp_internal_cache_vec(patchsets, Q, config)

    partner_in = np.asarray(partner_in, dtype=int)
    resid_in = np.asarray(resid_in, dtype=float)
    partner_out = np.asarray(partner_out, dtype=int)
    resid_out = np.asarray(resid_out, dtype=float)

    N = ps_in.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.maximum(resid_out[:, None], resid_in[None, :])

    for (sa, sb), info in internal_cache.items():
        blk_left = (s1, s2, sa, sb)
        blk_right = (sa, sb, s3, s4)
        if not (_allowed(blk_left, allowed) and _allowed(blk_right, allowed)):
            continue

        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)
        Na = len(partner)

        L = np.empty((Na, N), dtype=complex)
        R = np.empty((Na, N), dtype=complex)
        for a in range(Na):
            b = int(partner[a])
            for pin in range(N):
                L[a, pin] = gamma_fn(pin, s1, int(partner_in[pin]), s2, a, sa, b, sb)
            for pout in range(N):
                R[a, pout] = gamma_fn(a, sa, b, sb, pout, s3, int(partner_out[pout]), s4)

        K += (weights[:, None] * R).T @ L
        residuals = np.maximum(residuals, float(np.max(np.asarray(info["residual"], dtype=float))))

    return ChannelKernel(
        name="pp_kernel_fast2",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=partner_out,
        col_partner_patches=partner_in,
        row_spins=(s3, s4),
        col_spins=(s1, s2),
        residuals=residuals,
    )



def compute_ph_kernel_fast2(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "up"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
    internal_cache: Optional[InternalCache] = None,
    partner_in_resid: Optional[ShiftMap] = None,
    partner_out_resid: Optional[ShiftMap] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s3 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s4, s2 = s1, s3
    else:
        s4, s2 = map(normalize_spin, outgoing_spins)

    ps1 = patchset_for_spin(patchsets, s1)
    ps4 = patchset_for_spin(patchsets, s4)
    if ps1.Npatch != ps4.Npatch:
        raise ValueError("ph-direct kernel requires matching patch counts for legs 1 and 4")

    if partner_in_resid is None:
        kplus_in, resid_in = shifted_patch_map(patchsets, s3, Q, mode="k_plus_Q")
    else:
        kplus_in, resid_in = partner_in_resid
    if partner_out_resid is None:
        kplus_out, resid_out = shifted_patch_map(patchsets, s2, Q, mode="k_plus_Q")
    else:
        kplus_out, resid_out = partner_out_resid
    if internal_cache is None:
        internal_cache = build_ph_internal_cache_vec(patchsets, Q, config)

    kplus_in = np.asarray(kplus_in, dtype=int)
    resid_in = np.asarray(resid_in, dtype=float)
    kplus_out = np.asarray(kplus_out, dtype=int)
    resid_out = np.asarray(resid_out, dtype=float)

    N = ps1.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.maximum(resid_out[:, None], resid_in[None, :])

    for (sa, sb), info in internal_cache.items():
        term1_left = (s1, sa, s3, sb)
        term1_right = (sb, s2, sa, s4)
        term2_left = (s1, sb, s3, sa)
        term2_right = (sa, s2, sb, s4)
        keep1 = _allowed(term1_left, allowed) and _allowed(term1_right, allowed)
        keep2 = _allowed(term2_left, allowed) and _allowed(term2_right, allowed)
        if not (keep1 or keep2):
            continue

        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)
        Na = len(partner)

        if keep1:
            L1 = np.empty((Na, N), dtype=complex)
            R1 = np.empty((Na, N), dtype=complex)
        if keep2:
            L2 = np.empty((Na, N), dtype=complex)
            R2 = np.empty((Na, N), dtype=complex)

        for a in range(Na):
            b = int(partner[a])
            for pin in range(N):
                p1 = pin
                p3 = int(kplus_in[pin])
                if keep1:
                    L1[a, pin] = gamma_fn(p1, s1, a, sa, p3, s3, b, sb)
                if keep2:
                    L2[a, pin] = gamma_fn(p1, s1, b, sb, p3, s3, a, sa)
            for pout in range(N):
                p4 = pout
                p2 = int(kplus_out[pout])
                if keep1:
                    R1[a, pout] = gamma_fn(b, sb, p2, s2, a, sa, p4, s4)
                if keep2:
                    R2[a, pout] = gamma_fn(a, sa, p2, s2, b, sb, p4, s4)

        if keep1:
            K -= (weights[:, None] * R1).T @ L1
        if keep2:
            K -= (weights[:, None] * R2).T @ L2
        residuals = np.maximum(residuals, float(np.max(np.asarray(info["residual"], dtype=float))))

    return ChannelKernel(
        name="ph_direct_kernel_fast2",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=kplus_out,
        col_partner_patches=kplus_in,
        row_spins=(s4, s2),
        col_spins=(s1, s3),
        residuals=residuals,
    )



def compute_phc_kernel_fast2(
    gamma: GammaInput,
    patchsets: PatchSetMap,
    Q: Sequence[float],
    *,
    incoming_spins: Tuple[SpinLike, SpinLike] = ("up", "up"),
    outgoing_spins: Optional[Tuple[SpinLike, SpinLike]] = None,
    config: Optional[FlowConfig] = None,
    allowed_spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
    internal_cache: Optional[InternalCache] = None,
    partner_in_resid: Optional[ShiftMap] = None,
    partner_out_resid: Optional[ShiftMap] = None,
) -> ChannelKernel:
    if config is None:
        raise ValueError("config must be provided")
    gamma_fn = build_gamma_accessor(gamma)
    allowed = _normalize_allowed_spin_blocks(allowed_spin_blocks)
    Q = np.asarray(Q, dtype=float)
    s1, s2 = map(normalize_spin, incoming_spins)
    if outgoing_spins is None:
        s3, s4 = s2, s1
    else:
        s3, s4 = map(normalize_spin, outgoing_spins)

    ps1 = patchset_for_spin(patchsets, s1)
    ps2 = patchset_for_spin(patchsets, s2)
    if ps1.Npatch != ps2.Npatch:
        raise ValueError("ph-crossed kernel requires matching patch counts for legs 1 and 2")

    if partner_out_resid is None:
        kplus_out, resid_out = shifted_patch_map(patchsets, s3, Q, mode="k_plus_Q")
    else:
        kplus_out, resid_out = partner_out_resid
    if partner_in_resid is None:
        kminus_in, resid_in = shifted_patch_map(patchsets, s4, Q, mode="k_minus_Q")
    else:
        kminus_in, resid_in = partner_in_resid
    if internal_cache is None:
        internal_cache = build_ph_internal_cache_vec(patchsets, Q, config)

    kplus_out = np.asarray(kplus_out, dtype=int)
    resid_out = np.asarray(resid_out, dtype=float)
    kminus_in = np.asarray(kminus_in, dtype=int)
    resid_in = np.asarray(resid_in, dtype=float)

    N = ps1.Npatch
    K = np.zeros((N, N), dtype=complex)
    residuals = np.maximum(resid_out[:, None], resid_in[None, :])

    for (sa, sb), info in internal_cache.items():
        term1_left = (s1, sa, s4, sb)
        term1_right = (sb, s2, sa, s3)
        term2_left = (s1, sb, s4, sa)
        term2_right = (sa, s2, sb, s3)
        keep1 = _allowed(term1_left, allowed) and _allowed(term1_right, allowed)
        keep2 = _allowed(term2_left, allowed) and _allowed(term2_right, allowed)
        if not (keep1 or keep2):
            continue

        partner = np.asarray(info["partner"], dtype=int)
        weights = np.asarray(info["weights"], dtype=complex)
        Na = len(partner)

        if keep1:
            L1 = np.empty((Na, N), dtype=complex)
            R1 = np.empty((Na, N), dtype=complex)
        if keep2:
            L2 = np.empty((Na, N), dtype=complex)
            R2 = np.empty((Na, N), dtype=complex)

        for a in range(Na):
            b = int(partner[a])
            for pin in range(N):
                p1 = pin
                p4 = int(kminus_in[pin])
                if keep1:
                    L1[a, pin] = gamma_fn(p1, s1, a, sa, p4, s4, b, sb)
                if keep2:
                    L2[a, pin] = gamma_fn(p1, s1, b, sb, p4, s4, a, sa)
            for pout in range(N):
                p2 = pout
                p3 = int(kplus_out[pout])
                if keep1:
                    R1[a, pout] = gamma_fn(b, sb, p2, s2, a, sa, p3, s3)
                if keep2:
                    R2[a, pout] = gamma_fn(a, sa, p2, s2, b, sb, p3, s3)

        if keep1:
            K += (weights[:, None] * R1).T @ L1
        if keep2:
            K += (weights[:, None] * R2).T @ L2
        residuals = np.maximum(residuals, float(np.max(np.asarray(info["residual"], dtype=float))))

    return ChannelKernel(
        name="ph_crossed_kernel_fast2",
        Q=Q,
        matrix=K,
        row_patches=np.arange(N, dtype=int),
        col_patches=np.arange(N, dtype=int),
        row_partner_patches=kplus_out,
        col_partner_patches=kminus_in,
        row_spins=(s2, s3),
        col_spins=(s1, s4),
        residuals=residuals,
    )


__all__ = [
    "ChannelKernel",
    "FlowConfig",
    "GammaInput",
    "PatchSetMap",
    "SpinBlock",
    "SpinLike",
    "build_gamma_accessor",
    "normalize_spin",
    "patchset_for_spin",
    "shifted_patch_map",
    "has_patchset",
    "physical_propagator",
    "compute_pp_kernel",
    "compute_ph_kernel",
    "compute_phc_kernel",
    "build_pp_internal_cache_vec",
    "build_ph_internal_cache_vec",
    "compute_pp_kernel_fast2",
    "compute_ph_kernel_fast2",
    "compute_phc_kernel_fast2",
]
