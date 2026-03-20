from __future__ import annotations

from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frg_kernel_optimized import (
    ChannelKernel,
    FlowConfig,
    GammaInput,
    PatchSetMap,
    SpinBlock,
    SpinLike,
    _allowed,
    _normalize_allowed_spin_blocks,
    available_internal_spin_pairs,
    build_gamma_accessor,
    compute_pp_kernel,
    compute_ph_kernel,
    compute_phc_kernel,
    find_shifted_patch_index,
    has_patchset,
    normalize_spin,
    patchset_for_spin,
    physical_propagator,
    shifted_patch_map,
)


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
