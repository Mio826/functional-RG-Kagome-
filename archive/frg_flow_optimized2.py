from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import frg_flow_optimized as base
from frg_kernel_optimized2 import (
    FlowConfig,
    available_internal_spin_pairs,
    build_ph_internal_cache_vec,
    build_pp_internal_cache_vec,
    compute_ph_kernel_fast2,
    compute_phc_kernel_fast2,
    compute_pp_kernel_fast2,
    normalize_spin,
)

SpinLike = base.SpinLike
PatchSetMap = base.PatchSetMap
SpinBlock = base.SpinBlock
ChannelKey = base.ChannelKey


class FRGFlowSolver(base.FRGFlowSolver):
    """Stage-2 optimized solver.

    Adds two further optimizations on top of the first optimized solver:
    1) Q-level internal cache build once per channel/Q and reuse across all external spin blocks.
    2) Vectorized bubble frequency sums inside the kernel-side internal cache builder.

    Physics and flow equations remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._precompute_shift_maps()

    def _precompute_shift_maps(self) -> None:
        spins = self._all_spins_present()
        self._pp_qminus: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._ph_kplus: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._phc_kplus: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._phc_kminus: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}

        for iq, Q in enumerate(self.pp_grid.q_list):
            Q = np.asarray(Q, dtype=float)
            for s in spins:
                self._pp_qminus[(iq, s)] = base.shifted_patch_map(self.patchsets, s, Q, mode="Q_minus_k")

        for iq, Q in enumerate(self.phd_grid.q_list):
            Q = np.asarray(Q, dtype=float)
            for s in spins:
                self._ph_kplus[(iq, s)] = base.shifted_patch_map(self.patchsets, s, Q, mode="k_plus_Q")

        for iq, Q in enumerate(self.phc_grid.q_list):
            Q = np.asarray(Q, dtype=float)
            for s in spins:
                self._phc_kplus[(iq, s)] = base.shifted_patch_map(self.patchsets, s, Q, mode="k_plus_Q")
                self._phc_kminus[(iq, s)] = base.shifted_patch_map(self.patchsets, s, Q, mode="k_minus_Q")

    def _build_q_level_internal_caches(self, T: float):
        cfg = self._flow_config(T)
        pp_internal_by_iq = {}
        ph_internal_by_iq = {}
        for iq, Q in enumerate(self.pp_grid.q_list):
            shift_cache = {(sa, sb): self._pp_qminus[(iq, sb)] for sa, sb in available_internal_spin_pairs(self.patchsets)}
            pp_internal_by_iq[iq] = build_pp_internal_cache_vec(self.patchsets, Q, cfg, shift_cache=shift_cache)
        for iq, Q in enumerate(self.phd_grid.q_list):
            shift_cache = {(sa, sb): self._ph_kplus[(iq, sb)] for sa, sb in available_internal_spin_pairs(self.patchsets)}
            ph_internal_by_iq[iq] = build_ph_internal_cache_vec(self.patchsets, Q, cfg, shift_cache=shift_cache)
        return cfg, pp_internal_by_iq, ph_internal_by_iq

    def compute_channel_rhs(self, T: float):
        gamma = self._fast_gamma
        cfg, pp_internal_by_iq, ph_internal_by_iq = self._build_q_level_internal_caches(T)

        rhs_pp = self._empty_channel_store(self.pp_grid)
        rhs_phd = self._empty_channel_store(self.phd_grid)
        rhs_phc = self._empty_channel_store(self.phc_grid)

        for iq, Q in enumerate(self.pp_grid.q_list):
            internal_cache = pp_internal_by_iq[iq]
            for s1, s2, s3, s4 in self.spin_blocks:
                ker = compute_pp_kernel_fast2(
                    gamma,
                    self.patchsets,
                    Q,
                    incoming_spins=(s1, s2),
                    outgoing_spins=(s3, s4),
                    config=cfg,
                    allowed_spin_blocks=self.allowed_spin_blocks,
                    internal_cache=internal_cache,
                    partner_in_resid=self._pp_qminus[(iq, s2)],
                    partner_out_resid=self._pp_qminus[(iq, s4)],
                )
                rhs_pp[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)

        for iq, Q in enumerate(self.phd_grid.q_list):
            internal_cache = ph_internal_by_iq[iq]
            for s1, s2, s3, s4 in self.spin_blocks:
                ker = compute_ph_kernel_fast2(
                    gamma,
                    self.patchsets,
                    Q,
                    incoming_spins=(s1, s3),
                    outgoing_spins=(s4, s2),
                    config=cfg,
                    allowed_spin_blocks=self.allowed_spin_blocks,
                    internal_cache=internal_cache,
                    partner_in_resid=self._ph_kplus[(iq, s3)],
                    partner_out_resid=self._ph_kplus[(iq, s2)],
                )
                rhs_phd[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)

        for iq, Q in enumerate(self.phc_grid.q_list):
            shift_cache = {(sa, sb): self._phc_kplus[(iq, sb)] for sa, sb in available_internal_spin_pairs(self.patchsets)}
            internal_cache = build_ph_internal_cache_vec(self.patchsets, Q, cfg, shift_cache=shift_cache)
            for s1, s2, s3, s4 in self.spin_blocks:
                ker = compute_phc_kernel_fast2(
                    gamma,
                    self.patchsets,
                    Q,
                    incoming_spins=(s1, s2),
                    outgoing_spins=(s3, s4),
                    config=cfg,
                    allowed_spin_blocks=self.allowed_spin_blocks,
                    internal_cache=internal_cache,
                    partner_in_resid=self._phc_kminus[(iq, s4)],
                    partner_out_resid=self._phc_kplus[(iq, s3)],
                )
                rhs_phc[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)

        return rhs_pp, rhs_phd, rhs_phc

    def current_gamma_accessor(self):
        return self._fast_gamma


__all__ = [
    "FRGFlowSolver",
    "FlowConfig",
    "normalize_spin",
]
