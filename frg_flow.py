
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frg_kernel import (
    FlowConfig,
    normalize_spin,
    patchset_for_spin,
    has_patchset,
    available_internal_spin_pairs,
    build_pp_internal_cache_vec,
    build_ph_internal_cache_vec,
    canonicalize_q_for_patchsets,
    partner_map_from_q_index,
)

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
SpinBlock = Tuple[str, str, str, str]
GammaAccessor = Callable[[int, str, int, str, int, str, int, str], complex]


def canonical_spin_tuple(key: Tuple[SpinLike, SpinLike, SpinLike, SpinLike]) -> SpinBlock:
    return tuple(normalize_spin(x) for x in key)  # type: ignore[return-value]


def available_physical_spins(patchsets: PatchSetMap) -> List[str]:
    out: List[str] = []
    for s in ("up", "dn"):
        if has_patchset(patchsets, s):
            out.append(s)
    if not out:
        raise ValueError("No non-empty spin patch sets found.")
    return out


def default_spin_blocks(patchsets: PatchSetMap) -> List[SpinBlock]:
    spins = available_physical_spins(patchsets)
    blocks: List[SpinBlock] = []
    if "up" in spins:
        blocks.append(("up", "up", "up", "up"))
    if "dn" in spins:
        blocks.append(("dn", "dn", "dn", "dn"))
    if set(spins) == {"up", "dn"}:
        blocks.extend(
            [
                ("up", "dn", "up", "dn"),
                ("up", "dn", "dn", "up"),
                ("dn", "up", "up", "dn"),
                ("dn", "up", "dn", "up"),
            ]
        )
    return blocks


class BareVertexFromInteraction:
    def __init__(self, interaction: Any, patchsets: PatchSetMap):
        self.interaction = interaction
        self.patchsets = patchsets

    def __call__(self, p1: int, s1: str, p2: int, s2: str, p3: int, s3: str, p4: int, s4: str) -> complex:
        return complex(
            self.interaction.patch_vertex(
                self.patchsets,
                p1,
                s1,
                p2,
                s2,
                p3,
                s3,
                p4,
                s4,
                antisym=True,
                check_momentum=False,
            )
        )


@dataclass
class TransferGrid:
    patchsets: PatchSetMap
    q_list: List[np.ndarray]
    decimals: int = 10
    merge_tol_red: float = 5e-2

    def __post_init__(self) -> None:
        ref_spin = available_physical_spins(self.patchsets)[0]
        ps = patchset_for_spin(self.patchsets, ref_spin)
        self.b1 = np.asarray(ps.b1, dtype=float)
        self.b2 = np.asarray(ps.b2, dtype=float)
        self.q_list = [canonicalize_q_for_patchsets(self.patchsets, q) for q in self.q_list]
        self.rep_uv_list: List[np.ndarray] = []
        self.key_to_index: Dict[Tuple[float, float], int] = {}
        unique_qs: List[np.ndarray] = []
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
) -> List[np.ndarray]:
    ref_spin = available_physical_spins(patchsets)[0]
    ps = patchset_for_spin(patchsets, ref_spin)
    ks = _canonicalize_patch_k_array(np.asarray([p.k_cart for p in ps.patches], dtype=float), patchsets)

    raw_candidates: List[np.ndarray] = [np.zeros(2, dtype=float)]
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


@dataclass
class GammaTensorBlock:
    data: np.ndarray
    p4_index: np.ndarray
    p4_residual: np.ndarray


@dataclass
class FullVertexState:
    patchsets: PatchSetMap
    bare_gamma: GammaAccessor
    spin_blocks: List[SpinBlock]
    pp_grid: TransferGrid
    phd_grid: TransferGrid
    phc_grid: TransferGrid
    T: float
    gamma_blocks: Dict[SpinBlock, GammaTensorBlock] = field(default_factory=dict)

    def channel_norm(self) -> float:
        vals = [float(np.max(np.abs(block.data))) for block in self.gamma_blocks.values() if block.data.size]
        return max(vals) if vals else 0.0


@dataclass
class FlowStepRecord:
    step_index: int
    temperature: float
    dT: float
    channel_norm: float
    rhs_norm: float
    accepted_substeps: int
    max_rel_update: float
    instability: bool = False
    instability_reason: Optional[str] = None
    terminated_early: bool = False
    termination_reason: Optional[str] = None
    diagnosis_payload: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "temperature": float(self.temperature),
            "dT": float(self.dT),
            "channel_norm": float(self.channel_norm),
            "rhs_norm": float(self.rhs_norm),
            "accepted_substeps": int(self.accepted_substeps),
            "max_rel_update": float(self.max_rel_update),
            "instability": bool(self.instability),
            "instability_reason": self.instability_reason,
            "terminated_early": bool(self.terminated_early),
            "termination_reason": self.termination_reason,
            "diagnosis_payload": dict(self.diagnosis_payload),
        }


class FullVertexAccessor:
    def __init__(self, solver: "FRGFlowSolver"):
        self.solver = solver

    def __call__(self, p1: int, s1: str, p2: int, s2: str, p3: int, s3: str, p4: int, s4: str) -> complex:
        key = canonical_spin_tuple((s1, s2, s3, s4))
        block = self.solver.state.gamma_blocks.get(key)
        if block is None:
            return 0.0 + 0.0j
        p4_expected = int(block.p4_index[p1, p2, p3])
        if p4_expected < 0 or int(p4) != p4_expected:
            return 0.0 + 0.0j
        return complex(block.data[p1, p2, p3])


class FRGFlowSolver:
    def __init__(
        self,
        *,
        patchsets: PatchSetMap,
        bare_gamma: GammaAccessor,
        spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
        pp_Qs: Optional[Sequence[Sequence[float]]] = None,
        ph_Qs: Optional[Sequence[Sequence[float]]] = None,
        phc_Qs: Optional[Sequence[Sequence[float]]] = None,
        T_start: float = 0.5,
        T_stop: float = 0.02,
        n_steps: int = 24,
        temperature_grid: str = "log",
        nfreq: int = 128,
        include_explicit_T_prefactor: bool = True,
        max_relative_update: float = 0.15,
        min_substep_fraction: float = 1.0 / 128.0,
        channel_divergence_threshold: float = 1e3,
        diagnose_every: int = 1,
        q_merge_tol_red: float = 5e-2,
        q_key_decimals: int = 10,
    ) -> None:
        self.patchsets = patchsets
        self.bare_gamma = bare_gamma
        self.spin_blocks = [canonical_spin_tuple(x) for x in (spin_blocks or default_spin_blocks(patchsets))]
        self.allowed_spin_blocks = frozenset(self.spin_blocks)
        self.q_merge_tol_red = float(q_merge_tol_red)
        self.q_key_decimals = int(q_key_decimals)

        if pp_Qs is None:
            pp_Qs = build_unique_q_list(patchsets, mode="pp", decimals=self.q_key_decimals, merge_tol_red=self.q_merge_tol_red)
        if ph_Qs is None:
            ph_Qs = build_unique_q_list(patchsets, mode="ph", decimals=self.q_key_decimals, merge_tol_red=self.q_merge_tol_red)
        if phc_Qs is None:
            phc_Qs = build_unique_q_list(patchsets, mode="phc", decimals=self.q_key_decimals, merge_tol_red=self.q_merge_tol_red)

        self.pp_grid = TransferGrid(patchsets, list(pp_Qs), decimals=self.q_key_decimals, merge_tol_red=self.q_merge_tol_red)
        self.phd_grid = TransferGrid(patchsets, list(ph_Qs), decimals=self.q_key_decimals, merge_tol_red=self.q_merge_tol_red)
        self.phc_grid = TransferGrid(patchsets, list(phc_Qs), decimals=self.q_key_decimals, merge_tol_red=self.q_merge_tol_red)

        self.T_start = float(T_start)
        self.T_stop = float(T_stop)
        if self.T_start <= self.T_stop:
            raise ValueError("Require T_start > T_stop for a descending temperature flow.")
        self.n_steps = int(n_steps)
        self.temperature_grid = str(temperature_grid).lower()
        self.nfreq = int(nfreq)
        self.include_explicit_T_prefactor = bool(include_explicit_T_prefactor)
        self.max_relative_update = float(max_relative_update)
        self.min_substep_fraction = float(min_substep_fraction)
        self.channel_divergence_threshold = float(channel_divergence_threshold)
        self.diagnose_every = int(diagnose_every)

        self._spins = available_physical_spins(self.patchsets)
        self._internal_spin_pairs = tuple(available_internal_spin_pairs(self.patchsets))
        self._patch_k = {
            s: _canonicalize_patch_k_array(
                np.asarray([p.k_cart for p in patchset_for_spin(self.patchsets, s).patches], dtype=float),
                self.patchsets,
            )
            for s in self._spins
        }
        self._Npatch_by_spin = {s: patchset_for_spin(self.patchsets, s).Npatch for s in self._spins}
        self._validate_patch_counts()
        self.Npatch = self._Npatch_by_spin[self._spins[0]]

        self._precompute_transfer_tables()
        self._precompute_external_closure_maps()
        self._precompute_shift_maps()
        self._precompute_static_energies()

        self.state = FullVertexState(
            patchsets=patchsets,
            bare_gamma=bare_gamma,
            spin_blocks=list(self.spin_blocks),
            pp_grid=self.pp_grid,
            phd_grid=self.phd_grid,
            phc_grid=self.phc_grid,
            T=float(T_start),
            gamma_blocks=self._initialize_bare_vertex_blocks(),
        )
        self._fast_gamma = FullVertexAccessor(self)
        self.bare_vertex_norm = self._estimate_bare_vertex_norm()
        self.temperature_path = self._build_temperature_path()
        self.history: List[FlowStepRecord] = []
        self.instability_record: Optional[FlowStepRecord] = None

    def _validate_patch_counts(self) -> None:
        counts = [self._Npatch_by_spin[s] for s in self._spins]
        if len(set(counts)) != 1:
            raise ValueError("Full-vertex flow currently requires identical patch counts across available spin sectors.")

    def _flow_config(self, T: float) -> FlowConfig:
        return FlowConfig(
            temperature=float(T),
            nfreq=self.nfreq,
            include_explicit_T_prefactor=self.include_explicit_T_prefactor,
        )

    def _build_temperature_path(self) -> np.ndarray:
        if self.temperature_grid == "log":
            return np.geomspace(self.T_start, self.T_stop, self.n_steps)
        if self.temperature_grid == "linear":
            return np.linspace(self.T_start, self.T_stop, self.n_steps)
        raise ValueError("temperature_grid must be 'log' or 'linear'.")

    def _precompute_static_energies(self) -> None:
        self._energies_by_spin = {
            s: np.asarray([float(p.energy) for p in patchset_for_spin(self.patchsets, s).patches], dtype=float)
            for s in self._spins
        }

    def _precompute_transfer_tables(self) -> None:
        self._pp_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._phd_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        for s1 in self._spins:
            k1s = self._patch_k[s1]
            for s2 in self._spins:
                k2s = self._patch_k[s2]
                arr_pp = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_ph = np.zeros((self.Npatch, self.Npatch), dtype=int)
                for p1, k1 in enumerate(k1s):
                    for p2, k2 in enumerate(k2s):
                        arr_pp[p1, p2] = self.pp_grid.nearest_index(k1 + k2)
                        arr_ph[p1, p2] = self.phd_grid.nearest_index(k1 - k2)
                self._pp_q_index[(s1, s2)] = arr_pp
                self._phd_q_index[(s1, s2)] = arr_ph
                self._phc_q_index[(s1, s2)] = arr_ph.copy()

    def _partner_map_pp_from_iq(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            self.patchsets,
            self._pp_q_index[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=self.pp_grid.canonicalize(Q),
            mode="Q_minus_k",
        )

    def _partner_map_phd_from_iq(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            self.patchsets,
            self._phd_q_index[(normalize_spin(second_spin), normalize_spin(first_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=self.phd_grid.canonicalize(Q),
            mode="k_plus_Q",
        )

    def _partner_map_phc_from_iq(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float], mode: str):
        return partner_map_from_q_index(
            self.patchsets,
            self._phc_q_index[(normalize_spin(second_spin), normalize_spin(first_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=self.phc_grid.canonicalize(Q),
            mode=mode,
        )

    def _precompute_shift_maps(self) -> None:
        self._pp_qminus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._ph_kplus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._phc_kplus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._phc_kminus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        for iq, Q in enumerate(self.pp_grid.q_list):
            for s1 in self._spins:
                for s2 in self._spins:
                    self._pp_qminus[(iq, s1, s2)] = self._partner_map_pp_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q)
        for iq, Q in enumerate(self.phd_grid.q_list):
            for s1 in self._spins:
                for s2 in self._spins:
                    self._ph_kplus[(iq, s1, s2)] = self._partner_map_phd_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q)
        for iq, Q in enumerate(self.phc_grid.q_list):
            for s1 in self._spins:
                for s2 in self._spins:
                    self._phc_kplus[(iq, s1, s2)] = self._partner_map_phd_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q)
                    self._phc_kminus[(iq, s1, s2)] = self._partner_map_phc_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q, mode="k_minus_Q")

    def _precompute_external_closure_maps(self) -> None:
        self._closure_map: Dict[SpinBlock, Tuple[np.ndarray, np.ndarray]] = {}
        for key in self.spin_blocks:
            s1, s2, s3, s4 = key
            p4_idx = np.full((self.Npatch, self.Npatch, self.Npatch), -1, dtype=int)
            p4_res = np.full((self.Npatch, self.Npatch, self.Npatch), np.inf, dtype=float)
            target_cache: Dict[Tuple[int, int], np.ndarray] = {}
            for p1, k1 in enumerate(self._patch_k[s1]):
                for p2, k2 in enumerate(self._patch_k[s2]):
                    target_cache[(p1, p2)] = canonicalize_q_for_patchsets(self.patchsets, k1 + k2)
            for p1 in range(self.Npatch):
                for p2 in range(self.Npatch):
                    total = target_cache[(p1, p2)]
                    for p3, k3 in enumerate(self._patch_k[s3]):
                        target_k4 = canonicalize_q_for_patchsets(self.patchsets, total - k3)
                        idx, dist = self._find_shifted_patch_index(s4, target_k4)
                        p4_idx[p1, p2, p3] = idx
                        p4_res[p1, p2, p3] = dist
            self._closure_map[key] = (p4_idx, p4_res)

    def _find_shifted_patch_index(self, spin: str, target_k: Sequence[float]) -> Tuple[int, float]:
        ps = patchset_for_spin(self.patchsets, spin)
        ks = np.asarray([p.k_cart for p in ps.patches], dtype=float)
        b1 = np.asarray(ps.b1, dtype=float)
        b2 = np.asarray(ps.b2, dtype=float)
        best_idx = 0
        best_norm = np.inf
        target_k = np.asarray(target_k, dtype=float)
        for i, k_ref in enumerate(ks):
            local_best = np.inf
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    disp = target_k - (k_ref + n1 * b1 + n2 * b2)
                    nd = np.linalg.norm(disp)
                    if nd < local_best:
                        local_best = nd
            if local_best < best_norm:
                best_norm = local_best
                best_idx = i
        return int(best_idx), float(best_norm)

    def _initialize_bare_vertex_blocks(self) -> Dict[SpinBlock, GammaTensorBlock]:
        out: Dict[SpinBlock, GammaTensorBlock] = {}
        for key in self.spin_blocks:
            s1, s2, s3, s4 = key
            p4_idx, p4_res = self._closure_map[key]
            data = np.zeros((self.Npatch, self.Npatch, self.Npatch), dtype=complex)
            for p1 in range(self.Npatch):
                for p2 in range(self.Npatch):
                    for p3 in range(self.Npatch):
                        p4 = int(p4_idx[p1, p2, p3])
                        if p4 >= 0:
                            data[p1, p2, p3] = self.bare_gamma(p1, s1, p2, s2, p3, s3, p4, s4)
            out[key] = GammaTensorBlock(data=data, p4_index=p4_idx, p4_residual=p4_res)
        return out

    def _estimate_bare_vertex_norm(self) -> float:
        vals = [float(np.max(np.abs(block.data))) for block in self.state.gamma_blocks.values()] if hasattr(self, "state") else []
        if vals:
            return max(max(vals), 1e-14)
        vals = [float(np.max(np.abs(block.data))) for block in self._initialize_bare_vertex_blocks().values()]
        return max(max(vals), 1e-14)

    def current_gamma_accessor(self) -> GammaAccessor:
        return self._fast_gamma

    def _compress_internal_cache(self, cache: Dict[Tuple[str, str], Dict[str, np.ndarray]]):
        compressed = []
        for (sa, sb), info in cache.items():
            partner = np.asarray(info["partner"], dtype=int)
            weights = np.asarray(info["weights"], dtype=complex)
            valid = np.flatnonzero((partner >= 0) & (weights != 0))
            if valid.size == 0:
                continue
            compressed.append(
                (
                    sa,
                    sb,
                    partner[valid].astype(np.int64, copy=False),
                    weights[valid].astype(np.complex128, copy=False),
                    valid.astype(np.int64, copy=False),
                )
            )
        return tuple(compressed)

    def _build_q_level_internal_caches(self, T: float):
        cfg = self._flow_config(T)
        pp_internal_by_iq = {}
        ph_internal_by_iq = {}
        phc_internal_by_iq = {}
        for iq, _Q in enumerate(self.pp_grid.q_list):
            shift_cache = {(sa, sb): self._pp_qminus[(iq, sa, sb)] for sa, sb in self._internal_spin_pairs}
            pp_internal_by_iq[iq] = self._compress_internal_cache(
                build_pp_internal_cache_vec(self.patchsets, cfg, shift_cache=shift_cache)
            )
        for iq, _Q in enumerate(self.phd_grid.q_list):
            shift_cache = {(sa, sb): self._ph_kplus[(iq, sa, sb)] for sa, sb in self._internal_spin_pairs}
            ph_internal_by_iq[iq] = self._compress_internal_cache(
                build_ph_internal_cache_vec(self.patchsets, cfg, shift_cache=shift_cache)
            )
        for iq, _Q in enumerate(self.phc_grid.q_list):
            shift_cache = {(sa, sb): self._phc_kplus[(iq, sa, sb)] for sa, sb in self._internal_spin_pairs}
            phc_internal_by_iq[iq] = self._compress_internal_cache(
                build_ph_internal_cache_vec(self.patchsets, cfg, shift_cache=shift_cache)
            )
        return pp_internal_by_iq, ph_internal_by_iq, phc_internal_by_iq

    def compute_vertex_rhs(self, T: float) -> Dict[SpinBlock, np.ndarray]:
        pp_internal_by_iq, ph_internal_by_iq, phc_internal_by_iq = self._build_q_level_internal_caches(T)

        rhs: Dict[SpinBlock, np.ndarray] = {
            key: np.zeros((self.Npatch, self.Npatch, self.Npatch), dtype=complex) for key in self.spin_blocks
        }

        block_data = {key: blk.data for key, blk in self.state.gamma_blocks.items()}
        block_p4 = {key: blk.p4_index for key, blk in self.state.gamma_blocks.items()}
        zero = 0.0 + 0.0j

        def gval(key: SpinBlock, p1: int, p2: int, p3: int, p4: int) -> complex:
            data = block_data.get(key)
            if data is None:
                return zero
            p4_idx = block_p4[key]
            if int(p4_idx[p1, p2, p3]) != int(p4):
                return zero
            return data[p1, p2, p3]

        for key in self.spin_blocks:
            s1, s2, s3, s4 = key
            p4_idx = block_p4[key]
            qpp = self._pp_q_index[(s1, s2)]
            qphd = self._phd_q_index[(s3, s1)]
            qphc = self._phc_q_index[(s3, s2)]
            rhs_block = rhs[key]

            for p1 in range(self.Npatch):
                for p2 in range(self.Npatch):
                    pp_entries = pp_internal_by_iq[int(qpp[p1, p2])]
                    for p3 in range(self.Npatch):
                        p4 = int(p4_idx[p1, p2, p3])
                        if p4 < 0:
                            continue

                        val_pp = zero
                        for sa, sb, partner_valid, weights_valid, valid_a in pp_entries:
                            left_key = (s1, s2, sa, sb)
                            right_key = (sa, sb, s3, s4)
                            if left_key not in block_data or right_key not in block_data:
                                continue
                            for idx, a in enumerate(valid_a):
                                b = int(partner_valid[idx])
                                w = weights_valid[idx]
                                val_pp += w * gval(left_key, p1, p2, int(a), b) * gval(right_key, int(a), b, p3, p4)

                        val_phd = zero
                        phd_entries = ph_internal_by_iq[int(qphd[p3, p1])]
                        for sa, sb, partner_valid, weights_valid, valid_a in phd_entries:
                            key_1a = (s1, sa, s3, sb)
                            key_1b = (sb, s2, sa, s4)
                            key_2a = (s1, sb, s3, sa)
                            key_2b = (sa, s2, sb, s4)
                            keep1 = key_1a in block_data and key_1b in block_data
                            keep2 = key_2a in block_data and key_2b in block_data
                            if not (keep1 or keep2):
                                continue
                            for idx, a in enumerate(valid_a):
                                b = int(partner_valid[idx])
                                w = weights_valid[idx]
                                term1 = zero
                                term2 = zero
                                if keep1:
                                    term1 = gval(key_1a, p1, int(a), p3, b) * gval(key_1b, b, p2, int(a), p4)
                                if keep2:
                                    term2 = gval(key_2a, p1, b, p3, int(a)) * gval(key_2b, int(a), p2, b, p4)
                                val_phd -= w * (term1 + term2)

                        val_phc = zero
                        phc_entries = phc_internal_by_iq[int(qphc[p3, p2])]
                        for sa, sb, partner_valid, weights_valid, valid_a in phc_entries:
                            key_1a = (s1, sa, s4, sb)
                            key_1b = (sb, s2, sa, s3)
                            key_2a = (s1, sb, s4, sa)
                            key_2b = (sa, s2, sb, s3)
                            keep1 = key_1a in block_data and key_1b in block_data
                            keep2 = key_2a in block_data and key_2b in block_data
                            if not (keep1 or keep2):
                                continue
                            for idx, a in enumerate(valid_a):
                                b = int(partner_valid[idx])
                                w = weights_valid[idx]
                                term1 = zero
                                term2 = zero
                                if keep1:
                                    term1 = gval(key_1a, p1, int(a), p4, b) * gval(key_1b, b, p2, int(a), p3)
                                if keep2:
                                    term2 = gval(key_2a, p1, b, p4, int(a)) * gval(key_2b, int(a), p2, b, p3)
                                val_phc += w * (term1 + term2)

                        rhs_block[p1, p2, p3] = val_pp + val_phd + val_phc
        return rhs

    def _rhs_norm(self, rhs: Dict[SpinBlock, np.ndarray]) -> float:
        vals = [float(np.max(np.abs(v))) for v in rhs.values() if v.size]
        return max(vals) if vals else 0.0

    def _apply_rhs(self, rhs: Dict[SpinBlock, np.ndarray], scale: float) -> None:
        for key, mat in rhs.items():
            self.state.gamma_blocks[key].data += scale * np.asarray(mat, dtype=complex)

    def diagnose_current_state(self) -> Dict[str, Any]:
        return {
            "representation": "full_vertex",
            "stored_object": "Gamma(s1,s2,s3,s4; p1,p2,p3)",
            "channel_norm": self.state.channel_norm(),
        }

    def check_instability(self, record: FlowStepRecord) -> Tuple[bool, Optional[str]]:
        if record.terminated_early:
            return True, record.termination_reason or "flow terminated early"
        if record.channel_norm >= self.channel_divergence_threshold:
            return True, f"channel norm={record.channel_norm:.3e} exceeded channel_divergence_threshold"
        return False, None

    def step(self, T_old: float, dT: float) -> FlowStepRecord:
        rhs = self.compute_vertex_rhs(T_old)
        effective_norm = max(self.state.channel_norm(), self.bare_vertex_norm, 1e-14)
        rhs_norm = self._rhs_norm(rhs)
        rel_update = abs(dT) * rhs_norm / effective_norm
        n_sub = max(1, int(np.ceil(rel_update / self.max_relative_update))) if rel_update > self.max_relative_update else 1
        if (1.0 / n_sub) < self.min_substep_fraction:
            attempted_T_new = float(T_old + dT)
            message = (
                "Adaptive step control requested too many substeps; stopping flow early. "
                f"Current state remains at T={float(T_old):.8f}, attempted T_new={attempted_T_new:.8f}, "
                f"rhs_norm={rhs_norm:.3e}, rel_update={rel_update:.3e}, proposed_n_sub={n_sub}."
            )
            return FlowStepRecord(
                step_index=len(self.history),
                temperature=float(T_old),
                dT=float(dT),
                channel_norm=self.state.channel_norm(),
                rhs_norm=rhs_norm,
                accepted_substeps=0,
                max_rel_update=rel_update,
                terminated_early=True,
                termination_reason=message,
                diagnosis_payload=self.diagnose_current_state(),
            )
        sub_dT = dT / n_sub
        for _ in range(n_sub):
            self._apply_rhs(rhs, sub_dT)
        self.state.T = float(T_old + dT)
        return FlowStepRecord(
            step_index=len(self.history),
            temperature=float(T_old + dT),
            dT=float(dT),
            channel_norm=self.state.channel_norm(),
            rhs_norm=rhs_norm,
            accepted_substeps=n_sub,
            max_rel_update=rel_update / n_sub if n_sub > 0 else rel_update,
            diagnosis_payload=self.diagnose_current_state(),
        )

    def run(self) -> List[FlowStepRecord]:
        temps = self.temperature_path
        rec0 = FlowStepRecord(
            step_index=0,
            temperature=float(temps[0]),
            dT=0.0,
            channel_norm=self.state.channel_norm(),
            rhs_norm=0.0,
            accepted_substeps=0,
            max_rel_update=0.0,
            diagnosis_payload=self.diagnose_current_state(),
        )
        rec0.instability, rec0.instability_reason = self.check_instability(rec0)
        self.history = [rec0]
        if rec0.instability:
            self.instability_record = rec0
            return self.history
        for i in range(len(temps) - 1):
            T_old = float(temps[i])
            T_new = float(temps[i + 1])
            rec = self.step(T_old, T_new - T_old)
            rec.instability, rec.instability_reason = self.check_instability(rec)
            self.history.append(rec)
            if rec.instability:
                self.instability_record = rec
                break
        return self.history

    def history_as_dicts(self) -> List[Dict[str, Any]]:
        return [x.summary_dict() for x in self.history]

    def closure_map(self) -> Dict[SpinBlock, Tuple[np.ndarray, np.ndarray]]:
        return {k: (v[0].copy(), v[1].copy()) for k, v in self._closure_map.items()}


__all__ = [
    "BareVertexFromInteraction",
    "FRGFlowSolver",
    "FullVertexState",
    "FlowStepRecord",
    "TransferGrid",
    "available_physical_spins",
    "build_unique_q_list",
    "canonical_spin_tuple",
    "default_spin_blocks",
]
