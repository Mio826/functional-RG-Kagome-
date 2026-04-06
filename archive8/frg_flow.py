from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from frg_kernel import (
    FlowConfig,
    MinimalInternalCache,
    PatchSetMap,
    TransferGrid,
    build_ph_internal_cache_vec,
    build_pp_internal_cache_vec,
    build_unique_q_list,
    canonicalize_q_for_patchsets,
    compute_phc_vertex_contribution_sz0,
    compute_phd_vertex_contribution_sz0,
    compute_pp_vertex_contribution_sz0,
    has_patchset,
    normalize_spin,
    partner_map_from_q_index,
    patchset_for_spin,
)


@dataclass
class SZ0Tensor:
    """Store V(p1,p2,p3) with p4 determined by closure map."""
    data: np.ndarray
    p4_index: np.ndarray
    p4_residual: np.ndarray


@dataclass
class SZ0FlowState:
    patchsets: PatchSetMap
    bare_vertex: Callable[[int, int, int, int], complex]
    pp_grid: TransferGrid
    phd_grid: TransferGrid
    phc_grid: TransferGrid
    T: float
    vertex: SZ0Tensor

    def channel_norm(self) -> float:
        return float(np.max(np.abs(self.vertex.data))) if self.vertex.data.size else 0.0


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


class BareSZ0VertexFromInteraction:
    """
    Adapter for a PRL-compatible minimal S_z=0 bare vertex.

    Convention:
        V(1,2;3,4) = Gamma_{up,dn -> dn,up}(1,2;3,4)

    Important
    ---------
    The minimal Sz=0 object is *not* the raw direct density-density matrix
    element.  Bare level consistency with the PRL-compatible convention requires
    the antisymmetrized opposite-spin exchanged amplitude.
    """

    def __init__(self, interaction: Any, patchsets: PatchSetMap):
        self.interaction = interaction
        self.patchsets = patchsets

    def __call__(self, p1: int, p2: int, p3: int, p4: int) -> complex:
        if hasattr(self.interaction, "patch_vertex_sz0"):
            return complex(self.interaction.patch_vertex_sz0(self.patchsets, p1, p2, p3, p4))

        # Fallback for older interaction classes: use the *antisymmetrized*
        # opposite-spin exchanged object, not the raw direct vertex.
        return complex(
            self.interaction.patch_vertex(
                self.patchsets,
                p1, "up",
                p2, "dn",
                p3, "dn",
                p4, "up",
                antisym=True,
                check_momentum=False,
            )
        )


class FRGFlowSolverSZ0:
    def __init__(
        self,
        *,
        patchsets: PatchSetMap,
        bare_vertex: Callable[[int, int, int, int], complex],
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
        diagnosis_Qs: Optional[Sequence[Sequence[float]]] = None,
        diagnosis_score_threshold: Optional[float] = None,
        diagnosis_landau_F: bool = False,
    ) -> None:
        self.patchsets = patchsets
        self.bare_vertex = bare_vertex
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
        self.diagnosis_Qs = None if diagnosis_Qs is None else [
            np.asarray(canonicalize_q_for_patchsets(patchsets, q), dtype=float) for q in diagnosis_Qs
        ]
        self.diagnosis_score_threshold = None if diagnosis_score_threshold is None else float(diagnosis_score_threshold)
        self.diagnosis_landau_F = bool(diagnosis_landau_F)

        self._spins = ("up", "dn")
        self._validate_patch_counts()
        self.Npatch = patchset_for_spin(self.patchsets, "up").Npatch
        self._patch_k = {
            s: np.asarray(
                [canonicalize_q_for_patchsets(self.patchsets, p.k_cart) for p in patchset_for_spin(self.patchsets, s).patches],
                dtype=float,
            )
            for s in self._spins
        }

        self._precompute_transfer_tables()
        self._precompute_closure_map_sz0()
        self._precompute_shift_maps()

        self.state = SZ0FlowState(
            patchsets=patchsets,
            bare_vertex=bare_vertex,
            pp_grid=self.pp_grid,
            phd_grid=self.phd_grid,
            phc_grid=self.phc_grid,
            T=float(T_start),
            vertex=self._initialize_bare_vertex(),
        )
        self._fast_vertex = SZ0VertexAccessor(self)
        self.bare_vertex_norm = self._estimate_bare_vertex_norm()
        self.temperature_path = self._build_temperature_path()
        self.history: List[FlowStepRecord] = []
        self.instability_record: Optional[FlowStepRecord] = None

    def _validate_patch_counts(self) -> None:
        if not (has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn")):
            raise ValueError("SZ0 flow requires both up and dn patchsets.")
        counts = [patchset_for_spin(self.patchsets, s).Npatch for s in self._spins]
        if len(set(counts)) != 1:
            raise ValueError("SZ0 flow currently requires identical patch counts across up/dn patchsets.")

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

    def _precompute_transfer_tables(self) -> None:
        self._pp_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._phd_q_index_plus: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index_plus: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index_minus: Dict[Tuple[str, str], np.ndarray] = {}

        for s_src in self._spins:
            ksrcs = self._patch_k[s_src]
            for s_tgt in self._spins:
                ktgts = self._patch_k[s_tgt]
                arr_pp = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_phd_plus = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_phc_plus = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_phc_minus = np.zeros((self.Npatch, self.Npatch), dtype=int)
                for p_src, k_src in enumerate(ksrcs):
                    for p_tgt, k_tgt in enumerate(ktgts):
                        arr_pp[p_src, p_tgt] = self.pp_grid.nearest_index(k_src + k_tgt)
                        arr_phd_plus[p_src, p_tgt] = self.phd_grid.nearest_index(k_tgt - k_src)
                        arr_phc_plus[p_src, p_tgt] = self.phc_grid.nearest_index(k_tgt - k_src)
                        arr_phc_minus[p_src, p_tgt] = self.phc_grid.nearest_index(k_src - k_tgt)
                self._pp_q_index[(s_src, s_tgt)] = arr_pp
                self._phd_q_index_plus[(s_src, s_tgt)] = arr_phd_plus
                # For phc we keep both signs explicitly.  The flow itself uses
                # Q = k3 - k2, i.e. the "plus" table when indexed as [p2, p3].
                self._phc_q_index_plus[(s_src, s_tgt)] = arr_phc_plus
                self._phc_q_index_minus[(s_src, s_tgt)] = arr_phc_minus

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
            self._phd_q_index_plus[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=self.phd_grid.canonicalize(Q),
            mode="k_plus_Q",
        )

    def _partner_map_phc_from_iq(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float], mode: str):
        table = self._phc_q_index_plus if mode == "k_plus_Q" else self._phc_q_index_minus
        return partner_map_from_q_index(
            self.patchsets,
            table[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=self.phc_grid.canonicalize(Q),
            mode=mode,
        )

    def _precompute_shift_maps(self) -> None:
        self._pp_qminus: Dict[int, MinimalInternalCache] = {}
        self._ph_kplus: Dict[int, MinimalInternalCache] = {}
        self._phc_kplus: Dict[int, MinimalInternalCache] = {}

        for iq, Q in enumerate(self.pp_grid.q_list):
            legacy = build_pp_internal_cache_vec(
                self.patchsets,
                self._flow_config(self.T_start),
                shift_cache={("up", "dn"): self._partner_map_pp_from_iq(iq, first_spin="up", second_spin="dn", Q=Q)},
            )[("up", "dn")]
            self._pp_qminus[iq] = {
                "partner": np.asarray(legacy["partner"], dtype=int),
                "residual": np.asarray(legacy["residual"], dtype=float),
                "weights": np.asarray(legacy["weights"], dtype=complex),
            }

        # For ph/phc the partner geometry is temperature-independent but the
        # weights are temperature-dependent.  We store the partner/residual part
        # here and refresh the weights each step.
        for iq, Q in enumerate(self.phd_grid.q_list):
            partner, residual = self._partner_map_phd_from_iq(iq, first_spin="up", second_spin="dn", Q=Q)
            self._ph_kplus[iq] = {
                "partner": np.asarray(partner, dtype=int),
                "residual": np.asarray(residual, dtype=float),
                "weights": np.zeros(self.Npatch, dtype=complex),
            }
        for iq, Q in enumerate(self.phc_grid.q_list):
            partner, residual = self._partner_map_phc_from_iq(iq, first_spin="up", second_spin="dn", Q=Q, mode="k_plus_Q")
            self._phc_kplus[iq] = {
                "partner": np.asarray(partner, dtype=int),
                "residual": np.asarray(residual, dtype=float),
                "weights": np.zeros(self.Npatch, dtype=complex),
            }

    def _precompute_closure_map_sz0(self) -> None:
        s1, s2, s3, s4 = "up", "dn", "dn", "up"
        p4_idx = np.full((self.Npatch, self.Npatch, self.Npatch), -1, dtype=int)
        p4_res = np.full((self.Npatch, self.Npatch, self.Npatch), np.inf, dtype=float)
        ks4 = self._patch_k[s4]
        ps4 = patchset_for_spin(self.patchsets, s4)
        b1 = np.asarray(ps4.b1, dtype=float)
        b2 = np.asarray(ps4.b2, dtype=float)

        for p1, k1 in enumerate(self._patch_k[s1]):
            for p2, k2 in enumerate(self._patch_k[s2]):
                total = canonicalize_q_for_patchsets(self.patchsets, k1 + k2)
                for p3, k3 in enumerate(self._patch_k[s3]):
                    target_k4 = canonicalize_q_for_patchsets(self.patchsets, total - k3)
                    best_idx = 0
                    best_norm = np.inf
                    for i, k_ref in enumerate(ks4):
                        local_best = np.inf
                        for n1 in (-1, 0, 1):
                            for n2 in (-1, 0, 1):
                                disp = target_k4 - (k_ref + n1 * b1 + n2 * b2)
                                nd = np.linalg.norm(disp)
                                if nd < local_best:
                                    local_best = nd
                        if local_best < best_norm:
                            best_norm = local_best
                            best_idx = i
                    p4_idx[p1, p2, p3] = int(best_idx)
                    p4_res[p1, p2, p3] = float(best_norm)

        self._closure_map = {(s1, s2, s3, s4): (p4_idx, p4_res)}

    def _initialize_bare_vertex(self) -> SZ0Tensor:
        key = ("up", "dn", "dn", "up")
        p4_idx, p4_res = self._closure_map[key]
        data = np.zeros((self.Npatch, self.Npatch, self.Npatch), dtype=complex)
        for p1 in range(self.Npatch):
            for p2 in range(self.Npatch):
                for p3 in range(self.Npatch):
                    p4 = int(p4_idx[p1, p2, p3])
                    if p4 >= 0:
                        data[p1, p2, p3] = self.bare_vertex(p1, p2, p3, p4)
        return SZ0Tensor(data=data, p4_index=p4_idx, p4_residual=p4_res)

    def _estimate_bare_vertex_norm(self) -> float:
        return max(float(np.max(np.abs(self.state.vertex.data))), 1e-14)

    def current_vertex_accessor(self) -> Callable[[int, int, int, int], complex]:
        return self._fast_vertex

    def _refresh_cache_weights(self, T: float) -> Tuple[Dict[int, MinimalInternalCache], Dict[int, MinimalInternalCache], Dict[int, MinimalInternalCache]]:
        cfg = self._flow_config(T)

        pp_internal_by_iq: Dict[int, MinimalInternalCache] = {}
        ph_internal_by_iq: Dict[int, MinimalInternalCache] = {}
        phc_internal_by_iq: Dict[int, MinimalInternalCache] = {}

        # pp: partner geometry and weights are both simplest to refresh through the
        # legacy helper, then immediately strip the explicit-spin wrapper.
        for iq in range(len(self.pp_grid.q_list)):
            Q = self.pp_grid.q_list[iq]
            legacy = build_pp_internal_cache_vec(
                self.patchsets,
                cfg,
                shift_cache={("up", "dn"): self._partner_map_pp_from_iq(iq, first_spin="up", second_spin="dn", Q=Q)},
            )[("up", "dn")]
            pp_internal_by_iq[iq] = {
                "partner": np.asarray(legacy["partner"], dtype=int),
                "residual": np.asarray(legacy["residual"], dtype=float),
                "weights": np.asarray(legacy["weights"], dtype=complex),
            }

        # ph / phc: refresh only the temperature-dependent weights.
        for iq, template in self._ph_kplus.items():
            legacy = build_ph_internal_cache_vec(
                self.patchsets,
                cfg,
                shift_cache={("up", "dn"): (template["partner"], template["residual"])}
            )[("up", "dn")]
            ph_internal_by_iq[iq] = {
                "partner": template["partner"],
                "residual": template["residual"],
                "weights": np.asarray(legacy["weights"], dtype=complex),
            }

        for iq, template in self._phc_kplus.items():
            legacy = build_ph_internal_cache_vec(
                self.patchsets,
                cfg,
                shift_cache={("up", "dn"): (template["partner"], template["residual"])}
            )[("up", "dn")]
            phc_internal_by_iq[iq] = {
                "partner": template["partner"],
                "residual": template["residual"],
                "weights": np.asarray(legacy["weights"], dtype=complex),
            }

        return pp_internal_by_iq, ph_internal_by_iq, phc_internal_by_iq

    def compute_vertex_rhs(self, T: float) -> np.ndarray:
        pp_internal_by_iq, ph_internal_by_iq, phc_internal_by_iq = self._refresh_cache_weights(T)
        rhs = np.zeros_like(self.state.vertex.data)
        p4_idx = self.state.vertex.p4_index
        qpp = self._pp_q_index[("up", "dn")]
        qphd = self._phd_q_index_plus[("up", "dn")]
        # Crossed-ph transfer indices must use the same spin ordering as the
        # phc internal cache built in _precompute_shift_maps/_refresh_cache_weights.
        # That cache is constructed for ("up","dn"), so using ("dn","dn") here is
        # inconsistent and can select the wrong phc transfer sector.
        qphc = self._phc_q_index_plus[("up", "dn")]

        v_accessor = self.current_vertex_accessor()

        for p1 in range(self.Npatch):
            for p2 in range(self.Npatch):
                pp_cache = pp_internal_by_iq[int(qpp[p1, p2])]
                for p3 in range(self.Npatch):
                    p4 = int(p4_idx[p1, p2, p3])
                    if p4 < 0:
                        continue

                    # direct ph transfer: Q = k3 - k1 -> index[p1, p3]
                    phd_cache = ph_internal_by_iq[int(qphd[p1, p3])]

                    # crossed ph transfer: Q = k3 - k2 -> index[p2, p3]
                    # Keep the q-index lookup consistent with the phc cache spin
                    # convention chosen above.
                    phc_cache = phc_internal_by_iq[int(qphc[p2, p3])]

                    rhs[p1, p2, p3] = (
                        compute_pp_vertex_contribution_sz0(v_accessor, p1=p1, p2=p2, p3=p3, p4=p4, internal_cache=pp_cache)
                        + compute_phd_vertex_contribution_sz0(v_accessor, p1=p1, p2=p2, p3=p3, p4=p4, internal_cache=phd_cache)
                        + compute_phc_vertex_contribution_sz0(v_accessor, p1=p1, p2=p2, p3=p3, p4=p4, internal_cache=phc_cache)
                    )
        return rhs

    def _rhs_norm(self, rhs: np.ndarray) -> float:
        return float(np.max(np.abs(rhs))) if rhs.size else 0.0

    def _apply_rhs(self, rhs: np.ndarray, scale: float) -> None:
        self.state.vertex.data += scale * np.asarray(rhs, dtype=complex)


    @staticmethod
    def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
        M = np.asarray(matrix, dtype=complex)
        return 0.5 * (M + M.conjugate().T)

    @staticmethod
    def _sign_aware_channel_score(channel_name: str, matrix: np.ndarray) -> Dict[str, Any]:
        H = FRGFlowSolverSZ0._hermitian_part(matrix)
        evals, _ = np.linalg.eigh(H)
        eval_pos_max = float(np.max(evals)) if evals.size else 0.0
        eval_neg_min = float(np.min(evals)) if evals.size else 0.0

        if channel_name.startswith("pp"):
            physical_score = max(eval_pos_max, 0.0)
            chosen_eval = eval_pos_max
            chosen_sign = "positive"
        elif channel_name.startswith("ph"):
            physical_score = max(-eval_neg_min, 0.0)
            chosen_eval = eval_neg_min
            chosen_sign = "negative"
        else:
            raise ValueError(f"Unknown channel name: {channel_name}")

        return {
            "eval_pos_max": eval_pos_max,
            "eval_neg_min": eval_neg_min,
            "physical_score": float(physical_score),
            "chosen_eval": float(chosen_eval),
            "chosen_sign": chosen_sign,
            "herm_resid": float(np.max(np.abs(np.asarray(matrix) - np.asarray(matrix).conjugate().T))),
        }

    def _build_channel_builder(self):
        from channels import SZ0ChannelBuilder
        return SZ0ChannelBuilder.from_solver(
            self.current_vertex_accessor(),
            self,
            Landau_F=self.diagnosis_landau_F,
        )

    def _diagnose_sign_aware_channels(self) -> Dict[str, Any]:
        if not self.diagnosis_Qs:
            return {}

        builder = self._build_channel_builder()
        rows: List[Dict[str, Any]] = []
        q_payload: List[Dict[str, Any]] = []

        for iq, Q in enumerate(self.diagnosis_Qs):
            kernel_dict = builder.build_kernel_dict(Q, Landau_F=self.diagnosis_landau_F)
            q_entry: Dict[str, Any] = {
                "Q_index": int(iq),
                "Q": np.asarray(Q, dtype=float),
                "channels": {},
            }
            for ch_name, ker in kernel_dict.items():
                info = self._sign_aware_channel_score(ch_name, ker.matrix)
                row = {
                    "channel": ch_name,
                    "Q_index": int(iq),
                    "Q": np.asarray(Q, dtype=float),
                    **info,
                }
                rows.append(row)
                q_entry["channels"][ch_name] = row
            q_payload.append(q_entry)

        if not rows:
            return {}

        best = max(rows, key=lambda x: x["physical_score"])
        return {
            "diagnosis_mode": "sign_aware_hermitian_channel_score",
            "Landau_F": bool(self.diagnosis_landau_F),
            "Qs": [np.asarray(q, dtype=float) for q in self.diagnosis_Qs],
            "rows": rows,
            "per_Q": q_payload,
            "leading_channel": str(best["channel"]),
            "leading_Q_index": int(best["Q_index"]),
            "leading_Q": np.asarray(best["Q"], dtype=float),
            "leading_score": float(best["physical_score"]),
            "leading_chosen_eval": float(best["chosen_eval"]),
            "leading_chosen_sign": str(best["chosen_sign"]),
        }

    def diagnose_current_state(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "representation": "sz0_minimal",
            "stored_object": "V(p1,p2,p3; p4 via closure)",
            "channel_norm": self.state.channel_norm(),
        }
        if self.diagnosis_Qs:
            try:
                payload["sign_aware"] = self._diagnose_sign_aware_channels()
            except Exception as exc:
                payload["sign_aware_error"] = repr(exc)
        return payload

    def check_instability(self, record: FlowStepRecord) -> Tuple[bool, Optional[str]]:
        if record.terminated_early:
            return True, record.termination_reason or "flow terminated early"

        if self.diagnosis_score_threshold is not None:
            sign_aware = record.diagnosis_payload.get("sign_aware", {}) if isinstance(record.diagnosis_payload, dict) else {}
            leading_score = sign_aware.get("leading_score") if isinstance(sign_aware, dict) else None
            if leading_score is not None and float(leading_score) >= float(self.diagnosis_score_threshold):
                leading_channel = sign_aware.get("leading_channel", "unknown")
                leading_q = sign_aware.get("leading_Q", None)
                q_str = np.array2string(np.asarray(leading_q, dtype=float), precision=6) if leading_q is not None else "unknown"
                return True, (
                    f"sign-aware diagnosis score={float(leading_score):.3e} exceeded diagnosis_score_threshold "
                    f"for channel={leading_channel} at Q={q_str}"
                )

        if record.channel_norm >= self.channel_divergence_threshold:
            return True, f"channel norm={record.channel_norm:.3e} exceeded channel_divergence_threshold"
        return False, None

    def step(self, T_old: float, dT: float) -> FlowStepRecord:
        # First estimate the required substepping from the RHS at the beginning
        # of the macro step.
        rhs0 = self.compute_vertex_rhs(T_old)
        effective_norm = max(self.state.channel_norm(), self.bare_vertex_norm, 1e-14)
        rhs_norm0 = self._rhs_norm(rhs0)
        rel_update0 = abs(dT) * rhs_norm0 / effective_norm
        n_sub = max(1, int(np.ceil(rel_update0 / self.max_relative_update))) if rel_update0 > self.max_relative_update else 1
        if (1.0 / n_sub) < self.min_substep_fraction:
            attempted_T_new = float(T_old + dT)
            message = (
                "Adaptive step control requested too many substeps; stopping flow early. "
                f"Current state remains at T={float(T_old):.8f}, attempted T_new={attempted_T_new:.8f}, "
                f"rhs_norm={rhs_norm0:.3e}, rel_update={rel_update0:.3e}, proposed_n_sub={n_sub}."
            )
            return FlowStepRecord(
                step_index=len(self.history),
                temperature=float(T_old),
                dT=float(dT),
                channel_norm=self.state.channel_norm(),
                rhs_norm=rhs_norm0,
                accepted_substeps=0,
                max_rel_update=rel_update0,
                terminated_early=True,
                termination_reason=message,
                diagnosis_payload=self.diagnose_current_state(),
            )

        sub_dT = dT / n_sub
        T_sub = float(T_old)
        rhs_norm_max = rhs_norm0
        rel_update_max = rel_update0 / n_sub if n_sub > 0 else rel_update0

        # Refreshed-Euler substepping: recompute RHS after every accepted substep
        # so that late-stage strong-coupling flow is not integrated with a frozen RHS.
        for _ in range(n_sub):
            rhs_sub = self.compute_vertex_rhs(T_sub)
            rhs_norm_sub = self._rhs_norm(rhs_sub)
            effective_norm_sub = max(self.state.channel_norm(), self.bare_vertex_norm, 1e-14)
            rel_update_sub = abs(sub_dT) * rhs_norm_sub / effective_norm_sub
            rhs_norm_max = max(rhs_norm_max, rhs_norm_sub)
            rel_update_max = max(rel_update_max, rel_update_sub)
            self._apply_rhs(rhs_sub, sub_dT)
            T_sub = float(T_sub + sub_dT)
            self.state.T = T_sub

        return FlowStepRecord(
            step_index=len(self.history),
            temperature=float(T_old + dT),
            dT=float(dT),
            channel_norm=self.state.channel_norm(),
            rhs_norm=rhs_norm_max,
            accepted_substeps=n_sub,
            max_rel_update=rel_update_max,
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

    def closure_map(self) -> Dict[Tuple[str, str, str, str], Tuple[np.ndarray, np.ndarray]]:
        return {k: (v[0].copy(), v[1].copy()) for k, v in self._closure_map.items()}

    def transfer_context(self) -> Dict[str, Any]:
        return {
            "pp_grid": self.pp_grid,
            "phd_grid": self.phd_grid,
            "phc_grid": self.phc_grid,
            "pp_q_index": {k: v.copy() for k, v in self._pp_q_index.items()},
            "phd_q_index_plus": {k: v.copy() for k, v in self._phd_q_index_plus.items()},
            "phc_q_index_plus": {k: v.copy() for k, v in self._phc_q_index_plus.items()},
            "phc_q_index_minus": {k: v.copy() for k, v in self._phc_q_index_minus.items()},
        }


class SZ0VertexAccessor:
    def __init__(self, solver: FRGFlowSolverSZ0):
        self.solver = solver

    def __call__(self, p1: int, p2: int, p3: int, p4: int) -> complex:
        p4_expected = int(self.solver.state.vertex.p4_index[p1, p2, p3])
        if p4_expected < 0 or int(p4) != p4_expected:
            return 0.0 + 0.0j
        return complex(self.solver.state.vertex.data[p1, p2, p3])


__all__ = [
    "BareSZ0VertexFromInteraction",
    "FRGFlowSolverSZ0",
    "FlowStepRecord",
    "SZ0FlowState",
    "SZ0Tensor",
]
