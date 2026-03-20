
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

        def eig(self, sort_by: str = "abs"):
            vals, vecs = np.linalg.eig(self.matrix)
            if sort_by == "abs":
                order = np.argsort(-np.abs(vals))
            elif sort_by == "real":
                order = np.argsort(-np.real(vals))
            else:
                raise ValueError("sort_by must be 'abs' or 'real'.")
            return vals[order], vecs[:, order]

from frg_kernel import (
    FlowConfig,
    compute_pp_kernel_fast,
    compute_ph_kernel_fast,
    compute_phc_kernel_fast,
    normalize_spin,
    patchset_for_spin,
    shifted_patch_map,
    has_patchset,
)


from frg_kernel import (
    available_internal_spin_pairs,
    build_ph_internal_cache_vec,
    build_pp_internal_cache_vec,
    compute_ph_kernel_fast2,
    compute_phc_kernel_fast2,
    compute_pp_kernel_fast2,
)

try:
    from kagome_order_diagnosis import KagomeOrderDiagnoser
except Exception:
    KagomeOrderDiagnoser = None


SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
SpinBlock = Tuple[str, str, str, str]
ChannelKey = Tuple[str, str, str, str, int]
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
                p1, s1,
                p2, s2,
                p3, s3,
                p4, s4,
                antisym=True,
                check_momentum=False,
            )
        )


def _reduced_coords(k: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    B = np.column_stack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)])
    return np.linalg.solve(B, np.asarray(k, dtype=float))


class TransferGrid:
    def __init__(self, patchsets: PatchSetMap, q_list: Sequence[Sequence[float]], *, decimals: int = 10):
        ref_spin = available_physical_spins(patchsets)[0]
        ps = patchset_for_spin(patchsets, ref_spin)
        self.b1 = np.asarray(ps.b1, dtype=float)
        self.b2 = np.asarray(ps.b2, dtype=float)
        self.decimals = int(decimals)
        self.q_list = [np.asarray(q, dtype=float) for q in q_list]
        self.key_to_index: Dict[Tuple[float, float], int] = {}
        for i, q in enumerate(self.q_list):
            self.key_to_index[self.key(q)] = i

    def key(self, q: Sequence[float]) -> Tuple[float, float]:
        uv = _reduced_coords(np.asarray(q, dtype=float), self.b1, self.b2)
        uv = uv - np.floor(uv)
        return tuple(np.round(uv, decimals=self.decimals))

    def nearest_index(self, q: Sequence[float]) -> int:
        key = self.key(q)
        if key in self.key_to_index:
            return int(self.key_to_index[key])
        target = np.asarray(q, dtype=float)
        dists = [np.linalg.norm(target - qq) for qq in self.q_list]
        return int(np.argmin(dists))


def build_unique_q_list(patchsets: PatchSetMap, *, mode: str, decimals: int = 10) -> List[np.ndarray]:
    ref_spin = available_physical_spins(patchsets)[0]
    ps = patchset_for_spin(patchsets, ref_spin)
    ks = np.asarray([p.k_cart for p in ps.patches], dtype=float)
    grid = TransferGrid(patchsets, [np.zeros(2, dtype=float)], decimals=decimals)
    seen: Dict[Tuple[float, float], np.ndarray] = {grid.key(np.zeros(2, dtype=float)): np.zeros(2, dtype=float)}

    if mode == "pp":
        for k1 in ks:
            for k2 in ks:
                q = np.asarray(k1 + k2, dtype=float)
                seen.setdefault(grid.key(q), q)
    elif mode in {"ph", "phc"}:
        for k1 in ks:
            for k3 in ks:
                q = np.asarray(k3 - k1, dtype=float)
                seen.setdefault(grid.key(q), q)
    else:
        raise ValueError("mode must be one of {'pp', 'ph', 'phc'}")
    return list(seen.values())


@dataclass
class FRGState:
    patchsets: PatchSetMap
    bare_gamma: GammaAccessor
    pp_grid: TransferGrid
    phd_grid: TransferGrid
    phc_grid: TransferGrid
    spin_blocks: List[SpinBlock]
    T: float
    pp_corr: Dict[ChannelKey, np.ndarray] = field(default_factory=dict)
    phd_corr: Dict[ChannelKey, np.ndarray] = field(default_factory=dict)
    phc_corr: Dict[ChannelKey, np.ndarray] = field(default_factory=dict)

    def channel_norm(self) -> float:
        vals: List[float] = []
        for store in (self.pp_corr, self.phd_corr, self.phc_corr):
            vals.extend([float(np.max(np.abs(v))) for v in store.values() if v.size])
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
    leading_channel_name: Optional[str] = None
    leading_Q: Optional[np.ndarray] = None
    leading_eigenvalue_abs: Optional[float] = None
    leading_order_label: Optional[str] = None
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
            "leading_channel_name": self.leading_channel_name,
            "leading_Q": None if self.leading_Q is None else np.asarray(self.leading_Q, dtype=float).tolist(),
            "leading_eigenvalue_abs": None if self.leading_eigenvalue_abs is None else float(self.leading_eigenvalue_abs),
            "leading_order_label": self.leading_order_label,
            "diagnosis_payload": self.diagnosis_payload,
        }


@dataclass
class FastVertexEvaluator:
    solver: "FRGFlowSolver"

    def __call__(self, p1: int, s1: str, p2: int, s2: str, p3: int, s3: str, p4: int, s4: str) -> complex:
        slv = self.solver
        s1n, s2n, s3n, s4n = canonical_spin_tuple((s1, s2, s3, s4))
        val = complex(slv.state.bare_gamma(p1, s1n, p2, s2n, p3, s3n, p4, s4n))

        iq_pp = slv._pp_q_index[(s1n, s2n)][p1, p2]
        val += complex(slv.state.pp_corr[(s1n, s2n, s3n, s4n, iq_pp)][p3, p1])

        iq_phd = slv._phd_q_index[(s3n, s1n)][p3, p1]
        val += complex(slv.state.phd_corr[(s1n, s2n, s3n, s4n, iq_phd)][p4, p1])

        iq_phc = slv._phc_q_index[(s3n, s2n)][p3, p2]
        val += complex(slv.state.phc_corr[(s1n, s2n, s3n, s4n, iq_phc)][p2, p1])
        return val


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
        diagnoser: Optional[Any] = None,
        T_start: float = 0.5,
        T_stop: float = 0.02,
        n_steps: int = 24,
        temperature_grid: str = "log",
        nfreq: int = 128,
        include_explicit_T_prefactor: bool = True,
        max_relative_update: float = 0.15,
        min_substep_fraction: float = 1.0 / 128.0,
        channel_divergence_threshold: float = 1e3,
        eigenvalue_threshold: float = 1e2,
        diagnose_every: int = 1,
        diagnosis_Qs: Optional[Sequence[Sequence[float]]] = None,
        diagnosis_sort_by: str = "abs",
        track_crossed_channel: bool = True,
    ) -> None:
        self.patchsets = patchsets
        self.bare_gamma = bare_gamma
        self.spin_blocks = [canonical_spin_tuple(x) for x in (spin_blocks or default_spin_blocks(patchsets))]
        self.allowed_spin_blocks = frozenset(self.spin_blocks)

        if pp_Qs is None:
            pp_Qs = build_unique_q_list(patchsets, mode="pp")
        if ph_Qs is None:
            ph_Qs = build_unique_q_list(patchsets, mode="ph")
        if phc_Qs is None:
            phc_Qs = build_unique_q_list(patchsets, mode="phc")

        self.pp_grid = TransferGrid(patchsets, pp_Qs)
        self.phd_grid = TransferGrid(patchsets, ph_Qs)
        self.phc_grid = TransferGrid(patchsets, phc_Qs)

        self.diagnoser = diagnoser if diagnoser is not None else (
            KagomeOrderDiagnoser(patchsets_by_spin=patchsets) if KagomeOrderDiagnoser is not None else None
        )

        self.T_start = float(T_start)
        self.T_stop = float(T_stop)
        if self.T_start <= self.T_stop:
            raise ValueError("Require T_start > T_stop for a descending temperature flow.")
        self.n_steps = int(n_steps)
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        self.temperature_grid = str(temperature_grid).lower()
        if self.temperature_grid not in {"log", "linear"}:
            raise ValueError("temperature_grid must be 'log' or 'linear'.")

        self.nfreq = int(nfreq)
        self.include_explicit_T_prefactor = bool(include_explicit_T_prefactor)
        self.max_relative_update = float(max_relative_update)
        self.min_substep_fraction = float(min_substep_fraction)
        self.channel_divergence_threshold = float(channel_divergence_threshold)
        self.eigenvalue_threshold = float(eigenvalue_threshold)
        self.diagnose_every = int(diagnose_every)
        self.diagnosis_sort_by = str(diagnosis_sort_by)
        self.track_crossed_channel = bool(track_crossed_channel)

        if diagnosis_Qs is None:
            self.diagnosis_Qs = [np.asarray(q, dtype=float) for q in self.phd_grid.q_list]
        else:
            self.diagnosis_Qs = [np.asarray(q, dtype=float) for q in diagnosis_Qs]

        self.state = FRGState(
            patchsets=patchsets,
            bare_gamma=bare_gamma,
            pp_grid=self.pp_grid,
            phd_grid=self.phd_grid,
            phc_grid=self.phc_grid,
            spin_blocks=list(self.spin_blocks),
            T=float(T_start),
            pp_corr=self._empty_channel_store(self.pp_grid),
            phd_corr=self._empty_channel_store(self.phd_grid),
            phc_corr=self._empty_channel_store(self.phc_grid),
        )

        self._precompute_transfer_tables()
        self._fast_gamma = FastVertexEvaluator(self)
        self.bare_vertex_norm = self._estimate_bare_vertex_norm()

        self.temperature_path = self._build_temperature_path()
        self.history: List[FlowStepRecord] = []
        self.instability_record: Optional[FlowStepRecord] = None

    def _all_spins_present(self) -> List[str]:
        return available_physical_spins(self.patchsets)

    def _patch_k_array(self, spin: str) -> np.ndarray:
        ps = patchset_for_spin(self.patchsets, spin)
        return np.asarray([p.k_cart for p in ps.patches], dtype=float)

    def _precompute_transfer_tables(self) -> None:
        spins = self._all_spins_present()
        self._pp_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._phd_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index: Dict[Tuple[str, str], np.ndarray] = {}

        for s1 in spins:
            k1s = self._patch_k_array(s1)
            for s2 in spins:
                k2s = self._patch_k_array(s2)
                arr = np.zeros((len(k1s), len(k2s)), dtype=int)
                for p1, k1 in enumerate(k1s):
                    for p2, k2 in enumerate(k2s):
                        arr[p1, p2] = self.pp_grid.nearest_index(k1 + k2)
                self._pp_q_index[(s1, s2)] = arr

        for s3 in spins:
            k3s = self._patch_k_array(s3)
            for s1 in spins:
                k1s = self._patch_k_array(s1)
                arr = np.zeros((len(k3s), len(k1s)), dtype=int)
                for p3, k3 in enumerate(k3s):
                    for p1, k1 in enumerate(k1s):
                        arr[p3, p1] = self.phd_grid.nearest_index(k3 - k1)
                self._phd_q_index[(s3, s1)] = arr

        for s3 in spins:
            k3s = self._patch_k_array(s3)
            for s2 in spins:
                k2s = self._patch_k_array(s2)
                arr = np.zeros((len(k3s), len(k2s)), dtype=int)
                for p3, k3 in enumerate(k3s):
                    for p2, k2 in enumerate(k2s):
                        arr[p3, p2] = self.phc_grid.nearest_index(k3 - k2)
                self._phc_q_index[(s3, s2)] = arr

    def _empty_channel_store(self, grid: TransferGrid) -> Dict[ChannelKey, np.ndarray]:
        ref_spin = available_physical_spins(self.patchsets)[0]
        Np = patchset_for_spin(self.patchsets, ref_spin).Npatch
        out: Dict[ChannelKey, np.ndarray] = {}
        for key in self.spin_blocks:
            for iq in range(len(grid.q_list)):
                out[(key[0], key[1], key[2], key[3], iq)] = np.zeros((Np, Np), dtype=complex)
        return out

    def _flow_config(self, T: float) -> FlowConfig:
        return FlowConfig(
            temperature=float(T),
            nfreq=self.nfreq,
            include_explicit_T_prefactor=self.include_explicit_T_prefactor,
        )

    def _build_temperature_path(self) -> np.ndarray:
        if self.temperature_grid == "log":
            return np.geomspace(self.T_start, self.T_stop, self.n_steps)
        return np.linspace(self.T_start, self.T_stop, self.n_steps)

    def _channel_elem(self, store: Dict[ChannelKey, np.ndarray], key: ChannelKey, i: int, j: int) -> complex:
        mat = store.get(key)
        if mat is None:
            return 0.0 + 0.0j
        return complex(mat[i, j])

    def _estimate_bare_vertex_norm(self, sample_cap: int = 6) -> float:
        ref_spin = available_physical_spins(self.patchsets)[0]
        Np = patchset_for_spin(self.patchsets, ref_spin).Npatch
        P = min(sample_cap, Np)
        vals = []
        for (s1, s2, s3, s4) in self.spin_blocks:
            for p1 in range(P):
                for p2 in range(P):
                    for p3 in range(P):
                        p4 = 0
                        vals.append(abs(self.bare_gamma(p1, s1, p2, s2, p3, s3, p4, s4)))
        if not vals:
            return 1e-14
        return max(float(np.max(vals)), 1e-14)

    def gamma_accessor(self) -> GammaAccessor:
        state = self.state

        def gamma(p1: int, s1: str, p2: int, s2: str, p3: int, s3: str, p4: int, s4: str) -> complex:
            s1n, s2n, s3n, s4n = canonical_spin_tuple((s1, s2, s3, s4))
            k1 = np.asarray(patchset_for_spin(state.patchsets, s1n).patches[p1].k_cart, dtype=float)
            k2 = np.asarray(patchset_for_spin(state.patchsets, s2n).patches[p2].k_cart, dtype=float)
            k3 = np.asarray(patchset_for_spin(state.patchsets, s3n).patches[p3].k_cart, dtype=float)

            val = complex(state.bare_gamma(p1, s1n, p2, s2n, p3, s3n, p4, s4n))

            iq_pp = state.pp_grid.nearest_index(k1 + k2)
            val += self._channel_elem(state.pp_corr, (s1n, s2n, s3n, s4n, iq_pp), p3, p1)

            iq_phd = state.phd_grid.nearest_index(k3 - k1)
            val += self._channel_elem(state.phd_corr, (s1n, s2n, s3n, s4n, iq_phd), p4, p1)

            iq_phc = state.phc_grid.nearest_index(k3 - k2)
            val += self._channel_elem(state.phc_corr, (s1n, s2n, s3n, s4n, iq_phc), p2, p1)

            return complex(val)

        return gamma

    def _rhs_norm(self, rhs_pp, rhs_phd, rhs_phc) -> float:
        vals: List[float] = []
        for store in (rhs_pp, rhs_phd, rhs_phc):
            vals.extend([float(np.max(np.abs(v))) for v in store.values() if v.size])
        return max(vals) if vals else 0.0

    def compute_channel_rhs(self, T: float):
        gamma = self._fast_gamma
        cfg = self._flow_config(T)

        rhs_pp = self._empty_channel_store(self.pp_grid)
        rhs_phd = self._empty_channel_store(self.phd_grid)
        rhs_phc = self._empty_channel_store(self.phc_grid)

        for iq, Q in enumerate(self.pp_grid.q_list):
            for s1, s2, s3, s4 in self.spin_blocks:
                ker = compute_pp_kernel_fast(
                    gamma,
                    self.patchsets,
                    Q,
                    incoming_spins=(s1, s2),
                    outgoing_spins=(s3, s4),
                    config=cfg,
                    allowed_spin_blocks=self.allowed_spin_blocks,
                )
                rhs_pp[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)

        for iq, Q in enumerate(self.phd_grid.q_list):
            for s1, s2, s3, s4 in self.spin_blocks:
                ker = compute_ph_kernel_fast(
                    gamma,
                    self.patchsets,
                    Q,
                    incoming_spins=(s1, s3),
                    outgoing_spins=(s4, s2),
                    config=cfg,
                    allowed_spin_blocks=self.allowed_spin_blocks,
                )
                rhs_phd[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)

        for iq, Q in enumerate(self.phc_grid.q_list):
            for s1, s2, s3, s4 in self.spin_blocks:
                ker = compute_phc_kernel_fast(
                    gamma,
                    self.patchsets,
                    Q,
                    incoming_spins=(s1, s2),
                    outgoing_spins=(s3, s4),
                    config=cfg,
                    allowed_spin_blocks=self.allowed_spin_blocks,
                )
                rhs_phc[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)

        return rhs_pp, rhs_phd, rhs_phc

    def _vertex_pp_kernel(self, Q: Sequence[float], *, incoming_spins: Tuple[str, str], outgoing_spins: Tuple[str, str]) -> ChannelKernel:
        Q = np.asarray(Q, dtype=float)
        gamma = self._fast_gamma
        s1, s2 = map(normalize_spin, incoming_spins)
        s3, s4 = map(normalize_spin, outgoing_spins)
        ps_in = patchset_for_spin(self.patchsets, s1)
        partner_in, resid_in = shifted_patch_map(self.patchsets, s2, Q, mode="Q_minus_k")
        partner_out, resid_out = shifted_patch_map(self.patchsets, s4, Q, mode="Q_minus_k")
        N = ps_in.Npatch
        M = np.zeros((N, N), dtype=complex)
        residuals = np.zeros((N, N), dtype=float)
        for pout in range(N):
            p4 = int(partner_out[pout])
            for pin in range(N):
                p2 = int(partner_in[pin])
                M[pout, pin] = gamma(pin, s1, p2, s2, pout, s3, p4, s4)
                residuals[pout, pin] = max(resid_in[pin], resid_out[pout])
        return ChannelKernel(
            name="pp",
            Q=Q,
            matrix=M,
            row_patches=np.arange(N, dtype=int),
            col_patches=np.arange(N, dtype=int),
            row_partner_patches=np.asarray(partner_out, dtype=int),
            col_partner_patches=np.asarray(partner_in, dtype=int),
            row_spins=(s3, s4),
            col_spins=(s1, s2),
            residuals=residuals,
        )

    def _vertex_phd_kernel(self, Q: Sequence[float], *, incoming_spins: Tuple[str, str], outgoing_spins: Tuple[str, str]) -> ChannelKernel:
        Q = np.asarray(Q, dtype=float)
        gamma = self._fast_gamma
        s1, s3 = map(normalize_spin, incoming_spins)
        s4, s2 = map(normalize_spin, outgoing_spins)
        ps1 = patchset_for_spin(self.patchsets, s1)
        kplus_in, resid_in = shifted_patch_map(self.patchsets, s3, Q, mode="k_plus_Q")
        kplus_out, resid_out = shifted_patch_map(self.patchsets, s2, Q, mode="k_plus_Q")
        N = ps1.Npatch
        M = np.zeros((N, N), dtype=complex)
        residuals = np.zeros((N, N), dtype=float)
        for pout in range(N):
            p2 = int(kplus_out[pout])
            for pin in range(N):
                p3 = int(kplus_in[pin])
                M[pout, pin] = gamma(pin, s1, p2, s2, p3, s3, pout, s4)
                residuals[pout, pin] = max(resid_in[pin], resid_out[pout])
        return ChannelKernel(
            name="ph_direct",
            Q=Q,
            matrix=M,
            row_patches=np.arange(N, dtype=int),
            col_patches=np.arange(N, dtype=int),
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kplus_in, dtype=int),
            row_spins=(s4, s2),
            col_spins=(s1, s3),
            residuals=residuals,
        )

    def _vertex_phc_kernel(self, Q: Sequence[float], *, incoming_spins: Tuple[str, str], outgoing_spins: Tuple[str, str]) -> ChannelKernel:
        Q = np.asarray(Q, dtype=float)
        gamma = self._fast_gamma
        s1, s2 = map(normalize_spin, incoming_spins)
        s3, s4 = map(normalize_spin, outgoing_spins)
        ps1 = patchset_for_spin(self.patchsets, s1)
        kplus_out, resid_out = shifted_patch_map(self.patchsets, s3, Q, mode="k_plus_Q")
        kminus_in, resid_in = shifted_patch_map(self.patchsets, s4, Q, mode="k_minus_Q")
        N = ps1.Npatch
        M = np.zeros((N, N), dtype=complex)
        residuals = np.zeros((N, N), dtype=float)
        for pout in range(N):
            p3 = int(kplus_out[pout])
            for pin in range(N):
                p4 = int(kminus_in[pin])
                M[pout, pin] = gamma(pin, s1, pout, s2, p3, s3, p4, s4)
                residuals[pout, pin] = max(resid_in[pin], resid_out[pout])
        return ChannelKernel(
            name="ph_crossed",
            Q=Q,
            matrix=M,
            row_patches=np.arange(N, dtype=int),
            col_patches=np.arange(N, dtype=int),
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kminus_in, dtype=int),
            row_spins=(s2, s3),
            col_spins=(s1, s4),
            residuals=residuals,
        )

    def build_diagnosis_kernel_dict(self, Q: Sequence[float]) -> Dict[str, ChannelKernel]:
        Q = np.asarray(Q, dtype=float)
        out: Dict[str, ChannelKernel] = {}

        if has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn"):
            K_ud_ud = self._vertex_pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("up", "dn"))
            K_ud_du = self._vertex_pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("dn", "up"))
            K_du_ud = self._vertex_pp_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("up", "dn"))
            K_du_du = self._vertex_pp_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("dn", "up"))
            out["pp_ud_to_ud"] = K_ud_ud
            out["pp_ud_to_du"] = K_ud_du
            out["pp_du_to_ud"] = K_du_ud
            out["pp_du_to_du"] = K_du_du

            template = K_ud_ud
            out["pp_singlet_sz0"] = ChannelKernel(
                name="pp_singlet_sz0",
                Q=Q,
                matrix=0.5 * (K_ud_ud.matrix - K_ud_du.matrix - K_du_ud.matrix + K_du_du.matrix),
                row_patches=template.row_patches.copy(),
                col_patches=template.col_patches.copy(),
                row_partner_patches=template.row_partner_patches.copy(),
                col_partner_patches=template.col_partner_patches.copy(),
                row_spins=("S", "S"),
                col_spins=("S", "S"),
                residuals=template.residuals.copy(),
            )
            out["pp_triplet_sz0"] = ChannelKernel(
                name="pp_triplet_sz0",
                Q=Q,
                matrix=0.5 * (K_ud_ud.matrix + K_ud_du.matrix + K_du_ud.matrix + K_du_du.matrix),
                row_patches=template.row_patches.copy(),
                col_patches=template.col_patches.copy(),
                row_partner_patches=template.row_partner_patches.copy(),
                col_partner_patches=template.col_partner_patches.copy(),
                row_spins=("T", "T"),
                col_spins=("T", "T"),
                residuals=template.residuals.copy(),
            )

        if has_patchset(self.patchsets, "up"):
            out["pp_triplet_uu"] = self._vertex_pp_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
            out["phd_uu"] = self._vertex_phd_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
            if self.track_crossed_channel:
                out["phc_uu"] = self._vertex_phc_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))

        if has_patchset(self.patchsets, "dn"):
            out["pp_triplet_dd"] = self._vertex_pp_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
            out["phd_dd"] = self._vertex_phd_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
            if self.track_crossed_channel:
                out["phc_dd"] = self._vertex_phc_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))

        if "phd_uu" in out and "phd_dd" in out:
            K_uu = out["phd_uu"]
            K_dd = out["phd_dd"]
            residuals = np.maximum(K_uu.residuals, K_dd.residuals)
            out["ph_charge_longitudinal"] = ChannelKernel(
                name="ph_charge_longitudinal",
                Q=Q,
                matrix=0.5 * (K_uu.matrix + K_dd.matrix),
                row_patches=K_uu.row_patches.copy(),
                col_patches=K_uu.col_patches.copy(),
                row_partner_patches=K_uu.row_partner_patches.copy(),
                col_partner_patches=K_uu.col_partner_patches.copy(),
                row_spins=("rho", "rho"),
                col_spins=("rho", "rho"),
                residuals=residuals,
            )
            out["ph_spin_longitudinal"] = ChannelKernel(
                name="ph_spin_longitudinal",
                Q=Q,
                matrix=0.5 * (K_uu.matrix - K_dd.matrix),
                row_patches=K_uu.row_patches.copy(),
                col_patches=K_uu.col_patches.copy(),
                row_partner_patches=K_uu.row_partner_patches.copy(),
                col_partner_patches=K_uu.col_partner_patches.copy(),
                row_spins=("sz", "sz"),
                col_spins=("sz", "sz"),
                residuals=residuals,
            )
        return out

    def diagnose_current_state(self) -> Dict[str, Any]:
        best_abs_eval = -np.inf
        best = {
            "channel_name": None,
            "Q": None,
            "abs_eigenvalue": None,
            "order_label": None,
            "diagnosis": None,
        }

        for Q in self.diagnosis_Qs:
            kernels = self.build_diagnosis_kernel_dict(Q)
            for name, kernel in kernels.items():
                vals, _ = kernel.eig(sort_by=self.diagnosis_sort_by)
                if len(vals) == 0:
                    continue
                lam = float(np.abs(vals[0]))
                if lam > best_abs_eval:
                    best_abs_eval = lam
                    if self.diagnoser is not None:
                        diag = self.diagnoser.diagnose_kernel(kernel, sort_by=self.diagnosis_sort_by)
                        best = {
                            "channel_name": name,
                            "Q": np.asarray(Q, dtype=float),
                            "abs_eigenvalue": lam,
                            "order_label": diag.paper_label,
                            "diagnosis": diag.summary_dict(),
                        }
                    else:
                        best = {
                            "channel_name": name,
                            "Q": np.asarray(Q, dtype=float),
                            "abs_eigenvalue": lam,
                            "order_label": None,
                            "diagnosis": None,
                        }
        return best

    def _apply_rhs(self, rhs_pp, rhs_phd, rhs_phc, scale: float) -> None:
        for key, mat in rhs_pp.items():
            self.state.pp_corr[key] += scale * np.asarray(mat, dtype=complex)
        for key, mat in rhs_phd.items():
            self.state.phd_corr[key] += scale * np.asarray(mat, dtype=complex)
        for key, mat in rhs_phc.items():
            self.state.phc_corr[key] += scale * np.asarray(mat, dtype=complex)

    def check_instability(self, record: FlowStepRecord) -> Tuple[bool, Optional[str]]:
        if record.channel_norm >= self.channel_divergence_threshold:
            return True, f"channel norm={record.channel_norm:.3e} exceeded channel_divergence_threshold"
        if record.leading_eigenvalue_abs is not None and record.leading_eigenvalue_abs >= self.eigenvalue_threshold:
            return True, f"leading eigenvalue={record.leading_eigenvalue_abs:.3e} exceeded eigenvalue_threshold"
        return False, None

    def step(self, T_old: float, dT: float) -> FlowStepRecord:
        rhs_pp, rhs_phd, rhs_phc = self.compute_channel_rhs(T_old)

        effective_norm = max(self.state.channel_norm(), self.bare_vertex_norm, 1e-14)
        rhs_norm = self._rhs_norm(rhs_pp, rhs_phd, rhs_phc)
        rel_update = abs(dT) * rhs_norm / effective_norm

        n_sub = max(1, int(np.ceil(rel_update / self.max_relative_update))) if rel_update > self.max_relative_update else 1
        if (1.0 / n_sub) < self.min_substep_fraction:
            raise RuntimeError(
                "Adaptive step control requested too many substeps. Reduce the temperature spacing or relax max_relative_update."
            )

        sub_dT = dT / n_sub
        for _ in range(n_sub):
            self._apply_rhs(rhs_pp, rhs_phd, rhs_phc, sub_dT)

        self.state.T = float(T_old + dT)

        return FlowStepRecord(
            step_index=len(self.history),
            temperature=float(T_old + dT),
            dT=float(dT),
            channel_norm=self.state.channel_norm(),
            rhs_norm=rhs_norm,
            accepted_substeps=n_sub,
            max_rel_update=rel_update / n_sub if n_sub > 0 else rel_update,
        )

    def run(self) -> List[FlowStepRecord]:
        temps = self.temperature_path

        initial = self.diagnose_current_state()
        rec0 = FlowStepRecord(
            step_index=0,
            temperature=float(temps[0]),
            dT=0.0,
            channel_norm=self.state.channel_norm(),
            rhs_norm=0.0,
            accepted_substeps=0,
            max_rel_update=0.0,
            leading_channel_name=initial.get("channel_name"),
            leading_Q=initial.get("Q"),
            leading_eigenvalue_abs=initial.get("abs_eigenvalue"),
            leading_order_label=initial.get("order_label"),
            diagnosis_payload=initial.get("diagnosis") or {},
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

            if ((i + 1) % self.diagnose_every) == 0 or i == len(temps) - 2:
                payload = self.diagnose_current_state()
                rec.leading_channel_name = payload.get("channel_name")
                rec.leading_Q = payload.get("Q")
                rec.leading_eigenvalue_abs = payload.get("abs_eigenvalue")
                rec.leading_order_label = payload.get("order_label")
                rec.diagnosis_payload = payload.get("diagnosis") or {}

            rec.instability, rec.instability_reason = self.check_instability(rec)
            self.history.append(rec)

            if rec.instability:
                self.instability_record = rec
                break

        return self.history

    def history_as_dicts(self) -> List[Dict[str, Any]]:
        return [x.summary_dict() for x in self.history]

    def current_gamma_accessor(self) -> GammaAccessor:
        return self._fast_gamma



# ===== Final stage-2 optimized solver =====
class FRGFlowSolver(FRGFlowSolver):
    """Final optimized solver.

    Keeps the stage-1 fast vertex evaluator / transfer-index cache, and adds
    Q-level internal-cache reuse plus vectorized bubble frequency sums.
    Physics is unchanged relative to the reference solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._precompute_shift_maps()

    def _precompute_shift_maps(self) -> None:
        spins = self._all_spins_present()
        self._pp_qminus = {}
        self._ph_kplus = {}
        self._phc_kplus = {}
        self._phc_kminus = {}

        for iq, Q in enumerate(self.pp_grid.q_list):
            Q = np.asarray(Q, dtype=float)
            for s in spins:
                self._pp_qminus[(iq, s)] = shifted_patch_map(self.patchsets, s, Q, mode="Q_minus_k")

        for iq, Q in enumerate(self.phd_grid.q_list):
            Q = np.asarray(Q, dtype=float)
            for s in spins:
                self._ph_kplus[(iq, s)] = shifted_patch_map(self.patchsets, s, Q, mode="k_plus_Q")

        for iq, Q in enumerate(self.phc_grid.q_list):
            Q = np.asarray(Q, dtype=float)
            for s in spins:
                self._phc_kplus[(iq, s)] = shifted_patch_map(self.patchsets, s, Q, mode="k_plus_Q")
                self._phc_kminus[(iq, s)] = shifted_patch_map(self.patchsets, s, Q, mode="k_minus_Q")

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
    "BareVertexFromInteraction",
    "FRGFlowSolver",
    "FRGState",
    "FlowStepRecord",
    "TransferGrid",
    "available_physical_spins",
    "build_unique_q_list",
    "canonical_spin_tuple",
    "default_spin_blocks",
]
