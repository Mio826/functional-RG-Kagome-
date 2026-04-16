from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from channels import ChannelKernel
from frg_kernel import (
    FlowConfig,
    PatchSetMap,
    bubble_dot_ph,
    build_ph_internal_cache_vec,
    build_pp_internal_cache_vec,
    canonicalize_q_for_patchsets,
    partner_map_from_q_index,
    patch_measure_vector,
    patchset_for_spin,
)


# ---------------------------------------------------------------------------
# Bubble data
# ---------------------------------------------------------------------------


@dataclass
class BubbleWeights:
    """Patch-diagonal bubble weights used in instability diagnosis."""

    channel_type: str
    Q: np.ndarray
    weights: np.ndarray
    partner_patches: np.ndarray
    residuals: np.ndarray
    temperature: float
    source: str
    notes: Tuple[str, ...] = ()

    @property
    def Npatch(self) -> int:
        return int(self.weights.shape[0])


# ---------------------------------------------------------------------------
# Spectrum / multiplet / complex-order metadata
# ---------------------------------------------------------------------------


@dataclass
class LeadingModeInfo:
    score: float
    eval: float
    evec: np.ndarray
    source: str
    projection_name: Optional[str] = None


@dataclass
class MultipletCandidate:
    channel_name: str
    channel_type: str
    spin_structure: str
    Q: np.ndarray
    source: str
    indices: np.ndarray
    evals: np.ndarray
    basis_vectors: np.ndarray
    dimension: int
    spread_abs: float
    spread_rel: float
    eval_max_abs: float
    degeneracy_type: str
    physical_irrep_like: bool
    notes: Tuple[str, ...] = ()

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "channel_name": self.channel_name,
            "channel_type": self.channel_type,
            "spin_structure": self.spin_structure,
            "Q": np.asarray(self.Q, dtype=float).tolist(),
            "source": self.source,
            "indices": [int(i) for i in self.indices],
            "evals": [float(x) for x in self.evals],
            "dimension": int(self.dimension),
            "spread_abs": float(self.spread_abs),
            "spread_rel": float(self.spread_rel),
            "eval_max_abs": float(self.eval_max_abs),
            "degeneracy_type": self.degeneracy_type,
            "physical_irrep_like": bool(self.physical_irrep_like),
            "notes": list(self.notes),
        }


@dataclass
class ComplexOrderCandidate:
    parent_source: str
    channel_name: str
    channel_type: str
    spin_structure: str
    Q: np.ndarray
    combo_type: str
    basis_indices: Tuple[int, int]
    vector_complex: np.ndarray
    candidate_label: str
    tr_breaking_possible: bool
    notes: Tuple[str, ...] = ()

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "parent_source": self.parent_source,
            "channel_name": self.channel_name,
            "channel_type": self.channel_type,
            "spin_structure": self.spin_structure,
            "Q": np.asarray(self.Q, dtype=float).tolist(),
            "combo_type": self.combo_type,
            "basis_indices": [int(x) for x in self.basis_indices],
            "candidate_label": self.candidate_label,
            "tr_breaking_possible": bool(self.tr_breaking_possible),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Config and result objects
# ---------------------------------------------------------------------------


@dataclass
class InstabilityConfig:
    """Configuration for bubble-weighted instability diagnosis.

    Notes
    -----
    - ``ph_sign`` is fixed by convention for the current code base.
    - ``pp_sign`` defaults to ``+1`` in the repaired pp diagnosis, together with
      ``abs`` bubble weights, to avoid re-promoting the local repulsive
      Q=0 pp-singlet mode as a fake pairing instability.
    - The patch-measure fields must match the flow setup whenever you want the
      instability diagnosis to analyze the same discretized problem as the flow.
    """

    ph_sign: int = -1
    pp_sign: Optional[int] = +1
    use_hermitian_part: bool = True
    bubble_floor: float = 0.0
    q0_tol: float = 1e-10
    ph_bubble_mode: str = "patchrep"  # "patchrep" or "internal_cache"
    project_ph_charge_q0_uniform: bool = True
    report_pp_singlet_q0_local_gram_both: bool = True
    project_pp_singlet_q0_local_gram_default: bool = False
    store_operator_matrices: bool = True
    store_all_evals: bool = False
    projection_tol: float = 1e-12

    # Must be synchronized with the flow if flow uses non-unit patch measure.
    patch_measure_mode: str = "unit"
    patch_measure_soft_vf_eps: Optional[float] = None
    patch_measure_normalize_mean: bool = False

    # New multiplet / complex-order analysis
    analyze_multiplets: bool = True
    multiplet_top_n: int = 6
    multiplet_rtol = 1e-2   # 1%
    multiplet_atol = 1e-8
    multiplet_eval_floor = 1e-8
    multiplet_eval_rel_floor: float = 1e-6
    build_complex_candidates: bool = True
    max_complex_pairs_per_multiplet: int = 1


@dataclass
class InstabilityResult:
    """Diagnosis result for one physical channel at fixed Q."""

    channel_name: str
    channel_type: str
    spin_structure: str
    Q: np.ndarray
    sign_used: Optional[int]
    hermitian_residual: float
    bubble: BubbleWeights

    score: float
    leading_eval: float
    leading_evec: np.ndarray

    score_unprojected: float
    leading_eval_unprojected: float
    leading_evec_unprojected: np.ndarray

    score_projected: Optional[float] = None
    leading_eval_projected: Optional[float] = None
    leading_evec_projected: Optional[np.ndarray] = None

    projection_name: Optional[str] = None
    projection_rank: int = 0
    projection_basis_vectors: Optional[np.ndarray] = None

    operator_unprojected: Optional[np.ndarray] = None
    operator_projected: Optional[np.ndarray] = None
    hermitian_matrix: Optional[np.ndarray] = None
    all_evals_unprojected: Optional[np.ndarray] = None
    all_evals_projected: Optional[np.ndarray] = None

    # New physics-aware diagnostics
    single_mode_label: Optional[str] = None
    single_mode_notes: Tuple[str, ...] = ()
    leading_mode_info: Optional[LeadingModeInfo] = None
    multiplet_candidates: Tuple[MultipletCandidate, ...] = ()
    leading_multiplet_candidate: Optional[MultipletCandidate] = None
    complex_order_candidates: Tuple[ComplexOrderCandidate, ...] = ()

    notes: Tuple[str, ...] = ()

    def summary_dict(self) -> Dict[str, Any]:
        def _maybe_float(x: Optional[float]) -> Optional[float]:
            return None if x is None else float(x)

        return {
            "channel_name": self.channel_name,
            "channel_type": self.channel_type,
            "spin_structure": self.spin_structure,
            "Q": np.asarray(self.Q, dtype=float).tolist(),
            "sign_used": self.sign_used,
            "score": float(self.score),
            "leading_eval": float(self.leading_eval),
            "score_unprojected": float(self.score_unprojected),
            "leading_eval_unprojected": float(self.leading_eval_unprojected),
            "score_projected": _maybe_float(self.score_projected),
            "leading_eval_projected": _maybe_float(self.leading_eval_projected),
            "projection_name": self.projection_name,
            "projection_rank": int(self.projection_rank),
            "hermitian_residual": float(self.hermitian_residual),
            "bubble_source": self.bubble.source,
            "bubble_temperature": float(self.bubble.temperature),
            "single_mode_label": self.single_mode_label,
            "single_mode_notes": list(self.single_mode_notes),
            "n_multiplet_candidates": len(self.multiplet_candidates),
            "leading_multiplet_candidate": None if self.leading_multiplet_candidate is None else self.leading_multiplet_candidate.summary_dict(),
            "n_complex_order_candidates": len(self.complex_order_candidates),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def infer_channel_type(channel_kernel: ChannelKernel) -> str:
    channel_type = getattr(channel_kernel, "channel_type", None)
    if isinstance(channel_type, str) and channel_type in {"pp", "ph"}:
        return channel_type

    name = str(getattr(channel_kernel, "name", "")).lower()
    if name.startswith("pp"):
        return "pp"
    if name.startswith("ph"):
        return "ph"
    raise ValueError(f"Could not infer channel type from kernel name={channel_kernel.name!r}.")


def infer_spin_structure(channel_kernel: ChannelKernel) -> str:
    spin_structure = getattr(channel_kernel, "spin_structure", None)
    if isinstance(spin_structure, str) and spin_structure:
        return spin_structure

    name = str(getattr(channel_kernel, "name", "")).lower()
    for token in ("singlet", "triplet", "charge", "spin", "direct", "exchange"):
        if token in name:
            return token
    return "unknown"


def is_q0(Q: Sequence[float], *, tol: float = 1e-10) -> bool:
    q = np.asarray(Q, dtype=float)
    return bool(np.allclose(q, 0.0, atol=tol, rtol=0.0))


def _is_constant_like(vec: np.ndarray, *, tol: float = 1e-8) -> bool:
    v = np.asarray(vec, dtype=complex).reshape(-1)
    n = v.size
    if n == 0:
        return False
    u = np.ones(n, dtype=complex) / np.sqrt(float(n))
    ov = np.abs(np.vdot(u, v)) / max(np.linalg.norm(v), 1e-30)
    return bool(abs(1.0 - ov) <= tol)


# ---------------------------------------------------------------------------
# Bubble construction helpers
# ---------------------------------------------------------------------------


def _tuple_key(table: Mapping[Tuple[str, str], np.ndarray], first: str, second: str) -> np.ndarray:
    key = (str(first), str(second))
    if key in table:
        return np.asarray(table[key], dtype=int)
    raise KeyError(f"Missing q-index table for spin pair {key}.")


def _patch_energies(ps) -> np.ndarray:
    return np.array([float(p.energy) for p in ps.patches], dtype=float)


def _resolve_measure_config(
    transfer_context: Mapping[str, Any],
    config: InstabilityConfig,
) -> Tuple[str, Optional[float], bool, Tuple[str, ...]]:
    notes = []
    mode = str(config.patch_measure_mode)
    eps = config.patch_measure_soft_vf_eps
    norm = bool(config.patch_measure_normalize_mean)

    ctx_mode = transfer_context.get("patch_measure_mode", None)
    ctx_eps = transfer_context.get("patch_measure_soft_vf_eps", None)
    ctx_norm = transfer_context.get("patch_measure_normalize_mean", None)

    if ctx_mode is not None and str(ctx_mode) != mode:
        notes.append(f"measure_mode_override: config={mode}, context={ctx_mode}; using config")
    if ctx_eps is not None and eps is not None and float(ctx_eps) != float(eps):
        notes.append(f"measure_soft_vf_eps_override: config={eps}, context={ctx_eps}; using config")
    if ctx_norm is not None and bool(ctx_norm) != norm:
        notes.append(f"measure_normalize_override: config={norm}, context={bool(ctx_norm)}; using config")

    return mode, eps, norm, tuple(notes)


def _ph_shift_from_context(
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    Q: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    phd_grid = transfer_context["phd_grid"]
    q_table = _tuple_key(transfer_context["phd_q_index_plus"], "up", "dn")
    Q_can = canonicalize_q_for_patchsets(patchsets, Q)
    iq = phd_grid.nearest_index(Q_can)
    return partner_map_from_q_index(
        patchsets,
        q_table,
        source_spin="up",
        target_spin="dn",
        iq_target=int(iq),
        Q=Q_can,
        mode="k_plus_Q",
    )


def _pp_shift_from_context(
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    Q: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    pp_grid = transfer_context["pp_grid"]
    q_table = _tuple_key(transfer_context["pp_q_index"], "up", "dn")
    Q_can = canonicalize_q_for_patchsets(patchsets, Q)
    iq = pp_grid.nearest_index(Q_can)
    return partner_map_from_q_index(
        patchsets,
        q_table,
        source_spin="up",
        target_spin="dn",
        iq_target=int(iq),
        Q=Q_can,
        mode="Q_minus_k",
    )


def _sanitize_bubble_weights(weights: np.ndarray, *, floor: Optional[float] = 0.0) -> Tuple[np.ndarray, Tuple[str, ...]]:
    notes: List[str] = []
    arr = np.asarray(weights)
    if np.iscomplexobj(arr):
        imag_max = float(np.max(np.abs(np.imag(arr)))) if arr.size else 0.0
        if imag_max > 1e-10:
            notes.append(f"bubble_weights_had_complex_part_max={imag_max:.3e}; took real part")
        arr = np.real(arr)
    arr = np.asarray(arr, dtype=float)
    if floor is not None:
        neg_mask = arr < floor
        if np.any(neg_mask):
            notes.append(f"bubble_weights_clipped_count={int(np.count_nonzero(neg_mask))}")
            arr = arr.copy()
            arr[neg_mask] = float(floor)
    return arr, tuple(notes)


def build_ph_bubble_weights_internal_cache(
    channel_kernel: ChannelKernel,
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    bubble_floor: float = 0.0,
    patch_measure_mode: str = "unit",
    patch_measure_soft_vf_eps: Optional[float] = None,
    patch_measure_normalize_mean: bool = False,
) -> BubbleWeights:
    Q = canonicalize_q_for_patchsets(patchsets, channel_kernel.Q)
    partner, residual = _ph_shift_from_context(patchsets, transfer_context, Q)
    legacy = build_ph_internal_cache_vec(
        patchsets,
        flow_config,
        shift_cache={("up", "dn"): (np.asarray(partner, dtype=int), np.asarray(residual, dtype=float))},
        patch_measure_mode=patch_measure_mode,
        patch_measure_soft_vf_eps=patch_measure_soft_vf_eps,
        patch_measure_normalize_mean=patch_measure_normalize_mean,
    )[("up", "dn")]
    weights, notes = _sanitize_bubble_weights(np.asarray(legacy["weights"], dtype=complex), floor=bubble_floor)
    notes = tuple(list(notes) + [
        f"patch_measure_mode={patch_measure_mode}",
        f"patch_measure_normalize_mean={bool(patch_measure_normalize_mean)}",
    ])
    return BubbleWeights(
        channel_type="ph",
        Q=np.asarray(Q, dtype=float),
        weights=weights,
        partner_patches=np.asarray(legacy["partner"], dtype=int),
        residuals=np.asarray(legacy["residual"], dtype=float),
        temperature=float(flow_config.temperature),
        source="build_ph_internal_cache_vec[(up,dn)]",
        notes=notes,
    )


def build_ph_bubble_weights_patchrep(
    channel_kernel: ChannelKernel,
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    bubble_floor: float = 0.0,
    patch_measure_mode: str = "unit",
    patch_measure_soft_vf_eps: Optional[float] = None,
    patch_measure_normalize_mean: bool = False,
) -> BubbleWeights:
    Q = canonicalize_q_for_patchsets(patchsets, channel_kernel.Q)
    partner, residual = _ph_shift_from_context(patchsets, transfer_context, Q)
    ps_src = patchset_for_spin(patchsets, "up")
    ps_tgt = patchset_for_spin(patchsets, "dn")
    eps_src = _patch_energies(ps_src)
    eps_tgt = _patch_energies(ps_tgt)
    measure_src = patch_measure_vector(
        ps_src,
        mode=patch_measure_mode,
        soft_vf_eps=patch_measure_soft_vf_eps,
        normalize_mean=patch_measure_normalize_mean,
    )

    weights = np.zeros(ps_src.Npatch, dtype=complex)
    valid = np.asarray(partner, dtype=int) >= 0
    if np.any(valid):
        vals = [
            bubble_dot_ph(float(eps_src[i]), float(eps_tgt[int(partner[i])]), flow_config)
            for i in np.flatnonzero(valid)
        ]
        idx = np.flatnonzero(valid)
        weights[idx] = np.asarray(measure_src[idx], dtype=float) * np.asarray(vals, dtype=complex)

    weights, notes = _sanitize_bubble_weights(weights, floor=bubble_floor)
    notes = tuple(list(notes) + [
        "ph_bubble_mode=patchrep",
        f"patch_measure_mode={patch_measure_mode}",
        f"patch_measure_normalize_mean={bool(patch_measure_normalize_mean)}",
    ])
    return BubbleWeights(
        channel_type="ph",
        Q=np.asarray(Q, dtype=float),
        weights=weights,
        partner_patches=np.asarray(partner, dtype=int),
        residuals=np.asarray(residual, dtype=float),
        temperature=float(flow_config.temperature),
        source="patchrep:bubble_dot_ph(eps_p,eps_p+Q)*measure_p",
        notes=notes,
    )


def build_ph_bubble_weights(
    channel_kernel: ChannelKernel,
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    bubble_floor: float = 0.0,
    mode: str = "patchrep",
    patch_measure_mode: str = "unit",
    patch_measure_soft_vf_eps: Optional[float] = None,
    patch_measure_normalize_mean: bool = False,
) -> BubbleWeights:
    if mode == "patchrep":
        return build_ph_bubble_weights_patchrep(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=bubble_floor,
            patch_measure_mode=patch_measure_mode,
            patch_measure_soft_vf_eps=patch_measure_soft_vf_eps,
            patch_measure_normalize_mean=patch_measure_normalize_mean,
        )
    if mode == "internal_cache":
        return build_ph_bubble_weights_internal_cache(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=bubble_floor,
            patch_measure_mode=patch_measure_mode,
            patch_measure_soft_vf_eps=patch_measure_soft_vf_eps,
            patch_measure_normalize_mean=patch_measure_normalize_mean,
        )
    raise ValueError("ph bubble mode must be 'patchrep' or 'internal_cache'.")


def build_pp_bubble_weights(
    channel_kernel: ChannelKernel,
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    bubble_floor: Optional[float] = None,
    patch_measure_mode: str = "unit",
    patch_measure_soft_vf_eps: Optional[float] = None,
    patch_measure_normalize_mean: bool = False,
) -> BubbleWeights:
    Q = canonicalize_q_for_patchsets(patchsets, channel_kernel.Q)
    partner, residual = _pp_shift_from_context(patchsets, transfer_context, Q)
    legacy = build_pp_internal_cache_vec(
        patchsets,
        flow_config,
        shift_cache={("up", "dn"): (np.asarray(partner, dtype=int), np.asarray(residual, dtype=float))},
        patch_measure_mode=patch_measure_mode,
        patch_measure_soft_vf_eps=patch_measure_soft_vf_eps,
        patch_measure_normalize_mean=patch_measure_normalize_mean,
    )[("up", "dn")]
    weights, notes = _sanitize_bubble_weights(np.asarray(legacy["weights"], dtype=complex), floor=bubble_floor)
    notes = tuple(list(notes) + [
        f"patch_measure_mode={patch_measure_mode}",
        f"patch_measure_normalize_mean={bool(patch_measure_normalize_mean)}",
    ])
    return BubbleWeights(
        channel_type="pp",
        Q=np.asarray(Q, dtype=float),
        weights=weights,
        partner_patches=np.asarray(legacy["partner"], dtype=int),
        residuals=np.asarray(legacy["residual"], dtype=float),
        temperature=float(flow_config.temperature),
        source="build_pp_internal_cache_vec[(up,dn)]",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def build_uniform_projection_basis(npatch: int) -> np.ndarray:
    if npatch <= 0:
        raise ValueError("npatch must be positive")
    return np.ones((npatch, 1), dtype=complex) / np.sqrt(float(npatch))


def _extract_patch_eigvec(patch: object, patch_index: Optional[int] = None) -> np.ndarray:
    for attr in ("eigvec", "u", "bloch_vec", "eigenvector"):
        if hasattr(patch, attr):
            vec = np.asarray(getattr(patch, attr), dtype=complex)
            if vec.ndim == 1 and vec.size > 0:
                return vec
    available = sorted(a for a in dir(patch) if not a.startswith("_"))
    idx_txt = "" if patch_index is None else f" for patch index {patch_index}"
    raise AttributeError(
        f"Could not find a Bloch eigenvector on the patch object{idx_txt}. "
        f"Tried attrs=('eigvec','u','bloch_vec','eigenvector'). "
        f"Available public attrs include: {available[:30]}"
    )


def build_local_gram_projection_basis(patchsets: PatchSetMap) -> np.ndarray:
    """Construct the Q=0 pp local Gram basis span{w_A, w_B, w_C}."""
    ps = patchset_for_spin(patchsets, "up")
    npatch = int(ps.Npatch)
    if npatch <= 0:
        raise ValueError("Patch set is empty.")
    first_vec = _extract_patch_eigvec(ps.patches[0], patch_index=0)
    norb = int(first_vec.size)
    W = np.zeros((npatch, norb), dtype=float)
    for i, patch in enumerate(ps.patches):
        eigvec = _extract_patch_eigvec(patch, patch_index=i)
        if eigvec.size != norb:
            raise ValueError("Inconsistent orbital dimension across patches.")
        W[i, :] = np.abs(eigvec) ** 2
    Qmat, _ = np.linalg.qr(W.astype(complex), mode="reduced")
    return np.asarray(Qmat, dtype=complex)


def _orthonormalize_columns(basis: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    B = np.asarray(basis, dtype=complex)
    if B.ndim != 2:
        raise ValueError("Projection basis must be a 2D array with column vectors.")
    if B.shape[1] == 0:
        return B
    Qmat, Rmat = np.linalg.qr(B, mode="reduced")
    keep = np.abs(np.diag(Rmat)) > tol
    if not np.any(keep):
        return np.zeros((B.shape[0], 0), dtype=complex)
    return np.asarray(Qmat[:, keep], dtype=complex)


def complement_from_basis(basis_vectors: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    B = _orthonormalize_columns(basis_vectors, tol=tol)
    n = int(B.shape[0])
    if B.shape[1] == 0:
        return np.eye(n, dtype=complex)
    P = np.eye(n, dtype=complex) - B @ B.conjugate().T
    vals, vecs = np.linalg.eigh(P)
    keep = vals > 0.5
    C = np.asarray(vecs[:, keep], dtype=complex)
    return _orthonormalize_columns(C, tol=tol)

def project_kernel_to_complement(kernel: np.ndarray, basis_vectors: np.ndarray, *, projection_tol: float = 1e-12) -> np.ndarray:
    """Project a full-space Hermitian kernel into the complement of basis_vectors."""
    H = np.asarray(kernel, dtype=complex)
    C = complement_from_basis(basis_vectors, tol=projection_tol)
    return C.conjugate().T @ H @ C



# ---------------------------------------------------------------------------
# Operator construction / eig diagnosis
# ---------------------------------------------------------------------------


def build_hermitian_kernel(channel_kernel: ChannelKernel, *, use_hermitian_part: bool = True) -> Tuple[np.ndarray, float]:
    K = np.asarray(channel_kernel.matrix, dtype=complex)
    resid = channel_kernel.hermitian_residual()
    if use_hermitian_part:
        return 0.5 * (K + K.conjugate().T), resid
    return K, resid


def build_instability_operator(
    hermitian_kernel: np.ndarray,
    bubble_weights: BubbleWeights,
    *,
    sign: int,
    channel_type: str,
    basis_vectors: Optional[np.ndarray] = None,
    projection_tol: float = 1e-12,
) -> np.ndarray:
    if sign not in (+1, -1):
        raise ValueError("sign must be either +1 or -1.")

    H = np.asarray(hermitian_kernel, dtype=complex)
    weights = np.asarray(bubble_weights.weights, dtype=float)
    if H.shape[0] != H.shape[1] or H.shape[0] != weights.shape[0]:
        raise ValueError("Kernel dimension and bubble weight dimension are inconsistent.")

    if channel_type == "pp":
        sqrtw = np.sqrt(np.abs(weights))
    elif channel_type == "ph":
        sqrtw = np.sqrt(weights)
    else:
        raise ValueError("channel_type must be 'pp' or 'ph'.")

    Bhalf = np.diag(sqrtw.astype(complex))
    M = int(sign) * (Bhalf @ H @ Bhalf)

    if basis_vectors is None:
        return M
    C = complement_from_basis(basis_vectors, tol=projection_tol)
    return C.conjugate().T @ M @ C


def _eigh_desc(operator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = np.asarray(operator, dtype=complex)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(-vals)
    return np.asarray(vals[order], dtype=float), np.asarray(vecs[:, order], dtype=complex)


def _leading_eig(operator: np.ndarray, *, store_all: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    vals, vecs = _eigh_desc(operator)
    leading_eval = float(vals[0]) if vals.size else 0.0
    leading_evec = np.asarray(vecs[:, 0], dtype=complex) if vecs.size else np.zeros((operator.shape[0],), dtype=complex)
    all_vals = np.asarray(vals, dtype=float) if store_all else None
    return leading_eval, leading_evec, all_vals


def _degenerate_groups_from_sorted_evals(
    evals: np.ndarray,
    *,
    top_n: int,
    atol: float,
    rtol: float,
) -> List[np.ndarray]:
    vals = np.asarray(evals, dtype=float)[: int(top_n)]
    if vals.size == 0:
        return []
    groups: List[List[int]] = []
    current: List[int] = [0]
    for i in range(1, vals.size):
        ref = max(abs(vals[i - 1]), abs(vals[i]), 1.0)
        close = abs(vals[i] - vals[i - 1]) <= (atol + rtol * ref)
        if close:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return [np.asarray(g, dtype=int) for g in groups]


# ---------------------------------------------------------------------------
# Physics-aware classification helpers
# ---------------------------------------------------------------------------


def classify_single_mode_candidate(
    channel_name: str,
    channel_type: str,
    spin_structure: str,
    Q: Sequence[float],
    leading_evec: np.ndarray,
    *,
    q0_tol: float = 1e-10,
) -> Tuple[str, Tuple[str, ...]]:
    q0 = is_q0(Q, tol=q0_tol)
    const_like = _is_constant_like(leading_evec)
    notes: List[str] = [f"constant_like={const_like}", f"q0={q0}"]

    if channel_type == "ph" and spin_structure == "spin":
        if q0:
            return ("FM" if const_like else "spin_nematic_like_or_q0_spin_bond_like", tuple(notes))
        return ("SDW" if const_like else "sBO", tuple(notes))

    if channel_type == "ph" and spin_structure == "charge":
        if q0:
            return ("uniform_charge_or_landau" if const_like else "PI_or_charge_nematic", tuple(notes))
        return ("CDW" if const_like else "cBO", tuple(notes))

    if channel_type == "pp" and spin_structure == "singlet":
        return (("singlet_SC" if q0 else "PDW_singlet"), tuple(notes))

    if channel_type == "pp" and spin_structure == "triplet":
        return (("triplet_SC" if q0 else "PDW_triplet"), tuple(notes))

    return ("unknown_single_mode", tuple(notes))


def _classify_multiplet_candidate(
    channel_name: str,
    channel_type: str,
    spin_structure: str,
    Q: Sequence[float],
    dimension: int,
    *,
    q0_tol: float = 1e-10,
) -> str:
    q0 = is_q0(Q, tol=q0_tol)
    if dimension <= 1:
        return "single_mode"
    if channel_type == "ph" and spin_structure == "charge":
        return "PI_irrep" if q0 else "cBO_irrep"
    if channel_type == "ph" and spin_structure == "spin":
        return "q0_spin_irrep" if q0 else "sBO_irrep"
    if channel_type == "pp" and spin_structure == "singlet":
        return "singlet_SC_irrep" if q0 else "PDW_singlet_irrep"
    if channel_type == "pp" and spin_structure == "triplet":
        return "triplet_SC_irrep" if q0 else "PDW_triplet_irrep"
    return "unknown_irrep"


def analyze_operator_multiplets(
    *,
    operator: np.ndarray,
    channel_kernel: ChannelKernel,
    source: str,
    config: InstabilityConfig,
    projection_name: Optional[str] = None,
) -> List[MultipletCandidate]:
    """Analyze near-degenerate leading subspaces of a Hermitian matrix.

    Despite the historical name, this function is now intended to be used on
    the Hermitian channel kernel (or its projected complement-space version),
    not on the bubble-dressed operator. This makes irrep / form-factor
    detection much more robust against tiny bubble scales.
    """
    vals, vecs = _eigh_desc(operator)
    if vals.size == 0:
        return []

    channel_name = str(channel_kernel.name)
    channel_type = infer_channel_type(channel_kernel)
    spin_structure = infer_spin_structure(channel_kernel)
    q0 = is_q0(channel_kernel.Q, tol=config.q0_tol)

    leading_abs = float(abs(vals[0]))
    floor_abs = max(float(config.multiplet_eval_floor), float(config.multiplet_eval_rel_floor) * leading_abs)

    # If even the leading mode is below floor, do not report a physical multiplet.
    if leading_abs < floor_abs:
        return []

    n = min(int(config.multiplet_top_n), vals.size)
    # Only track the leading contiguous group that contains index 0.
    group = [0]
    for i in range(1, n):
        if abs(vals[i]) < floor_abs:
            break
        ref = max(abs(vals[i - 1]), abs(vals[i]), abs(vals[0]), 1.0)
        close = abs(vals[i] - vals[i - 1]) <= (config.multiplet_atol + config.multiplet_rtol * ref)
        if close:
            group.append(i)
        else:
            break

    g = np.asarray(group, dtype=int)
    g_evals = np.asarray(vals[g], dtype=float)
    g_vecs = np.asarray(vecs[:, g], dtype=complex)
    dim = int(g.size)
    eval_max_abs = float(np.max(np.abs(g_evals))) if g_evals.size else 0.0
    spread_abs = float(np.max(g_evals) - np.min(g_evals)) if g_evals.size else 0.0
    denom = max(float(np.max(np.abs(g_evals))), 1.0)
    spread_rel = float(spread_abs / denom)
    notes: List[str] = []
    degeneracy_type = "physical_irrep_like"
    physical = dim > 1

    # Special notes for projected Q=0 charge / pp-singlet cases.
    if channel_type == "ph" and spin_structure == "charge" and q0:
        if projection_name == "uniform_q0_charge":
            notes.append("uniform_q0_charge_projected")
    if channel_type == "pp" and spin_structure == "singlet" and q0 and projection_name == "local_gram_q0_pp_singlet":
        notes.append("local_gram_projection_applied")

    return [
        MultipletCandidate(
            channel_name=channel_name,
            channel_type=channel_type,
            spin_structure=spin_structure,
            Q=np.asarray(channel_kernel.Q, dtype=float),
            source=source,
            indices=g,
            evals=g_evals,
            basis_vectors=g_vecs,
            dimension=dim,
            spread_abs=spread_abs,
            spread_rel=spread_rel,
            eval_max_abs=eval_max_abs,
            degeneracy_type=degeneracy_type,
            physical_irrep_like=physical,
            notes=tuple(notes + [f"floor_abs={floor_abs:.3e}"]),
        )
    ]


def _pick_leading_physical_multiplet(candidates: Sequence[MultipletCandidate]) -> Optional[MultipletCandidate]:
    if not candidates:
        return None
    for cand in candidates:
        if cand.indices.size > 0 and int(cand.indices[0]) == 0 and cand.physical_irrep_like and cand.dimension > 1:
            return cand
    return None


def build_complex_order_candidates(
    multiplet: MultipletCandidate,
    *,
    max_pairs: int = 1,
    q0_tol: float = 1e-10,
) -> List[ComplexOrderCandidate]:
    if multiplet.dimension < 2:
        return []
    B = np.asarray(multiplet.basis_vectors, dtype=complex)
    out: List[ComplexOrderCandidate] = []
    npairs = 0
    for i in range(multiplet.dimension):
        for j in range(i + 1, multiplet.dimension):
            if npairs >= max_pairs:
                return out
            f1 = B[:, i]
            f2 = B[:, j]
            for combo_type, vec in (("plus_i", f1 + 1j * f2), ("minus_i", f1 - 1j * f2)):
                if multiplet.channel_type == "ph" and multiplet.spin_structure == "charge":
                    label = "charge_loop_current_possible"
                elif multiplet.channel_type == "ph" and multiplet.spin_structure == "spin":
                    label = "spin_loop_current_possible"
                elif multiplet.channel_type == "pp":
                    label = "chiral_SC_possible"
                else:
                    label = "complex_order_possible"
                out.append(
                    ComplexOrderCandidate(
                        parent_source=multiplet.source,
                        channel_name=multiplet.channel_name,
                        channel_type=multiplet.channel_type,
                        spin_structure=multiplet.spin_structure,
                        Q=np.asarray(multiplet.Q, dtype=float),
                        combo_type=combo_type,
                        basis_indices=(int(multiplet.indices[i]), int(multiplet.indices[j])),
                        vector_complex=np.asarray(vec / max(np.linalg.norm(vec), 1e-30), dtype=complex),
                        candidate_label=label,
                        tr_breaking_possible=True,
                        notes=(f"built_from_dim={multiplet.dimension}", f"q0={is_q0(multiplet.Q, tol=q0_tol)}"),
                    )
                )
            npairs += 1
    return out


# ---------------------------------------------------------------------------
# High-level diagnosis entry points
# ---------------------------------------------------------------------------


def diagnose_channel_instability(
    channel_kernel: ChannelKernel,
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    config: Optional[InstabilityConfig] = None,
) -> InstabilityResult:
    if config is None:
        config = InstabilityConfig()

    channel_type = infer_channel_type(channel_kernel)
    spin_structure = infer_spin_structure(channel_kernel)

    measure_mode, measure_eps, measure_norm, measure_notes = _resolve_measure_config(transfer_context, config)

    notes: List[str] = list(measure_notes)
    if channel_type == "ph":
        bubble = build_ph_bubble_weights(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=config.bubble_floor,
            mode=config.ph_bubble_mode,
            patch_measure_mode=measure_mode,
            patch_measure_soft_vf_eps=measure_eps,
            patch_measure_normalize_mean=measure_norm,
        )
        sign_used: Optional[int] = int(config.ph_sign)
    elif channel_type == "pp":
        bubble = build_pp_bubble_weights(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=None,
            patch_measure_mode=measure_mode,
            patch_measure_soft_vf_eps=measure_eps,
            patch_measure_normalize_mean=measure_norm,
        )
        sign_used = None if config.pp_sign is None else int(config.pp_sign)
        notes.append("pp_operator_uses_abs_bubble_weights")
    else:
        raise ValueError(f"Unsupported channel_type={channel_type!r}")

    H, herm_resid = build_hermitian_kernel(channel_kernel, use_hermitian_part=config.use_hermitian_part)
    notes.extend(bubble.notes)

    if channel_type == "pp" and sign_used is None:
        raise ValueError("pp_sign is not set. Pass InstabilityConfig(pp_sign=...).")
    assert sign_used is not None

    op_unproj = build_instability_operator(
        H,
        bubble,
        sign=sign_used,
        channel_type=channel_type,
        basis_vectors=None,
        projection_tol=config.projection_tol,
    )
    eval_unproj, evec_unproj, all_unproj = _leading_eig(op_unproj, store_all=config.store_all_evals)

    projection_name: Optional[str] = None
    projection_basis: Optional[np.ndarray] = None
    projected_is_default = False

    kernel_name_lower = str(channel_kernel.name).lower()
    already_reduced_upstream = "reduced" in kernel_name_lower

    if channel_type == "ph" and spin_structure == "charge" and is_q0(channel_kernel.Q, tol=config.q0_tol):
        if config.project_ph_charge_q0_uniform and not already_reduced_upstream:
            projection_name = "uniform_q0_charge"
            projection_basis = build_uniform_projection_basis(channel_kernel.Npatch)
            projected_is_default = True
        elif config.project_ph_charge_q0_uniform and already_reduced_upstream:
            notes.append("ph_charge_q0_kernel_already_reduced_upstream; skipped_extra_projection")
    elif channel_type == "pp" and spin_structure == "singlet" and is_q0(channel_kernel.Q, tol=config.q0_tol):
        if config.report_pp_singlet_q0_local_gram_both or config.project_pp_singlet_q0_local_gram_default:
            projection_name = "local_gram_q0_pp_singlet"
            projection_basis = build_local_gram_projection_basis(patchsets)
            projected_is_default = bool(config.project_pp_singlet_q0_local_gram_default)

    op_proj = None
    eval_proj = None
    evec_proj_full = None
    all_proj = None
    projection_rank = 0
    projection_basis_store = None

    if projection_basis is not None:
        projection_basis_store = _orthonormalize_columns(projection_basis, tol=config.projection_tol)
        projection_rank = int(projection_basis_store.shape[1])
        C = complement_from_basis(projection_basis_store, tol=config.projection_tol)
        op_proj = build_instability_operator(
            H,
            bubble,
            sign=sign_used,
            channel_type=channel_type,
            basis_vectors=projection_basis_store,
            projection_tol=config.projection_tol,
        )
        eval_proj, evec_proj_red, all_proj = _leading_eig(op_proj, store_all=config.store_all_evals)
        evec_proj_full = C @ evec_proj_red
        notes.append(f"projection_applied={projection_name}, rank={projection_rank}")

    score = eval_proj if (projected_is_default and eval_proj is not None) else eval_unproj
    leading_eval = float(score)
    leading_evec = evec_proj_full if (projected_is_default and evec_proj_full is not None) else evec_unproj
    leading_source = projection_name if (projected_is_default and projection_name is not None) else "unprojected"

    single_label, single_notes = classify_single_mode_candidate(
        str(channel_kernel.name), channel_type, spin_structure, channel_kernel.Q, leading_evec, q0_tol=config.q0_tol
    )

    multiplet_candidates: List[MultipletCandidate] = []
    leading_multiplet_candidate: Optional[MultipletCandidate] = None
    complex_candidates: List[ComplexOrderCandidate] = []

    if config.analyze_multiplets:
        # Detect irrep / degeneracy on the Hermitian channel kernel itself, not
        # on the bubble-dressed operator. This is much more robust against tiny
        # bubble scales and preserves the form-factor structure.
        multiplet_candidates.extend(
            analyze_operator_multiplets(
                operator=H,
                channel_kernel=channel_kernel,
                source="unprojected",
                config=config,
                projection_name=None,
            )
        )
        if projection_basis_store is not None and projection_name is not None:
            H_proj = project_kernel_to_complement(H, projection_basis_store, projection_tol=config.projection_tol)
            multiplet_candidates.extend(
                analyze_operator_multiplets(
                    operator=H_proj,
                    channel_kernel=channel_kernel,
                    source=str(projection_name),
                    config=config,
                    projection_name=projection_name,
                )
            )
        leading_multiplet_candidate = _pick_leading_physical_multiplet(multiplet_candidates)
        if config.build_complex_candidates and leading_multiplet_candidate is not None:
            complex_candidates = build_complex_order_candidates(
                leading_multiplet_candidate,
                max_pairs=config.max_complex_pairs_per_multiplet,
                q0_tol=config.q0_tol,
            )

    return InstabilityResult(
        channel_name=str(channel_kernel.name),
        channel_type=channel_type,
        spin_structure=spin_structure,
        Q=np.asarray(channel_kernel.Q, dtype=float),
        sign_used=sign_used,
        hermitian_residual=float(herm_resid),
        bubble=bubble,
        score=float(score),
        leading_eval=float(leading_eval),
        leading_evec=np.asarray(leading_evec, dtype=complex),
        score_unprojected=float(eval_unproj),
        leading_eval_unprojected=float(eval_unproj),
        leading_evec_unprojected=np.asarray(evec_unproj, dtype=complex),
        score_projected=None if eval_proj is None else float(eval_proj),
        leading_eval_projected=None if eval_proj is None else float(eval_proj),
        leading_evec_projected=None if evec_proj_full is None else np.asarray(evec_proj_full, dtype=complex),
        projection_name=projection_name,
        projection_rank=projection_rank,
        projection_basis_vectors=projection_basis_store,
        operator_unprojected=op_unproj if config.store_operator_matrices else None,
        operator_projected=op_proj if (config.store_operator_matrices and op_proj is not None) else None,
        hermitian_matrix=H if config.store_operator_matrices else None,
        all_evals_unprojected=all_unproj,
        all_evals_projected=all_proj,
        single_mode_label=single_label,
        single_mode_notes=tuple(single_notes),
        leading_mode_info=LeadingModeInfo(
            score=float(score),
            eval=float(leading_eval),
            evec=np.asarray(leading_evec, dtype=complex),
            source=str(leading_source),
            projection_name=projection_name if projected_is_default else None,
        ),
        multiplet_candidates=tuple(multiplet_candidates),
        leading_multiplet_candidate=leading_multiplet_candidate,
        complex_order_candidates=tuple(complex_candidates),
        notes=tuple(notes),
    )


def diagnose_kernel_collection(
    kernels: Union[Mapping[str, ChannelKernel], Sequence[ChannelKernel]],
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    config: Optional[InstabilityConfig] = None,
) -> Dict[str, InstabilityResult]:
    if isinstance(kernels, Mapping):
        items = list(kernels.items())
    else:
        items = [(str(k.name), k) for k in kernels]

    out: Dict[str, InstabilityResult] = {}
    for key, kernel in items:
        out[str(key)] = diagnose_channel_instability(
            kernel,
            patchsets,
            transfer_context,
            flow_config,
            config=config,
        )
    return out


__all__ = [
    "BubbleWeights",
    "LeadingModeInfo",
    "MultipletCandidate",
    "ComplexOrderCandidate",
    "InstabilityConfig",
    "InstabilityResult",
    "build_ph_bubble_weights",
    "build_ph_bubble_weights_internal_cache",
    "build_ph_bubble_weights_patchrep",
    "build_pp_bubble_weights",
    "build_uniform_projection_basis",
    "build_local_gram_projection_basis",
    "project_kernel_to_complement",
    "build_instability_operator",
    "build_hermitian_kernel",
    "classify_single_mode_candidate",
    "analyze_operator_multiplets",
    "build_complex_order_candidates",
    "diagnose_channel_instability",
    "diagnose_kernel_collection",
    "infer_channel_type",
    "infer_spin_structure",
    "is_q0",
]
