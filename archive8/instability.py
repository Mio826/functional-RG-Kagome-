from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

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
    patchset_for_spin,
)


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


@dataclass
class InstabilityConfig:
    """Configuration for bubble-weighted instability diagnosis.

    Notes
    -----
    - ``ph_sign`` is fixed by convention for the current code base.
    - ``pp_sign`` defaults to ``+1`` in the repaired pp diagnosis, together with
      ``abs`` bubble weights, to avoid re-promoting the local repulsive
      Q=0 pp-singlet mode as a fake pairing instability.
    - ``ph_bubble_mode='patchrep'`` is the new default. It computes a genuine
      patch-resolved ph bubble using the partner map and patch energies, instead
      of reusing the flow's internal-cache weights, which were effectively only
      a support mask with constant active-patch weight.
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
    notes = []
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
            nneg = int(np.count_nonzero(neg_mask))
            notes.append(f"bubble_weights_clipped_count={nneg}")
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
) -> BubbleWeights:
    """Legacy ph bubble path: directly reuse flow internal-cache weights."""
    Q = canonicalize_q_for_patchsets(patchsets, channel_kernel.Q)
    partner, residual = _ph_shift_from_context(patchsets, transfer_context, Q)
    legacy = build_ph_internal_cache_vec(
        patchsets,
        flow_config,
        shift_cache={("up", "dn"): (np.asarray(partner, dtype=int), np.asarray(residual, dtype=float))},
    )[("up", "dn")]
    weights, notes = _sanitize_bubble_weights(np.asarray(legacy["weights"], dtype=complex), floor=bubble_floor)
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
) -> BubbleWeights:
    """Patch-representative ph bubble.

    For each patch representative p we explicitly evaluate
        chi0(Q,p) ~ bubble_dot_ph(eps_p, eps_{p+Q})
    using the same partner map / Q semantics as the flow, but *not* reusing the
    flow's coarse internal-cache weight tensor. This preserves patch-resolved
    energy dependence and avoids collapsing ph diagnosis into a support mask.
    """
    Q = canonicalize_q_for_patchsets(patchsets, channel_kernel.Q)
    partner, residual = _ph_shift_from_context(patchsets, transfer_context, Q)
    ps_src = patchset_for_spin(patchsets, "up")
    ps_tgt = patchset_for_spin(patchsets, "dn")
    eps_src = _patch_energies(ps_src)
    eps_tgt = _patch_energies(ps_tgt)

    weights = np.zeros(ps_src.Npatch, dtype=complex)
    valid = np.asarray(partner, dtype=int) >= 0
    if np.any(valid):
        vals = [
            bubble_dot_ph(float(eps_src[i]), float(eps_tgt[int(partner[i])]), flow_config)
            for i in np.flatnonzero(valid)
        ]
        weights[np.flatnonzero(valid)] = np.asarray(vals, dtype=complex)

    weights, notes = _sanitize_bubble_weights(weights, floor=bubble_floor)
    notes = tuple(list(notes) + ["ph_bubble_mode=patchrep"])
    return BubbleWeights(
        channel_type="ph",
        Q=np.asarray(Q, dtype=float),
        weights=weights,
        partner_patches=np.asarray(partner, dtype=int),
        residuals=np.asarray(residual, dtype=float),
        temperature=float(flow_config.temperature),
        source="patchrep:bubble_dot_ph(eps_p,eps_p+Q)",
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
) -> BubbleWeights:
    if mode == "patchrep":
        return build_ph_bubble_weights_patchrep(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=bubble_floor,
        )
    if mode == "internal_cache":
        return build_ph_bubble_weights_internal_cache(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=bubble_floor,
        )
    raise ValueError("ph bubble mode must be 'patchrep' or 'internal_cache'.")


def build_pp_bubble_weights(
    channel_kernel: ChannelKernel,
    patchsets: PatchSetMap,
    transfer_context: Mapping[str, Any],
    flow_config: FlowConfig,
    *,
    bubble_floor: Optional[float] = None,
) -> BubbleWeights:
    """Build patch-diagonal pp bubble weights using the same helper as the flow.

    For pp we intentionally keep the raw sign information and do *not* default to
    clipping negative values. The repaired pp operator uses abs(weights) later.
    """
    Q = canonicalize_q_for_patchsets(patchsets, channel_kernel.Q)
    partner, residual = _pp_shift_from_context(patchsets, transfer_context, Q)
    legacy = build_pp_internal_cache_vec(
        patchsets,
        flow_config,
        shift_cache={("up", "dn"): (np.asarray(partner, dtype=int), np.asarray(residual, dtype=float))},
    )[("up", "dn")]
    weights, notes = _sanitize_bubble_weights(np.asarray(legacy["weights"], dtype=complex), floor=bubble_floor)
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
    if getattr(ps, "Npatch", None) is None:
        raise ValueError("Patch set does not expose Npatch.")
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


def _leading_eig(operator: np.ndarray, *, store_all: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    H = np.asarray(operator, dtype=complex)
    vals, vecs = np.linalg.eigh(H)
    idx = int(np.argmax(vals))
    leading_eval = float(np.real(vals[idx]))
    leading_evec = np.asarray(vecs[:, idx], dtype=complex)
    all_vals = np.asarray(vals, dtype=float) if store_all else None
    return leading_eval, leading_evec, all_vals


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

    notes = []
    if channel_type == "ph":
        bubble = build_ph_bubble_weights(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=config.bubble_floor,
            mode=config.ph_bubble_mode,
        )
        sign_used: Optional[int] = int(config.ph_sign)
    elif channel_type == "pp":
        bubble = build_pp_bubble_weights(
            channel_kernel,
            patchsets,
            transfer_context,
            flow_config,
            bubble_floor=None,
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
    leading_eval = score
    leading_evec = evec_proj_full if (projected_is_default and evec_proj_full is not None) else evec_unproj

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
    "InstabilityConfig",
    "InstabilityResult",
    "build_ph_bubble_weights",
    "build_ph_bubble_weights_internal_cache",
    "build_ph_bubble_weights_patchrep",
    "build_pp_bubble_weights",
    "build_uniform_projection_basis",
    "build_local_gram_projection_basis",
    "build_instability_operator",
    "build_hermitian_kernel",
    "diagnose_channel_instability",
    "diagnose_kernel_collection",
    "infer_channel_type",
    "infer_spin_structure",
    "is_q0",
]
