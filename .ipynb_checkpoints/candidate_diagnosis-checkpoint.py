from __future__ import annotations

"""Candidate-based diagnosis helpers for the kagome fRG pipeline.

This module is intentionally separated from ``instability.py``.
Its job is *not* to build the bubble-dressed instability operator itself,
but to construct physically motivated candidate templates and compare them to
instability eigenmodes in a metric-consistent way.

Current scope
-------------
This first version only supports **particle-hole real orders**:

* Q = 0 onsite ferromagnetism-like constant mode
* Q = 0 real bond A1 / E-type modes
* Q = M onsite density-wave templates (CDW/SDW momentum structure)
* Q = M real bond-wave templates (CBO/SBO momentum structure)

Not included yet
----------------
* particle-particle real orders
* particle-hole imaginary / loop-current orders
* particle-particle imaginary / chiral orders

Design principle
----------------
A candidate is *not* stored as a fixed form-factor array detached from the
flow context. Instead we store a compact specification and instantiate the
candidate against a specific ``model`` + ``patchset`` (+ ``Q`` when needed).
This keeps the candidate representation aligned with:

* the same patch ordering used by the flow / diagnosis,
* the same Bloch basis and gauge conventions,
* the same momentum representative conventions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from frg_kernel import canonicalize_q_for_patchsets
except Exception as exc:  # pragma: no cover - import-time fallback
    raise ImportError(
        "candidate_diagnosis.py requires frg_kernel.canonicalize_q_for_patchsets"
    ) from exc

try:
    from patching import exact_M6_points_1bz, canonicalize_k_to_centered_1bz
except Exception:
    # Older notebook copies may not export canonicalize_k_to_centered_1bz.
    from patching import exact_M6_points_1bz  # type: ignore

    def canonicalize_k_to_centered_1bz(model, k, *, search_range: int = 2):
        return np.asarray(k, dtype=float)


ArrayLike = Sequence[float] | np.ndarray
MetricName = Literal["euclidean", "bubble_weighted", "whitened"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    """Abstract physical candidate definition.

    Notes
    -----
    ``builder_key`` selects the concrete template builder. The builder later
    needs a model + patchset to instantiate the patch-space vector.
    """

    name: str
    family: str
    channel_type: str          # currently only "ph"
    spin_structure: str        # "charge" or "spin"
    q_type: str                # "q0", "M", or "custom"
    builder_key: str
    irrep_label: Optional[str] = None
    representation_kind: str = "unknown"
    notes: Tuple[str, ...] = ()


@dataclass
class CandidateVector:
    """Concrete candidate instantiated on one patch set."""

    spec: CandidateSpec
    Q: np.ndarray
    patch_k: np.ndarray
    vector_patch: np.ndarray
    band_index: int
    gauge_info: Dict[str, Any] = field(default_factory=dict)
    metric_ready: Dict[str, np.ndarray] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    @property
    def Npatch(self) -> int:
        return int(self.vector_patch.shape[0])

    def normalized(self) -> np.ndarray:
        return _normalize(self.vector_patch)


@dataclass
class CandidateMatch:
    candidate_name: str
    metric: str
    overlap_abs: float
    projected_score: Optional[float]
    best_phase: float
    best_shift: int
    reversed_order: bool
    conjugated: bool
    notes: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


A, B, C = 0, 1, 2


def _normalize(v: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    v = np.asarray(v, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm <= tol:
        return v.copy()
    return v / nrm


def _check_patchset(patchset: Any) -> None:
    for attr in ("patches", "patch_k", "patch_eigvec", "Npatch"):
        if not hasattr(patchset, attr):
            raise TypeError(f"patchset is missing required attribute {attr!r}")


def _resolve_Q(spec: CandidateSpec, model: Any, patchset: Any, Q: Optional[ArrayLike], m_index: int) -> np.ndarray:
    if spec.q_type == "q0":
        q = np.zeros(2, dtype=float)
    elif spec.q_type == "M":
        if Q is None:
            q = np.asarray(exact_M6_points_1bz(model)[int(m_index)], dtype=float)
        else:
            q = np.asarray(Q, dtype=float)
    elif spec.q_type == "custom":
        if Q is None:
            raise ValueError(f"Candidate {spec.name!r} requires an explicit Q.")
        q = np.asarray(Q, dtype=float)
    else:
        raise ValueError(f"Unsupported q_type={spec.q_type!r}")
    return np.asarray(canonicalize_q_for_patchsets({"up": patchset}, q), dtype=float)


def _anchor_phase(u: np.ndarray, method: str = "max_component") -> np.ndarray:
    u = np.asarray(u, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(u)
    if nrm == 0:
        raise ValueError("Encountered zero-norm eigenvector.")
    u = u / nrm
    if method == "max_component":
        idx = int(np.argmax(np.abs(u)))
        if np.abs(u[idx]) > 0:
            u = u * np.exp(-1j * np.angle(u[idx]))
    elif method == "first_component":
        if np.abs(u[0]) > 0:
            u = u * np.exp(-1j * np.angle(u[0]))
    else:
        raise ValueError("anchor_method must be 'max_component' or 'first_component'.")
    return u


def _band_eigvec_at_k(
    model: Any,
    k: ArrayLike,
    band_index: int,
    *,
    anchor_method: str = "max_component",
) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    k = canonicalize_k_to_centered_1bz(model, k)
    evals, evecs = model.eigenstate(float(k[0]), float(k[1]))
    u = np.asarray(evecs[:, int(band_index)], dtype=complex)
    return _anchor_phase(u, method=anchor_method)


def _bond_matrix_ab() -> np.ndarray:
    M = np.zeros((3, 3), dtype=complex)
    M[A, B] = 1.0
    M[B, A] = 1.0
    return M


def _bond_matrix_ac() -> np.ndarray:
    M = np.zeros((3, 3), dtype=complex)
    M[A, C] = 1.0
    M[C, A] = 1.0
    return M


def _bond_matrix_bc() -> np.ndarray:
    M = np.zeros((3, 3), dtype=complex)
    M[B, C] = 1.0
    M[C, B] = 1.0
    return M


M_AB = _bond_matrix_ab()
M_AC = _bond_matrix_ac()
M_BC = _bond_matrix_bc()


def _density_diag(vals: Sequence[float]) -> np.ndarray:
    return np.diag(np.asarray(vals, dtype=float))


D_M_A = _density_diag([1.0, -1.0, 0.0])
D_M_B = _density_diag([1.0, 1.0, -2.0]) / np.sqrt(3.0)


def _real_bond_ops_q0(model: Any, kx: float, ky: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.array([kx, ky], dtype=float)
    c1 = np.cos(np.dot(model.delta1, k))
    c2 = np.cos(np.dot(model.delta2, k))
    c3 = np.cos(np.dot(model.delta3, k))

    B1 = (-2.0 * c1) * M_AB
    B2 = (-2.0 * c2) * M_AC
    B3 = (-2.0 * c3) * M_BC
    return B1, B2, B3


def _real_bond_wave_ops_q(model: Any, kx: float, ky: float, Q: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.array([kx, ky], dtype=float)
    q = np.asarray(Q, dtype=float)

    x1 = np.cos(np.dot(model.delta1, k + 0.5 * q))
    x2 = np.cos(np.dot(model.delta2, k + 0.5 * q))
    x3 = np.cos(np.dot(model.delta3, k + 0.5 * q))

    B1 = (2.0 * x1) * M_AB
    B2 = (2.0 * x2) * M_AC
    B3 = (2.0 * x3) * M_BC
    return B1, B2, B3


def _op_q0_fm(_: Any, __: float, ___: float) -> np.ndarray:
    return np.eye(3, dtype=complex)


def _op_q0_bond_a1(model: Any, kx: float, ky: float) -> np.ndarray:
    B1, B2, B3 = _real_bond_ops_q0(model, kx, ky)
    O = B1 + B2 + B3
    return 0.5 * (O + O.conjugate().T)


def _op_q0_bond_ea(model: Any, kx: float, ky: float) -> np.ndarray:
    B1, B2, B3 = _real_bond_ops_q0(model, kx, ky)
    O = 2.0 * B1 - B2 - B3
    return 0.5 * (O + O.conjugate().T)


def _op_q0_bond_eb(model: Any, kx: float, ky: float) -> np.ndarray:
    B1, B2, B3 = _real_bond_ops_q0(model, kx, ky)
    O = B2 - B3
    return 0.5 * (O + O.conjugate().T)


def _op_qm_bond_a(model: Any, kx: float, ky: float, Q: ArrayLike) -> np.ndarray:
    B1, B2, B3 = _real_bond_wave_ops_q(model, kx, ky, Q)
    O = 2.0 * B1 - B2 - B3
    return 0.5 * (O + O.conjugate().T)


def _op_qm_bond_b(model: Any, kx: float, ky: float, Q: ArrayLike) -> np.ndarray:
    B1, B2, B3 = _real_bond_wave_ops_q(model, kx, ky, Q)
    O = B2 - B3
    return 0.5 * (O + O.conjugate().T)


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------


def ph_real_candidate_library() -> Dict[str, CandidateSpec]:
    """Return the initial particle-hole real-order candidate dictionary."""

    specs = {
        "FM_Q0": CandidateSpec(
            name="FM_Q0",
            family="FM",
            channel_type="ph",
            spin_structure="spin",
            q_type="q0",
            builder_key="q0_fm",
            irrep_label="A1",
            representation_kind="site",
            notes=("Uniform onsite spin-density template.",),
        ),
        "BOND_Q0_A1": CandidateSpec(
            name="BOND_Q0_A1",
            family="BOND_Q0",
            channel_type="ph",
            spin_structure="charge",
            q_type="q0",
            builder_key="q0_bond_a1",
            irrep_label="A1",
            representation_kind="bond_real",
            notes=("Uniform real bond A1 template.",),
        ),
        "BOND_Q0_Ea": CandidateSpec(
            name="BOND_Q0_Ea",
            family="BOND_Q0",
            channel_type="ph",
            spin_structure="charge",
            q_type="q0",
            builder_key="q0_bond_ea",
            irrep_label="E",
            representation_kind="bond_real",
            notes=("Real bond E-type partner a.",),
        ),
        "BOND_Q0_Eb": CandidateSpec(
            name="BOND_Q0_Eb",
            family="BOND_Q0",
            channel_type="ph",
            spin_structure="charge",
            q_type="q0",
            builder_key="q0_bond_eb",
            irrep_label="E",
            representation_kind="bond_real",
            notes=("Real bond E-type partner b.",),
        ),
        "CDW_M_A": CandidateSpec(
            name="CDW_M_A",
            family="CDW_M",
            channel_type="ph",
            spin_structure="charge",
            q_type="M",
            builder_key="qm_density_a",
            irrep_label=None,
            representation_kind="site",
            notes=("Finite-Q onsite density-wave template, basis A.",),
        ),
        "CDW_M_B": CandidateSpec(
            name="CDW_M_B",
            family="CDW_M",
            channel_type="ph",
            spin_structure="charge",
            q_type="M",
            builder_key="qm_density_b",
            irrep_label=None,
            representation_kind="site",
            notes=("Finite-Q onsite density-wave template, basis B.",),
        ),
        "SDW_M_A": CandidateSpec(
            name="SDW_M_A",
            family="SDW_M",
            channel_type="ph",
            spin_structure="spin",
            q_type="M",
            builder_key="qm_density_a",
            irrep_label=None,
            representation_kind="site",
            notes=("Finite-Q onsite spin-density-wave momentum template, basis A.",),
        ),
        "SDW_M_B": CandidateSpec(
            name="SDW_M_B",
            family="SDW_M",
            channel_type="ph",
            spin_structure="spin",
            q_type="M",
            builder_key="qm_density_b",
            irrep_label=None,
            representation_kind="site",
            notes=("Finite-Q onsite spin-density-wave momentum template, basis B.",),
        ),
        "CBO_M_A": CandidateSpec(
            name="CBO_M_A",
            family="CBO_M",
            channel_type="ph",
            spin_structure="charge",
            q_type="M",
            builder_key="qm_bond_a",
            irrep_label=None,
            representation_kind="bond_real",
            notes=("Finite-Q real bond-wave template, basis A.",),
        ),
        "CBO_M_B": CandidateSpec(
            name="CBO_M_B",
            family="CBO_M",
            channel_type="ph",
            spin_structure="charge",
            q_type="M",
            builder_key="qm_bond_b",
            irrep_label=None,
            representation_kind="bond_real",
            notes=("Finite-Q real bond-wave template, basis B.",),
        ),
        "SBO_M_A": CandidateSpec(
            name="SBO_M_A",
            family="SBO_M",
            channel_type="ph",
            spin_structure="spin",
            q_type="M",
            builder_key="qm_bond_a",
            irrep_label=None,
            representation_kind="bond_real",
            notes=("Finite-Q spin-bond-wave momentum template, basis A.",),
        ),
        "SBO_M_B": CandidateSpec(
            name="SBO_M_B",
            family="SBO_M",
            channel_type="ph",
            spin_structure="spin",
            q_type="M",
            builder_key="qm_bond_b",
            irrep_label=None,
            representation_kind="bond_real",
            notes=("Finite-Q spin-bond-wave momentum template, basis B.",),
        ),
    }
    return specs


# ---------------------------------------------------------------------------
# Candidate builders
# ---------------------------------------------------------------------------


def _build_q0_candidate(
    op_fn,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
) -> np.ndarray:
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)

    for ip, (k, u) in enumerate(zip(patch_k, patch_eigvec)):
        O = op_fn(model, float(k[0]), float(k[1]))
        out[ip] = np.vdot(u, O @ u)
    return out


def _build_qm_candidate_matrix(
    M: np.ndarray,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: np.ndarray,
    anchor_method: str,
    use_patchset_eigvec_at_k: bool,
) -> np.ndarray:
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)

    for ip, k in enumerate(patch_k):
        uk = patch_eigvec[ip] if use_patchset_eigvec_at_k else _band_eigvec_at_k(model, k, band_index, anchor_method=anchor_method)
        uq = _band_eigvec_at_k(model, k + Q, band_index, anchor_method=anchor_method)
        out[ip] = np.vdot(uq, M @ uk)
    return out


def _build_qm_candidate_operator(
    op_fn,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: np.ndarray,
    anchor_method: str,
    use_patchset_eigvec_at_k: bool,
) -> np.ndarray:
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)

    for ip, k in enumerate(patch_k):
        uk = patch_eigvec[ip] if use_patchset_eigvec_at_k else _band_eigvec_at_k(model, k, band_index, anchor_method=anchor_method)
        uq = _band_eigvec_at_k(model, k + Q, band_index, anchor_method=anchor_method)
        O = op_fn(model, float(k[0]), float(k[1]), Q)
        out[ip] = np.vdot(uq, O @ uk)
    return out


def build_ph_real_candidate(
    spec: CandidateSpec,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: Optional[ArrayLike] = None,
    m_index: int = 0,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
) -> CandidateVector:
    """Instantiate a particle-hole real-order candidate on one patch set.

    Parameters
    ----------
    spec
        Candidate specification from :func:`ph_real_candidate_library`.
    model, patchset, band_index
        Must match the FRG diagnosis context you want to compare against.
    Q, m_index
        For finite-Q templates. If ``spec.q_type == 'M'`` and ``Q is None``,
        the geometric M point with index ``m_index`` is used.
    anchor_method
        Local phase-fixing rule for finite-Q shifted Bloch states.
    use_patchset_eigvec_at_k
        If True, the unshifted leg uses the patchset's gauge-fixed eigenvectors.
        This is recommended to stay aligned with the flow / diagnosis patch basis.
    """
    _check_patchset(patchset)
    if spec.channel_type != "ph":
        raise ValueError("This first module version only supports particle-hole candidates.")

    q_vec = _resolve_Q(spec, model, patchset, Q, m_index)

    if spec.builder_key == "q0_fm":
        vec = np.ones(int(patchset.Npatch), dtype=complex)
    elif spec.builder_key == "q0_bond_a1":
        vec = _build_q0_candidate(_op_q0_bond_a1, model=model, patchset=patchset, band_index=band_index)
    elif spec.builder_key == "q0_bond_ea":
        vec = _build_q0_candidate(_op_q0_bond_ea, model=model, patchset=patchset, band_index=band_index)
    elif spec.builder_key == "q0_bond_eb":
        vec = _build_q0_candidate(_op_q0_bond_eb, model=model, patchset=patchset, band_index=band_index)
    elif spec.builder_key == "qm_density_a":
        vec = _build_qm_candidate_matrix(
            D_M_A,
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=q_vec,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
    elif spec.builder_key == "qm_density_b":
        vec = _build_qm_candidate_matrix(
            D_M_B,
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=q_vec,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
    elif spec.builder_key == "qm_bond_a":
        vec = _build_qm_candidate_operator(
            _op_qm_bond_a,
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=q_vec,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
    elif spec.builder_key == "qm_bond_b":
        vec = _build_qm_candidate_operator(
            _op_qm_bond_b,
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=q_vec,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
    else:
        raise ValueError(f"Unknown builder_key={spec.builder_key!r}")

    return CandidateVector(
        spec=spec,
        Q=np.asarray(q_vec, dtype=float),
        patch_k=np.asarray(patchset.patch_k, dtype=float),
        vector_patch=np.asarray(vec, dtype=complex),
        band_index=int(band_index),
        gauge_info={
            "anchor_method": str(anchor_method),
            "use_patchset_eigvec_at_k": bool(use_patchset_eigvec_at_k),
        },
        notes=spec.notes,
    )


# ---------------------------------------------------------------------------
# Metric alignment
# ---------------------------------------------------------------------------


def metric_vector_from_candidate(
    cand: CandidateVector,
    *,
    metric: MetricName = "euclidean",
    bubble_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the candidate vector represented in a chosen metric space.

    Metrics
    -------
    euclidean
        No weighting.
    bubble_weighted
        Return W * T.
    whitened
        Return sqrt(W) * T.
    """
    v = np.asarray(cand.vector_patch, dtype=complex).reshape(-1)
    if metric == "euclidean":
        return v.copy()

    if bubble_weights is None:
        raise ValueError(f"metric={metric!r} requires bubble_weights.")
    w = np.asarray(bubble_weights, dtype=float).reshape(-1)
    if w.shape[0] != v.shape[0]:
        raise ValueError("bubble_weights length must match candidate vector length.")

    if metric == "bubble_weighted":
        return w * v
    if metric == "whitened":
        return np.sqrt(np.clip(w, 0.0, None)) * v
    raise ValueError(f"Unsupported metric={metric!r}")


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _iter_relabelings(v: np.ndarray) -> Iterable[Tuple[np.ndarray, int, bool]]:
    """Yield all cyclic shifts and reversed cyclic shifts."""
    v = np.asarray(v, dtype=complex).reshape(-1)
    N = v.size
    for shift in range(N):
        yield np.roll(v, shift), shift, False
    vr = v[::-1]
    for shift in range(N):
        yield np.roll(vr, shift), shift, True


def best_overlap_against_vector(
    cand_vec: np.ndarray,
    target_vec: np.ndarray,
    *,
    allow_shift: bool = True,
    allow_reverse: bool = True,
    allow_conjugation: bool = True,
) -> Tuple[float, float, int, bool, bool]:
    """Return the best absolute overlap after simple relabelings.

    Returns
    -------
    overlap_abs, best_phase, best_shift, reversed_order, conjugated
    """
    t = _normalize(target_vec)
    candidates: List[Tuple[np.ndarray, int, bool, bool]] = []

    base = np.asarray(cand_vec, dtype=complex).reshape(-1)
    if allow_shift:
        for vv, shift, rev in _iter_relabelings(base):
            if rev and not allow_reverse:
                continue
            candidates.append((vv, shift, rev, False))
    else:
        candidates.append((base, 0, False, False))

    if allow_conjugation:
        basec = np.conjugate(base)
        if allow_shift:
            for vv, shift, rev in _iter_relabelings(basec):
                if rev and not allow_reverse:
                    continue
                candidates.append((vv, shift, rev, True))
        else:
            candidates.append((basec, 0, False, True))

    best = (-1.0, 0.0, 0, False, False)
    for vv, shift, rev, conj_flag in candidates:
        v = _normalize(vv)
        z = np.vdot(v, t)
        val = float(np.abs(z))
        phase = float(np.angle(z)) if np.abs(z) > 1e-30 else 0.0
        if val > best[0]:
            best = (val, phase, int(shift), bool(rev), bool(conj_flag))
    return best


def candidate_projected_score(
    cand_metric_vec: np.ndarray,
    operator_matrix: np.ndarray,
) -> float:
    v = _normalize(cand_metric_vec)
    M = np.asarray(operator_matrix, dtype=complex)
    if M.shape[0] != v.shape[0] or M.shape[1] != v.shape[0]:
        raise ValueError("operator_matrix shape incompatible with candidate vector length.")
    val = np.vdot(v, M @ v)
    val = np.real_if_close(val, tol=1000)
    return float(np.real(val))


def compare_candidate_to_instability(
    cand: CandidateVector,
    *,
    leading_evec: np.ndarray,
    operator_matrix: Optional[np.ndarray] = None,
    metric: MetricName = "euclidean",
    bubble_weights: Optional[np.ndarray] = None,
    allow_shift: bool = True,
    allow_reverse: bool = True,
    allow_conjugation: bool = True,
) -> CandidateMatch:
    """Compare one candidate to one instability eigenvector.

    Parameters
    ----------
    cand
        Candidate instantiated on the same patch set as the instability result.
    leading_evec
        Usually the leading instability eigenvector from instability.py.
    operator_matrix
        If supplied, ``<T|M|T>`` is also reported in the same metric space.
    metric, bubble_weights
        Must match the metric convention used for the eigenvector you want to
        compare against.
    """
    cvec = metric_vector_from_candidate(cand, metric=metric, bubble_weights=bubble_weights)
    overlap_abs, phase, shift, rev, conj_flag = best_overlap_against_vector(
        cvec,
        leading_evec,
        allow_shift=allow_shift,
        allow_reverse=allow_reverse,
        allow_conjugation=allow_conjugation,
    )
    proj = None
    if operator_matrix is not None:
        proj = candidate_projected_score(cvec, operator_matrix)

    return CandidateMatch(
        candidate_name=cand.spec.name,
        metric=str(metric),
        overlap_abs=float(overlap_abs),
        projected_score=proj,
        best_phase=float(phase),
        best_shift=int(shift),
        reversed_order=bool(rev),
        conjugated=bool(conj_flag),
        notes=cand.notes,
    )


# ---------------------------------------------------------------------------
# Convenience batch API
# ---------------------------------------------------------------------------


def build_default_ph_real_candidates(
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: Optional[ArrayLike] = None,
    m_index: int = 0,
    include: Optional[Iterable[str]] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
) -> Dict[str, CandidateVector]:
    specs = ph_real_candidate_library()
    names = list(specs.keys()) if include is None else [str(x) for x in include]
    out: Dict[str, CandidateVector] = {}
    for name in names:
        if name not in specs:
            raise KeyError(f"Unknown candidate name {name!r}")
        out[name] = build_ph_real_candidate(
            specs[name],
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=Q,
            m_index=m_index,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
    return out


def compare_default_ph_real_candidates(
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    leading_evec: np.ndarray,
    Q: Optional[ArrayLike] = None,
    m_index: int = 0,
    operator_matrix: Optional[np.ndarray] = None,
    metric: MetricName = "euclidean",
    bubble_weights: Optional[np.ndarray] = None,
    include: Optional[Iterable[str]] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
    allow_shift: bool = True,
    allow_reverse: bool = True,
    allow_conjugation: bool = True,
) -> List[CandidateMatch]:
    candidates = build_default_ph_real_candidates(
        model=model,
        patchset=patchset,
        band_index=band_index,
        Q=Q,
        m_index=m_index,
        include=include,
        anchor_method=anchor_method,
        use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
    )
    matches = []
    for cand in candidates.values():
        matches.append(
            compare_candidate_to_instability(
                cand,
                leading_evec=leading_evec,
                operator_matrix=operator_matrix,
                metric=metric,
                bubble_weights=bubble_weights,
                allow_shift=allow_shift,
                allow_reverse=allow_reverse,
                allow_conjugation=allow_conjugation,
            )
        )
    return sorted(matches, key=lambda x: x.overlap_abs, reverse=True)


__all__ = [
    "CandidateSpec",
    "CandidateVector",
    "CandidateMatch",
    "ph_real_candidate_library",
    "build_ph_real_candidate",
    "build_default_ph_real_candidates",
    "metric_vector_from_candidate",
    "candidate_projected_score",
    "compare_candidate_to_instability",
    "compare_default_ph_real_candidates",
]
