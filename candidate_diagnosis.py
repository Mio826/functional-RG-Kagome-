
from __future__ import annotations

"""Unified candidate-based diagnosis helpers for the kagome fRG pipeline.

This module merges three previously separate candidate libraries into one place:

1. Particle-hole real orders
   Q=0:
     * FM_Q0
     * PI_Q0_E2 = {PI_Q0_dx2y2, PI_Q0_dxy}
   Q=M:
     * CDW_M
     * SDW_M
     * CBO_M
     * SBO_M

2. Particle-particle real orders
   Q=0:
     * PP_S_Q0
     * PP_D_Q0 = {PP_D_Q0_dx2y2, PP_D_Q0_dxy}
     * PP_P_Q0 = {PP_P_Q0_px, PP_P_Q0_py}
     * PP_F_Q0 = {PP_F_Q0_f1, PP_F_Q0_f2}
   Q=M:
     * PDW_S_M
     * PDW_D_M = {PDW_D_M_d1, PDW_D_M_d2}
     * PDW_P_M = {PDW_P_M_p1, PDW_P_M_p2}
     * PDW_F_M = {PDW_F_M_f1, PDW_F_M_f2}

3. Particle-hole current / imaginary orders
   Q=0:
     * LC_Q0_NAGAOSA
     * LC_Q0_FLOWA
   Q=M:
     * LC_M_D6A
     * LC_M_D6B
     * LC_M_D6C
     * LC_M_D6PA

Important notes
---------------
* Q=M current classes are the high-symmetry proxy templates validated in the
  notebook overlap/rank tests. They are suitable for sector identification and
  subspace analysis, but they are not yet a literal bond-by-bond reconstruction
  of Fig. 6 from the flux-classification paper.
* There is intentionally no dedicated "pp-imaginary/chiral" library here.
  Chiral superconductivity should be analyzed as a complex combination inside a
  nearly-degenerate multi-dimensional pp-real subspace, rather than as a new
  basis candidate.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from frg_kernel import canonicalize_q_for_patchsets
from patching import exact_M6_points_1bz

ArrayLike = Sequence[float] | np.ndarray
MetricName = Literal["euclidean", "bubble_weighted", "whitened"]

A, B, C = 0, 1, 2


# ============================================================================
# Shared dataclasses
# ============================================================================

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    channel_type: str            # "ph" or "pp"
    spin_structure: str          # charge/spin or singlet/triplet
    q_type: str                  # "q0" or "M"
    builder_key: str
    irrep_label: Optional[str] = None
    angular_momentum: Optional[int] = None
    representation_kind: str = "scalar_patch"
    notes: Tuple[str, ...] = ()


@dataclass
class CandidateVector:
    spec: CandidateSpec
    Q: np.ndarray
    patch_k: np.ndarray
    vector_patch: np.ndarray
    band_index: Optional[int] = None
    gauge_info: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    @property
    def Npatch(self) -> int:
        return int(np.asarray(self.vector_patch).shape[0])


@dataclass
class FamilyMatch:
    family: str
    metric: str
    subspace_overlap: float
    member_overlaps: Dict[str, float]
    best_member: str
    best_member_overlap: float


# ============================================================================
# Shared linear algebra helpers
# ============================================================================

def _normalize(v: np.ndarray, tol: float = 1e-30) -> np.ndarray:
    v = np.asarray(v, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm <= tol:
        return np.zeros_like(v)
    return v / nrm


def _abs_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(_normalize(a), _normalize(b))))


def _orthonormalize_columns(cols: Sequence[np.ndarray], tol: float = 1e-12) -> np.ndarray:
    out: List[np.ndarray] = []
    for c in cols:
        w = _normalize(np.asarray(c, dtype=complex).reshape(-1))
        for b in out:
            w = w - np.vdot(b, w) * b
        nrm = np.linalg.norm(w)
        if nrm > tol:
            out.append(w / nrm)
    if len(out) == 0:
        n = len(np.asarray(cols[0]).reshape(-1)) if len(cols) else 0
        return np.zeros((n, 0), dtype=complex)
    return np.column_stack(out)


def _subspace_overlap_abs(v: np.ndarray, basis_cols: Sequence[np.ndarray]) -> float:
    v = _normalize(v)
    B = _orthonormalize_columns(basis_cols)
    if B.shape[1] == 0:
        return 0.0
    coeff = B.conjugate().T @ v
    return float(np.sqrt(np.real(np.vdot(coeff, coeff))))


def _metric_vector(v: np.ndarray, metric: MetricName, bubble_weights: Optional[np.ndarray]) -> np.ndarray:
    v = np.asarray(v, dtype=complex).reshape(-1)
    if metric == "euclidean":
        return v.copy()
    if bubble_weights is None:
        raise ValueError(f"metric={metric!r} requires bubble_weights.")
    w = np.asarray(bubble_weights, dtype=float).reshape(-1)
    if w.shape[0] != v.shape[0]:
        raise ValueError("bubble_weights length must match vector length.")
    if metric == "bubble_weighted":
        return w * v
    if metric == "whitened":
        return np.sqrt(np.clip(w, 0.0, None)) * v
    raise ValueError(f"Unsupported metric={metric!r}")


# ============================================================================
# Shared patch / model helpers
# ============================================================================

def _check_patchset(patchset: Any) -> None:
    for attr in ("patch_k", "Npatch"):
        if not hasattr(patchset, attr):
            raise TypeError(f"patchset is missing required attribute {attr!r}")


def _check_patchset_with_eigvec(patchset: Any) -> None:
    _check_patchset(patchset)
    if not hasattr(patchset, "patch_eigvec"):
        raise TypeError("patchset is missing required attribute 'patch_eigvec'")


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


def _band_eigvec_at_k(model: Any, k: ArrayLike, band_index: int, *, anchor_method: str = "max_component") -> np.ndarray:
    k = np.asarray(k, dtype=float)
    evals, evecs = model.eigenstate(float(k[0]), float(k[1]))
    u = np.asarray(evecs[:, int(band_index)], dtype=complex)
    return _anchor_phase(u, method=anchor_method)


def _local_block_eigvec_at_k(model: Any, k: ArrayLike, spin_slice: slice, local_band_index: int, *, anchor_method: str = "max_component") -> np.ndarray:
    k = np.asarray(k, dtype=float)
    Hk = np.asarray(model.Hk(float(k[0]), float(k[1])), dtype=complex)
    Hloc = Hk[spin_slice, spin_slice]
    evals, evecs = np.linalg.eigh(Hloc)
    u = np.asarray(evecs[:, int(local_band_index)], dtype=complex)
    return _anchor_phase(u, method=anchor_method)


def _unique_M3(model: Any, patchset: Any) -> List[np.ndarray]:
    Ms = [np.asarray(canonicalize_q_for_patchsets({"up": patchset}, q), dtype=float)
          for q in exact_M6_points_1bz(model)]
    reps: List[np.ndarray] = []
    keys: List[Tuple[float, float]] = []
    for q in Ms:
        qn = q.copy()
        if (qn[1] < 0) or (abs(qn[1]) < 1e-10 and qn[0] < 0):
            qn = -qn
        key = tuple(np.round(qn, 8))
        if key not in keys:
            keys.append(key)
            reps.append(qn)

    def _ang(q: np.ndarray) -> float:
        a = np.arctan2(q[1], q[0])
        if a < 0:
            a += np.pi
        return float(a)

    reps = sorted(reps, key=_ang)
    if len(reps) != 3:
        raise RuntimeError(f"Expected 3 unique M vectors, found {len(reps)}")
    return reps


def _resolve_Q_and_mindex(
    spec: CandidateSpec,
    *,
    model: Any,
    patchset: Any,
    Q: Optional[ArrayLike],
    m_index: Optional[int],
) -> Tuple[np.ndarray, int]:
    if spec.q_type == "q0":
        return np.zeros(2, dtype=float), 0

    M3 = _unique_M3(model, patchset)
    if Q is None:
        idx = 0 if m_index is None else int(m_index)
        if idx < 0 or idx >= 3:
            raise ValueError("m_index must be 0,1,2 for q_type='M'")
        return np.asarray(M3[idx], dtype=float), idx

    q = np.asarray(canonicalize_q_for_patchsets({"up": patchset}, np.asarray(Q, dtype=float)), dtype=float)
    qn = q.copy()
    if (qn[1] < 0) or (abs(qn[1]) < 1e-10 and qn[0] < 0):
        qn = -qn
    d2 = [float(np.sum((qn - m) ** 2)) for m in M3]
    idx = int(np.argmin(d2))
    return np.asarray(q, dtype=float), idx


# ============================================================================
# PH real
# ============================================================================

_QM_PAIR_MAP = {
    0: (B, C),
    1: (A, C),
    2: (A, B),
}


def _bond_matrix(pair: Tuple[int, int]) -> np.ndarray:
    i, j = pair
    M = np.zeros((3, 3), dtype=complex)
    M[i, j] = 1.0
    M[j, i] = 1.0
    return M


def _density_matrix_for_m(m_idx: int) -> np.ndarray:
    pair = _QM_PAIR_MAP[int(m_idx)]
    vals = np.zeros(3, dtype=float)
    vals[pair[0]] = 1.0
    vals[pair[1]] = -1.0
    return np.diag(vals)


def _bond_matrix_for_m(m_idx: int) -> np.ndarray:
    return _bond_matrix(_QM_PAIR_MAP[int(m_idx)])


def ph_real_candidate_library() -> Dict[str, CandidateSpec]:
    return {
        "FM_Q0": CandidateSpec("FM_Q0", "FM_Q0", "ph", "spin", "q0", "fm_q0", "A1", 0, notes=("Uniform onsite ferromagnetism.",)),
        "PI_Q0_dx2y2": CandidateSpec("PI_Q0_dx2y2", "PI_Q0_E2", "ph", "charge", "q0", "pi_q0_dx2y2", "E2", 2, notes=("Pomeranchuk partner f_dx2-y2 from PRL Eq. (3).",)),
        "PI_Q0_dxy": CandidateSpec("PI_Q0_dxy", "PI_Q0_E2", "ph", "charge", "q0", "pi_q0_dxy", "E2", 2, notes=("Pomeranchuk partner f_dxy from PRL Eq. (3).",)),
        "CDW_M": CandidateSpec("CDW_M", "CDW_M", "ph", "charge", "M", "cdw_m", "M", 0, notes=("Single-Q finite-M charge density wave.",)),
        "SDW_M": CandidateSpec("SDW_M", "SDW_M", "ph", "spin", "M", "sdw_m", "M", 0, notes=("Single-Q finite-M spin density wave.",)),
        "CBO_M": CandidateSpec("CBO_M", "CBO_M", "ph", "charge", "M", "cbo_m", "M", 1, notes=("Single-Q finite-M charge bond order, PRL Eq. (2)-type.",)),
        "SBO_M": CandidateSpec("SBO_M", "SBO_M", "ph", "spin", "M", "sbo_m", "M", 1, notes=("Single-Q finite-M spin bond order, PRL Eq. (2)-type.",)),
    }


PH_REAL_FAMILY_MEMBERS: Dict[str, List[str]] = {
    "FM_Q0": ["FM_Q0"],
    "PI_Q0_E2": ["PI_Q0_dx2y2", "PI_Q0_dxy"],
    "CDW_M": ["CDW_M"],
    "SDW_M": ["SDW_M"],
    "CBO_M": ["CBO_M"],
    "SBO_M": ["SBO_M"],
}


def _build_scalar_q0_fm(patchset: Any) -> np.ndarray:
    return np.ones(int(patchset.Npatch), dtype=complex)


def _build_scalar_q0_pi_dx2y2(patchset: Any) -> np.ndarray:
    k = np.asarray(patchset.patch_k, dtype=float)
    return np.asarray(np.cos(2.0 * k[:, 0]) - np.cos(k[:, 0]) * np.cos(np.sqrt(3.0) * k[:, 1]), dtype=complex)


def _build_scalar_q0_pi_dxy(patchset: Any) -> np.ndarray:
    k = np.asarray(patchset.patch_k, dtype=float)
    return np.asarray(np.sqrt(3.0) * np.sin(k[:, 0]) * np.sin(np.sqrt(3.0) * k[:, 1]), dtype=complex)


def _build_qm_density(*, model: Any, patchset: Any, band_index: int, Q: np.ndarray, m_idx: int, anchor_method: str, use_patchset_eigvec_at_k: bool) -> np.ndarray:
    D = _density_matrix_for_m(m_idx)
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)
    for ip, k in enumerate(patch_k):
        uk = patch_eigvec[ip] if use_patchset_eigvec_at_k else _band_eigvec_at_k(model, k, band_index, anchor_method=anchor_method)
        uq = _band_eigvec_at_k(model, k + Q, band_index, anchor_method=anchor_method)
        out[ip] = np.vdot(uq, D @ uk)
    return out


def _build_qm_bond(*, model: Any, patchset: Any, band_index: int, Q: np.ndarray, m_idx: int, anchor_method: str, use_patchset_eigvec_at_k: bool) -> np.ndarray:
    Bm = _bond_matrix_for_m(m_idx)
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)
    for ip, k in enumerate(patch_k):
        uk = patch_eigvec[ip] if use_patchset_eigvec_at_k else _band_eigvec_at_k(model, k, band_index, anchor_method=anchor_method)
        uq = _band_eigvec_at_k(model, k + Q, band_index, anchor_method=anchor_method)
        phase = np.sin(float(np.dot(Q, k)))
        out[ip] = phase * np.vdot(uq, Bm @ uk)
    return out


def build_ph_real_candidate(
    spec: CandidateSpec,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: Optional[ArrayLike] = None,
    m_index: Optional[int] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
) -> CandidateVector:
    _check_patchset_with_eigvec(patchset)
    q_vec, m_idx = _resolve_Q_and_mindex(spec, model=model, patchset=patchset, Q=Q, m_index=m_index)
    if spec.builder_key == "fm_q0":
        vec = _build_scalar_q0_fm(patchset)
    elif spec.builder_key == "pi_q0_dx2y2":
        vec = _build_scalar_q0_pi_dx2y2(patchset)
    elif spec.builder_key == "pi_q0_dxy":
        vec = _build_scalar_q0_pi_dxy(patchset)
    elif spec.builder_key in {"cdw_m", "sdw_m"}:
        vec = _build_qm_density(model=model, patchset=patchset, band_index=band_index, Q=q_vec, m_idx=m_idx, anchor_method=anchor_method, use_patchset_eigvec_at_k=use_patchset_eigvec_at_k)
    elif spec.builder_key in {"cbo_m", "sbo_m"}:
        vec = _build_qm_bond(model=model, patchset=patchset, band_index=band_index, Q=q_vec, m_idx=m_idx, anchor_method=anchor_method, use_patchset_eigvec_at_k=use_patchset_eigvec_at_k)
    else:
        raise ValueError(f"Unknown builder_key={spec.builder_key!r}")
    return CandidateVector(
        spec=spec,
        Q=np.asarray(q_vec, dtype=float),
        patch_k=np.asarray(patchset.patch_k, dtype=float),
        vector_patch=np.asarray(vec, dtype=complex),
        band_index=int(band_index),
        gauge_info={"anchor_method": str(anchor_method), "use_patchset_eigvec_at_k": bool(use_patchset_eigvec_at_k), "m_index": int(m_idx)},
        notes=spec.notes,
    )


def build_ph_real_family(
    family_name: str,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: Optional[ArrayLike] = None,
    m_index: Optional[int] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
) -> List[CandidateVector]:
    specs = ph_real_candidate_library()
    if family_name not in PH_REAL_FAMILY_MEMBERS:
        raise KeyError(f"Unknown family {family_name!r}")
    return [
        build_ph_real_candidate(
            specs[nm],
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=Q,
            m_index=m_index,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
        for nm in PH_REAL_FAMILY_MEMBERS[family_name]
    ]


def build_default_ph_real_candidates(
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    families: Optional[Iterable[str]] = None,
    Q: Optional[ArrayLike] = None,
    m_index: Optional[int] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
) -> Dict[str, List[CandidateVector]]:
    fams = list(PH_REAL_FAMILY_MEMBERS.keys()) if families is None else [str(x) for x in families]
    return {
        fam: build_ph_real_family(
            fam,
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=Q,
            m_index=m_index,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
        )
        for fam in fams
    }


# ============================================================================
# PP real
# ============================================================================

def pp_real_candidate_library() -> Dict[str, CandidateSpec]:
    return {
        "PP_S_Q0": CandidateSpec("PP_S_Q0", "PP_S_Q0", "pp", "singlet", "q0", "pp_s_q0", "A1", 0, notes=("Q=0 singlet s-wave constant.",)),
        "PP_D_Q0_dx2y2": CandidateSpec("PP_D_Q0_dx2y2", "PP_D_Q0", "pp", "singlet", "q0", "pp_d_q0_dx2y2", "E2", 2, notes=("Q=0 singlet d-wave partner 1.",)),
        "PP_D_Q0_dxy": CandidateSpec("PP_D_Q0_dxy", "PP_D_Q0", "pp", "singlet", "q0", "pp_d_q0_dxy", "E2", 2, notes=("Q=0 singlet d-wave partner 2.",)),
        "PP_P_Q0_px": CandidateSpec("PP_P_Q0_px", "PP_P_Q0", "pp", "triplet", "q0", "pp_p_q0_px", "E1", 1, notes=("Q=0 triplet p-wave partner 1.",)),
        "PP_P_Q0_py": CandidateSpec("PP_P_Q0_py", "PP_P_Q0", "pp", "triplet", "q0", "pp_p_q0_py", "E1", 1, notes=("Q=0 triplet p-wave partner 2.",)),
        "PP_F_Q0_f1": CandidateSpec("PP_F_Q0_f1", "PP_F_Q0", "pp", "triplet", "q0", "pp_f_q0_f1", None, 3, notes=("Q=0 triplet f-wave partner 1.",)),
        "PP_F_Q0_f2": CandidateSpec("PP_F_Q0_f2", "PP_F_Q0", "pp", "triplet", "q0", "pp_f_q0_f2", None, 3, notes=("Q=0 triplet f-wave partner 2.",)),
        "PDW_S_M": CandidateSpec("PDW_S_M", "PDW_S_M", "pp", "singlet", "M", "pdw_s_m", None, 0, notes=("Q=M PDW singlet s-like in relative momentum.",)),
        "PDW_D_M_d1": CandidateSpec("PDW_D_M_d1", "PDW_D_M", "pp", "singlet", "M", "pdw_d_m_d1", None, 2, notes=("Q=M PDW singlet d-like partner 1.",)),
        "PDW_D_M_d2": CandidateSpec("PDW_D_M_d2", "PDW_D_M", "pp", "singlet", "M", "pdw_d_m_d2", None, 2, notes=("Q=M PDW singlet d-like partner 2.",)),
        "PDW_P_M_p1": CandidateSpec("PDW_P_M_p1", "PDW_P_M", "pp", "triplet", "M", "pdw_p_m_p1", None, 1, notes=("Q=M PDW triplet p-like partner 1.",)),
        "PDW_P_M_p2": CandidateSpec("PDW_P_M_p2", "PDW_P_M", "pp", "triplet", "M", "pdw_p_m_p2", None, 1, notes=("Q=M PDW triplet p-like partner 2.",)),
        "PDW_F_M_f1": CandidateSpec("PDW_F_M_f1", "PDW_F_M", "pp", "triplet", "M", "pdw_f_m_f1", None, 3, notes=("Q=M PDW triplet f-like partner 1.",)),
        "PDW_F_M_f2": CandidateSpec("PDW_F_M_f2", "PDW_F_M", "pp", "triplet", "M", "pdw_f_m_f2", None, 3, notes=("Q=M PDW triplet f-like partner 2.",)),
    }


PP_REAL_FAMILY_MEMBERS: Dict[str, List[str]] = {
    "PP_S_Q0": ["PP_S_Q0"],
    "PP_D_Q0": ["PP_D_Q0_dx2y2", "PP_D_Q0_dxy"],
    "PP_P_Q0": ["PP_P_Q0_px", "PP_P_Q0_py"],
    "PP_F_Q0": ["PP_F_Q0_f1", "PP_F_Q0_f2"],
    "PDW_S_M": ["PDW_S_M"],
    "PDW_D_M": ["PDW_D_M_d1", "PDW_D_M_d2"],
    "PDW_P_M": ["PDW_P_M_p1", "PDW_P_M_p2"],
    "PDW_F_M": ["PDW_F_M_f1", "PDW_F_M_f2"],
}


def _theta_from_k(patch_k: np.ndarray) -> np.ndarray:
    patch_k = np.asarray(patch_k, dtype=float)
    return np.arctan2(patch_k[:, 1], patch_k[:, 0])


def _theta_from_relative_k(patch_k: np.ndarray, Q: ArrayLike) -> np.ndarray:
    patch_k = np.asarray(patch_k, dtype=float)
    Q = np.asarray(Q, dtype=float)
    p = patch_k - 0.5 * Q[None, :]
    return np.arctan2(p[:, 1], p[:, 0])


def _build_pp_real_family_patchvectors(patch_k: np.ndarray, Q: np.ndarray, family_name: str) -> Dict[str, np.ndarray]:
    if np.allclose(Q, 0.0):
        theta = _theta_from_k(patch_k)
        if family_name == "PP_S_Q0":
            return {"PP_S_Q0": np.ones(len(theta), dtype=float)}
        if family_name == "PP_D_Q0":
            return {"PP_D_Q0_dx2y2": np.cos(2.0 * theta), "PP_D_Q0_dxy": np.sin(2.0 * theta)}
        if family_name == "PP_P_Q0":
            return {"PP_P_Q0_px": np.cos(theta), "PP_P_Q0_py": np.sin(theta)}
        if family_name == "PP_F_Q0":
            return {"PP_F_Q0_f1": np.cos(3.0 * theta), "PP_F_Q0_f2": np.sin(3.0 * theta)}
    else:
        theta = _theta_from_relative_k(patch_k, Q)
        if family_name == "PDW_S_M":
            return {"PDW_S_M": np.ones(len(theta), dtype=float)}
        if family_name == "PDW_D_M":
            return {"PDW_D_M_d1": np.cos(2.0 * theta), "PDW_D_M_d2": np.sin(2.0 * theta)}
        if family_name == "PDW_P_M":
            return {"PDW_P_M_p1": np.cos(theta), "PDW_P_M_p2": np.sin(theta)}
        if family_name == "PDW_F_M":
            return {"PDW_F_M_f1": np.cos(3.0 * theta), "PDW_F_M_f2": np.sin(3.0 * theta)}
    raise KeyError(f"Unsupported pp-real family {family_name!r}")


def build_pp_real_family(
    family_name: str,
    *,
    patchset: Any,
    Q: ArrayLike,
) -> Dict[str, CandidateVector]:
    _check_patchset(patchset)
    specs = pp_real_candidate_library()
    if family_name not in PP_REAL_FAMILY_MEMBERS:
        raise KeyError(f"Unknown family {family_name!r}")
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    Q = np.asarray(Q, dtype=float)
    raw = _build_pp_real_family_patchvectors(patch_k, Q, family_name)
    out: Dict[str, CandidateVector] = {}
    for name, vec in raw.items():
        spec = specs[name]
        out[name] = CandidateVector(
            spec=spec,
            Q=Q.copy(),
            patch_k=patch_k.copy(),
            vector_patch=np.asarray(vec, dtype=complex),
            band_index=None,
            gauge_info={"representation_kind": "scalar_patch"},
            notes=spec.notes,
        )
    return out


def build_default_pp_real_candidates(
    *,
    patchset: Any,
    families: Sequence[str],
    Q: ArrayLike,
) -> Dict[str, Dict[str, CandidateVector]]:
    return {fam: build_pp_real_family(fam, patchset=patchset, Q=Q) for fam in families}


# ============================================================================
# PH current / imaginary
# ============================================================================

def current_candidate_library() -> Dict[str, CandidateSpec]:
    return {
        "LC_Q0_NAGAOSA": CandidateSpec("LC_Q0_NAGAOSA", "LC_Q0_NAGAOSA", "ph", "charge", "q0", "lc_q0_nagaosa", "Q0_current", None, notes=("Q=0 Nagaosa-type loop-current representative.",)),
        "LC_Q0_FLOWA": CandidateSpec("LC_Q0_FLOWA", "LC_Q0_FLOWA", "ph", "charge", "q0", "lc_q0_flowa", "Q0_current", None, notes=("Q=0 second high-symmetry 1x1 current representative.",)),
        "LC_M_D6A": CandidateSpec("LC_M_D6A", "LC_M_D6A", "ph", "charge", "M", "lc_m_d6a", "M_current", None, notes=("Q=M high-symmetry 2x2 current proxy for D6a class.",)),
        "LC_M_D6B": CandidateSpec("LC_M_D6B", "LC_M_D6B", "ph", "charge", "M", "lc_m_d6b", "M_current", None, notes=("Q=M high-symmetry 2x2 current proxy for D6b class.",)),
        "LC_M_D6C": CandidateSpec("LC_M_D6C", "LC_M_D6C", "ph", "charge", "M", "lc_m_d6c", "M_current", None, notes=("Q=M high-symmetry 2x2 current proxy for D6c class.",)),
        "LC_M_D6PA": CandidateSpec("LC_M_D6PA", "LC_M_D6PA", "ph", "charge", "M", "lc_m_d6pa", "M_current", None, notes=("Q=M high-symmetry 2x2 current proxy for D'6a class.",)),
    }


CURRENT_FAMILY_MEMBERS: Dict[str, List[str]] = {
    "LC_Q0_NAGAOSA": ["LC_Q0_NAGAOSA"],
    "LC_Q0_FLOWA": ["LC_Q0_FLOWA"],
    "LC_M_D6A": ["LC_M_D6A"],
    "LC_M_D6B": ["LC_M_D6B"],
    "LC_M_D6C": ["LC_M_D6C"],
    "LC_M_D6PA": ["LC_M_D6PA"],
}


def _current_matrix(pair: Tuple[int, int], sign: float = +1.0) -> np.ndarray:
    i, j = pair
    M = np.zeros((3, 3), dtype=complex)
    M[i, j] = -1j * sign
    M[j, i] = +1j * sign
    return M


J_AB = _current_matrix((A, B), sign=+1.0)
J_AC = _current_matrix((A, C), sign=+1.0)
J_BC = _current_matrix((B, C), sign=+1.0)
J_BONDS = [J_AB, J_AC, J_BC]


def _cos_bond_operator(model: Any, kx: float, ky: float, eta_ab: float, eta_ac: float, eta_bc: float, prefactor: float = 1.0) -> np.ndarray:
    kvec = np.array([kx, ky], dtype=float)
    c1 = np.cos(np.dot(model.delta1, kvec))
    c2 = np.cos(np.dot(model.delta2, kvec))
    c3 = np.cos(np.dot(model.delta3, kvec))
    O = np.array(
        [
            [0.0, 1j * prefactor * eta_ab * c1, 1j * prefactor * eta_ac * c2],
            [-1j * prefactor * eta_ab * c1, 0.0, 1j * prefactor * eta_bc * c3],
            [-1j * prefactor * eta_ac * c2, -1j * prefactor * eta_bc * c3, 0.0],
        ],
        dtype=complex,
    )
    return 0.5 * (O + O.conjugate().T)


def _dH_dphi_nagaosa(model: Any, kx: float, ky: float) -> np.ndarray:
    if not hasattr(model, "parameters") or "phi" not in model.parameters:
        raise TypeError("Exact Nagaosa current requires a model with parameters['phi'] and kagome delta vectors.")
    t = model.parameters["t"]
    phi = model.parameters["phi"]
    kvec = np.array([kx, ky], dtype=float)
    c1 = np.cos(np.dot(model.delta1, kvec))
    c2 = np.cos(np.dot(model.delta2, kvec))
    c3 = np.cos(np.dot(model.delta3, kvec))
    AB = -2.0 * t * np.exp(1j * phi / 3.0) * c1
    AC = -2.0 * t * np.exp(-1j * phi / 3.0) * c2
    BC = -2.0 * t * np.exp(1j * phi / 3.0) * c3
    dAB = (1j / 3.0) * AB
    dAC = (-1j / 3.0) * AC
    dBC = (1j / 3.0) * BC
    O = np.array(
        [
            [0.0, dAB, dAC],
            [np.conjugate(dAB), 0.0, dBC],
            [np.conjugate(dAC), np.conjugate(dBC), 0.0],
        ],
        dtype=complex,
    )
    return 0.5 * (O + O.conjugate().T)


def _q0_current_operator(model: Any, kx: float, ky: float, builder_key: str, *, use_exact_q0_nagaosa: bool) -> np.ndarray:
    if builder_key == "lc_q0_nagaosa":
        if use_exact_q0_nagaosa:
            return _dH_dphi_nagaosa(model, kx, ky)
        return _cos_bond_operator(model, kx, ky, +1.0, -1.0, +1.0, prefactor=2.0)
    if builder_key == "lc_q0_flowa":
        return _cos_bond_operator(model, kx, ky, +1.0, +1.0, +1.0, prefactor=2.0)
    raise ValueError(f"Unknown Q=0 builder_key={builder_key!r}")


def _build_q0_current(
    *,
    spec: CandidateSpec,
    model: Any,
    patchset: Any,
    band_index: int,
    anchor_method: str,
    use_patchset_eigvec_at_k: bool,
    use_exact_q0_nagaosa: bool,
) -> np.ndarray:
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    for ip, k in enumerate(patch_k):
        uk = patch_eigvec[ip] if use_patchset_eigvec_at_k else _band_eigvec_at_k(model, k, band_index, anchor_method=anchor_method)
        uq = _band_eigvec_at_k(model, k, band_index, anchor_method=anchor_method)
        O = _q0_current_operator(model, float(k[0]), float(k[1]), spec.builder_key, use_exact_q0_nagaosa=use_exact_q0_nagaosa)
        out[ip] = np.vdot(uq, O @ uk)
    return np.real_if_close(out, tol=1000)


QM_CLASS_PROXY = {
    "lc_m_d6a": {"cell_signs": np.array([+1, +1, +1, +1], dtype=float), "bond_signs": np.array([+1, +1, +1], dtype=float)},
    "lc_m_d6b": {"cell_signs": np.array([+1, -1, -1, -1], dtype=float), "bond_signs": np.array([+1, +1, +1], dtype=float)},
    "lc_m_d6c": {"cell_signs": np.array([+1, +1, -1, +1], dtype=float), "bond_signs": np.array([+1, -1, +1], dtype=float)},
    "lc_m_d6pa": {"cell_signs": np.array([+1, -1, +1, -1], dtype=float), "bond_signs": np.array([+1, -1, -1], dtype=float)},
}


def _build_qm_proxy(
    *,
    spec: CandidateSpec,
    model: Any,
    patchset: Any,
    spin_slice: slice,
    local_band_index: int,
    Q: np.ndarray,
    anchor_method: str,
) -> np.ndarray:
    if spec.builder_key not in QM_CLASS_PROXY:
        raise ValueError(f"Unknown Q=M builder_key={spec.builder_key!r}")
    info = QM_CLASS_PROXY[spec.builder_key]
    cell_signs = info["cell_signs"]
    bond_signs = info["bond_signs"]
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_eigvec = np.asarray(patchset.patch_eigvec, dtype=complex)
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    Rcells = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3) / 2]),
        np.array([1.5, np.sqrt(3) / 2]),
    ]
    for ip, k in enumerate(patch_k):
        uk = patch_eigvec[ip].reshape(-1)
        uq = _local_block_eigvec_at_k(model, k + Q, spin_slice, local_band_index, anchor_method=anchor_method)
        val = 0.0 + 0.0j
        for ic, R in enumerate(Rcells):
            phase_cell = np.exp(1j * np.dot(Q, R))
            for ib, Jb in enumerate(J_BONDS):
                val += cell_signs[ic] * bond_signs[ib] * phase_cell * np.vdot(uq, Jb @ uk)
        out[ip] = val
    return np.real_if_close(out, tol=1000)


def build_current_candidate(
    spec: CandidateSpec,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: Optional[ArrayLike] = None,
    m_index: Optional[int] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
    use_exact_q0_nagaosa: bool = True,
    spin_slice: Optional[slice] = None,
    local_band_index: Optional[int] = None,
) -> CandidateVector:
    _check_patchset_with_eigvec(patchset)
    q_vec, m_idx = _resolve_Q_and_mindex(spec, model=model, patchset=patchset, Q=Q, m_index=m_index)
    if spec.q_type == "q0":
        vec = _build_q0_current(
            spec=spec,
            model=model,
            patchset=patchset,
            band_index=band_index,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
            use_exact_q0_nagaosa=use_exact_q0_nagaosa,
        )
        gauge_info = {
            "anchor_method": str(anchor_method),
            "use_patchset_eigvec_at_k": bool(use_patchset_eigvec_at_k),
            "use_exact_q0_nagaosa": bool(use_exact_q0_nagaosa),
            "m_index": int(m_idx),
        }
    else:
        if spin_slice is None or local_band_index is None:
            raise ValueError("Q=M current candidates require spin_slice and local_band_index.")
        vec = _build_qm_proxy(
            spec=spec,
            model=model,
            patchset=patchset,
            spin_slice=spin_slice,
            local_band_index=int(local_band_index),
            Q=q_vec,
            anchor_method=anchor_method,
        )
        gauge_info = {
            "anchor_method": str(anchor_method),
            "spin_slice": spin_slice,
            "local_band_index": int(local_band_index),
            "m_index": int(m_idx),
            "proxy_kind": "high_symmetry_2x2_current",
        }
    return CandidateVector(
        spec=spec,
        Q=np.asarray(q_vec, dtype=float),
        patch_k=np.asarray(patchset.patch_k, dtype=float),
        vector_patch=np.asarray(vec, dtype=complex),
        band_index=int(band_index),
        gauge_info=gauge_info,
        notes=spec.notes,
    )


def build_current_family(
    family_name: str,
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    Q: Optional[ArrayLike] = None,
    m_index: Optional[int] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
    use_exact_q0_nagaosa: bool = True,
    spin_slice: Optional[slice] = None,
    local_band_index: Optional[int] = None,
) -> List[CandidateVector]:
    specs = current_candidate_library()
    if family_name not in CURRENT_FAMILY_MEMBERS:
        raise KeyError(f"Unknown family {family_name!r}")
    return [
        build_current_candidate(
            specs[nm],
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=Q,
            m_index=m_index,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
            use_exact_q0_nagaosa=use_exact_q0_nagaosa,
            spin_slice=spin_slice,
            local_band_index=local_band_index,
        )
        for nm in CURRENT_FAMILY_MEMBERS[family_name]
    ]


def build_default_current_candidates(
    *,
    model: Any,
    patchset: Any,
    band_index: int,
    families: Optional[Iterable[str]] = None,
    Q: Optional[ArrayLike] = None,
    m_index: Optional[int] = None,
    anchor_method: str = "max_component",
    use_patchset_eigvec_at_k: bool = True,
    use_exact_q0_nagaosa: bool = True,
    spin_slice: Optional[slice] = None,
    local_band_index: Optional[int] = None,
) -> Dict[str, List[CandidateVector]]:
    fams = list(CURRENT_FAMILY_MEMBERS.keys()) if families is None else [str(x) for x in families]
    return {
        fam: build_current_family(
            fam,
            model=model,
            patchset=patchset,
            band_index=band_index,
            Q=Q,
            m_index=m_index,
            anchor_method=anchor_method,
            use_patchset_eigvec_at_k=use_patchset_eigvec_at_k,
            use_exact_q0_nagaosa=use_exact_q0_nagaosa,
            spin_slice=spin_slice,
            local_band_index=local_band_index,
        )
        for fam in fams
    }


# ============================================================================
# Shared family-to-instability comparison
# ============================================================================

def metric_vector_from_candidate(cand: CandidateVector, *, metric: MetricName = "euclidean", bubble_weights: Optional[np.ndarray] = None) -> np.ndarray:
    return _metric_vector(cand.vector_patch, metric=metric, bubble_weights=bubble_weights)


def compare_family_to_instability(
    family_name: str,
    family_candidates: Sequence[CandidateVector] | Mapping[str, CandidateVector],
    *,
    leading_evec: np.ndarray,
    metric: MetricName = "euclidean",
    bubble_weights: Optional[np.ndarray] = None,
) -> FamilyMatch:
    if isinstance(family_candidates, Mapping):
        family_candidates = list(family_candidates.values())
    target = _metric_vector(leading_evec, metric=metric, bubble_weights=bubble_weights)
    target = _normalize(target)
    member_overlaps: Dict[str, float] = {}
    fam_cols: List[np.ndarray] = []
    for cand in family_candidates:
        v = metric_vector_from_candidate(cand, metric=metric, bubble_weights=bubble_weights)
        fam_cols.append(v)
        member_overlaps[cand.spec.name] = float(np.abs(np.vdot(_normalize(v), target)))
    sub_ov = _subspace_overlap_abs(target, fam_cols)
    best_member = max(member_overlaps, key=member_overlaps.get) if member_overlaps else ""
    best_val = float(member_overlaps[best_member]) if best_member else 0.0
    return FamilyMatch(
        family=family_name,
        metric=str(metric),
        subspace_overlap=float(sub_ov),
        member_overlaps=member_overlaps,
        best_member=str(best_member),
        best_member_overlap=best_val,
    )


__all__ = [
    "CandidateSpec",
    "CandidateVector",
    "FamilyMatch",
    # shared comparison / metric
    "metric_vector_from_candidate",
    "compare_family_to_instability",
    # ph real
    "ph_real_candidate_library",
    "PH_REAL_FAMILY_MEMBERS",
    "build_ph_real_candidate",
    "build_ph_real_family",
    "build_default_ph_real_candidates",
    # pp real
    "pp_real_candidate_library",
    "PP_REAL_FAMILY_MEMBERS",
    "build_pp_real_family",
    "build_default_pp_real_candidates",
    # current
    "current_candidate_library",
    "CURRENT_FAMILY_MEMBERS",
    "build_current_candidate",
    "build_current_family",
    "build_default_current_candidates",
]
