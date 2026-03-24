from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Sequence[float]
SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]


@dataclass
class TemplateProjection:
    """Projection result onto one template family / irrep."""

    name: str
    score: float
    coefficients: np.ndarray
    basis_dim: int


@dataclass
class ModeAnalysis:
    """Structured analysis of one eigenmode (or a nearly-degenerate subspace)."""

    kernel_name: str
    Q: np.ndarray
    eigenvalues: np.ndarray
    basis_vectors: np.ndarray  # shape (Npatch, subspace_dim)
    subspace_dim: int
    dominant_label: str
    dominant_score: float
    projections: List[TemplateProjection]
    patch_weight: np.ndarray
    dominant_sublattice: Optional[int]
    sublattice_weights: Optional[np.ndarray]
    partner_parity: Optional[float]
    notes: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kernel": self.kernel_name,
            "Q": np.asarray(self.Q, dtype=float).tolist(),
            "subspace_dim": int(self.subspace_dim),
            "dominant_label": self.dominant_label,
            "dominant_score": float(self.dominant_score),
            "eigenvalues": [complex(v) for v in self.eigenvalues],
            "partner_parity": None if self.partner_parity is None else float(self.partner_parity),
            "notes": list(self.notes),
        }
        if self.sublattice_weights is not None:
            out["sublattice_weights"] = [float(x) for x in self.sublattice_weights]
            out["dominant_sublattice"] = None if self.dominant_sublattice is None else int(self.dominant_sublattice)
        return out


@dataclass
class TemplateFamily:
    """A named set of orthonormal basis vectors on patch space."""

    name: str
    basis: np.ndarray  # shape (Npatch, n_basis)

    def project_subspace(self, subspace_basis: np.ndarray) -> TemplateProjection:
        overlap = self.basis.conjugate().T @ subspace_basis
        basis_dim = int(self.basis.shape[1])
        denom = max(1, min(basis_dim, int(subspace_basis.shape[1])))
        score = float(np.linalg.norm(overlap, ord="fro") ** 2 / denom)
        coeffs = np.sum(np.abs(overlap) ** 2, axis=1)
        return TemplateProjection(
            name=self.name,
            score=score,
            coefficients=np.asarray(coeffs, dtype=float),
            basis_dim=basis_dim,
        )


def _normalize_spin_label(spin: SpinLike) -> str:
    s = str(spin).lower()
    aliases = {
        "up": "up",
        "u": "up",
        "0": "up",
        "dn": "dn",
        "down": "dn",
        "d": "dn",
        "1": "dn",
        "s": "S",
        "t": "T",
        "rho": "rho",
        "sz": "sz",
    }
    return aliases.get(s, str(spin))


def _normalize_columns(mat: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    mat = np.asarray(mat, dtype=complex)
    if mat.ndim == 1:
        mat = mat[:, None]
    cols: List[np.ndarray] = []
    for j in range(mat.shape[1]):
        v = mat[:, j].astype(complex, copy=True)
        nrm = float(np.linalg.norm(v))
        if nrm > tol:
            cols.append(v / nrm)
    if not cols:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    return np.column_stack(cols)


def _qr_orthonormalize(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    mat = _normalize_columns(mat, tol=tol)
    if mat.shape[1] == 0:
        return mat
    q, _ = np.linalg.qr(mat)
    keep = []
    for j in range(q.shape[1]):
        if np.linalg.norm(q[:, j]) > tol:
            keep.append(q[:, j])
    if not keep:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    return np.column_stack(keep)


def _extract_patch_positions(patchset: Any) -> np.ndarray:
    if patchset is None or not hasattr(patchset, "patches"):
        raise ValueError("patchset must have a .patches attribute.")
    ks = []
    for p in patchset.patches:
        if hasattr(p, "k_cart"):
            k = np.asarray(p.k_cart, dtype=float)
        elif hasattr(p, "k"):
            k = np.asarray(p.k, dtype=float)
        else:
            raise ValueError("Each patch must provide k_cart (preferred) or k.")
        if k.shape != (2,):
            raise ValueError(f"Expected 2D patch momenta, got shape {k.shape}.")
        ks.append(k)
    return np.asarray(ks, dtype=float)


def _extract_patch_eigvecs(patchset: Any) -> np.ndarray:
    if patchset is None or not hasattr(patchset, "patches"):
        raise ValueError("patchset must have a .patches attribute.")
    eigvecs = []
    for p in patchset.patches:
        if not hasattr(p, "eigvec"):
            return np.zeros((len(patchset.patches), 0), dtype=complex)
        eigvecs.append(np.asarray(p.eigvec, dtype=complex))
    if not eigvecs:
        return np.zeros((0, 0), dtype=complex)
    return np.asarray(eigvecs, dtype=complex)


def _centered_angles(ks: np.ndarray, Q: Optional[np.ndarray] = None) -> np.ndarray:
    center = np.zeros(2, dtype=float) if Q is None else np.asarray(Q, dtype=float) / 2.0
    rel = ks - center[None, :]
    return np.arctan2(rel[:, 1], rel[:, 0])


def _orthonormal_family(name: str, vectors: Sequence[np.ndarray]) -> TemplateFamily:
    mat = np.column_stack([np.asarray(v, dtype=complex) for v in vectors])
    mat = _qr_orthonormalize(mat)
    return TemplateFamily(name=name, basis=mat)


def default_template_families(
    patchset: Any,
    kernel_name: str,
    Q: Optional[ArrayLike] = None,
) -> List[TemplateFamily]:
    ks = _extract_patch_positions(patchset)
    theta = _centered_angles(ks, None if Q is None else np.asarray(Q, dtype=float))
    kx = ks[:, 0]
    ky = ks[:, 1]
    ones = np.ones(len(ks), dtype=float)

    fams: List[TemplateFamily] = []
    fams.append(_orthonormal_family("s", [ones]))
    fams.append(_orthonormal_family("p", [np.cos(theta), np.sin(theta)]))
    fams.append(_orthonormal_family("d", [np.cos(2 * theta), np.sin(2 * theta)]))
    fams.append(_orthonormal_family("f", [np.cos(3 * theta), np.sin(3 * theta)]))

    f_dx2y2 = np.cos(2 * kx) - np.cos(kx) * np.cos(np.sqrt(3.0) * ky)
    f_dxy = np.sqrt(3.0) * np.sin(kx) * np.sin(np.sqrt(3.0) * ky)
    fams.append(_orthonormal_family("paper_d_Q0", [f_dx2y2, f_dxy]))

    if Q is not None:
        Q = np.asarray(Q, dtype=float)
        phase = ks @ Q
        fams.append(_orthonormal_family("finiteQ_cos_sin", [np.cos(phase), np.sin(phase)]))

    if kernel_name.startswith("pp"):
        fams.append(_orthonormal_family("pp_even", [ones, np.cos(2 * theta), np.sin(2 * theta)]))
        fams.append(_orthonormal_family("pp_odd", [np.cos(theta), np.sin(theta), np.cos(3 * theta), np.sin(3 * theta)]))
    else:
        fams.append(_orthonormal_family("ph_nematic", [np.cos(2 * theta), np.sin(2 * theta)]))

    return fams


class OrderRecognizer:
    """
    Spin-aware version of the coarse recognizer.

    Backward compatible:
      - OrderRecognizer(patchset)
      - OrderRecognizer(patchsets_by_spin={"up": ..., "dn": ...})

    The generic symmetry diagnosis still lives in patch space. The only thing we
    make spin-aware here is: when summarizing sublattice weight, choose the
    appropriate Bloch eigenvectors for the kernel sector instead of always using a
    single global patchset.
    """

    def __init__(
        self,
        patchset: Any = None,
        *,
        patchsets_by_spin: Optional[PatchSetMap] = None,
        default_spin: str = "up",
        degeneracy_tol: float = 1e-6,
        projection_threshold: float = 0.20,
    ) -> None:
        if patchsets_by_spin is None:
            if patchset is None:
                raise ValueError("Provide either patchset or patchsets_by_spin.")
            patchsets_by_spin = {default_spin: patchset}
        elif patchset is not None:
            raise ValueError("Provide patchset or patchsets_by_spin, not both.")

        self.patchsets_by_spin: Dict[str, Any] = {
            _normalize_spin_label(k): v for k, v in patchsets_by_spin.items()
        }
        if len(self.patchsets_by_spin) == 0:
            raise ValueError("patchsets_by_spin is empty.")

        self.default_spin = _normalize_spin_label(default_spin)
        if self.default_spin not in self.patchsets_by_spin:
            self.default_spin = next(iter(self.patchsets_by_spin.keys()))

        self.patchset = self.patchsets_by_spin[self.default_spin]
        self.ks = _extract_patch_positions(self.patchset)
        self.eigvecs = _extract_patch_eigvecs(self.patchset)
        self.ks_by_spin: Dict[str, np.ndarray] = {}
        self.eigvecs_by_spin: Dict[str, np.ndarray] = {}
        self._validate_patchsets()

        self.degeneracy_tol = float(degeneracy_tol)
        self.projection_threshold = float(projection_threshold)

    def _validate_patchsets(self) -> None:
        ref_ks = None
        ref_npatch = None
        for spin, ps in self.patchsets_by_spin.items():
            ks = _extract_patch_positions(ps)
            eig = _extract_patch_eigvecs(ps)
            self.ks_by_spin[spin] = ks
            self.eigvecs_by_spin[spin] = eig
            if ref_ks is None:
                ref_ks = ks
                ref_npatch = ks.shape[0]
            else:
                if ks.shape[0] != ref_npatch:
                    raise ValueError("All spin patchsets used by Benchmark2 must have the same Npatch.")
                if ks.shape == ref_ks.shape and not np.allclose(ks, ref_ks, atol=1e-10, rtol=0.0):
                    raise ValueError(
                        "Benchmark2 assumes all spin sectors share the same patch representatives in momentum space. "
                        "Their Bloch eigenvectors may differ, but the patch coordinates must match."
                    )

    def _get_patchset_for_spin(self, spin: SpinLike):
        key = _normalize_spin_label(spin)
        if key in self.patchsets_by_spin:
            return self.patchsets_by_spin[key]
        if self.default_spin in self.patchsets_by_spin:
            return self.patchsets_by_spin[self.default_spin]
        return next(iter(self.patchsets_by_spin.values()))

    def _get_eigvecs_for_spin(self, spin: SpinLike) -> np.ndarray:
        ps = self._get_patchset_for_spin(spin)
        return _extract_patch_eigvecs(ps)

    def _primary_physical_spin_for_kernel(self, kernel: Any) -> str:
        # Prefer explicit kernel metadata from channels.py over name heuristics.
        if hasattr(kernel, "col_spins"):
            for s in kernel.col_spins:
                key = _normalize_spin_label(s)
                if key in {"up", "dn"}:
                    return key
        name = getattr(kernel, "name", "").lower()
        if "dd" in name or "dn" in name:
            return "dn"
        return self.default_spin

    def _sorted_eigensystem(self, kernel: Any, sort_by: str = "abs") -> Tuple[np.ndarray, np.ndarray]:
        vals, vecs = kernel.eig(sort_by=sort_by)
        vals = np.asarray(vals)
        vecs = np.asarray(vecs, dtype=complex)
        return vals, _normalize_columns(vecs)

    def _group_leading_subspaces(self, vals: np.ndarray) -> List[List[int]]:
        groups: List[List[int]] = []
        if vals.size == 0:
            return groups
        mags = np.abs(vals)
        i = 0
        while i < len(vals):
            group = [i]
            ref = mags[i]
            j = i + 1
            while j < len(vals):
                scale = max(1.0, abs(ref), abs(mags[j]))
                if abs(mags[j] - ref) / scale <= self.degeneracy_tol:
                    group.append(j)
                    j += 1
                else:
                    break
            groups.append(group)
            i = j
        return groups

    def _subspace_basis(self, vecs: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        basis = vecs[:, list(indices)]
        return _qr_orthonormalize(basis)

    def _template_projections(
        self,
        subspace_basis: np.ndarray,
        kernel_name: str,
        Q: np.ndarray,
        families: Optional[Sequence[TemplateFamily]] = None,
    ) -> List[TemplateProjection]:
        if families is None:
            families = default_template_families(self.patchset, kernel_name, Q)
        out = [fam.project_subspace(subspace_basis) for fam in families]
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _patch_weight(self, subspace_basis: np.ndarray) -> np.ndarray:
        weight = np.sum(np.abs(subspace_basis) ** 2, axis=1)
        total = float(np.sum(weight))
        if total > 0:
            weight = weight / total
        return np.asarray(weight, dtype=float)

    def _sublattice_weights(self, kernel: Any, patch_weight: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[int]]:
        eigvecs = self._get_eigvecs_for_spin(self._primary_physical_spin_for_kernel(kernel))
        if eigvecs.size == 0:
            return None, None
        orb_w = np.abs(eigvecs) ** 2
        if orb_w.ndim != 2:
            return None, None
        if orb_w.shape[0] != patch_weight.shape[0]:
            return None, None
        weights = np.sum(patch_weight[:, None] * orb_w, axis=0)
        total = float(np.sum(weights))
        if total > 0:
            weights = weights / total
        dominant = int(np.argmax(weights))
        return np.asarray(weights, dtype=float), dominant

    def _partner_parity(self, kernel: Any, vec: np.ndarray) -> Optional[float]:
        if not hasattr(kernel, "col_partner_patches"):
            return None
        if getattr(kernel, "name", "").lower() in {"pp_singlet_sz0", "pp_triplet_sz0"}:
            # For mixed-spin pair kernels, patch-space parity alone is not a clean proxy.
            return None
        partner = np.asarray(kernel.col_partner_patches, dtype=int)
        if partner.shape[0] != vec.shape[0]:
            return None
        vp = vec[partner]
        denom = float(np.linalg.norm(vec) ** 2)
        if denom < 1e-14:
            return None
        overlap = np.vdot(vec, vp) / denom
        return float(np.real(overlap))

    def _paper_notes(
        self,
        kernel_name: str,
        Q: np.ndarray,
        projections: Sequence[TemplateProjection],
        subspace_dim: int,
    ) -> List[str]:
        notes: List[str] = []
        qnorm = float(np.linalg.norm(Q))
        if kernel_name.startswith("ph") and qnorm < 1e-10:
            notes.append("Q≈0 particle-hole mode: candidate Pomeranchuk / nematic sector.")
        if kernel_name.startswith("ph") and qnorm >= 1e-10:
            notes.append("Finite-Q particle-hole mode: candidate bond / density-wave sector.")
        if kernel_name.startswith("pp"):
            notes.append("Particle-particle mode: candidate superconducting sector.")
        if subspace_dim >= 2 and projections and projections[0].name in {"d", "paper_d_Q0", "ph_nematic"}:
            notes.append("Leading space is at least two-dimensional; treat it as an irrep subspace, not one fixed basis vector.")
        if kernel_name.startswith("pp"):
            notes.append(
                "Patch-space scalar harmonics can diagnose even/odd and approximate angular momentum, "
                "but do not by themselves reconstruct the paper's full sublattice-pair f-wave tensor structure."
            )
        if kernel_name.startswith("ph"):
            notes.append(
                "Particle-hole patch harmonics can diagnose Q and rough lattice symmetry, "
                "but real-space bond-order identification still needs a separate bond/sublattice reconstruction layer."
            )
        return notes

    def analyze_kernel(
        self,
        kernel: Any,
        *,
        n_groups: int = 3,
        sort_by: str = "abs",
        template_families: Optional[Sequence[TemplateFamily]] = None,
    ) -> List[ModeAnalysis]:
        vals, vecs = self._sorted_eigensystem(kernel, sort_by=sort_by)
        groups = self._group_leading_subspaces(vals)[: int(n_groups)]
        analyses: List[ModeAnalysis] = []
        for group in groups:
            subspace = self._subspace_basis(vecs, group)
            projections = self._template_projections(subspace, kernel.name, np.asarray(kernel.Q, dtype=float), template_families)
            dominant = projections[0] if projections else TemplateProjection("unclassified", 0.0, np.zeros(0), 0)
            patch_weight = self._patch_weight(subspace)
            sublat_w, dominant_sublat = self._sublattice_weights(kernel, patch_weight)
            partner_parity = None
            if kernel.name.startswith("pp") and subspace.shape[1] >= 1:
                partner_parity = self._partner_parity(kernel, subspace[:, 0])
            notes = self._paper_notes(kernel.name, np.asarray(kernel.Q, dtype=float), projections, subspace.shape[1])
            if dominant.score < self.projection_threshold:
                notes.append("No template family has strong overlap; keep this mode as numerically discovered / potentially novel.")
                dominant_label = "unclassified"
            else:
                dominant_label = dominant.name
            analyses.append(
                ModeAnalysis(
                    kernel_name=kernel.name,
                    Q=np.asarray(kernel.Q, dtype=float),
                    eigenvalues=np.asarray(vals[group]),
                    basis_vectors=subspace,
                    subspace_dim=int(subspace.shape[1]),
                    dominant_label=dominant_label,
                    dominant_score=float(dominant.score),
                    projections=projections,
                    patch_weight=patch_weight,
                    dominant_sublattice=dominant_sublat,
                    sublattice_weights=sublat_w,
                    partner_parity=partner_parity,
                    notes=notes,
                )
            )
        return analyses

    def analyze_kernel_dict(
        self,
        kernels: Mapping[str, Any],
        *,
        n_groups: int = 2,
        sort_by: str = "abs",
    ) -> Dict[str, List[ModeAnalysis]]:
        out: Dict[str, List[ModeAnalysis]] = {}
        for key, kernel in kernels.items():
            out[key] = self.analyze_kernel(kernel, n_groups=n_groups, sort_by=sort_by)
        return out


__all__ = [
    "ModeAnalysis",
    "OrderRecognizer",
    "TemplateFamily",
    "TemplateProjection",
    "default_template_families",
    "_extract_patch_eigvecs",
    "_extract_patch_positions",
]
