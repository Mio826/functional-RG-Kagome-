
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from form_factor import ModeAnalysis, OrderRecognizer, _extract_patch_eigvecs, _extract_patch_positions
except Exception:
    from order_recognition import ModeAnalysis, OrderRecognizer, _extract_patch_eigvecs, _extract_patch_positions


@dataclass
class PaperTemplateScore:
    name: str
    score: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class InternalModeTensors:
    """
    Reconstructed internal-index mode basis.

    tensor_basis has shape (Npatch, Norb, Norb, dim)
      - ph channel: Phi_{l n}(p;Q)
      - pp channel: Delta_{m n}(p)
    """
    channel_sector: str
    spin_sector: str
    tensor_basis: np.ndarray
    pair_weights: np.ndarray
    dominant_pair: Optional[Tuple[int, int]]
    same_sublattice_weight: float
    inter_sublattice_weight: float

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "channel_sector": self.channel_sector,
            "spin_sector": self.spin_sector,
            "tensor_basis_shape": tuple(int(x) for x in self.tensor_basis.shape),
            "dominant_pair": None if self.dominant_pair is None else tuple(int(x) for x in self.dominant_pair),
            "same_sublattice_weight": float(self.same_sublattice_weight),
            "inter_sublattice_weight": float(self.inter_sublattice_weight),
            "pair_weights": np.asarray(self.pair_weights, dtype=float).tolist(),
        }


@dataclass
class KagomeOrderDiagnosis:
    kernel_name: str
    Q: np.ndarray
    coarse_label: str
    coarse_score: float
    paper_label: str
    paper_score: float
    recognition_status: str
    top_template_name: str
    top_template_score: float
    spin_sector: str
    channel_sector: str
    degeneracy: int
    internal_mode: InternalModeTensors
    template_scores: List[PaperTemplateScore]
    notes: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "kernel": self.kernel_name,
            "Q": np.asarray(self.Q, dtype=float).tolist(),
            "coarse_label": str(self.coarse_label),
            "coarse_score": float(self.coarse_score),
            "paper_label": str(self.paper_label),
            "paper_score": float(self.paper_score),
            "recognition_status": str(self.recognition_status),
            "top_template_name": str(self.top_template_name),
            "top_template_score": float(self.top_template_score),
            "spin_sector": str(self.spin_sector),
            "channel_sector": str(self.channel_sector),
            "degeneracy": int(self.degeneracy),
            "internal_mode": self.internal_mode.summary_dict(),
            "template_scores": [{"name": t.name, "score": float(t.score), **t.details} for t in self.template_scores],
            "notes": list(self.notes),
        }


class KagomeOrderDiagnoser:
    """
    Paper-oriented Kagome diagnosis layer.

    Role of form_factor.py
    ----------------------
    This class still *depends* on the coarse recognizer from form_factor.py
    (or order_recognition.py as a fallback). That earlier layer remains useful:
      1) it finds the leading eigenspace / degeneracy robustly,
      2) it gives generic symmetry labels (s, d, f, finite-Q ph, ...)
      3) it should remain model-agnostic.

    This file adds the next layer:
      - reconstruct the leading mode into internal-index tensors
            ph : Phi_{l n}(p;Q)
            pp : Delta_{m n}(p)
      - compare with Kagome-paper-specific templates
      - still keep an open 'unclassified / novel_candidate' exit
    """

    def __init__(
        self,
        patchset: Any,
        *,
        recognizer: Optional[OrderRecognizer] = None,
        q_zero_tol: float = 1e-8,
        min_paper_template_score: float = 0.12,
    ):
        self.patchset = patchset
        self.ks = _extract_patch_positions(patchset)
        self.eigvecs = _extract_patch_eigvecs(patchset)
        if self.eigvecs.ndim != 2:
            raise ValueError("Patch eigvec array must have shape (Npatch, Norb).")
        self.Npatch = int(self.ks.shape[0])
        self.Norb = int(self.eigvecs.shape[1]) if self.eigvecs.size else 0
        self.recognizer = recognizer if recognizer is not None else OrderRecognizer(patchset)
        self.q_zero_tol = float(q_zero_tol)
        self.min_paper_template_score = float(min_paper_template_score)
        self._pair_labels = {0: "A", 1: "B", 2: "C"}

    # ---------- generic helpers ----------
    def _channel_sector(self, kernel_name: str) -> str:
        return "pp" if kernel_name.startswith("pp") else "ph"

    def _spin_sector(self, kernel_name: str) -> str:
        name = kernel_name.lower()
        if "triplet" in name or "spin" in name or "fm" in name:
            return "spin/triplet"
        if "singlet" in name or "charge" in name:
            return "charge/singlet"
        return "unknown"

    def _orthonormalize(self, mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        mat = np.asarray(mat, dtype=complex)
        if mat.ndim == 1:
            mat = mat[:, None]
        if mat.size == 0:
            return np.zeros((mat.shape[0], 0), dtype=complex)
        q, _ = np.linalg.qr(mat)
        keep = [q[:, i] for i in range(q.shape[1]) if np.linalg.norm(q[:, i]) > tol]
        if not keep:
            return np.zeros((mat.shape[0], 0), dtype=complex)
        return np.column_stack(keep)

    def _flatten_tensor_basis(self, tensors: Sequence[np.ndarray]) -> np.ndarray:
        mats = [np.asarray(t, dtype=complex).reshape(-1) for t in tensors]
        return self._orthonormalize(np.column_stack(mats))

    def _pair_weight_summary(self, tensor_basis: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]], float, float]:
        if tensor_basis.ndim != 4:
            raise ValueError("tensor_basis must have shape (Npatch, Norb, Norb, dim).")
        weights = np.sum(np.abs(tensor_basis) ** 2, axis=(0, 3))  # (Norb, Norb)
        total = float(np.sum(weights))
        if total > 0:
            weights = weights / total
        dominant = None
        if weights.size > 0:
            dominant = tuple(int(x) for x in np.unravel_index(np.argmax(weights), weights.shape))
        same = float(np.trace(weights))
        inter = float(np.sum(weights) - same)
        return np.asarray(weights, dtype=float), dominant, same, inter

    # ---------- internal-index reconstruction ----------
    def reconstruct_mode_tensor(self, kernel: Any, mode: ModeAnalysis) -> InternalModeTensors:
        """
        Reconstruct the leading mode into tensors with explicit orbital/sublattice indices.

        ph  : Phi_{l n}(p;Q) ~ g(p) * u_l^*(p) * u_n(p_Q)
        pp  : Delta_{m n}(p) ~ g(p) * [u_m(p) u_n(p_partner) +/- u_n(p) u_m(p_partner)] / 2

        This uses the *column* patch basis convention to stay consistent with the
        channel-kernel definitions in channels.py.
        """
        basis = np.asarray(mode.basis_vectors, dtype=complex)
        partner = np.asarray(kernel.col_partner_patches, dtype=int)
        if partner.shape[0] != self.Npatch:
            raise ValueError("Kernel partner map is incompatible with patchset size.")
        if self.Norb == 0:
            raise ValueError("Patch eigvecs are required for kagome order diagnosis.")

        out = np.zeros((self.Npatch, self.Norb, self.Norb, basis.shape[1]), dtype=complex)
        channel = self._channel_sector(kernel.name)
        spin_sector = self._spin_sector(kernel.name)
        is_triplet = ("triplet" in kernel.name.lower()) or ("spin" in kernel.name.lower())

        for a in range(basis.shape[1]):
            g = basis[:, a]
            for p in range(self.Npatch):
                u = np.asarray(self.eigvecs[p], dtype=complex)
                v = np.asarray(self.eigvecs[partner[p]], dtype=complex)
                if channel == "ph":
                    out[p, :, :, a] = g[p] * np.outer(np.conjugate(u), v)
                else:
                    outer_uv = np.outer(u, v)
                    outer_vu = np.outer(v, u)
                    sign = +1.0 if is_triplet else -1.0
                    out[p, :, :, a] = 0.5 * g[p] * (outer_uv + sign * outer_vu)

        pair_weights, dominant_pair, same_w, inter_w = self._pair_weight_summary(out)
        return InternalModeTensors(
            channel_sector=channel,
            spin_sector=spin_sector,
            tensor_basis=out,
            pair_weights=pair_weights,
            dominant_pair=dominant_pair,
            same_sublattice_weight=float(same_w),
            inter_sublattice_weight=float(inter_w),
        )

    # ---------- paper-specific templates ----------
    def _template_tensors(self, kernel: Any) -> Dict[str, np.ndarray]:
        Q = np.asarray(kernel.Q, dtype=float)
        kx = self.ks[:, 0]
        ky = self.ks[:, 1]
        z = np.zeros((self.Npatch, self.Norb, self.Norb), dtype=complex)
        templates: Dict[str, np.ndarray] = {}

        # PI templates: diagonal / same-sublattice, Eq. (3) of the paper.
        f_dx2y2 = np.cos(2 * kx) - np.cos(kx) * np.cos(np.sqrt(3.0) * ky)
        f_dxy = np.sqrt(3.0) * np.sin(kx) * np.sin(np.sqrt(3.0) * ky)
        for name, f in [("PI_dx2y2", f_dx2y2), ("PI_dxy", f_dxy)]:
            t = np.zeros_like(z)
            for m in range(min(self.Norb, 3)):
                t[:, m, m] = f
            templates[name] = t

        # f-SC templates: Eq. (5) in the paper.
        f_ab = np.sin(1.5 * kx + 0.5 * np.sqrt(3.0) * ky)
        f_bc = np.sin(1.5 * kx - 0.5 * np.sqrt(3.0) * ky)
        f_ac = np.sin(np.sqrt(3.0) * ky)
        pair_specs = {
            "fSC_AB": (0, 1, f_ab),
            "fSC_BC": (1, 2, f_bc),
            "fSC_AC": (0, 2, f_ac),
        }
        for name, (i, j, f) in pair_specs.items():
            t = np.zeros_like(z)
            if self.Norb > max(i, j):
                t[:, i, j] = f
                t[:, j, i] = f
            templates[name] = t

        # finite-Q BO templates: one-Q components only. Full three-Q superposition is a later layer.
        qnorm = float(np.linalg.norm(Q))
        if qnorm > self.q_zero_tol:
            phase = self.ks @ Q
            f_q = np.sin(phase)
            Qs = [
                np.array([-0.5, -np.sqrt(3.0) / 2.0]),
                np.array([1.0, 0.0]),
                np.array([-0.5, np.sqrt(3.0) / 2.0]),
            ]
            idx = int(np.argmin([np.linalg.norm(Q - q0) for q0 in Qs]))
            pair_by_q = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
            i, j = pair_by_q[idx]
            t = np.zeros_like(z)
            if self.Norb > max(i, j):
                t[:, i, j] = f_q
                t[:, j, i] = f_q
            templates["BO_finiteQ"] = t

        # FM template: Q=0 diagonal constant spin-like ph order.
        t = np.zeros_like(z)
        for m in range(min(self.Norb, 3)):
            t[:, m, m] = 1.0
        templates["FM_constant"] = t
        return templates

    def _project_tensor_basis(self, tensor_basis: np.ndarray, templates: Mapping[str, np.ndarray]) -> List[PaperTemplateScore]:
        flat_basis = self._flatten_tensor_basis([tensor_basis[..., i] for i in range(tensor_basis.shape[-1])])
        scores: List[PaperTemplateScore] = []
        for name, tensor in templates.items():
            flat_t = self._flatten_tensor_basis([tensor])
            if flat_t.shape[1] == 0 or flat_basis.shape[1] == 0:
                score = 0.0
            else:
                overlap = flat_t.conjugate().T @ flat_basis
                denom = max(1, min(flat_t.shape[1], flat_basis.shape[1]))
                score = float(np.linalg.norm(overlap, ord="fro") ** 2 / denom)
            scores.append(PaperTemplateScore(name=name, score=score))
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores

    def _build_unclassified(
        self,
        mode: ModeAnalysis,
        top_template: PaperTemplateScore,
        *,
        reason: str,
        extra_notes: Optional[List[str]] = None,
    ) -> Tuple[str, float, str, List[str]]:
        notes = [reason]
        if extra_notes:
            notes.extend(extra_notes)
        notes.append(
            f"Coarse symmetry label retained as '{mode.dominant_label}' "
            f"with score {mode.dominant_score:.4f}; top paper template was "
            f"'{top_template.name}' with score {top_template.score:.4f}."
        )
        return "unclassified", 0.0, "novel_candidate", notes

    def _final_label(
        self,
        kernel: Any,
        mode: ModeAnalysis,
        internal_mode: InternalModeTensors,
        template_scores: List[PaperTemplateScore],
    ) -> Tuple[str, float, str, List[str]]:
        notes: List[str] = []
        top = template_scores[0] if template_scores else PaperTemplateScore("none", 0.0)
        name = kernel.name.lower()
        qnorm = float(np.linalg.norm(kernel.Q))
        pair_weights = internal_mode.pair_weights

        if qnorm < self.q_zero_tol and "spin" in name and mode.dominant_label == "s" and top.name == "FM_constant":
            if top.score >= self.min_paper_template_score:
                notes.append("Q≈0 spin-like particle-hole mode with strong constant diagonal signal.")
                return "FM", top.score, "recognized", notes
            return self._build_unclassified(mode, top, reason="FM-like kernel found, but paper-template support is too weak.")

        if qnorm < self.q_zero_tol and name.startswith("ph"):
            pi_score = max([t.score for t in template_scores if t.name in {"PI_dx2y2", "PI_dxy"}] + [0.0])
            if mode.dominant_label in {"d", "paper_d_Q0", "ph_nematic"} and pi_score > self.min_paper_template_score:
                notes.append("Q≈0 particle-hole d-wave-like mode with diagonal / same-sublattice dominance.")
                if mode.subspace_dim >= 2:
                    notes.append("Treat as twofold-degenerate d-wave PI subspace rather than one fixed basis vector.")
                return "PI", pi_score, "recognized", notes
            if mode.dominant_label in {"d", "paper_d_Q0", "ph_nematic"}:
                return self._build_unclassified(
                    mode, top,
                    reason="Q≈0 particle-hole d-wave-like mode detected, but support for kagome PI templates is insufficient."
                )

        if qnorm >= self.q_zero_tol and name.startswith("ph"):
            bo_score = max([t.score for t in template_scores if t.name == "BO_finiteQ"] + [0.0])
            offdiag_weight = float(np.sum(np.abs(pair_weights - np.diag(np.diag(pair_weights)))))
            if top.name == "BO_finiteQ" or (mode.dominant_label == "finiteQ_cos_sin" and offdiag_weight > np.trace(pair_weights)):
                if bo_score > self.min_paper_template_score:
                    if "spin" in name:
                        notes.append("Finite-Q spin-like off-diagonal bond-order template matched.")
                        notes.append("This is a one-Q component diagnosis; three-Q cBO/sBO superposition is a later layer.")
                        return "sBO", bo_score, "recognized", notes
                    if "charge" in name or "singlet" in name or "ph" in name:
                        notes.append("Finite-Q charge-like off-diagonal bond-order template matched.")
                        notes.append("This is a one-Q component diagnosis; three-Q cBO/sBO superposition is a later layer.")
                        return "cBO", bo_score, "recognized", notes
                return self._build_unclassified(
                    mode, top,
                    reason="Finite-Q particle-hole mode detected, but bond-order template support is not strong enough for cBO/sBO."
                )

        if name.startswith("pp"):
            fscore = max([t.score for t in template_scores if t.name.startswith("fSC_")] + [0.0])
            if ("triplet" in name or (mode.partner_parity is not None and mode.partner_parity < 0.0)) and fscore > self.min_paper_template_score:
                notes.append("Odd / triplet-like pp mode with strong f-wave kagome pair template overlap.")
                return "f-SC", fscore, "recognized", notes
            if mode.partner_parity is not None:
                return self._build_unclassified(
                    mode, top,
                    reason="Superconducting pp mode detected, but it does not match the kagome f-SC templates strongly enough."
                )

        return self._build_unclassified(
            mode,
            top,
            reason="No paper-specific template achieved strong enough support; keep this mode open as a possible new order.",
        )

    # ---------- public API ----------
    def leading_internal_mode(self, kernel: Any, *, sort_by: str = "abs") -> InternalModeTensors:
        analyses = self.recognizer.analyze_kernel(kernel, n_groups=1, sort_by=sort_by)
        if not analyses:
            raise ValueError("Recognizer returned no leading mode.")
        return self.reconstruct_mode_tensor(kernel, analyses[0])

    def diagnose_kernel(self, kernel: Any, *, sort_by: str = "abs") -> KagomeOrderDiagnosis:
        analyses = self.recognizer.analyze_kernel(kernel, n_groups=1, sort_by=sort_by)
        if not analyses:
            raise ValueError("Recognizer returned no leading mode.")
        mode = analyses[0]
        internal_mode = self.reconstruct_mode_tensor(kernel, mode)
        templates = self._template_tensors(kernel)
        template_scores = self._project_tensor_basis(internal_mode.tensor_basis, templates)
        paper_label, paper_score, recognition_status, notes = self._final_label(kernel, mode, internal_mode, template_scores)
        notes = list(mode.notes) + notes
        if internal_mode.dominant_pair is not None:
            i, j = internal_mode.dominant_pair
            lbl = (self._pair_labels.get(i, str(i)), self._pair_labels.get(j, str(j)))
            notes.append(f"Dominant reconstructed sublattice pair: {lbl[0]}-{lbl[1]}.")
        top_template = template_scores[0] if template_scores else PaperTemplateScore("none", 0.0)
        return KagomeOrderDiagnosis(
            kernel_name=kernel.name,
            Q=np.asarray(kernel.Q, dtype=float),
            coarse_label=mode.dominant_label,
            coarse_score=float(mode.dominant_score),
            paper_label=paper_label,
            paper_score=float(paper_score),
            recognition_status=recognition_status,
            top_template_name=top_template.name,
            top_template_score=float(top_template.score),
            spin_sector=self._spin_sector(kernel.name),
            channel_sector=self._channel_sector(kernel.name),
            degeneracy=int(mode.subspace_dim),
            internal_mode=internal_mode,
            template_scores=template_scores,
            notes=notes,
        )

    def diagnose_kernel_dict(self, kernels: Mapping[str, Any], *, sort_by: str = "abs") -> Dict[str, KagomeOrderDiagnosis]:
        return {name: self.diagnose_kernel(kernel, sort_by=sort_by) for name, kernel in kernels.items()}


__all__ = [
    "InternalModeTensors",
    "KagomeOrderDiagnosis",
    "KagomeOrderDiagnoser",
    "PaperTemplateScore",
]
