import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class FormFactorCandidate:
    name: str
    values: np.ndarray
    irrep: str
    parity: str
    family: str
    metadata: Optional[dict] = None

    def normalized_values(self) -> np.ndarray:
        v = np.asarray(self.values, dtype=complex).reshape(-1)
        nrm = np.linalg.norm(v)
        if nrm == 0:
            raise ValueError(f"Form factor {self.name!r} has zero norm.")
        return v / nrm


@dataclass
class ModeMatch:
    name: str
    overlap: complex
    weight: float
    irrep: str
    family: str
    parity: str


@dataclass
class KernelModeAnalysis:
    rank: int
    eigenvalue: complex
    mode: np.ndarray
    phase_fixed_mode: np.ndarray
    matches: List[ModeMatch]
    grouped_weights: Dict[str, float]
    best_name: str
    best_weight: float


@dataclass
class KernelFormFactorAnalysis:
    kernel_name: str
    Q: np.ndarray
    sort_by: str
    candidates: List[FormFactorCandidate]
    modes: List[KernelModeAnalysis]


class StandardKagomeFormFactorLibrary:
    """
    Build simple patch-space candidate form factors from kagome / triangular bond harmonics.

    Important note
    --------------
    This is a *recognition* library, not a mathematically complete irrep basis.
    It is meant to answer the practical question:
        "which familiar form-factor family does this kernel eigenmode most resemble?"

    The returned labels are therefore heuristic but physically useful, especially after
    your patch eigenvectors have already been gauge-smoothed.
    """

    def __init__(self, patchset, *, delta_vectors: Optional[Sequence[np.ndarray]] = None):
        self.patchset = patchset
        self.k = np.asarray([p.k_cart for p in patchset.patches], dtype=float)
        if self.k.ndim != 2 or self.k.shape[1] != 2:
            raise ValueError("patchset must contain 2D patch momenta.")

        if delta_vectors is None:
            delta_vectors = self._infer_bond_vectors_from_patchset(patchset)
        self.delta_vectors = [np.asarray(d, dtype=float) for d in delta_vectors]
        if len(self.delta_vectors) != 3:
            raise ValueError("delta_vectors must contain exactly three kagome NN bond vectors.")

    @staticmethod
    def _infer_direct_lattice_vectors_from_reciprocal(b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a_i · b_j = 2π δ_ij for the direct primitive vectors.
        """
        B = np.column_stack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)])
        A = 2.0 * np.pi * np.linalg.inv(B).T
        a1 = A[:, 0]
        a2 = A[:, 1]
        return a1, a2

    @classmethod
    def _infer_bond_vectors_from_patchset(cls, patchset) -> List[np.ndarray]:
        a1, a2 = cls._infer_direct_lattice_vectors_from_reciprocal(patchset.b1, patchset.b2)
        # kagome NN bonds inside one triangular Bravais cell
        delta1 = 0.5 * a1
        delta2 = 0.5 * a2
        delta3 = 0.5 * (a2 - a1)
        return [delta1, delta2, delta3]

    def _angles(self) -> np.ndarray:
        return np.arctan2([d[1] for d in self.delta_vectors], [d[0] for d in self.delta_vectors])

    def _cos_sum(self, weights: Sequence[complex]) -> np.ndarray:
        out = np.zeros(len(self.k), dtype=complex)
        for w, d in zip(weights, self.delta_vectors):
            out += complex(w) * np.cos(self.k @ d)
        return out

    def _sin_sum(self, weights: Sequence[complex]) -> np.ndarray:
        out = np.zeros(len(self.k), dtype=complex)
        for w, d in zip(weights, self.delta_vectors):
            out += complex(w) * np.sin(self.k @ d)
        return out

    def build(self, *, include_constant: bool = True, include_extended: bool = True) -> List[FormFactorCandidate]:
        th = self._angles()
        c2 = np.cos(2.0 * th)
        s2 = np.sin(2.0 * th)

        candidates: List[FormFactorCandidate] = []

        def add(name, values, irrep, parity, family, metadata=None):
            candidates.append(
                FormFactorCandidate(
                    name=name,
                    values=np.asarray(values, dtype=complex),
                    irrep=irrep,
                    parity=parity,
                    family=family,
                    metadata={} if metadata is None else dict(metadata),
                )
            )

        if include_constant:
            add("s_const", np.ones(len(self.k)), "A1", "even", "s")

        if include_extended:
            add("s_nn", self._cos_sum([1.0, 1.0, 1.0]), "A1", "even", "s")

        # 2D p-wave pair from odd bond harmonics projected onto x/y directions.
        bond_x = np.array([d[0] for d in self.delta_vectors], dtype=float)
        bond_y = np.array([d[1] for d in self.delta_vectors], dtype=float)
        add("p_x", self._sin_sum(bond_x), "E1", "odd", "p")
        add("p_y", self._sin_sum(bond_y), "E1", "odd", "p")

        # 2D d-wave pair from even bond harmonics with l=2 angular weights.
        add("d_x2_y2", self._cos_sum(c2), "E2", "even", "d")
        add("d_xy", self._cos_sum(s2), "E2", "even", "d")

        # 2D f-wave-like pair from odd bond harmonics with l=2 angular weights.
        add("f_x", self._sin_sum(c2), "E2", "odd", "f")
        add("f_y", self._sin_sum(s2), "E2", "odd", "f")

        return candidates


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("Encountered zero-norm vector.")
    return v / nrm


def fix_global_phase(v: np.ndarray) -> np.ndarray:
    """
    Fix a mode's overall U(1) phase so the largest-amplitude component becomes real positive.
    """
    v = _normalize_vector(v)
    idx = int(np.argmax(np.abs(v)))
    phase = np.exp(-1j * np.angle(v[idx]))
    return v * phase


def orthonormalize_candidates(candidates: Sequence[FormFactorCandidate]) -> List[FormFactorCandidate]:
    """
    Modified Gram-Schmidt. Keeps labels but replaces values by an orthonormalized version.
    This avoids double-counting overlap between, e.g., s_const and s_nn.
    """
    out: List[FormFactorCandidate] = []
    vecs: List[np.ndarray] = []
    for cand in candidates:
        v = np.asarray(cand.values, dtype=complex).reshape(-1).copy()
        for q in vecs:
            v = v - np.vdot(q, v) * q
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            continue
        q = v / nrm
        vecs.append(q)
        out.append(
            FormFactorCandidate(
                name=cand.name,
                values=q,
                irrep=cand.irrep,
                parity=cand.parity,
                family=cand.family,
                metadata=cand.metadata,
            )
        )
    return out


class FormFactorAnalyzer:
    def __init__(self, patchset, candidates: Sequence[FormFactorCandidate]):
        self.patchset = patchset
        self.Npatch = int(patchset.Npatch)
        self.candidates = orthonormalize_candidates(candidates)
        for cand in self.candidates:
            if np.asarray(cand.values).shape != (self.Npatch,):
                raise ValueError(
                    f"Candidate {cand.name!r} has shape {np.asarray(cand.values).shape}, "
                    f"expected ({self.Npatch},)."
                )

    @classmethod
    def from_standard_kagome(
        cls,
        patchset,
        *,
        model=None,
        include_constant: bool = True,
        include_extended: bool = True,
    ):
        delta_vectors = None
        if model is not None and all(hasattr(model, name) for name in ("delta1", "delta2", "delta3")):
            delta_vectors = [model.delta1, model.delta2, model.delta3]
        lib = StandardKagomeFormFactorLibrary(
            patchset,
            delta_vectors=delta_vectors,
        )
        candidates = lib.build(include_constant=include_constant, include_extended=include_extended)
        return cls(patchset, candidates)

    def analyze_mode(self, mode: np.ndarray, *, rank: int = 1, eigenvalue: complex = np.nan) -> KernelModeAnalysis:
        mode = fix_global_phase(mode)
        matches: List[ModeMatch] = []
        grouped: Dict[str, float] = {}

        for cand in self.candidates:
            basis = cand.normalized_values()
            ov = np.vdot(basis, mode)
            wt = float(np.abs(ov) ** 2)
            matches.append(
                ModeMatch(
                    name=cand.name,
                    overlap=ov,
                    weight=wt,
                    irrep=cand.irrep,
                    family=cand.family,
                    parity=cand.parity,
                )
            )
            grouped[cand.family] = grouped.get(cand.family, 0.0) + wt

        matches.sort(key=lambda x: x.weight, reverse=True)
        best = matches[0]
        return KernelModeAnalysis(
            rank=int(rank),
            eigenvalue=eigenvalue,
            mode=_normalize_vector(mode),
            phase_fixed_mode=mode,
            matches=matches,
            grouped_weights=dict(sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)),
            best_name=best.name,
            best_weight=best.weight,
        )

    def analyze_kernel(
        self,
        kernel,
        *,
        top_n: int = 6,
        sort_by: str = "abs",
        hermitize_if_close: bool = True,
        hermitian_tol: float = 1e-9,
    ) -> KernelFormFactorAnalysis:
        M = np.asarray(kernel.matrix, dtype=complex)
        if M.shape != (self.Npatch, self.Npatch):
            raise ValueError(
                f"Kernel matrix has shape {M.shape}, but patchset has Npatch={self.Npatch}."
            )

        if hermitize_if_close:
            resid = np.max(np.abs(M - M.conjugate().T))
            if resid < hermitian_tol:
                evals, evecs = np.linalg.eigh(0.5 * (M + M.conjugate().T))
                if sort_by == "abs":
                    order = np.argsort(-np.abs(evals))
                elif sort_by == "real":
                    order = np.argsort(-np.real(evals))
                elif sort_by == "most_negative_real":
                    order = np.argsort(np.real(evals))
                else:
                    raise ValueError("sort_by must be 'abs', 'real', or 'most_negative_real'.")
                evals, evecs = evals[order], evecs[:, order]
            else:
                evals, evecs = np.linalg.eig(M)
                if sort_by == "abs":
                    order = np.argsort(-np.abs(evals))
                elif sort_by == "real":
                    order = np.argsort(-np.real(evals))
                elif sort_by == "most_negative_real":
                    order = np.argsort(np.real(evals))
                else:
                    raise ValueError("sort_by must be 'abs', 'real', or 'most_negative_real'.")
                evals, evecs = evals[order], evecs[:, order]
        else:
            evals, evecs = np.linalg.eig(M)
            if sort_by == "abs":
                order = np.argsort(-np.abs(evals))
            elif sort_by == "real":
                order = np.argsort(-np.real(evals))
            elif sort_by == "most_negative_real":
                order = np.argsort(np.real(evals))
            else:
                raise ValueError("sort_by must be 'abs', 'real', or 'most_negative_real'.")
            evals, evecs = evals[order], evecs[:, order]

        modes: List[KernelModeAnalysis] = []
        for i in range(min(top_n, len(evals))):
            modes.append(self.analyze_mode(evecs[:, i], rank=i + 1, eigenvalue=evals[i]))

        return KernelFormFactorAnalysis(
            kernel_name=getattr(kernel, "name", "unknown_kernel"),
            Q=np.asarray(getattr(kernel, "Q", np.zeros(2)), dtype=float),
            sort_by=sort_by,
            candidates=self.candidates,
            modes=modes,
        )


def summarize_analysis(analysis: KernelFormFactorAnalysis, *, top_matches: int = 4) -> str:
    lines: List[str] = []
    lines.append(f"Kernel = {analysis.kernel_name}, Q = {np.array2string(analysis.Q, precision=6)}")
    lines.append(f"sorted by = {analysis.sort_by}")
    for mode in analysis.modes:
        lines.append(
            f"mode #{mode.rank}: eigenvalue = {mode.eigenvalue.real:+.6e}"
            + (f" {mode.eigenvalue.imag:+.6e}i" if abs(mode.eigenvalue.imag) > 1e-12 else "")
        )
        lines.append(
            "  grouped weights: "
            + ", ".join([f"{k}={v:.3f}" for k, v in mode.grouped_weights.items()])
        )
        best = mode.matches[:top_matches]
        lines.append(
            "  top matches: "
            + ", ".join([f"{m.name} ({m.weight:.3f})" for m in best])
        )
    return "\n".join(lines)


def plot_mode_on_fs(
    patchset,
    mode: np.ndarray,
    *,
    ax=None,
    component: str = "real",
    cmap: str = "coolwarm",
    s: int = 55,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    mode = fix_global_phase(mode)
    k = np.asarray([p.k_cart for p in patchset.patches], dtype=float)

    if component == "real":
        c = np.real(mode)
        label = r"Re $f(k)$"
    elif component == "imag":
        c = np.imag(mode)
        label = r"Im $f(k)$"
    elif component == "abs":
        c = np.abs(mode)
        label = r"$|f(k)|$"
    elif component == "phase":
        c = np.angle(mode)
        label = r"arg $f(k)$"
        cmap = "twilight"
    else:
        raise ValueError("component must be one of {'real', 'imag', 'abs', 'phase'}.")

    sc = ax.scatter(k[:, 0], k[:, 1], c=c, s=s, cmap=cmap)
    ax.plot(patchset.fs_contour_k[:, 0], patchset.fs_contour_k[:, 1], alpha=0.25)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(title or f"mode on FS ({component})")
    plt.colorbar(sc, ax=ax, label=label)
    return ax


def print_top_matches(mode_analysis: KernelModeAnalysis, *, top_matches: int = 6) -> None:
    print(f"rank = {mode_analysis.rank}")
    print(f"eigenvalue = {mode_analysis.eigenvalue}")
    print("grouped weights =", mode_analysis.grouped_weights)
    for m in mode_analysis.matches[:top_matches]:
        print(
            f"  {m.name:10s}  family={m.family:>2s}  irrep={m.irrep:>2s}  "
            f"parity={m.parity:>4s}  weight={m.weight:.6f}  overlap={m.overlap}"
        )
