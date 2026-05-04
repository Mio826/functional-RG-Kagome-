from __future__ import annotations

"""Unified Level-3 order-parameter fingerprint diagnostics for the kagome fRG pipeline.

This file combines the previously separate modules:

* order_parameter_diagnosis_q0.py
* order_parameter_diagnosis_qm.py
* order_parameter_diagnosis_qm_current.py

Conceptual scope
----------------
This module is a Level-3 post-processing / fingerprint module. It takes scalar
patch-space fRG eigenvectors and reconstructs onsite, bond, current, and finite-Q
orbital fingerprints. It does not compute RG flows, susceptibilities, or true
many-body expectation values.

Main sections
-------------
1. Q=0 PH fingerprints: FM, PI scalar fingerprints, Nagaosa/FlowA bond-current
   textures.
2. Q=M real-order fingerprints: strict retained-partner CDW/SDW/CBO/SBO
   diagnostics.
3. Q=M current proxy diagnostics: LC_M_D6A, LC_M_D6B, LC_M_D6C, LC_M_D6PA,
   including Gram/SVD/subspace analysis because these proxy templates are not
   generally orthogonal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Q=0 fingerprints
# =============================================================================

"""Level-3 Q=0 order-parameter fingerprint diagnostics for the kagome fRG pipeline.

This module implements Level-3 order-parameter fingerprint diagnostics. It
reconstructs real/orbital-space patterns associated with a scalar fRG eigenmode.
It does not compute independent order-parameter susceptibilities or replace
scalar candidate overlap when such overlap is well-defined.

Scope of this first file
------------------------
Only Q=0 particle-hole fingerprints are implemented here:

* FM-like onsite spin/charge pattern from the lifted orbital density matrix.
* PI/d-wave momentum-space scalar fingerprint. PI is treated as a Fermi-surface
  deformation, not as a fake real-space current pattern.
* Q=0 bond/current fingerprints, including Nagaosa and Flow-A current-pattern
  scores, from the same oriented-bond bilinear

      R_l = t_l sum_i w_i exp(i k_i dot delta_l) rho_i[a,b].

The real bond and current components are only the real and imaginary parts of
this same R_l:

      bond_l    = 2 Re R_l,
      current_l = 2 Im R_l.

Interpretation warning
----------------------
The input vector v_i is an fRG eigenvector direction, not an actual many-body
expectation value <gamma_k^dagger gamma_k>. All amplitudes are normalization-
and phase-convention dependent. Use these results as pattern diagnostics only.
"""



ArrayLike = Sequence[float] | np.ndarray

A, B, C = 0, 1, 2
ORBITAL_LABELS = ("A", "B", "C")


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class BondSpec:
    """One oriented real-space bond used for Q=0 bond/current fingerprints.

    Parameters
    ----------
    name:
        Human-readable label, e.g. "AB+".
    a, b:
        Orbital indices in rho[a,b]. The bond is interpreted as
        a -> b + delta.
    delta:
        Real-space displacement vector from orbital a to orbital b in the
        chosen unit-cell convention.
    hopping:
        The physical hopping amplitude t_l for this oriented bond. Do not add
        an extra imaginary source by hand; pass the hopping used by the actual
        Hamiltonian convention being diagnosed.
    pair:
        Optional pair label such as "AB", "AC", or "BC". If omitted it is
        inferred from a,b.
    target_signs:
        Optional named target signs for pattern scoring. For example
        {"nagaosa": +1, "flowA": +1}.
    """

    name: str
    a: int
    b: int
    delta: np.ndarray
    hopping: complex
    pair: Optional[str] = None
    target_signs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", np.asarray(self.delta, dtype=float))
        if self.delta.shape != (2,):
            raise ValueError(f"BondSpec.delta must have shape (2,), got {self.delta.shape}")
        if self.pair is None:
            object.__setattr__(self, "pair", f"{ORBITAL_LABELS[int(self.a)]}{ORBITAL_LABELS[int(self.b)]}")


@dataclass
class Q0FingerprintResult:
    """Container for one Q=0 Level-3 fingerprint result."""

    pattern_name: str
    channel: str
    site: pd.DataFrame = field(default_factory=pd.DataFrame)
    bonds: pd.DataFrame = field(default_factory=pd.DataFrame)
    scalar: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        if len(self.site):
            for _, r in self.site.iterrows():
                rows.append({
                    "pattern_name": self.pattern_name,
                    "channel": self.channel,
                    "kind": "site",
                    "name": r.get("orbital", "site"),
                    "value_real": float(r.get("value_real", np.nan)),
                    "value_imag": float(r.get("value_imag", np.nan)),
                    "abs": float(r.get("abs", np.nan)),
                })
        if len(self.bonds):
            for _, r in self.bonds.iterrows():
                rows.append({
                    "pattern_name": self.pattern_name,
                    "channel": self.channel,
                    "kind": "bond",
                    "name": r.get("bond", "bond"),
                    "value_real": float(r.get("bond_real", np.nan)),
                    "value_imag": float(r.get("current", np.nan)),
                    "abs": float(abs(r.get("R", 0.0))),
                })
        if len(self.scalar):
            for _, r in self.scalar.iterrows():
                rows.append({
                    "pattern_name": self.pattern_name,
                    "channel": self.channel,
                    "kind": "scalar",
                    "name": r.get("component", "scalar"),
                    "value_real": float(r.get("value_real", np.nan)),
                    "value_imag": float(r.get("value_imag", np.nan)),
                    "abs": float(r.get("abs", np.nan)),
                })
        return pd.DataFrame(rows)


# =============================================================================
# Generic helpers
# =============================================================================

def _require_patchset_q0(patchset: Any) -> None:
    for attr in ("Npatch", "patch_k", "patch_eigvec"):
        if not hasattr(patchset, attr):
            raise TypeError(f"patchset is missing required attribute {attr!r}")


def as_weights(weights: Optional[ArrayLike], n: int, *, normalize_sum: bool = False) -> np.ndarray:
    if weights is None:
        w = np.ones(int(n), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != int(n):
            raise ValueError(f"weights length must be {n}, got {w.size}")
    w = w.copy()
    w[~np.isfinite(w)] = 0.0
    if normalize_sum:
        s = float(np.sum(w))
        if abs(s) > 0:
            w = w / s
    return w


def prepare_mode_vector(
    v: ArrayLike,
    *,
    normalize_v: bool = False,
    phase_fix: str = "none",
    tol: float = 1e-30,
) -> np.ndarray:
    """Return a cleaned scalar patch vector.

    phase_fix options:
      - "none": no global phase change.
      - "max_real": make the largest-magnitude component real positive.
      - "sum_real": make sum(v) real positive if possible.
    """
    out = np.asarray(v, dtype=complex).reshape(-1).copy()
    out[~np.isfinite(out)] = 0.0

    if phase_fix == "none":
        pass
    elif phase_fix == "max_real":
        if out.size:
            idx = int(np.argmax(np.abs(out)))
            if abs(out[idx]) > tol:
                out *= np.exp(-1j * np.angle(out[idx]))
    elif phase_fix == "sum_real":
        s = np.sum(out)
        if abs(s) > tol:
            out *= np.exp(-1j * np.angle(s))
    else:
        raise ValueError("phase_fix must be one of {'none','max_real','sum_real'}")

    if normalize_v:
        nrm = float(np.linalg.norm(out))
        if nrm > tol:
            out = out / nrm
    return out


def _safe_complex_inner_weighted(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> complex:
    return complex(np.sum(np.asarray(weights, dtype=float) * np.conjugate(a) * b))


def _weighted_norm(a: np.ndarray, weights: np.ndarray, tol: float = 1e-30) -> float:
    val = np.real(_safe_complex_inner_weighted(a, a, weights))
    return float(np.sqrt(max(val, 0.0))) if val > tol else 0.0


def weighted_projection(v: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> complex:
    """Weighted projection coefficient <basis|v>_w / <basis|basis>_w."""
    denom = _safe_complex_inner_weighted(basis, basis, weights)
    if abs(denom) <= 1e-30:
        return 0.0 + 0.0j
    return _safe_complex_inner_weighted(basis, v, weights) / denom


def weighted_abs_overlap(v: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> float:
    """Cosine-like weighted absolute overlap between two patch vectors."""
    nv = _weighted_norm(v, weights)
    nb = _weighted_norm(basis, weights)
    if nv <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(abs(_safe_complex_inner_weighted(basis, v, weights)) / (nv * nb))


# =============================================================================
# Q=0 orbital reconstruction
# =============================================================================

def reconstruct_orbital_ph_pattern_q0(
    v: ArrayLike,
    patchset: Any,
    weights: Optional[ArrayLike] = None,
    *,
    normalize_v: bool = True,
    phase_fix: str = "none",
) -> Dict[str, Any]:
    """Lift a scalar Q=0 PH eigenvector to orbital bilinears.

    Computes

        rho_i[a,b] = v_i conj(u_a(k_i)) u_b(k_i).

    weights are returned for downstream sums but are not absorbed into rho.

    If normalize_v=True, v is normalized by the weighted norm

        sqrt(sum_i w_i |v_i|^2).
    """
    _require_patchset_q0(patchset)
    n = int(patchset.Npatch)

    ww = as_weights(weights, n)

    # First do phase fixing / basic conversion using existing helper.
    # Keep normalize_v=False here because we want weighted normalization below.
    vv = prepare_mode_vector(v, normalize_v=False, phase_fix=phase_fix)

    if vv.size != n:
        raise ValueError(f"v length must equal patchset.Npatch={n}, got {vv.size}")

    # Weighted normalization: sum_i w_i |v_i|^2 = 1
    if normalize_v:
        norm = np.sqrt(np.sum(ww * np.abs(vv) ** 2))
        if norm > 0:
            vv = vv / norm

    u = np.asarray(patchset.patch_eigvec, dtype=complex)
    if u.ndim != 2 or u.shape[0] != n:
        raise ValueError(f"patchset.patch_eigvec must have shape (Npatch,Norb), got {u.shape}")

    rho = vv[:, None, None] * np.conjugate(u[:, :, None]) * u[:, None, :]

    return {
        "rho": rho,
        "v": vv,
        "u": u,
        "patch_k": np.asarray(patchset.patch_k, dtype=float),
        "weights": ww,
        "metadata": {
            "Npatch": n,
            "Norb": int(u.shape[1]),
            "normalize_v": bool(normalize_v),
            "phase_fix": str(phase_fix),
            "normalization_note": "v normalized by sqrt(sum_i w_i |v_i|^2); Level-3 amplitudes remain fingerprint diagnostics.",
        },
    }

def compute_site_pattern_q0(
    rho: np.ndarray,
    weights: Optional[ArrayLike] = None,
    *,
    channel: str = "spin",
) -> pd.DataFrame:
    """Compute Q=0 onsite orbital pattern sum_i w_i rho_i[a,a]."""
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 3 or rho.shape[1] != rho.shape[2]:
        raise ValueError("rho must have shape (Npatch,Norb,Norb)")
    n, norb, _ = rho.shape
    w = as_weights(weights, n)
    vals = np.einsum("i,iaa->a", w, rho)

    rows = []
    for a in range(norb):
        label = ORBITAL_LABELS[a] if a < len(ORBITAL_LABELS) else f"orb{a}"
        rows.append({
            "channel": str(channel),
            "orbital": label,
            "orbital_index": int(a),
            "value": vals[a],
            "value_real": float(np.real(vals[a])),
            "value_imag": float(np.imag(vals[a])),
            "abs": float(abs(vals[a])),
        })
    total = np.sum(vals)
    rows.append({
        "channel": str(channel),
        "orbital": "total",
        "orbital_index": -1,
        "value": total,
        "value_real": float(np.real(total)),
        "value_imag": float(np.imag(total)),
        "abs": float(abs(total)),
    })
    return pd.DataFrame(rows)


def compute_q0_bond_bilinears(
    rho: np.ndarray,
    patch_k: np.ndarray,
    bonds: Sequence[BondSpec],
    weights: Optional[ArrayLike] = None,
) -> pd.DataFrame:
    """Compute Q=0 oriented-bond bilinears, real bond, and current.

    For each bond ell=(a -> b + delta):

        R_ell = t_ell sum_i w_i exp(i k_i dot delta_ell) rho_i[a,b]

    Then:

        bond_real = 2 Re R_ell,
        current   = 2 Im R_ell.
    """
    rho = np.asarray(rho, dtype=complex)
    patch_k = np.asarray(patch_k, dtype=float)
    if rho.ndim != 3:
        raise ValueError("rho must have shape (Npatch,Norb,Norb)")
    n, norb, _ = rho.shape
    if patch_k.shape != (n, 2):
        raise ValueError(f"patch_k must have shape ({n},2), got {patch_k.shape}")
    w = as_weights(weights, n)

    rows = []
    for bond in bonds:
        a, b = int(bond.a), int(bond.b)
        if not (0 <= a < norb and 0 <= b < norb):
            raise ValueError(f"bond {bond.name!r} has invalid orbital indices {(a,b)} for Norb={norb}")
        phase = np.exp(1j * (patch_k @ np.asarray(bond.delta, dtype=float)))
        raw_sum = np.sum(w * phase * rho[:, a, b])
        R = complex(bond.hopping) * raw_sum
        row = {
            "bond": bond.name,
            "pair": bond.pair,
            "a": a,
            "b": b,
            "a_label": ORBITAL_LABELS[a] if a < len(ORBITAL_LABELS) else str(a),
            "b_label": ORBITAL_LABELS[b] if b < len(ORBITAL_LABELS) else str(b),
            "delta_x": float(bond.delta[0]),
            "delta_y": float(bond.delta[1]),
            "hopping": complex(bond.hopping),
            "raw_sum": complex(raw_sum),
            "R": R,
            "R_real": float(np.real(R)),
            "R_imag": float(np.imag(R)),
            "bond_real": float(2.0 * np.real(R)),
            "current": float(2.0 * np.imag(R)),
        }
        for key, val in dict(bond.target_signs).items():
            row[f"target_{key}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Q=0 FM / PI / current diagnostics
# =============================================================================

def diagnose_q0_fm_fingerprint(
    v: ArrayLike,
    patchset: Any,
    weights: Optional[ArrayLike] = None,
    *,
    channel: str = "spin",
    normalize_v: bool = False,
    phase_fix: str = "sum_real",
    pattern_name: str = "FM_Q0_level3",
) -> Q0FingerprintResult:
    pack = reconstruct_orbital_ph_pattern_q0(
        v, patchset, weights=weights, normalize_v=normalize_v, phase_fix=phase_fix
    )
    site = compute_site_pattern_q0(pack["rho"], pack["weights"], channel=channel)
    return Q0FingerprintResult(
        pattern_name=pattern_name,
        channel=channel,
        site=site,
        metadata={**pack["metadata"], "interpretation": "onsite orbital spin/charge fingerprint"},
    )


def q0_pi_basis_vectors(patch_k: np.ndarray) -> Dict[str, np.ndarray]:
    """Return PI/d-wave scalar basis vectors used in candidate_diagnosis.py plus angular harmonics."""
    k = np.asarray(patch_k, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2:
        raise ValueError("patch_k must have shape (Npatch,2)")
    theta = np.arctan2(k[:, 1], k[:, 0])
    return {
        "PI_dx2y2_candidate": np.asarray(np.cos(2.0 * k[:, 0]) - np.cos(k[:, 0]) * np.cos(np.sqrt(3.0) * k[:, 1]), dtype=complex),
        "PI_dxy_candidate": np.asarray(np.sqrt(3.0) * np.sin(k[:, 0]) * np.sin(np.sqrt(3.0) * k[:, 1]), dtype=complex),
        "angular_cos2theta": np.asarray(np.cos(2.0 * theta), dtype=complex),
        "angular_sin2theta": np.asarray(np.sin(2.0 * theta), dtype=complex),
        "constant": np.ones(k.shape[0], dtype=complex),
    }


def compute_q0_scalar_harmonics(
    v: ArrayLike,
    patchset: Any,
    weights: Optional[ArrayLike] = None,
    *,
    normalize_v: bool = False,
    phase_fix: str = "none",
) -> pd.DataFrame:
    _require_patchset_q0(patchset)
    n = int(patchset.Npatch)
    vv = prepare_mode_vector(v, normalize_v=normalize_v, phase_fix=phase_fix)
    if vv.size != n:
        raise ValueError(f"v length must equal patchset.Npatch={n}, got {vv.size}")
    w = as_weights(weights, n)
    basis = q0_pi_basis_vectors(np.asarray(patchset.patch_k, dtype=float))
    rows = []
    for name, b in basis.items():
        coeff = weighted_projection(vv, b, w)
        ov = weighted_abs_overlap(vv, b, w)
        rows.append({
            "component": name,
            "coefficient": coeff,
            "value": coeff,
            "value_real": float(np.real(coeff)),
            "value_imag": float(np.imag(coeff)),
            "abs": float(abs(coeff)),
            "abs_overlap": float(ov),
            "overlap_sq": float(ov * ov),
            "note": "momentum-space scalar fingerprint; not a real-space current",
        })
    return pd.DataFrame(rows)


def patch_scalar_dataframe(v: ArrayLike, patchset: Any, weights: Optional[ArrayLike] = None) -> pd.DataFrame:
    _require_patchset_q0(patchset)
    n = int(patchset.Npatch)
    vv = np.asarray(v, dtype=complex).reshape(-1)
    if vv.size != n:
        raise ValueError(f"v length must equal patchset.Npatch={n}, got {vv.size}")
    k = np.asarray(patchset.patch_k, dtype=float)
    w = as_weights(weights, n)
    return pd.DataFrame({
        "patch": np.arange(n, dtype=int),
        "kx": k[:, 0],
        "ky": k[:, 1],
        "weight": w,
        "v": vv,
        "v_real": np.real(vv),
        "v_imag": np.imag(vv),
        "v_abs": np.abs(vv),
        "theta": np.arctan2(k[:, 1], k[:, 0]),
    })


def diagnose_q0_pi_fingerprint(
    v: ArrayLike,
    patchset: Any,
    weights: Optional[ArrayLike] = None,
    *,
    normalize_v: bool = False,
    phase_fix: str = "none",
    pattern_name: str = "PI_Q0_level3",
) -> Q0FingerprintResult:
    scalar = compute_q0_scalar_harmonics(
        v, patchset, weights=weights, normalize_v=normalize_v, phase_fix=phase_fix
    )
    return Q0FingerprintResult(
        pattern_name=pattern_name,
        channel="charge",
        scalar=scalar,
        metadata={
            "Npatch": int(patchset.Npatch),
            "normalize_v": bool(normalize_v),
            "phase_fix": str(phase_fix),
            "interpretation": "PI/d-wave Fermi-surface deformation scalar fingerprint",
            "warning": "PI is not diagnosed here as a real-space current or bond texture.",
        },
    )


def score_bond_pattern(
    bond_df: pd.DataFrame,
    target_signs: Mapping[str, float] | Sequence[float],
    *,
    value_col: str = "current",
    target_name: str = "target",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Score a bond/current vector against a target sign vector.

    The returned score is a pattern diagnostic only. It is not a susceptibility.
    """
    x = np.asarray(bond_df[value_col].to_numpy(dtype=float), dtype=float)
    if isinstance(target_signs, Mapping):
        eta = np.asarray([float(target_signs.get(str(b), np.nan)) for b in bond_df["bond"]], dtype=float)
    else:
        eta = np.asarray(list(target_signs), dtype=float)
        if eta.size != x.size:
            raise ValueError(f"target_signs length must be {x.size}, got {eta.size}")

    valid = np.isfinite(x) & np.isfinite(eta)
    raw_signed_sum = float(np.sum(eta[valid] * x[valid])) if np.any(valid) else np.nan
    norm_x = float(np.linalg.norm(x[valid])) if np.any(valid) else 0.0
    norm_eta = float(np.linalg.norm(eta[valid])) if np.any(valid) else 0.0
    cosine = raw_signed_sum / (norm_x * norm_eta) if (norm_x > 0 and norm_eta > 0) else np.nan
    if not normalize:
        cosine = raw_signed_sum
    return {
        "target_name": str(target_name),
        "value_col": str(value_col),
        "n_valid_bonds": int(np.count_nonzero(valid)),
        "raw_signed_sum": raw_signed_sum,
        "cosine_similarity": float(cosine) if np.isfinite(cosine) else np.nan,
        "norm_value": norm_x,
        "norm_target": norm_eta,
        "interpretation": "bond/current pattern diagnostic only; not an LC susceptibility",
    }


def diagnose_q0_current_fingerprint(
    v: ArrayLike,
    patchset: Any,
    bonds: Sequence[BondSpec],
    weights: Optional[ArrayLike] = None,
    *,
    channel: str = "charge",
    pattern: Optional[str] = None,
    target_signs: Optional[Mapping[str, float] | Sequence[float]] = None,
    normalize_v: bool = False,
    phase_fix: str = "none",
    pattern_name: Optional[str] = None,
) -> Q0FingerprintResult:
    """Compute Q=0 bond/current Level-3 fingerprint from a scalar eigenmode."""
    pack = reconstruct_orbital_ph_pattern_q0(
        v, patchset, weights=weights, normalize_v=normalize_v, phase_fix=phase_fix
    )
    bonds_df = compute_q0_bond_bilinears(pack["rho"], pack["patch_k"], bonds, weights=pack["weights"])
    site = compute_site_pattern_q0(pack["rho"], pack["weights"], channel=channel)

    scalar_rows: List[Dict[str, Any]] = []
    if target_signs is None and pattern is not None:
        # Try to read target signs from BondSpec.target_signs.
        key = pattern.lower()
        extracted = {}
        for b in bonds:
            if key in b.target_signs:
                extracted[b.name] = float(b.target_signs[key])
        if len(extracted):
            target_signs = extracted
    if target_signs is not None:
        for col in ("current", "bond_real"):
            score = score_bond_pattern(
                bonds_df, target_signs, value_col=col, target_name=str(pattern or "custom")
            )
            scalar_rows.append({
                "component": f"{score['target_name']}_{col}_pattern_cosine",
                "value": score["cosine_similarity"],
                "value_real": score["cosine_similarity"],
                "value_imag": 0.0,
                "abs": abs(score["cosine_similarity"]) if np.isfinite(score["cosine_similarity"]) else np.nan,
                **score,
            })
            scalar_rows.append({
                "component": f"{score['target_name']}_{col}_signed_sum",
                "value": score["raw_signed_sum"],
                "value_real": score["raw_signed_sum"],
                "value_imag": 0.0,
                "abs": abs(score["raw_signed_sum"]) if np.isfinite(score["raw_signed_sum"]) else np.nan,
                **score,
            })
    scalar = pd.DataFrame(scalar_rows)

    return Q0FingerprintResult(
        pattern_name=pattern_name or (f"{pattern}_Q0_current_level3" if pattern else "Q0_bond_current_level3"),
        channel=channel,
        site=site,
        bonds=bonds_df,
        scalar=scalar,
        metadata={
            **pack["metadata"],
            "pattern": pattern,
            "interpretation": "Q=0 bond/current texture from lifted orbital bilinears",
            "warning": "current score is a pattern diagnostic only, not an LC susceptibility or order magnitude.",
        },
    )


def diagnose_q0_order_fingerprint(
    v: ArrayLike,
    patchset: Any,
    bonds: Optional[Sequence[BondSpec]] = None,
    weights: Optional[ArrayLike] = None,
    *,
    order: str = "fm",
    channel: str = "spin",
    pattern: Optional[str] = None,
    target_signs: Optional[Mapping[str, float] | Sequence[float]] = None,
    normalize_v: bool = False,
    phase_fix: str = "none",
) -> Q0FingerprintResult:
    """Small dispatcher for Q=0 Level-3 fingerprints."""
    key = str(order).lower()
    if key in {"fm", "onsite"}:
        return diagnose_q0_fm_fingerprint(
            v, patchset, weights=weights, channel=channel, normalize_v=normalize_v, phase_fix=phase_fix
        )
    if key in {"pi", "pomeranchuk", "d-wave", "dwave"}:
        return diagnose_q0_pi_fingerprint(
            v, patchset, weights=weights, normalize_v=normalize_v, phase_fix=phase_fix
        )
    if key in {"current", "lc", "bond", "bond_current"}:
        if bonds is None:
            raise ValueError("bonds must be provided for Q=0 bond/current fingerprints")
        return diagnose_q0_current_fingerprint(
            v,
            patchset,
            bonds,
            weights=weights,
            channel=channel,
            pattern=pattern,
            target_signs=target_signs,
            normalize_v=normalize_v,
            phase_fix=phase_fix,
        )
    raise ValueError(f"Unknown Q=0 order={order!r}")


# =============================================================================
# Kagome default bond conventions and Q=0 target patterns
# =============================================================================

def q0_current_target_signs(pattern: str, bonds: Sequence[BondSpec]) -> Dict[str, float]:
    """Return target current signs for known Q=0 patterns on the supplied bonds.

    Conventions match the Q=0 current candidate signs used in candidate_diagnosis:
      Nagaosa: eta_AB=+1, eta_AC=-1, eta_BC=+1
      FlowA:   eta_AB=+1, eta_AC=+1, eta_BC=+1

    The same pair sign is assigned to all oriented bonds with that pair label.
    """
    key = str(pattern).lower().replace("-", "_")
    if key in {"nagaosa", "lc_q0_nagaosa"}:
        pair_eta = {"AB": +1.0, "AC": -1.0, "BC": +1.0}
    elif key in {"flowa", "flow_a", "lc_q0_flowa"}:
        pair_eta = {"AB": +1.0, "AC": +1.0, "BC": +1.0}
    else:
        raise ValueError("pattern must be one of {'nagaosa','flowA'}")
    return {b.name: float(pair_eta.get(str(b.pair), np.nan)) for b in bonds}


def default_kagome_nn_bonds(
    model: Any,
    *,
    hopping: Optional[complex] = None,
    include_reverse: bool = False,
    pattern_signs: bool = True,
) -> List[BondSpec]:
    """Build a simple set of oriented NN kagome bonds from model.delta1/2/3.

    The default three bonds use the same pair/displacement convention as many
    kagome Hamiltonians in this project:

      AB: delta1, AC: -delta2, BC: delta3.

    If include_reverse=True, the reverse-oriented Hermitian partners are also
    included with conjugate hopping.

    By default this function uses a real NN hopping of -t. If you diagnose a
    flux/SOC Hamiltonian, pass the actual oriented hopping values manually via
    custom BondSpec objects. Do not use this helper if its hopping convention
    does not match your Hamiltonian.
    """
    for attr in ("delta1", "delta2", "delta3"):
        if not hasattr(model, attr):
            raise AttributeError(f"model must have {attr} to build default kagome bonds")

    tval = None
    if hopping is not None:
        tval = complex(hopping)
    elif hasattr(model, "parameters") and isinstance(model.parameters, Mapping) and "t" in model.parameters:
        tval = -complex(model.parameters["t"])
    else:
        tval = -1.0 + 0.0j

    base = [
        ("AB+", A, B, np.asarray(model.delta1, dtype=float), tval, "AB"),
        ("AC-", A, C, -np.asarray(model.delta2, dtype=float), tval, "AC"),
        ("BC+", B, C, np.asarray(model.delta3, dtype=float), tval, "BC"),
    ]
    signs = {
        "AB": {"nagaosa": +1.0, "flowa": +1.0},
        "AC": {"nagaosa": -1.0, "flowa": +1.0},
        "BC": {"nagaosa": +1.0, "flowa": +1.0},
    }
    bonds: List[BondSpec] = []
    for name, a, b, d, hop, pair in base:
        bonds.append(BondSpec(name, a, b, d, hop, pair=pair, target_signs=signs[pair] if pattern_signs else {}))
        if include_reverse:
            bonds.append(BondSpec(name + "_rev", b, a, -d, np.conjugate(hop), pair=pair, target_signs=signs[pair] if pattern_signs else {}))
    return bonds


def q0_bond_kernel_for_mock(
    patchset: Any,
    bond: BondSpec,
    weights: Optional[ArrayLike] = None,
    *,
    include_hopping: bool = True,
) -> np.ndarray:
    """Kernel K_i for the linear bond bilinear R_l[v] = sum_i w_i v_i K_i.

    K_i = t_l exp(i k_i dot delta_l) conj(u_a(k_i)) u_b(k_i)
    if include_hopping=True. This is useful for mock validation tests.
    """
    _require_patchset_q0(patchset)
    k = np.asarray(patchset.patch_k, dtype=float)
    u = np.asarray(patchset.patch_eigvec, dtype=complex)
    a, b = int(bond.a), int(bond.b)
    K = np.exp(1j * (k @ bond.delta)) * np.conjugate(u[:, a]) * u[:, b]
    if include_hopping:
        K = complex(bond.hopping) * K
    return np.asarray(K, dtype=complex)


def q0_current_pattern_kernel_for_mock(
    patchset: Any,
    bonds: Sequence[BondSpec],
    target_signs: Mapping[str, float] | Sequence[float],
    *,
    include_hopping: bool = True,
) -> np.ndarray:
    """Combined kernel sum_l eta_l K_l for mock current-pattern tests."""
    if isinstance(target_signs, Mapping):
        etas = [float(target_signs[b.name]) for b in bonds]
    else:
        etas = list(map(float, target_signs))
        if len(etas) != len(bonds):
            raise ValueError("target_signs length must match bonds")
    out = np.zeros(int(patchset.Npatch), dtype=complex)
    for eta, b in zip(etas, bonds):
        out = out + eta * q0_bond_kernel_for_mock(patchset, b, include_hopping=include_hopping)
    return out


# =============================================================================
# Dataframe / plotting helpers
# =============================================================================

def fingerprint_to_dataframe(result: Q0FingerprintResult) -> pd.DataFrame:
    return result.summary_frame()


def plot_q0_bond_fingerprint(
    bond_df: pd.DataFrame,
    *,
    value_col: str = "current",
    ax: Optional[Any] = None,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    vals = bond_df[value_col].to_numpy(dtype=float)
    labels = bond_df["bond"].astype(str).tolist()
    ax.bar(labels, vals)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"Q=0 bond fingerprint: {value_col}")
    ax.tick_params(axis="x", rotation=25)
    return ax


def plot_q0_patch_scalar_fingerprint(
    patch_df: pd.DataFrame,
    *,
    value_col: str = "v_real",
    ax: Optional[Any] = None,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(patch_df["kx"], patch_df["ky"], c=patch_df[value_col], s=60)
    ax.set_aspect("equal")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(title or f"patch scalar fingerprint: {value_col}")
    plt.colorbar(sc, ax=ax)
    return ax


# =============================================================================
# Q=M real-order fingerprints
# =============================================================================

"""Level-3 Q=M order-parameter fingerprint diagnostics for kagome fRG.

This module implements Q=M Level-3 order-parameter fingerprint diagnostics.
It reconstructs real/orbital-space finite-Q patterns associated with a scalar
fRG eigenmode. It does not compute independent order-parameter susceptibilities,
new RG eigenvalues, or replace scalar candidate overlap when such overlap is
well-defined.

Important convention
--------------------
For a scalar particle-hole mode v_i at momentum transfer Q, the orbital lift is

    rho_i[a,b] = v_i * conj(u_a(k_i + Q)) * u_b(k_i)

but only for patches where k_i + Q is represented by another retained patch
modulo reciprocal lattice vectors within a strict tolerance. Invalid rows are
stored as NaN and ignored in masked sums.
"""



try:
    from frg_kernel import canonicalize_q_for_patchsets
except Exception:  # pragma: no cover
    canonicalize_q_for_patchsets = None

ArrayLike = Sequence[complex] | np.ndarray

A, B, C = 0, 1, 2
ORBITAL_NAMES = ("A", "B", "C")
QM_PAIR_MAP: Dict[int, Tuple[int, int]] = {
    0: (B, C),
    1: (A, C),
    2: (A, B),
}


@dataclass(frozen=True)
class QMPartnerMap:
    Q: np.ndarray
    partner_indices: np.ndarray
    residuals: np.ndarray
    valid_mask: np.ndarray
    tol: float
    direction: str = "k_plus_Q"

    @property
    def Npatch(self) -> int:
        return int(self.partner_indices.size)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def valid_fraction(self) -> float:
        return float(self.n_valid / max(self.Npatch, 1))

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "patch": np.arange(self.Npatch, dtype=int),
            "partner": self.partner_indices.astype(int),
            "residual": self.residuals.astype(float),
            "valid": self.valid_mask.astype(bool),
        })


@dataclass
class QMOrbitalPattern:
    rho: np.ndarray
    v: np.ndarray
    u_k: np.ndarray
    u_kq: np.ndarray
    patch_k: np.ndarray
    weights: np.ndarray
    partner: QMPartnerMap
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QMRealOrderResult:
    pattern_name: str
    channel: str
    Q: np.ndarray
    m_index: int
    density: pd.DataFrame
    bond: pd.DataFrame
    scalar: Dict[str, Any]
    partner: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, val in self.scalar.items():
            if np.isscalar(val) or isinstance(val, (str, bool, int, float, complex)):
                rows.append({"quantity": key, "value": val})
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _require_patchset(patchset: Any) -> None:
    for attr in ("Npatch", "patch_k", "patch_eigvec", "b1", "b2"):
        if not hasattr(patchset, attr):
            raise TypeError(f"patchset is missing required attribute {attr!r}")


def as_weights(weights: Optional[ArrayLike], n: int) -> np.ndarray:
    if weights is None:
        return np.ones(int(n), dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != int(n):
        raise ValueError(f"weights length must be {n}, got {w.size}")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    return w


def prepare_mode_vector(
    v: ArrayLike,
    *,
    weights: Optional[ArrayLike] = None,
    normalize_v: bool = True,
    phase_fix: str = "none",
) -> np.ndarray:
    vv = np.asarray(v, dtype=complex).reshape(-1).copy()

    if phase_fix == "none":
        pass
    elif phase_fix == "sum_real":
        z = np.sum(vv)
        if abs(z) > 0:
            vv *= np.exp(-1j * np.angle(z))
    elif phase_fix == "max_component":
        if vv.size:
            idx = int(np.argmax(np.abs(vv)))
            if abs(vv[idx]) > 0:
                vv *= np.exp(-1j * np.angle(vv[idx]))
    else:
        raise ValueError("phase_fix must be 'none', 'sum_real', or 'max_component'")

    if normalize_v:
        if weights is None:
            norm = np.sqrt(np.sum(np.abs(vv) ** 2))
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != vv.size:
                raise ValueError("weights length must match v length for normalization")
            norm = np.sqrt(np.sum(w * np.abs(vv) ** 2))
        if norm > 0:
            vv = vv / norm
    return vv


def _canonicalize_q(patchset: Any, q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(2)
    if canonicalize_q_for_patchsets is not None:
        try:
            return np.asarray(canonicalize_q_for_patchsets({"up": patchset}, q), dtype=float)
        except Exception:
            pass
    Bmat = np.column_stack([np.asarray(patchset.b1, dtype=float), np.asarray(patchset.b2, dtype=float)])
    uv = np.linalg.solve(Bmat, q)
    uv = uv - np.floor(uv)
    uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
    uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
    out = Bmat @ uv
    out[np.isclose(out, 0.0, atol=1e-12)] = 0.0
    return out


def minimum_image_displacement(k_target: ArrayLike, k_ref: ArrayLike, b1: ArrayLike, b2: ArrayLike, search_range: int = 1) -> np.ndarray:
    k_target = np.asarray(k_target, dtype=float)
    k_ref = np.asarray(k_ref, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    best = None
    best_norm = np.inf
    for n1 in range(-search_range, search_range + 1):
        for n2 in range(-search_range, search_range + 1):
            disp = k_target - (k_ref + n1 * b1 + n2 * b2)
            nd = float(np.linalg.norm(disp))
            if nd < best_norm:
                best_norm = nd
                best = disp
    return np.asarray(best, dtype=float)


def periodic_distance(k_target: ArrayLike, k_ref: ArrayLike, b1: ArrayLike, b2: ArrayLike, search_range: int = 1) -> float:
    return float(np.linalg.norm(minimum_image_displacement(k_target, k_ref, b1, b2, search_range=search_range)))


# -----------------------------------------------------------------------------
# Strict Q=M partner map and orbital lift
# -----------------------------------------------------------------------------

def partner_indices_from_Q_masked(
    patchset: Any,
    Q: ArrayLike,
    *,
    tol: float = 1e-6,
    direction: str = "k_plus_Q",
    search_range: int = 1,
) -> QMPartnerMap:
    """Strictly find retained-patch partners for k -> k+Q or k-Q.

    This deliberately does NOT use nearest-patch approximation as a valid result.
    The nearest retained patch is found only to compute a residual; the row is
    valid iff residual <= tol.
    """
    _require_patchset(patchset)
    if direction not in {"k_plus_Q", "k_minus_Q"}:
        raise ValueError("direction must be 'k_plus_Q' or 'k_minus_Q'")

    n = int(patchset.Npatch)
    ks = np.asarray(patchset.patch_k, dtype=float)
    b1 = np.asarray(patchset.b1, dtype=float)
    b2 = np.asarray(patchset.b2, dtype=float)
    Qc = _canonicalize_q(patchset, Q)

    partner = np.full(n, -1, dtype=int)
    residuals = np.full(n, np.inf, dtype=float)

    sign = +1.0 if direction == "k_plus_Q" else -1.0
    for i, k in enumerate(ks):
        target = k + sign * Qc
        best_j = -1
        best_d = np.inf
        for j, kj in enumerate(ks):
            d = periodic_distance(target, kj, b1, b2, search_range=search_range)
            if d < best_d - 1e-14 or (abs(d - best_d) <= 1e-14 and (best_j < 0 or j < best_j)):
                best_d = d
                best_j = j
        partner[i] = int(best_j)
        residuals[i] = float(best_d)

    valid = residuals <= float(tol)
    partner_strict = partner.copy()
    partner_strict[~valid] = -1

    return QMPartnerMap(
        Q=np.asarray(Qc, dtype=float),
        partner_indices=partner_strict,
        residuals=residuals,
        valid_mask=valid,
        tol=float(tol),
        direction=direction,
    )


def reconstruct_orbital_ph_pattern_qm_masked(
    v: ArrayLike,
    patchset: Any,
    Q: ArrayLike,
    weights: Optional[ArrayLike] = None,
    *,
    partner_map: Optional[QMPartnerMap] = None,
    partner_indices: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    normalize_v: bool = True,
    phase_fix: str = "none",
) -> QMOrbitalPattern:
    """Lift scalar finite-Q PH eigenvector to masked orbital bilinears.

    Invalid rows are filled with NaN and are ignored by downstream masked sums.
    """
    _require_patchset(patchset)
    n = int(patchset.Npatch)
    ww = as_weights(weights, n)
    vv = prepare_mode_vector(v, weights=ww, normalize_v=normalize_v, phase_fix=phase_fix)
    if vv.size != n:
        raise ValueError(f"v length must equal patchset.Npatch={n}, got {vv.size}")

    u = np.asarray(patchset.patch_eigvec, dtype=complex)
    if u.ndim != 2 or u.shape[0] != n:
        raise ValueError(f"patchset.patch_eigvec must have shape (Npatch,Norb), got {u.shape}")
    norb = int(u.shape[1])

    if partner_map is None:
        if partner_indices is not None and valid_mask is not None:
            Qc = _canonicalize_q(patchset, Q)
            partner_map = QMPartnerMap(
                Q=Qc,
                partner_indices=np.asarray(partner_indices, dtype=int),
                residuals=np.full(n, np.nan, dtype=float),
                valid_mask=np.asarray(valid_mask, dtype=bool),
                tol=float(tol),
            )
        else:
            partner_map = partner_indices_from_Q_masked(patchset, Q, tol=tol)

    if partner_map.partner_indices.size != n or partner_map.valid_mask.size != n:
        raise ValueError("partner map has wrong length")

    u_kq = np.full_like(u, np.nan + 1j * np.nan)
    valid = np.asarray(partner_map.valid_mask, dtype=bool)
    pidx = np.asarray(partner_map.partner_indices, dtype=int)
    if np.any(valid):
        u_kq[valid] = u[pidx[valid]]

    rho = np.full((n, norb, norb), np.nan + 1j * np.nan, dtype=complex)
    if np.any(valid):
        rho[valid] = vv[valid, None, None] * np.conjugate(u_kq[valid, :, None]) * u[valid, None, :]

    return QMOrbitalPattern(
        rho=rho,
        v=vv,
        u_k=u,
        u_kq=u_kq,
        patch_k=np.asarray(patchset.patch_k, dtype=float),
        weights=ww,
        partner=partner_map,
        metadata={
            "Npatch": n,
            "Norb": norb,
            "Q": np.asarray(partner_map.Q, dtype=float),
            "tol": float(partner_map.tol),
            "n_valid": int(partner_map.n_valid),
            "valid_fraction": float(partner_map.valid_fraction),
            "normalize_v": bool(normalize_v),
            "phase_fix": str(phase_fix),
            "normalization_note": "v normalized by sqrt(sum_i w_i |v_i|^2) if normalize_v=True; Level-3 amplitudes are pattern diagnostics.",
        },
    )


# -----------------------------------------------------------------------------
# Q=M real-order matrices and kernels
# -----------------------------------------------------------------------------

def qm_pair(m_index: int) -> Tuple[int, int]:
    if int(m_index) not in QM_PAIR_MAP:
        raise ValueError("m_index must be 0, 1, or 2")
    return QM_PAIR_MAP[int(m_index)]


def qm_density_matrix(m_index: int) -> np.ndarray:
    i, j = qm_pair(m_index)
    D = np.zeros((3, 3), dtype=complex)
    D[i, i] = 1.0
    D[j, j] = -1.0
    return D


def qm_bond_matrix(m_index: int) -> np.ndarray:
    i, j = qm_pair(m_index)
    Bm = np.zeros((3, 3), dtype=complex)
    Bm[i, j] = 1.0
    Bm[j, i] = 1.0
    return Bm


def _trace_M_rho(M: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Tr[M rho_i] = sum_ab M_ab rho_i[ba]
    return np.einsum("ab,iba->i", np.asarray(M, dtype=complex), np.asarray(rho, dtype=complex))


def qm_density_kernel_from_pattern(pattern: QMOrbitalPattern, m_index: int) -> np.ndarray:
    D = qm_density_matrix(m_index)
    h = _trace_M_rho(D, pattern.rho)
    h[~pattern.partner.valid_mask] = np.nan + 1j * np.nan
    return h


def qm_bond_kernel_from_pattern(
    pattern: QMOrbitalPattern,
    m_index: int,
    *,
    use_candidate_phase: bool = False,
) -> np.ndarray:
    """Return Q=M bond kernel from a lifted orbital pattern.

    Raw Level-3 bond fingerprint uses Tr[B_m rho_i].  If
    use_candidate_phase=True, multiply by sin(Q·k_i); that is a
    Level-2 candidate-style convention and is intentionally not the default.
    """
    Bm = qm_bond_matrix(m_index)
    h = _trace_M_rho(Bm, pattern.rho)
    if use_candidate_phase:
        phase = np.sin(np.asarray(pattern.patch_k, dtype=float) @ np.asarray(pattern.partner.Q, dtype=float))
        h = phase * h
    h[~pattern.partner.valid_mask] = np.nan + 1j * np.nan
    return h


def masked_weighted_sum(values: np.ndarray, weights: np.ndarray, valid_mask: np.ndarray) -> complex:
    values = np.asarray(values, dtype=complex).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.any(valid):
        return np.nan + 1j * np.nan
    return complex(np.sum(weights[valid] * values[valid]))


def kernel_for_mock_density_qm(
    patchset: Any,
    Q: ArrayLike,
    *,
    m_index: int,
    weights: Optional[ArrayLike] = None,
    partner_map: Optional[QMPartnerMap] = None,
    tol: float = 1e-6,
    normalize_kernel: bool = False,
) -> Tuple[np.ndarray, QMPartnerMap]:
    """Return h_i = Tr[D_m u(k+Q)^† u(k)] on valid rows for mock tests."""
    n = int(patchset.Npatch)
    dummy = np.ones(n, dtype=complex)
    pat = reconstruct_orbital_ph_pattern_qm_masked(dummy, patchset, Q, weights=weights, partner_map=partner_map, tol=tol, normalize_v=False)
    h = qm_density_kernel_from_pattern(pat, m_index)
    h[~pat.partner.valid_mask] = 0.0
    if normalize_kernel:
        ww = as_weights(weights, n)
        norm = np.sqrt(np.sum(ww[pat.partner.valid_mask] * np.abs(h[pat.partner.valid_mask]) ** 2))
        if norm > 0:
            h = h / norm
    return h, pat.partner


def kernel_for_mock_bond_qm(
    patchset: Any,
    Q: ArrayLike,
    *,
    m_index: int,
    weights: Optional[ArrayLike] = None,
    partner_map: Optional[QMPartnerMap] = None,
    tol: float = 1e-6,
    use_candidate_phase: bool = False,
    normalize_kernel: bool = False,
) -> Tuple[np.ndarray, QMPartnerMap]:
    """Return Q=M bond kernel h_i on valid rows for mock tests.

    Raw Level-3 uses h_i = Tr[B_m u(k+Q)^† u(k)].
    If use_candidate_phase=True, multiply by sin(Q·k_i), which is the
    Level-2 candidate convention rather than the default raw fingerprint.
    """
    n = int(patchset.Npatch)
    dummy = np.ones(n, dtype=complex)
    pat = reconstruct_orbital_ph_pattern_qm_masked(dummy, patchset, Q, weights=weights, partner_map=partner_map, tol=tol, normalize_v=False)
    h = qm_bond_kernel_from_pattern(pat, m_index, use_candidate_phase=use_candidate_phase)
    h[~pat.partner.valid_mask] = 0.0
    if normalize_kernel:
        ww = as_weights(weights, n)
        norm = np.sqrt(np.sum(ww[pat.partner.valid_mask] * np.abs(h[pat.partner.valid_mask]) ** 2))
        if norm > 0:
            h = h / norm
    return h, pat.partner


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def _site_density_dataframe(pattern: QMOrbitalPattern, m_index: int) -> pd.DataFrame:
    D = qm_density_matrix(m_index)
    vals = np.diag(D).astype(complex)
    rows = []
    for a, name in enumerate(ORBITAL_NAMES):
        contrib_i = pattern.rho[:, a, a]
        amp = masked_weighted_sum(vals[a] * contrib_i, pattern.weights, pattern.partner.valid_mask)
        rows.append({
            "orbital": name,
            "orbital_index": a,
            "D_weight": vals[a],
            "amplitude": amp,
            "real": float(np.real(amp)) if np.isfinite(np.real(amp)) else np.nan,
            "imag": float(np.imag(amp)) if np.isfinite(np.imag(amp)) else np.nan,
        })
    return pd.DataFrame(rows)


def _bond_dataframe(pattern: QMOrbitalPattern, m_index: int, *, use_candidate_phase: bool) -> pd.DataFrame:
    i, j = qm_pair(m_index)
    Bm = qm_bond_matrix(m_index)
    h_raw = _trace_M_rho(Bm, pattern.rho)
    phase = np.sin(np.asarray(pattern.patch_k, dtype=float) @ np.asarray(pattern.partner.Q, dtype=float))
    h = phase * h_raw if use_candidate_phase else h_raw
    amp = masked_weighted_sum(h, pattern.weights, pattern.partner.valid_mask)
    return pd.DataFrame([{
        "m_index": int(m_index),
        "pair": f"{ORBITAL_NAMES[i]}{ORBITAL_NAMES[j]}",
        "a": int(i),
        "b": int(j),
        "use_candidate_phase": bool(use_candidate_phase),
        "amplitude": amp,
        "real": float(np.real(amp)) if np.isfinite(np.real(amp)) else np.nan,
        "imag": float(np.imag(amp)) if np.isfinite(np.imag(amp)) else np.nan,
        "bond_order_realpart": float(2 * np.real(amp)) if np.isfinite(np.real(amp)) else np.nan,
    }])


def diagnose_qm_real_orders_masked(
    v: ArrayLike,
    patchset: Any,
    Q: ArrayLike,
    weights: Optional[ArrayLike] = None,
    *,
    m_index: int = 0,
    channel: str = "charge",
    partner_tol: float = 1e-6,
    use_candidate_phase: bool = False,
    normalize_v: bool = True,
    phase_fix: str = "none",
    partner_map: Optional[QMPartnerMap] = None,
    pattern_name: Optional[str] = None,
) -> QMRealOrderResult:
    """Diagnose Q=M CDW/SDW/CBO/SBO Level-3 fingerprints.

    channel is metadata only: use 'charge' for CDW/CBO, 'spin' for SDW/SBO.
    The same scalar finite-Q bilinear is used; physical interpretation is set by
    the channel from which the input eigenvector came.

    By default, the bond fingerprint is raw Level-3: Tr[B_m rho_i].
    Set use_candidate_phase=True only when you intentionally want the
    Level-2 candidate convention sin(Q·k_i) Tr[B_m rho_i].
    """
    if channel not in {"charge", "spin"}:
        raise ValueError("channel must be 'charge' or 'spin'")
    pat = reconstruct_orbital_ph_pattern_qm_masked(
        v,
        patchset,
        Q,
        weights=weights,
        partner_map=partner_map,
        tol=partner_tol,
        normalize_v=normalize_v,
        phase_fix=phase_fix,
    )
    D = qm_density_matrix(m_index)
    Bm = qm_bond_matrix(m_index)
    hD = _trace_M_rho(D, pat.rho)
    hB_raw = _trace_M_rho(Bm, pat.rho)
    phase = np.sin(np.asarray(pat.patch_k, dtype=float) @ np.asarray(pat.partner.Q, dtype=float))
    hB = phase * hB_raw if use_candidate_phase else hB_raw

    ampD = masked_weighted_sum(hD, pat.weights, pat.partner.valid_mask)
    ampB = masked_weighted_sum(hB, pat.weights, pat.partner.valid_mask)

    density_df = _site_density_dataframe(pat, m_index)
    bond_df = _bond_dataframe(pat, m_index, use_candidate_phase=use_candidate_phase)

    if pattern_name is None:
        pattern_name = f"QM_{channel}_m{int(m_index)}"
    scalar = {
        "density_amplitude": ampD,
        "density_real": float(np.real(ampD)) if np.isfinite(np.real(ampD)) else np.nan,
        "density_imag": float(np.imag(ampD)) if np.isfinite(np.imag(ampD)) else np.nan,
        "bond_amplitude": ampB,
        "bond_real": float(np.real(ampB)) if np.isfinite(np.real(ampB)) else np.nan,
        "bond_imag": float(np.imag(ampB)) if np.isfinite(np.imag(ampB)) else np.nan,
        "bond_order_realpart": float(2 * np.real(ampB)) if np.isfinite(np.real(ampB)) else np.nan,
        "n_valid": int(pat.partner.n_valid),
        "Npatch": int(pat.partner.Npatch),
        "valid_fraction": float(pat.partner.valid_fraction),
        "partner_tol": float(pat.partner.tol),
        "use_candidate_phase": bool(use_candidate_phase),
    }

    return QMRealOrderResult(
        pattern_name=str(pattern_name),
        channel=str(channel),
        Q=np.asarray(pat.partner.Q, dtype=float),
        m_index=int(m_index),
        density=density_df,
        bond=bond_df,
        scalar=scalar,
        partner=pat.partner.dataframe(),
        metadata={
            **pat.metadata,
            "channel": str(channel),
            "m_index": int(m_index),
            "pair": "".join(ORBITAL_NAMES[x] for x in qm_pair(m_index)),
            "level3_warning": "Pattern diagnostic only; not an independent susceptibility or true order-parameter magnitude.",
        },
    )


def diagnose_all_qm_real_orders_masked(
    v_charge: Optional[ArrayLike],
    v_spin: Optional[ArrayLike],
    patchset: Any,
    Q: ArrayLike,
    weights: Optional[ArrayLike] = None,
    *,
    m_index: int = 0,
    partner_tol: float = 1e-6,
    normalize_v: bool = True,
    phase_fix: str = "none",
    use_candidate_phase: bool = False,
) -> Dict[str, QMRealOrderResult]:
    """Convenience wrapper for CDW, CBO, SDW, SBO at one Q and m_index."""
    pmap = partner_indices_from_Q_masked(patchset, Q, tol=partner_tol)
    out: Dict[str, QMRealOrderResult] = {}
    if v_charge is not None:
        out["CDW_M"] = diagnose_qm_real_orders_masked(v_charge, patchset, Q, weights, m_index=m_index, channel="charge", partner_map=pmap, partner_tol=partner_tol, normalize_v=normalize_v, phase_fix=phase_fix, use_candidate_phase=use_candidate_phase, pattern_name="CDW_M")
        out["CBO_M"] = diagnose_qm_real_orders_masked(v_charge, patchset, Q, weights, m_index=m_index, channel="charge", partner_map=pmap, partner_tol=partner_tol, normalize_v=normalize_v, phase_fix=phase_fix, use_candidate_phase=use_candidate_phase, pattern_name="CBO_M")
    if v_spin is not None:
        out["SDW_M"] = diagnose_qm_real_orders_masked(v_spin, patchset, Q, weights, m_index=m_index, channel="spin", partner_map=pmap, partner_tol=partner_tol, normalize_v=normalize_v, phase_fix=phase_fix, use_candidate_phase=use_candidate_phase, pattern_name="SDW_M")
        out["SBO_M"] = diagnose_qm_real_orders_masked(v_spin, patchset, Q, weights, m_index=m_index, channel="spin", partner_map=pmap, partner_tol=partner_tol, normalize_v=normalize_v, phase_fix=phase_fix, use_candidate_phase=use_candidate_phase, pattern_name="SBO_M")
    return out


# -----------------------------------------------------------------------------
# Display / plotting helpers
# -----------------------------------------------------------------------------

def results_summary_dataframe(results: Mapping[str, QMRealOrderResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            "name": name,
            "channel": res.channel,
            "m_index": res.m_index,
            "pair": res.metadata.get("pair"),
            "density_real": res.scalar.get("density_real"),
            "density_abs": abs(res.scalar.get("density_amplitude", np.nan)),
            "bond_real": res.scalar.get("bond_real"),
            "bond_abs": abs(res.scalar.get("bond_amplitude", np.nan)),
            "bond_order_realpart": res.scalar.get("bond_order_realpart"),
            "n_valid": res.scalar.get("n_valid"),
            "Npatch": res.scalar.get("Npatch"),
            "valid_fraction": res.scalar.get("valid_fraction"),
            "use_candidate_phase": res.scalar.get("use_candidate_phase"),
        })
    return pd.DataFrame(rows)


def partner_mask_dataframe(patchset: Any, partner: QMPartnerMap) -> pd.DataFrame:
    ks = np.asarray(patchset.patch_k, dtype=float)
    df = partner.dataframe()
    df["kx"] = ks[:, 0]
    df["ky"] = ks[:, 1]
    return df


def plot_partner_mask(patchset: Any, partner: QMPartnerMap, *, ax=None, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    df = partner_mask_dataframe(patchset, partner)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    valid = df["valid"].to_numpy(dtype=bool)
    ax.scatter(df.loc[~valid, "kx"], df.loc[~valid, "ky"], s=35, marker="x", label="invalid")
    ax.scatter(df.loc[valid, "kx"], df.loc[valid, "ky"], s=45, marker="o", label="valid")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(title or f"Q=M partner mask: n_valid={partner.n_valid}/{partner.Npatch}")
    ax.legend()
    return ax


def plot_partner_residuals(partner: QMPartnerMap, *, ax=None, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(partner.Npatch)
    ax.semilogy(x, partner.residuals, marker="o", linestyle="none")
    ax.axhline(partner.tol, linestyle="--", linewidth=1)
    ax.set_xlabel("patch index")
    ax.set_ylabel("periodic residual")
    ax.set_title(title or "Q=M partner residuals")
    return ax


__all__ = [
    "QMPartnerMap", "QMOrbitalPattern", "QMRealOrderResult",
    "QM_PAIR_MAP", "ORBITAL_NAMES",
    "as_weights", "prepare_mode_vector",
    "partner_indices_from_Q_masked", "reconstruct_orbital_ph_pattern_qm_masked",
    "qm_pair", "qm_density_matrix", "qm_bond_matrix",
    "qm_density_kernel_from_pattern", "qm_bond_kernel_from_pattern",
    "kernel_for_mock_density_qm", "kernel_for_mock_bond_qm",
    "diagnose_qm_real_orders_masked", "diagnose_all_qm_real_orders_masked",
    "results_summary_dataframe", "partner_mask_dataframe", "plot_partner_mask", "plot_partner_residuals",
]


# =============================================================================
# Q=M current proxy fingerprints
# =============================================================================

"""Level-3 / candidate-style Q=M current proxy diagnostics for kagome fRG.

This module handles the four Q=M high-symmetry current proxy candidates
used in candidate_diagnosis.py:

    LC_M_D6A, LC_M_D6B, LC_M_D6C, LC_M_D6PA.

Important interpretation
------------------------
These are proxy fingerprints for 2x2 flux classes. They are not a literal
bond-by-bond reconstruction of every arrow in Fig. 6 of the flux-classification
paper, and they are not independent susceptibilities or RG eigenvalues.

For a scalar PH mode v_i at transfer Q, the proxy kernel is

    T_i^alpha(Q) = sum_{cell c, bond b}
        s_c^alpha eta_b^alpha exp(i Q.R_c)
        <u(k_i+Q)| J_b |u(k_i)>,

where J_b is the orbital current matrix on AB, AC, or BC. The diagnostic
projects v onto these kernels as a pattern fingerprint.

Two modes are supported:
    mode='direct': evaluate u(k_i+Q) by direct local-block diagonalization.
                   This mirrors candidate_diagnosis.py.
    mode='strict': use only retained patch partners k_j = k_i + Q from a strict
                   partner map. Invalid patches are masked. This is stricter but
                   may miss candidate_diagnosis-style information.
"""



# In the combined module, these names are defined above in the Q=M real-order section.

ArrayLike = Sequence[complex] | np.ndarray
A, B, C = 0, 1, 2
ORBITAL_NAMES = ("A", "B", "C")


@dataclass(frozen=True)
class QMCurrentProxySpec:
    name: str
    builder_key: str
    cell_signs: np.ndarray
    bond_signs: np.ndarray
    notes: str = ""


@dataclass
class QMCurrentProxyResult:
    Q: np.ndarray
    mode: str
    pattern_name: str
    v: np.ndarray
    weights: np.ndarray
    templates: Dict[str, np.ndarray]
    scores: pd.DataFrame
    per_patch: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dataframe(self) -> pd.DataFrame:
        return self.scores.copy()


# Exactly match candidate_diagnosis.py proxy sign convention.
QM_CLASS_PROXY: Dict[str, Dict[str, np.ndarray]] = {
    "lc_m_d6a":  {"cell_signs": np.array([+1, +1, +1, +1], dtype=float), "bond_signs": np.array([+1, +1, +1], dtype=float)},
    "lc_m_d6b":  {"cell_signs": np.array([+1, -1, -1, -1], dtype=float), "bond_signs": np.array([+1, +1, +1], dtype=float)},
    "lc_m_d6c":  {"cell_signs": np.array([+1, +1, -1, +1], dtype=float), "bond_signs": np.array([+1, -1, +1], dtype=float)},
    "lc_m_d6pa": {"cell_signs": np.array([+1, -1, +1, -1], dtype=float), "bond_signs": np.array([+1, -1, -1], dtype=float)},
}

PUBLIC_TO_KEY = {
    "LC_M_D6A": "lc_m_d6a",
    "LC_M_D6B": "lc_m_d6b",
    "LC_M_D6C": "lc_m_d6c",
    "LC_M_D6PA": "lc_m_d6pa",
    "LC-M-D6a": "lc_m_d6a",
    "LC-M-D6b": "lc_m_d6b",
    "LC-M-D6c": "lc_m_d6c",
    "LC-M-D6'a": "lc_m_d6pa",
}


def qm_current_proxy_specs() -> Dict[str, QMCurrentProxySpec]:
    out: Dict[str, QMCurrentProxySpec] = {}
    for pub, key in [("LC_M_D6A", "lc_m_d6a"), ("LC_M_D6B", "lc_m_d6b"), ("LC_M_D6C", "lc_m_d6c"), ("LC_M_D6PA", "lc_m_d6pa")]:
        info = QM_CLASS_PROXY[key]
        out[pub] = QMCurrentProxySpec(
            name=pub,
            builder_key=key,
            cell_signs=np.asarray(info["cell_signs"], dtype=float).copy(),
            bond_signs=np.asarray(info["bond_signs"], dtype=float).copy(),
            notes="Q=M high-symmetry 2x2 current proxy; candidate-style, not full bond-by-bond reconstruction.",
        )
    return out


def _current_matrix(pair: Tuple[int, int], sign: float = +1.0) -> np.ndarray:
    i, j = pair
    M = np.zeros((3, 3), dtype=complex)
    M[i, j] = -1j * sign
    M[j, i] = +1j * sign
    return M


J_AB = _current_matrix((A, B), +1.0)
J_AC = _current_matrix((A, C), +1.0)
J_BC = _current_matrix((B, C), +1.0)
J_BONDS = [J_AB, J_AC, J_BC]
J_BOND_NAMES = ["AB", "AC", "BC"]


def proxy_Rcells() -> list[np.ndarray]:
    # Same convention as candidate_diagnosis.py. These are 2x2 cell positions
    # in the coordinate convention used by that helper, not a full lattice plotter.
    return [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3.0) / 2.0]),
        np.array([1.5, np.sqrt(3.0) / 2.0]),
    ]


def _anchor_phase(u: np.ndarray, method: str = "max_component") -> np.ndarray:
    u = np.asarray(u, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(u)
    if nrm == 0:
        raise ValueError("Encountered zero-norm eigenvector")
    u = u / nrm
    if method == "none":
        return u
    if method == "max_component":
        idx = int(np.argmax(np.abs(u)))
        if abs(u[idx]) > 0:
            u = u * np.exp(-1j * np.angle(u[idx]))
        return u
    if method == "first_component":
        if abs(u[0]) > 0:
            u = u * np.exp(-1j * np.angle(u[0]))
        return u
    raise ValueError("anchor_method must be 'none', 'max_component', or 'first_component'")


def local_block_eigvec_at_k(
    model: Any,
    k: Sequence[float],
    spin_slice: slice,
    local_band_index: int,
    *,
    anchor_method: str = "max_component",
) -> np.ndarray:
    kk = np.asarray(k, dtype=float).reshape(2)
    H = np.asarray(model.Hk(float(kk[0]), float(kk[1])), dtype=complex)
    Hloc = H[spin_slice, spin_slice]
    evals, evecs = np.linalg.eigh(Hloc)
    return _anchor_phase(evecs[:, int(local_band_index)], method=anchor_method)


def _require_patchset(patchset: Any) -> None:
    for attr in ("Npatch", "patch_k", "patch_eigvec"):
        if not hasattr(patchset, attr):
            raise TypeError(f"patchset is missing required attribute {attr!r}")


def _fallback_as_weights(weights: Optional[ArrayLike], n: int) -> np.ndarray:
    if as_weights is not None:
        return as_weights(weights, n)
    if weights is None:
        return np.ones(int(n), dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != int(n):
        raise ValueError(f"weights length must be {n}, got {w.size}")
    return w


def _fallback_prepare(v: ArrayLike, weights: Optional[ArrayLike], normalize_v: bool, phase_fix: str) -> np.ndarray:
    if prepare_mode_vector is not None:
        return prepare_mode_vector(v, weights=weights, normalize_v=normalize_v, phase_fix=phase_fix)
    vv = np.asarray(v, dtype=complex).reshape(-1).copy()
    if phase_fix == "sum_real":
        z = np.sum(vv)
        if abs(z) > 0:
            vv *= np.exp(-1j * np.angle(z))
    elif phase_fix == "max_component":
        idx = int(np.argmax(np.abs(vv)))
        if abs(vv[idx]) > 0:
            vv *= np.exp(-1j * np.angle(vv[idx]))
    elif phase_fix != "none":
        raise ValueError("phase_fix must be 'none', 'sum_real', or 'max_component'")
    if normalize_v:
        if weights is None:
            norm = np.sqrt(np.sum(np.abs(vv) ** 2))
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            norm = np.sqrt(np.sum(w * np.abs(vv) ** 2))
        if norm > 0:
            vv = vv / norm
    return vv


def qm_current_proxy_template(
    *,
    model: Any,
    patchset: Any,
    Q: Sequence[float],
    proxy_name: str,
    spin_slice: slice,
    local_band_index: int,
    mode: str = "direct",
    partner_map: Optional[Any] = None,
    partner_tol: float = 1e-6,
    anchor_method: str = "max_component",
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Build one Q=M current proxy patch vector.

    mode='direct' mirrors candidate_diagnosis.py: u(k+Q) is evaluated directly.
    mode='strict' uses only retained partner patches from a strict mask.
    """
    _require_patchset(patchset)
    n = int(patchset.Npatch)
    Q = np.asarray(Q, dtype=float).reshape(2)
    key = PUBLIC_TO_KEY.get(str(proxy_name), str(proxy_name).lower())
    if key not in QM_CLASS_PROXY:
        raise KeyError(f"Unknown Q=M current proxy {proxy_name!r}")
    info = QM_CLASS_PROXY[key]
    cell_signs = np.asarray(info["cell_signs"], dtype=float)
    bond_signs = np.asarray(info["bond_signs"], dtype=float)
    patch_k = np.asarray(patchset.patch_k, dtype=float)
    patch_u = np.asarray(patchset.patch_eigvec, dtype=complex)
    if patch_u.shape[0] != n:
        raise ValueError("patchset.patch_eigvec first dimension must equal Npatch")

    if mode not in {"direct", "strict"}:
        raise ValueError("mode must be 'direct' or 'strict'")

    if mode == "strict":
        if partner_map is None:
            if partner_indices_from_Q_masked is None:
                raise RuntimeError("strict mode needs order_parameter_diagnosis_qm.partner_indices_from_Q_masked")
            partner_map = partner_indices_from_Q_masked(patchset, Q, tol=partner_tol)
        partner_indices = np.asarray(partner_map.partner_indices, dtype=int)
        valid_mask = np.asarray(partner_map.valid_mask, dtype=bool)
        residuals = np.asarray(partner_map.residuals, dtype=float)
    else:
        partner_indices = np.full(n, -1, dtype=int)
        valid_mask = np.ones(n, dtype=bool)
        residuals = np.zeros(n, dtype=float)

    Rcells = proxy_Rcells()
    T = np.full(n, np.nan + 1j * np.nan, dtype=complex)
    rows = []

    for ip, k in enumerate(patch_k):
        if not valid_mask[ip]:
            rows.append({"patch": ip, "valid": False, "partner": int(partner_indices[ip]), "residual": float(residuals[ip]), "template": np.nan + 1j*np.nan})
            continue
        uk = patch_u[ip].reshape(-1)
        if mode == "direct":
            uq = local_block_eigvec_at_k(model, k + Q, spin_slice, local_band_index, anchor_method=anchor_method)
        else:
            uq = patch_u[int(partner_indices[ip])].reshape(-1)
        val = 0.0 + 0.0j
        for ic, R in enumerate(Rcells):
            phase_cell = np.exp(1j * float(np.dot(Q, R)))
            for ib, Jb in enumerate(J_BONDS):
                val += cell_signs[ic] * bond_signs[ib] * phase_cell * np.vdot(uq, Jb @ uk)
        T[ip] = val
        rows.append({
            "patch": ip,
            "valid": True,
            "partner": int(partner_indices[ip]),
            "residual": float(residuals[ip]),
            "template": val,
            "template_real": float(np.real(val)),
            "template_imag": float(np.imag(val)),
            "template_abs": float(abs(val)),
        })

    df = pd.DataFrame(rows)
    meta = {
        "proxy_name": proxy_name,
        "builder_key": key,
        "mode": mode,
        "cell_signs": cell_signs.copy(),
        "bond_signs": bond_signs.copy(),
        "n_valid": int(np.count_nonzero(valid_mask)),
        "Npatch": n,
        "valid_fraction": float(np.count_nonzero(valid_mask) / max(n, 1)),
        "partner_tol": float(partner_tol),
        "note": "Q=M current proxy matching candidate_diagnosis.py; not a full bond-by-bond Fig.6 reconstruction.",
    }
    return T, df, meta


def build_all_qm_current_proxy_templates(
    *,
    model: Any,
    patchset: Any,
    Q: Sequence[float],
    spin_slice: slice,
    local_band_index: int,
    mode: str = "direct",
    partner_map: Optional[Any] = None,
    partner_tol: float = 1e-6,
    anchor_method: str = "max_component",
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    templates: Dict[str, np.ndarray] = {}
    frames = []
    metas = {}
    for name in ["LC_M_D6A", "LC_M_D6B", "LC_M_D6C", "LC_M_D6PA"]:
        T, df, meta = qm_current_proxy_template(
            model=model,
            patchset=patchset,
            Q=Q,
            proxy_name=name,
            spin_slice=spin_slice,
            local_band_index=local_band_index,
            mode=mode,
            partner_map=partner_map,
            partner_tol=partner_tol,
            anchor_method=anchor_method,
        )
        templates[name] = T
        df = df.copy()
        df["proxy"] = name
        frames.append(df)
        metas[name] = meta
    per_patch = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta_all = {"mode": mode, "Q": np.asarray(Q, dtype=float), "proxy_metas": metas}
    return templates, per_patch, meta_all


def weighted_overlap(v: np.ndarray, T: np.ndarray, weights: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    v = np.asarray(v, dtype=complex).reshape(-1)
    T = np.asarray(T, dtype=complex).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if valid_mask is None:
        valid = np.isfinite(T.real) & np.isfinite(T.imag) & np.isfinite(v.real) & np.isfinite(v.imag)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(T.real) & np.isfinite(T.imag) & np.isfinite(v.real) & np.isfinite(v.imag)
    if not np.any(valid):
        return np.nan
    vv = v[valid]
    TT = T[valid]
    ww = w[valid]
    nv = np.sum(ww * np.abs(vv) ** 2)
    nT = np.sum(ww * np.abs(TT) ** 2)
    if nv <= 1e-30 or nT <= 1e-30:
        return np.nan
    return float(np.abs(np.sum(ww * np.conjugate(vv) * TT)) ** 2 / (nv * nT))


def qm_current_weighted_projection(v: np.ndarray, T: np.ndarray, weights: np.ndarray) -> complex:
    v = np.asarray(v, dtype=complex).reshape(-1)
    T = np.asarray(T, dtype=complex).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(T.real) & np.isfinite(T.imag) & np.isfinite(v.real) & np.isfinite(v.imag)
    if not np.any(valid):
        return np.nan + 1j*np.nan
    return np.sum(w[valid] * np.conjugate(v[valid]) * T[valid])



def qm_current_proxy_gram_matrix(
    templates: Mapping[str, np.ndarray],
    weights: Optional[ArrayLike] = None,
    *,
    normalize: bool = False,
) -> pd.DataFrame:
    """Weighted Gram matrix of Q=M current proxy templates.

    G_ab = sum_i w_i T_a(i)^* T_b(i).  If normalize=True, return the
    normalized Gram/cosine matrix.  Large off-diagonal values mean that the
    proxies are not independent scalar templates, so a largest-overlap
    "winner" should not be interpreted as a unique flux class.
    """
    names = list(templates.keys())
    if not names:
        return pd.DataFrame()
    n = len(np.asarray(templates[names[0]]).reshape(-1))
    w = _fallback_as_weights(weights, n)
    G = np.zeros((len(names), len(names)), dtype=complex)

    for ia, a in enumerate(names):
        Ta = np.asarray(templates[a], dtype=complex).reshape(-1)
        for ib, b in enumerate(names):
            Tb = np.asarray(templates[b], dtype=complex).reshape(-1)
            valid = (
                np.isfinite(Ta.real) & np.isfinite(Ta.imag)
                & np.isfinite(Tb.real) & np.isfinite(Tb.imag)
            )
            if not np.any(valid):
                G[ia, ib] = np.nan + 1j * np.nan
                continue
            val = np.sum(w[valid] * np.conjugate(Ta[valid]) * Tb[valid])
            if normalize:
                na = np.sqrt(np.sum(w[valid] * np.abs(Ta[valid]) ** 2))
                nb = np.sqrt(np.sum(w[valid] * np.abs(Tb[valid]) ** 2))
                val = val / (na * nb) if (na > 0 and nb > 0) else np.nan + 1j * np.nan
            G[ia, ib] = val
    return pd.DataFrame(G, index=names, columns=names)


def qm_current_proxy_svd(
    templates: Mapping[str, np.ndarray],
    weights: Optional[ArrayLike] = None,
    *,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """SVD/effective rank of the Q=M current proxy template subspace."""
    names = list(templates.keys())
    if not names:
        return {"names": [], "singular_values": np.array([]), "rank": 0, "Vh": np.zeros((0, 0)), "valid_mask": np.array([], dtype=bool), "tol": tol}
    T = np.column_stack([np.asarray(templates[n], dtype=complex).reshape(-1) for n in names])
    n = T.shape[0]
    w = _fallback_as_weights(weights, n)
    valid = np.all(np.isfinite(T.real) & np.isfinite(T.imag), axis=1)
    if not np.any(valid):
        return {"names": names, "singular_values": np.array([]), "rank": 0, "Vh": np.zeros((0, len(names))), "valid_mask": valid, "tol": tol}
    Tw = T[valid] * np.sqrt(w[valid])[:, None]
    U, svals, Vh = np.linalg.svd(Tw, full_matrices=False)
    rank = int(np.count_nonzero(svals > tol * svals[0])) if svals.size and svals[0] > 0 else 0
    return {"names": names, "singular_values": svals, "rank": rank, "Vh": Vh, "valid_mask": valid, "tol": tol}


def qm_current_proxy_subspace_overlap(
    v: ArrayLike,
    templates: Mapping[str, np.ndarray],
    weights: Optional[ArrayLike] = None,
    *,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Projection of v onto span{Q=M current proxy templates}.

    This is the safer diagnostic when the individual proxies are not
    orthogonal.  It returns ||P_S v||^2 / ||v||^2 in the weighted metric.
    """
    names = list(templates.keys())
    vv = np.asarray(v, dtype=complex).reshape(-1)
    n = vv.size
    w = _fallback_as_weights(weights, n)
    if not names:
        return {"subspace_overlap_sq": np.nan, "rank": 0, "singular_values": np.array([]), "names": names}
    T = np.column_stack([np.asarray(templates[nm], dtype=complex).reshape(-1) for nm in names])
    valid = (
        np.isfinite(vv.real) & np.isfinite(vv.imag)
        & np.all(np.isfinite(T.real) & np.isfinite(T.imag), axis=1)
    )
    if not np.any(valid):
        return {"subspace_overlap_sq": np.nan, "rank": 0, "singular_values": np.array([]), "names": names}
    Tw = T[valid] * np.sqrt(w[valid])[:, None]
    vw = vv[valid] * np.sqrt(w[valid])
    U, svals, Vh = np.linalg.svd(Tw, full_matrices=False)
    rank = int(np.count_nonzero(svals > tol * svals[0])) if svals.size and svals[0] > 0 else 0
    if rank == 0:
        return {"subspace_overlap_sq": np.nan, "rank": 0, "singular_values": svals, "names": names}
    Qbasis = U[:, :rank]
    proj = Qbasis @ (Qbasis.conjugate().T @ vw)
    nv = float(np.real(np.vdot(vw, vw)))
    overlap_sq = float(np.real(np.vdot(proj, proj)) / nv) if nv > 0 else np.nan
    return {"subspace_overlap_sq": overlap_sq, "rank": rank, "singular_values": svals, "names": names}

def diagnose_qm_current_proxies(
    v: ArrayLike,
    *,
    model: Any,
    patchset: Any,
    Q: Sequence[float],
    weights: Optional[ArrayLike] = None,
    spin_slice: slice = slice(0, 3),
    local_band_index: Optional[int] = None,
    mode: str = "direct",
    partner_map: Optional[Any] = None,
    partner_tol: float = 1e-6,
    normalize_v: bool = True,
    phase_fix: str = "none",
    anchor_method: str = "max_component",
) -> QMCurrentProxyResult:
    _require_patchset(patchset)
    n = int(patchset.Npatch)
    ww = _fallback_as_weights(weights, n)
    vv = _fallback_prepare(v, ww, normalize_v=normalize_v, phase_fix=phase_fix)
    if vv.size != n:
        raise ValueError(f"v length must equal Npatch={n}, got {vv.size}")
    if local_band_index is None:
        local_band_index = int(getattr(patchset, "band_index", 0))

    templates, per_patch, meta = build_all_qm_current_proxy_templates(
        model=model,
        patchset=patchset,
        Q=Q,
        spin_slice=spin_slice,
        local_band_index=int(local_band_index),
        mode=mode,
        partner_map=partner_map,
        partner_tol=partner_tol,
        anchor_method=anchor_method,
    )

    gram = qm_current_proxy_gram_matrix(templates, ww, normalize=False)
    normalized_gram = qm_current_proxy_gram_matrix(templates, ww, normalize=True)
    svd_info = qm_current_proxy_svd(templates, ww)
    subspace_info = qm_current_proxy_subspace_overlap(vv, templates, ww)

    rows = []
    for name, T in templates.items():
        valid = np.isfinite(T.real) & np.isfinite(T.imag)
        norm2 = float(np.sum(ww[valid] * np.abs(T[valid]) ** 2)) if np.any(valid) else 0.0
        proj = qm_current_weighted_projection(vv, T, ww)
        ov = weighted_overlap(vv, T, ww)
        rows.append({
            "proxy": name,
            "mode": mode,
            "template_norm2": norm2,
            "template_max_abs": float(np.nanmax(np.abs(T))) if np.any(valid) else np.nan,
            "projection": proj,
            "projection_real": float(np.real(proj)) if np.isfinite(np.real(proj)) else np.nan,
            "projection_imag": float(np.imag(proj)) if np.isfinite(np.imag(proj)) else np.nan,
            "projection_abs": float(abs(proj)) if np.isfinite(np.real(proj)) else np.nan,
            "overlap_sq": ov,
            "n_valid": int(np.count_nonzero(valid)),
            "Npatch": n,
            "valid_fraction": float(np.count_nonzero(valid) / max(n, 1)),
            "note": "individual proxy overlap; proxies may be non-orthogonal, check Gram/SVD/subspace diagnostics",
        })
    scores = pd.DataFrame(rows).sort_values("overlap_sq", ascending=False, na_position="last").reset_index(drop=True)
    metadata = {
        "Q": np.asarray(Q, dtype=float),
        "mode": mode,
        "normalize_v": bool(normalize_v),
        "phase_fix": str(phase_fix),
        "spin_slice": (spin_slice.start, spin_slice.stop, spin_slice.step),
        "local_band_index": int(local_band_index),
        "normalization_note": "v normalized by sqrt(sum_i w_i |v_i|^2) if normalize_v=True.",
        "interpretation_note": "Q=M current proxies mirror candidate_diagnosis.py; they classify proxy patterns, not full real-space flux arrows.",
        "gram_matrix": gram,
        "normalized_gram_matrix": normalized_gram,
        "proxy_rank": svd_info["rank"],
        "proxy_singular_values": svd_info["singular_values"],
        "subspace_overlap_sq": subspace_info["subspace_overlap_sq"],
        "warning": (
            "Q=M current proxies are generally non-orthogonal. Do not interpret "
            "the largest individual overlap as a unique winner unless the "
            "normalized Gram matrix is close to diagonal. Use Gram/SVD/subspace "
            "diagnostics for robust interpretation."
        ),
    }
    metadata.update(meta)
    return QMCurrentProxyResult(
        Q=np.asarray(Q, dtype=float),
        mode=mode,
        pattern_name="QM_current_proxies",
        v=vv,
        weights=ww,
        templates=templates,
        scores=scores,
        per_patch=per_patch,
        metadata=metadata,
    )


def mock_vector_for_qm_current_proxy(
    proxy_name: str,
    *,
    model: Any,
    patchset: Any,
    Q: Sequence[float],
    spin_slice: slice = slice(0, 3),
    local_band_index: Optional[int] = None,
    mode: str = "direct",
    partner_map: Optional[Any] = None,
    partner_tol: float = 1e-6,
    anchor_method: str = "max_component",
    convention: str = "conj",
) -> np.ndarray:
    if local_band_index is None:
        local_band_index = int(getattr(patchset, "band_index", 0))
    T, _, _ = qm_current_proxy_template(
        model=model,
        patchset=patchset,
        Q=Q,
        proxy_name=proxy_name,
        spin_slice=spin_slice,
        local_band_index=int(local_band_index),
        mode=mode,
        partner_map=partner_map,
        partner_tol=partner_tol,
        anchor_method=anchor_method,
    )
    if convention == "conj":
        return np.conjugate(T)
    if convention == "i_conj":
        return 1j * np.conjugate(T)
    if convention == "minus_i_conj":
        return -1j * np.conjugate(T)
    raise ValueError("convention must be 'conj', 'i_conj', or 'minus_i_conj'")


def templates_summary_dataframe(templates: Mapping[str, np.ndarray], weights: Optional[ArrayLike] = None) -> pd.DataFrame:
    first = next(iter(templates.values()))
    n = len(first)
    w = _fallback_as_weights(weights, n)
    rows = []
    for name, T in templates.items():
        T = np.asarray(T, dtype=complex)
        valid = np.isfinite(T.real) & np.isfinite(T.imag)
        rows.append({
            "proxy": name,
            "norm2": float(np.sum(w[valid] * np.abs(T[valid]) ** 2)) if np.any(valid) else 0.0,
            "max_abs": float(np.nanmax(np.abs(T))) if np.any(valid) else np.nan,
            "n_valid": int(np.count_nonzero(valid)),
            "Npatch": int(T.size),
        })
    return pd.DataFrame(rows)


def plot_qm_current_proxy_scores(scores: pd.DataFrame, *, ax=None, title: str = "Q=M current proxy overlaps"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    df = scores.copy()
    ax.bar(df["proxy"].astype(str), df["overlap_sq"].astype(float))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("overlap_sq")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    return ax


# =============================================================================
# Combined module export list
# =============================================================================

__all__ = [name for name in globals() if not name.startswith("_")]
