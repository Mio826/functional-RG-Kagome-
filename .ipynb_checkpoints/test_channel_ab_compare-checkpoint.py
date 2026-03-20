
from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# A/B test for the longitudinal ph-channel definition.
#
# What this test is trying to answer:
#   Is the FM-vs-PI failure already present at the level of the channel
#   construction, before form-factor diagnosis does anything fancy?
#
# It compares:
#   old solver: legacy longitudinal kernel from diagonal same-spin blocks only
#   new solver: corrected longitudinal kernel including uu->dd and dd->uu blocks
#
# Important:
#   The RG flow itself is unchanged.  If both solvers end with identical channel
#   correction tensors but produce different diagnostic ph kernels, then the bug
#   is upstream of order diagnosis.
# -----------------------------------------------------------------------------


def _load_module(name: str):
    return importlib.import_module(name)


def _max_store_diff(a: Dict[Any, np.ndarray], b: Dict[Any, np.ndarray]) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = 0.0
    for key in keys:
        aa = np.asarray(a.get(key, 0.0), dtype=complex)
        bb = np.asarray(b.get(key, 0.0), dtype=complex)
        if aa.shape != bb.shape:
            return np.inf
        out = max(out, float(np.max(np.abs(aa - bb))))
    return out


def _kernel_summary(kernel, diagnoser=None, sort_by: str = "abs"):
    vals, vecs = kernel.eig(sort_by=sort_by)
    out = {
        "name": kernel.name,
        "Q": np.asarray(kernel.Q, dtype=float).tolist(),
        "largest_abs_eval": float(np.abs(vals[0])) if len(vals) else 0.0,
        "largest_real_eval": float(np.real(vals[0])) if len(vals) else 0.0,
        "hermitian_residual": float(np.max(np.abs(kernel.matrix - kernel.matrix.conjugate().T))),
        "vector_std_abs": float(np.std(np.abs(vecs[:, 0]))) if len(vals) else 0.0,
    }
    if diagnoser is not None:
        try:
            diag = diagnoser.diagnose_kernel(kernel, sort_by=sort_by)
            out["paper_label"] = diag.paper_label
            out["coarse_label"] = diag.coarse_label
            out["paper_score"] = float(diag.paper_score)
            out["coarse_score"] = float(diag.coarse_score)
            out["degeneracy"] = int(diag.degeneracy)
        except Exception as exc:
            out["diagnosis_error"] = repr(exc)
    return out


def run_channel_ab_test(
    *,
    patchsets,
    bare_gamma,
    old_flow_module: str = "frg_flow",
    new_flow_module: str = "frg_flow_channelfix",
    solver_kwargs: Optional[Dict[str, Any]] = None,
    qs_to_check: Optional[Sequence[np.ndarray]] = None,
    sort_by: str = "abs",
):
    solver_kwargs = dict(solver_kwargs or {})

    old_mod = _load_module(old_flow_module)
    new_mod = _load_module(new_flow_module)

    old_solver = old_mod.FRGFlowSolver(patchsets=patchsets, bare_gamma=bare_gamma, **solver_kwargs)
    new_solver = new_mod.FRGFlowSolver(patchsets=patchsets, bare_gamma=bare_gamma, **solver_kwargs)

    old_hist = old_solver.run()
    new_hist = new_solver.run()

    state_diff = {
        "pp_corr_max_diff": _max_store_diff(old_solver.state.pp_corr, new_solver.state.pp_corr),
        "phd_corr_max_diff": _max_store_diff(old_solver.state.phd_corr, new_solver.state.phd_corr),
        "phc_corr_max_diff": _max_store_diff(old_solver.state.phc_corr, new_solver.state.phc_corr),
        "final_channel_norm_old": float(old_solver.state.channel_norm()),
        "final_channel_norm_new": float(new_solver.state.channel_norm()),
        "n_history_old": len(old_hist),
        "n_history_new": len(new_hist),
    }

    if qs_to_check is None:
        qs_to_check = [np.zeros(2, dtype=float)]
        for q in old_solver.diagnosis_Qs:
            q = np.asarray(q, dtype=float)
            if np.linalg.norm(q) < 1e-12:
                qs_to_check = [q]
                break

    kernel_report = []
    for Q in qs_to_check:
        Q = np.asarray(Q, dtype=float)
        old_k = old_solver.build_diagnosis_kernel_dict(Q)
        new_k = new_solver.build_diagnosis_kernel_dict(Q)

        entry = {
            "Q": Q.tolist(),
            "old_available": sorted(old_k.keys()),
            "new_available": sorted(new_k.keys()),
        }

        for key in [
            "phd_uu_to_uu",
            "phd_uu_to_dd",
            "phd_dd_to_uu",
            "phd_dd_to_dd",
            "ph_charge_longitudinal",
            "ph_spin_longitudinal",
            "ph_charge_longitudinal_legacy",
            "ph_spin_longitudinal_legacy",
        ]:
            if key in new_k:
                entry[f"new::{key}"] = _kernel_summary(new_k[key], diagnoser=new_solver.diagnoser, sort_by=sort_by)
            if key in old_k:
                entry[f"old::{key}"] = _kernel_summary(old_k[key], diagnoser=old_solver.diagnoser, sort_by=sort_by)

        if "phd_uu_to_dd" in new_k and "phd_dd_to_uu" in new_k:
            entry["mixed_block_norms"] = {
                "uu_to_dd": float(np.max(np.abs(new_k["phd_uu_to_dd"].matrix))),
                "dd_to_uu": float(np.max(np.abs(new_k["phd_dd_to_uu"].matrix))),
            }

        kernel_report.append(entry)

    report = {
        "state_comparison": state_diff,
        "kernel_report": kernel_report,
        "interpretation_hint": (
            "If pp/phd/phc correction tensors are identical but the corrected longitudinal kernels differ strongly "
            "from the legacy ones, then the FM-vs-PI issue already appears at the channel-construction stage, "
            "before form-factor diagnosis."
        ),
    }
    return report


def print_channel_ab_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))
