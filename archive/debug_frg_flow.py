
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from frg_flow import FRGFlowSolver, BareVertexFromInteraction
from frg_kernel import compute_pp_kernel, compute_ph_kernel, compute_phc_kernel


@dataclass
class KernelAttempt:
    channel: str
    iq: int
    Q: np.ndarray
    spin_block: Tuple[str, str, str, str]
    success: bool
    max_abs: Optional[float] = None
    exc_type: Optional[str] = None
    exc_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "iq": int(self.iq),
            "Q": np.asarray(self.Q, dtype=float).tolist(),
            "spin_block": list(self.spin_block),
            "success": bool(self.success),
            "max_abs": None if self.max_abs is None else float(self.max_abs),
            "exc_type": self.exc_type,
            "exc_message": self.exc_message,
        }


class DebugFRGFlowSolver(FRGFlowSolver):
    """
    Non-invasive debugger wrapper for FRGFlowSolver.

    It does NOT modify your original code.
    It exposes:
      1. detailed exception tracing for compute_channel_rhs
      2. summary counts for successful/failed kernel builds
      3. one-step flow diagnostics
    """

    def compute_channel_rhs_debug(self, T: float, *, raise_on_error: bool = False):
        gamma = self.gamma_accessor()
        cfg = self._flow_config(T)

        rhs_pp = self._empty_channel_store(self.pp_grid)
        rhs_phd = self._empty_channel_store(self.phd_grid)
        rhs_phc = self._empty_channel_store(self.phc_grid)

        attempts: List[KernelAttempt] = []

        def _record_success(channel: str, iq: int, Q, spin_block, mat):
            attempts.append(
                KernelAttempt(
                    channel=channel,
                    iq=iq,
                    Q=np.asarray(Q, dtype=float),
                    spin_block=tuple(spin_block),
                    success=True,
                    max_abs=float(np.max(np.abs(mat))) if np.size(mat) else 0.0,
                )
            )

        def _record_failure(channel: str, iq: int, Q, spin_block, exc: Exception):
            attempts.append(
                KernelAttempt(
                    channel=channel,
                    iq=iq,
                    Q=np.asarray(Q, dtype=float),
                    spin_block=tuple(spin_block),
                    success=False,
                    exc_type=type(exc).__name__,
                    exc_message=str(exc),
                )
            )

        for iq, Q in enumerate(self.pp_grid.q_list):
            for s1, s2, s3, s4 in self.spin_blocks:
                block = (s1, s2, s3, s4)
                try:
                    ker = compute_pp_kernel(
                        gamma,
                        self.patchsets,
                        Q,
                        incoming_spins=(s1, s2),
                        outgoing_spins=(s3, s4),
                        config=cfg,
                    )
                    rhs_pp[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)
                    _record_success("pp", iq, Q, block, ker.matrix)
                except Exception as exc:
                    _record_failure("pp", iq, Q, block, exc)
                    if raise_on_error:
                        raise

        for iq, Q in enumerate(self.phd_grid.q_list):
            for s1, s2, s3, s4 in self.spin_blocks:
                block = (s1, s2, s3, s4)
                try:
                    ker = compute_ph_kernel(
                        gamma,
                        self.patchsets,
                        Q,
                        incoming_spins=(s1, s3),
                        outgoing_spins=(s4, s2),
                        config=cfg,
                    )
                    rhs_phd[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)
                    _record_success("phd", iq, Q, block, ker.matrix)
                except Exception as exc:
                    _record_failure("phd", iq, Q, block, exc)
                    if raise_on_error:
                        raise

        for iq, Q in enumerate(self.phc_grid.q_list):
            for s1, s2, s3, s4 in self.spin_blocks:
                block = (s1, s2, s3, s4)
                try:
                    ker = compute_phc_kernel(
                        gamma,
                        self.patchsets,
                        Q,
                        incoming_spins=(s1, s2),
                        outgoing_spins=(s3, s4),
                        config=cfg,
                    )
                    rhs_phc[(s1, s2, s3, s4, iq)] = np.asarray(ker.matrix, dtype=complex)
                    _record_success("phc", iq, Q, block, ker.matrix)
                except Exception as exc:
                    _record_failure("phc", iq, Q, block, exc)
                    if raise_on_error:
                        raise

        summary = summarize_attempts(attempts)
        return rhs_pp, rhs_phd, rhs_phc, attempts, summary

    def debug_single_step(self, T: Optional[float] = None, dT: Optional[float] = None) -> Dict[str, Any]:
        if T is None:
            T = float(self.state.T)
        if dT is None:
            temps = self.temperature_path
            if len(temps) >= 2:
                dT = float(temps[1] - temps[0])
            else:
                dT = -1e-3

        rhs_pp, rhs_phd, rhs_phc, attempts, summary = self.compute_channel_rhs_debug(T, raise_on_error=False)
        rhs_norm = self._rhs_norm(rhs_pp, rhs_phd, rhs_phc)
        before_norm = self.state.channel_norm()

        self._apply_rhs(rhs_pp, rhs_phd, rhs_phc, dT)
        after_norm = self.state.channel_norm()

        return {
            "T": float(T),
            "dT": float(dT),
            "rhs_norm": float(rhs_norm),
            "channel_norm_before": float(before_norm),
            "channel_norm_after": float(after_norm),
            "summary": summary,
            "attempts": [a.to_dict() for a in attempts],
        }


def summarize_attempts(attempts: Sequence[KernelAttempt]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "total": len(attempts),
        "success": 0,
        "failure": 0,
        "by_channel": {},
        "top_errors": {},
        "nonzero_success_count": 0,
        "max_success_abs": 0.0,
    }

    for a in attempts:
        ch = a.channel
        if ch not in out["by_channel"]:
            out["by_channel"][ch] = {"success": 0, "failure": 0, "nonzero_success": 0, "max_abs": 0.0}
        bucket = out["by_channel"][ch]

        if a.success:
            out["success"] += 1
            bucket["success"] += 1
            if a.max_abs is not None and a.max_abs > 0:
                out["nonzero_success_count"] += 1
                bucket["nonzero_success"] += 1
                out["max_success_abs"] = max(out["max_success_abs"], a.max_abs)
                bucket["max_abs"] = max(bucket["max_abs"], a.max_abs)
        else:
            out["failure"] += 1
            bucket["failure"] += 1
            key = f"{a.channel}:{a.exc_type}:{a.exc_message}"
            out["top_errors"][key] = out["top_errors"].get(key, 0) + 1

    out["top_errors"] = dict(sorted(out["top_errors"].items(), key=lambda kv: (-kv[1], kv[0]))[:10])
    return out


def compare_kernel_interfaces(
    *,
    patchsets: Mapping[str, object],
    interaction: Any,
    Q,
    config,
    spin_block: Tuple[str, str, str, str] = ("up", "dn", "up", "dn"),
    solver: Optional[FRGFlowSolver] = None,
) -> Dict[str, Any]:
    """
    Compare kernel outputs for:
      1. bare callable vertex
      2. current flow-reconstructed gamma_accessor
    This isolates whether the failure is in the solver gamma path.
    """
    s1, s2, s3, s4 = spin_block
    gamma_bare = BareVertexFromInteraction(interaction, patchsets)

    out: Dict[str, Any] = {
        "Q": np.asarray(Q, dtype=float).tolist(),
        "spin_block": list(spin_block),
    }

    for name, fn, kwargs in [
        ("pp", compute_pp_kernel, dict(incoming_spins=(s1, s2), outgoing_spins=(s3, s4))),
        ("phd", compute_ph_kernel, dict(incoming_spins=(s1, s3), outgoing_spins=(s4, s2))),
        ("phc", compute_phc_kernel, dict(incoming_spins=(s1, s2), outgoing_spins=(s3, s4))),
    ]:
        ch_info: Dict[str, Any] = {}

        try:
            ker = fn(gamma_bare, patchsets, Q, config=config, **kwargs)
            ch_info["bare_success"] = True
            ch_info["bare_max_abs"] = float(np.max(np.abs(ker.matrix)))
        except Exception as exc:
            ch_info["bare_success"] = False
            ch_info["bare_error"] = f"{type(exc).__name__}: {exc}"

        if solver is not None:
            gamma_flow = solver.gamma_accessor()
            try:
                ker = fn(gamma_flow, patchsets, Q, config=config, **kwargs)
                ch_info["flow_success"] = True
                ch_info["flow_max_abs"] = float(np.max(np.abs(ker.matrix)))
            except Exception as exc:
                ch_info["flow_success"] = False
                ch_info["flow_error"] = f"{type(exc).__name__}: {exc}"

        out[name] = ch_info

    return out
