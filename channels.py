from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frg_kernel import (
    PatchSetMap,
    SZ0VertexInput,
    build_sz0_vertex_accessor,
    canonicalize_q_for_patchsets,
    has_patchset,
    patchset_for_spin,
    partner_map_from_q_index,
)


SpinLike = Union[str, int]


@dataclass
class ChannelKernel:
    """Patch-space kernel for one physical channel at fixed Q."""
    name: str
    Q: np.ndarray
    matrix: np.ndarray
    row_patches: np.ndarray
    col_patches: np.ndarray
    row_partner_patches: np.ndarray
    col_partner_patches: np.ndarray
    residuals: np.ndarray

    @property
    def Npatch(self) -> int:
        return int(self.matrix.shape[0])

    def hermitian_residual(self) -> float:
        return float(np.max(np.abs(self.matrix - self.matrix.conjugate().T)))

    def eig(self, sort_by: str = "abs"):
        vals, vecs = np.linalg.eig(self.matrix)
        if sort_by == "abs":
            order = np.argsort(-np.abs(vals))
        elif sort_by == "real":
            order = np.argsort(-np.real(vals))
        elif sort_by == "hermitian":
            vals, vecs = np.linalg.eigh(0.5 * (self.matrix + self.matrix.conjugate().T))
            order = np.argsort(-vals)
        else:
            raise ValueError("sort_by must be 'abs', 'real', or 'hermitian'.")
        return vals[order], vecs[:, order]


class SZ0ChannelBuilder:
    r"""
    Build PRL-compatible physical channel kernels directly from the minimal
    S_z=0 vertex V(1,2;3,4).

    Convention
    ----------
    V(1,2;3,4) ≡ Γ_{up,dn -> dn,up}(1,2;3,4)

    Physical channel kernels:
      pp singlet = V + V_out_exchange
      pp triplet = V - V_out_exchange
      ph charge  = V_d - 2 V_x
      ph spin    = V_d
    """

    def __init__(
        self,
        vertex: SZ0VertexInput,
        patchsets: PatchSetMap,
        *,
        closure_map: Optional[Mapping[Tuple[str, str, str, str], Tuple[np.ndarray, np.ndarray]]] = None,
        transfer_context: Optional[Mapping[str, object]] = None,
        q_merge_tol_red: float = 5e-2,
        q_key_decimals: int = 10,
    ) -> None:
        self.vertex = build_sz0_vertex_accessor(vertex)
        self.patchsets = patchsets
        if not (has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn")):
            raise ValueError("SZ0 channel builder requires both up and dn patchsets.")
        self.Npatch = patchset_for_spin(self.patchsets, "up").Npatch
        if patchset_for_spin(self.patchsets, "dn").Npatch != self.Npatch:
            raise ValueError("SZ0 channel builder requires identical up/dn patch counts.")
        if closure_map is None:
            raise ValueError("SZ0ChannelBuilder requires closure_map from the flow solver.")
        if transfer_context is None:
            raise ValueError("SZ0ChannelBuilder requires transfer_context from the flow solver.")

        self.closure_map = dict(closure_map)
        self.q_merge_tol_red = float(q_merge_tol_red)
        self.q_key_decimals = int(q_key_decimals)
        self._load_transfer_context(dict(transfer_context))

    def _load_transfer_context(self, ctx: Mapping[str, object]) -> None:
        self.pp_grid = ctx.get("pp_grid")
        self.phd_grid = ctx.get("phd_grid")
        self.phc_grid = ctx.get("phc_grid")
        if self.pp_grid is None or self.phd_grid is None or self.phc_grid is None:
            raise ValueError("transfer_context must provide pp_grid, phd_grid, and phc_grid.")
        self._pp_q_index = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["pp_q_index"].items()}
        self._phd_q_index_plus = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["phd_q_index_plus"].items()}
        self._phc_q_index_plus = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["phc_q_index_plus"].items()}
        self._phc_q_index_minus = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["phc_q_index_minus"].items()}

    def _stored_p4(self, p1: int, p2: int, p3: int) -> int:
        key = ("up", "dn", "dn", "up")
        entry = self.closure_map.get(key)
        if entry is None:
            raise KeyError(f"Missing closure_map entry for spin block {key}.")
        return int(entry[0][p1, p2, p3])

    def _stored_p4_residual(self, p1: int, p2: int, p3: int) -> float:
        key = ("up", "dn", "dn", "up")
        entry = self.closure_map.get(key)
        if entry is None:
            raise KeyError(f"Missing closure_map entry for spin block {key}.")
        return float(entry[1][p1, p2, p3])

    def _pp_partner(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            self.patchsets,
            self._pp_q_index[(first_spin, second_spin)],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=canonicalize_q_for_patchsets(self.patchsets, Q),
            mode="Q_minus_k",
        )

    def _ph_partner(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            self.patchsets,
            self._phd_q_index_plus[(first_spin, second_spin)],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=canonicalize_q_for_patchsets(self.patchsets, Q),
            mode="k_plus_Q",
        )

    def _pp_raw_v(self, Q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Raw matrix V(k,Q-k ; k',Q-k').
        Row variable = k', column variable = k.
        """
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        iq = self.pp_grid.nearest_index(Q)
        partner_in, resid_in = self._pp_partner(iq, first_spin="up", second_spin="dn", Q=Q)
        partner_out, resid_out = self._pp_partner(iq, first_spin="dn", second_spin="up", Q=Q)

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p4_partner = int(partner_out[pout])
            for pin in range(self.Npatch):
                p2 = int(partner_in[pin])
                if p2 >= 0:
                    p4 = self._stored_p4(pin, p2, pout)
                    if int(p4) == p4_partner and int(p4) >= 0:
                        M[pout, pin] = self.vertex(pin, p2, pout, int(p4))
                    residuals[pout, pin] = max(
                        float(resid_in[pin]),
                        float(resid_out[pout]),
                        self._stored_p4_residual(pin, p2, pout),
                    )
                else:
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]))
        return M, partner_in, residuals

    def _pp_out_exchange_v(self, Q: Sequence[float]) -> np.ndarray:
        """
        Raw matrix V(k,Q-k ; Q-k',k').
        Same incoming basis, outgoing legs exchanged.
        """
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        iq = self.pp_grid.nearest_index(Q)
        partner_in, _ = self._pp_partner(iq, first_spin="up", second_spin="dn", Q=Q)
        partner_out_ex, _ = self._pp_partner(iq, first_spin="up", second_spin="dn", Q=Q)

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        for pout in range(self.Npatch):
            p3 = int(partner_out_ex[pout])  # Q-k'
            p4 = pout                      # k'
            for pin in range(self.Npatch):
                p2 = int(partner_in[pin])
                if p2 >= 0 and p3 >= 0:
                    p4_expected = self._stored_p4(pin, p2, p3)
                    if int(p4_expected) == int(p4):
                        M[pout, pin] = self.vertex(pin, p2, p3, p4)
        return M

    def pp_singlet(self, Q: Sequence[float]) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        Vraw, partner_in, residuals = self._pp_raw_v(Q)
        Vex = self._pp_out_exchange_v(Q)
        return ChannelKernel(
            name="pp_singlet",
            Q=np.asarray(Q, dtype=float),
            matrix=Vraw + Vex,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(partner_in, dtype=int),
            col_partner_patches=np.asarray(partner_in, dtype=int),
            residuals=residuals,
        )

    def pp_triplet(self, Q: Sequence[float]) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        Vraw, partner_in, residuals = self._pp_raw_v(Q)
        Vex = self._pp_out_exchange_v(Q)
        return ChannelKernel(
            name="pp_triplet",
            Q=np.asarray(Q, dtype=float),
            matrix=Vraw - Vex,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(partner_in, dtype=int),
            col_partner_patches=np.asarray(partner_in, dtype=int),
            residuals=residuals,
        )

    def ph_direct(self, Q: Sequence[float]) -> ChannelKernel:
        """
        V_d(k',k;Q) = V(k, k'+Q ; k+Q, k')
        Row variable = k', column variable = k.
        """
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        iq = self.phd_grid.nearest_index(Q)
        kplus_in, resid_in = self._ph_partner(iq, first_spin="up", second_spin="dn", Q=Q)
        kplus_out, resid_out = self._ph_partner(iq, first_spin="up", second_spin="dn", Q=Q)

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p2 = int(kplus_out[pout])   # k'+Q
            for pin in range(self.Npatch):
                p3 = int(kplus_in[pin]) # k+Q
                if p2 >= 0 and p3 >= 0:
                    p4_expected = self._stored_p4(pin, p2, p3)
                    if int(p4_expected) == pout:
                        M[pout, pin] = self.vertex(pin, p2, p3, pout)
                    residuals[pout, pin] = max(
                        float(resid_in[pin]),
                        float(resid_out[pout]),
                        self._stored_p4_residual(pin, p2, p3),
                    )
                else:
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]))
        return ChannelKernel(
            name="ph_direct",
            Q=np.asarray(Q, dtype=float),
            matrix=M,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kplus_in, dtype=int),
            residuals=residuals,
        )

    def ph_exchange(self, Q: Sequence[float]) -> ChannelKernel:
        """
        V_x(k',k;Q) = V(k, k'+Q ; k', k+Q)
        Row variable = k', column variable = k.
        """
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        iq = self.phd_grid.nearest_index(Q)
        kplus_in, resid_in = self._ph_partner(iq, first_spin="up", second_spin="dn", Q=Q)
        kplus_out, resid_out = self._ph_partner(iq, first_spin="up", second_spin="dn", Q=Q)

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p2 = int(kplus_out[pout])   # k'+Q
            p3 = pout                   # k'
            for pin in range(self.Npatch):
                p4 = int(kplus_in[pin]) # k+Q
                if p2 >= 0 and p4 >= 0:
                    p4_expected = self._stored_p4(pin, p2, p3)
                    if int(p4_expected) == int(p4):
                        M[pout, pin] = self.vertex(pin, p2, p3, p4)
                    residuals[pout, pin] = max(
                        float(resid_in[pin]),
                        float(resid_out[pout]),
                        self._stored_p4_residual(pin, p2, p3),
                    )
                else:
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]))
        return ChannelKernel(
            name="ph_exchange",
            Q=np.asarray(Q, dtype=float),
            matrix=M,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kplus_in, dtype=int),
            residuals=residuals,
        )

    def ph_charge(self, Q: Sequence[float]) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        Vd = self.ph_direct(Q)
        Vx = self.ph_exchange(Q)
        return ChannelKernel(
            name="ph_charge",
            Q=np.asarray(Q, dtype=float),
            matrix=Vd.matrix - 2.0 * Vx.matrix,
            row_patches=Vd.row_patches.copy(),
            col_patches=Vd.col_patches.copy(),
            row_partner_patches=Vd.row_partner_patches.copy(),
            col_partner_patches=Vd.col_partner_patches.copy(),
            residuals=np.maximum(Vd.residuals, Vx.residuals),
        )

    def ph_spin(self, Q: Sequence[float]) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        Vd = self.ph_direct(Q)
        return ChannelKernel(
            name="ph_spin",
            Q=np.asarray(Q, dtype=float),
            matrix=Vd.matrix.copy(),
            row_patches=Vd.row_patches.copy(),
            col_patches=Vd.col_patches.copy(),
            row_partner_patches=Vd.row_partner_patches.copy(),
            col_partner_patches=Vd.col_partner_patches.copy(),
            residuals=Vd.residuals.copy(),
        )

    def build_kernel_dict(self, Q: Sequence[float]) -> Dict[str, ChannelKernel]:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        return {
            "pp_singlet": self.pp_singlet(Q),
            "pp_triplet": self.pp_triplet(Q),
            "ph_charge": self.ph_charge(Q),
            "ph_spin": self.ph_spin(Q),
        }

    @classmethod
    def from_solver(cls, vertex: SZ0VertexInput, solver: object) -> "SZ0ChannelBuilder":
        transfer_context = solver.transfer_context() if callable(getattr(solver, "transfer_context", None)) else None
        closure_map = solver.closure_map() if callable(getattr(solver, "closure_map", None)) else getattr(solver, "_closure_map", None)
        return cls(
            vertex,
            solver.patchsets,
            closure_map=closure_map,
            transfer_context=transfer_context,
        )


__all__ = [
    "ChannelKernel",
    "SZ0ChannelBuilder",
]
