from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frg_kernel import normalize_spin, has_patchset, patchset_for_spin, partner_map_from_q_index, canonicalize_q_for_patchsets
from frg_geometry_shared import SharedGeometry

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
GammaAccessor = callable


@dataclass
class ChannelKernel:
    name: str
    Q: np.ndarray
    matrix: np.ndarray
    row_patches: np.ndarray
    col_patches: np.ndarray
    row_partner_patches: np.ndarray
    col_partner_patches: np.ndarray
    row_spins: Tuple[str, str]
    col_spins: Tuple[str, str]
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


@dataclass
class MotherChannelKernel:
    name: str
    Q: np.ndarray
    matrix: np.ndarray
    basis_labels: Tuple[str, ...]
    Npatch: int
    residuals: np.ndarray

    @property
    def nblocks(self) -> int:
        return len(self.basis_labels)

    def hermitian_part(self) -> np.ndarray:
        return 0.5 * (self.matrix + self.matrix.conjugate().T)

    def hermitian_residual(self) -> float:
        return float(np.max(np.abs(self.matrix - self.matrix.conjugate().T)))

    def eig(self, sort_by: str = "abs"):
        if sort_by == "hermitian":
            vals, vecs = np.linalg.eigh(self.hermitian_part())
            order = np.argsort(-vals)
            return vals[order], vecs[:, order]
        vals, vecs = np.linalg.eig(self.matrix)
        if sort_by == "abs":
            order = np.argsort(-np.abs(vals))
        elif sort_by == "real":
            order = np.argsort(-np.real(vals))
        else:
            raise ValueError("sort_by must be 'abs', 'real', or 'hermitian'.")
        return vals[order], vecs[:, order]


def assemble_mother_kernel(*, name: str, Q: np.ndarray, blocks, basis_labels) -> MotherChannelKernel:
    N = blocks[0][0].Npatch
    nb = len(basis_labels)
    matrix = np.zeros((nb * N, nb * N), dtype=complex)
    residuals = np.zeros((nb * N, nb * N), dtype=float)
    for i in range(nb):
        for j in range(nb):
            ker = blocks[i][j]
            matrix[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.asarray(ker.matrix, dtype=complex)
            residuals[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.asarray(ker.residuals, dtype=float)
    return MotherChannelKernel(name=name, Q=np.asarray(Q, dtype=float), matrix=matrix, basis_labels=tuple(basis_labels), Npatch=N, residuals=residuals)


class FullVertexChannelBuilder:
    def __init__(
        self,
        gamma,
        patchsets: PatchSetMap,
        *,
        closure_map: Optional[Mapping[Tuple[str, str, str, str], Tuple[np.ndarray, np.ndarray]]] = None,
        q_merge_tol_red: float = 5e-2,
        q_key_decimals: int = 10,
        shared_geometry: Optional[SharedGeometry] = None,
    ) -> None:
        self.gamma = gamma
        self.patchsets = patchsets
        if closure_map is None:
            raise ValueError("FullVertexChannelBuilder requires closure_map from the flow solver.")
        self.closure_map = dict(closure_map)
        self.q_merge_tol_red = float(q_merge_tol_red)
        self.q_key_decimals = int(q_key_decimals)
        self.spins = [s for s in ("up", "dn") if has_patchset(self.patchsets, s)]
        self.Npatch = patchset_for_spin(self.patchsets, self.spins[0]).Npatch

        self._shared_geometry = shared_geometry
        if shared_geometry is not None:
            self._pp_q_index = shared_geometry.pp_q_index
            self._ph_q_index = shared_geometry.phd_q_index
            self._phc_q_index = shared_geometry.phc_q_index
            self._pp_grid = shared_geometry.pp_grid
            self._phd_grid = shared_geometry.phd_grid
            self._phc_grid = shared_geometry.phc_grid
        else:
            # fallback: local construction compatible with current interface
            self._patch_k = {
                s: np.asarray([canonicalize_q_for_patchsets(self.patchsets, p.k_cart) for p in patchset_for_spin(self.patchsets, s).patches], dtype=float)
                for s in self.spins
            }
            self._build_q_index_tables()
            self._pp_grid = None
            self._phd_grid = None
            self._phc_grid = None

    def _build_q_index_tables(self) -> None:
        self._pp_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._ph_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._q_rep_uv: list[np.ndarray] = []
        self._q_key_to_index: Dict[Tuple[float, float], int] = {}
        for s1 in self.spins:
            for s2 in self.spins:
                arr_pp = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_ph = np.zeros((self.Npatch, self.Npatch), dtype=int)
                for p1, k1 in enumerate(self._patch_k[s1]):
                    for p2, k2 in enumerate(self._patch_k[s2]):
                        arr_pp[p1, p2] = self._q_nearest_index_local(k1 + k2)
                        arr_ph[p1, p2] = self._q_nearest_index_local(k1 - k2)
                self._pp_q_index[(s1, s2)] = arr_pp
                self._ph_q_index[(s1, s2)] = arr_ph
                self._phc_q_index[(s1, s2)] = arr_ph.copy()

    def _uv(self, q: Sequence[float]) -> np.ndarray:
        q_can = canonicalize_q_for_patchsets(self.patchsets, q)
        ps = patchset_for_spin(self.patchsets, self.spins[0])
        B = np.column_stack([np.asarray(ps.b1, dtype=float), np.asarray(ps.b2, dtype=float)])
        uv = np.linalg.solve(B, np.asarray(q_can, dtype=float))
        uv = uv - np.floor(uv)
        uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
        uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
        return uv

    def _find_matching_rep(self, uv: np.ndarray) -> Optional[int]:
        key = tuple(np.round(np.asarray(uv, dtype=float), self.q_key_decimals))
        if key in self._q_key_to_index:
            return int(self._q_key_to_index[key])
        best_idx = None
        best_dist = np.inf
        for irep, uv_rep in enumerate(self._q_rep_uv):
            duv = np.asarray(uv, dtype=float) - np.asarray(uv_rep, dtype=float)
            duv = duv - np.round(duv)
            dist = float(np.linalg.norm(duv))
            if dist <= self.q_merge_tol_red and (
                dist < best_dist - 1e-14 or (abs(dist - best_dist) <= 1e-14 and (best_idx is None or irep < best_idx))
            ):
                best_idx = int(irep)
                best_dist = float(dist)
        return best_idx

    def _q_nearest_index_local(self, q: Sequence[float]) -> int:
        q_can = canonicalize_q_for_patchsets(self.patchsets, q)
        uv = self._uv(q_can)
        idx = self._find_matching_rep(uv)
        if idx is not None:
            return int(idx)
        idx = len(self._q_rep_uv)
        self._q_rep_uv.append(uv)
        self._q_key_to_index[tuple(np.round(uv, self.q_key_decimals))] = idx
        return int(idx)

    def _stored_p4(self, s1: str, s2: str, s3: str, s4: str, p1: int, p2: int, p3: int) -> int:
        return int(self.closure_map[(normalize_spin(s1), normalize_spin(s2), normalize_spin(s3), normalize_spin(s4))][0][p1, p2, p3])

    def _stored_p4_residual(self, s1: str, s2: str, s3: str, s4: str, p1: int, p2: int, p3: int) -> float:
        return float(self.closure_map[(normalize_spin(s1), normalize_spin(s2), normalize_spin(s3), normalize_spin(s4))][1][p1, p2, p3])

    def pp_block(self, Q: Sequence[float], *, incoming_pair, outgoing_pair=None) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        s1, s2 = map(normalize_spin, incoming_pair)
        s3, s4 = map(normalize_spin, outgoing_pair or incoming_pair)
        iq = self._pp_grid.nearest_index(Q) if self._pp_grid is not None else self._q_nearest_index_local(Q)
        partner_in, resid_in = partner_map_from_q_index(
            self.patchsets, self._pp_q_index[(s1, s2)], source_spin=s1, target_spin=s2, iq_target=iq, Q=Q, mode="Q_minus_k"
        )
        partner_out, resid_out = partner_map_from_q_index(
            self.patchsets, self._pp_q_index[(s3, s4)], source_spin=s3, target_spin=s4, iq_target=iq, Q=Q, mode="Q_minus_k"
        )
        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p4_partner = int(partner_out[pout])
            for pin in range(self.Npatch):
                p2 = int(partner_in[pin])
                if p2 >= 0:
                    p4 = self._stored_p4(s1, s2, s3, s4, pin, p2, pout)
                    if p4 == p4_partner and p4 >= 0:
                        M[pout, pin] = self.gamma(pin, s1, p2, s2, pout, s3, p4, s4)
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]), self._stored_p4_residual(s1, s2, s3, s4, pin, p2, pout))
                else:
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]))
        return ChannelKernel("pp_raw", np.asarray(Q, dtype=float), M, np.arange(self.Npatch), np.arange(self.Npatch), np.asarray(partner_out), np.asarray(partner_in), (s3, s4), (s1, s2), residuals)

    def phd_block(self, Q: Sequence[float], *, incoming_bilinear, outgoing_bilinear=None) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        s1, s3 = map(normalize_spin, incoming_bilinear)
        s4, s2 = map(normalize_spin, outgoing_bilinear or incoming_bilinear)
        iq = self._phd_grid.nearest_index(Q) if self._phd_grid is not None else self._q_nearest_index_local(Q)
        kplus_in, resid_in = partner_map_from_q_index(
            self.patchsets, self._ph_q_index[(s3, s1)], source_spin=s1, target_spin=s3, iq_target=iq, Q=Q, mode="k_plus_Q"
        )
        kplus_out, resid_out = partner_map_from_q_index(
            self.patchsets, self._ph_q_index[(s2, s4)], source_spin=s4, target_spin=s2, iq_target=iq, Q=Q, mode="k_plus_Q"
        )
        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p2 = int(kplus_out[pout])
            for pin in range(self.Npatch):
                p3 = int(kplus_in[pin])
                if p2 >= 0 and p3 >= 0 and self._stored_p4(s1, s2, s3, s4, pin, p2, p3) == pout:
                    M[pout, pin] = self.gamma(pin, s1, p2, s2, p3, s3, pout, s4)
                residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]), self._stored_p4_residual(s1, s2, s3, s4, pin, p2, p3) if (p2 >= 0 and p3 >= 0) else 0.0)
        return ChannelKernel("phd_raw", np.asarray(Q, dtype=float), M, np.arange(self.Npatch), np.arange(self.Npatch), np.asarray(kplus_out), np.asarray(kplus_in), (s4, s2), (s1, s3), residuals)

    def build_pp_mother_sz0(self, Q: Sequence[float]) -> MotherChannelKernel:
        K_ud_ud = self.pp_block(Q, incoming_pair=("up", "dn"), outgoing_pair=("up", "dn"))
        K_ud_du = self.pp_block(Q, incoming_pair=("up", "dn"), outgoing_pair=("dn", "up"))
        K_du_ud = self.pp_block(Q, incoming_pair=("dn", "up"), outgoing_pair=("up", "dn"))
        K_du_du = self.pp_block(Q, incoming_pair=("dn", "up"), outgoing_pair=("dn", "up"))
        return assemble_mother_kernel(name="pp_mother_sz0", Q=np.asarray(Q, dtype=float), basis_labels=("ud", "du"), blocks=[[K_ud_ud, K_ud_du], [K_du_ud, K_du_du]])

    def build_ph_mother_longitudinal(self, Q: Sequence[float]) -> MotherChannelKernel:
        K_uu_uu = self.phd_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("up", "up"))
        K_uu_dd = self.phd_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("dn", "dn"))
        K_dd_uu = self.phd_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("up", "up"))
        K_dd_dd = self.phd_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("dn", "dn"))
        return assemble_mother_kernel(name="ph_mother_longitudinal", Q=np.asarray(Q, dtype=float), basis_labels=("uu", "dd"), blocks=[[K_uu_uu, K_uu_dd], [K_dd_uu, K_dd_dd]])

    def build_mother_kernel_dict(self, Q: Sequence[float]) -> Dict[str, MotherChannelKernel]:
        return {
            "pp_mother_sz0": self.build_pp_mother_sz0(Q),
            "ph_mother_longitudinal": self.build_ph_mother_longitudinal(Q),
        }


ChannelDecomposer = FullVertexChannelBuilder

__all__ = [
    "ChannelKernel",
    "MotherChannelKernel",
    "assemble_mother_kernel",
    "FullVertexChannelBuilder",
    "ChannelDecomposer",
]
