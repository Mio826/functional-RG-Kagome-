from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frg_kernel import normalize_spin, has_patchset, patchset_for_spin, partner_map_from_q_index, canonicalize_q_for_patchsets

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
GammaAccessor = callable


@dataclass
class ChannelKernel:
    """Patch-space kernel for a fixed transfer momentum Q and fixed external spin block.

    The matrix acts on a momentum basis whose physical interpretation depends on the
    channel:

    - pp: basis state is a pair ``(k, Q-k)``
    - ph: basis state is a bilinear ``k -> k+Q`` (direct) or ``k -> k-Q`` / ``k -> k+Q``
      depending on the crossed routing
    """

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
    """Tensor-product mother kernel in patch ⊗ spin(bilinear/pair) space."""

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

    def split_vector_by_block(self, vec: np.ndarray) -> Dict[str, np.ndarray]:
        vec = np.asarray(vec, dtype=complex).reshape(self.nblocks, self.Npatch)
        return {label: vec[i].copy() for i, label in enumerate(self.basis_labels)}

    def block(self, row_label: str, col_label: str) -> np.ndarray:
        ir = self.basis_labels.index(row_label)
        ic = self.basis_labels.index(col_label)
        N = self.Npatch
        return self.matrix[ir * N:(ir + 1) * N, ic * N:(ic + 1) * N]


def assemble_mother_kernel(
    *,
    name: str,
    Q: np.ndarray,
    blocks: Sequence[Sequence[ChannelKernel]],
    basis_labels: Sequence[str],
) -> MotherChannelKernel:
    if len(blocks) == 0 or len(blocks) != len(basis_labels):
        raise ValueError("blocks must be a non-empty square list matching basis_labels.")
    N = blocks[0][0].Npatch
    nb = len(basis_labels)
    matrix = np.zeros((nb * N, nb * N), dtype=complex)
    residuals = np.zeros((nb * N, nb * N), dtype=float)
    for i in range(nb):
        if len(blocks[i]) != nb:
            raise ValueError("blocks must be square.")
        for j in range(nb):
            ker = blocks[i][j]
            if ker.Npatch != N:
                raise ValueError("All blocks must have the same patch dimension.")
            matrix[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.asarray(ker.matrix, dtype=complex)
            residuals[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.asarray(ker.residuals, dtype=float)
    return MotherChannelKernel(
        name=name,
        Q=np.asarray(Q, dtype=float),
        matrix=matrix,
        basis_labels=tuple(str(x) for x in basis_labels),
        Npatch=N,
        residuals=residuals,
    )


class FullVertexChannelBuilder:
    r"""Build mother kernels directly from the full antisymmetrized vertex.

    Stored / input object
    ---------------------
    The input ``gamma`` must be the *full* four-leg vertex accessor with convention

        (p1,s1), (p2,s2) -> (p3,s3), (p4,s4).

    This class never assumes that pp / phd / phc are stored pieces of the vertex.
    They are used only as *external-leg routings* for constructing diagnosis kernels.

    Channel conventions
    -------------------
    1) pp pair basis at fixed Q:
           in  = (k,   Q-k)
           out = (k',  Q-k')
       K_pp[out, in] = Γ(k, Q-k -> k', Q-k')

    2) ph direct bilinear basis at fixed Q:
           in  = k   -> k+Q
           out = k'  -> k'+Q
       K_phd[out, in] = Γ(k, k'+Q -> k+Q, k')

    3) ph crossed bilinear basis at fixed Q:
           in  = k   -> k-Q
           out = k'  -> k'+Q
       K_phc[out, in] = Γ(k, k' -> k'+Q, k-Q)

    The important point is that the ph basis is a bilinear basis, not a pp-like pair basis.
    """

    def __init__(
        self,
        gamma,
        patchsets: PatchSetMap,
        *,
        closure_map: Optional[Mapping[Tuple[str, str, str, str], Tuple[np.ndarray, np.ndarray]]] = None,
        transfer_context: Optional[Mapping[str, object]] = None,
        q_merge_tol_red: float = 5e-2,
        q_key_decimals: int = 10,
    ) -> None:
        self.gamma = gamma
        self.patchsets = patchsets
        if closure_map is None:
            raise ValueError("FullVertexChannelBuilder requires closure_map from the flow solver; diagnosis must use the same p4 closure as the stored full vertex.")
        self.closure_map = dict(closure_map)
        self.q_merge_tol_red = float(q_merge_tol_red)
        self.q_key_decimals = int(q_key_decimals)
        self.spins = [s for s in ("up", "dn") if has_patchset(self.patchsets, s)]
        if not self.spins:
            raise ValueError("No non-empty spin patchsets found.")
        self.Npatch = patchset_for_spin(self.patchsets, self.spins[0]).Npatch
        for s in self.spins:
            if patchset_for_spin(self.patchsets, s).Npatch != self.Npatch:
                raise ValueError("FullVertexChannelBuilder requires identical patch counts across available spin sectors.")
        self._patch_k = {
            s: np.asarray([canonicalize_q_for_patchsets(self.patchsets, p.k_cart) for p in patchset_for_spin(self.patchsets, s).patches], dtype=float)
            for s in self.spins
        }
        self.transfer_context = dict(transfer_context) if transfer_context is not None else None
        if self.transfer_context is not None:
            self._load_transfer_context(self.transfer_context)
        else:
            self._build_q_index_tables()

    def _build_q_index_tables(self) -> None:
        self._pp_q_index: Dict[Tuple[str, str], np.ndarray] = {}
        self._ph_q_index_plus: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index_plus: Dict[Tuple[str, str], np.ndarray] = {}
        self._phc_q_index_minus: Dict[Tuple[str, str], np.ndarray] = {}
        self._q_rep_uv: list[np.ndarray] = []
        self._q_key_to_index: Dict[Tuple[float, float], int] = {}

        for s_src in self.spins:
            for s_tgt in self.spins:
                arr_pp = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_ph_plus = np.zeros((self.Npatch, self.Npatch), dtype=int)
                arr_ph_minus = np.zeros((self.Npatch, self.Npatch), dtype=int)
                for p_src, k_src in enumerate(self._patch_k[s_src]):
                    for p_tgt, k_tgt in enumerate(self._patch_k[s_tgt]):
                        arr_pp[p_src, p_tgt] = self._q_nearest_index(k_src + k_tgt)
                        arr_ph_plus[p_src, p_tgt] = self._q_nearest_index(k_tgt - k_src)
                        arr_ph_minus[p_src, p_tgt] = self._q_nearest_index(k_src - k_tgt)
                self._pp_q_index[(s_src, s_tgt)] = arr_pp
                self._ph_q_index_plus[(s_src, s_tgt)] = arr_ph_plus
                self._phc_q_index_plus[(s_src, s_tgt)] = arr_ph_plus.copy()
                self._phc_q_index_minus[(s_src, s_tgt)] = arr_ph_minus

    def _load_transfer_context(self, ctx: Mapping[str, object]) -> None:
        self.pp_grid = ctx.get("pp_grid")  # type: ignore[assignment]
        self.phd_grid = ctx.get("phd_grid")  # type: ignore[assignment]
        self.phc_grid = ctx.get("phc_grid")  # type: ignore[assignment]
        if self.pp_grid is None or self.phd_grid is None or self.phc_grid is None:
            raise ValueError("transfer_context must provide pp_grid, phd_grid, and phc_grid.")
        self._pp_q_index = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["pp_q_index"].items()}  # type: ignore[index]
        self._ph_q_index_plus = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["phd_q_index_plus"].items()}  # type: ignore[index]
        self._phc_q_index_plus = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["phc_q_index_plus"].items()}  # type: ignore[index]
        self._phc_q_index_minus = {tuple(map(str, k)): np.asarray(v, dtype=int).copy() for k, v in ctx["phc_q_index_minus"].items()}  # type: ignore[index]
        self._q_rep_uv = [self._uv(q) for q in self.phd_grid.q_list]
        self._q_key_to_index = {tuple(np.round(uv, self.q_key_decimals)): i for i, uv in enumerate(self._q_rep_uv)}

    def _uv(self, q: Sequence[float]) -> np.ndarray:
        q_can = canonicalize_q_for_patchsets(self.patchsets, q)
        ps = patchset_for_spin(self.patchsets, self.spins[0])
        B = np.column_stack([np.asarray(ps.b1, dtype=float), np.asarray(ps.b2, dtype=float)])
        uv = np.linalg.solve(B, np.asarray(q_can, dtype=float))
        uv = uv - np.floor(uv)
        uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
        uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
        return uv

    def _q_key(self, q: Sequence[float]) -> Tuple[float, float]:
        return tuple(np.round(self._uv(q), self.q_key_decimals))

    @staticmethod
    def _uv_distance(uv_a: np.ndarray, uv_b: np.ndarray) -> float:
        duv = np.asarray(uv_a, dtype=float) - np.asarray(uv_b, dtype=float)
        duv = duv - np.round(duv)
        return float(np.linalg.norm(duv))

    def _find_matching_rep(self, uv: np.ndarray) -> Optional[int]:
        key = tuple(np.round(np.asarray(uv, dtype=float), self.q_key_decimals))
        if key in self._q_key_to_index:
            return int(self._q_key_to_index[key])
        if self.q_merge_tol_red <= 0:
            return None
        best_idx = None
        best_dist = np.inf
        for irep, uv_rep in enumerate(self._q_rep_uv):
            dist = self._uv_distance(uv, uv_rep)
            if dist <= self.q_merge_tol_red and (
                dist < best_dist - 1e-14 or (abs(dist - best_dist) <= 1e-14 and (best_idx is None or irep < best_idx))
            ):
                best_idx = int(irep)
                best_dist = float(dist)
        return best_idx

    def _q_nearest_index(self, q: Sequence[float]) -> int:
        q_can = canonicalize_q_for_patchsets(self.patchsets, q)
        uv = self._uv(q_can)
        idx = self._find_matching_rep(uv)
        if idx is not None:
            return int(idx)
        idx = len(self._q_rep_uv)
        self._q_rep_uv.append(uv)
        self._q_key_to_index[tuple(np.round(uv, self.q_key_decimals))] = idx
        return int(idx)

    def _pp_partner(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            self.patchsets,
            self._pp_q_index[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=canonicalize_q_for_patchsets(self.patchsets, Q),
            mode="Q_minus_k",
        )

    def _ph_partner(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            self.patchsets,
            self._ph_q_index_plus[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=canonicalize_q_for_patchsets(self.patchsets, Q),
            mode="k_plus_Q",
        )

    def _phc_partner(self, iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float], mode: str):
        return partner_map_from_q_index(
            self.patchsets,
            (self._phc_q_index_plus if mode == "k_plus_Q" else self._phc_q_index_minus)[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=canonicalize_q_for_patchsets(self.patchsets, Q),
            mode=mode,
        )

    def _stored_p4(self, s1: str, s2: str, s3: str, s4: str, p1: int, p2: int, p3: int) -> int:
        key = (normalize_spin(s1), normalize_spin(s2), normalize_spin(s3), normalize_spin(s4))
        entry = self.closure_map.get(key)
        if entry is None:
            raise KeyError(f"Missing closure_map entry for spin block {key}.")
        p4_idx = entry[0]
        return int(p4_idx[p1, p2, p3])

    def _stored_p4_residual(self, s1: str, s2: str, s3: str, s4: str, p1: int, p2: int, p3: int) -> float:
        key = (normalize_spin(s1), normalize_spin(s2), normalize_spin(s3), normalize_spin(s4))
        entry = self.closure_map.get(key)
        if entry is None:
            raise KeyError(f"Missing closure_map entry for spin block {key}.")
        p4_res = entry[1]
        return float(p4_res[p1, p2, p3])

    def pp_block(self, Q: Sequence[float], *, incoming_pair: Tuple[SpinLike, SpinLike], outgoing_pair: Optional[Tuple[SpinLike, SpinLike]] = None) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        s1, s2 = map(normalize_spin, incoming_pair)
        if outgoing_pair is None:
            s3, s4 = s1, s2
        else:
            s3, s4 = map(normalize_spin, outgoing_pair)
        iq = self._q_nearest_index(Q)
        partner_in, resid_in = self._pp_partner(iq, first_spin=s1, second_spin=s2, Q=Q)
        partner_out, resid_out = self._pp_partner(iq, first_spin=s3, second_spin=s4, Q=Q)

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p4_partner = int(partner_out[pout])
            for pin in range(self.Npatch):
                p2 = int(partner_in[pin])
                if p2 >= 0:
                    p4 = self._stored_p4(s1, s2, s3, s4, pin, p2, pout)
                    if int(p4) == p4_partner and int(p4) >= 0:
                        M[pout, pin] = self.gamma(pin, s1, p2, s2, pout, s3, int(p4), s4)
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]), self._stored_p4_residual(s1, s2, s3, s4, pin, p2, pout))
                else:
                    residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]))
        return ChannelKernel(
            name="pp_raw",
            Q=np.asarray(Q, dtype=float),
            matrix=M,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(partner_out, dtype=int),
            col_partner_patches=np.asarray(partner_in, dtype=int),
            row_spins=(s3, s4),
            col_spins=(s1, s2),
            residuals=residuals,
        )

    def phd_block(self, Q: Sequence[float], *, incoming_bilinear: Tuple[SpinLike, SpinLike], outgoing_bilinear: Optional[Tuple[SpinLike, SpinLike]] = None) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        s1, s3 = map(normalize_spin, incoming_bilinear)
        if outgoing_bilinear is None:
            s4, s2 = s1, s3
        else:
            s4, s2 = map(normalize_spin, outgoing_bilinear)
        iq = self._q_nearest_index(Q)
        kplus_in, resid_in = self._ph_partner(iq, first_spin=s1, second_spin=s3, Q=Q)
        kplus_out, resid_out = self._ph_partner(iq, first_spin=s4, second_spin=s2, Q=Q)

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p2 = int(kplus_out[pout])
            for pin in range(self.Npatch):
                p3 = int(kplus_in[pin])
                p4_expected = self._stored_p4(s1, s2, s3, s4, pin, p2, p3) if (p2 >= 0 and p3 >= 0) else None
                if p2 >= 0 and p3 >= 0 and int(p4_expected) == pout:
                    M[pout, pin] = self.gamma(pin, s1, p2, s2, p3, s3, pout, s4)
                residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]), self._stored_p4_residual(s1, s2, s3, s4, pin, p2, p3) if (p2 >= 0 and p3 >= 0) else 0.0)
        return ChannelKernel(
            name="phd_raw",
            Q=np.asarray(Q, dtype=float),
            matrix=M,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kplus_in, dtype=int),
            row_spins=(s4, s2),
            col_spins=(s1, s3),
            residuals=residuals,
        )

    def phc_block(self, Q: Sequence[float], *, incoming_bilinear: Tuple[SpinLike, SpinLike], outgoing_bilinear: Optional[Tuple[SpinLike, SpinLike]] = None) -> ChannelKernel:
        Q = canonicalize_q_for_patchsets(self.patchsets, Q)
        s1, s2 = map(normalize_spin, incoming_bilinear)
        if outgoing_bilinear is None:
            s3, s4 = s2, s1
        else:
            s3, s4 = map(normalize_spin, outgoing_bilinear)
        iq = self._q_nearest_index(Q)
        kplus_out, resid_out = self._phc_partner(iq, first_spin=s2, second_spin=s3, Q=Q, mode="k_plus_Q")
        kminus_in, resid_in = self._phc_partner(iq, first_spin=s1, second_spin=s4, Q=Q, mode="k_minus_Q")

        M = np.zeros((self.Npatch, self.Npatch), dtype=complex)
        residuals = np.zeros((self.Npatch, self.Npatch), dtype=float)
        for pout in range(self.Npatch):
            p3 = int(kplus_out[pout])
            for pin in range(self.Npatch):
                p4 = int(kminus_in[pin])
                p4_expected = self._stored_p4(s1, s2, s3, s4, pin, pout, p3) if (p3 >= 0 and p4 >= 0) else None
                if p3 >= 0 and p4 >= 0 and int(p4_expected) == p4:
                    M[pout, pin] = self.gamma(pin, s1, pout, s2, p3, s3, p4, s4)
                residuals[pout, pin] = max(float(resid_in[pin]), float(resid_out[pout]), self._stored_p4_residual(s1, s2, s3, s4, pin, pout, p3) if (p3 >= 0 and p4 >= 0) else 0.0)
        return ChannelKernel(
            name="phc_raw",
            Q=np.asarray(Q, dtype=float),
            matrix=M,
            row_patches=np.arange(self.Npatch, dtype=int),
            col_patches=np.arange(self.Npatch, dtype=int),
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kminus_in, dtype=int),
            row_spins=(s2, s3),
            col_spins=(s1, s4),
            residuals=residuals,
        )

    def build_pp_mother_sz0(self, Q: Sequence[float]) -> MotherChannelKernel:
        if not (has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn")):
            raise ValueError("pp Sz=0 mother kernel requires both up and dn patchsets.")
        K_ud_ud = self.pp_block(Q, incoming_pair=("up", "dn"), outgoing_pair=("up", "dn"))
        K_ud_du = self.pp_block(Q, incoming_pair=("up", "dn"), outgoing_pair=("dn", "up"))
        K_du_ud = self.pp_block(Q, incoming_pair=("dn", "up"), outgoing_pair=("up", "dn"))
        K_du_du = self.pp_block(Q, incoming_pair=("dn", "up"), outgoing_pair=("dn", "up"))
        return assemble_mother_kernel(
            name="pp_mother_sz0",
            Q=np.asarray(Q, dtype=float),
            basis_labels=("ud", "du"),
            blocks=[
                [K_ud_ud, K_ud_du],
                [K_du_ud, K_du_du],
            ],
        )

    def build_ph_mother_longitudinal(self, Q: Sequence[float]) -> MotherChannelKernel:
        if not (has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn")):
            raise ValueError("longitudinal ph mother kernel requires both up and dn patchsets.")
        # Basis labels are longitudinal bilinears: uu ≡ c†_up c_up, dd ≡ c†_dn c_dn
        K_uu_uu = self.phd_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("up", "up"))
        K_uu_dd = self.phd_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("dn", "dn"))
        K_dd_uu = self.phd_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("up", "up"))
        K_dd_dd = self.phd_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("dn", "dn"))
        return assemble_mother_kernel(
            name="ph_mother_longitudinal",
            Q=np.asarray(Q, dtype=float),
            basis_labels=("uu", "dd"),
            blocks=[
                [K_uu_uu, K_uu_dd],
                [K_dd_uu, K_dd_dd],
            ],
        )

    def build_raw_block_dict(self, Q: Sequence[float]) -> Dict[str, ChannelKernel]:
        out: Dict[str, ChannelKernel] = {}
        if has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn"):
            out["pp_ud_to_ud"] = self.pp_block(Q, incoming_pair=("up", "dn"), outgoing_pair=("up", "dn"))
            out["pp_ud_to_du"] = self.pp_block(Q, incoming_pair=("up", "dn"), outgoing_pair=("dn", "up"))
            out["pp_du_to_ud"] = self.pp_block(Q, incoming_pair=("dn", "up"), outgoing_pair=("up", "dn"))
            out["pp_du_to_du"] = self.pp_block(Q, incoming_pair=("dn", "up"), outgoing_pair=("dn", "up"))
            out["phd_uu_to_uu"] = self.phd_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("up", "up"))
            out["phd_uu_to_dd"] = self.phd_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("dn", "dn"))
            out["phd_dd_to_uu"] = self.phd_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("up", "up"))
            out["phd_dd_to_dd"] = self.phd_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("dn", "dn"))
        if has_patchset(self.patchsets, "up"):
            out["pp_uu_to_uu"] = self.pp_block(Q, incoming_pair=("up", "up"), outgoing_pair=("up", "up"))
            out["phc_uu"] = self.phc_block(Q, incoming_bilinear=("up", "up"), outgoing_bilinear=("up", "up"))
        if has_patchset(self.patchsets, "dn"):
            out["pp_dd_to_dd"] = self.pp_block(Q, incoming_pair=("dn", "dn"), outgoing_pair=("dn", "dn"))
            out["phc_dd"] = self.phc_block(Q, incoming_bilinear=("dn", "dn"), outgoing_bilinear=("dn", "dn"))
        return out

    def build_mother_kernel_dict(self, Q: Sequence[float]) -> Dict[str, MotherChannelKernel]:
        out: Dict[str, MotherChannelKernel] = {}
        if has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn"):
            out["pp_mother_sz0"] = self.build_pp_mother_sz0(Q)
            out["ph_mother_longitudinal"] = self.build_ph_mother_longitudinal(Q)
        return out


    @classmethod
    def from_solver(cls, gamma, solver: object) -> "FullVertexChannelBuilder":
        transfer_context = solver.transfer_context() if callable(getattr(solver, "transfer_context", None)) else None
        closure_map = solver.closure_map() if callable(getattr(solver, "closure_map", None)) else getattr(solver, "_closure_map", None)
        return cls(
            gamma,
            solver.patchsets,
            closure_map=closure_map,
            transfer_context=transfer_context,
        )

# Backward-compatible alias.
ChannelDecomposer = FullVertexChannelBuilder


__all__ = [
    "ChannelKernel",
    "MotherChannelKernel",
    "assemble_mother_kernel",
    "FullVertexChannelBuilder",
    "ChannelDecomposer",
]
