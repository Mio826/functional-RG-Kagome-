from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frg_kernel import (
    normalize_spin,
    patchset_for_spin,
    has_patchset,
    available_internal_spin_pairs,
    canonicalize_q_for_patchsets,
    partner_map_from_q_index,
)

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]
SpinBlock = Tuple[str, str, str, str]


def canonical_spin_tuple(key: Tuple[SpinLike, SpinLike, SpinLike, SpinLike]) -> SpinBlock:
    return tuple(normalize_spin(x) for x in key)  # type: ignore[return-value]


def available_physical_spins(patchsets: PatchSetMap) -> List[str]:
    out: List[str] = []
    for s in ("up", "dn"):
        if has_patchset(patchsets, s):
            out.append(s)
    if not out:
        raise ValueError("No non-empty spin patch sets found.")
    return out


def default_spin_blocks(patchsets: PatchSetMap) -> List[SpinBlock]:
    spins = available_physical_spins(patchsets)
    blocks: List[SpinBlock] = []
    if "up" in spins:
        blocks.append(("up", "up", "up", "up"))
    if "dn" in spins:
        blocks.append(("dn", "dn", "dn", "dn"))
    if set(spins) == {"up", "dn"}:
        blocks.extend(
            [
                ("up", "dn", "up", "dn"),
                ("up", "dn", "dn", "up"),
                ("dn", "up", "up", "dn"),
                ("dn", "up", "dn", "up"),
            ]
        )
    return blocks


@dataclass
class TransferGrid:
    patchsets: PatchSetMap
    q_list: List[np.ndarray]
    decimals: int = 10
    merge_tol_red: float = 5e-2

    def __post_init__(self) -> None:
        ref_spin = available_physical_spins(self.patchsets)[0]
        ps = patchset_for_spin(self.patchsets, ref_spin)
        self.b1 = np.asarray(ps.b1, dtype=float)
        self.b2 = np.asarray(ps.b2, dtype=float)
        self.q_list = [canonicalize_q_for_patchsets(self.patchsets, q) for q in self.q_list]
        self.rep_uv_list: List[np.ndarray] = []
        self.key_to_index: Dict[Tuple[float, float], int] = {}
        unique_qs: List[np.ndarray] = []
        for q in self.q_list:
            uv = self._uv(q)
            idx = self._find_matching_rep(uv)
            if idx is None:
                idx = len(unique_qs)
                unique_qs.append(np.asarray(q, dtype=float))
                self.rep_uv_list.append(uv)
                self.key_to_index[self._exact_key_from_uv(uv)] = idx
        self.q_list = unique_qs

    def _reduced_coords(self, q: Sequence[float]) -> np.ndarray:
        B = np.column_stack([self.b1, self.b2])
        uv = np.linalg.solve(B, np.asarray(q, dtype=float))
        uv = uv - np.floor(uv)
        uv[np.isclose(uv, 1.0, atol=1e-12)] = 0.0
        uv[np.isclose(uv, 0.0, atol=1e-12)] = 0.0
        return uv

    def _uv(self, q: Sequence[float]) -> np.ndarray:
        return self._reduced_coords(canonicalize_q_for_patchsets(self.patchsets, q))

    def _exact_key_from_uv(self, uv: np.ndarray) -> Tuple[float, float]:
        return tuple(np.round(np.asarray(uv, dtype=float), decimals=self.decimals))

    def _reduced_distance(self, uv_a: np.ndarray, uv_b: np.ndarray) -> float:
        duv = np.asarray(uv_a, dtype=float) - np.asarray(uv_b, dtype=float)
        duv = duv - np.round(duv)
        return float(np.linalg.norm(duv))

    def _find_matching_rep(self, uv: np.ndarray) -> Optional[int]:
        if len(self.rep_uv_list) == 0:
            return None
        key = self._exact_key_from_uv(uv)
        if key in self.key_to_index:
            return int(self.key_to_index[key])
        best_idx = None
        best_dist = np.inf
        for irep, uv_rep in enumerate(self.rep_uv_list):
            dist = self._reduced_distance(uv, uv_rep)
            if dist <= self.merge_tol_red and (
                dist < best_dist - 1e-14
                or (abs(dist - best_dist) <= 1e-14 and (best_idx is None or irep < best_idx))
            ):
                best_idx = int(irep)
                best_dist = float(dist)
        return best_idx

    def canonicalize(self, q: Sequence[float]) -> np.ndarray:
        return canonicalize_q_for_patchsets(self.patchsets, q)

    def nearest_index(self, q: Sequence[float]) -> int:
        q_can = self.canonicalize(q)
        uv = self._uv(q_can)
        key = self._exact_key_from_uv(uv)
        if key in self.key_to_index:
            return int(self.key_to_index[key])
        idx = self._find_matching_rep(uv)
        if idx is not None:
            return int(idx)
        dists = [self._reduced_distance(uv, uv_rep) for uv_rep in self.rep_uv_list]
        return int(np.argmin(dists))


def _canonicalize_patch_k_array(ks: np.ndarray, patchsets: PatchSetMap) -> np.ndarray:
    out = np.zeros_like(ks, dtype=float)
    for i, k in enumerate(np.asarray(ks, dtype=float)):
        out[i] = canonicalize_q_for_patchsets(patchsets, k)
    return out


def build_unique_q_list(
    patchsets: PatchSetMap,
    *,
    mode: str,
    decimals: int = 10,
    merge_tol_red: float = 5e-2,
) -> List[np.ndarray]:
    ref_spin = available_physical_spins(patchsets)[0]
    ps = patchset_for_spin(patchsets, ref_spin)
    ks = _canonicalize_patch_k_array(np.asarray([p.k_cart for p in ps.patches], dtype=float), patchsets)

    raw_candidates: List[np.ndarray] = [np.zeros(2, dtype=float)]
    if mode == "pp":
        for k1 in ks:
            for k2 in ks:
                raw_candidates.append(np.asarray(k1 + k2, dtype=float))
    elif mode in {"ph", "phc"}:
        for k1 in ks:
            for k3 in ks:
                raw_candidates.append(np.asarray(k3 - k1, dtype=float))
    else:
        raise ValueError("mode must be one of {'pp', 'ph', 'phc'}")

    grid = TransferGrid(
        patchsets,
        raw_candidates,
        decimals=decimals,
        merge_tol_red=merge_tol_red,
    )
    return [np.asarray(q, dtype=float) for q in grid.q_list]


@dataclass
class SharedGeometry:
    patchsets: PatchSetMap
    spin_blocks: Tuple[SpinBlock, ...]
    q_merge_tol_red: float
    q_key_decimals: int
    spins: Tuple[str, ...]
    internal_spin_pairs: Tuple[Tuple[str, str], ...]
    Npatch: int
    patch_k: Dict[str, np.ndarray]
    energies_by_spin: Dict[str, np.ndarray]
    pp_grid: TransferGrid
    phd_grid: TransferGrid
    phc_grid: TransferGrid
    pp_q_index: Dict[Tuple[str, str], np.ndarray]
    phd_q_index: Dict[Tuple[str, str], np.ndarray]
    phc_q_index: Dict[Tuple[str, str], np.ndarray]
    pp_qminus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]]
    ph_kplus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]]
    phc_kplus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]]
    phc_kminus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]]
    closure_map: Dict[SpinBlock, Tuple[np.ndarray, np.ndarray]]


def _find_shifted_patch_index(patchsets: PatchSetMap, spin: str, target_k: Sequence[float]) -> Tuple[int, float]:
    ps = patchset_for_spin(patchsets, spin)
    ks = np.asarray([p.k_cart for p in ps.patches], dtype=float)
    b1 = np.asarray(ps.b1, dtype=float)
    b2 = np.asarray(ps.b2, dtype=float)
    best_idx = 0
    best_norm = np.inf
    target_k = np.asarray(target_k, dtype=float)
    for i, k_ref in enumerate(ks):
        local_best = np.inf
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                disp = target_k - (k_ref + n1 * b1 + n2 * b2)
                nd = np.linalg.norm(disp)
                if nd < local_best:
                    local_best = nd
        if local_best < best_norm:
            best_norm = local_best
            best_idx = i
    return int(best_idx), float(best_norm)


def build_shared_geometry(
    patchsets: PatchSetMap,
    *,
    spin_blocks: Optional[Sequence[Tuple[SpinLike, SpinLike, SpinLike, SpinLike]]] = None,
    pp_Qs: Optional[Sequence[Sequence[float]]] = None,
    ph_Qs: Optional[Sequence[Sequence[float]]] = None,
    phc_Qs: Optional[Sequence[Sequence[float]]] = None,
    q_merge_tol_red: float = 5e-2,
    q_key_decimals: int = 10,
) -> SharedGeometry:
    spin_blocks_can = tuple(canonical_spin_tuple(x) for x in (spin_blocks or default_spin_blocks(patchsets)))
    spins = tuple(available_physical_spins(patchsets))
    internal_spin_pairs = tuple(available_internal_spin_pairs(patchsets))
    Npatch_by_spin = {s: patchset_for_spin(patchsets, s).Npatch for s in spins}
    if len(set(Npatch_by_spin.values())) != 1:
        raise ValueError("SharedGeometry currently requires identical patch counts across available spin sectors.")
    Npatch = Npatch_by_spin[spins[0]]
    patch_k = {
        s: _canonicalize_patch_k_array(
            np.asarray([p.k_cart for p in patchset_for_spin(patchsets, s).patches], dtype=float),
            patchsets,
        )
        for s in spins
    }
    energies_by_spin = {
        s: np.asarray([float(p.energy) for p in patchset_for_spin(patchsets, s).patches], dtype=float)
        for s in spins
    }

    if pp_Qs is None:
        pp_Qs = build_unique_q_list(patchsets, mode="pp", decimals=q_key_decimals, merge_tol_red=q_merge_tol_red)
    if ph_Qs is None:
        ph_Qs = build_unique_q_list(patchsets, mode="ph", decimals=q_key_decimals, merge_tol_red=q_merge_tol_red)
    if phc_Qs is None:
        phc_Qs = build_unique_q_list(patchsets, mode="phc", decimals=q_key_decimals, merge_tol_red=q_merge_tol_red)

    pp_grid = TransferGrid(patchsets, list(pp_Qs), decimals=q_key_decimals, merge_tol_red=q_merge_tol_red)
    phd_grid = TransferGrid(patchsets, list(ph_Qs), decimals=q_key_decimals, merge_tol_red=q_merge_tol_red)
    phc_grid = TransferGrid(patchsets, list(phc_Qs), decimals=q_key_decimals, merge_tol_red=q_merge_tol_red)

    pp_q_index: Dict[Tuple[str, str], np.ndarray] = {}
    phd_q_index: Dict[Tuple[str, str], np.ndarray] = {}
    phc_q_index: Dict[Tuple[str, str], np.ndarray] = {}
    for s1 in spins:
        k1s = patch_k[s1]
        for s2 in spins:
            k2s = patch_k[s2]
            arr_pp = np.zeros((Npatch, Npatch), dtype=int)
            arr_ph = np.zeros((Npatch, Npatch), dtype=int)
            for p1, k1 in enumerate(k1s):
                for p2, k2 in enumerate(k2s):
                    arr_pp[p1, p2] = pp_grid.nearest_index(k1 + k2)
                    arr_ph[p1, p2] = phd_grid.nearest_index(k1 - k2)
            pp_q_index[(s1, s2)] = arr_pp
            phd_q_index[(s1, s2)] = arr_ph
            phc_q_index[(s1, s2)] = arr_ph.copy()

    def _partner_map_pp_from_iq(iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            patchsets,
            pp_q_index[(normalize_spin(first_spin), normalize_spin(second_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=pp_grid.canonicalize(Q),
            mode="Q_minus_k",
        )

    def _partner_map_phd_from_iq(iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float]):
        return partner_map_from_q_index(
            patchsets,
            phd_q_index[(normalize_spin(second_spin), normalize_spin(first_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=phd_grid.canonicalize(Q),
            mode="k_plus_Q",
        )

    def _partner_map_phc_from_iq(iq: int, *, first_spin: str, second_spin: str, Q: Sequence[float], mode: str):
        return partner_map_from_q_index(
            patchsets,
            phc_q_index[(normalize_spin(second_spin), normalize_spin(first_spin))],
            source_spin=first_spin,
            target_spin=second_spin,
            iq_target=int(iq),
            Q=phc_grid.canonicalize(Q),
            mode=mode,
        )

    pp_qminus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
    ph_kplus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
    phc_kplus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
    phc_kminus: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for iq, Q in enumerate(pp_grid.q_list):
        for s1 in spins:
            for s2 in spins:
                pp_qminus[(iq, s1, s2)] = _partner_map_pp_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q)
    for iq, Q in enumerate(phd_grid.q_list):
        for s1 in spins:
            for s2 in spins:
                ph_kplus[(iq, s1, s2)] = _partner_map_phd_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q)
    for iq, Q in enumerate(phc_grid.q_list):
        for s1 in spins:
            for s2 in spins:
                phc_kplus[(iq, s1, s2)] = _partner_map_phd_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q)
                phc_kminus[(iq, s1, s2)] = _partner_map_phc_from_iq(iq, first_spin=s1, second_spin=s2, Q=Q, mode="k_minus_Q")

    closure_map: Dict[SpinBlock, Tuple[np.ndarray, np.ndarray]] = {}
    for key in spin_blocks_can:
        s1, s2, s3, s4 = key
        p4_idx = np.full((Npatch, Npatch, Npatch), -1, dtype=int)
        p4_res = np.full((Npatch, Npatch, Npatch), np.inf, dtype=float)
        target_cache: Dict[Tuple[int, int], np.ndarray] = {}
        for p1, k1 in enumerate(patch_k[s1]):
            for p2, k2 in enumerate(patch_k[s2]):
                target_cache[(p1, p2)] = canonicalize_q_for_patchsets(patchsets, k1 + k2)
        for p1 in range(Npatch):
            for p2 in range(Npatch):
                total = target_cache[(p1, p2)]
                for p3, k3 in enumerate(patch_k[s3]):
                    target_k4 = canonicalize_q_for_patchsets(patchsets, total - k3)
                    idx, dist = _find_shifted_patch_index(patchsets, s4, target_k4)
                    p4_idx[p1, p2, p3] = idx
                    p4_res[p1, p2, p3] = dist
        closure_map[key] = (p4_idx, p4_res)

    return SharedGeometry(
        patchsets=patchsets,
        spin_blocks=spin_blocks_can,
        q_merge_tol_red=float(q_merge_tol_red),
        q_key_decimals=int(q_key_decimals),
        spins=spins,
        internal_spin_pairs=internal_spin_pairs,
        Npatch=Npatch,
        patch_k=patch_k,
        energies_by_spin=energies_by_spin,
        pp_grid=pp_grid,
        phd_grid=phd_grid,
        phc_grid=phc_grid,
        pp_q_index=pp_q_index,
        phd_q_index=phd_q_index,
        phc_q_index=phc_q_index,
        pp_qminus=pp_qminus,
        ph_kplus=ph_kplus,
        phc_kplus=phc_kplus,
        phc_kminus=phc_kminus,
        closure_map=closure_map,
    )


__all__ = [
    "SharedGeometry",
    "TransferGrid",
    "available_physical_spins",
    "build_shared_geometry",
    "build_unique_q_list",
    "canonical_spin_tuple",
    "default_spin_blocks",
]
