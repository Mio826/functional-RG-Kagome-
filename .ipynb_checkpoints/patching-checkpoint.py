from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Standard patch data structures
# =============================================================================


@dataclass
class PatchPoint:
    patch_id: int
    k_cart: np.ndarray
    k_red: np.ndarray
    energy: float
    vF: np.ndarray
    vF_norm: float
    eigvec: np.ndarray
    orbital_weight: np.ndarray

    tangent: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None

    fs_arc_length: Optional[float] = None
    weight_length: Optional[float] = None
    weight_length_over_vf: Optional[float] = None


@dataclass
class PatchSet:
    mu: float
    mu_used_for_contour: float
    band_index: int
    filling: float
    patches: List[PatchPoint]

    fs_contour_k: np.ndarray
    bz_vertices: np.ndarray
    b1: np.ndarray
    b2: np.ndarray

    gauge_method: Optional[str] = None
    gauge_loop_phase: float = 0.0
    name: Optional[str] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def Npatch(self) -> int:
        return len(self.patches)

    @property
    def patch_k(self) -> np.ndarray:
        return np.array([p.k_cart for p in self.patches], dtype=float)

    @property
    def patch_k_red(self) -> np.ndarray:
        return np.array([p.k_red for p in self.patches], dtype=float)

    @property
    def patch_energy(self) -> np.ndarray:
        return np.array([p.energy for p in self.patches], dtype=float)

    @property
    def patch_vF(self) -> np.ndarray:
        return np.array([p.vF for p in self.patches], dtype=float)

    @property
    def patch_vF_norm(self) -> np.ndarray:
        return np.array([p.vF_norm for p in self.patches], dtype=float)

    @property
    def patch_weight(self) -> np.ndarray:
        return np.array([p.orbital_weight for p in self.patches], dtype=float)

    @property
    def patch_eigvec(self) -> np.ndarray:
        return np.array([p.eigvec for p in self.patches], dtype=complex)

    @property
    def patch_arc_length(self) -> np.ndarray:
        return np.array(
            [0.0 if p.fs_arc_length is None else p.fs_arc_length for p in self.patches],
            dtype=float,
        )

    @property
    def patch_weight_length(self) -> np.ndarray:
        return np.array(
            [0.0 if p.weight_length is None else p.weight_length for p in self.patches],
            dtype=float,
        )

    @property
    def patch_weight_length_over_vf(self) -> np.ndarray:
        return np.array(
            [
                0.0 if p.weight_length_over_vf is None else p.weight_length_over_vf
                for p in self.patches
            ],
            dtype=float,
        )

    def copy_with_eigvecs(
        self,
        eigvecs: np.ndarray,
        *,
        gauge_method: Optional[str] = None,
        gauge_loop_phase: Optional[float] = None,
    ) -> "PatchSet":
        eigvecs = np.asarray(eigvecs, dtype=complex)
        if eigvecs.shape != self.patch_eigvec.shape:
            raise ValueError(
                f"eigvecs must have shape {self.patch_eigvec.shape}, got {eigvecs.shape}"
            )

        new_patches: List[PatchPoint] = []
        for p, u in zip(self.patches, eigvecs):
            uu = _normalize_eigvec(u)
            new_patches.append(
                replace(
                    p,
                    eigvec=uu,
                    orbital_weight=_orbital_weight(uu),
                )
            )

        return PatchSet(
            mu=self.mu,
            mu_used_for_contour=self.mu_used_for_contour,
            band_index=self.band_index,
            filling=self.filling,
            patches=new_patches,
            fs_contour_k=self.fs_contour_k.copy(),
            bz_vertices=self.bz_vertices.copy(),
            b1=self.b1.copy(),
            b2=self.b2.copy(),
            gauge_method=self.gauge_method if gauge_method is None else gauge_method,
            gauge_loop_phase=(
                self.gauge_loop_phase if gauge_loop_phase is None else gauge_loop_phase
            ),
            name=self.name,
            model_name=self.model_name,
            metadata=dict(self.metadata),
        )


# =============================================================================
# Reciprocal helpers
# =============================================================================


def _wrap_unit_interval(x, tol=1e-12):
    x = np.asarray(x, dtype=float)
    x = x - np.floor(x)
    x[np.isclose(x, 1.0, atol=tol)] = 0.0
    x[np.isclose(x, 0.0, atol=tol)] = 0.0
    return x


def _B(model) -> np.ndarray:
    return np.column_stack([
        np.asarray(model.b1, dtype=float),
        np.asarray(model.b2, dtype=float),
    ])


def _cart_to_red(model, k):
    return np.linalg.solve(_B(model), np.asarray(k, dtype=float))


def _red_to_cart(model, uv):
    return _B(model) @ np.asarray(uv, dtype=float)


def _canonicalize_k_mod_G(model, k):
    """
    Parallelogram fundamental-domain rep.
    只用于 modulo-G 等价类检查，不用于 centered 1BZ 可视化。
    """
    uv = _cart_to_red(model, k)
    uv = _wrap_unit_interval(uv)
    k_can = _red_to_cart(model, uv)
    k_can[np.isclose(k_can, 0.0, atol=1e-12)] = 0.0
    return k_can


def _minimum_image_displacement(k_target, k_ref, model) -> np.ndarray:
    k_target = np.asarray(k_target, dtype=float)
    k_ref = np.asarray(k_ref, dtype=float)
    b1 = np.asarray(model.b1, dtype=float)
    b2 = np.asarray(model.b2, dtype=float)

    best = None
    best_norm = np.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            disp = k_target - (k_ref + n1 * b1 + n2 * b2)
            nd = np.linalg.norm(disp)
            if nd < best_norm:
                best_norm = nd
                best = disp
    return np.asarray(best, dtype=float)


def _periodic_distance(k1, k2, model) -> float:
    return float(np.linalg.norm(_minimum_image_displacement(k1, k2, model)))


# =============================================================================
# 1BZ geometry: centered hexagon, consistent with notebook helper
# =============================================================================


def hex_bz_vertices(model):
    """
    Standard centered 1BZ hexagon vertices for triangular/kagome reciprocal lattice.

    For reciprocal vectors b1, b2:
        ±(2b1-b2)/3, ±(b1+b2)/3, ±(-b1+2b2)/3
    """
    b1 = np.asarray(model.b1, dtype=float)
    b2 = np.asarray(model.b2, dtype=float)

    verts = np.array([
        (2 * b1 - b2) / 3.0,
        (b1 + b2) / 3.0,
        (-b1 + 2 * b2) / 3.0,
        -(2 * b1 - b2) / 3.0,
        -(b1 + b2) / 3.0,
        -(-b1 + 2 * b2) / 3.0,
    ], dtype=float)

    ang = np.arctan2(verts[:, 1], verts[:, 0])
    order = np.argsort(ang)
    return verts[order]


def exact_M6_points_1bz(model):
    """
    6 geometric M points on the boundary of the centered 1BZ hexagon.
    """
    V = hex_bz_vertices(model)
    M = []

    n = len(V)
    for i in range(n):
        v0 = V[i]
        v1 = V[(i + 1) % n]
        M.append(0.5 * (v0 + v1))

    M = np.asarray(M, dtype=float)
    ang = np.arctan2(M[:, 1], M[:, 0])
    order = np.argsort(ang)
    return M[order]


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, tol: float = 1e-10) -> bool:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)

    sign = None
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = np.cross(b - a, p - a)
        if abs(cross) <= tol:
            continue
        this_sign = cross > 0
        if sign is None:
            sign = this_sign
        elif sign != this_sign:
            return False
    return True


def canonicalize_k_to_centered_1bz(model, k, *, search_range: int = 2) -> np.ndarray:
    """
    Choose a reciprocal-lattice-shifted representative in the centered hexagonal 1BZ.
    """
    k = np.asarray(k, dtype=float)
    b1 = np.asarray(model.b1, dtype=float)
    b2 = np.asarray(model.b2, dtype=float)
    poly = hex_bz_vertices(model)

    candidates = []
    for n1 in range(-search_range, search_range + 1):
        for n2 in range(-search_range, search_range + 1):
            kc = k - n1 * b1 - n2 * b2
            if _point_in_convex_polygon(kc, poly, tol=1e-10):
                candidates.append(kc)

    if len(candidates) == 0:
        best = None
        best_norm = np.inf
        for n1 in range(-search_range, search_range + 1):
            for n2 in range(-search_range, search_range + 1):
                kc = k - n1 * b1 - n2 * b2
                nd = float(np.linalg.norm(kc))
                if nd < best_norm:
                    best = kc
                    best_norm = nd
        out = np.asarray(best, dtype=float)
    else:
        out = np.asarray(min(candidates, key=lambda x: float(np.linalg.norm(x))), dtype=float)

    out[np.isclose(out, 0.0, atol=1e-12)] = 0.0
    return out


# =============================================================================
# Notebook-consistent exact-M loop generators
# =============================================================================


def exact_M_hex_loop_points(model, points_per_edge=1):
    """
    Per-edge interpolation, same naming/logic as your helper txt.

    points_per_edge = 1  ->  6 patches
    points_per_edge = 2  -> 12 patches
    points_per_edge = 4  -> 24 patches
    """
    if points_per_edge < 1:
        raise ValueError("points_per_edge must be >= 1")

    M6 = exact_M6_points_1bz(model)
    pts = []

    n = len(M6)
    for i in range(n):
        k0 = M6[i]
        k1 = M6[(i + 1) % n]
        for m in range(points_per_edge):
            t = m / points_per_edge
            k = (1.0 - t) * k0 + t * k1
            pts.append(k)

    return np.asarray(pts, dtype=float)


def _ray_polygon_first_intersection(theta, vertices, *, tol=1e-12):
    vertices = np.asarray(vertices, dtype=float)
    e = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    hits = []
    nv = len(vertices)
    for i in range(nv):
        A = vertices[i]
        B = vertices[(i + 1) % nv]
        d = B - A

        M = np.column_stack([e, -d])
        det = np.linalg.det(M)
        if abs(det) < tol:
            continue

        r, t = np.linalg.solve(M, A)
        if r >= -tol and (-tol <= t <= 1 + tol):
            r = max(r, 0.0)
            t = min(max(t, 0.0), 1.0)
            P = A + t * d
            hits.append((r, i, P))

    if not hits:
        raise ValueError(f"No polygon intersection found for theta={theta}")

    hits.sort(key=lambda x: x[0])
    rmin, edge_index, P = hits[0]
    return np.asarray(P, dtype=float), int(edge_index), float(rmin)


def exact_M_hex_loop_points_global_angular(model, points_per_edge=1, *, angle_offset=0.0):
    """
    Global-uniform angular sampling, same spirit as your helper txt.
    """
    p = int(points_per_edge)
    if p <= 0:
        raise ValueError("points_per_edge must be >= 1")

    M6 = np.asarray(exact_M6_points_1bz(model), dtype=float)
    th_v = np.arctan2(M6[:, 1], M6[:, 0])
    order = np.argsort(th_v)
    poly = M6[order]

    N = 6 * p
    theta0 = float(np.arctan2(poly[0, 1], poly[0, 0])) + float(angle_offset)
    thetas = theta0 + 2.0 * np.pi * np.arange(N) / N

    pts = []
    for th in thetas:
        P, _, _ = _ray_polygon_first_intersection(th, poly)
        pts.append(P)

    return np.asarray(pts, dtype=float)


# =============================================================================
# duplicated-M removal (same notebook semantics)
# =============================================================================


def _patch_keep_indices_remove_duplicate_M_anchors(model, points_per_edge, tol=1e-10):
    """
    Same behavior as your helper txt:
    only the 6 anchor points are checked modulo G.
    """
    p = int(points_per_edge)
    n_full = 6 * p
    keep = np.ones(n_full, dtype=bool)

    anchor_ids = [i * p for i in range(6)]
    M6 = exact_M6_points_1bz(model)

    seen_classes = []
    for local_anchor_idx, global_idx in enumerate(anchor_ids):
        k = M6[local_anchor_idx]
        k_can = _canonicalize_k_mod_G(model, k)

        duplicated = False
        for q_can in seen_classes:
            if np.linalg.norm(k_can - q_can) < tol:
                duplicated = True
                break

        if duplicated:
            keep[global_idx] = False
        else:
            seen_classes.append(k_can)

    return np.flatnonzero(keep)


def _patch_keep_indices_exclude_strict_M_anchors(points_per_edge):
    p = int(points_per_edge)
    if p < 2:
        raise ValueError(
            "keep_strict_M_anchors=False requires points_per_edge >= 2, "
            "because for points_per_edge=1 all loop points are strict M anchors."
        )

    n_full = 6 * p
    keep = np.ones(n_full, dtype=bool)
    strict_M_ids = [i * p for i in range(6)]
    keep[strict_M_ids] = False
    return np.flatnonzero(keep)


# =============================================================================
# Sector eigensystem / velocity / FS projection
# =============================================================================


def _sector_hamiltonian(model, k, orbital_slice):
    k = np.asarray(k, dtype=float)
    H = np.asarray(model.Hk(k[0], k[1]), dtype=complex)
    if orbital_slice is None:
        return H
    return H[orbital_slice, orbital_slice]


def _sector_eig(model, k, orbital_slice, band_index):
    H = _sector_hamiltonian(model, k, orbital_slice)
    evals, evecs = np.linalg.eigh(H)
    if not (0 <= band_index < len(evals)):
        raise ValueError(
            f"band_index={band_index} out of range for sector with {len(evals)} bands"
        )
    return float(evals[band_index]), np.asarray(evecs[:, band_index], dtype=complex)


def _finite_diff_velocity(model, k, orbital_slice, band_index, h=1e-5):
    k = np.asarray(k, dtype=float)

    ex1, _ = _sector_eig(model, [k[0] + h, k[1]], orbital_slice, band_index)
    ex2, _ = _sector_eig(model, [k[0] - h, k[1]], orbital_slice, band_index)
    ey1, _ = _sector_eig(model, [k[0], k[1] + h], orbital_slice, band_index)
    ey2, _ = _sector_eig(model, [k[0], k[1] - h], orbital_slice, band_index)

    return np.array([(ex1 - ex2) / (2.0 * h), (ey1 - ey2) / (2.0 * h)], dtype=float)


def project_to_fs(
    model,
    k0,
    orbital_slice,
    band_index,
    *,
    mu=0.0,
    nstep=40,
    tol=1e-12,
    fd_step=1e-5,
):
    """
    Newton projection onto E(k)=mu using the velocity direction.

    这是这次相对 notebook helper 的“刻意升级”：
    你的 helper 本身只给 manual loop，并不保证 E=mu；
    但你现在明确希望 patch energy 尽量是 0，所以这里默认加上。
    """
    k = canonicalize_k_to_centered_1bz(model, k0)

    for _ in range(nstep):
        e, _ = _sector_eig(model, k, orbital_slice, band_index)
        de = e - mu
        if abs(de) < tol:
            return canonicalize_k_to_centered_1bz(model, k)

        v = _finite_diff_velocity(model, k, orbital_slice, band_index, h=fd_step)
        vv = float(np.dot(v, v))
        if vv < 1e-20:
            return canonicalize_k_to_centered_1bz(model, k)

        k = k - (de / vv) * v
        k = canonicalize_k_to_centered_1bz(model, k)

    return canonicalize_k_to_centered_1bz(model, k)


def _normalize_eigvec(u):
    u = np.asarray(u, dtype=complex)
    nrm = np.linalg.norm(u)
    if nrm == 0:
        raise ValueError("Encountered zero-norm eigenvector.")
    return u / nrm


def _orbital_weight(u):
    u = np.asarray(u, dtype=complex)
    w = np.abs(u) ** 2
    s = float(np.sum(w))
    return w / s if s > 0 else w


# =============================================================================
# Gauge fixing
# =============================================================================


def _anchor_phase(u, method="max_component"):
    u = _normalize_eigvec(u)
    if method == "max_component":
        idx = int(np.argmax(np.abs(u)))
        if np.abs(u[idx]) > 0:
            u = u * np.exp(-1j * np.angle(u[idx]))
    elif method == "first_component":
        if np.abs(u[0]) > 0:
            u = u * np.exp(-1j * np.angle(u[0]))
    else:
        raise ValueError("method must be 'max_component' or 'first_component'")
    return u


def smooth_patch_eigvecs(eigvecs, *, close_loop=True, anchor_method="max_component"):
    U = np.asarray(eigvecs, dtype=complex).copy()
    if U.ndim != 2:
        raise ValueError("eigvecs must have shape (Npatch, Norb).")

    N = U.shape[0]
    if N == 0:
        return U, 0.0

    U[0] = _anchor_phase(U[0], method=anchor_method)

    for p in range(1, N):
        U[p] = _normalize_eigvec(U[p])
        ov = np.vdot(U[p - 1], U[p])
        if np.abs(ov) > 1e-14:
            U[p] *= np.exp(-1j * np.angle(ov))
        else:
            U[p] = _anchor_phase(U[p], method=anchor_method)

    loop_phase = 0.0
    if N > 1:
        ov_last = np.vdot(U[-1], U[0])
        if np.abs(ov_last) > 1e-14:
            loop_phase = float(np.angle(ov_last))

    if close_loop and N > 1 and np.abs(loop_phase) > 1e-14:
        for p in range(N):
            U[p] *= np.exp(1j * (p / N) * loop_phase)

        U[0] = _anchor_phase(U[0], method=anchor_method)
        for p in range(1, N):
            ov = np.vdot(U[p - 1], U[p])
            if np.abs(ov) > 1e-14:
                U[p] *= np.exp(-1j * np.angle(ov))

    for p in range(N):
        U[p] = _normalize_eigvec(U[p])

    return U, loop_phase


# =============================================================================
# Patch geometry on the kept loop
# =============================================================================


def _compute_patch_arc_lengths(k_points: np.ndarray, model) -> np.ndarray:
    k_points = np.asarray(k_points, dtype=float)
    n = len(k_points)
    out = np.zeros(n, dtype=float)

    if n == 0:
        return out
    if n == 1:
        return out

    for p in range(n):
        km = k_points[(p - 1) % n]
        k0 = k_points[p]
        kp = k_points[(p + 1) % n]

        d_left = _periodic_distance(k0, km, model)
        d_right = _periodic_distance(kp, k0, model)
        out[p] = 0.5 * (d_left + d_right)

    return out


def _compute_tangent_normal(k_points: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
    k_points = np.asarray(k_points, dtype=float)
    n = len(k_points)

    tangents = np.zeros((n, 2), dtype=float)
    normals = np.zeros((n, 2), dtype=float)

    if n == 0:
        return tangents, normals

    for p in range(n):
        km = k_points[(p - 1) % n]
        kp = k_points[(p + 1) % n]

        dm = _minimum_image_displacement(k_points[p], km, model)
        dp = _minimum_image_displacement(kp, k_points[p], model)

        t = dm + dp
        nt = np.linalg.norm(t)
        if nt < 1e-14:
            t = dp if np.linalg.norm(dp) > np.linalg.norm(dm) else dm
            nt = np.linalg.norm(t)

        if nt < 1e-14:
            tangents[p] = np.array([1.0, 0.0], dtype=float)
            normals[p] = np.array([0.0, 1.0], dtype=float)
            continue

        t = t / nt
        tangents[p] = t
        normals[p] = np.array([-t[1], t[0]], dtype=float)

    return tangents, normals


# =============================================================================
# Internal builder
# =============================================================================


def _build_patchset_from_loop(
    model,
    K_full,
    orbital_slice,
    band_index,
    *,
    points_per_edge=1,
    remove_duplicate_M_modG=False,
    keep_strict_M_anchors=True,
    gauge_fix=True,
    close_loop_gauge=True,
    gauge_anchor="max_component",
    project_to_fs_points=True,
    mu=0.0,
    fd_step=1e-5,
    builder_name="manual_exact_M_hex",
):
    if not keep_strict_M_anchors:
        keep_idx = _patch_keep_indices_exclude_strict_M_anchors(points_per_edge)
    elif remove_duplicate_M_modG:
        keep_idx = _patch_keep_indices_remove_duplicate_M_anchors(
            model,
            points_per_edge=points_per_edge,
        )
    else:
        keep_idx = np.arange(len(K_full))

    K_patch0 = np.asarray(K_full[keep_idx], dtype=float)

    if project_to_fs_points:
        K_patch = np.asarray(
            [
                project_to_fs(
                    model,
                    k,
                    orbital_slice,
                    band_index,
                    mu=mu,
                    fd_step=fd_step,
                )
                for k in K_patch0
            ],
            dtype=float,
        )
    else:
        K_patch = np.asarray(
            [canonicalize_k_to_centered_1bz(model, k) for k in K_patch0],
            dtype=float,
        )

    # for plotting, canonicalize the full loop to the centered 1BZ too
    K_full_plot = np.asarray(
        [canonicalize_k_to_centered_1bz(model, k) for k in K_full],
        dtype=float,
    )

    bz_vertices = hex_bz_vertices(model)

    raw_eigvecs = []
    energies = []
    velocities = []

    for k in K_patch:
        e, u = _sector_eig(model, k, orbital_slice, band_index)
        vF = _finite_diff_velocity(model, k, orbital_slice, band_index, h=fd_step)

        raw_eigvecs.append(u)
        energies.append(e)
        velocities.append(vF)

    raw_eigvecs = np.asarray(raw_eigvecs, dtype=complex)

    if gauge_fix:
        fixed_eigvecs, loop_phase = smooth_patch_eigvecs(
            raw_eigvecs,
            close_loop=close_loop_gauge,
            anchor_method=gauge_anchor,
        )
        gauge_method = f"{builder_name}_parallel_transport"
    else:
        fixed_eigvecs = np.asarray(
            [_normalize_eigvec(u) for u in raw_eigvecs],
            dtype=complex,
        )
        loop_phase = 0.0
        gauge_method = f"{builder_name}_raw"

    arc = _compute_patch_arc_lengths(K_patch, model)
    tangent, normal = _compute_tangent_normal(K_patch, model)

    patches = []
    for pid, (k, e, vF, u, ell, t, n) in enumerate(
        zip(K_patch, energies, velocities, fixed_eigvecs, arc, tangent, normal)
    ):
        vf_norm = float(np.linalg.norm(vF))
        patches.append(
            PatchPoint(
                patch_id=pid,
                k_cart=np.asarray(k, dtype=float),
                k_red=_cart_to_red(model, k),
                energy=float(e),
                vF=np.asarray(vF, dtype=float),
                vF_norm=vf_norm,
                eigvec=np.asarray(u, dtype=complex),
                orbital_weight=_orbital_weight(u),
                tangent=np.asarray(t, dtype=float),
                normal=np.asarray(n, dtype=float),
                fs_arc_length=float(ell),
                weight_length=float(ell),
                weight_length_over_vf=float(ell / vf_norm) if vf_norm > 1e-14 else np.inf,
            )
        )

    npatch = len(K_patch)
    suffix_dup = "_dropDupM" if (keep_strict_M_anchors and remove_duplicate_M_modG) else ""
    suffix_noM = "_noStrictM" if (not keep_strict_M_anchors) else ""
    suffix_g = "_gaugeFixed" if gauge_fix else "_rawGauge"
    suffix_fs = "_projFS" if project_to_fs_points else "_rawLoop"

    return PatchSet(
        mu=float(mu),
        mu_used_for_contour=float(mu),
        band_index=int(band_index),
        filling=np.nan,
        patches=patches,
        # fs_contour_k=np.asarray(K_full_plot, dtype=float),
        fs_contour_k=np.asarray(np.asarray(exact_M6_points_1bz(model), dtype=float), dtype=float),
        bz_vertices=np.asarray(bz_vertices, dtype=float),
        b1=np.asarray(model.b1, dtype=float),
        b2=np.asarray(model.b2, dtype=float),
        gauge_method=f"{gauge_method}_{npatch}{suffix_dup}{suffix_noM}{suffix_g}{suffix_fs}",
        gauge_loop_phase=float(loop_phase),
        name=f"{builder_name}_{npatch}",
        model_name=type(model).__name__,
        metadata={
            "builder": builder_name,
            "points_per_edge": int(points_per_edge),
            "remove_duplicate_M_modG": bool(remove_duplicate_M_modG),
            "keep_strict_M_anchors": bool(keep_strict_M_anchors),
            "project_to_fs_points": bool(project_to_fs_points),
            "kept_indices": keep_idx.tolist(),
        },
    )


# =============================================================================
# Public builders
# =============================================================================


def build_exactM_patchset(
    model,
    orbital_slice,
    band_index,
    *,
    points_per_edge=1,
    remove_duplicate_M_modG=False,
    keep_strict_M_anchors=True,
    gauge_fix=True,
    close_loop_gauge=True,
    gauge_anchor="max_component",
    project_to_fs_points=True,
    mu=0.0,
    fd_step=1e-5,
):
    """
    Notebook-consistent manual exact-M builder using per-edge interpolation,
    plus two deliberate fixes:
      1. centered-1BZ canonicalization for plotting/representatives
      2. optional FS projection of kept patch reps
    """
    K_full = exact_M_hex_loop_points(model, points_per_edge=points_per_edge)
    return _build_patchset_from_loop(
        model,
        K_full,
        orbital_slice,
        band_index,
        points_per_edge=points_per_edge,
        remove_duplicate_M_modG=remove_duplicate_M_modG,
        keep_strict_M_anchors=keep_strict_M_anchors,
        gauge_fix=gauge_fix,
        close_loop_gauge=close_loop_gauge,
        gauge_anchor=gauge_anchor,
        project_to_fs_points=project_to_fs_points,
        mu=mu,
        fd_step=fd_step,
        builder_name="manual_exact_M_hex",
    )


def build_exactM_patchset_global_angular(
    model,
    orbital_slice,
    band_index,
    *,
    points_per_edge=1,
    remove_duplicate_M_modG=False,
    keep_strict_M_anchors=True,
    gauge_fix=True,
    close_loop_gauge=True,
    gauge_anchor="max_component",
    project_to_fs_points=True,
    mu=0.0,
    fd_step=1e-5,
):
    K_full = exact_M_hex_loop_points_global_angular(
        model,
        points_per_edge=points_per_edge,
    )
    return _build_patchset_from_loop(
        model,
        K_full,
        orbital_slice,
        band_index,
        points_per_edge=points_per_edge,
        remove_duplicate_M_modG=remove_duplicate_M_modG,
        keep_strict_M_anchors=keep_strict_M_anchors,
        gauge_fix=gauge_fix,
        close_loop_gauge=close_loop_gauge,
        gauge_anchor=gauge_anchor,
        project_to_fs_points=project_to_fs_points,
        mu=mu,
        fd_step=fd_step,
        builder_name="manual_exact_M_hex_globalAngular",
    )


# =============================================================================
# Plotting
# =============================================================================


def plot_patchset(patchset, ax=None, show_contour=True, show_velocity=False, show_bz=True):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if show_bz and patchset.bz_vertices is not None:
        bz = np.vstack([patchset.bz_vertices, patchset.bz_vertices[0]])
        ax.plot(bz[:, 0], bz[:, 1], color="black", lw=1.2, alpha=0.8, label="1st BZ")

    if show_contour and patchset.fs_contour_k is not None:
        kk = np.asarray(patchset.fs_contour_k, dtype=float)
        kk2 = np.vstack([kk, kk[0]])
        ax.plot(kk2[:, 0], kk2[:, 1], lw=1.3, alpha=0.85, label="manual M-loop")

    pk = patchset.patch_k
    ax.scatter(pk[:, 0], pk[:, 1], s=28, label="patch reps", zorder=3)

    for p in patchset.patches:
        ax.text(p.k_cart[0], p.k_cart[1], str(p.patch_id), fontsize=7)

    if show_velocity:
        v = patchset.patch_vF
        ax.quiver(
            pk[:, 0],
            pk[:, 1],
            v[:, 0],
            v[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
        )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.legend()
    return ax