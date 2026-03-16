import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


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

    @property
    def Npatch(self) -> int:
        return len(self.patches)

    @property
    def patch_k(self) -> np.ndarray:
        return np.array([p.k_cart for p in self.patches])

    @property
    def patch_vF(self) -> np.ndarray:
        return np.array([p.vF for p in self.patches])

    @property
    def patch_weight(self) -> np.ndarray:
        return np.array([p.orbital_weight for p in self.patches])


class FSPatcher:
    """
    Unified first-BZ contour-based patcher for 2D kagome-like models.

    Robust features:
      - Always works inside the hexagonal 1st BZ
      - Scores contours by closure / length / enclosure of target point
      - Automatically tries tiny contour-level shifts to avoid van Hove degeneracies
    """

    def __init__(
        self,
        model,
        *,
        band_index: int,
        filling: Optional[float] = None,
        mu: Optional[float] = None,
        grid_size: int = 320,
        Npatch: int = 48,
        fd_step: float = 1e-4,
        orbital_slice: Optional[slice] = None,
        contour_min_points: int = 40,
        contour_target_k: Optional[np.ndarray] = None,
        auto_level_shifts: Optional[List[float]] = None,
        prefer_closed_contour: bool = True,
        verbose: bool = False,
    ):
        self.model = model
        self.band_index = band_index
        self.filling = filling
        self.mu = mu
        self.grid_size = grid_size
        self.Npatch = Npatch
        self.fd_step = fd_step
        self.orbital_slice = orbital_slice
        self.contour_min_points = contour_min_points
        self.contour_target_k = None if contour_target_k is None else np.asarray(contour_target_k, dtype=float)
        self.prefer_closed_contour = prefer_closed_contour
        self.verbose = verbose

        if auto_level_shifts is None:
            self.auto_level_shifts = [0.0, 1e-4, -1e-4, 5e-4, -5e-4]
        else:
            self.auto_level_shifts = list(auto_level_shifts)

        if (filling is None) == (mu is None):
            raise ValueError("Exactly one of filling or mu must be provided.")

    # ------------------------
    # reciprocal helpers
    # ------------------------
    @property
    def b1(self) -> np.ndarray:
        return np.asarray(self.model.b1, dtype=float)

    @property
    def b2(self) -> np.ndarray:
        return np.asarray(self.model.b2, dtype=float)

    @property
    def Bmat(self) -> np.ndarray:
        return np.column_stack([self.b1, self.b2])

    @property
    def Binv(self) -> np.ndarray:
        return np.linalg.inv(self.Bmat)

    def red_to_cart(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        return self.Bmat @ uv

    def cart_to_red(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        return self.Binv @ k

    def wrap_red(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        return uv - np.floor(uv)

    def wrap_cart(self, k: np.ndarray) -> np.ndarray:
        return self.red_to_cart(self.wrap_red(self.cart_to_red(k)))

    # ------------------------
    # band / eigvec / mu
    # ------------------------
    def band_energy(self, kx: float, ky: float) -> float:
        evals, _ = self.model.eigenstate(kx, ky)
        return float(evals[self.band_index])

    def band_eigvec(self, kx: float, ky: float) -> np.ndarray:
        _, evecs = self.model.eigenstate(kx, ky)
        return np.asarray(evecs[:, self.band_index])

    def get_mu(self) -> float:
        if self.mu is not None:
            return float(self.mu)
        return float(self.model.EF_from_filling(self.filling))

    # ------------------------
    # velocity / orbital weight
    # ------------------------
    def fermi_velocity(self, kx: float, ky: float) -> np.ndarray:
        h = self.fd_step
        ex1 = self.band_energy(kx + h, ky)
        ex2 = self.band_energy(kx - h, ky)
        ey1 = self.band_energy(kx, ky + h)
        ey2 = self.band_energy(kx, ky - h)
        vx = (ex1 - ex2) / (2.0 * h)
        vy = (ey1 - ey2) / (2.0 * h)
        return np.array([vx, vy], dtype=float)

    def orbital_weight(self, eigvec: np.ndarray) -> np.ndarray:
        vec = eigvec if self.orbital_slice is None else eigvec[self.orbital_slice]
        w = np.abs(vec) ** 2
        s = np.sum(w)
        if s > 0:
            w = w / s
        return w

    # ------------------------
    # refine point to exact FS
    # ------------------------
    def project_to_fs(self, k0: np.ndarray, mu: float, nstep: int = 8) -> np.ndarray:
        k = np.array(k0, dtype=float)
        for _ in range(nstep):
            e = self.band_energy(k[0], k[1])
            v = self.fermi_velocity(k[0], k[1])
            vv = np.dot(v, v)
            if vv < 1e-14:
                break
            k = k - ((e - mu) / vv) * v
        return k

    # ------------------------
    # first BZ geometry
    # ------------------------
    def first_bz_vertices(self) -> np.ndarray:
        Gs = []
        for n1 in [-1, 0, 1]:
            for n2 in [-1, 0, 1]:
                if n1 == 0 and n2 == 0:
                    continue
                G = n1 * self.b1 + n2 * self.b2
                Gs.append(G)
        Gs = np.array(Gs, dtype=float)

        norms = np.linalg.norm(Gs, axis=1)
        min_norm = np.min(norms)
        close = Gs[np.abs(norms - min_norm) < 1e-8]

        verts = []
        for i in range(len(close)):
            for j in range(i + 1, len(close)):
                G1 = close[i]
                G2 = close[j]
                A = np.array([G1, G2], dtype=float)
                if abs(np.linalg.det(A)) < 1e-12:
                    continue
                rhs = np.array([np.dot(G1, G1) / 2.0, np.dot(G2, G2) / 2.0], dtype=float)
                k = np.linalg.solve(A, rhs)

                good = True
                for G in close:
                    if np.dot(k, G) - np.dot(G, G) / 2.0 > 1e-8:
                        good = False
                        break
                if good:
                    verts.append(k)

        verts = np.array(verts, dtype=float)

        uniq = []
        for v in verts:
            if not any(np.linalg.norm(v - u) < 1e-8 for u in uniq):
                uniq.append(v)
        verts = np.array(uniq, dtype=float)

        center = np.mean(verts, axis=0)
        ang = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        order = np.argsort(ang)
        verts = verts[order]

        return verts

    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
                if x < xinters:
                    inside = not inside
        return inside

    # ------------------------
    # energy grid + contour extraction
    # ------------------------
    def build_energy_grid_first_bz(self):
        verts = self.first_bz_vertices()

        xmin, ymin = np.min(verts, axis=0)
        xmax, ymax = np.max(verts, axis=0)

        kxs = np.linspace(xmin, xmax, self.grid_size)
        kys = np.linspace(ymin, ymax, self.grid_size)
        KX, KY = np.meshgrid(kxs, kys, indexing="xy")

        E = np.full_like(KX, np.nan, dtype=float)
        mask = np.zeros_like(KX, dtype=bool)

        for i in range(KX.shape[0]):
            for j in range(KX.shape[1]):
                k = np.array([KX[i, j], KY[i, j]])
                if self.point_in_polygon(k, verts):
                    mask[i, j] = True
                    E[i, j] = self.band_energy(k[0], k[1])

        return KX, KY, E, mask, verts

    def extract_fs_contours_first_bz(self, KX, KY, E, mask, mu_level):
        import matplotlib.pyplot as plt

        Em = np.ma.array(E, mask=~mask)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contour(KX, KY, Em, levels=[mu_level])
        plt.close(fig)

        contours = []
        if len(cs.allsegs) == 0 or len(cs.allsegs[0]) == 0:
            return contours

        for seg in cs.allsegs[0]:
            if seg.shape[0] >= self.contour_min_points:
                contours.append(np.array(seg, dtype=float))

        return contours

    # ------------------------
    # contour scoring
    # ------------------------
    @staticmethod
    def contour_length(contour: np.ndarray, closed: bool = True) -> float:
        if contour.shape[0] < 2:
            return 0.0
        diffs = np.diff(contour, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        if closed:
            length += np.linalg.norm(contour[0] - contour[-1])
        return float(length)

    def is_contour_closed(self, contour: np.ndarray, tol: float = 5e-2) -> bool:
        if contour.shape[0] < 3:
            return False
        return np.linalg.norm(contour[0] - contour[-1]) < tol

    def contour_centroid(self, contour: np.ndarray) -> np.ndarray:
        return np.mean(contour, axis=0)

    def polygon_contains_point(self, contour: np.ndarray, point: np.ndarray) -> bool:
        return self.point_in_polygon(point, contour)

    def score_contour(self, contour: np.ndarray) -> Tuple:
        closed = self.is_contour_closed(contour, tol=5e-2)
        length = self.contour_length(contour, closed=closed)

        encloses_target = False
        target_dist = 0.0
        if self.contour_target_k is not None:
            centroid = self.contour_centroid(contour)
            target_dist = np.linalg.norm(centroid - self.contour_target_k)
            if closed:
                try:
                    encloses_target = self.polygon_contains_point(contour, self.contour_target_k)
                except Exception:
                    encloses_target = False

        sort_key = (
            1 if (self.prefer_closed_contour and closed) else 0,
            1 if encloses_target else 0,
            length,
            -target_dist,
        )
        return sort_key

    def choose_best_contour_over_shifts(self, KX, KY, E, mask, mu):
        best = None

        for shift in self.auto_level_shifts:
            mu_level = mu + shift
            contours = self.extract_fs_contours_first_bz(KX, KY, E, mask, mu_level)

            if self.verbose:
                print(f"[FSPatcher] shift={shift:.2e}, n_contours={len(contours)}")

            if len(contours) == 0:
                continue

            scored = []
            for c in contours:
                key = self.score_contour(c)
                scored.append((key, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            local_best_key, local_best_contour = scored[0]

            if self.verbose:
                print(f"[FSPatcher]   best key={local_best_key}")

            if best is None or local_best_key > best[0]:
                best = (local_best_key, local_best_contour, mu_level)

        if best is None:
            raise RuntimeError("No valid FS contour found for any tested contour level.")

        return best[1], best[2]

    # ------------------------
    # contour resampling
    # ------------------------
    def close_contour_if_needed(self, contour: np.ndarray) -> np.ndarray:
        if contour.shape[0] < 2:
            return contour
        if np.linalg.norm(contour[0] - contour[-1]) > 1e-10:
            contour = np.vstack([contour, contour[0]])
        return contour

    def resample_contour_by_arclength(self, contour: np.ndarray, Npatch: int):
        contour = self.close_contour_if_needed(contour)

        diffs = np.diff(contour, axis=0)
        seglen = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seglen)])
        total = s[-1]
        if total <= 0:
            raise RuntimeError("Contour length is zero.")

        targets = np.linspace(0.0, total, Npatch, endpoint=False)
        reps = []

        for t in targets:
            idx = np.searchsorted(s, t, side="right") - 1
            idx = min(max(idx, 0), len(seglen) - 1)

            ds = seglen[idx]
            if ds < 1e-14:
                reps.append(contour[idx].copy())
                continue

            alpha = (t - s[idx]) / ds
            k = (1.0 - alpha) * contour[idx] + alpha * contour[idx + 1]
            reps.append(k)

        return reps

    # ------------------------
    # main
    # ------------------------
    def build(self) -> PatchSet:
        mu = self.get_mu()
        filling = self.filling if self.filling is not None else np.nan

        KX, KY, E, mask, verts = self.build_energy_grid_first_bz()
        main_contour, mu_used_for_contour = self.choose_best_contour_over_shifts(KX, KY, E, mask, mu)
        reps = self.resample_contour_by_arclength(main_contour, self.Npatch)

        patches = []
        for pid, k0 in enumerate(reps):
            kf = self.project_to_fs(k0, mu)  # 注意：投回真实 mu，而不是 shifted mu
            uv = self.wrap_red(self.cart_to_red(kf))
            e = self.band_energy(kf[0], kf[1])
            vf = self.fermi_velocity(kf[0], kf[1])
            eig = self.band_eigvec(kf[0], kf[1])
            w = self.orbital_weight(eig)

            patches.append(
                PatchPoint(
                    patch_id=pid,
                    k_cart=kf,
                    k_red=uv,
                    energy=e,
                    vF=vf,
                    vF_norm=float(np.linalg.norm(vf)),
                    eigvec=eig,
                    orbital_weight=w,
                )
            )

        return PatchSet(
            mu=mu,
            mu_used_for_contour=mu_used_for_contour,
            band_index=self.band_index,
            filling=filling,
            patches=patches,
            fs_contour_k=main_contour,
            bz_vertices=verts,
            b1=self.b1,
            b2=self.b2,
        )


def plot_patchset(patchset, ax=None, show_contour=True, show_velocity=False, show_bz=True):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if show_bz and patchset.bz_vertices is not None:
        bz = np.vstack([patchset.bz_vertices, patchset.bz_vertices[0]])
        ax.plot(bz[:, 0], bz[:, 1], color="black", lw=1.2, alpha=0.8, label="1st BZ")

    if show_contour and patchset.fs_contour_k is not None:
        kk = patchset.fs_contour_k
        kk2 = np.vstack([kk, kk[0]])
        ax.plot(kk2[:, 0], kk2[:, 1], lw=1.3, alpha=0.85, label="FS contour")

    pk = patchset.patch_k
    ax.scatter(pk[:, 0], pk[:, 1], s=24, label="patch reps", zorder=3)

    for p in patchset.patches:
        ax.text(p.k_cart[0], p.k_cart[1], str(p.patch_id), fontsize=7)

    if show_velocity:
        v = patchset.patch_vF
        ax.quiver(
            pk[:, 0], pk[:, 1], v[:, 0], v[:, 1],
            angles="xy", scale_units="xy", scale=1.0
        )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.legend()
    return ax