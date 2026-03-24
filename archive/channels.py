import numpy as np
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, Union

SpinLike = Union[str, int]
PatchSetMap = Mapping[SpinLike, object]


@dataclass
class ChannelKernel:
    """
    Discrete patch-space kernel for one FRG channel at fixed transfer momentum Q.
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
        else:
            raise ValueError("sort_by must be 'abs' or 'real'.")
        return vals[order], vecs[:, order]


class ChannelDecomposer:
    r"""
    Build patch-space kernels for pp / direct-ph / crossed-ph channels from the
    antisymmetrized patch vertex provided by BareExtendedHubbard.

    Important conventions
    ---------------------
    1) particle-particle (pp)
       incoming  : (k_in,  Q-k_in)
       outgoing  : (k_out, Q-k_out)

       M_pp[out, in] = Gamma(k_in, Q-k_in -> k_out, Q-k_out)

    2) particle-hole direct (ph_direct)
       incoming bilinear  : k_in  -> k_in + Q
       outgoing bilinear  : k_out -> k_out + Q

       M_phd[out, in] = Gamma(k_in, k_out+Q -> k_in+Q, k_out)

    3) particle-hole crossed (ph_crossed)
       incoming bilinear  : k_in  -> k_in - Q
       outgoing bilinear  : k_out -> k_out + Q

       M_phc[out, in] = Gamma(k_in, k_out -> k_out+Q, k_in-Q)

    Notes
    -----
    - This class only does channel bookkeeping.
    - It assumes the input vertex is already the antisymmetrized fermionic
      vertex from interaction.patch_vertex(..., antisym=True).
    - Shifted momenta like Q-k, k+Q, k-Q are matched to the nearest patch
      modulo reciprocal lattice vectors.
    """

    def __init__(self, interaction, patchsets: PatchSetMap):
        self.interaction = interaction
        self.patchsets = patchsets

        # Early validation: avoid deep stack-trace failures later.
        expected = int(self.interaction.Norb)
        for spin, ps in patchsets.items():
            if ps is None:
                raise ValueError(f"patchsets['{spin}'] is None.")
            if not hasattr(ps, "patches"):
                raise ValueError(f"patchsets['{spin}'] does not look like a PatchSet.")
            if len(ps.patches) == 0:
                # empty patchsets are allowed to exist, but channels using them must be skipped
                continue
            u0 = np.asarray(ps.patches[0].eigvec)
            if u0.shape != (expected,):
                raise ValueError(
                    f"patchsets['{spin}'] has eigvec shape {u0.shape}, "
                    f"but interaction expects length-{expected} kagome eigenvectors."
                )

    def normalize_spin(self, spin: SpinLike) -> str:
        return self.interaction.normalize_spin(spin)

    def _patchset_for_spin(self, spin: SpinLike):
        ps = self.interaction._patchset_for_spin(self.patchsets, spin)
        return ps

    def _require_nonempty_patchset(self, spin: SpinLike):
        ps = self._patchset_for_spin(spin)
        if ps.Npatch == 0:
            raise ValueError(
                f"patchset for spin='{spin}' is empty. "
                "This usually means that spin sector does not cross the Fermi surface, "
                "so channels involving it cannot be constructed."
            )
        return ps

    @staticmethod
    def _minimum_image_displacement(k_target, k_ref, b1, b2):
        k_target = np.asarray(k_target, dtype=float)
        k_ref = np.asarray(k_ref, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        b2 = np.asarray(b2, dtype=float)

        best = None
        best_norm = np.inf
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                disp = k_target - (k_ref + n1 * b1 + n2 * b2)
                norm = float(np.linalg.norm(disp))
                if norm < best_norm:
                    best_norm = norm
                    best = disp
        return best

    def find_shifted_patch_index(self, spin: SpinLike, k_target: np.ndarray):
        """
        Find nearest patch to k_target modulo reciprocal lattice vectors.
        """
        PS = self._require_nonempty_patchset(spin)
        ks = np.array([p.k_cart for p in PS.patches], dtype=float)
        dists = []
        for k_ref in ks:
            disp = self._minimum_image_displacement(k_target, k_ref, PS.b1, PS.b2)
            dists.append(np.linalg.norm(disp))
        dists = np.asarray(dists, dtype=float)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def shifted_patch_map(self, spin: SpinLike, Q: Sequence[float], *, mode: str):
        """
        Map each patch p to the nearest patch representing a shifted momentum.

        mode:
            'Q_minus_k' : target = Q - k
            'k_plus_Q'  : target = k + Q
            'k_minus_Q' : target = k - Q
        """
        Q = np.asarray(Q, dtype=float)
        PS = self._require_nonempty_patchset(spin)
        idxs = np.zeros(PS.Npatch, dtype=int)
        residuals = np.zeros(PS.Npatch, dtype=float)

        for p, patch in enumerate(PS.patches):
            k = np.asarray(patch.k_cart, dtype=float)
            if mode == "Q_minus_k":
                target = Q - k
            elif mode == "k_plus_Q":
                target = k + Q
            elif mode == "k_minus_Q":
                target = k - Q
            else:
                raise ValueError("mode must be one of {'Q_minus_k', 'k_plus_Q', 'k_minus_Q'}.")
            idx, dist = self.find_shifted_patch_index(spin, target)
            idxs[p] = idx
            residuals[p] = dist

        return idxs, residuals

    def gamma(self, p1, s1, p2, s2, p3, s3, p4, s4):
        return self.interaction.patch_vertex(
            self.patchsets,
            p1, s1,
            p2, s2,
            p3, s3,
            p4, s4,
            antisym=True,
            check_momentum=False,
        )

    def gamma_tensor(self, s1, s2, s3, s4):
        return self.interaction.patch_tensor(
            self.patchsets, s1, s2, s3, s4, antisym=True, enforce_momentum=False
        )

    def pp_kernel(self, Q, *, incoming_spins=("up", "dn"), outgoing_spins=None):
        Q = np.asarray(Q, dtype=float)
        s1, s2 = map(self.normalize_spin, incoming_spins)
        if outgoing_spins is None:
            s3, s4 = s1, s2
        else:
            s3, s4 = map(self.normalize_spin, outgoing_spins)

        PS_in = self._require_nonempty_patchset(s1)
        PS_out = self._require_nonempty_patchset(s3)
        self._require_nonempty_patchset(s2)
        self._require_nonempty_patchset(s4)

        if PS_in.Npatch != PS_out.Npatch:
            raise ValueError("For pp kernel, first-leg incoming/outgoing patch counts must match.")

        partner_in, resid_in = self.shifted_patch_map(s2, Q, mode="Q_minus_k")
        partner_out, resid_out = self.shifted_patch_map(s4, Q, mode="Q_minus_k")

        N = PS_in.Npatch
        M = np.zeros((N, N), dtype=complex)
        residuals = np.zeros((N, N), dtype=float)
        rows = np.arange(N, dtype=int)
        cols = np.arange(N, dtype=int)

        for pout in range(N):
            p4 = int(partner_out[pout])
            for pin in range(N):
                p2 = int(partner_in[pin])
                M[pout, pin] = self.gamma(pin, s1, p2, s2, pout, s3, p4, s4)
                residuals[pout, pin] = max(resid_in[pin], resid_out[pout])

        return ChannelKernel(
            name="pp",
            Q=Q,
            matrix=M,
            row_patches=rows,
            col_patches=cols,
            row_partner_patches=np.asarray(partner_out, dtype=int),
            col_partner_patches=np.asarray(partner_in, dtype=int),
            row_spins=(s3, s4),
            col_spins=(s1, s2),
            residuals=residuals,
        )

    def ph_direct_kernel(self, Q, *, incoming_spins=("up", "up"), outgoing_spins=None):
        Q = np.asarray(Q, dtype=float)
        s1, s2 = map(self.normalize_spin, incoming_spins)
        if outgoing_spins is None:
            s3, s4 = s1, s2
        else:
            s3, s4 = map(self.normalize_spin, outgoing_spins)

        PS1 = self._require_nonempty_patchset(s1)
        PS4 = self._require_nonempty_patchset(s4)
        self._require_nonempty_patchset(s2)
        self._require_nonempty_patchset(s3)

        if PS1.Npatch != PS4.Npatch:
            raise ValueError("For ph-direct kernel, row/col patch counts must match.")

        kplus_in, resid_in = self.shifted_patch_map(s3, Q, mode="k_plus_Q")
        kplus_out, resid_out = self.shifted_patch_map(s2, Q, mode="k_plus_Q")

        N = PS1.Npatch
        M = np.zeros((N, N), dtype=complex)
        residuals = np.zeros((N, N), dtype=float)
        rows = np.arange(N, dtype=int)
        cols = np.arange(N, dtype=int)

        for pout in range(N):
            p2 = int(kplus_out[pout])
            for pin in range(N):
                p3 = int(kplus_in[pin])
                M[pout, pin] = self.gamma(pin, s1, p2, s2, p3, s3, pout, s4)
                residuals[pout, pin] = max(resid_in[pin], resid_out[pout])

        return ChannelKernel(
            name="ph_direct",
            Q=Q,
            matrix=M,
            row_patches=rows,
            col_patches=cols,
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kplus_in, dtype=int),
            row_spins=(s4, s2),
            col_spins=(s1, s3),
            residuals=residuals,
        )

    def ph_crossed_kernel(self, Q, *, incoming_spins=("up", "up"), outgoing_spins=None):
        Q = np.asarray(Q, dtype=float)
        s1, s2 = map(self.normalize_spin, incoming_spins)
        if outgoing_spins is None:
            s3, s4 = s2, s1
        else:
            s3, s4 = map(self.normalize_spin, outgoing_spins)

        PS1 = self._require_nonempty_patchset(s1)
        PS2 = self._require_nonempty_patchset(s2)
        self._require_nonempty_patchset(s3)
        self._require_nonempty_patchset(s4)

        if PS1.Npatch != PS2.Npatch:
            raise ValueError("For ph-crossed kernel, first and second spin patch counts must match.")

        kplus_out, resid_out = self.shifted_patch_map(s3, Q, mode="k_plus_Q")
        kminus_in, resid_in = self.shifted_patch_map(s4, Q, mode="k_minus_Q")

        N = PS1.Npatch
        M = np.zeros((N, N), dtype=complex)
        residuals = np.zeros((N, N), dtype=float)
        rows = np.arange(N, dtype=int)
        cols = np.arange(N, dtype=int)

        for pout in range(N):
            p3 = int(kplus_out[pout])
            for pin in range(N):
                p4 = int(kminus_in[pin])
                M[pout, pin] = self.gamma(pin, s1, pout, s2, p3, s3, p4, s4)
                residuals[pout, pin] = max(resid_in[pin], resid_out[pout])

        return ChannelKernel(
            name="ph_crossed",
            Q=Q,
            matrix=M,
            row_patches=rows,
            col_patches=cols,
            row_partner_patches=np.asarray(kplus_out, dtype=int),
            col_partner_patches=np.asarray(kminus_in, dtype=int),
            row_spins=(s2, s3),
            col_spins=(s1, s4),
            residuals=residuals,
        )

    def pp_spin_blocks(self, Q) -> Dict[str, ChannelKernel]:
        out = {}

        # mixed-spin blocks only if both sectors exist
        try:
            self._require_nonempty_patchset("up")
            self._require_nonempty_patchset("dn")
            out.update({
                "ud_to_ud": self.pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("up", "dn")),
                "ud_to_du": self.pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("dn", "up")),
                "du_to_ud": self.pp_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("up", "dn")),
                "du_to_du": self.pp_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("dn", "up")),
            })
        except ValueError:
            pass

        # same-spin blocks if available
        try:
            out["uu_to_uu"] = self.pp_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
        except ValueError:
            pass

        try:
            out["dd_to_dd"] = self.pp_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
        except ValueError:
            pass

        if len(out) == 0:
            raise ValueError("No valid pp spin blocks can be constructed from the current patchsets.")

        return out

    def pp_singlet_triplet_sz0(self, Q) -> Dict[str, ChannelKernel]:
        """
        Build Sz=0 singlet / triplet pp kernels by algebraic spin combination.

        Requires both up and down patchsets to be nonempty.
        """
        blocks = self.pp_spin_blocks(Q)
        required = ["ud_to_ud", "ud_to_du", "du_to_ud", "du_to_du"]
        missing = [key for key in required if key not in blocks]
        if missing:
            raise ValueError(
                "pp_singlet_triplet_sz0 requires both spin sectors to cross the Fermi surface. "
                f"Missing blocks: {missing}"
            )

        K_ud_ud = blocks["ud_to_ud"].matrix
        K_ud_du = blocks["ud_to_du"].matrix
        K_du_ud = blocks["du_to_ud"].matrix
        K_du_du = blocks["du_to_du"].matrix

        singlet = 0.5 * (K_ud_ud - K_ud_du - K_du_ud + K_du_du)
        triplet = 0.5 * (K_ud_ud + K_ud_du + K_du_ud + K_du_du)

        template = blocks["ud_to_ud"]
        singlet_kernel = ChannelKernel(
            name="pp_singlet_sz0",
            Q=np.asarray(Q, dtype=float),
            matrix=singlet,
            row_patches=template.row_patches.copy(),
            col_patches=template.col_patches.copy(),
            row_partner_patches=template.row_partner_patches.copy(),
            col_partner_patches=template.col_partner_patches.copy(),
            row_spins=("S", "S"),
            col_spins=("S", "S"),
            residuals=template.residuals.copy(),
        )
        triplet_kernel = ChannelKernel(
            name="pp_triplet_sz0",
            Q=np.asarray(Q, dtype=float),
            matrix=triplet,
            row_patches=template.row_patches.copy(),
            col_patches=template.col_patches.copy(),
            row_partner_patches=template.row_partner_patches.copy(),
            col_partner_patches=template.col_partner_patches.copy(),
            row_spins=("T", "T"),
            col_spins=("T", "T"),
            residuals=template.residuals.copy(),
        )

        out = {
            "singlet_sz0": singlet_kernel,
            "triplet_sz0": triplet_kernel,
        }
        if "uu_to_uu" in blocks:
            out["triplet_uu"] = blocks["uu_to_uu"]
        if "dd_to_dd" in blocks:
            out["triplet_dd"] = blocks["dd_to_dd"]
        return out

    def ph_longitudinal_blocks(self, Q) -> Dict[str, ChannelKernel]:
        """
        Raw longitudinal direct-ph blocks in the physical {up, dn} basis.

        These are the ingredients needed to form charge / spin bilinears in the
        {rho, sz} basis.  In particular, the mixed blocks

            uu -> dd,   dd -> uu

        must be retained; using only the diagonal same-spin blocks can
        artificially suppress the longitudinal spin channel in SU(2)-symmetric
        benchmarks.
        """
        Q = np.asarray(Q, dtype=float)
        K_uu_uu = self.ph_direct_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
        K_uu_dd = self.ph_direct_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("up", "dn"))
        K_dd_uu = self.ph_direct_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("dn", "up"))
        K_dd_dd = self.ph_direct_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
        return {
            "uu_to_uu": K_uu_uu,
            "uu_to_dd": K_uu_dd,
            "dd_to_uu": K_dd_uu,
            "dd_to_dd": K_dd_dd,
        }

    def ph_charge_spin_longitudinal(self, Q) -> Dict[str, ChannelKernel]:
        """
        Longitudinal charge / spin kernels in the {rho, sz} bilinear basis.

        Correct combinations:

            K_rho = 0.5 * (K_uu->uu + K_uu->dd + K_dd->uu + K_dd->dd)
            K_sz  = 0.5 * (K_uu->uu - K_uu->dd - K_dd->uu + K_dd->dd)

        We also return the raw blocks for debugging.
        """
        blocks = self.ph_longitudinal_blocks(Q)
        K_uu_uu = blocks["uu_to_uu"]
        K_uu_dd = blocks["uu_to_dd"]
        K_dd_uu = blocks["dd_to_uu"]
        K_dd_dd = blocks["dd_to_dd"]

        charge = 0.5 * (
            K_uu_uu.matrix + K_uu_dd.matrix + K_dd_uu.matrix + K_dd_dd.matrix
        )
        spin = 0.5 * (
            K_uu_uu.matrix - K_uu_dd.matrix - K_dd_uu.matrix + K_dd_dd.matrix
        )
        residuals = np.maximum.reduce([
            K_uu_uu.residuals,
            K_uu_dd.residuals,
            K_dd_uu.residuals,
            K_dd_dd.residuals,
        ])

        def _make(name: str, matrix: np.ndarray, tag: str) -> ChannelKernel:
            return ChannelKernel(
                name=name,
                Q=np.asarray(Q, dtype=float),
                matrix=np.asarray(matrix, dtype=complex),
                row_patches=K_uu_uu.row_patches.copy(),
                col_patches=K_uu_uu.col_patches.copy(),
                row_partner_patches=K_uu_uu.row_partner_patches.copy(),
                col_partner_patches=K_uu_uu.col_partner_patches.copy(),
                row_spins=(tag, tag),
                col_spins=(tag, tag),
                residuals=np.asarray(residuals, dtype=float),
            )

        return {
            "charge": _make("ph_charge_longitudinal", charge, "rho"),
            "spin": _make("ph_spin_longitudinal", spin, "sz"),
            **blocks,
        }

    def build_all_basic_channels(self, Q) -> Dict[str, ChannelKernel]:
        out = {}
        try:
            out["pp_ud"] = self.pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("up", "dn"))
        except ValueError:
            pass
        try:
            out["phd_uu"] = self.ph_direct_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
        except ValueError:
            pass
        try:
            out["phd_dd"] = self.ph_direct_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
        except ValueError:
            pass
        try:
            out["phc_uu"] = self.ph_crossed_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
        except ValueError:
            pass
        try:
            out["phc_dd"] = self.ph_crossed_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
        except ValueError:
            pass
        if len(out) == 0:
            raise ValueError("No valid basic channels can be constructed from the current patchsets.")
        return out

    def summarize(self, kernel: ChannelKernel) -> Dict[str, float]:
        vals, _ = kernel.eig(sort_by="abs")
        return {
            "Npatch": float(kernel.Npatch),
            "max_patch_match_residual": float(np.max(kernel.residuals)),
            "mean_patch_match_residual": float(np.mean(kernel.residuals)),
            "hermitian_residual": float(kernel.hermitian_residual()),
            "largest_abs_eigenvalue": float(np.abs(vals[0])) if len(vals) else 0.0,
            "largest_real_part": float(np.max(np.real(vals))) if len(vals) else 0.0,
        }