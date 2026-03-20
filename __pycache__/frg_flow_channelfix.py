
from __future__ import annotations

import numpy as np

from frg_flow import *  # noqa: F401,F403
from frg_flow import FRGFlowSolver as _BaseFRGFlowSolver
from channels import ChannelKernel
from frg_kernel import has_patchset


class FRGFlowSolver(_BaseFRGFlowSolver):
    """
    Drop-in diagnostic A/B solver.

    Physics of the flow is unchanged.  Only the diagnosis-stage construction of
    particle-hole longitudinal charge/spin kernels is modified so you can test
    whether the old channel definition was already the source of the FM-vs-PI
    failure.
    """

    def _make_ph_combo_kernel(self, template: ChannelKernel, *, name: str, matrix: np.ndarray, residuals: np.ndarray, tag: str) -> ChannelKernel:
        return ChannelKernel(
            name=name,
            Q=np.asarray(template.Q, dtype=float),
            matrix=np.asarray(matrix, dtype=complex),
            row_patches=template.row_patches.copy(),
            col_patches=template.col_patches.copy(),
            row_partner_patches=template.row_partner_patches.copy(),
            col_partner_patches=template.col_partner_patches.copy(),
            row_spins=(tag, tag),
            col_spins=(tag, tag),
            residuals=np.asarray(residuals, dtype=float),
        )

    def build_diagnosis_kernel_dict(self, Q):
        Q = np.asarray(Q, dtype=float)
        out = {}

        if has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn"):
            K_ud_ud = self._vertex_pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("up", "dn"))
            K_ud_du = self._vertex_pp_kernel(Q, incoming_spins=("up", "dn"), outgoing_spins=("dn", "up"))
            K_du_ud = self._vertex_pp_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("up", "dn"))
            K_du_du = self._vertex_pp_kernel(Q, incoming_spins=("dn", "up"), outgoing_spins=("dn", "up"))
            out["pp_ud_to_ud"] = K_ud_ud
            out["pp_ud_to_du"] = K_ud_du
            out["pp_du_to_ud"] = K_du_ud
            out["pp_du_to_du"] = K_du_du

            template = K_ud_ud
            out["pp_singlet_sz0"] = ChannelKernel(
                name="pp_singlet_sz0",
                Q=Q,
                matrix=0.5 * (K_ud_ud.matrix - K_ud_du.matrix - K_du_ud.matrix + K_du_du.matrix),
                row_patches=template.row_patches.copy(),
                col_patches=template.col_patches.copy(),
                row_partner_patches=template.row_partner_patches.copy(),
                col_partner_patches=template.col_partner_patches.copy(),
                row_spins=("S", "S"),
                col_spins=("S", "S"),
                residuals=template.residuals.copy(),
            )
            out["pp_triplet_sz0"] = ChannelKernel(
                name="pp_triplet_sz0",
                Q=Q,
                matrix=0.5 * (K_ud_ud.matrix + K_ud_du.matrix + K_du_ud.matrix + K_du_du.matrix),
                row_patches=template.row_patches.copy(),
                col_patches=template.col_patches.copy(),
                row_partner_patches=template.row_partner_patches.copy(),
                col_partner_patches=template.col_partner_patches.copy(),
                row_spins=("T", "T"),
                col_spins=("T", "T"),
                residuals=template.residuals.copy(),
            )

        if has_patchset(self.patchsets, "up"):
            out["pp_triplet_uu"] = self._vertex_pp_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
            out["phd_uu"] = self._vertex_phd_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
            if self.track_crossed_channel:
                out["phc_uu"] = self._vertex_phc_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))

        if has_patchset(self.patchsets, "dn"):
            out["pp_triplet_dd"] = self._vertex_pp_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
            out["phd_dd"] = self._vertex_phd_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
            if self.track_crossed_channel:
                out["phc_dd"] = self._vertex_phc_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))

        if has_patchset(self.patchsets, "up") and has_patchset(self.patchsets, "dn"):
            K_uu_uu = self._vertex_phd_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
            K_uu_dd = self._vertex_phd_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("dn", "dn"))
            K_dd_uu = self._vertex_phd_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("up", "up"))
            K_dd_dd = self._vertex_phd_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))

            out["phd_uu_to_uu"] = K_uu_uu
            out["phd_uu_to_dd"] = K_uu_dd
            out["phd_dd_to_uu"] = K_dd_uu
            out["phd_dd_to_dd"] = K_dd_dd

            residuals_all = np.maximum.reduce([
                K_uu_uu.residuals,
                K_uu_dd.residuals,
                K_dd_uu.residuals,
                K_dd_dd.residuals,
            ])
            residuals_diag = np.maximum(K_uu_uu.residuals, K_dd_dd.residuals)

            corrected_charge = 0.5 * (K_uu_uu.matrix + K_uu_dd.matrix + K_dd_uu.matrix + K_dd_dd.matrix)
            corrected_spin = 0.5 * (K_uu_uu.matrix - K_uu_dd.matrix - K_dd_uu.matrix + K_dd_dd.matrix)
            legacy_charge = 0.5 * (K_uu_uu.matrix + K_dd_dd.matrix)
            legacy_spin = 0.5 * (K_uu_uu.matrix - K_dd_dd.matrix)

            out["ph_charge_longitudinal"] = self._make_ph_combo_kernel(
                K_uu_uu, name="ph_charge_longitudinal", matrix=corrected_charge, residuals=residuals_all, tag="rho"
            )
            out["ph_spin_longitudinal"] = self._make_ph_combo_kernel(
                K_uu_uu, name="ph_spin_longitudinal", matrix=corrected_spin, residuals=residuals_all, tag="sz"
            )
            out["ph_charge_longitudinal_legacy"] = self._make_ph_combo_kernel(
                K_uu_uu, name="ph_charge_longitudinal_legacy", matrix=legacy_charge, residuals=residuals_diag, tag="rho"
            )
            out["ph_spin_longitudinal_legacy"] = self._make_ph_combo_kernel(
                K_uu_uu, name="ph_spin_longitudinal_legacy", matrix=legacy_spin, residuals=residuals_diag, tag="sz"
            )
        return out
