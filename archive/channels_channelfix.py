
import numpy as np
from typing import Dict

from channels import ChannelKernel, ChannelDecomposer as _BaseChannelDecomposer


class ChannelDecomposer(_BaseChannelDecomposer):
    """
    Diagnostic A/B version of ChannelDecomposer.

    Main change
    -----------
    The old code defined ph longitudinal charge/spin using only the same-spin
    diagonal blocks

        charge_legacy = 0.5 * (K_uu->uu + K_dd->dd)
        spin_legacy   = 0.5 * (K_uu->uu - K_dd->dd)

    This can miss the mixed longitudinal blocks

        K_uu->dd,  K_dd->uu

    which are needed once one asks for the kernel in the {rho, sz} basis of
    bilinears.  The corrected combinations are

        K_rho = 0.5 * (K_uu->uu + K_uu->dd + K_dd->uu + K_dd->dd)
        K_sz  = 0.5 * (K_uu->uu - K_uu->dd - K_dd->uu + K_dd->dd)

    This file keeps both the corrected and legacy combinations so you can run a
    clean A/B test.
    """

    def ph_longitudinal_blocks(self, Q) -> Dict[str, ChannelKernel]:
        Q = np.asarray(Q, dtype=float)
        K_uu_uu = self.ph_direct_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("up", "up"))
        K_uu_dd = self.ph_direct_kernel(Q, incoming_spins=("up", "up"), outgoing_spins=("dn", "dn"))
        K_dd_uu = self.ph_direct_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("up", "up"))
        K_dd_dd = self.ph_direct_kernel(Q, incoming_spins=("dn", "dn"), outgoing_spins=("dn", "dn"))
        return {
            "uu_to_uu": K_uu_uu,
            "uu_to_dd": K_uu_dd,
            "dd_to_uu": K_dd_uu,
            "dd_to_dd": K_dd_dd,
        }

    def ph_charge_spin_longitudinal(self, Q):
        """
        Build both the legacy and corrected longitudinal ph combinations.

        Returns keys:
            charge, spin              : corrected rho/sz combinations
            charge_legacy, spin_legacy: old diagonal-only combinations
            uu_to_uu, uu_to_dd, dd_to_uu, dd_to_dd: raw blocks
        """
        blocks = self.ph_longitudinal_blocks(Q)
        K_uu_uu = blocks["uu_to_uu"]
        K_uu_dd = blocks["uu_to_dd"]
        K_dd_uu = blocks["dd_to_uu"]
        K_dd_dd = blocks["dd_to_dd"]

        corrected_charge = 0.5 * (
            K_uu_uu.matrix + K_uu_dd.matrix + K_dd_uu.matrix + K_dd_dd.matrix
        )
        corrected_spin = 0.5 * (
            K_uu_uu.matrix - K_uu_dd.matrix - K_dd_uu.matrix + K_dd_dd.matrix
        )

        legacy_charge = 0.5 * (K_uu_uu.matrix + K_dd_dd.matrix)
        legacy_spin = 0.5 * (K_uu_uu.matrix - K_dd_dd.matrix)

        residuals_all = np.maximum.reduce([
            K_uu_uu.residuals,
            K_uu_dd.residuals,
            K_dd_uu.residuals,
            K_dd_dd.residuals,
        ])
        residuals_diag = np.maximum(K_uu_uu.residuals, K_dd_dd.residuals)

        def _make(name, matrix, residuals, tag):
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

        out = {
            "charge": _make("ph_charge_longitudinal", corrected_charge, residuals_all, "rho"),
            "spin": _make("ph_spin_longitudinal", corrected_spin, residuals_all, "sz"),
            "charge_legacy": _make("ph_charge_longitudinal_legacy", legacy_charge, residuals_diag, "rho"),
            "spin_legacy": _make("ph_spin_longitudinal_legacy", legacy_spin, residuals_diag, "sz"),
            **blocks,
        }
        return out
