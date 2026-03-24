import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from scipy.linalg import block_diag  
from typing import Tuple, Protocol
from abc import ABC, abstractmethod
from functools import partial


def expect_ED(OP,eig_vec):
    OP = np.array(OP)
    ket = np.array(eig_vec).reshape(-1,1)
    bra = np.array(ket.conjugate().reshape(1,-1))
    return (bra@(OP@ket)).item()

class Noninteracting_Model(ABC):
    """
    Father class, with virtual function def Hk
    parameters:
        t: real hopping 
        h: imaginary hopping
    """
    def __init__(self):
        self.Emin,self.Emax,self._H_rt,self.a = None,None,None,2

    @property
    @abstractmethod
    def b1(self):
        pass

    @property
    @abstractmethod
    def b2(self):
        pass
        
    @abstractmethod
    def Hk(self, kx, ky, *args, **kwargs) -> np.ndarray:
        pass

    def eigenstate(self,kx,ky):
        eig_vals, eig_vecs = np.linalg.eigh(self.Hk(kx,ky))
        return eig_vals, eig_vecs
        
    @property
    def E_min_max(self):
        #return the minimal and maximal energy
        N=400
        if self.Emin is None:
            b1,b2 = self.b1,self.b2
            low_E,high_E = 1e10,-1e10
            for i in range(N):
                for j in range(N):
                    # alpha, beta = (i + 0.5) / N, (j + 0.5) / N
                    alpha, beta = (i) / N, (j) / N
                    k = alpha * b1 + beta * b2
                    E_min = self.eigenstate(k[0],k[1])[0].min()
                    E_max = self.eigenstate(k[0],k[1])[0].max()
                    if low_E > E_min:
                        low_E = E_min
                    if high_E < E_max:
                        high_E = E_max
            self.Emin, self.Emax = low_E, high_E
        return [self.Emin, self.Emax]
        
    def filling_from_EF(self, EF: float, N: int = 200, *, tie_rule: str = "le") -> float:
        b1, b2 = self.b1, self.b2
        M = self.Hk(0.0, 0.0).shape[0]
        count = 0
        for i in range(N):
            a = (i + 0.5) / N
            for j in range(N):
                b = (j + 0.5) / N
                kx, ky = (a * b1 + b * b2)
                evals = self.eigenstate(kx,ky)[0]
                count += np.count_nonzero(evals < EF) if tie_rule == "lt" else np.count_nonzero(evals <= EF)
        nelec_per_cell = count / (N * N) 
        return float(nelec_per_cell)/M

    def EF_from_filling(self, target_filling: float, N: int = 100, *,
                        tie_rule: str = "le", tol: float = 1e-5, maxiter: int = 80, seed: int | None = None) -> float:
        
        def bisect_monotone(f, target: float, lo: float, hi: float, *, tol: float = 1e-5, maxiter: int = 80) -> float:
            f_lo = f(lo)
            f_hi = f(hi)
            if not (f_lo <= target <= f_hi):
                raise ValueError("Bisection bracket invalid: f(lo) <= target <= f(hi) violated.")
        
            for _ in range(maxiter):
                mid = 0.5 * (lo + hi)
                f_mid = f(mid)
                if abs(f_mid - target) <= tol:
                    return mid
                if f_mid < target:
                    lo, f_lo = mid, f_mid
                else:
                    hi, f_hi = mid, f_mid
            return 0.5 * (lo + hi)
        if not (0.0 <= target_filling <= 1):
            raise ValueError(f"target_filling must be in [0, 1].")
    
        b1, b2 = self.b1, self.b2
        Emin, Emax = self.E_min_max
        lo, hi = float(Emin) - 0.1, float(Emax) + 0.1
    
        return bisect_monotone(partial(self.filling_from_EF, N=N, tie_rule=tie_rule), target_filling, lo, hi, tol=tol, maxiter=maxiter)
    
    def berry_curvature(self, kx, ky, dk=1e-4):
        # Use 4th-order central difference for derivatives
        def dH_dkx(kx, ky):
            return (-self.Hk(kx + 2*dk, ky)
                    + 8*self.Hk(kx + dk, ky)
                    - 8*self.Hk(kx - dk, ky)
                    + self.Hk(kx - 2*dk, ky)) / (12*dk)
    
        def dH_dky(kx, ky):
            return (-self.Hk(kx, ky + 2*dk)
                    + 8*self.Hk(kx, ky + dk)
                    - 8*self.Hk(kx, ky - dk)
                    + self.Hk(kx, ky - 2*dk)) / (12*dk)
    
        H_k = self.Hk(kx, ky)
        eigenvalues, eigenvectors = np.linalg.eigh(H_k)
        curvature = np.zeros(len(eigenvalues))
        dxH = dH_dkx(kx, ky)
        dyH = dH_dky(kx, ky)
    
        for n in range(len(eigenvalues)):
            sum_term = 0
            for m in range(len(eigenvalues)):
                if n != m:
                    num = (np.vdot(eigenvectors[:, n], dxH @ eigenvectors[:, m]) *
                           np.vdot(eigenvectors[:, m], dyH @ eigenvectors[:, n]))
                    den = (eigenvalues[n] - eigenvalues[m])**2
                    if den == 0 and num == 0:
                        continue
                    sum_term += num / den
            curvature[n] = -2 * np.imag(sum_term)
        return curvature
        
    def Chern_number(self,N=300):
        b1,b2 = self.b1,self.b2       # a=2       
        total = 0.0
        area = np.abs(np.cross(b1, b2))  
        dk_area = area / (N * N)
        for i in range(N):
            for j in range(N):
                # alpha, beta = (i + 0.5) / N, (j + 0.5) / N
                alpha, beta = (i) / N, (j) / N
                k = alpha * b1 + beta * b2
                total += self.berry_curvature(k[0], k[1]) * dk_area / (2*np.pi)
        return total     
        
    # def Chern_number_FHS(self, N=40):
    #     """
    #     Per-band Chern numbers via Fukui–Hatsugai–Suzuki (Abelian) method.
    #     """
    #     b1, b2 = self.b1, self.b2
    
    #     grid = np.arange(N, dtype=float) / N
    #     kgrid = np.array([[i*b1 + j*b2 for j in grid] for i in grid], dtype=float) 
    
    #     M = self.Hk(0.0, 0.0).shape[0]
    #     V = np.empty((N, N, M, M), dtype=np.complex128)
    #     for i in range(N):
    #         for j in range(N):
    #             kx, ky = kgrid[i, j]
    #             _, vecs = np.linalg.eigh(self.Hk(kx, ky))
    #             vecs = vecs / np.linalg.norm(vecs, axis=0, keepdims=True)
    #             V[i, j] = vecs
    
    #     ip = lambda x: (x + 1) % N
    #     eps = 1e-16
    #     chern = np.zeros(M, dtype=float)
    #     for i in range(N):
    #         for j in range(N):
    #             V00 = V[i,     j    ]
    #             V10 = V[ip(i), j    ]
    #             V11 = V[ip(i), ip(j)]
    #             V01 = V[i,     ip(j)]
    #             Lx_bot = (V00.conj().T @ V10).diagonal().copy()   # U_x(k)
    #             Ly_rgt = (V10.conj().T @ V11).diagonal().copy()   # U_y(k+e_x)
    #             Lx_top_fwd = (V01.conj().T @ V11).diagonal().copy()  # U_x(k+e_y)
    #             Ly_lft_fwd = (V00.conj().T @ V01).diagonal().copy()  # U_y(k)
    #             Lx_bot = Lx_bot / (np.abs(Lx_bot) + eps)
    #             Ly_rgt = Ly_rgt / (np.abs(Ly_rgt) + eps)
    #             Lx_top_fwd = Lx_top_fwd / (np.abs(Lx_top_fwd) + eps)
    #             Ly_lft_fwd = Ly_lft_fwd / (np.abs(Ly_lft_fwd) + eps)
    #             W = Lx_bot * Ly_rgt * np.conj(Lx_top_fwd) * np.conj(Ly_lft_fwd)
    #             chern += p.angle(W)
    #     return chern / (2*np.pi)

        
    def Morb_integrand(self, kx, ky, mu=0, dk=1e-4) -> Tuple[np.ndarray, np.ndarray] :
        """
        Parameters:
            kx, ky: Momentum values
            mu: Fermi energy
            dk: Small value for numerical derivative
        Returns:
            integrand: Orbital magnetization integrand for each band: [LC, LC_reg, IC, IC_reg, Nontopo, Topo, Total]
        """
        def dH_dkx(kx, ky):
            return (-self.Hk(kx + 2*dk, ky)
                    + 8*self.Hk(kx + dk, ky)
                    - 8*self.Hk(kx - dk, ky)
                    + self.Hk(kx - 2*dk, ky)) / (12*dk)
    
        def dH_dky(kx, ky):
            return (-self.Hk(kx, ky + 2*dk)
                    + 8*self.Hk(kx, ky + dk)
                    - 8*self.Hk(kx, ky - dk)
                    + self.Hk(kx, ky - 2*dk)) / (12*dk)

        H_k = self.Hk(kx, ky)
        dH_kx = dH_dkx(kx, ky)
        dH_ky = dH_dky(kx, ky)
        eigvals, eigvecs = np.linalg.eigh(H_k)
        occupied_mask = eigvals < mu
        occupied_vecs = eigvecs[:, occupied_mask]
        P = occupied_vecs @ occupied_vecs.conj().T  #occupied space projector
        Q = np.eye(P.shape[0], dtype=complex) - P   #unoccupied space projector
        N = len(eigvals)
        nontopo_LC,nontopo_IC,nontopo_LC_reg,nontopo_IC_reg,topo = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    
        for n in range(N):
            En = eigvals[n]
            shift = self.E_min_max[0]*np.eye(N) #energy shift
            Q_nontopo_LC, Q_nontopo_LC_reg = H_k-shift,  Q@(H_k-shift)@Q #non topological term LC
            Q_nontopo_IC, Q_nontopo_IC_reg = En*np.eye(N)-shift, Q@(En*np.eye(N)-shift)@Q #non topological term IC
            Q_topo = -2*mu*np.eye(N) + 2*shift #topological term
    
            for m in range(N):
                if m == n: continue
                for l in range(N):
                    if l == n: continue
                    vnm = np.vdot(eigvecs[:, n], dH_kx @ eigvecs[:, m])
                    vlm = np.vdot(eigvecs[:, l], dH_ky @ eigvecs[:, n])
                    denom = (En - eigvals[m]) * (En - eigvals[l])
                    
                    Qml = np.vdot(eigvecs[:, m], Q_nontopo_LC @ eigvecs[:, l])
                    nontopo_LC[n] += np.imag(vnm * Qml * vlm / denom)

                    Qml = np.vdot(eigvecs[:, m], Q_nontopo_LC_reg @ eigvecs[:, l])
                    nontopo_LC_reg[n] += np.imag(vnm * Qml * vlm / denom)
                    
                    Qml = np.vdot(eigvecs[:, m], Q_nontopo_IC @ eigvecs[:, l])
                    nontopo_IC[n] += np.imag(vnm * Qml * vlm / denom)

                    Qml = np.vdot(eigvecs[:, m], Q_nontopo_IC_reg @ eigvecs[:, l])
                    nontopo_IC_reg[n] += np.imag(vnm * Qml * vlm / denom)
                    
                    Qml = np.vdot(eigvecs[:, m], Q_topo @ eigvecs[:, l])
                    topo[n] += np.imag(vnm * Qml * vlm / denom)
                    
        return nontopo_LC,nontopo_LC_reg,nontopo_IC,nontopo_IC_reg,topo
        
    def Morb_integral(self, Ef, N=100, dk=1e-4):
        """
        Total orbital magnetization M(T=0, Ef) using projector-based Berry curvature
        and orbital moment expressions. at T=0K
        """
        b1,b2 = self.b1,self.b2
        nband = self.Hk(0,0).shape[0]
        area = np.abs(np.cross(b1, b2))
        dk_area = area / (N * N)
        e, hbar, c = 1.0, 1.0, 1.0
        prefactor = dk_area*e / (hbar * c) / (2 * np.pi)**2
        int_nontopo_LC,int_nontopo_LC_reg,int_nontopo_IC,int_nontopo_IC_reg,int_topo = np.zeros(nband),np.zeros(nband),np.zeros(nband),np.zeros(nband),np.zeros(nband)

        for i in range(N):
            for j in range(N):
                # alpha, beta = (i + 0.5) / N, (j + 0.5) / N
                alpha, beta = (i) / N, (j) / N
                kx, ky = (alpha * b1 + beta * b2)
                H_k = self.Hk(kx, ky)
                eigvals, eigvecs = np.linalg.eigh(H_k)
                nontopo_LC,nontopo_LC_reg,nontopo_IC,nontopo_IC_reg,topo = self.Morb_integrand(kx, ky, mu=Ef, dk=1e-4)
                
                occ = eigvals<Ef
                int_nontopo_LC[occ] += nontopo_LC[occ]
                int_nontopo_LC_reg[occ] += nontopo_LC_reg[occ]
                int_nontopo_IC[occ] += nontopo_IC[occ]
                int_nontopo_IC_reg[occ] += nontopo_IC_reg[occ]
                int_topo[occ] += topo[occ]
        
        return [prefactor*int_nontopo_LC, prefactor*int_nontopo_LC_reg, prefactor*int_nontopo_IC, prefactor*int_nontopo_IC_reg, prefactor*int_topo, prefactor*(int_nontopo_LC+int_nontopo_IC+int_topo)]

        
    def Longitudinal_conductivity(self, EF, *, N=60, dk=1e-4,
                                  eta_reg=1e-2,
                                  tau=1.0,
                                  eta_D=1e-2,
                                  omega=0.0):
        """
        Real-part-only, directly consistent with the note:
          - Fermi-surface delta: δ(E-EF) -> (1/π) η_D/(...)
          - Drude delta in frequency: πδ(ω) -> (1/τ)/(ω^2+(1/τ)^2) = τ/(1+ω^2τ^2)
          - Interband delta: πδ(ω-ΔE) -> η_reg/((ω-ΔE)^2+η_reg^2)
        """

        def dH_dkx(kx, ky):
            return (-self.Hk(kx + 2*dk, ky)
                    + 8*self.Hk(kx + dk,  ky)
                    - 8*self.Hk(kx - dk,  ky)
                    +    self.Hk(kx - 2*dk, ky)) / (12*dk)
    
        def dH_dky(kx, ky):
            return (-self.Hk(kx, ky + 2*dk)
                    + 8*self.Hk(kx, ky + dk)
                    - 8*self.Hk(kx, ky - dk)
                    +    self.Hk(kx, ky - 2*dk)) / (12*dk)
    
        # one Lorentz shape, two normalizations:
        #   norm="delta":   δ(x) ≈ (1/π) η/(x^2+η^2)
        #   norm="pi_delta":πδ(x)≈       η/(x^2+η^2)
        def lorentz(x, eta, norm="delta"):
            if norm == "delta":
                return (1.0/np.pi) * (eta / (x*x + eta*eta))
            elif norm == "pi_delta":
                return (eta / (x*x + eta*eta))
    
        b1, b2 = self.b1, self.b2
        area = abs(np.cross(b1, b2))
        dk_area = area / (N * N)
        prefactor = dk_area / (2*np.pi)**2
    
        acc_xx_reg = 0.0
        acc_yy_reg = 0.0
        acc_xx_dr  = 0.0   
        acc_yy_dr  = 0.0
    
        grid = np.arange(N, dtype=float) / N
        for a in grid:
            for b in grid:
                kx, ky = (a * b1 + b * b2)
                evals, evecs = self.eigenstate(kx, ky)
    
                vx = evecs.conj().T @ dH_dkx(kx, ky) @ evecs
                vy = evecs.conj().T @ dH_dky(kx, ky) @ evecs
    
                # occupations
                fn = (evals < EF).astype(float)
                occ_diff = fn[:, None] - fn[None, :]
    
                # ΔE_mn = E_m - E_n
                dE_mn = evals[None, :] - evals[:, None]
                offdiag = ~np.eye(len(evals), dtype=bool)
    
                # velocities
                v2x = np.abs(vx)**2
                v2y = np.abs(vy)**2
    
                # -------- (1) interband regular: πδ(ω-ΔE) -> η_reg/((ω-ΔE)^2+η_reg^2)
                inter_kernel = lorentz(dE_mn-omega, eta_reg, norm="pi_delta")
                acc_xx_reg += np.sum((occ_diff * v2x * inter_kernel)[offdiag])
                acc_yy_reg += np.sum((occ_diff * v2y * inter_kernel)[offdiag])
    
                # -------- (2) intraband Fermi surface: δ(E-EF) -> (1/π)η_D/(...)
                intra_kernel = lorentz(evals - EF, eta_D, norm="delta") *lorentz(omega, 1/tau, norm="pi_delta")
                vxx = np.real(np.diag(vx))
                vyy = np.real(np.diag(vy))
                acc_xx_dr += np.sum(vxx*vxx * intra_kernel)
                acc_yy_dr += np.sum(vyy*vyy * intra_kernel)
    
        sigma_xx_reg = prefactor * acc_xx_reg
        sigma_yy_reg = prefactor * acc_yy_reg
        sigma_xx_dr  = prefactor * acc_xx_dr 
        sigma_yy_dr  = prefactor * acc_yy_dr 
    
        return {
            "xx": sigma_xx_reg + sigma_xx_dr,
            "yy": sigma_yy_reg + sigma_yy_dr,
            "xx_reg": sigma_xx_reg, "yy_reg": sigma_yy_reg,
            "xx_drude": sigma_xx_dr, "yy_drude": sigma_yy_dr
        }


    

    def Hall_conductivity(self, EF, N=200):
        """
        Intrinsic Hall conductivity σ_xy from Berry curvature of occupied bands.
        """
        b1, b2 = self.b1, self.b2
        area = abs(np.cross(b1, b2))
        dk_area = area / (N * N)
        total = 0.0
        for i in range(N):
            for j in range(N):
                a, b = i / N, j / N
                kx, ky = (a * b1 + b * b2)
                curv = self.berry_curvature(kx, ky)     
                evals, _ = self.eigenstate(kx, ky)
                occ = evals < EF
                total += np.sum(curv[occ]) * dk_area
        # σ_xy = -(e^2/ℏ) ∫ Ω / (2π)^2 ; with e=ℏ=1 → just a minus sign and (2π)^2 factor
        return - total / (2*np.pi)**2


#-------------------------------------------------------------subclass models--------------------------------------------------
class KagomeNagaosa(Noninteracting_Model):
    """
    Kagome 1x1 Nagaosa flux model.

    Flux pattern:
        up triangle    : +phi
        down triangle  : +phi
        hexagon        : -2phi

    parameters:
        t   : real NN hopping magnitude
        phi : flux through each triangle
    """
    def __init__(self, parameters: dict, spin: bool = False, B: float = None, *args, **kwargs):
        super().__init__()
        required_keys = {"t", "phi"}
        if not (isinstance(parameters, dict) and parameters.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")

        self._parameters = parameters.copy()
        self.spin = spin
        self.B = B

        a = 1.0
        self.delta1 = np.array([ a/2,  a*np.sqrt(3)/2])   # A -> B
        self.delta2 = np.array([ a/2, -a*np.sqrt(3)/2])   # C -> A
        self.delta3 = np.array([-a,   0.0])               # B -> C

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        required_keys = {"t", "phi"}
        if not (isinstance(value, dict) and value.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")
        if self._parameters != value:
            self._parameters = value.copy()
            print(">>reset called")

    @property
    def b1(self):
        return np.array([0.0, 2*np.pi/np.sqrt(3)])

    @property
    def b2(self):
        return np.array([np.pi, np.pi/np.sqrt(3)])

    def _H_flux_block(self, kx, ky, phi_u, phi_d):
        t = self.parameters["t"]
        ku = phi_u / 3.0
        kd = phi_d / 3.0
        kvec = np.array([kx, ky], dtype=float)

        AB = -t * (
            np.exp(1j * (np.dot(self.delta1,  kvec) + ku)) +
            np.exp(1j * (np.dot(-self.delta1, kvec) + kd))
        )

        AC = -t * (
            np.exp(1j * (np.dot(-self.delta2, kvec) - ku)) +
            np.exp(1j * (np.dot( self.delta2, kvec) - kd))
        )

        BC = -t * (
            np.exp(1j * (np.dot(self.delta3,  kvec) + ku)) +
            np.exp(1j * (np.dot(-self.delta3, kvec) + kd))
        )

        return np.array([
            [0.0,               AB,               AC],
            [np.conjugate(AB),  0.0,              BC],
            [np.conjugate(AC),  np.conjugate(BC), 0.0]
        ], dtype=complex)

    def Hk_spin(self, kx, ky):
        phi = self.parameters["phi"]
        return self._H_flux_block(kx, ky, phi, phi)

    def Hk(self, kx, ky):
        if not self.spin:
            return self.Hk_spin(kx, ky)

        phi = self.parameters["phi"]
        H_up = self._H_flux_block(kx, ky,  phi,  phi)
        H_dn = self._H_flux_block(kx, ky, -phi, -phi)

        if self.B is None:
            return np.block([
                [H_up, np.zeros((3, 3), dtype=complex)],
                [np.zeros((3, 3), dtype=complex), H_dn]
            ])
        else:
            return np.block([
                [H_up + self.B*np.eye(3), np.zeros((3, 3), dtype=complex)],
                [np.zeros((3, 3), dtype=complex), H_dn - self.B*np.eye(3)]
            ])
            
                
class KagomeStaggerFlux(Noninteracting_Model):
    """
    Kagome nearest-neighbor staggered-flux model.

    Flux pattern:
        up triangle    : +phi
        down triangle  : -phi
        hexagon        : 0

    parameters:
        t   : real NN hopping magnitude
        phi : total flux through the up triangle
              (down triangle is fixed to -phi)

    Notes:
        1. spin=False:
           returns the spinless staggered-flux model.

        2. spin=True:
           returns TR-paired blocks:
               H_up(+phi,-phi) ⊕ H_dn(-phi,+phi)
    """
    def __init__(self, parameters: dict, spin: bool = False, B: float = None, *args, **kwargs):
        super().__init__()
        required_keys = {"t", "phi"}
        if not (isinstance(parameters, dict) and parameters.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")

        self._parameters = parameters.copy()
        self.spin = spin
        self.B = B

        a = 1.0
        self.delta1 = np.array([ a/2,  a*np.sqrt(3)/2])   # A -> B
        self.delta2 = np.array([ a/2, -a*np.sqrt(3)/2])   # C -> A
        self.delta3 = np.array([-a,   0.0])               # B -> C

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        required_keys = {"t", "phi"}
        if not (isinstance(value, dict) and value.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")
        if self._parameters != value:
            self._parameters = value.copy()
            print(">>reset called")

    @property
    def b1(self):
        return np.array([0.0, 2*np.pi/np.sqrt(3)])

    @property
    def b2(self):
        return np.array([np.pi, np.pi/np.sqrt(3)])

    def _H_flux_block(self, kx, ky, phi_u, phi_d):
        t = self.parameters["t"]
        ku = phi_u / 3.0
        kd = phi_d / 3.0
        kvec = np.array([kx, ky], dtype=float)

        AB = -t * (
            np.exp(1j * (np.dot(self.delta1,  kvec) + ku)) +
            np.exp(1j * (np.dot(-self.delta1, kvec) + kd))
        )

        AC = -t * (
            np.exp(1j * (np.dot(-self.delta2, kvec) - ku)) +
            np.exp(1j * (np.dot( self.delta2, kvec) - kd))
        )

        BC = -t * (
            np.exp(1j * (np.dot(self.delta3,  kvec) + ku)) +
            np.exp(1j * (np.dot(-self.delta3, kvec) + kd))
        )

        return np.array([
            [0.0,               AB,               AC],
            [np.conjugate(AB),  0.0,              BC],
            [np.conjugate(AC),  np.conjugate(BC), 0.0]
        ], dtype=complex)

    def Hk_spin(self, kx, ky):
        phi = self.parameters["phi"]
        return self._H_flux_block(kx, ky, phi, -phi)

    def Hk(self, kx, ky):
        if not self.spin:
            return self.Hk_spin(kx, ky)

        phi = self.parameters["phi"]
        H_up = self._H_flux_block(kx, ky,  phi, -phi)
        H_dn = self._H_flux_block(kx, ky, -phi,  phi)

        if self.B is None:
            return np.block([
                [H_up, np.zeros((3, 3), dtype=complex)],
                [np.zeros((3, 3), dtype=complex), H_dn]
            ])
        else:
            return np.block([
                [H_up + self.B*np.eye(3), np.zeros((3, 3), dtype=complex)],
                [np.zeros((3, 3), dtype=complex), H_dn - self.B*np.eye(3)]
            ])

   
class KagomeKaneMeleSOC(Noninteracting_Model):
    """
    Noninteracting kagome model with explicit Kane-Mele-type SOC hopping
    (both NN and NNN).

    parameters:
        t  : real hopping
        l1 : nearest-neighbor Kane-Mele SOC strength
        l2 : second-neighbor Kane-Mele SOC strength

    spin=False:
        returns a 3x3 single-spin / spinless block.

    spin=True:
        returns a 6x6 block-diagonal Hamiltonian
            H_up ⊕ H_dn
        where the SOC terms flip sign between up/down spin.
    """
    def __init__(self, parameters: dict, spin: bool = False, B: float = None, *args, **kwargs):
        super().__init__()
        required_keys = {"t", "l1", "l2"}
        if not (isinstance(parameters, dict) and parameters.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")

        self._parameters = parameters.copy()
        self.spin = spin
        self.B = B

        a = 1.0
        self.delta1  = np.array([ a/2,  a*np.sqrt(3)/2])   # A-B
        self.delta2  = np.array([ a/2, -a*np.sqrt(3)/2])   # C->A
        self.delta3  = np.array([-a,    0.0])              # B-C

        self.delta1p = np.array([-3*a/2,  a*np.sqrt(3)/2])
        self.delta2p = np.array([ 3*a/2,  a*np.sqrt(3)/2])
        self.delta3p = np.array([ 0.0,   -a*np.sqrt(3)])

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        required_keys = {"t", "l1", "l2"}
        if not (isinstance(value, dict) and value.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")
        if self._parameters != value:
            self._parameters = value.copy()
            print(">>reset called")

    @property
    def b1(self):
        return np.array([0.0, 2*np.pi/np.sqrt(3)])

    @property
    def b2(self):
        return np.array([np.pi, np.pi/np.sqrt(3)])

    def f(self, kx, ky):
        f1  = 2*np.cos(kx*self.delta1[0]  + ky*self.delta1[1])
        f2  = 2*np.cos(kx*self.delta2[0]  + ky*self.delta2[1])
        f3  = 2*np.cos(kx*self.delta3[0]  + ky*self.delta3[1])
        f1p = 2*np.cos(kx*self.delta1p[0] + ky*self.delta1p[1])
        f2p = 2*np.cos(kx*self.delta2p[0] + ky*self.delta2p[1])
        f3p = 2*np.cos(kx*self.delta3p[0] + ky*self.delta3p[1])
        return f1, f2, f3, f1p, f2p, f3p

    def _Hk_spin_block(self, kx, ky, s=+1):
        """
        s = +1 for spin-up, s = -1 for spin-down.
        SOC terms are odd in s.
        """
        f1, f2, f3, f1p, f2p, f3p = self.f(kx, ky)

        t  = self.parameters["t"]
        l1 = self.parameters["l1"]
        l2 = self.parameters["l2"]

        H0 = np.array([
            [0,    -t*f1,   -t*f2],
            [-t*f1, 0,      -t*f3],
            [-t*f2, -t*f3,   0   ]
        ], dtype=complex)

        H1 = np.array([
            [0,            1j*s*l1*f1,   -1j*s*l1*f2],
            [-1j*s*l1*f1,  0,             1j*s*l1*f3],
            [1j*s*l1*f2,  -1j*s*l1*f3,    0]
        ], dtype=complex)

        H2 = np.array([
            [0,            1j*s*l2*f1p,  -1j*s*l2*f2p],
            [-1j*s*l2*f1p, 0,             1j*s*l2*f3p],
            [1j*s*l2*f2p, -1j*s*l2*f3p,   0]
        ], dtype=complex)

        return H0 + H1 + H2

    def Hk_spin(self, kx, ky):
        """
        For compatibility with your other classes:
        returns the spin-up block.
        """
        return self._Hk_spin_block(kx, ky, s=+1)

    def Hk(self, kx, ky):
        if not self.spin:
            return self._Hk_spin_block(kx, ky, s=+1)

        H_up = self._Hk_spin_block(kx, ky, s=+1)
        H_dn = self._Hk_spin_block(kx, ky, s=-1)

        if self.B is None:
            return np.block([
                [H_up, np.zeros((3, 3), dtype=complex)],
                [np.zeros((3, 3), dtype=complex), H_dn]
            ])
        else:
            return np.block([
                [H_up + self.B*np.eye(3), np.zeros((3, 3), dtype=complex)],
                [np.zeros((3, 3), dtype=complex), H_dn - self.B*np.eye(3)]
            ])


                
class threesites_model_singleorb(Noninteracting_Model):
    """
    Son class, with upcasting function deg Hk
    parameters:
    t0,t1,h0,h1: real and imaginary hopping strength in center triangle / edge triangles
    """
    def __init__(self, parameters:dict, spin:bool=False, B:float=None, *args, **kwargs): 
        super().__init__() #constructor for father class
        required_keys = {"t0","t1","h0","h1"}
        if not (isinstance(parameters, dict) and parameters.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")
        self._parameters, self.spin, self.B = parameters, spin, B
        a = 2 # a: length of bravis vector
        self.delta1, self.delta2, self.delta3 = np.array([0,-a/np.sqrt(3)]), np.array([a/2, a/(np.sqrt(3)*2)]), np.array([-a/2, a/(np.sqrt(3)*2)]) # NN hopping vector
        
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if self._parameters != value:
            self._parameters = value
            print(">>reset called")

    @property  #bravis vector
    def b1(self):
        return np.array([0,2*np.pi/np.sqrt(3)])
    @property
    def b2(self):
        return np.array([np.pi,np.pi/np.sqrt(3)])
        
    def Hk_spin(self,kx,ky):   
        t0,t1,h0,h1 = self.parameters["t0"],self.parameters["t1"],self.parameters["h0"],self.parameters["h1"]
        #hopping: A->B
        AB = 0
        #(A->B)_1  
        AB += (-t0+1j*h0)*np.exp(1j*np.dot(self.delta1,[kx,ky])) 
        #(A->B)_2 + (A->B)_3
        AB += (-t1+1j*h1)*np.exp(1j*np.dot(self.delta2,[kx,ky])) + (-t1+1j*h1)*np.exp(1j*np.dot(self.delta3,[kx,ky]))
        #hopping: A->C
        AC = 0
        #(A->C)_1  
        AC += (-t0-1j*h0)*np.exp(1j*np.dot(-self.delta3,[kx,ky])) 
        #(A->C)_2 + (A->C)_3
        AC += (-t1-1j*h1)*np.exp(1j*np.dot(-self.delta1,[kx,ky])) + (-t1-1j*h1)*np.exp(1j*np.dot(-self.delta2,[kx,ky]))
        #hopping: B->C
        BC = 0
        #(B->C)_1  
        BC += (-t0+1j*h0)*np.exp(1j*np.dot(self.delta2,[kx,ky])) 
        #(B->C)_2 + (B->C)_3
        BC += (-t1+1j*h1)*np.exp(1j*np.dot(self.delta3,[kx,ky])) + (-t1+1j*h1)*np.exp(1j*np.dot(self.delta1,[kx,ky]))
        
        return np.array([
            [0, AB, AC],
            [np.conjugate(AB), 0, BC],
            [np.conjugate(AC), np.conjugate(BC), 0]
        ])

    def Hk(self,kx,ky):
        if not self.spin: #spinless
            return self.Hk_spin(kx,ky)
        else:
            if self.B is None:
                raise ValueError("Zeeman field should be given with spinful system") 
            else:
                return np.block([
        [self.Hk_spin(kx,ky)+self.B*np.eye(3) , np.zeros((3, 3))],
        [np.zeros((3, 3)), self.Hk_spin(kx,ky)-self.B*np.eye(3)]])
                
class threesites_model(Noninteracting_Model):
    """
    Son class, with upcasting function deg Hk
    parameters:
    t0,t1,h0,h1: real and imaginary hopping strength in center triangle / edge triangles
    s,p: slater-koster coefficients
    """
    def __init__(self, parameters:dict, spin:bool=False, B:float=None, *args, **kwargs): 
        super().__init__() #constructor for father class
        required_keys = {"t0","t1","h0","h1","s","p"}
        if not (isinstance(parameters, dict) and parameters.keys() == required_keys):
            raise ValueError(f"input parameters must be a dict, with .keys = {required_keys}")
        self._parameters, self.spin, self.B = parameters, spin, B
        self._bond_xy, self._bond_rt= None, None
        a = 2 # a: length of bravis vector
        self.delta1, self.delta2, self.delta3 = np.array([0,-a/np.sqrt(3)]), np.array([a/2, a/(np.sqrt(3)*2)]), np.array([-a/2, a/(np.sqrt(3)*2)]) # NN hopping vector
        
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if self._parameters != value:
            self._bond_xy, self._bond_rt= None, None #reset the bond operator
            self._parameters = value
            print(">>reset called")
            
    @property  #bravis vector
    def b1(self):
        return np.array([0,2*np.pi/np.sqrt(3)])
    @property
    def b2(self):
        return np.array([np.pi,np.pi/np.sqrt(3)])
        
    @property
    def bond_xy(self): 
        '''
        there are nine bonds in 3 sites triangle lattice, each bonds has a 2by2 hopping matrix for px,py orbitals
        unit vectors of bond order is given as: 
        (A-B)_1 : delta1 * (within UC)
        (A-B)_2 : delta2
        (A-B)_3 : delta3
        (A-C)_1 : -delta3  * (within UC)
        (A-C)_2 : -delta1 
        (A-C)_3 : -delta2 
        (B-C)_1 : delta2 * (within UC)
        (B-C)_2 : delta3  
        (B-C)_3 : delta1 
        '''
        if self._bond_xy is None:
            unit_vectors = [v / np.linalg.norm(v) for v in [self.delta1, self.delta2, self.delta3,
                -self.delta3, -self.delta1, -self.delta2,
                self.delta2, self.delta3, self.delta1]]
            def slater_koster(nx,ny):
                s,p = self.parameters["s"], self.parameters["p"]
                return np.array([[nx*nx*s + (1-nx*nx)*p, nx*ny*(s-p)],
                                 [nx*ny*(s-p),           ny*ny*s + (1-ny*ny)*p]], dtype=float)
            self._bond_xy = []
            for n in unit_vectors:
                self._bond_xy.append(slater_koster(n[0],n[1]))
        return self._bond_xy

    @property
    def bond_rt(self): 
        '''
        there are nine bonds in 3 sites triangle lattice, each bonds has a 2by2 hopping matrix for pr,pt orbitals
        unit vectors of bond order is the same as bond_xy
        '''
        if self._bond_rt is None:
            
            def C3_rotation():
                theta = [0.0, 4*np.pi/3, 2*np.pi/3]  
                def R_theta(theta):
                    M3 = np.array([[-1,0],[0,-1]])  #(pr,pt)_c = (-px,-py)_c = M3 * (px,py)
                    return np.dot(np.array([[cos(theta),  -sin(theta)],
                                     [sin(theta), cos(theta)]], dtype=float),M3) #clockwise
                return [R_theta(theta[-1]), R_theta(theta[-2]), R_theta(theta[-3])] # [UA, UB, UC]
            UC3 = C3_rotation()
            
            #rotation operator corresponding to bond_xy
            U_R = [UC3[0]]*6 + [UC3[1]]*3
            U_L = [UC3[1].T]*3 + [UC3[2].T]*6
            
            self._bond_rt = []
            for b,bonds in enumerate(self.bond_xy):
                self._bond_rt.append(U_R[b]@bonds@U_L[b])
                
        return self._bond_rt
        
    def Hk_spin(self,kx,ky):
        t0,t1,h0,h1 = self.parameters["t0"],self.parameters["t1"],self.parameters["h0"],self.parameters["h1"]
        #hopping: A->B
        AB = np.zeros((2,2),dtype=complex)
        #(A->B)_1  
        AB += (-t0+1j*h0)*self.bond_rt[0]*np.exp(1j*np.dot(self.delta1,[kx,ky])) 
        #(A->B)_2 + (A->B)_3
        AB += (-t1+1j*h1)*self.bond_rt[1]*np.exp(1j*np.dot(self.delta2,[kx,ky])) + (-t1+1j*h1)*self.bond_rt[2]*np.exp(1j*np.dot(self.delta3,[kx,ky]))
        #hopping: A->C
        AC = np.zeros((2,2),dtype=complex)
        #(A->C)_1  
        AC += (-t0-1j*h0)*self.bond_rt[3]*np.exp(1j*np.dot(-self.delta3,[kx,ky])) 
        #(A->C)_2 + (A->C)_3
        AC += (-t1-1j*h1)*self.bond_rt[4]*np.exp(1j*np.dot(-self.delta1,[kx,ky])) + (-t1-1j*h1)*self.bond_rt[5]*np.exp(1j*np.dot(-self.delta2,[kx,ky]))
        #hopping: B->C
        BC = np.zeros((2,2),dtype=complex)
        #(B->C)_1  
        BC += (-t0+1j*h0)*self.bond_rt[6]*np.exp(1j*np.dot(self.delta2,[kx,ky])) 
        #(B->C)_2 + (B->C)_3
        BC += (-t1+1j*h1)*self.bond_rt[7]*np.exp(1j*np.dot(self.delta3,[kx,ky])) + (-t1+1j*h1)*self.bond_rt[8]*np.exp(1j*np.dot(self.delta1,[kx,ky]))
        return np.block([
            [np.zeros((2,2)), AB, AC],
            [np.conjugate(AB).T, np.zeros((2,2)), BC],
            [np.conjugate(AC).T, np.conjugate(BC).T, np.zeros((2,2))]]) 
        
        
    def Hk(self,kx,ky):
        if not self.spin: #spinless
            return self.Hk_spin(kx,ky)
        else:
            if self.B is None:
                raise ValueError("Zeeman field should be given with spinful system") 
            else:
                return np.block([
        [self.Hk_spin(kx,ky)+self.B*np.eye(6) , np.zeros((6, 6))],
        [np.zeros((6, 6)), self.Hk_spin(kx,ky)-self.B*np.eye(6)]])