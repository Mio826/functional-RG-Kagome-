I am building a functional RG (fRG) pipeline for kagome lattice systems.

🎯 **Ultimate goal**
- Reproduce the PRL phase diagram of the kagome Hubbard model (van Hove filling)
- Identify leading instabilities:
    FM, PI, cBO, sBO, f-SC, etc.
- Extend to modified models:
    - flux patterns
    - spin-orbit coupling (e.g. Rashba)
    - generalized interaction structures

This project is both:
- a **benchmark reproduction**
- and a **general-purpose fRG framework**

--------------------------------------------------

🧠 **Pipeline structure (4 modules)**

The pipeline follows standard one-loop fRG (Metzner RMP), implemented in a modular way.

----------------------------------------

### Module 1: Model construction (input layer)

Defines the microscopic system and prepares the initial vertex.

Includes:

1. Non-interacting model
   - Kagome tight-binding Hamiltonian
   - Supports flux / SOC extensions
   - Computes:
        eigenvalues ε(k)
        eigenvectors u(k)

2. Interaction (extended Hubbard)
   - Defined in orbital basis
   - Projected to band basis via Bloch eigenvectors
   - Produces antisymmetrized vertex:
        Γ(k1,k2,k3,k4)

3. Fermi surface patching
   - Discretizes FS into patches
   - Stores:
        patch momenta k_i
        Bloch eigenvectors u(k_i)
   - Spin-resolved patch sets (up/down treated explicitly)

4. Channel decomposition (index backbone)
   - Converts Γ into:
        pp / ph / ph' channel matrices
   - Defines:
        momentum routing conventions
        mapping between patch indices and transfer momentum Q

IMPORTANT:
- This module defines **all index conventions**
- Any inconsistency here propagates through the entire pipeline

----------------------------------------

### Module 2: Order diagnosis (analysis layer)

Analyzes channel kernels during the flow.

Two levels:

1. Form factor analysis
   - Diagonalizes channel kernels
   - Extracts leading eigenmodes
   - Handles degeneracies
   - Classifies symmetry:
        s, p, d, f, etc.

2. Kagome-specific diagnosis
   - Reconstructs internal order tensors:
        Φ (particle-hole)
        Δ (particle-particle)
   - Matches against known kagome orders:
        FM, PI, cBO, sBO, f-SC
   - Outputs:
        identified phase OR "unclassified"

----------------------------------------

### Module 3: One-loop kernel (physics core)

Implements the fRG flow equation (Metzner Eq. 52).

Includes:

- Temperature-flow cutoff
- Propagators:
    G (full)
    S (single-scale)

- Three channels:
    particle-particle (pp)
    particle-hole direct (ph)
    particle-hole crossed (ph')

Key implementation:

- Internal loops evaluated in patch basis
- Momentum conservation enforced via patch mapping
- Vectorized Matsubara summation (performance-critical)

Outputs:
    dΓ/dT in channel representation

----------------------------------------

### Module 4: RG flow (orchestration layer)

Drives the full flow.

Representation:

    Γ = Γ_bare + Φ_pp + Φ_phd + Φ_phc

Design choices:

- Avoid full Γ(p1,p2,p3,p4) tensor (O(Np^4))
- Work in channel representation:
    O(NQ · Np^2)

Flow procedure:

1. Initialize Γ_bare
2. For each temperature step:
    - compute RHS (one-loop kernel)
    - update channel corrections
3. Periodically:
    - reconstruct vertex (via accessor)
    - run order diagnosis
4. Detect instability via:
    - leading eigenvalue growth

----------------------------------------

🧩 **Momentum structure (CRITICAL DESIGN)**

This pipeline uses a **patch-driven definition of transfer momentum Q**:

- Q is NOT externally imposed
- Q is generated from patch combinations:
    pp:  Q = k1 + k2
    ph:  Q = k3 - k1

- All Q values are:
    - canonicalized modulo reciprocal lattice vectors
    - indexed via transfer grids

- Channel storage is organized by:
    (spin block, Q index, patch indices)

----------------------------------------

🚨 **Key consistency requirement (recently fixed)**

Physical invariance:

    Q ≡ Q + G   (G: reciprocal lattice vector)

must hold exactly.

Implementation details:

- All Q are canonicalized in reciprocal space
- Partner patch mapping is constrained by:
    fixed transfer index (iq)
- No free "nearest patch" mapping across different Q sectors
- External and internal legs use the SAME Q definition

This avoids:

- artificial Q-dependence
- unphysical divergence
- incorrect order diagnosis

----------------------------------------

🧠 **How modules interact**

- Module 1 → provides:
    patchsets, Γ_bare

- Module 3 → computes:
    dΓ/dT from current Γ

- Module 4 → updates:
    channel representation of Γ

- Module 2 → analyzes:
    channel kernels at selected steps

----------------------------------------

⚙️ **Current approximation level**

- One-loop fRG
- Temperature-flow cutoff
- Static vertex (no frequency dependence)
- No self-energy flow

Vertex structure:

- Stored in reduced channel form
- Only 6 spin-conserving spin blocks retained
- Forbidden spin sectors set to zero

Momentum treatment:

- Patch discretization on FS
- Transfer momenta discretized from patch combinations
- All momenta treated modulo reciprocal lattice vectors

----------------------------------------

📌 **Notes**

- Spin is treated explicitly
- Momentum routing is fixed and consistent across modules
- The pipeline is designed for:
    correctness first → then performance → then extensibility