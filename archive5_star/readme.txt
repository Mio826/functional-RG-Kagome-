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

4. Momentum indexing backbone
   - Defines:
        momentum conservation
        mapping between patch indices
        transfer momentum Q conventions

IMPORTANT:
- This module defines **all index conventions**
- Any inconsistency here propagates through the entire pipeline

----------------------------------------

### Module 2: Order diagnosis (analysis layer)

Analyzes channel kernels reconstructed from the full vertex.

Two levels:

1. Mother-kernel construction
   - Builds channel kernels from Γ:
        pp: pair space (k, Q-k)
        ph: bilinear space (k → k+Q)
   - Keeps full spin ⊗ momentum tensor structure
   - No premature projection into charge/spin or singlet/triplet

2. Eigenmode analysis
   - Diagonalizes mother kernels
   - Extracts leading eigenmodes
   - Interprets eigenvectors as:
        pairing symmetry (pp)
        density/spin patterns (ph)

3. Kagome-specific diagnosis (optional)
   - Matches eigenmodes against known kagome orders:
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

- FULL vertex Γ is used directly in all diagrams
- No reduced-channel storage
- Internal loops evaluated in patch basis
- Momentum conservation enforced explicitly
- Q handled via canonicalization:
        Q ≡ Q + G

Outputs:
    dΓ/dT in full vertex representation

----------------------------------------

### Module 4: RG flow (orchestration layer)

Drives the full flow.

Representation:

    Γ(s1,s2,s3,s4; p1,p2,p3)

with:
    p4 determined by momentum conservation

Design choices:

- Store full vertex (3 independent momentum indices)
- No decomposition:
        Γ ≠ Γ_bare + Φ_pp + Φ_phd + Φ_phc
- Channel decomposition only used inside one-loop kernel

Flow procedure:

1. Initialize Γ_bare
2. For each temperature step:
    - compute RHS using one-loop diagrams
    - update full Γ
3. Periodically:
    - build channel kernels from Γ
    - perform eigenmode analysis
4. Detect instability via:
    - divergence of leading eigenvalue

----------------------------------------

🧩 **Momentum structure (CRITICAL DESIGN)**

This pipeline uses a **patch-driven definition of transfer momentum Q**:

- Q is NOT externally imposed
- Q is generated from patch combinations:
    pp:  Q = k1 + k2
    ph:  Q = k3 - k1

- All Q values are:
    - canonicalized modulo reciprocal lattice vectors
    - grouped with tolerance in reduced coordinates

- Internal loops use Q-constrained partner mapping:
    pp: k ↔ Q - k
    ph: k ↔ k + Q

----------------------------------------

🚨 **Key consistency requirement**

Physical invariance:

    Q ≡ Q + G   (G: reciprocal lattice vector)

must hold exactly.

Implementation details:

- All Q are canonicalized in reciprocal space
- Same Q definition used in:
    - flow (internal loops)
    - vertex storage (closure)
    - diagnosis (kernel construction)
- No fallback or mixed closure rules

This avoids:

- artificial Q-dependence
- incorrect kernel reconstruction
- silent zeroing of matrix elements

----------------------------------------

🧠 **How modules interact**

- Module 1 → provides:
    patchsets, Γ_bare

- Module 3 → computes:
    dΓ/dT from Γ

- Module 4 → updates:
    full vertex Γ

- Module 2 → analyzes:
    kernels reconstructed from Γ

----------------------------------------

⚙️ **Current approximation level**

- One-loop fRG
- Temperature-flow cutoff
- Static vertex (no frequency dependence)
- No self-energy flow

Vertex structure:

- Full vertex stored explicitly
- Only 6 spin-conserving blocks retained
- Forbidden spin sectors set to zero

Momentum treatment:

- Patch discretization on FS
- Transfer momenta generated from patch combinations
- All momenta treated modulo reciprocal lattice vectors

----------------------------------------

📌 **Notes**

- Spin is treated explicitly
- Momentum routing is fixed and consistent across modules
- Channel decomposition is used only for:
    diagram evaluation and diagnosis
- The pipeline is designed for:
    correctness → consistency → extensibility → performance