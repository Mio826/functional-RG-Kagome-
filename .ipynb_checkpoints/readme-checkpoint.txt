I am building a functional RG (fRG) pipeline for kagome lattice systems.

🎯 **Ultimate goal**
- Reproduce the phase diagram in the PRL paper on kagome Hubbard model (van Hove filling)
- Identify leading instabilities (FM, PI, cBO, sBO, f-SC, etc.)
- Then extend the pipeline to study modified models:
    - flux patterns
    - SOC (e.g. Rashba)
    - different interaction structures

So this is both:
- a **reproduction project**
- and a **general-purpose fRG framework**

--------------------------------------------------

🧠 **Current pipeline structure (4 modules)**

The whole pipeline is modular and follows the standard fRG logic (Metzner RMP, one-loop truncation).

----------------------------------------

### Module 1: Model construction (input layer)

Defines the microscopic system and prepares the initial vertex.

Includes:

1. Non-interacting model
   - Tight-binding Hamiltonian (kagome, possibly spinful)
   - Supports flux / SOC extensions
   - Computes:
        eigenvalues ε(k)
        eigenvectors u(k)

2. Interaction (extended Hubbard)
   - Bare interaction in orbital basis
   - Projection to band basis via Bloch eigenstates
   - Produces antisymmetrized vertex:
        Γ(k1,k2,k3,k4)

3. Fermi surface patching
   - Discretizes FS into patches
   - Stores:
        patch momenta
        Bloch eigenvectors
   - Spin-aware patch sets (up/down handled explicitly)

4. Channel decomposition (bookkeeping layer)
   - Converts Γ into:
        pp / ph / ph' channel matrices
   - Defines:
        momentum routing conventions
        patch index mapping (Q ± k → nearest patch)

IMPORTANT:
- This module defines the **index conventions of the vertex**
- Everything downstream depends on this

----------------------------------------

### Module 2: Order diagnosis (output layer)

Analyzes a given vertex or channel kernel.

Two levels:

1. Form factor analysis
   - Diagonalizes channel kernels
   - Extracts leading eigenmodes
   - Handles degeneracy
   - Classifies symmetry (s, p, d, f, etc.)

2. Kagome-specific diagnosis
   - Reconstructs internal tensor structure:
        Φ (particle-hole)
        Δ (particle-particle)
   - Matches against known kagome orders:
        FM, PI, cBO, sBO, f-SC
   - Outputs:
        identified phase OR "unclassified"

----------------------------------------

### Module 3: One-loop kernel

Implements the fRG flow equation (Metzner Eq. 52).

Includes:

- Temperature cutoff scheme
- Propagators:
    G (full)
    S (single-scale)

- Three one-loop contributions:
    particle-particle (pp)
    particle-hole direct (ph)
    particle-hole crossed (ph')

Outputs:
    dΓ/dT in channel representation

----------------------------------------

### Module 4: RG flow (current focus)

Combines everything into an actual flow.

Key design:

- Vertex stored in channel form:
    Γ = Γ_bare + Φ_pp + Φ_phd + Φ_phc

- Avoids full Γ(p1,p2,p3,p4) tensor (O(Np^4))
- Uses:
    transfer momentum grids (Q)
    patch representation

Flow procedure:

1. Initialize Γ_bare
2. At each step:
    - compute RHS via one-loop kernel
    - update channel corrections
3. Periodically:
    - reconstruct vertex
    - run order diagnosis
4. Detect instability via:
    - divergence of eigenvalues

----------------------------------------

🧩 **How modules interact**

- Module 1 → provides:
    patchsets, bare interaction, Γ_bare

- Module 3 → uses:
    patchsets + Γ (via accessor)
    to compute RHS

- Module 4 → orchestrates:
    flow loop and state update

- Module 2 → analyzes:
    intermediate Γ during flow

----------------------------------------

🚨 **Important implementation details**

- Spin is explicitly treated (spinful system)
- Only spin-conserving processes are kept
- Channel representation is used for scalability:
    reduces O(Np^4) → O(NQ · Np^2)

- Momentum conservation handled via patch mapping
- Transfer momentum Q is discretized

--------------------------------------------------

IMPORTANT:
- Do not give generic explanations
- Always connect to my pipeline structure
- Be precise about:
    indices, channels, momentum routing, spin structure

--------------------------------------------------
