fRG Kagome Pipeline README

Overview

This project is a temperature-flow one-loop functional RG pipeline for kagome-type lattice models with spin-conserving single-particle Hamiltonians and density-density interactions. The code is organized as a staged pipeline:

noninteracting.py → patching.py → interaction.py → frg_kernel.py → frg_flow.py → channels.py → instability.py

The design principle is:

1. Build the noninteracting band structure and Bloch eigenvectors.
2. Construct spin-resolved patch sets on the Fermi-surface loop.
3. Project the bare density-density interaction into the Bloch basis.
4. Run a temperature-flow one-loop RG for a PRL-compatible minimal S_z=0 vertex.
5. Reconstruct physical channels: pp singlet/triplet and ph charge/spin.
6. Diagnose the leading ordering tendency using both raw channel kernels and bubble-dressed instability operators.

The important point is that the code does not store the full spin-dependent four-point vertex as an independent flow object. Instead, it stores a minimal PRL-compatible scalar vertex and reconstructs the needed physical channels later.

1. Module-by-module pipeline and coordination

1.1 noninteracting.py

Role:

* Defines the noninteracting lattice models.
* Supplies H(k), eigenvalues, eigenvectors, reciprocal vectors, and band geometry.

Typical outputs used downstream:

* model.Hk(kx, ky)
* model.eigenstate(kx, ky)
* model.b1, model.b2
* model.delta1, delta2, delta3 for kagome nearest-neighbor structure

Why this matters:

* patching.py uses the model to build patch representatives on a chosen loop.
* interaction.py uses the Bloch eigenvectors on those patch points.
* patching.py also uses the band energy and velocity derived from H(k).

1.2 patching.py

Role:

* Builds the patch sets that discretize the Fermi-surface loop or the chosen M-loop.
* Stores standard patch data in PatchPoint / PatchSet.

Key outputs:

* PatchPoint: k_cart, k_red, energy, vF, vF_norm, eigvec, orbital_weight
* Optional geometric data: tangent, normal, fs_arc_length, weight_length, weight_length_over_vf
* PatchSet: the full collection for one spin sector

Why this matters:

* interaction.py evaluates the bare vertex on patch representatives.
* frg_kernel.py needs patch energies, patch momenta, and optionally patch measures.
* frg_flow.py uses the patch sets to build transfer grids, partner maps, and closure maps.
* instability.py reuses the same patch data when constructing bubble weights.

1.3 interaction.py

Role:

* Defines the bare extended Hubbard interaction in the Bloch basis.
* Converts orbital-space density-density interactions into band / patch vertices.

Important conceptual point:

* The raw density-density interaction is not the same object as the PRL-compatible minimal flow variable.
* The code constructs a minimal S_z=0 scalar vertex:
  V_sz0(1,2;3,4) = Γ_{up,dn -> dn,up}(1,2 -> 3,4)

This is produced from the antisymmetrized vertex, not from the raw direct density-density matrix element.

Why this matters:

* frg_flow.py evolves this minimal scalar vertex.
* channels.py later reconstructs physical pp/ph kernels from that minimal object.

1.4 frg_kernel.py

Role:

* Provides the technical core for one-loop building blocks.
* Handles Q canonicalization, transfer grids, partner maps, Matsubara bubbles, patch measures, and minimal-S_z=0 one-loop formulas.

Important outputs:

* TransferGrid
* build_unique_q_list(...)
* canonicalize_q_for_patchsets(...)
* partner_map_from_q_index(...)
* build_pp_internal_cache_vec(...)
* build_ph_internal_cache_vec(...)
* compute_pp_vertex_contribution_sz0(...)
* compute_phd_vertex_contribution_sz0(...)
* compute_phc_vertex_contribution_sz0(...)

Why this matters:

* frg_flow.py calls frg_kernel.py every RG step to build one-loop right-hand-side contributions.

1.5 frg_flow.py

Role:

* Owns the actual RG state and time stepping.
* Stores the flowing minimal vertex V(p1,p2,p3), with p4 reconstructed from closure.
* Evolves the RG equation dV/dT = Φ_pp + Φ_phd + Φ_phc.

Important outputs:

* FRGFlowSolverSZ0
* FlowStepRecord
* diagnosis_payload
* closure_map()
* transfer_context()

Why this matters:

* channels.py uses the solver’s closure and transfer information to reconstruct physical channel kernels.
* instability.py uses the transfer context plus the solver’s measure settings to stay consistent with flow.

1.6 channels.py

Role:

* Takes the flowed minimal S_z=0 vertex and reconstructs physical channel kernels at fixed Q.

Physical channels:

* pp_singlet
* pp_triplet
* ph_charge
* ph_spin

This is a diagnosis / interpretation layer. It does not change the RG flow itself.

1.7 instability.py

Role:

* Builds bubble-dressed instability operators from channel kernels.
* Produces a more physical diagnosis than raw kernel eigenvalues alone.

This is where the code asks whether the current flowed kernel at a given Q already looks unstable in a physically meaningful way.

2. The main mathematical object of the whole code

The central design choice is:

V(1,2;3,4) ≡ Γ_{up,dn -> dn,up}(1,2;3,4)

This is the minimal S_z=0 scalar vertex.

Why store this instead of the full spin tensor?

* For a spin-conserving density-density interaction with SU(2)-compatible structure, the full spin-dependent vertex can be reconstructed from this object and its outgoing-leg exchange.
* This avoids explicitly storing many spin blocks that are not independent.

The reconstruction logic is:

Γ_{σ1σ2σ3σ4}(1,2;3,4)
= V(1,2;3,4) δ_{σ1σ4} δ_{σ2σ3}

* V(1,2;4,3) δ_{σ1σ3} δ_{σ2σ4}

This formula encodes spin structure, fermionic antisymmetry, and the fact that same-spin scattering is not stored independently but reconstructed later.

3. Important technical handling in each module

3.1 noninteracting.py: model, bands, Bloch states

Important techniques:

* Every model provides reciprocal vectors b1, b2.
* Every model provides H(k), and hence energies and Bloch eigenvectors.
* The kagome-specific models also expose bond vectors delta1, delta2, delta3, needed by interaction.py.

Coding detail:

* The code assumes spin-conserving one-body structure. Even if the model is spinful, it should not mix up and down sectors directly.
* This allows separate patch sets for different spin sectors.

Physics detail:

* Bloch eigenvectors matter because the bare interaction is not used in raw orbital form in the RG. It is first projected into the band basis.

3.2 patching.py: patch data structure, geometry, gauge, Fermi-surface projection

Important techniques:

A. Standard data structure

* PatchPoint stores momentum, reduced coordinates, energy, Fermi velocity, Bloch eigenvector, orbital weights, and optional geometric data like arc length and tangent/normal.
* PatchSet stores the whole patch collection plus reciprocal vectors and metadata.

B. Canonicalization to centered 1BZ

* canonicalize_k_to_centered_1bz(...) chooses a representative in the centered hexagonal first BZ.
* This is used for plotting and for a stable geometric representation of patch points.

C. Modulo-G equivalence handling

* _canonicalize_k_mod_G(...) is used for reciprocal-lattice equivalence checks.
* This is separate from centered-1BZ plotting canonicalization.

D. Exact M-loop construction

* exact_M6_points_1bz(...) builds the six geometric M points.
* exact_M_hex_loop_points(...) interpolates along the M-hexagon edges.
* exact_M_hex_loop_points_global_angular(...) samples the loop by global angle.

E. Optional patch concentration near M points

* The current file includes _edge_cluster_parameter(...) and edge_cluster_alpha to cluster points toward edge endpoints, namely toward M points.

F. Gauge fixing

* smooth_patch_eigvecs(...) parallel-transports Bloch phases patch to patch.
* This removes arbitrary random phase jumps between neighboring patch eigenvectors.

Why this matters physically:

* The interaction projection depends on Bloch eigenvectors.
* If neighboring patch eigenvectors carry random phases, cancellations or fake structures can appear in projected bare vertices.

G. Fermi-surface projection

* project_to_fs(...) uses a Newton step along the velocity direction to move each patch representative onto E(k)=μ.

Why this matters:

* The manual M-loop does not automatically lie exactly on the Fermi surface.
* The code chooses to project the kept patch points onto E=μ so patch energies are close to 0.

H. Patch measure data

* _compute_patch_arc_lengths(...) defines a discrete FS arc-length proxy.
* patch_weight_length ≈ arc length
* patch_weight_length_over_vf ≈ arc length / |v_F|

These are later used by frg_kernel.py to construct patch measures.

3.3 interaction.py: bare density-density interaction and minimal S_z=0 object

Important techniques:

A. Direct density-density orbital kernel

* orbital_interaction_matrix(q, s1,s2,s3,s4) builds the orbital-space direct interaction W_ab(q).

Physics:

* For density-density interactions, the raw direct process obeys s3 = s1 and s4 = s2.
* So the bare direct interaction itself does not directly look like up,dn -> dn,up.

B. Direct vs antisymmetrized vertex

* direct_band_vertex(...) computes V_dir.
* antisym_band_vertex(...) computes Γ = V_dir(1,2->3,4) - V_dir(1,2->4,3).

C. Minimal S_z=0 convention

* band_vertex_sz0(...) and patch_vertex_sz0(...) define V_sz0 = Γ_{up,dn -> dn,up}.
* This is the PRL-compatible convention used by the flow.

Important physics point:

* Because the original interaction is density-density, the meaningful bare minimal object must come from the antisymmetrized vertex.
* Therefore antisym=True is the correct default.

Why this matters:

* If one mistakenly uses the raw direct density-density matrix element as the minimal PRL object, the downstream pp/ph decomposition becomes inconsistent.

3.4 frg_kernel.py: Q handling, transfer grids, partner maps, bubbles, and one-loop kernels

This is the most technical module.

A. Q canonicalization

* canonicalize_q_for_patchsets(...) maps a momentum transfer q to a unique reciprocal-lattice representative using patchset reciprocal vectors.
* TransferGrid then works in reduced coordinates and merges nearly equivalent q values.

Why this is necessary:

* In a lattice, Q and Q+G are physically equivalent.
* Numerically, different patch pairs often generate q values that differ by reciprocal vectors or tiny floating-point noise.
* If the code treated them as different, one physical Q would be split into many fake copies.

B. TransferGrid and small numerical differences

* TransferGrid stores a unique q_list.
* It uses exact rounded keys in reduced coordinates plus a reduced-coordinate merge tolerance merge_tol_red.
* nearest_index(...) first tries exact matching, then near-matching, then nearest reduced-distance fallback.

Coding meaning:

* This is the main defense against floating-point Q mismatch.
* It prevents the same physical M point from being treated as several different transfer momenta.

C. build_unique_q_list(...)

* Builds the candidate Q catalogue from patch momenta.
* For pp: Q = k1 + k2
* For ph/phc: Q = k3 - k1
  Then it canonicalizes and deduplicates them through TransferGrid.

D. partner_map_from_q_index(...)

* Given a target Q and a source patch, this finds the target patch that best matches Q-k, k+Q, or k-Q depending on mode.
* It also returns a residual distance.

Physics meaning:

* The discrete patch language replaces exact momentum equalities by nearest representative matches.
* The residual tells you how well the closure works numerically.

E. Matsubara bubbles

* bubble_dot_pp(...)
* bubble_dot_ph(...)
* vectorized versions _bubble_dot_pp_vec / _bubble_dot_ph_vec

These compute the temperature-flow derivatives of the pp and ph bubbles.

F. Patch measure
The cleaned-up code supports:

* unit
* length
* length_over_vf
* length_over_vf_soft

Through patch_measure_vector(...), the one-loop cache weight becomes:

weight_p = measure_p × bubble_p

rather than only bubble_p.

Why this matters physically:

* A continuum FS integral is not the same as an equal-weight patch sum.
* The measure attempts to approximate FS phase-space weighting using arc length and possibly 1/|v_F|.

G. Minimal internal cache

* build_pp_internal_cache_vec(...)
* build_ph_internal_cache_vec(...)

These store, for each source patch:

* partner patch
* residual
* weight

This is the actual discrete approximation to the internal momentum sum.

H. One-loop minimal-S_z=0 formulas

* compute_pp_vertex_contribution_sz0(...)
* compute_phd_vertex_contribution_sz0(...)
* compute_phc_vertex_contribution_sz0(...)

These are not generic full-spin formulas. They are the reduced one-loop expressions for the minimal PRL-compatible vertex V(1,2;3,4)=Γ_{up,dn->dn,up}.

This is a crucial point:

* The flow object is already reduced.
* Therefore the one-loop kernels are carefully derived minimal formulas, not naive full-spin copies.

3.5 frg_flow.py: solver, closure map, state tensor, adaptive integration

Important techniques:

A. Bare vertex adapter

* BareSZ0VertexFromInteraction wraps interaction.py and ensures the flow sees the minimal PRL-compatible bare object.

B. Stored tensor shape

* The solver stores V(p1,p2,p3), and p4 is not treated as an independent axis.
* p4 is reconstructed from the closure map.

This reduces memory relative to a naive full four-index storage.

C. Closure map

* _precompute_closure_map_sz0() finds, for every (p1,p2,p3), the best p4 compatible with momentum conservation modulo reciprocal lattice vectors.
* It stores p4 index and p4 residual.

Why this matters:

* In patch language, exact closure is generally impossible.
* The solver must define one canonical p4 representative and record how good that match is.

D. drop_inexact_closure and closure_tol

* If enabled, tuples with too-large closure residual are masked out.

This is a numerical consistency control. It avoids trusting highly inexact discrete closures.

E. Shift maps and internal caches

* _precompute_shift_maps() and _refresh_cache_weights() prepare internal pp/ph/phc cache data for each Q.

F. RG state

* SZ0Tensor holds data, p4_index, and p4_residual.
* SZ0FlowState adds current temperature and transfer grids.

G. Flow equation

* compute_vertex_rhs(T) computes RHS = Φ_pp + Φ_phd + Φ_phc.
* Each term uses the minimal-S_z=0 one-loop formula from frg_kernel.py.

H. Adaptive step control

* step(...) estimates relative update size.
* If the step is too large, it subdivides into substeps.
* If too many substeps would be required, it terminates early.

This is why flow can stop for numerical reasons even before a true physical instability threshold is reached.

I. Flow measure lives here too

* FRGFlowSolverSZ0 stores patch_measure_mode, patch_measure_soft_vf_eps, and patch_measure_normalize_mean.
* These are passed to frg_kernel.py when building one-loop internal caches.

J. Built-in sign-aware diagnosis

* diagnose_current_state() runs a quick per-step channel diagnosis on selected Q values.
* This is lightweight compared with a full instability analysis.

3.6 channels.py: restoring physical pp/ph channels from the minimal vertex

This module is conceptually very important.

The minimal flow variable is not yet one of the physical channels. channels.py rebuilds the physical kernels at fixed Q.

A. Raw pp objects

* _pp_raw_v(Q)
* _pp_out_exchange_v(Q)

Then:

* pp_singlet = Vraw + Vex
* pp_triplet = Vraw - Vex

B. Raw ph objects

* ph_direct(Q) = V_d
* ph_exchange(Q) = V_x

Then:

* ph_charge = V_d - 2 V_x
* ph_spin   = V_d

These are the PRL-compatible formulas after reducing the flow variable to the minimal scalar object.

C. Q=0 ph_charge uniform-mode projection
The code can project out the uniform L=0 charge mode at Q=0:

K_reduced = P_perp K P_perp,
P_perp = I - |u><u|,
u = (1,1,...,1)/sqrt(N)

Why?

* ph_charge(Q=0) contains the forward-scattering / compressibility-like uniform mode.
* In diagnosis, this mode can dominate simply because it is a Landau parameter, not because it is the interesting ordered state one wants to identify.
* Therefore the code removes this mode at diagnosis stage by default, unless Landau_F=True.

Important:

* This projection changes diagnosis only.
* It does not change the RG flow itself.

D. Closure-aware reconstruction

* channels.py uses solver-provided closure_map and transfer_context.
* Therefore even bare sign-aware kernel scores depend not only on interaction.py, but also on how frg_flow.py defines closure and transfer indexing.

3.7 instability.py: physical instability diagnosis

This is the more physical diagnosis module.

A. Why raw channel kernels are not enough
A channel kernel by itself is not yet the full instability criterion. One also needs the bubble or phase-space weighting.

So instability.py builds a bubble-dressed operator from:

* the Hermitian channel kernel
* the appropriate diagonal bubble weights

B. Hermitian part

* The code first uses the Hermitian part of the channel kernel.
* This suppresses non-Hermitian numerical noise and makes the spectrum physically interpretable.

C. BubbleWeights
For each channel and Q, instability.py stores:

* channel type
* Q
* diagonal bubble weights
* partner patches
* residuals
* source description

D. ph bubble modes

* patchrep
* internal_cache

The default is patchrep, but internal_cache can be used to mirror the flow’s internal cache more directly.

E. Measure consistency with flow
The cleaned-up InstabilityConfig also contains:

* patch_measure_mode
* patch_measure_soft_vf_eps
* patch_measure_normalize_mean

This is critical. If the flow used non-unit patch measure, instability diagnosis must use the same measure; otherwise the diagnosis and the flow are analyzing different discretized problems.

F. Sign-aware logic and why it exists
There are two related but distinct ideas in this project:

1. sign-aware kernel score
   This is a quick diagnosis used on the raw channel kernel K(Q).
   The rule is:

   * pp: use the largest positive eigenvalue
   * ph: use the magnitude of the most negative eigenvalue
     after Hermitianization.

Why?

* In the current conventions, a large negative pp eigenvalue can represent strong local Cooper repulsion, not attractive pairing.
* In ph channels, the ordering-favored direction is encoded in the negative side of the Hermitian spectrum.

So sign-aware means:

* do not rank channels by max(|λ|)
* instead, use the sign that corresponds to the physically relevant instability direction for that channel type

This was introduced to avoid fake leaders, especially in pp singlet for large U.

2. physical instability operator
   instability.py goes further and constructs a bubble-dressed operator, then diagonalizes that.

So:

* sign-aware score = fast, kernel-level diagnosis
* physical instability score = more complete, bubble-dressed diagnosis

G. Why project out Q=0 ph_charge in instability diagnosis too
The same logic as in channels.py applies:

* Q=0 ph_charge includes a uniform compressibility-like mode
* It can dominate the diagnosis without representing the ordered pattern of interest
* Therefore instability.py can project out that uniform mode by default.

H. Additional pp-singlet local Gram projection
For Q=0 pp singlet, the code can also report or project out a local Gram-type basis related to trivial local pairing structures. This is meant as a diagnosis aid, not a change to the flow equation.

4. How the whole pipeline works in one sentence

5. noninteracting.py defines the band problem.

6. patching.py discretizes the chosen loop into gauge-fixed patch representatives and geometric weights.

7. interaction.py projects the bare extended Hubbard interaction into a PRL-compatible minimal S_z=0 vertex.

8. frg_kernel.py supplies all one-loop technical ingredients: Q handling, bubbles, partner maps, and minimal one-loop formulas.

9. frg_flow.py stores and evolves the minimal vertex under temperature-flow RG.

10. channels.py reconstructs physical pp/ph kernels from the flowed minimal vertex.

11. instability.py builds bubble-dressed operators and diagnoses the leading physical instability.

12. Practical interpretation guide

If a result looks strange, first ask which layer it belongs to.

* If the bare interaction itself looks wrong, inspect interaction.py and the Bloch-gauge and patch construction in patching.py.
* If different Q values that should be equivalent behave differently, inspect Q canonicalization, TransferGrid, and partner-map logic in frg_kernel.py and frg_flow.py.
* If flow strongly depends on patch geometry, inspect patch measure handling in patching.py and frg_kernel.py.
* If the same flowed vertex gives confusing channel rankings, compare sign-aware kernel score and physical instability score, and check whether Q=0 ph_charge uniform mode projection is active.
* If diagnosis disagrees with flow measure, check whether instability.py is using the same patch_measure settings as frg_flow.py.

6. Main design philosophy of the cleaned-up code

The cleaned-up codebase is built around four principles:

1. Store the right flow variable.
   Do not store many explicit spin blocks if the physical problem only has one independent minimal S_z=0 scalar object.

2. Canonicalize lattice momenta carefully.
   Q and Q+G must be identified, and small floating-point differences must not create fake distinct transfer channels.

3. Separate flow from diagnosis.
   The flow evolves one object. Physical channels and projections are reconstructed later for interpretation.

4. Keep numerical and physical semantics aligned.
   If the flow uses a certain patch measure, instability diagnosis must use the same measure. If a mode is projected out, it should be clear that this is a diagnosis-stage filter, not a change to the flow equation.

End of README.
