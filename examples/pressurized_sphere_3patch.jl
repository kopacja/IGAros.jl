# examples/pressurized_sphere_3patch.jl
#
# Internally pressurized thick sphere — 3-patch deltoidal icositetrahedron octant.
# Single shell (r_i → r_o), NO mortar interface.
#
# Purpose: isolate the geometric advantage of the deltoidal tiling (no pole
# singularity) from the mortar coupling.  Compare convergence rates against the
# classical single-patch pole-based parametrization.
#
# Total patches: 3 (one per tile, spanning the full radial thickness).

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─── Reuse exact solution, error functions, and tile data ────────────────────
include(joinpath(@__DIR__, "pressurized_sphere.jl"))
include(joinpath(@__DIR__, "pressurized_sphere_deltoidal.jl"))

# ─── Exact tile evaluation & degree-p approximation ─────────────────────────

"""
    eval_tile_surface(tile_homo, u, v) -> (x, y, z)

Evaluate the exact degree-4 rational Bézier tile at parameter (u,v) ∈ [0,1]².
`tile_homo` is 25×4 in homogeneous form [xw, yw, zw, w].
"""
function eval_tile_surface(tile_homo::Matrix{Float64}, u::Float64, v::Float64)
    s_u = 1.0 - u;  s_v = 1.0 - v
    Bu = [s_u^4, 4*s_u^3*u, 6*s_u^2*u^2, 4*s_u*u^3, u^4]
    Bv = [s_v^4, 4*s_v^3*v, 6*s_v^2*v^2, 4*s_v*v^3, v^4]

    xw = yw = zw = w = 0.0
    for j in 1:5
        for i in 1:5
            Nij = Bu[i] * Bv[j]
            k = (j-1)*5 + i
            xw += Nij * tile_homo[k, 1]
            yw += Nij * tile_homo[k, 2]
            zw += Nij * tile_homo[k, 3]
            w  += Nij * tile_homo[k, 4]
        end
    end
    return (xw/w, yw/w, zw/w)
end

"""
    approximate_tile_at_degree(tile_homo, p) -> Matrix{Float64}

Create a degree-p polynomial (weight=1) Bézier surface approximating the exact
degree-4 rational tile.  Uses Greville-point interpolation: evaluates the exact
surface at (p+1)² Greville abscissae and inverts the Bernstein basis to find CPs.

Returns ((p+1)² × 4) with ξ-inner, η-outer ordering.
"""
function approximate_tile_at_degree(tile_homo::Matrix{Float64}, p::Int)
    n = p + 1

    # Greville abscissae for degree-p Bezier: ξ_i = i/p
    greville = Float64[i/p for i in 0:p]

    # Bernstein basis matrix: B[row, col] = B^p_{col-1}(greville[row])
    Bmat = zeros(n, n)
    for (row, t) in enumerate(greville)
        s = 1.0 - t
        for k in 0:p
            Bmat[row, k+1] = binomial(p, k) * s^(p-k) * t^k
        end
    end
    Bmat_inv = inv(Bmat)

    # Evaluate exact surface at Greville grid → points on the sphere
    points = zeros(n*n, 3)
    for j in 1:n
        for i in 1:n
            k = (j-1)*n + i
            x, y, z = eval_tile_surface(tile_homo, greville[i], greville[j])
            points[k, :] = [x, y, z]
        end
    end

    # Tensor-product interpolation: CPs = (Bmat_inv ⊗ Bmat_inv) * points
    # Step 1: solve in ξ for each η-row
    temp = zeros(n*n, 3)
    for j in 1:n
        temp[(j-1)*n+1 : j*n, :] = Bmat_inv * points[(j-1)*n+1 : j*n, :]
    end
    # Step 2: solve in η for each ξ-column
    cps = zeros(n*n, 3)
    for i in 1:n
        col = temp[i : n : end, :]   # n×3
        result = Bmat_inv * col
        for j in 1:n
            cps[(j-1)*n + i, :] = result[j, :]
        end
    end

    out = zeros(n*n, 4)
    out[:, 1:3] = cps
    out[:, 4] .= 1.0
    return out
end

# ─── 3-patch geometry builder ────────────────────────────────────────────────

"""
    sphere_geometry_3patch(p_ord; r_i, r_o) -> (B, P)

Build a 3-patch solid sphere octant using deltoidal icositetrahedron tiles.
Each tile spans the full radial thickness from r_i to r_o.

For p ≥ 4: uses Dedoncker's exact biquartic rational Bézier CPs (exact sphere).
For p < 4: Greville-interpolated degree-p polynomial CPs (approximate sphere,
           geometry error O(h^{p+1}) subordinate to solution error O(h^p)).
"""
function sphere_geometry_3patch(p_ord::Int;
                                 r_i::Float64 = 1.0,
                                 r_o::Float64 = 1.2)

    tile_rotations = [ROT_TILE_A, ROT_TILE_B, ROT_TILE_C]

    p_surf = 4   # Dedoncker tile native degree
    p_use  = p_ord

    n_surf_per_dir = p_use + 1
    n_rad          = p_use + 1
    n_surf_cp      = n_surf_per_dir^2
    ncp_per_patch  = n_surf_cp * n_rad

    npc = 3
    B_all = zeros(npc * ncp_per_patch, 4)
    P     = Vector{Vector{Int}}(undef, npc)

    for (t, R) in enumerate(tile_rotations)
        if p_use >= p_surf
            # Exact rational tile (p=4), optionally degree-elevated
            tile_cp = rotate_tile(TILE1_CP, R)
            if p_use > p_surf
                tile_cp = elevate_tile_surface(tile_cp, p_use)
            end
        else
            # Approximate polynomial tile at degree p
            tile_homo_rot = rotate_tile(_TILE1_HOMO, R)
            tile_cp = approximate_tile_at_degree(tile_homo_rot, p_use)
        end

        # Reverse ξ direction so that (ξ×η) normal points outward → positive detJ
        nsd_dir = n_surf_per_dir
        tile_flipped = similar(tile_cp)
        for j in 1:nsd_dir
            for i in 1:nsd_dir
                tile_flipped[(j-1)*nsd_dir + (nsd_dir - i + 1), :] =
                    tile_cp[(j-1)*nsd_dir + i, :]
            end
        end

        surf_ri = scale_tile_to_radius(tile_flipped, r_i)
        surf_ro = scale_tile_to_radius(tile_flipped, r_o)

        B_solid = zeros(ncp_per_patch, 4)
        for k in 0:n_rad-1
            frac = (n_rad > 1) ? k / (n_rad - 1) : 0.0
            B_solid[k*n_surf_cp+1 : (k+1)*n_surf_cp, :] =
                (1 - frac) .* surf_ri .+ frac .* surf_ro
        end
        offset = (t - 1) * ncp_per_patch
        B_all[offset+1 : offset+ncp_per_patch, :] = B_solid
        P[t] = collect(offset+1 : offset+ncp_per_patch)
    end

    return B_all, P
end

# ─── Post-refinement CP projection for p < 4 ───────────────────────────────
#
# After h-refinement the NURBS geometry is unchanged (knot insertion preserves
# the surface).  For p < 4 the initial Bézier surface is only an approximate
# sphere, and this approximation does NOT improve with h-refinement.
#
# Fix: replace each CP with the Greville quasi-interpolant of the exact
# Dedoncker surface → geometry error becomes O(h^{p+1}).

"""
    project_cps_to_sphere!(B, P, KV, n_mat, p_mat, r_i, r_o)

For each of the 3 deltoidal patches (p < 4 only), replace CPs with values
of the exact tile surface evaluated at the CP's Greville abscissae, scaled
to the appropriate radius.  Weights are set to 1.0 (polynomial surface).
"""
function project_cps_to_sphere!(
    B::Matrix{Float64}, P::Vector{Vector{Int}},
    KV::Vector{<:Vector{<:Vector{Float64}}},
    n_mat::Matrix{Int}, p_mat::Matrix{Int},
    r_i::Float64, r_o::Float64
)
    tile_rotations = [ROT_TILE_A, ROT_TILE_B, ROT_TILE_C]

    for (t, R) in enumerate(tile_rotations)
        pc = t
        tile_homo = rotate_tile(_TILE1_HOMO, R)

        n1 = n_mat[pc, 1]; n2 = n_mat[pc, 2]; n3 = n_mat[pc, 3]
        p1 = p_mat[pc, 1]; p2 = p_mat[pc, 2]; p3 = p_mat[pc, 3]

        # Greville abscissae from the refined knot vectors
        function _greville(kv, n, p)
            ga = zeros(n)
            for i in 1:n
                ga[i] = sum(kv[i+1 : i+p]) / p
            end
            return ga
        end
        ga_xi   = _greville(KV[pc][1], n1, p1)
        ga_eta  = _greville(KV[pc][2], n2, p2)
        ga_zeta = _greville(KV[pc][3], n3, p3)

        for k in 1:n3
            r = r_i + ga_zeta[k] * (r_o - r_i)   # linear radius mapping
            for j in 1:n2
                for i in 1:n1
                    # ξ was flipped in geometry builder → undo for tile eval
                    xi_tile  = 1.0 - ga_xi[i]
                    eta_tile = ga_eta[j]
                    x, y, z  = eval_tile_surface(tile_homo, xi_tile, eta_tile)

                    local_idx  = (k-1)*n1*n2 + (j-1)*n1 + i
                    global_idx = P[pc][local_idx]
                    B[global_idx, 1] = r * x
                    B[global_idx, 2] = r * y
                    B[global_idx, 3] = r * z
                    B[global_idx, 4] = 1.0
                end
            end
        end
    end
end

# ─── Solver ──────────────────────────────────────────────────────────────────

"""
    solve_sphere_3patch(p_ord, exp_level; ...) -> NamedTuple

Solve pressurized sphere with 3-patch deltoidal geometry (no mortar).
Neumann traction on face 1 (ζ=1, inner surface r_i).
"""
function solve_sphere_3patch(
    p_ord::Int,
    exp_level::Int;
    r_i::Float64  = 1.0,
    r_o::Float64  = 2.0,
    E::Float64    = 1.0,
    nu::Float64   = 0.3,
    p_i::Float64  = 0.01,
    NQUAD::Int    = p_ord + 1,
    n_base::Int   = 4,
    vtk_prefix::String = "",
    n_vis::Int    = 4
)
    nsd = 3; npd = 3; ned = 3; npc = 3

    # ── Geometry ──────────────────────────────────────────────────────────
    B0, P = sphere_geometry_3patch(p_ord; r_i=r_i, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_ang = p_ord + 1
    n_rad = p_ord + 1
    n_mat = fill(0, npc, npd)
    for pc in 1:npc
        n_mat[pc, :] = [n_ang, n_ang, n_rad]
    end

    KV = [[vcat(zeros(p_ord+1), ones(p_ord+1)) for _ in 1:3] for _ in 1:npc]

    # ── h-refinement ──────────────────────────────────────────────────────
    n_elem     = n_base * 2^exp_level
    n_rad_elem = 2^exp_level

    u_surf = Float64[i/n_elem for i in 1:n_elem-1]
    u_rad  = Float64[i/n_rad_elem for i in 1:n_rad_elem-1]

    kref_data = Vector{Float64}[]
    for t in 1:npc
        push!(kref_data, vcat([Float64(t), 1.0], u_surf))
        push!(kref_data, vcat([Float64(t), 2.0], u_surf))
        push!(kref_data, vcat([Float64(t), 3.0], u_rad))
    end

    n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data
    )
    ncp = size(B_ref, 1)

    # ── Project CPs onto exact sphere (p < 4 only) ──────────────────────
    if p_ord < 4
        project_cps_to_sphere!(B_ref, P_ref, KV_ref, n_mat_ref, p_mat, r_i, r_o)
    end

    # ── Connectivity ──────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs ────────────────────────────────────────────────────
    dBC = deltoidal_symmetry_bcs(B_ref, P_ref, ned)
    neq, ID = build_id(dBC, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ─────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :three_d) for _ in 1:npc]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # ── Load: internal pressure on face 1 (ζ=1, r=r_i) of all patches ──
    stress_fn = (x, y, z) -> lame_stress_sphere(x, y, z; p_i=p_i, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    for pc in 1:npc
        F = segment_load(n_mat_ref[pc,:], p_mat[pc,:], KV_ref[pc], P_ref[pc], B_ref,
                         nnp[pc], nen[pc], nsd, npd, ned,
                         Int[], 1, ID, F, stress_fn, 1.0, NQUAD)
    end

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Solve (direct, no mortar) ─────────────────────────────────────────
    U = K_bc \ F_bc

    # ── L2 stress error ───────────────────────────────────────────────────
    σ_abs, σ_ref = l2_stress_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── L2 displacement error ─────────────────────────────────────────────
    disp_fn = (x, y, z) -> lame_displacement_sphere(x, y, z;
        p_i=p_i, r_i=r_i, r_o=r_o, E=E, nu=nu)
    l2_abs, l2_ref = l2_disp_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )

    # ── Energy-norm error ─────────────────────────────────────────────────
    en_abs, en_ref = energy_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── Optional VTK export ───────────────────────────────────────────────
    if !isempty(vtk_prefix)
        write_vtk_sphere(vtk_prefix, U, ID, npc, nsd, npd,
                          p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                          nen, IEN, INC, E, nu;
                          n_vis=n_vis, p_i=p_i, r_i=r_i, r_o=r_o)
    end

    return (σ_rel=σ_abs/σ_ref, σ_abs=σ_abs,
            l2_rel=l2_abs/l2_ref, l2_abs=l2_abs,
            en_rel=en_abs/en_ref, en_abs=en_abs)
end

# ─── Mortar-coupled solver ───────────────────────────────────────────────────

"""
    solve_sphere_3patch_mortar(p_ord, exp_level; ...) -> NamedTuple

Solve pressurized sphere with 3-patch deltoidal geometry and Twin Mortar
coupling at the 3 inter-patch interfaces.  Non-conforming meshes: patches
1,2,3 can have different surface refinement levels.
"""
function solve_sphere_3patch_mortar(
    p_ord::Int,
    exp_level::Int;
    r_i::Float64  = 1.0,
    r_o::Float64  = 1.2,
    E::Float64    = 1000.0,
    nu::Float64   = 0.3,
    p_i::Float64  = 1.0,
    epss::Float64 = 0.0,
    NQUAD::Int    = p_ord + 1,
    NQUAD_mortar::Int = p_ord + 2,
    n_base::Int   = 2,
    conforming::Bool = false,
    mesh_ratio::Float64 = 2.0,
    strategy::IntegrationStrategy  = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    vtk_prefix::String = "",
    n_vis::Int    = 4
)
    nsd = 3; npd = 3; ned = 3; npc = 3

    # ── Geometry ──────────────────────────────────────────────────────────
    B0, P = sphere_geometry_3patch(p_ord; r_i=r_i, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_ang = p_ord + 1
    n_rad = p_ord + 1
    n_mat = fill(0, npc, npd)
    for pc in 1:npc
        n_mat[pc, :] = [n_ang, n_ang, n_rad]
    end
    KV = [[vcat(zeros(p_ord+1), ones(p_ord+1)) for _ in 1:3] for _ in 1:npc]

    # ── h-refinement ────────────────────────────────────────────────────
    # Non-conforming strategy: for each patch, refine ξ and η differently.
    # Interface faces pair F3 (free: ξ,ζ) ↔ F4 (free: η,ζ), so having
    # n_ξ ≠ n_η in every patch makes ALL 3 interfaces non-conforming
    # with ratio n_ξ : n_η, while keeping the same total mesh per patch.
    n_elem_xi  = n_base * 2^exp_level
    n_elem_eta = conforming ? n_elem_xi : round(Int, mesh_ratio * n_elem_xi)
    n_rad_elem = 2^exp_level

    u_xi  = Float64[i/n_elem_xi  for i in 1:n_elem_xi-1]
    u_eta = Float64[i/n_elem_eta for i in 1:n_elem_eta-1]
    u_rad = Float64[i/n_rad_elem for i in 1:n_rad_elem-1]

    kref_data = Vector{Float64}[]
    for t in 1:npc
        push!(kref_data, vcat([Float64(t), 1.0], u_xi))    # ξ: coarse
        push!(kref_data, vcat([Float64(t), 2.0], u_eta))   # η: fine
        push!(kref_data, vcat([Float64(t), 3.0], u_rad))   # ζ: radial
    end

    n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data
    )
    ncp = size(B_ref, 1)

    # ── Project CPs onto exact sphere (p < 4 only) ──────────────────────
    if p_ord < 4
        project_cps_to_sphere!(B_ref, P_ref, KV_ref, n_mat_ref, p_mat, r_i, r_o)
    end

    epss_use = epss > 0.0 ? epss : 100.0

    # ── Connectivity ──────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs ────────────────────────────────────────────────────
    dBC = deltoidal_symmetry_bcs(B_ref, P_ref, ned)
    neq, ID = build_id(dBC, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ─────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :three_d) for _ in 1:npc]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # ── Load: internal pressure on face 1 (ζ=1, r=r_i) of all patches ──
    stress_fn = (x, y, z) -> lame_stress_sphere(x, y, z; p_i=p_i, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    for pc in 1:npc
        F = segment_load(n_mat_ref[pc,:], p_mat[pc,:], KV_ref[pc], P_ref[pc], B_ref,
                         nnp[pc], nen[pc], nsd, npd, ned,
                         Int[], 1, ID, F, stress_fn, 1.0, NQUAD)
    end
    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling: 3 inter-patch interfaces ─────────────────────────
    # Patch 1 face 4 ↔ Patch 2 face 3
    # Patch 1 face 3 ↔ Patch 3 face 4
    # Patch 2 face 4 ↔ Patch 3 face 3
    pairs = InterfacePair[]
    for (s_pc, s_face, m_pc, m_face) in [
        (1, 4, 2, 3), (1, 3, 3, 4), (2, 4, 3, 3)
    ]
        push!(pairs, InterfacePair(s_pc, s_face, m_pc, m_face))
        if !(formulation isa SinglePassFormulation)
            push!(pairs, InterfacePair(m_pc, m_face, s_pc, s_face))
        end
    end

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    # ── Solve ─────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── Error measures ────────────────────────────────────────────────────
    σ_abs, σ_ref = l2_stress_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )
    disp_fn = (x, y, z) -> lame_displacement_sphere(x, y, z;
        p_i=p_i, r_i=r_i, r_o=r_o, E=E, nu=nu)
    l2_abs, l2_ref = l2_disp_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )
    en_abs, en_ref = energy_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    if !isempty(vtk_prefix)
        write_vtk_sphere(vtk_prefix, U, ID, npc, nsd, npd,
                          p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                          nen, IEN, INC, E, nu;
                          n_vis=n_vis, p_i=p_i, r_i=r_i, r_o=r_o)
    end

    return (σ_rel=σ_abs/σ_ref, σ_abs=σ_abs,
            l2_rel=l2_abs/l2_ref, l2_abs=l2_abs,
            en_rel=en_abs/en_ref, en_abs=en_abs)
end

# ─── Mortar convergence study driver ──────────────────────────────────────────

function run_convergence_3patch_mortar(;
    p_range   = 1:4,
    exp_range = 0:3,
    n_base::Int = 2,
    E::Float64  = 1000.0,
    nu::Float64 = 0.3,
    p_i::Float64 = 1.0,
    epss::Float64 = 100.0,
    r_i::Float64 = 1.0,
    r_o::Float64 = 1.2,
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    kwargs...
)
    tag = formulation isa TwinMortarFormulation ? "TM" :
          formulation isa DualPassMortarFormulation ? "DPM" : "SP"
    for p in p_range
        @printf("\n=== p=%d  3-patch mortar sphere (%s, ε=%.0e, n_base=%d) ===\n",
                p, tag, epss, n_base)
        prev_l2 = prev_en = 0.0
        for exp in exp_range
            @printf("  computing exp=%d ...", exp); flush(stdout)
            res = solve_sphere_3patch_mortar(p, exp; E=E, nu=nu, p_i=p_i, epss=epss,
                                              r_i=r_i, r_o=r_o, n_base=n_base,
                                              formulation=formulation, strategy=strategy,
                                              kwargs...)
            rate_l2 = exp > first(exp_range) ? log2(prev_l2 / res.l2_rel) : NaN
            rate_en = exp > first(exp_range) ? log2(prev_en / res.en_rel) : NaN
            @printf("\r  exp=%d  σ=%.4e  l2=%.4e  en=%.4e  rate_l2=%5.2f  rate_en=%5.2f\n",
                    exp, res.σ_rel, res.l2_rel, res.en_rel, rate_l2, rate_en)
            flush(stdout)
            prev_l2 = res.l2_rel; prev_en = res.en_rel
        end
    end
end

# ─── Convergence study driver ─────────────────────────────────────────────────

function run_convergence_3patch(;
    p_range  = 1:4,
    exp_range = 0:3,
    n_base::Int = 4,
    E::Float64  = 1.0,
    nu::Float64 = 0.3,
    p_i::Float64 = 0.01,
    r_i::Float64 = 1.0,
    r_o::Float64 = 2.0,
    kwargs...
)
    for p in p_range
        @printf("\n=== p=%d  3-patch deltoidal sphere (no mortar, n_base=%d) ===\n", p, n_base)
        prev_l2 = prev_en = prev_σ = 0.0
        for exp in exp_range
            @printf("  computing exp=%d ...", exp); flush(stdout)
            res = solve_sphere_3patch(p, exp; E=E, nu=nu, p_i=p_i,
                                       r_i=r_i, r_o=r_o, n_base=n_base, kwargs...)
            rate_l2 = exp > first(exp_range) ? log2(prev_l2 / res.l2_rel) : NaN
            rate_en = exp > first(exp_range) ? log2(prev_en / res.en_rel) : NaN
            @printf("\r  exp=%d  σ=%.4e  l2=%.4e  en=%.4e  rate_l2=%5.2f  rate_en=%5.2f\n",
                    exp, res.σ_rel, res.l2_rel, res.en_rel, rate_l2, rate_en)
            flush(stdout)
            prev_l2 = res.l2_rel; prev_en = res.en_rel; prev_σ = res.σ_rel
        end
    end
end
