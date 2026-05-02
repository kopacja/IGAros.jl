# examples/pressurized_sphere_deltoidal.jl
#
# Internally pressurized thick sphere — 3-patch deltoidal icositetrahedron octant.
#
# Uses three kite-shaped Bézier tiles per octant (Dedoncker et al., 2018,
# "Bézier tilings of the sphere and their applications in benchmarking
# multipatch isogeometric methods"), eliminating the degenerate pole singularity
# of the classical single-patch parametrization.
#
# Each octant is covered by 3 biquadratic rational Bézier patches on the unit
# sphere, generated from a reference tile (Table A.9) by rotations (Table A.10).
# The solid is formed by radial extrusion; the mortar interface is at r = r_c.
#
# Total patches: 6 (3 inner + 3 outer).
# Mortar interface pairs: 3 (one per tile, inner ↔ outer).

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─── Reuse exact solution and error functions from the pole-based sphere ──────
include(joinpath(@__DIR__, "pressurized_sphere.jl"))

# ─── Reference tile (Table A.9, Dedoncker et al. 2018) ───────────────────────
# Biquadratic rational Bézier surface on the unit sphere.
# 5×5 control points in (x, y, z, w).  Index (i,j) with i=ξ (fast), j=η (slow).
# Tile 1 covers a kite from the south pole (0,0,-1) through two cuboctahedron
# edges to the cube vertex (1/√3, 1/√3, -1/√3).

# Raw Table A.9 data in HOMOGENEOUS form [xw, yw, zw, w].
# IGAros stores Euclidean [x, y, z, w], so we convert below.
const _TILE1_HOMO = Float64[
#   xw                   yw                   zw                   w
    0.0                  0.0                 -1.0                  1.0           ;  # (1,1)
    0.207106781186548    0.0                 -1.0                  1.0           ;  # (2,1)
    0.414213562373095    0.0                 -0.971404520791032    1.02859547920897;  # (3,1)
    0.621320343559643    0.0                 -0.914213562373095    1.0857864376269 ;  # (4,1)
    0.82842712474619     0.0                 -0.82842712474619     1.17157287525381;  # (5,1)

    0.0                  0.207106781186548   -1.0                  1.0           ;  # (1,2)
    0.20246636350282     0.20246636350282    -0.998930607671726    0.998930607671726;  # (2,2)
    0.407140425183055    0.197167079290905   -0.969834223875419    1.02446235037379;  # (3,2)
    0.613579228234834    0.191208928550803   -0.911155682104475    1.07815039461279;  # (4,2)
    0.821339815852291    0.184591911282515   -0.821339815852291    1.16154990689533;  # (5,2)

    0.0                  0.414213562373095   -0.971404520791032    1.02859547920897;  # (1,3)
    0.197167079290905    0.407140425183055   -0.969834223875419    1.02446235037379;  # (2,3)
    0.396726193243811    0.396726193243811   -0.942520631201463    1.04607658343011;  # (3,3)
    0.598682394601221    0.38300296952756    -0.887573768782017    1.09534035076458;  # (4,3)
    0.802955081964821    0.366025403784439   -0.802955081964822    1.17432880222789;  # (5,3)

    0.0                  0.621320343559643   -0.914213562373095    1.0857864376269 ;  # (1,4)
    0.191208928550803    0.613579228234834   -0.911155682104475    1.07815039461279;  # (2,4)
    0.38300296952756     0.598682394601221   -0.887573768782017    1.09534035076458;  # (3,4)
    0.576618885956231    0.576618885956231   -0.84219639933622     1.13862772915177;  # (4,4)
    0.773563913640501    0.547485916996871   -0.773563913640501    1.20932580910712;  # (5,4)

    0.0                  0.82842712474619    -0.82842712474619     1.17157287525381;  # (1,5)
    0.184591911282515    0.821339815852291   -0.821339815852291    1.16154990689533;  # (2,5)
    0.366025403784439    0.802955081964821   -0.802955081964822    1.17432880222789;  # (3,5)
    0.547485916996871    0.773563913640501   -0.773563913640501    1.20932580910712;  # (4,5)
    0.732050807568877    0.732050807568877   -0.732050807568877    1.26794919243112;  # (5,5)
]

# Convert homogeneous → Euclidean: divide xyz by w
const TILE1_CP = let cp = copy(_TILE1_HOMO)
    for i in 1:size(cp, 1)
        cp[i, 1:3] ./= cp[i, 4]
    end
    cp
end

# ─── Rotation matrices for the three first-octant tiles ───────────────────────
# Tile  5 = R_x(π/2)           * Tile 1 → octahedron vertex at (0,1,0)
# Tile 21 = R_y(-π/2)          * Tile 1 → octahedron vertex at (1,0,0)
# Tile 12 = R_x(π)·R_z(-π/2)  * Tile 1 → octahedron vertex at (0,0,1)

const ROT_TILE_A = Float64[1 0 0; 0 0 -1; 0 1 0]   # R_x(π/2):  (x,y,z)→(x,-z,y)
const ROT_TILE_B = Float64[0 0 -1; 0 1 0; 1 0 0]    # R_y(-π/2): (x,y,z)→(-z,y,x)
const ROT_TILE_C = Float64[0 1 0; 1 0 0; 0 0 -1]    # R_x(π)R_z(-π/2): (x,y,z)→(y,x,-z)

"""
    rotate_tile(tile_cp, R) -> Matrix{Float64}

Apply 3×3 rotation R to tile control points (n×4, columns = x,y,z,w).
Weights are preserved; only xyz are rotated.
"""
function rotate_tile(tile_cp::Matrix{Float64}, R::Matrix{Float64})::Matrix{Float64}
    out = copy(tile_cp)
    for i in 1:size(out, 1)
        out[i, 1:3] = R * tile_cp[i, 1:3]
    end
    return out
end

"""
    scale_tile_to_radius(tile_cp, r) -> Matrix{Float64}

Scale unit-sphere tile CPs to radius r.  Physical point = (x/w, y/w, z/w);
scaling by r gives (r·x/w, r·y/w, r·z/w) ≡ CPs (r·x, r·y, r·z, w).
"""
function scale_tile_to_radius(tile_cp::Matrix{Float64}, r::Float64)::Matrix{Float64}
    out = copy(tile_cp)
    out[:, 1:3] .*= r
    return out
end

# ─── Degree elevation for Bézier curves (1D, in homogeneous space) ────────────
# Reuse bezier_elevate_3d from pressurized_sphere.jl (already included).

"""
    elevate_tile_surface(tile_cp, p_target) -> Matrix{Float64}

Order-elevate a 5×5 (p=2) biquadratic Bézier surface tile to (p_target × p_target).
Returns ((p_target+1)² × 4) array with ξ-inner, η-outer ordering.
"""
function elevate_tile_surface(tile_cp::Matrix{Float64}, p_target::Int)::Matrix{Float64}
    @assert size(tile_cp, 1) == 25  "Expected 5×5 = 25 CPs for biquadratic tile"
    n_src = 5   # p=2 → 5 CPs per direction (Bézier: n = p+1 = 3, but wait...)

    # The tile from Table A.9 has 5×5 CPs → it is degree (4,4), not (2,2)!
    # Each direction has 5 CPs → polynomial degree 4 in each direction.
    p_src = 4

    if p_target <= p_src
        return copy(tile_cp)
    end

    n_tgt = p_target + 1

    # Step 1: elevate in ξ direction (for each η row of n_src CPs)
    n_xi_cur = n_src
    B_xi = copy(tile_cp)
    for _ in p_src+1:p_target
        n_xi_new = n_xi_cur + 1
        B_new = zeros(n_xi_new * n_src, 4)
        for j in 1:n_src
            row = B_xi[(j-1)*n_xi_cur+1 : j*n_xi_cur, :]
            Bh = copy(row)
            Bh[:, 1:3] .*= Bh[:, 4:4]  # homogeneous
            Bh = bezier_elevate_3d(Bh)
            Bh[:, 1:3] ./= Bh[:, 4:4]  # back to Euclidean
            B_new[(j-1)*n_xi_new+1 : j*n_xi_new, :] = Bh
        end
        B_xi = B_new
        n_xi_cur = n_xi_new
    end

    # Step 2: elevate in η direction (for each ξ column)
    n_eta_cur = n_src
    for _ in p_src+1:p_target
        n_eta_new = n_eta_cur + 1
        B_new = zeros(n_xi_cur * n_eta_new, 4)
        for i in 1:n_xi_cur
            col = B_xi[i : n_xi_cur : (n_eta_cur-1)*n_xi_cur+i, :]
            Bh = copy(col)
            Bh[:, 1:3] .*= Bh[:, 4:4]
            Bh = bezier_elevate_3d(Bh)
            Bh[:, 1:3] ./= Bh[:, 4:4]
            for j in 1:n_eta_new
                B_new[(j-1)*n_xi_cur + i, :] = Bh[j, :]
            end
        end
        B_xi = B_new
        n_eta_cur = n_eta_new
    end

    return B_xi  # (n_tgt² × 4)
end

# ─── Geometry builder ─────────────────────────────────────────────────────────

"""
    sphere_geometry_deltoidal(p_ord; r_i, r_c, r_o) -> (B, P)

Build a 6-patch solid sphere octant using the deltoidal icositetrahedron tiling.

  Patches 1–3: inner shell (r_i → r_c), tiles A, B, C
  Patches 4–6: outer shell (r_c → r_o), tiles A, B, C

Each tile is a biquadratic (degree 4) rational Bézier surface on the unit sphere,
elevated to p_ord if needed, then extruded radially with p_ord CPs.

Returns B (ncp_total × 4) and P (6-element vector of CP index vectors).
"""
function sphere_geometry_deltoidal(p_ord::Int;
                                    r_i::Float64 = 1.0,
                                    r_c::Float64 = 1.2,
                                    r_o::Float64 = 1.4)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    # Three first-octant tiles (rotations of Tile 1)
    tile_rotations = [ROT_TILE_A, ROT_TILE_B, ROT_TILE_C]

    p_surf = 4   # The tile from Table A.9 is degree 4 in each surface direction
    p_use  = max(p_ord, p_surf)   # Can't go below the geometric degree

    n_surf_per_dir = p_use + 1   # CPs per surface direction
    n_rad          = p_use + 1   # CPs in radial direction
    n_surf_cp      = n_surf_per_dir^2
    ncp_per_patch  = n_surf_cp * n_rad

    npc = 6
    B_all = zeros(npc * ncp_per_patch, 4)
    P     = Vector{Vector{Int}}(undef, npc)

    for (t, R) in enumerate(tile_rotations)
        # Rotate tile to first octant
        tile_cp = rotate_tile(TILE1_CP, R)

        # Elevate surface to p_use if needed
        if p_use > p_surf
            tile_cp = elevate_tile_surface(tile_cp, p_use)
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

        # Scale to inner and outer radii
        surf_ri = scale_tile_to_radius(tile_flipped, r_i)
        surf_rc = scale_tile_to_radius(tile_flipped, r_c)
        surf_ro = scale_tile_to_radius(tile_flipped, r_o)

        # Build solid patches by linear blend in radial direction
        for (layer, surf_a, surf_b, pc_idx) in [
            (1, surf_ri, surf_rc, t),       # inner shell: patches 1,2,3
            (2, surf_rc, surf_ro, t + 3),   # outer shell: patches 4,5,6
        ]
            B_solid = zeros(ncp_per_patch, 4)
            for k in 0:n_rad-1
                frac = (n_rad > 1) ? k / (n_rad - 1) : 0.0
                B_solid[k*n_surf_cp+1 : (k+1)*n_surf_cp, :] =
                    (1 - frac) .* surf_a .+ frac .* surf_b
            end
            offset = (pc_idx - 1) * ncp_per_patch
            B_all[offset+1 : offset+ncp_per_patch, :] = B_solid
            P[pc_idx] = collect(offset+1 : offset+ncp_per_patch)
        end
    end

    return B_all, P
end

# ─── Boundary condition helpers ───────────────────────────────────────────────

"""
    deltoidal_symmetry_bcs(B, P, ned) -> Vector{Vector{Int}}

For the 3-patch deltoidal octant, symmetry BCs are:
  ux=0 on x=0 plane, uy=0 on y=0 plane, uz=0 on z=0 plane.

Returns `dBC[d]` = list of CP global indices constrained in direction d,
compatible with `build_id`.
"""
function deltoidal_symmetry_bcs(
    B::Matrix{Float64}, P::Vector{Vector{Int}},
    ned::Int;
    tol::Float64 = 1e-12
)::Vector{Vector{Int}}

    dBC = [Int[] for _ in 1:ned]

    seen = Set{Int}()
    for pc in 1:length(P)
        for A in P[pc]
            A in seen && continue
            push!(seen, A)
            x, y, z = B[A, 1], B[A, 2], B[A, 3]
            if abs(x) < tol   # x=0 plane → ux=0
                push!(dBC[1], A)
            end
            if abs(y) < tol   # y=0 plane → uy=0
                push!(dBC[2], A)
            end
            if abs(z) < tol   # z=0 plane → uz=0
                push!(dBC[3], A)
            end
        end
    end

    return dBC
end

# ─── Facet identification for deltoidal patches ──────────────────────────────
# In a 3D NURBS solid with parametric directions (ξ, η, ζ):
#   Face 1: ζ=1  (inner radial surface)
#   Face 6: ζ=n₃ (outer radial surface)
# For the deltoidal shell:
#   Inner patch face 6 (ζ=n₃, at r_c) ↔ Outer patch face 1 (ζ=1, at r_c)
#   Inner patch face 1 (ζ=1, at r_i) is the load surface.

# ─── Solver ───────────────────────────────────────────────────────────────────

"""
    solve_sphere_deltoidal(p_ord, exp_level; ...) -> NamedTuple

Solve the pressurized sphere benchmark using the 6-patch deltoidal geometry.
Mortar coupling at r=r_c between each inner-outer tile pair (3 interface pairs).
"""
function solve_sphere_deltoidal(
    p_ord::Int,
    exp_level::Int;
    conforming::Bool           = false,
    mesh_ratio::Float64        = 2.0,
    r_i::Float64               = 1.0,
    r_c::Float64               = 1.2,
    r_o::Float64               = 1.4,
    E::Float64                 = 1.0,
    nu::Float64                = 0.3,
    p_i::Float64               = 0.01,
    epss::Float64              = 0.0,
    NQUAD::Int                 = p_ord + 1,
    NQUAD_mortar::Int          = p_ord + 2,
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    vtk_prefix::String         = "",
    n_vis::Int                 = 4
)
    nsd = 3; npd = 3; ned = 3; npc = 6

    # ── Geometry ────────────────────────────────────────────────────────────
    B0, P = sphere_geometry_deltoidal(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_geom = max(p_ord, 4)   # geometric degree (tile is degree 4)
    p_mat = fill(p_geom, npc, npd)
    n_ang = p_geom + 1
    n_rad = p_geom + 1
    n_mat = fill(0, npc, npd)
    for pc in 1:npc
        n_mat[pc, :] = [n_ang, n_ang, n_rad]
    end

    # Knot vectors: initially Bézier (open, no interior knots)
    function bezier_kv(p)
        vcat(zeros(p+1), ones(p+1))
    end
    KV = [[bezier_kv(p_geom), bezier_kv(p_geom), bezier_kv(p_geom)] for _ in 1:npc]

    # ── h-refinement ────────────────────────────────────────────────────────
    n_elem = 2^(exp_level + 2)  # elements per surface direction (4 at exp=0)
    n_rad_elem = 2^exp_level     # radial elements (1 at exp=0)

    # For non-conforming: inner patches get mesh_ratio× more surface elements
    n_elem_inner = conforming ? n_elem : round(Int, mesh_ratio * n_elem)

    u_surf_inner = Float64[i/n_elem_inner for i in 1:n_elem_inner-1]
    u_surf_outer = Float64[i/n_elem for i in 1:n_elem-1]
    u_rad        = Float64[i/n_rad_elem for i in 1:n_rad_elem-1]

    kref_data = Vector{Float64}[]
    for t in 1:3
        # Inner patches (1,2,3): finer surface mesh
        push!(kref_data, vcat([Float64(t), 1.0], u_surf_inner))  # ξ
        push!(kref_data, vcat([Float64(t), 2.0], u_surf_inner))  # η
        push!(kref_data, vcat([Float64(t), 3.0], u_rad))          # ζ
    end
    for t in 1:3
        # Outer patches (4,5,6): coarser surface mesh
        push!(kref_data, vcat([Float64(t+3), 1.0], u_surf_outer))  # ξ
        push!(kref_data, vcat([Float64(t+3), 2.0], u_surf_outer))  # η
        push!(kref_data, vcat([Float64(t+3), 3.0], u_rad))          # ζ
    end

    n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data
    )
    ncp = size(B_ref, 1)
    epss_use = epss > 0.0 ? epss : 100.0

    # ── Connectivity ────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs (coordinate-plane symmetry) ──────────────────────────
    dBC = deltoidal_symmetry_bcs(B_ref, P_ref, ned)
    neq, ID = build_id(dBC, ned, ncp)

    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :three_d) for _ in 1:npc]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # ── Load: internal pressure on inner patches face 1 (ζ=1, r=r_i) ─────
    stress_fn = (x, y, z) -> lame_stress_sphere(x, y, z; p_i=p_i, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    for pc in 1:3   # patches 1,2,3 are inner shell
        F = segment_load(n_mat_ref[pc,:], p_mat[pc,:], KV_ref[pc], P_ref[pc], B_ref,
                         nnp[pc], nen[pc], nsd, npd, ned,
                         Int[], 1, ID, F, stress_fn, 1.0, NQUAD)
    end

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling at r = r_c: 3 interface pairs ─────────────────────
    # Inner patch t face 6 (ζ=n₃) ↔ Outer patch t+3 face 1 (ζ=1)
    pairs = InterfacePair[]
    for t in 1:3
        push!(pairs, InterfacePair(t, 6, t+3, 1))
        if !(formulation isa SinglePassFormulation)
            push!(pairs, InterfacePair(t+3, 1, t, 6))
        end
    end

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    # ── Solve ───────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ─────────────────────────────────────────────────────
    σ_abs, σ_ref = l2_stress_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── L2 displacement error ───────────────────────────────────────────────
    disp_fn = (x, y, z) -> lame_displacement_sphere(x, y, z;
        p_i=p_i, r_i=r_i, r_o=r_o, E=E, nu=nu)
    l2_abs, l2_ref = l2_disp_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )

    # ── Energy-norm error ───────────────────────────────────────────────────
    en_abs, en_ref = energy_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── Optional VTK export ─────────────────────────────────────────────────
    if !isempty(vtk_prefix)
        write_vtk_sphere(vtk_prefix, U, ID, npc, nsd, npd,
                          p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                          nen, IEN, INC, E, nu;
                          n_vis=n_vis, p_i=p_i, r_i=r_i, r_o=r_o)
    end

    return (σ_rel=σ_abs/σ_ref, σ_abs=σ_abs,
            l2_rel=l2_abs/l2_ref, l2_abs=l2_abs,
            en_rel=en_abs/en_ref, en_abs=en_abs,
            K_bc=K_bc, C=C, Z=Z, neq=neq)
end

# ─── Convergence study driver ─────────────────────────────────────────────────

function run_convergence_deltoidal(;
    p_range = 4:4,
    exp_range = 0:2,
    E::Float64 = 1.0,
    nu::Float64 = 0.3,
    p_i::Float64 = 0.01,
    epss::Float64 = 100.0,
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    kwargs...
)
    for p in p_range
        tag = formulation isa TwinMortarFormulation ? "TM" :
              formulation isa DualPassMortarFormulation ? "DPM" : "SP"
        @printf("\n=== p=%d deltoidal sphere convergence (%s, ε=%.0e, p_i=%.2e) ===\n",
                p, tag, epss, p_i)
        prev_l2 = prev_en = prev_σ = 0.0
        for exp in exp_range
            @printf("  computing exp=%d ...", exp); flush(stdout)
            res = solve_sphere_deltoidal(p, exp; E=E, nu=nu, p_i=p_i, epss=epss,
                                          formulation=formulation, strategy=strategy, kwargs...)
            rate_l2 = exp > first(exp_range) ? log2(prev_l2 / res.l2_rel) : NaN
            rate_en = exp > first(exp_range) ? log2(prev_en / res.en_rel) : NaN
            @printf("\r  exp=%d  σ_rel=%.4e  l2_rel=%.4e  en_rel=%.4e  rate_l2=%5.2f  rate_en=%5.2f\n",
                    exp, res.σ_rel, res.l2_rel, res.en_rel, rate_l2, rate_en)
            flush(stdout)
            prev_l2 = res.l2_rel; prev_en = res.en_rel; prev_σ = res.σ_rel
        end
    end
end
