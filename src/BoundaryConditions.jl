# BoundaryConditions.jl
# Dirichlet BCs (homogeneous and non-homogeneous), segment extraction, traction loads.
# Ported from MATLAB: getDirichBCcontrolPoints.m, getSegmentPatch.m,
#   enforceNonHomogeneousDirichletBoundaryConditions.m, segmentLoad.m

# ─── Segment (facet) extraction ───────────────────────────────────────────────

"""
    facet_parametric_dirs(facet, npd) -> (free_dirs::Vector{Int}, fixed_dir::Int, fixed_val::Int, norm_sign::Int)

Return the free parametric directions, the fixed direction, the fixed NURBS index value,
and the outward normal sign for a given facet label (1-6).

Facet convention (MATLAB legacy):
  1 → η = 1  (bottom)         normal sign +1
  2 → ξ = n₁ (right)         normal sign +1
  3 → η = n₂ (top)           normal sign -1
  4 → ξ = 1  (left)          normal sign -1
  5 → ζ = 1  (front, 3D)     normal sign -1
  6 → ζ = n₃ (back, 3D)      normal sign +1
"""
function facet_parametric_dirs(facet::Int, npd::Int)
    if npd == 2
        table = Dict(
            1 => (2, 2, 1, 1),       # free: dim2, fixed: dim2=1
            2 => (1, 1, -1, 1),      # free: dim1, fixed: dim1=n[1]  (-1 = end)
            3 => (2, 2, -1, -1),     # free: dim1, fixed: dim2=n[2]
            4 => (1, 1, 1, -1),      # free: dim2, fixed: dim1=1
        )
        free_dir, fixed_dir, fixed_val, norm_sign = table[facet]
        return [free_dir == 2 ? 1 : 2], fixed_dir, fixed_val, norm_sign
    else
        error("facet_parametric_dirs: only npd=2 implemented")
    end
end

"""
    get_segment_patch(p, n, KV, P, npd, nnp, facet) ->
        (ps, ns, KVs, Ps, nsn, nsen, nsel, norm_sign, as, b_dirs, xi_fixed)

Extract the boundary segment (facet) data from a 2D NURBS patch.

- `facet`: face label (1=bottom, 2=right, 3=top, 4=left)
- Returns reduced-dimension geometry for the boundary segment.

Ported from getSegmentPatch.m + patchSupspaceProjection.m.
"""
function get_segment_patch(
    p::AbstractVector{Int},
    n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    P::AbstractVector{Int},
    npd::Int, nnp::Int,
    facet::Int
)
    # Determine which direction is free (surface) and which is fixed (normal)
    b_dirs, norm_sign = _facet_dirs(facet, npd)
    fixed_dim, fixed_nc = _facet_fixed(facet, npd, n)

    # Segment (boundary) polynomial degrees and knot vectors
    nspdims = npd - 1
    ps = p[b_dirs]
    ns = n[b_dirs]
    KVs = KV[b_dirs]

    nsn  = prod(ns)                     # #boundary CPs
    nsen = prod(ps .+ 1)               # #local boundary basis functions
    nsel = prod(ns .- ps)              # #boundary elements

    # Collect boundary control point indices
    Ps = Int[]
    as = Int[]
    xi_fixed = (fixed_nc == 1) ? 0.0 : 1.0

    for A in 1:nnp
        nc = nurbs_coords(A, npd, n)
        if nc[fixed_dim] == fixed_nc
            push!(Ps, P[A])
            push!(as, A)
        end
    end

    return ps, ns, KVs, Ps, nsn, nsen, nsel, norm_sign, as, b_dirs, xi_fixed
end

# Maps facet → (free parametric directions, norm_sign)
function _facet_dirs(facet::Int, npd::Int)
    if npd == 2
        # facet 1: bottom (η=1),  free: η-lines
        # facet 2: right  (ξ=n₁), free: ξ-lines
        # facet 3: top    (η=n₂), free: η-lines
        # facet 4: left   (ξ=1),  free: ξ-lines
        table = Dict(1 => ([1], 1), 2 => ([2], 1), 3 => ([1], -1), 4 => ([2], -1))
        return table[facet]
    end
    error("_facet_dirs: npd=$npd not implemented")
end

# Maps facet → (fixed dimension index, fixed NURBS coordinate value)
function _facet_fixed(facet::Int, npd::Int, n::AbstractVector{Int})
    if npd == 2
        table = Dict(
            1 => (2, 1),           # η = 1
            2 => (1, n[1]),        # ξ = n[1]  (last)
            3 => (2, n[2]),        # η = n[2]  (last)
            4 => (1, 1),           # ξ = 1
        )
        return table[facet]
    end
    error("_facet_fixed: npd=$npd not implemented")
end

# ─── Dirichlet BC control point extraction ────────────────────────────────────

"""
    dirichlet_bc_control_points(p, n, KV, CP, npd, nnp, ned, dBC) ->
        Vector{Vector{Int}}

For each DOF component i, collect the global CP indices with homogeneous Dirichlet BC.

`dBC` is a matrix where each row is:
  [dof, facet, n_patches, pc1, pc2, ...]

Returns `BC[i]` = sorted unique list of CP indices constrained in DOF i.

Ported from getDirichBCcontrolPoints.m.
"""
function dirichlet_bc_control_points(
    p::Matrix{Int},
    n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    CP::Vector{Vector{Int}},   # P{pc} = CP indices for patch pc
    npd::Int,
    nnp::Vector{Int},
    ned::Int,
    dBC::Matrix{Int}
)::Vector{Vector{Int}}

    BC = [Int[] for _ in 1:ned]
    isempty(dBC) && return BC

    for i in 1:size(dBC, 1)
        dof    = dBC[i, 1]
        facet  = dBC[i, 2]
        n_pat  = dBC[i, 3]
        for j in 1:n_pat
            pc = dBC[i, 3 + j]
            _, _, _, Ps, _, _, _, _, _, _, _ = get_segment_patch(
                p[pc, :], n[pc, :], KV[pc], CP[pc], npd, nnp[pc], facet
            )
            append!(BC[dof], Ps)
        end
    end
    return [sort(unique(bc)) for bc in BC]
end

# ─── Non-homogeneous Dirichlet enforcement ─────────────────────────────────────

"""
    enforce_dirichlet(IND, K0, F0) -> (K, F)

Modify the global system to enforce non-homogeneous Dirichlet BCs by row/column elimination.

- `IND`: vector of (global_dof, prescribed_value) pairs
- `K0`: original stiffness matrix (neq × neq, sparse)
- `F0`: original RHS vector (neq,)

For each constrained DOF A with value vA:
  F ← F - K[:,A] * vA
  K[A,:] = K[:,A] = 0; K[A,A] = 1; F[A] = vA

Ported from enforceNonHomogeneousDirichletBoundaryConditions.m.
"""
function enforce_dirichlet(
    IND::Vector{Tuple{Int, Float64}},
    K0::SparseMatrixCSC{Float64, Int},
    F0::Vector{Float64}
)::Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}

    K = copy(K0)
    F = copy(F0)

    for (A, vA) in IND
        # Load correction: subtract column A scaled by vA from RHS
        col_A = K[:, A]
        F .-= col_A .* vA

        # Zero row A and column A
        K[A, :] .= 0.0
        K[:, A] .= 0.0

        # Set diagonal and RHS
        K[A, A] = 1.0
        F[A]    = vA
    end
    return K, F
end

# ─── Surface traction load assembly ───────────────────────────────────────────

"""
    segment_load(n, p, KV, P, B, nnp, nen, nsd, npd, ned,
                 loaded_el, facet, ID, F, traction, thickness, NQUAD) -> F

Assemble surface traction contribution into global RHS vector F.

- `traction`: scalar (pressure) or (nsd,) vector
- `loaded_el`: 1-based element indices to load (Int[] = all elements)
- `facet`: facet label (1–4 for 2D)

Ported from segmentLoad.m.
"""
function segment_load(
    n::AbstractVector{Int},
    p::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    P::AbstractVector{Int},
    B::AbstractMatrix{Float64},
    nnp::Int, nsen_full::Int,
    nsd::Int, npd::Int, ned::Int,
    loaded_el::AbstractVector{Int},
    facet::Int,
    ID::Matrix{Int},
    F::Vector{Float64},
    traction,
    thickness::Float64,
    NQUAD::Int
)::Vector{Float64}

    F = copy(F)

    ps, ns, KVs, Ps, nsn, nsen, nsel, norm_sign, as, b_dirs, _ =
        get_segment_patch(p, n, KV, P, npd, nnp, facet)

    ien_s = build_ien(nsd, npd - 1, 1,
                      reshape(ps, 1, :), reshape(ns, 1, :),
                      [nsel], [nsn], [nsen])
    ien = ien_s[1]
    inc = build_inc(ns)

    els = isempty(loaded_el) ? (1:nsel) : loaded_el

    GPW = gauss_product(NQUAD, npd - 1)

    for el in els
        anchor = ien[el, 1]
        n0     = inc[anchor]

        Fs = zeros(ned, nsen)

        for (gp, gw) in GPW
            R, _, _, detJ, n_vec = shape_function(
                ps, ns, KVs, B, Ps, gp, nsen, nsd, npd - 1, el, n0, ien, inc
            )
            n_vec .*= norm_sign
            gwJ = gw * detJ * thickness

            if traction isa Function
                Xe = B[Ps[ien[el, :]], :]
                X  = Xe' * R               # physical coordinates (X[1]=x, X[2]=y)
                σ  = traction(X[1], X[2])  # must return (nsd×nsd) Cauchy stress matrix
                Fp = σ * n_vec             # traction = σ · n  (2×2 @ 2→2)
            elseif length(traction) > 1
                Fp = traction               # global traction vector
            else
                Fp = traction .* n_vec      # scalar normal traction (positive = outward)
            end

            Fs .+= Fp * R' .* gwJ
        end

        # Scatter into global F
        for a in 1:nsen
            cp = Ps[ien[el, a]]
            for i in 1:ned
                eq = ID[i, cp]
                eq != 0 && (F[eq] += Fs[i, a])
            end
        end
    end

    return F
end

"""
    build_homogeneous_dirichlet(ned, ncp, BC) -> Vector{Vector{Int}}

Build BC list compatible with `build_id`. Just a passthrough alias.
"""
build_homogeneous_dirichlet(ned::Int, ncp::Int, BC::Vector{Vector{Int}}) = BC
