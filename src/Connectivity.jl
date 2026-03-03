# Connectivity.jl
# IGA connectivity arrays: INC, IEN, ID, LM, IND.
# Ported from MATLAB: getNURBScoords.m, buildINC.m, BuildIEN.m,
#   Build_ID.m, Build_LM.m, buildIND.m

# ─── NURBS coordinate conversion ─────────────────────────────────────────────

"""
    nurbs_coords(A, npd, n) -> Vector{Int}

Convert linear (1-based) control point index `A` to multi-dimensional NURBS coordinates.

- `A`: linear index in [1, prod(n)]
- `npd`: number of parametric dimensions
- `n`: number of control points per direction

Returns `nc[i]` ∈ [1, n[i]] for each direction i.

Ported from getNURBScoords.m.
"""
function nurbs_coords(A::Int, npd::Int, n::AbstractVector{Int})::Vector{Int}
    nc = zeros(Int, npd)
    denom = 1
    for i in 1:npd
        nc[i] = floor(Int, mod((A - 1) / denom, n[i])) + 1
        denom *= n[i]
    end
    return nc
end

# ─── INC (element-interior NURBS coords) ─────────────────────────────────────

"""
    build_inc(n) -> Vector{Vector{Int}}

Build INC: for each control point A (1-based), store its NURBS coordinates.
`INC[A]` = multi-index vector of length npd.

Ported from buildINC.m.
"""
function build_inc(n::AbstractVector{Int})::Vector{Vector{Int}}
    npd = length(n)
    ncp = prod(n)
    INC = Vector{Vector{Int}}(undef, ncp)
    for A in 1:ncp
        INC[A] = nurbs_coords(A, npd, n)
    end
    return INC
end

# ─── IEN (element-to-node connectivity) ──────────────────────────────────────

"""
    build_ien(nsd, npd, npc, p, n, nel, nnp, nen) -> Vector{Matrix{Int}}

Build IEN connectivity for each patch.
`IEN[pc][el, a]` = local control point index (within patch) for element `el`, local node `a`.

- `p[pc, i]`: degree per patch and direction
- `n[pc, i]`: #control points per patch and direction
- `nel[pc]`, `nnp[pc]`, `nen[pc]`: #elements, #nodes, #local nodes per patch

Ported from BuildIEN.m.
"""
function build_ien(
    nsd::Int, npd::Int, npc::Int,
    p::Matrix{Int}, n::Matrix{Int},
    nel::Vector{Int}, nnp::Vector{Int}, nen::Vector{Int}
)::Vector{Matrix{Int}}

    IEN = Vector{Matrix{Int}}(undef, npc)

    for pc in 1:npc
        pp = p[pc, :]
        nn = n[pc, :]

        # coef[i] = stride in the local element index for direction i
        coef = ones(Int, npd + 1)
        for i in 1:npd
            coef[i + 1] = coef[i] * (pp[i] + 1)
        end

        el = 1
        ien = zeros(Int, nel[pc], nen[pc])

        for A in 1:nnp[pc]
            nc = nurbs_coords(A, npd, nn)

            # Skip the first p[i] control points in each direction
            skip = any(nc[i] <= pp[i] for i in 1:npd)
            skip && continue

            for a in 1:nen[pc]
                B = A
                b = 1
                denom_loc = 1
                mult = 1
                for i in 1:npd
                    loc = floor(Int, mod((a - 1) / denom_loc, pp[i] + 1))
                    b  += loc * coef[i]
                    B  -= loc * mult
                    denom_loc *= (pp[i] + 1)
                    mult      *= nn[i]
                end
                ien[el, b] = B
            end
            el += 1
        end

        IEN[pc] = ien
    end
    return IEN
end

# ─── ID (DOF numbering) ───────────────────────────────────────────────────────

"""
    build_id(bc_per_dof, ned, ncp) -> (neq::Int, ID::Matrix{Int})

Build global DOF numbering array.

- `bc_per_dof[i]`: sorted list of control point indices with homogeneous Dirichlet BC on DOF i
- `ned`: number of DOF components per node
- `ncp`: total number of control points

Returns:
- `neq`: total number of free DOFs
- `ID[i, A]`: global equation number for DOF i of control point A (0 = constrained)

Ported from Build_ID.m.
"""
function build_id(
    bc_per_dof::Vector{Vector{Int}},
    ned::Int, ncp::Int
)::Tuple{Int, Matrix{Int}}

    ID = zeros(Int, ned, ncp)
    eq = 0

    for A in 1:ncp
        for i in 1:ned
            if A in bc_per_dof[i]
                ID[i, A] = 0
            else
                eq += 1
                ID[i, A] = eq
            end
        end
    end
    return eq, ID
end

# ─── LM (local-to-global DOF map) ────────────────────────────────────────────

"""
    build_lm(nen, ned, npc, nel, ID, IEN, P) -> Vector{Matrix{Int}}

Build LM connectivity: `LM[pc][dof_local, el]` = global equation number.

- `dof_local` runs from 1 to ned*nen[pc] (node-major order: DOF 1 of node 1, DOF 2 of node 1, …)
- Returns 0 for constrained DOFs.

Ported from Build_LM.m.
"""
function build_lm(
    nen::Vector{Int}, ned::Int, npc::Int,
    nel::Vector{Int},
    ID::Matrix{Int},
    IEN::Vector{Matrix{Int}},
    P::Vector{Vector{Int}}
)::Vector{Matrix{Int}}

    LM = Vector{Matrix{Int}}(undef, npc)
    for pc in 1:npc
        Pp  = P[pc]
        ien = IEN[pc]
        lm  = zeros(Int, ned * nen[pc], nel[pc])
        for e in 1:nel[pc]
            for a in 1:nen[pc]
                for i in 1:ned
                    local_dof = ned * (a - 1) + i
                    lm[local_dof, e] = ID[i, Pp[ien[e, a]]]
                end
            end
        end
        LM[pc] = lm
    end
    return LM
end

# ─── IND (non-homogeneous Dirichlet BCs) ─────────────────────────────────────

"""
    build_ind(ndBC_rows, ndBCcp, ned, ID, p, n, KV, CP, npd, nnp) ->
        Vector{Tuple{Int,Float64}}

Build list of (global_dof, prescribed_value) pairs for non-homogeneous Dirichlet BCs.

- `ndBC_rows`: rows of ndBC matrix; each row is [dof, segment, npatch, patch_indices..., value]
- `ndBCcp[i]`: vector [value, cp1, cp2, ...] for DOF i (or empty)

Ported from buildIND.m.
"""
function build_ind(
    ndBC_rows::Vector{Vector{Float64}},
    ndBCcp::Vector{Vector{Float64}},
    ned::Int,
    ID::Matrix{Int},
    dirichlet_fn::Function  # dirichlet_bc_control_points(dBC_row, p, n, KV, CP, npd, nnp)
)::Vector{Tuple{Int, Float64}}

    IND = Tuple{Int, Float64}[]

    for row in ndBC_rows
        dof   = Int(row[1])
        value = row[end]
        dBC_row = row[1:end-1]
        bc = dirichlet_fn(dBC_row)
        for cp in bc[dof]
            eq = ID[dof, cp]
            @assert eq != 0 "Non-homo BC on constrained DOF"
            push!(IND, (eq, value))
        end
    end

    for dof in 1:ned
        isempty(ndBCcp[dof]) && continue
        value = ndBCcp[dof][1]
        cps   = Int.(ndBCcp[dof][2:end])
        for cp in cps
            eq = ID[dof, cp]
            @assert eq != 0 "Non-homo BC on constrained DOF"
            push!(IND, (eq, value))
        end
    end

    return IND
end

# ─── Patch metrics ────────────────────────────────────────────────────────────

"""
    patch_metrics(npc, npd, p, n) -> (nel, nnp, nen)

Compute number of elements, control points, and local basis functions per patch.
"""
function patch_metrics(
    npc::Int, npd::Int,
    p::Matrix{Int}, n::Matrix{Int}
)::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
    nel = ones(Int, npc)
    nen = ones(Int, npc)
    nnp = ones(Int, npc)
    for pc in 1:npc
        for i in 1:npd
            nel[pc] *= (n[pc, i] - p[pc, i])
            nen[pc] *= (p[pc, i] + 1)
            nnp[pc] *= n[pc, i]
        end
    end
    return nel, nnp, nen
end
