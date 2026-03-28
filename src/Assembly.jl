# Assembly.jl
# Element and global stiffness matrix assembly for small-deformation linear elasticity.
# Ported from MATLAB: buildElementStiffnessMatrix.m (small-deformation branch),
#   fem/buildStiffnessMatrix.m

"""
    element_stiffness(p, n, KV, P, B, Ub, ien, inc, el, nsd, npd, nen,
                      NQUAD, mat, thickness) -> (Ke, Fe)

Compute element stiffness matrix and internal force vector (small deformations only).

- `p`, `n`, `KV`, `P`, `B`: patch geometry
- `Ub`: displacement at control points (ncp × nsd); zero for first iteration
- `ien`: element-to-node connectivity for this patch (nel × nen)
- `inc`: INC table for this patch
- `el`: element index (1-based)
- `mat`: a `LinearElastic` material
- `thickness`: out-of-plane thickness (for plane problems)

Returns (Ke, Fe) both of size (ned*nen, ned*nen) and (ned*nen,) respectively.
"""
function element_stiffness(
    p::AbstractVector{Int},
    n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    P::AbstractVector{Int},
    B::AbstractMatrix{Float64},
    Ub::AbstractMatrix{Float64},
    ien::AbstractMatrix{Int},
    inc::AbstractVector{<:AbstractVector{Int}},
    el::Int,
    nsd::Int, npd::Int, nen::Int,
    NQUAD::Int,
    mat::LinearElastic,
    thickness::Float64;
    Bu::Union{AbstractMatrix{Float64}, Nothing} = nothing
)::Tuple{Matrix{Float64}, Vector{Float64}}

    ned  = nsd
    ndof = ned * nen

    Ke = zeros(ndof, ndof)
    Fe = zeros(ndof)

    # NURBS coordinates of the element anchor node
    anchor = ien[el, 1]
    n0 = inc[anchor]

    # Check zero-measure element
    for i in 1:npd
        kv = KV[i]
        if kv[n0[i]] ≈ kv[n0[i] + 1]
            return Ke, Fe
        end
    end

    # Elastic constants matrix
    D = elastic_constants(mat, nsd)

    # Current configuration (B + Ub)
    if Bu === nothing
        Bu_use = copy(B)
        Bu_use[:, 1:nsd] .+= Ub
    else
        Bu_use = Bu
    end

    # Element nodal displacements
    Ue  = Ub[P[ien[el, :]], 1:nsd]      # nen × nsd
    Ue_vec = vec(Ue')                    # [u1x, u1y, u2x, u2y, ...]

    GPW = gauss_product(NQUAD, npd)

    for (gp, gw) in GPW
        R, dR_dx, _, detJ, _ = shape_function(
            p, n, KV, Bu_use, P, gp, nen, nsd, npd, el, n0, ien, inc
        )

        @assert detJ > 0 "Negative Jacobian determinant in element $el"

        gwJ = gw * detJ * thickness

        # Strain-displacement matrix (uses physical gradients)
        B0 = strain_displacement_matrix(nsd, nen, dR_dx')   # nstrains × ndof

        # Small-strain stiffness
        Ke .+= B0' * D * B0 .* gwJ

        # Internal force (residual from existing displacement)
        ε    = B0 * Ue_vec                  # strain vector (Voigt)
        σ    = D * ε                         # stress vector
        Fe .+= B0' * σ .* gwJ
    end

    return Ke, Fe
end

"""
    _partition(n, k) -> Vector{Tuple{Int,Int}}

Split `1:n` into `k` contiguous chunks.  Returns a vector of `(start, stop)`
tuples; when `n < k` the trailing chunks are empty (`start > stop`).
"""
function _partition(n::Int, k::Int)
    base, rem = divrem(n, k)
    chunks = Vector{Tuple{Int,Int}}(undef, k)
    lo = 1
    for i in 1:k
        hi = lo + base - 1 + (i <= rem ? 1 : 0)
        chunks[i] = (lo, hi)
        lo = hi + 1
    end
    return chunks
end

"""
    build_stiffness_matrix(npc, nsd, npd, ned, neq, p, n, KV, P, B, Ub,
                           nen, nel, IEN, INC, LM, materials, NQUAD, thickness)
        -> (K::SparseMatrixCSC, F::Vector{Float64})

Assemble the global stiffness matrix K and internal force vector F.

- `materials[pc]`: material for patch pc
- All other args follow IGAros conventions.
"""
function build_stiffness_matrix(
    npc::Int, nsd::Int, npd::Int, ned::Int, neq::Int,
    p::Matrix{Int}, n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::AbstractMatrix{Float64},
    Ub::AbstractMatrix{Float64},
    nen::Vector{Int}, nel::Vector{Int},
    IEN::Vector{Matrix{Int}},
    INC::Vector{Vector{Vector{Int}}},
    LM::Vector{Matrix{Int}},
    materials::Vector{LinearElastic},
    NQUAD::Int,
    thickness::Float64
)::Tuple{SparseMatrixCSC{Float64,Int}, Vector{Float64}}

    # Pre-compute deformed configuration once (shared read-only across threads)
    Bu = copy(B)
    Bu[:, 1:nsd] .+= Ub

    # Flatten (patch, element) pairs into a work list
    work = Tuple{Int,Int}[]              # (patch_index, element_index)
    for pc in 1:npc, el in 1:nel[pc]
        push!(work, (pc, el))
    end
    ntasks = length(work)
    nt = max(Threads.nthreads(), 1)

    # Per-chunk accumulators (one per chunk to avoid data races)
    local_I = [Int[]     for _ in 1:nt]
    local_J = [Int[]     for _ in 1:nt]
    local_V = [Float64[] for _ in 1:nt]
    local_F = [zeros(neq) for _ in 1:nt]

    # Partition work into nt balanced chunks
    chunks = _partition(ntasks, nt)

    Threads.@threads for chunk_id in 1:nt
        a_start, a_end = chunks[chunk_id]
        for idx in a_start:a_end
            pc, el = work[idx]

            Ke, Fe = element_stiffness(
                p[pc, :], n[pc, :], KV[pc], P[pc], B, Ub,
                IEN[pc], INC[pc], el, nsd, npd, nen[pc], NQUAD,
                materials[pc], thickness; Bu=Bu
            )

            # Scatter into chunk-local triplets
            dofs = LM[pc][:, el]
            for a in 1:ned*nen[pc]
                row = dofs[a]
                row == 0 && continue
                local_F[chunk_id][row] += Fe[a]
                for b in 1:ned*nen[pc]
                    col = dofs[b]
                    col == 0 && continue
                    push!(local_I[chunk_id], row)
                    push!(local_J[chunk_id], col)
                    push!(local_V[chunk_id], Ke[a, b])
                end
            end
        end
    end

    # Merge chunk-local results
    I_idx = reduce(vcat, local_I)
    J_idx = reduce(vcat, local_J)
    K_val = reduce(vcat, local_V)
    F_vec = reduce(+, local_F)

    K = sparse(I_idx, J_idx, K_val, neq, neq)
    return K, F_vec
end

"""
    build_updated_geometry(nsd, ncp, ID, U, B) -> (Bu, Ub)

Update control point positions with displacement.

- `ID[i, A]`: global equation number for DOF i of CP A (0 = constrained)
- `U`: global displacement vector (length neq)
- `B`: reference control points (ncp × (nsd+1))

Returns:
- `Bu`: current positions (ncp × (nsd+1))  (only spatial cols updated)
- `Ub`: displacements at CPs (ncp × nsd)

Ported from buildBu.m.
"""
function build_updated_geometry(
    nsd::Int, ncp::Int,
    ID::Matrix{Int},
    U::Vector{Float64},
    B::Matrix{Float64}
)::Tuple{Matrix{Float64}, Matrix{Float64}}
    Ub = zeros(ncp, nsd)
    for i in 1:nsd
        for A in 1:ncp
            eq = ID[i, A]
            if eq != 0
                Ub[A, i] = U[eq]
            end
        end
    end
    Bu = copy(B)
    Bu[:, 1:nsd] .+= Ub
    return Bu, Ub
end
