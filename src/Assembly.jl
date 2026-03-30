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

    nt = max(Threads.nthreads(), 1)
    K = spzeros(neq, neq)
    F_vec = zeros(neq)

    # Assemble per-patch to limit peak memory (one patch's triplets at a time)
    for pc in 1:npc
        n_el = nel[pc]
        n_el == 0 && continue
        ndof_el = ned * nen[pc]

        # Partition elements across threads
        chunks = _partition(n_el, nt)
        local_I = [Int[]     for _ in 1:nt]
        local_J = [Int[]     for _ in 1:nt]
        local_V = [Float64[] for _ in 1:nt]
        local_F = [zeros(neq) for _ in 1:nt]

        # Pre-allocate per-chunk based on estimated nnz
        for c in 1:nt
            a_s, a_e = chunks[c]
            est = (a_e - a_s + 1) * ndof_el * ndof_el
            sizehint!(local_I[c], est)
            sizehint!(local_J[c], est)
            sizehint!(local_V[c], est)
        end

        Threads.@threads for chunk_id in 1:nt
            a_start, a_end = chunks[chunk_id]
            for el in a_start:a_end
                Ke, Fe = element_stiffness(
                    p[pc, :], n[pc, :], KV[pc], P[pc], B, Ub,
                    IEN[pc], INC[pc], el, nsd, npd, nen[pc], NQUAD,
                    materials[pc], thickness; Bu=Bu
                )

                dofs = LM[pc][:, el]
                for a in 1:ndof_el
                    row = dofs[a]
                    row == 0 && continue
                    local_F[chunk_id][row] += Fe[a]
                    for b in 1:ndof_el
                        col = dofs[b]
                        (col == 0 || Ke[a, b] == 0.0) && continue
                        push!(local_I[chunk_id], row)
                        push!(local_J[chunk_id], col)
                        push!(local_V[chunk_id], Ke[a, b])
                    end
                end
            end
        end

        # Merge chunk-local triplets into per-patch arrays
        total_nnz = sum(length, local_I)
        I_pc = Vector{Int}(undef, total_nnz)
        J_pc = Vector{Int}(undef, total_nnz)
        V_pc = Vector{Float64}(undef, total_nnz)
        off = 0
        for c in 1:nt
            n_c = length(local_I[c])
            copyto!(I_pc, off + 1, local_I[c], 1, n_c)
            copyto!(J_pc, off + 1, local_J[c], 1, n_c)
            copyto!(V_pc, off + 1, local_V[c], 1, n_c)
            local_I[c] = Int[]; local_J[c] = Int[]; local_V[c] = Float64[]
            off += n_c
        end
        F_vec .+= reduce(+, local_F)

        # Add patch contribution and free triplets
        K += sparse(I_pc, J_pc, V_pc, neq, neq)
    end

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
