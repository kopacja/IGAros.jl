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

    # Prevent BLAS from spawning its own threads (conflicts with Julia threads)
    old_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)

    # Pre-compute deformed configuration once (shared read-only across threads)
    Bu = copy(B)
    Bu[:, 1:nsd] .+= Ub

    nstrain = nsd == 2 ? 3 : 6
    nt = max(Threads.nthreads(), 1)
    F_vec = zeros(neq)

    # Global maxima for workspace sizing
    p_max   = maximum(p)
    nen_max = maximum(nen)
    ndof_max = ned * nen_max

    # Pre-compute PatchConstants (read-only, shared across threads)
    patch_consts = [PatchConstants(p[pc, :], nen[pc], nsd, npd, NQUAD, materials[pc])
                    for pc in 1:npc]
    ngp = NQUAD^npd

    # Pre-extract per-patch degree/count vectors (avoid p[pc,:] slice allocations)
    p_vecs = [p[pc, :] for pc in 1:npc]
    n_vecs = [n[pc, :] for pc in 1:npc]

    # Total element count for triplet sizing
    total_els = sum(nel)
    max_triplets_per_ws = cld(total_els, nt) * ndof_max * ndof_max

    # Create workspaces ONCE, reuse across all patches
    workspaces = [AssemblyWorkspace(p_max, nen_max, nsd, npd, neq, max_triplets_per_ws)
                  for _ in 1:nt]

    # Flatten (patch, element) pairs into a work list
    work = Vector{Tuple{Int,Int}}(undef, total_els)
    wi = 0
    for pc in 1:npc
        for el in 1:nel[pc]
            wi += 1
            work[wi] = (pc, el)
        end
    end

    # Partition ALL elements across threads (not per-patch)
    chunks = _partition(total_els, nt)

    Threads.@threads :static for chunk_id in 1:nt
        ws = workspaces[chunk_id]
        ws.triplet_pos[] = 0
        fill!(ws.F_buf, 0.0)

        a_start, a_end = chunks[chunk_id]
        a_start > a_end && continue

        for idx in a_start:a_end
            pc, el = work[idx]
            ndof_el = ned * nen[pc]
            nen_pc  = nen[pc]
            pc_const = patch_consts[pc]
            p_pc = p_vecs[pc]
            n_pc = n_vecs[pc]

            # --- Knot span indices (from INC, no allocation) ---
            anchor = IEN[pc][el, 1]
            n0 = INC[pc][anchor]

            # --- Zero-measure element check ---
            skip = false
            @inbounds for i in 1:npd
                kv = KV[pc][i]
                if kv[n0[i]] ≈ kv[n0[i] + 1]
                    skip = true
                    break
                end
            end
            skip && continue

            # --- Zero Ke, Fe ---
            @inbounds for j in 1:ndof_el
                @simd for i in 1:ndof_el
                    ws.Ke[i, j] = 0.0
                end
            end
            @inbounds @simd for i in 1:ndof_el
                ws.Fe[i] = 0.0
            end

            # --- Extract Ue_vec (displacement at element nodes) ---
            @inbounds for a in 1:nen_pc
                cp = P[pc][IEN[pc][el, a]]
                for d in 1:nsd
                    ws.Ue_vec[(a-1)*ned + d] = Ub[cp, d]
                end
            end

            # --- Gauss quadrature loop ---
            @inbounds for igp in 1:ngp
                detJ = shape_function!(
                    ws, pc_const, p_pc, n_pc, KV[pc], Bu, P[pc],
                    igp, nen_pc, nsd, npd, el, n0, IEN[pc], INC[pc]
                )

                detJ <= 0.0 && continue

                gwJ = pc_const.gp_weights[igp] * detJ * thickness

                # --- Build B0 in-place ---
                for j in 1:nen_pc
                    @simd for i in 1:nsd
                        ws.dN_dX_T[i, j] = ws.dR_dx[j, i]
                    end
                end
                strain_displacement_matrix!(ws.B0, nsd, nen_pc, ws.dN_dX_T)

                # --- Ke += B0' * D * B0 * gwJ ---
                if ndof_el >= 24  # 3D or high-order: BLAS mul!
                    B0v   = @view ws.B0[1:nstrain, 1:ndof_el]
                    BtDv  = @view ws.BtD[1:ndof_el, 1:nstrain]
                    BtDBv = @view ws.BtDB[1:ndof_el, 1:ndof_el]
                    Kev   = @view ws.Ke[1:ndof_el, 1:ndof_el]
                    mul!(BtDv, transpose(B0v), pc_const.D)
                    mul!(BtDBv, BtDv, B0v)
                    @simd for i in eachindex(Kev)
                        Kev[i] += BtDBv[i] * gwJ
                    end
                    # Fe via BLAS
                    eps_v = @view ws.eps_vec[1:nstrain]
                    sig_v = @view ws.sigma_vec[1:nstrain]
                    Ue_v  = @view ws.Ue_vec[1:ndof_el]
                    Bt_s  = @view ws.Bt_sigma[1:ndof_el]
                    mul!(eps_v, B0v, Ue_v)
                    mul!(sig_v, pc_const.D, eps_v)
                    mul!(Bt_s, transpose(B0v), sig_v)
                    @simd for i in 1:ndof_el
                        ws.Fe[i] += Bt_s[i] * gwJ
                    end
                else  # 2D small elements: scalar loops (faster than BLAS dispatch)
                    # BtD = B0' * D
                    for j in 1:nstrain
                        for i in 1:ndof_el
                            s = 0.0
                            @simd for k in 1:nstrain
                                s += ws.B0[k, i] * pc_const.D[k, j]
                            end
                            ws.BtD[i, j] = s
                        end
                    end
                    # Ke += BtD * B0 * gwJ
                    for j in 1:ndof_el
                        for i in 1:ndof_el
                            s = 0.0
                            @simd for k in 1:nstrain
                                s += ws.BtD[i, k] * ws.B0[k, j]
                            end
                            ws.Ke[i, j] += s * gwJ
                        end
                    end
                    # Fe: eps = B0 * Ue_vec
                    for i in 1:nstrain
                        s = 0.0
                        @simd for k in 1:ndof_el
                            s += ws.B0[i, k] * ws.Ue_vec[k]
                        end
                        ws.eps_vec[i] = s
                    end
                    # sigma = D * eps
                    for i in 1:nstrain
                        s = 0.0
                        @simd for k in 1:nstrain
                            s += pc_const.D[i, k] * ws.eps_vec[k]
                        end
                        ws.sigma_vec[i] = s
                    end
                    # Fe += B0' * sigma * gwJ
                    for i in 1:ndof_el
                        s = 0.0
                        @simd for k in 1:nstrain
                            s += ws.B0[k, i] * ws.sigma_vec[k]
                        end
                        ws.Fe[i] += s * gwJ
                    end
                end
            end  # igp

            # --- Scatter into pre-sized triplets ---
            pos = ws.triplet_pos[]
            @inbounds for a in 1:ndof_el
                row = LM[pc][a, el]
                row == 0 && continue
                ws.F_buf[row] += ws.Fe[a]
                for b in 1:ndof_el
                    col = LM[pc][b, el]
                    col == 0 && continue
                    ke_val = ws.Ke[a, b]
                    ke_val == 0.0 && continue
                    pos += 1
                    ws.I_buf[pos] = row
                    ws.J_buf[pos] = col
                    ws.V_buf[pos] = ke_val
                end
            end
            ws.triplet_pos[] = pos
        end  # idx
    end  # @threads

    # --- Single sparse() call from all threads ---
    total_nnz = sum(ws.triplet_pos[] for ws in workspaces)
    I_all = Vector{Int}(undef, total_nnz)
    J_all = Vector{Int}(undef, total_nnz)
    V_all = Vector{Float64}(undef, total_nnz)
    off = 0
    for c in 1:nt
        n_c = workspaces[c].triplet_pos[]
        n_c == 0 && continue
        copyto!(I_all, off + 1, workspaces[c].I_buf, 1, n_c)
        copyto!(J_all, off + 1, workspaces[c].J_buf, 1, n_c)
        copyto!(V_all, off + 1, workspaces[c].V_buf, 1, n_c)
        off += n_c
    end
    for c in 1:nt
        F_vec .+= workspaces[c].F_buf
    end

    K = sparse(I_all, J_all, V_all, neq, neq)

    LinearAlgebra.BLAS.set_num_threads(old_blas_threads)
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
