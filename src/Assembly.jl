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

    nstrain = nsd == 2 ? 3 : 6
    nt = max(Threads.nthreads(), 1)
    K = spzeros(neq, neq)
    F_vec = zeros(neq)

    # Global maxima for workspace sizing
    p_max   = maximum(p)
    nen_max = maximum(nen)

    # Pre-compute PatchConstants (read-only, shared across threads)
    patch_consts = [PatchConstants(p[pc, :], nen[pc], nsd, npd, NQUAD, materials[pc])
                    for pc in 1:npc]
    ngp = NQUAD^npd

    # Assemble per-patch to limit peak memory
    for pc in 1:npc
        n_el = nel[pc]
        n_el == 0 && continue
        ndof_el = ned * nen[pc]
        nen_pc  = nen[pc]
        pc_const = patch_consts[pc]

        # Partition elements across threads
        chunks = _partition(n_el, nt)

        # Compute max chunk size for triplet pre-allocation
        max_chunk_els = maximum(c[2] - c[1] + 1 for c in chunks)
        max_triplets  = max_chunk_els * ndof_el * ndof_el

        # Create one workspace per thread
        workspaces = [AssemblyWorkspace(p_max, nen_max, nsd, npd, neq, max_triplets)
                      for _ in 1:nt]

        Threads.@threads for chunk_id in 1:nt
            ws = workspaces[chunk_id]
            ws.triplet_pos[] = 0
            a_start, a_end = chunks[chunk_id]

            p_pc = p[pc, :]
            n_pc = n[pc, :]

            for el in a_start:a_end
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
                if skip
                    # Scatter zeros (Ke/Fe already zero-effect)
                    continue
                end

                # --- Zero Ke, Fe ---
                @inbounds for j in 1:ndof_el, i in 1:ndof_el
                    ws.Ke[i, j] = 0.0
                end
                @inbounds for i in 1:ndof_el
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
                    # shape_function! reads xi_tilde from pc_const.gp_coords
                    xi_tilde = @view pc_const.gp_coords[igp, :]

                    detJ = shape_function!(
                        ws, pc_const, p_pc, n_pc, KV[pc], Bu, P[pc],
                        xi_tilde, nen_pc, nsd, npd, el, n0, IEN[pc], INC[pc]
                    )

                    detJ <= 0.0 && continue

                    gwJ = pc_const.gp_weights[igp] * detJ * thickness

                    # --- Build B0 in-place ---
                    # dN_dX_T = transpose of dR_dx (nsd × nen)
                    for j in 1:nen_pc
                        for i in 1:nsd
                            ws.dN_dX_T[i, j] = ws.dR_dx[j, i]
                        end
                    end
                    strain_displacement_matrix!(ws.B0, nsd, nen_pc, ws.dN_dX_T)

                    # --- Ke += B0' * D * B0 * gwJ ---
                    # BtD = B0' * D  (ndof × nstrain)
                    for j in 1:nstrain
                        for i in 1:ndof_el
                            s = 0.0
                            for k in 1:nstrain
                                s += ws.B0[k, i] * pc_const.D[k, j]
                            end
                            ws.BtD[i, j] = s
                        end
                    end
                    # BtDB = BtD * B0  (ndof × ndof)
                    for j in 1:ndof_el
                        for i in 1:ndof_el
                            s = 0.0
                            for k in 1:nstrain
                                s += ws.BtD[i, k] * ws.B0[k, j]
                            end
                            ws.Ke[i, j] += s * gwJ
                        end
                    end

                    # --- Fe += B0' * sigma * gwJ ---
                    # eps = B0 * Ue_vec
                    for i in 1:nstrain
                        s = 0.0
                        for k in 1:ndof_el
                            s += ws.B0[i, k] * ws.Ue_vec[k]
                        end
                        ws.eps_vec[i] = s
                    end
                    # sigma = D * eps
                    for i in 1:nstrain
                        s = 0.0
                        for k in 1:nstrain
                            s += pc_const.D[i, k] * ws.eps_vec[k]
                        end
                        ws.sigma_vec[i] = s
                    end
                    # Bt_sigma = B0' * sigma
                    for i in 1:ndof_el
                        s = 0.0
                        for k in 1:nstrain
                            s += ws.B0[k, i] * ws.sigma_vec[k]
                        end
                        ws.Fe[i] += s * gwJ
                    end
                end  # igp

                # --- Scatter into pre-sized triplets ---
                pos = ws.triplet_pos[]
                for a in 1:ndof_el
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
            end  # el
        end  # @threads

        # --- Merge chunk-local results ---
        total_nnz = sum(ws.triplet_pos[] for ws in workspaces)
        I_pc = Vector{Int}(undef, total_nnz)
        J_pc = Vector{Int}(undef, total_nnz)
        V_pc = Vector{Float64}(undef, total_nnz)
        off = 0
        for c in 1:nt
            n_c = workspaces[c].triplet_pos[]
            n_c == 0 && continue
            copyto!(I_pc, off + 1, workspaces[c].I_buf, 1, n_c)
            copyto!(J_pc, off + 1, workspaces[c].J_buf, 1, n_c)
            copyto!(V_pc, off + 1, workspaces[c].V_buf, 1, n_c)
            off += n_c
        end
        for c in 1:nt
            F_vec .+= workspaces[c].F_buf
        end

        # Add patch contribution
        if total_nnz > 0
            K += sparse(I_pc, J_pc, V_pc, neq, neq)
        end
    end  # pc

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
