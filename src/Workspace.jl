# Workspace.jl
# Pre-allocated workspace structs for allocation-free threaded assembly.

# Local copy of _local_nc (originally in Geometry.jl) to break circular include dependency.
function _local_nc_ws(p::AbstractVector{Int}, npd::Int, nen::Int)::Matrix{Int}
    nc = zeros(Int, nen, npd)
    denom = 1
    for i in 1:npd
        for a in 1:nen
            nc[a, i] = floor(Int, mod((a - 1) / denom, p[i] + 1))
        end
        denom *= (p[i] + 1)
    end
    return nc
end

"""
    PatchConstants

Read-only data that is constant for all elements within a patch.
Computed once before the threaded loop and shared across threads.
"""
struct PatchConstants
    D::Matrix{Float64}           # nstrain × nstrain  (elastic constants)
    nc::Matrix{Int}              # nen × npd  (0-based tensor-product offsets)
    idx::Matrix{Int}             # nen × npd  (1-based lookup into 1D basis)
    gp_coords::Matrix{Float64}  # ngp × npd  (Gauss point coordinates)
    gp_weights::Vector{Float64}  # ngp        (Gauss point weights)
end

function PatchConstants(
    p_dir::AbstractVector{Int}, nen::Int,
    nsd::Int, npd::Int, NQUAD::Int, mat::LinearElastic
)
    D = elastic_constants(mat, nsd)
    nc = _local_nc_ws(p_dir, npd, nen)
    idx_mat = zeros(Int, nen, npd)
    for i in 1:npd
        for a in 1:nen
            idx_mat[a, i] = (p_dir[i] + 1) - nc[a, i]
        end
    end

    # Build flat Gauss rule (Matrix + Vector instead of Vector{Tuple{Vector,Float64}})
    pts1d, wts1d = gauss_rule(NQUAD)
    ngp = NQUAD^npd
    gp_coords = zeros(ngp, npd)
    gp_weights = zeros(ngp)
    for igp in 1:ngp
        gw = 1.0
        denom = 1
        for i in 1:npd
            ig = mod((igp - 1) ÷ denom, NQUAD) + 1
            gp_coords[igp, i] = pts1d[ig]
            gw *= wts1d[ig]
            denom *= NQUAD
        end
        gp_weights[igp] = gw
    end

    return PatchConstants(D, nc, idx_mat, gp_coords, gp_weights)
end

"""
    AssemblyWorkspace

Per-thread scratch buffers for the element assembly hot loop.
Sized for the maximum element dimensions across all patches.
No allocation occurs during assembly — all writes go into these buffers.
"""
struct AssemblyWorkspace
    # Element level
    Ke::Matrix{Float64}
    Fe::Vector{Float64}
    Ue_vec::Vector{Float64}
    dofs::Vector{Int}

    # Shape function outputs
    R::Vector{Float64}
    dR_dx::Matrix{Float64}
    dR_dXi::Matrix{Float64}
    dx_dXi::Matrix{Float64}
    n_vec::Vector{Float64}

    # Shape function intermediates
    Xi::Vector{Float64}
    dXi_dtildeXi::Matrix{Float64}
    N_num::Vector{Float64}
    dN_dXi_num::Matrix{Float64}
    dsum_W::Vector{Float64}
    W_buf::Vector{Float64}
    col_tmp::Vector{Float64}
    Xcp::Matrix{Float64}

    # 1D basis buffers (per direction)
    NN::Vector{Vector{Float64}}
    dNN_dXi::Vector{Vector{Float64}}

    # bspline_basis_and_deriv! buffers
    ndu::Matrix{Float64}
    bsp_left::Vector{Float64}
    bsp_right::Vector{Float64}
    ders::Matrix{Float64}
    a_bsp::Matrix{Float64}

    # Strain-displacement and matrix-product temporaries
    B0::Matrix{Float64}
    dN_dX_T::Matrix{Float64}
    BtD::Matrix{Float64}
    BtDB::Matrix{Float64}
    eps_vec::Vector{Float64}
    sigma_vec::Vector{Float64}
    Bt_sigma::Vector{Float64}

    # Jacobian intermediates
    J_mat::Matrix{Float64}
    dXi_dx::Matrix{Float64}

    # Triplet output (pre-sized per chunk)
    I_buf::Vector{Int}
    J_buf::Vector{Int}
    V_buf::Vector{Float64}
    F_buf::Vector{Float64}
    triplet_pos::Base.RefValue{Int}
end

function AssemblyWorkspace(
    p_max::Int, nen_max::Int, nsd::Int, npd::Int, neq::Int,
    max_triplets::Int
)
    ndof_max = nsd * nen_max
    nstrain = nsd == 2 ? 3 : 6

    return AssemblyWorkspace(
        # Element level
        zeros(ndof_max, ndof_max),      # Ke
        zeros(ndof_max),                 # Fe
        zeros(ndof_max),                 # Ue_vec
        zeros(Int, ndof_max),            # dofs

        # Shape function outputs
        zeros(nen_max),                  # R
        zeros(nen_max, nsd),             # dR_dx
        zeros(nen_max, npd),             # dR_dXi
        zeros(nsd, npd),                 # dx_dXi
        zeros(nsd),                      # n_vec

        # Shape function intermediates
        zeros(npd),                      # Xi
        zeros(npd, npd),                 # dXi_dtildeXi
        zeros(nen_max),                  # N_num
        zeros(nen_max, npd),             # dN_dXi_num
        zeros(npd),                      # dsum_W
        zeros(nen_max),                  # W_buf
        zeros(nen_max),                  # col_tmp
        zeros(nen_max, nsd),             # Xcp

        # 1D basis buffers
        [zeros(p_max + 1) for _ in 1:npd],    # NN
        [zeros(p_max + 1) for _ in 1:npd],    # dNN_dXi

        # bspline buffers
        zeros(p_max + 1, p_max + 1),     # ndu
        zeros(p_max + 1),                # bsp_left
        zeros(p_max + 1),                # bsp_right
        zeros(2, p_max + 1),             # ders (n_deriv=1)
        zeros(2, p_max + 1),             # a_bsp

        # Strain-displacement temporaries
        zeros(nstrain, ndof_max),        # B0
        zeros(nsd, nen_max),             # dN_dX_T
        zeros(ndof_max, nstrain),        # BtD
        zeros(ndof_max, ndof_max),       # BtDB
        zeros(nstrain),                  # eps_vec
        zeros(nstrain),                  # sigma_vec
        zeros(ndof_max),                 # Bt_sigma

        # Jacobian intermediates
        zeros(nsd, nsd),                 # J_mat
        zeros(npd, nsd),                 # dXi_dx

        # Triplet output
        zeros(Int, max_triplets),        # I_buf
        zeros(Int, max_triplets),        # J_buf
        zeros(max_triplets),             # V_buf
        zeros(neq),                      # F_buf
        Ref(0),                          # triplet_pos
    )
end
