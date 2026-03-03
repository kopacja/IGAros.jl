# StrainDisplacement.jl
# B₀ strain-displacement matrix (small-deformation, Voigt notation).
# Ported from MATLAB: getMatrixB0.m

"""
    strain_displacement_matrix(nsd, nen, dN_dX) -> Matrix{Float64}

Build the Voigt-notation strain-displacement (B₀) matrix.

- `nsd`: spatial dimension (2 or 3)
- `nen`: number of element nodes
- `dN_dX`: (nsd × nen) matrix of shape-function gradients w.r.t. physical coordinates

Returns:
- 3×(2·nen) for nsd=2  [ε_xx, ε_yy, γ_xy]
- 6×(3·nen) for nsd=3  [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]

Strain: ε = B₀ · u_vec  (u_vec assembled node-major: [u₁ᵢ, u₁ⱼ, u₂ᵢ, u₂ⱼ, ...])

Ported from getMatrixB0.m.
"""
function strain_displacement_matrix(
    nsd::Int, nen::Int, dN_dX::AbstractMatrix{Float64}
)::Matrix{Float64}

    if nsd == 2
        B = zeros(3, 2 * nen)
        for a in 1:nen
            col_x = 2a - 1  # x-displacement DOF
            col_y = 2a       # y-displacement DOF
            B[1, col_x] = dN_dX[1, a]   # ε_xx = ∂N/∂x · u_x
            B[2, col_y] = dN_dX[2, a]   # ε_yy = ∂N/∂y · u_y
            B[3, col_x] = dN_dX[2, a]   # γ_xy = ∂N/∂y · u_x + ∂N/∂x · u_y
            B[3, col_y] = dN_dX[1, a]
        end
        return B

    elseif nsd == 3
        B = zeros(6, 3 * nen)
        for a in 1:nen
            cx = 3a - 2; cy = 3a - 1; cz = 3a
            B[1, cx] = dN_dX[1, a]   # ε_xx
            B[2, cy] = dN_dX[2, a]   # ε_yy
            B[3, cz] = dN_dX[3, a]   # ε_zz
            B[4, cx] = dN_dX[2, a]   # γ_xy
            B[4, cy] = dN_dX[1, a]
            B[5, cy] = dN_dX[3, a]   # γ_yz
            B[5, cz] = dN_dX[2, a]
            B[6, cx] = dN_dX[3, a]   # γ_zx
            B[6, cz] = dN_dX[1, a]
        end
        return B

    else
        error("strain_displacement_matrix: unsupported nsd = $nsd")
    end
end
