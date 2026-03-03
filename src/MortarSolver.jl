# MortarSolver.jl
# KKT system solver for Twin Mortar tying.

"""
    solve_mortar(K, C, Z, F) -> (U, Lambda)

Solve the augmented KKT system arising from Twin Mortar mesh tying:

    [K    C ] [U]   [F]
    [C^T  -Z] [λ] = [0]

- `K` (neq × neq): IGA stiffness (sparse, with Dirichlet BCs already enforced)
- `C` (neq × 2·nlm): coupling matrix from `build_mortar_coupling`
- `Z` (2·nlm × 2·nlm): stabilization matrix from `build_mortar_coupling`
- `F` (neq,): load vector (with Dirichlet corrections applied)

Returns:
- `U` (neq,): displacement solution
- `Lambda` (2·nlm,): Lagrange multiplier solution [λ_n; λ_t]
"""
function solve_mortar(
    K::SparseMatrixCSC{Float64, Int},
    C::SparseMatrixCSC{Float64, Int},
    Z::SparseMatrixCSC{Float64, Int},
    F::Vector{Float64}
)::Tuple{Vector{Float64}, Vector{Float64}}

    nlm2 = size(Z, 1)
    A    = [K C; C' -Z]
    b    = [F; zeros(Float64, nlm2)]
    x    = A \ b

    neq = length(F)
    return x[1:neq], x[neq+1:end]
end
