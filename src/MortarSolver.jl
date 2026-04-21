# MortarSolver.jl
# KKT system solver for Twin Mortar tying.

"""
    solve_mortar(K, C, Z, F; g=zeros(size(Z,1))) -> (U, Lambda)

Solve the augmented KKT system arising from Twin Mortar mesh tying:

    [K    C ] [U]   [F]
    [C^T  +Z] [λ] = [g]

- `K` (neq × neq): IGA stiffness (sparse, with Dirichlet BCs already enforced)
- `C` (neq × n_lam): coupling matrix from `build_mortar_coupling`
- `Z` (n_lam × n_lam): stabilization matrix (positive definite, stored as +ε·[D̄,M̄;...])
- `F` (neq,): load vector (with Dirichlet corrections applied)
- `g` (n_lam,): constraint RHS (default zeros)

Returns:
- `U` (neq,): displacement solution
- `Lambda` (n_lam,): Lagrange multiplier solution
"""
function solve_mortar(
    K::SparseMatrixCSC{Float64, Int},
    C::SparseMatrixCSC{Float64, Int},
    Z::SparseMatrixCSC{Float64, Int},
    F::Vector{Float64};
    g::Vector{Float64} = zeros(Float64, size(Z, 1))
)::Tuple{Vector{Float64}, Vector{Float64}}

    A = [K C; C' Z]
    b = [F; g]
    x = A \ b

    neq = length(F)
    return x[1:neq], x[neq+1:end]
end
