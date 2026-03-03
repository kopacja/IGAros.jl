# Solver.jl
# Linear system solve and simple Newton iteration utilities.

"""
    linear_solve(K, F) -> Vector{Float64}

Solve the symmetric positive definite system K·u = F.
"""
function linear_solve(
    K::SparseMatrixCSC{Float64, Int},
    F::Vector{Float64}
)::Vector{Float64}
    return K \ F
end
