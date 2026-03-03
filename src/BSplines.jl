# BSplines.jl
# Core B-spline algorithms: knot span search, basis functions, derivatives.
# Ported from MATLAB: FindSpan.m, BasisFuns.m, Bspline_basis_and_deriv.m
# Reference: "The NURBS Book" by Piegl & Tiller (1997), Algorithms A2.1, A2.2, A2.3

"""
    find_span(n_basis, p, u, kv) -> Int

Find the knot span index for parameter `u` in knot vector `kv`.

- `n_basis`: number of basis functions minus 1 (= length(kv) - p - 2)
- `p`: polynomial degree
- `u`: parametric coordinate
- `kv`: knot vector

Returns index `i` such that `kv[i] ≤ u < kv[i+1]` (1-based, p+1 ≤ i ≤ n_basis+1).

Algorithm A2.1 from The NURBS Book.
"""
function find_span(n_basis::Int, p::Int, u::Float64, kv::Vector{Float64})::Int
    # Special case: u at the right endpoint
    if u >= kv[n_basis + 2]
        return n_basis + 1  # 1-based
    end
    if u <= kv[p + 1]
        return p + 1  # 1-based
    end

    low  = p + 1
    high = n_basis + 2
    mid  = (low + high) ÷ 2

    while u < kv[mid] || u >= kv[mid + 1]
        if u < kv[mid]
            high = mid
        else
            low = mid
        end
        mid = (low + high) ÷ 2
    end
    return mid
end

"""
    basis_funs(i, u, p, kv) -> Vector{Float64}

Compute the p+1 nonvanishing B-spline basis functions at parameter `u`.

- `i`: knot span index (from `find_span`)
- `u`: parametric coordinate
- `p`: polynomial degree
- `kv`: knot vector

Returns `N[1:p+1]` where `N[j]` = N_{i-p+j-1, p}(u).

Algorithm A2.2 from The NURBS Book.
"""
function basis_funs(i::Int, u::Float64, p::Int, kv::Vector{Float64})::Vector{Float64}
    N    = zeros(p + 1)
    left  = zeros(p + 1)
    right = zeros(p + 1)

    N[1] = 1.0
    for j in 1:p
        left[j + 1]  = u - kv[i + 1 - j]
        right[j + 1] = kv[i + j] - u
        saved = 0.0
        for r in 0:j-1
            temp    = N[r + 1] / (right[r + 2] + left[j - r + 1])
            N[r + 1] = saved + right[r + 2] * temp
            saved    = left[j - r + 1] * temp
        end
        N[j + 1] = saved
    end
    return N
end

"""
    bspline_basis_and_deriv(i, u, p, n_deriv, kv) -> Matrix{Float64}

Compute B-spline basis functions and their derivatives up to order `n_deriv`.

- `i`: knot span index (from `find_span`)
- `u`: parametric coordinate
- `p`: polynomial degree
- `n_deriv`: number of derivatives requested (≤ p)
- `kv`: knot vector

Returns matrix `ders` of size `(n_deriv+1) × (p+1)` where:
  - `ders[k+1, j+1]` = k-th derivative of the j-th nonvanishing basis function

Algorithm A2.3 from The NURBS Book.
"""
function bspline_basis_and_deriv(
    i::Int, u::Float64, p::Int, n_deriv::Int, kv::Vector{Float64}
)::Matrix{Float64}

    ndu   = zeros(p + 1, p + 1)
    left  = zeros(p + 1)
    right = zeros(p + 1)
    ders  = zeros(n_deriv + 1, p + 1)
    a     = zeros(2, p + 1)

    ndu[1, 1] = 1.0
    for j in 1:p
        left[j + 1]  = u - kv[i + 1 - j]
        right[j + 1] = kv[i + j] - u
        saved = 0.0
        for r in 0:j-1
            # Lower triangle
            ndu[j + 1, r + 1] = right[r + 2] + left[j - r + 1]
            temp = ndu[r + 1, j] / ndu[j + 1, r + 1]
            # Upper triangle
            ndu[r + 1, j + 1] = saved + right[r + 2] * temp
            saved = left[j - r + 1] * temp
        end
        ndu[j + 1, j + 1] = saved
    end

    # Load basis functions into first row of ders
    for j in 0:p
        ders[1, j + 1] = ndu[j + 1, p + 1]
    end

    # Compute derivatives
    for r in 0:p
        s1 = 1; s2 = 2
        a[1, 1] = 1.0
        for k in 1:n_deriv
            d = 0.0
            rk = r - k; pk = p - k
            if r >= k
                a[s2, 1] = a[s1, 1] / ndu[pk + 2, rk + 1]
                d = a[s2, 1] * ndu[rk + 1, pk + 1]
            end
            j1 = (rk >= -1) ? 1 : -rk
            j2 = (r - 1 <= pk) ? k - 1 : p - r
            for j in j1:j2
                a[s2, j + 1] = (a[s1, j + 1] - a[s1, j]) / ndu[pk + 2, rk + j + 1]
                d += a[s2, j + 1] * ndu[rk + j + 1, pk + 1]
            end
            if r <= pk
                a[s2, k + 1] = -a[s1, k] / ndu[pk + 2, r + 1]
                d += a[s2, k + 1] * ndu[r + 1, pk + 1]
            end
            ders[k + 1, r + 1] = d
            # Swap rows
            s1, s2 = s2, s1
        end
    end

    # Multiply through by the correct factors
    r = p
    for k in 1:n_deriv
        for j in 0:p
            ders[k + 1, j + 1] *= r
        end
        r *= (p - k)
    end

    return ders
end
