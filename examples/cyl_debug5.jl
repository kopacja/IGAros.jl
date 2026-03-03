# cyl_debug5.jl — test convergence for all degrees with various epss formulas
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, Printf

include("concentric_cylinders.jl")

println("=== epss = E / s_ang  (grows with refinement) ===")
for p in [2, 3, 4]
    @printf("p=%d:  ", p)
    errs_p = Float64[]
    for e in 0:5
        s_ang = (1/3) / 2^e
        epss_v = 100.0 / s_ang   # E / h — grows with refinement
        rel, abs_e = solve_cylinder(p, e; epss=epss_v, NQUAD=p+1)
        push!(errs_p, abs_e)
        @printf("%.3e  ", rel)
    end
    rates_p = [log2(errs_p[i]/errs_p[i+1]) for i in 1:5]
    @printf("  rates: %.2f %.2f %.2f %.2f %.2f\n", rates_p...)
end

println("\n=== epss = 1e4 (fixed) ===")
for p in [2, 3, 4]
    @printf("p=%d:  ", p)
    errs_p = Float64[]
    for e in 0:5
        rel, abs_e = solve_cylinder(p, e; epss=1e4, NQUAD=p+1)
        push!(errs_p, abs_e)
        @printf("%.3e  ", rel)
    end
    rates_p = [log2(errs_p[i]/errs_p[i+1]) for i in 1:5]
    @printf("  rates: %.2f %.2f %.2f %.2f %.2f\n", rates_p...)
end

println("\n=== epss = 1e6 (fixed) ===")
for p in [2, 3, 4]
    @printf("p=%d:  ", p)
    errs_p = Float64[]
    for e in 0:5
        rel, abs_e = solve_cylinder(p, e; epss=1e6, NQUAD=p+1)
        push!(errs_p, abs_e)
        @printf("%.3e  ", rel)
    end
    rates_p = [log2(errs_p[i]/errs_p[i+1]) for i in 1:5]
    @printf("  rates: %.2f %.2f %.2f %.2f %.2f\n", rates_p...)
end

# Check: does epss = E / s_ang correctly track the threshold E/h?
println("\n=== Threshold check: what is min working epss at each exp? ===")
for e in [1, 2, 3, 4]
    s_ang = (1/3) / 2^e
    E = 100.0
    h_over_E = E / s_ang
    @printf("exp=%d: E/s_ang=%.0f,  prev threshold=%.0f\n",
            e, h_over_E, E / (s_ang*2))
end
