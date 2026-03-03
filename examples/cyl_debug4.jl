# cyl_debug4.jl — find correct epss scaling for cylinder
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, Printf

include("concentric_cylinders.jl")

p_ord = 2

println("=== Fixed epss sweep (exp=1..4, p=2) ===")
for epss_val in [1e8, 1e6, 1e4, 1e2, 1e1, 1.0, 0.1, 1e-2, 1e-4]
    @printf("epss=%8.2e:  ", epss_val)
    for e in 1:4
        rel, _ = solve_cylinder(p_ord, e; epss=Float64(epss_val))
        @printf("%.3e  ", rel)
    end
    # Print log2 convergence rate (last 3 levels)
    errs = [solve_cylinder(p_ord, e; epss=Float64(epss_val))[2] for e in 1:4]
    rates = [log2(errs[i]/errs[i+1]) for i in 1:3]
    @printf("  rates: %.2f %.2f %.2f\n", rates...)
end

println("\n=== Adaptive epss = E * s_ang (plate-like formula) ===")
for e in 0:5
    s_ang = (1/3) / 2^e
    epss_v = 100.0 * s_ang    # E * h
    rel, abs_e = solve_cylinder(p_ord, e; epss=epss_v)
    @printf("exp=%d  epss=%.3e  rel=%.4e  abs=%.4e\n", e, epss_v, rel, abs_e)
end
errs = [solve_cylinder(p_ord, e; epss=100.0*(1/3)/2^e)[2] for e in 0:5]
rates = [log2(errs[i]/errs[i+1]) for i in 1:5]
@printf("  rates: %.2f %.2f %.2f %.2f %.2f\n", rates...)

println("\n=== Adaptive epss = s_ang (no E) ===")
for e in 0:5
    s_ang = (1/3) / 2^e
    epss_v = s_ang
    rel, abs_e = solve_cylinder(p_ord, e; epss=epss_v)
    @printf("exp=%d  epss=%.3e  rel=%.4e  abs=%.4e\n", e, epss_v, rel, abs_e)
end

println("\n=== All degrees with epss = E * s_ang ===")
for p in [2, 3, 4]
    @printf("p=%d:  ", p)
    errs_p = Float64[]
    for e in 0:5
        s_ang = (1/3) / 2^e
        epss_v = 100.0 * s_ang
        rel, abs_e = solve_cylinder(p, e; epss=epss_v, NQUAD=p+1)
        push!(errs_p, abs_e)
        @printf("%.3e  ", rel)
    end
    rates_p = [log2(errs_p[i]/errs_p[i+1]) for i in 1:5]
    @printf("  rates: %.2f %.2f %.2f %.2f %.2f\n", rates_p...)
end
