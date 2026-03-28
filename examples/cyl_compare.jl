import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))
include(joinpath(@__DIR__, "test_tm_variants.jl"))

println("\n  CONVERGENCE: Z_II (Current) vs DualZ vs Z_I — cylinders p=2, eps=1e6")
println("  " * "="^80)
@printf "  %3s | %12s %6s | %12s %6s | %12s %6s\n" "exp" "Z_II" "rate" "DualZ" "rate" "Z_I" "rate"
@printf "  %3s-|-%12s-%6s-|-%12s-%6s-|-%12s-%6s\n" "---" "------------" "------" "------------" "------" "------------" "------"

prev = [NaN, NaN, NaN]
for ex in 0:5
    errs = Float64[]
    for form in [TwinMortarFormulation(), DualPassZ_TM(), CorrectedTM()]
        try
            e, _, _, _, _, _ = solve_cylinder(2, ex; epss=1e6, formulation=form)
            push!(errs, e)
        catch err
            @warn "form=$(typeof(form)) exp=$ex: $err"
            push!(errs, NaN)
        end
    end
    rates = Float64[]
    for i in 1:3
        if ex > 0 && !isnan(prev[i]) && !isnan(errs[i]) && errs[i] > 0
            push!(rates, log2(prev[i] / errs[i]))
        else
            push!(rates, NaN)
        end
    end
    @printf("  %3d | %12.4e %6s | %12.4e %6s | %12.4e %6s\n",
        ex,
        errs[1], isnan(rates[1]) ? "" : @sprintf("%.2f", rates[1]),
        errs[2], isnan(rates[2]) ? "" : @sprintf("%.2f", rates[2]),
        errs[3], isnan(rates[3]) ? "" : @sprintf("%.2f", rates[3]))
    prev .= errs
end

# Z symmetry check
println("\n  Z SYMMETRY CHECK at exp=2, p=2, eps=1e6:")
for (label, form) in [("Z_II", TwinMortarFormulation()), ("DualZ", DualPassZ_TM()), ("Z_I", CorrectedTM())]
    p_ord = 2; exp_level = 2
    nsd = 2; npd = 2; ned = 2; npc = 2
    B0, P = cylinder_geometry(p_ord; r_i=1.0, r_c=1.5, r_o=2.0)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV = generate_knot_vectors(npc, npd, p_mat, n_mat)
    s_ang = 1.0 / (3 * 2^exp_level)
    s_ang_nc = 1.0 / (6 * 2^exp_level)
    s_rad = 0.5 / 2^exp_level
    u_ang = collect(s_ang:s_ang:1-s_ang/2)
    u_ang_nc = collect(s_ang_nc:s_ang_nc:1-s_ang_nc/2)
    u_rad = collect(s_rad:s_rad:1-s_rad/2)
    kref_data = Vector{Float64}[
        vcat([1.0,1.0], u_ang_nc), vcat([2.0,1.0], u_ang),
        vcat([1.0,2.0], u_rad), vcat([2.0,2.0], u_rad)]
    B0h = copy(B0); B0h[P[1],2] .+= 1000.0
    n_mat_ref, _, KV_ref, B_hack, P_ref = krefinement(nsd, npd, npc, n_mat, p_mat, KV, B0h, P, kref_data)
    B_ref = copy(B_hack)
    for i in axes(B_ref,1)
        if B_ref[i,2] > 100.0
            B_ref[i,2] -= 1000.0
        end
    end
    ncp = size(B_ref,1)
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc,:]) for pc in 1:npc]
    bc_per_dof = zeros(Int, ned, ncp)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    pairs = [InterfacePair(1,3,2,1), InterfacePair(2,1,1,3)]
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, form)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        ID, nnp, ned, nsd, npd, neq, 12, 1e6, ElementBasedIntegration(), form)
    Zd = Matrix(Z)
    asym = norm(Zd - Zd', Inf) / max(norm(Zd, Inf), 1e-30)
    eigs_sym = eigvals(Symmetric(0.5*(Zd + Zd')))
    mine = minimum(eigs_sym)
    maxe = maximum(eigs_sym)
    @printf("  %-6s: ||Z-Z^T||/||Z|| = %.2e,  eig range [%+.2e, %+.2e]\n", label, asym, mine, maxe)
end
