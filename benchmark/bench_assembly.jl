#!/usr/bin/env julia
# Benchmark: threaded stiffness matrix assembly on concentric cylinders
#
# Usage:  julia -t <N> --project=.. bench_assembly.jl
# SLURM:  see bench_assembly.sh

using IGAros
using Printf

include(joinpath(@__DIR__, "..", "examples", "concentric_cylinders.jl"))

function setup_problem(p_ord, exp_level)
    nsd=2; npd=2; ned=2; npc=2; thickness=1.0; NQUAD=p_ord+1
    B0, P = cylinder_geometry(p_ord)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord+1, npc, npd)
    KV = generate_knot_vectors(npc, npd, p_mat, n_mat)

    s_ang = (1/3)/2^exp_level; s_ang_nc = s_ang/2; s_rad = (1/2)/2^exp_level
    u_ang    = collect(s_ang   :s_ang   :1-s_ang/2)
    u_ang_nc = collect(s_ang_nc:s_ang_nc:1-s_ang_nc/2)
    u_rad    = collect(s_rad   :s_rad   :1-s_rad/2)
    kref = Vector{Float64}[
        vcat([1.0,1.0], u_ang_nc), vcat([2.0,1.0], u_ang),
        vcat([1.0,2.0], u_rad),    vcat([2.0,2.0], u_rad)]
    B0h = copy(B0); B0h[P[1],2] .+= 1000.0
    n_mat2, _, KV2, Bh, P2 = krefinement(nsd, npd, npc, n_mat, p_mat, KV, B0h, P, kref)
    B = copy(Bh)
    for i in axes(B,1); B[i,2] > 100 && (B[i,2] -= 1000); end
    ncp = size(B,1)
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat2)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat2, nel, nnp, nen)
    INC = [build_inc(n_mat2[pc,:]) for pc in 1:npc]
    dBC = [1 4 2 1 2; 2 2 2 1 2]
    bc = dirichlet_bc_control_points(p_mat, n_mat2, KV2, P2, npd, nnp, ned, dBC)
    neq, ID = build_id(bc, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P2)
    mats = [LinearElastic(100.0, 0.3, :plane_strain) for _ in 1:npc]
    Ub = zeros(ncp, nsd)
    return (; npc, nsd, npd, ned, neq, p_mat, n_mat2, KV2, P2, B, Ub,
              nen, nel, IEN, INC, LM, mats, NQUAD, thickness)
end

function bench(p_ord, exp_level; nreps=10)
    s = setup_problem(p_ord, exp_level)
    # Warmup
    build_stiffness_matrix(s.npc, s.nsd, s.npd, s.ned, s.neq,
        s.p_mat, s.n_mat2, s.KV2, s.P2, s.B, s.Ub,
        s.nen, s.nel, s.IEN, s.INC, s.LM, s.mats, s.NQUAD, s.thickness)

    times = Float64[]
    for _ in 1:nreps
        t = @elapsed build_stiffness_matrix(s.npc, s.nsd, s.npd, s.ned, s.neq,
            s.p_mat, s.n_mat2, s.KV2, s.P2, s.B, s.Ub,
            s.nen, s.nel, s.IEN, s.INC, s.LM, s.mats, s.NQUAD, s.thickness)
        push!(times, t)
    end
    nel_total = sum(s.nel)
    median_ms = sort(times)[div(nreps,2)+1] * 1000
    return nel_total, median_ms
end

function main()
    nt = Threads.nthreads()
    println("=" ^ 60)
    @printf("IGAros stiffness assembly benchmark — %d thread(s)\n", nt)
    println("=" ^ 60)
    @printf("%-6s %-6s %8s %12s\n", "p", "exp", "elements", "median [ms]")
    println("-" ^ 40)

    cases = [
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 3),
    ]
    for (p, exp) in cases
        nel, ms = bench(p, exp)
        @printf("%-6d %-6d %8d %12.2f\n", p, exp, nel, ms)
        flush(stdout)
    end
    println("=" ^ 60)
end

main()
