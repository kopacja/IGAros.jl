# benchmark/run_cylinders.jl
#
# Concentric cylinders (2D): convergence + ε sweep + NQUAD sweep + moments.
#
# Usage:
#   julia -t auto --project=.. run_cylinders.jl

import Pkg; Pkg.instantiate(; allow_autoprecomp=false)

include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "..", "examples", "concentric_cylinders.jl"))

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEGREES   = [1, 2, 3, 4]
const EXP_RANGE = Dict(1 => 0:5, 2 => 0:5, 3 => 0:4, 4 => 0:3)
const E_VAL     = 100.0
const NU_VAL    = 0.3
const R_I       = 1.0
const R_C       = 1.5
const R_O       = 2.0
const P_O       = 1.0
const EPS_RANGE = 10.0 .^ (-2:0.5:7)
const NQUAD_RANGE = 1:8
const EPS_SWEEP_EXP  = 3
const NQUAD_SWEEP_EXP = 2

# ═══════════════════════════════════════════════════════════════════════════════
# Solver adapter
# ═══════════════════════════════════════════════════════════════════════════════

function cyl_solve(p, exp_level;
                   formulation, strategy, epss,
                   NQUAD_mortar::Int = (p == 1 ? 3 : 10),
                   max_dof::Int = typemax(Int),
                   kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss

    if p == 1
        s_rel, s_abs, d_rel, d_abs, en_rel, en_abs = solve_cylinder_p1(
            exp_level; epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)
    else
        s_rel, s_abs, d_rel, d_abs, en_rel, en_abs = solve_cylinder(
            p, exp_level; epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)
    end

    return BenchmarkResult(l2_disp=d_abs, energy=en_abs, l2_stress=s_abs)
end

function h_fn(p, exp)
    return p == 1 ? 0.5 / 2^(exp+1) : 0.5 / 2^exp
end

function eps_fn(p, exp)
    return p == 1 ? 1e2 : 1e6
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh setup for force moments
# ═══════════════════════════════════════════════════════════════════════════════

function cyl_setup(p, exp_level; kwargs...)
    nsd = 2; npd = 2; npc = 2

    if p == 1
        n_ang_p2 = 3 * 2^exp_level
        n_ang_p1 = 6 * 2^exp_level
        n_rad = 2 * 2^exp_level
        B, P, p_mat, n_mat, KV = cylinder_geometry_direct_p1(
            n_ang_p1, n_ang_p2, n_rad; r_i=R_I, r_c=R_C, r_o=R_O)
        _, nnp_v, _ = patch_metrics(npc, npd, p_mat, n_mat)
    else
        B0, P = cylinder_geometry(p; r_i=R_I, r_c=R_C, r_o=R_O)
        p_mat = fill(p, npc, npd)
        n_mat = fill(p + 1, npc, npd)
        KV = generate_knot_vectors(npc, npd, p_mat, n_mat)

        n_ang = 2^exp_level; n_rad = 2^exp_level
        n_ang_inner = 2 * n_ang
        u_ang_o = Float64[i/n_ang for i in 1:n_ang-1]
        u_ang_i = Float64[i/n_ang_inner for i in 1:n_ang_inner-1]
        u_rad = Float64[i/n_rad for i in 1:n_rad-1]
        kref_data = Vector{Float64}[
            vcat([1.0, 1.0], u_ang_i), vcat([1.0, 2.0], u_rad),
            vcat([2.0, 1.0], u_ang_o), vcat([2.0, 2.0], u_rad)]
        n_mat, _, KV, B, P = krefinement(nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data)
        _, nnp_v, _ = patch_metrics(npc, npd, p_mat, n_mat)
    end

    pairs_full = [InterfacePair(1, 3, 2, 1), InterfacePair(2, 1, 1, 3)]
    pairs_sp   = [InterfacePair(1, 3, 2, 1)]

    return (p_mat=p_mat, n_mat=n_mat, KV=KV, P=P, B=B,
            nnp=nnp_v, nsd=nsd, npd=npd,
            pairs_full=pairs_full, pairs_sp=pairs_sp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

outdir = setup_output_dir("cylinders", "systematic")
t_start = time()

println("=" ^ 90)
println("  Concentric cylinders: systematic benchmark")
println("  Threads: $(Threads.nthreads()), Host: $(gethostname()), Date: $(now())")
println("=" ^ 90)

# ── 1. Convergence ───────────────────────────────────────────────────────────
println("\n>>> Convergence study...")
conv = run_convergence(cyl_solve, ALL_6_METHODS, DEGREES, EXP_RANGE;
    h_fn=h_fn, eps_fn=eps_fn, max_dof=typemax(Int))
write_csv(joinpath(outdir, "convergence.csv"), conv)

# ── 2. Force moments (p=2,3 at exp=2) ────────────────────────────────────────
println("\n>>> Force moments...")
moments = run_moments(cyl_setup, ALL_6_METHODS, [2, 3], [2];
    dims=[2], NQUAD_mortar_fn=(p, e) -> 10)
write_csv(joinpath(outdir, "moments.csv"), moments)

# ── 3. ε sweep ───────────────────────────────────────────────────────────────
println("\n>>> ε sweep...")
eps_methods = [m for m in ALL_6_METHODS if !is_single_pass(m)]
eps_rows = run_eps_sweep(cyl_solve, eps_methods, [1, 2, 3, 4], EPS_SWEEP_EXP, EPS_RANGE;
    max_dof=typemax(Int))
write_csv(joinpath(outdir, "eps_sweep.csv"), eps_rows)

# ── 4. NQUAD sweep ──────────────────────────────────────────────────────────
println("\n>>> NQUAD sweep...")
nq_rows = run_nquad_sweep(cyl_solve, NQUAD_METHODS, [2, 3, 4], NQUAD_SWEEP_EXP, NQUAD_RANGE;
    eps_fn=eps_fn)
write_csv(joinpath(outdir, "nquad_sweep.csv"), nq_rows)

# ── 5. Meta ───────────────────────────────────────────────────────────────────
write_meta(joinpath(outdir, "meta.toml");
    description="Concentric cylinders: convergence + ε sweep + NQUAD sweep + moments",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "r_i" => R_I, "r_c" => R_C, "r_o" => R_O,
                "p_o" => P_O, "degrees" => DEGREES),
    outputs=["convergence.csv", "moments.csv", "eps_sweep.csv", "nquad_sweep.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
