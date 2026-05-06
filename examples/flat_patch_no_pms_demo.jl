# examples/flat_patch_no_pms_demo.jl
#
# §3.3 demonstration: drop the master-master cross-mass block P^(ms) from Z
# in the Twin Mortar formulation and observe the patch test fail on a
# non-conforming flat interface.  Standalone driver — runs locally in a few
# seconds, writes a focused comparison CSV without touching the
# `results/flat_patch_test/current` symlink (which tracks the full factorial
# benchmark).  The same demonstration is also produced as `no_pms_demo.csv`
# by the canonical `benchmark/run_flat_patch_test.jl` cluster pipeline.
#
# Output: results/flat_patch_test/<YYYY-MM-DD>_no_pms_demo/
#   ├── no_pms_demo.csv   — TME and TME_NoP × p=1..4
#   └── meta.toml         — run provenance
#
# Usage:
#   julia --project=.. examples/flat_patch_no_pms_demo.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf, Dates

include(joinpath(@__DIR__, "..", "benchmark", "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "cz_cancellation.jl"))

# ─── Configuration (matches benchmark/run_flat_patch_test.jl) ────────────────

const DEGREES  = [1, 2, 3, 4]
const N_S      = 4
const N_M      = 7
const E_VAL    = 1000.0
const NU_VAL   = 0.0
const EPSS_DEF = 10.0

const NO_PMS_DEMO_METHODS = [
    MethodConfig("TME",     TwinMortarFormulation(),            ElementBasedIntegration()),
    MethodConfig("TME_NoP", TwinMortarFormulationNoCrossMass(), ElementBasedIntegration()),
]

# ─── Solver adapter (mirrors run_flat_patch_test.jl::flat_solve) ─────────────

function flat_solve(p, _exp_unused;
                    formulation, strategy, epss,
                    NQUAD_mortar::Int = p + 2,
                    max_dof::Int = typemax(Int),
                    kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss
    r = flat_patch_test_hp(p, N_S, N_M;
            E=E_VAL, nu=NU_VAL, epss=eps_use,
            NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)

    NQUAD_vol = p + 1
    d_abs, _ = l2_disp_error_flat(r.U, r.ID, r.npc, r.nsd, r.npd,
                                  r.p_mat, r.n_mat, r.KV, r.P, r.B,
                                  r.nen, r.nel, r.IEN, r.INC,
                                  NQUAD_vol, E_VAL)

    kappa = safe_kappa(r.K_bc, r.C, r.Z; max_dof=max_dof)

    lam_inf_s = if formulation isa SinglePassFormulation
        norm(r.Lambda, Inf)
    else
        half = length(r.Lambda) ÷ 2
        norm(r.Lambda[1:half], Inf)
    end
    lam_err = abs(lam_inf_s - 1.0)

    return BenchmarkResult(l2_disp=d_abs, energy=NaN, l2_stress=NaN,
                           kappa=kappa, lam_err=lam_err,
                           ndof=r.neq, n_lam=size(r.C, 2))
end

# ─── Run + write (no symlink update; standalone results dir) ─────────────────

t_start = time()

base   = joinpath(@__DIR__, "..", "..", "results", "flat_patch_test")
mkpath(base)
stamp  = Dates.format(now(), "yyyy-mm-dd")
outdir = joinpath(base, "$(stamp)_no_pms_demo")
mkpath(outdir)

println("=" ^ 90)
println("  §3.3 P^(ms) drop demonstration")
println("  TME (full Z) vs. TME_NoP (master-master block omitted)")
println("  n_s=$(N_S), n_m=$(N_M), ε=$(EPSS_DEF), p=1..4")
println("=" ^ 90)

rows = run_convergence(flat_solve, NO_PMS_DEMO_METHODS, DEGREES, [0];
    h_fn   = (p, e) -> 1.0 / N_S,
    eps_fn = (p, e) -> EPSS_DEF,
    max_dof = typemax(Int))

write_csv(joinpath(outdir, "no_pms_demo.csv"), rows)

write_meta(joinpath(outdir, "meta.toml");
    description="§3.3 P^(ms) drop demonstration: TME vs TME_NoP on non-conforming flat patch",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "n_s" => N_S, "n_m" => N_M,
                "epss" => EPSS_DEF, "degrees" => DEGREES,
                "n_quad_mortar" => "p+2"),
    outputs=["no_pms_demo.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
