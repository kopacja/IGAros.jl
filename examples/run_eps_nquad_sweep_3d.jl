# run_eps_nquad_sweep_3d.jl — 2D sweep: ε × NQUAD for the 3D flat patch test
#
# For each (method, p, slave), sweeps both ε and NQUAD_mortar to produce a
# displacement error surface e(ε, nq).  Shows that TME error is ε-controlled
# (iso-error lines are horizontal) while SPME error is NQUAD-controlled.
#
# Output: eps_nquad_2d.csv

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# Skip the study section in run_flat_patchtest_3d.jl
ENV["TM_SKIP_STUDY"] = "1"
include(joinpath(@__DIR__, "run_flat_patchtest_3d.jl"))
delete!(ENV, "TM_SKIP_STUDY")

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════

results_dir = get(ENV, "TM_RESULTS_DIR",
    joinpath(@__DIR__, "..", "..", "results", "flat_patch_test_3d",
             Dates.format(now(), "yyyy-mm-dd") * "_benchmark"))
mkpath(results_dir)

methods_2d = [
    ("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    ("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    ("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
]

slave_choices = [:upper]
slave_labels = Dict(:lower => "sL", :upper => "sU")
p_range = [1, 2]

epss_range = 10.0 .^ (-1:1.0:17)
nquad_range = [2, 3, 4, 5, 7, 10, 15, 20]

println("=== 2D sweep: ε × NQUAD ===")
println("  methods:  ", join([m[1] for m in methods_2d], ", "))
println("  p range:  ", p_range)
println("  ε range:  ", length(epss_range), " values (0.1 to 1e17)")
println("  nq range: ", nquad_range)
println("  output:   ", joinpath(results_dir, "eps_nquad_2d.csv"))
println()

open(joinpath(results_dir, "eps_nquad_2d.csv"), "w") do io
    println(io, "method,slave,p,eps,nquad,l2_disp,lam_err,wall_s")

    for sf in slave_choices, p in p_range, (mname, form, strat) in methods_2d
        tag = mname * "_" * slave_labels[sf]
        @printf("  %s p=%d: ", tag, p); flush(stdout)

        for nq in nquad_range
            for eps_val in epss_range
                eps_use = form isa SinglePassFormulation ? 0.0 : eps_val
                t0 = time()
                try
                    r = solve_farah3d(p; NQUAD_mortar=nq, epss=eps_use,
                        formulation=form, strategy=strat, slave_first=sf)
                    wall = time() - t0
                    @printf(io, "%s,%s,%d,%.6e,%d,%.6e,%.6e,%.6e\n",
                            mname, slave_labels[sf], p, eps_val, nq,
                            r.l2_disp, r.lam_err, wall)
                catch
                    @printf(io, "%s,%s,%d,%.6e,%d,NaN,NaN,%.6e\n",
                            mname, slave_labels[sf], p, eps_val, nq, time() - t0)
                end
            end
            @printf("."); flush(stdout)
        end
        println(" done")
    end
end

println("\n2D sweep complete. Results saved to:\n  ", joinpath(results_dir, "eps_nquad_2d.csv"))
