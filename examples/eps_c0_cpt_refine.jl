"""
3D CPT: Does the non-integer ratio discretization floor drop with refinement?
Compare 2:1 vs 3:2 across exp=0,1,2 with full ε-sweeps.
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

function run_cpt_refine()
    p = 2; E = 1e3

    ratios = [
        (2, 2, "1:1 (conf)", true),
        (4, 2, "2:1", false),
        (3, 2, "3:2", false),
        (5, 3, "5:3", false),
    ]

    println("═"^78)
    println("  3D CPT: ε-sweep × refinement for integer vs non-integer ratios")
    println("  p=$p, E=$E")
    println("═"^78)

    for (nxl, nxu, label, conf) in ratios
        println("\n\n── $label (base: $nxl:$nxu) ──")
        for exp in 0:2
            ns = nxl * 2^exp; nm = nxu * 2^exp
            h = 1.0 / (nm)  # approximate master element size
            println("\n  exp=$exp, n_slave=$ns, n_master=$nm, h≈$(round(h,digits=4))")
            @printf("  %10s  %14s  %14s\n", "ε", "RMS σ_zz", "RMS×ε")
            @printf("  %s\n", "─"^40)
            for k in 0:7
                eps = 10.0^k
                rms, _ = check_cpt(p, exp; epss=eps, E=E,
                    formulation=TwinMortarFormulation(),
                    strategy=ElementBasedIntegration(),
                    conforming=conf,
                    n_x_lower_base=nxl, n_x_upper_base=nxu,
                    n_y_lower_base=nxl, n_y_upper_base=nxu)
                @printf("  %10.0e  %14.4e  %14.4e\n", eps, rms, rms*eps)
            end
        end
    end
end

run_cpt_refine()
