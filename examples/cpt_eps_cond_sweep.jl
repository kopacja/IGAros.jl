"""
3D Curved Patch Test: ε-sweep with condition number analysis.
Analogous to the 2D flat patch test Figures 4 & 5 (patch_eps, patch_cond).

For each p=2,3,4 at a fixed mesh (exp=1), sweep ε and report:
  - RMS stress error (σ_zz)
  - condition number κ(A) of the augmented KKT system
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

function cpt_eps_cond_sweep(;
    degrees = [2, 3, 4],
    exp_level = 1,
    eps_range = [10.0^k for k in -2:0.5:7],
    E = 1e5, nu = 0.3,
    conforming = false,
    n_x_lower_base = 3, n_x_upper_base = 2,
    n_y_lower_base = 3, n_y_upper_base = 2,
)
    println("═"^80)
    println("  3D Curved Patch Test: ε-sweep + conditioning")
    println("  exp=$exp_level, E=$E, non-conforming $(n_x_lower_base):$(n_x_upper_base)")
    println("═"^80)

    for p in degrees
        println("\n── p = $p ──")
        @printf("  %12s  %14s  %14s  %8s  %8s\n",
                "ε", "RMS σ_zz", "κ(A)", "neq", "nlm")
        @printf("  %s\n", "─"^62)

        for eps_val in eps_range
            try
                d = _cpt_solve_diag(p, exp_level;
                    epss=eps_val, E=E, nu=nu, conforming=conforming,
                    formulation=TwinMortarFormulation(),
                    strategy=ElementBasedIntegration(),
                    n_x_lower_base=n_x_lower_base, n_x_upper_base=n_x_upper_base,
                    n_y_lower_base=n_y_lower_base, n_y_upper_base=n_y_upper_base)

                # Stress error
                rms_zz, _, _, _ = stress_error_cpt(
                    d.U, d.ID, 2, 3, 3, d.p_mat, d.n_mat_ref,
                    d.KV_ref, d.P_ref, d.B_ref,
                    d.nen, d.nel, d.IEN, d.INC, E, nu, d.NQUAD_use)

                # Condition number of KKT system [K C; C' -Z]
                K_dense = Matrix(d.K)
                C_dense = Matrix(d.C)
                Z_dense = Matrix(d.Z)
                neq = size(K_dense, 1)
                nlm = size(Z_dense, 1)
                A_kkt = [K_dense C_dense; C_dense' -Z_dense]
                kappa = cond(A_kkt)

                @printf("  %12.3e  %14.4e  %14.4e  %8d  %8d\n",
                        eps_val, rms_zz, kappa, neq, nlm)
                flush(stdout)
            catch e
                @printf("  %12.3e  %14s  %14s\n", eps_val, "ERROR", string(e)[1:min(14,length(string(e)))])
                flush(stdout)
            end
        end
    end
end

cpt_eps_cond_sweep()
