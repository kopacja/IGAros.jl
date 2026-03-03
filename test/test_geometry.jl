using Test
using IGAros
using LinearAlgebra

@testset "Geometry" begin

    # ── Unit square patch (p=1, 2×2 CPs, weight=1) ───────────────────────────
    # Control points: (0,0), (1,0), (0,1), (1,1) with weight=1
    # This is a bilinear map (identical to standard FEM)
    function make_unit_square_patch()
        p = [1, 1]
        n = [2, 2]
        kv = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
        B  = [0.0 0.0 0.0 1.0   # CP1: (0,0), weight=1
              1.0 0.0 0.0 1.0   # CP2: (1,0)
              0.0 1.0 0.0 1.0   # CP3: (0,1)
              1.0 1.0 0.0 1.0]  # CP4: (1,1)
        P  = [1, 2, 3, 4]
        nsd = 2; npd = 2; ncp = 4; nen = 4
        nel = [1]; nnp = [4]
        IEN = build_ien(nsd, npd, 1,
                        reshape(p, 1, :), reshape(n, 1, :),
                        nel, nnp, [nen])
        INC = [build_inc(n)]
        n0 = INC[1][IEN[1][1, 1]]   # knot span for element 1
        return p, n, kv, B, P, nsd, npd, nen, IEN, INC, n0
    end

    @testset "partition of unity" begin
        p, n, kv, B, P, nsd, npd, nen, IEN, INC, n0 = make_unit_square_patch()

        for xi in [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-0.5, 0.7]]
            R, _, _, detJ, _ = shape_function(
                p, n, kv, B, P, xi, nen, nsd, npd, 1, n0, IEN[1], INC[1]
            )
            @test sum(R) ≈ 1.0 atol=1e-13
            @test all(R .>= -1e-14)
        end
    end

    @testset "reproducing property" begin
        p, n, kv, B, P, nsd, npd, nen, IEN, INC, n0 = make_unit_square_patch()

        # ∑ R_a · X_a = X  (exact geometry reproduction)
        for xi in [[-0.5, -0.5], [0.0, 0.5], [0.8, -0.3]]
            R, _, _, _, _ = shape_function(
                p, n, kv, B, P, xi, nen, nsd, npd, 1, n0, IEN[1], INC[1]
            )
            Xcp = B[P[IEN[1][1, :]], 1:nsd]  # nen × nsd
            X_computed = Xcp' * R             # nsd
            # For unit square: X = (xi+1)/2 mapped from [-1,1]² to [0,1]²
            X_exact = [(xi[1] + 1) / 2, (xi[2] + 1) / 2]
            @test X_computed ≈ X_exact atol=1e-13
        end
    end

    @testset "positive Jacobian" begin
        p, n, kv, B, P, nsd, npd, nen, IEN, INC, n0 = make_unit_square_patch()

        for xi in [[-0.9, -0.9], [0.0, 0.0], [0.9, 0.9]]
            _, _, _, detJ, _ = shape_function(
                p, n, kv, B, P, xi, nen, nsd, npd, 1, n0, IEN[1], INC[1]
            )
            @test detJ > 0
        end
    end

    @testset "gradient (finite difference check)" begin
        p, n, kv, B, P, nsd, npd, nen, IEN, INC, n0 = make_unit_square_patch()

        # For the unit-square bilinear patch, dR/dx should match bilinear FEM gradients.
        xi = [0.0, 0.0]
        R, dR_dx, _, _, _ = shape_function(
            p, n, kv, B, P, xi, nen, nsd, npd, 1, n0, IEN[1], INC[1]
        )

        # FD verification: perturb parent coords ξ slightly and finite-diff
        h = 1e-6
        R_p1, _, _, _, _ = shape_function(
            p, n, kv, B, P, [xi[1]+h, xi[2]], nen, nsd, npd, 1, n0, IEN[1], INC[1]
        )
        R_p2, _, _, _, _ = shape_function(
            p, n, kv, B, P, [xi[1], xi[2]+h], nen, nsd, npd, 1, n0, IEN[1], INC[1]
        )

        # Parent-space derivative ≈ (R(ξ+h) - R(ξ)) / h
        dR_dxi1_fd = (R_p1 - R) ./ h
        dR_dxi2_fd = (R_p2 - R) ./ h

        # Physical gradient = parent gradient × (dξ/dx) = parent gradient × 2
        # (because ξ ∈ [-1,1] maps to x ∈ [0,1], so dx/dξ = 0.5 → dξ/dx = 2)
        @test dR_dx[:, 1] ≈ dR_dxi1_fd .* 2.0 atol=1e-5
        @test dR_dx[:, 2] ≈ dR_dxi2_fd .* 2.0 atol=1e-5
    end

end
