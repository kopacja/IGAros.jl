using Test
using IGAros

@testset "Quadrature" begin

    @testset "gauss_rule exactness" begin
        # An n-point Gauss-Legendre rule integrates polynomials of degree ≤ 2n-1 exactly.
        # Tolerance is 1e-8 because table values in Quadrature.jl have limited precision.
        for n in [1, 2, 3, 4, 5, 6, 7, 8]
            pts, wts = gauss_rule(n)
            for k in 0:2n-1
                # ∫₋₁¹ xᵏ dx = 0 if k odd, 2/(k+1) if k even
                exact = iseven(k) ? 2.0/(k+1) : 0.0
                approx = sum(wts .* pts .^ k)
                @test approx ≈ exact atol=1e-8
            end
        end
    end

    @testset "weights sum to 2" begin
        for n in [1,2,3,4,5,6,7,8,10,20]
            _, wts = gauss_rule(n)
            @test sum(wts) ≈ 2.0 atol=1e-8
        end
    end

    @testset "gauss_product 2D" begin
        # ∫₋₁¹∫₋₁¹ 1 dA = 4
        gpw = gauss_product(3, 2)
        total = sum(w for (_, w) in gpw)
        @test total ≈ 4.0 atol=1e-12

        # ∫₋₁¹∫₋₁¹ x·y dx dy = 0
        gpw = gauss_product(2, 2)
        integral = sum(gp[1] * gp[2] * gw for (gp, gw) in gpw)
        @test integral ≈ 0.0 atol=1e-12

        # ∫₋₁¹∫₋₁¹ x² dy dx = 4/3
        gpw = gauss_product(2, 2)
        integral = sum(gp[1]^2 * gw for (gp, gw) in gpw)
        @test integral ≈ 4/3 atol=1e-12
    end

    @testset "gauss_product 1D agrees with gauss_rule" begin
        for n in [2, 3, 5]
            pts1d, wts1d = gauss_rule(n)
            gpw = gauss_product(n, 1)
            @test length(gpw) == n
            for (i, (gp, gw)) in enumerate(gpw)
                @test gp[1] ≈ pts1d[i] atol=1e-14
                @test gw    ≈ wts1d[i] atol=1e-14
            end
        end
    end

end
