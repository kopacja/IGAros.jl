using Test
using IGAros
using LinearAlgebra

@testset "Materials & StrainDisplacement" begin

    @testset "LinearElastic plane_stress" begin
        mat = LinearElastic(1000.0, 0.0, :plane_stress)
        D = elastic_constants(mat, 2)
        @test size(D) == (3, 3)
        @test D ≈ [1000.0 0.0 0.0; 0.0 1000.0 0.0; 0.0 0.0 500.0] atol=1e-10

        # ν=0.3
        E = 210e3; ν = 0.3
        mat2 = LinearElastic(E, ν, :plane_stress)
        D2 = elastic_constants(mat2, 2)
        @test D2[1,1] ≈ E/(1-ν^2) atol=1.0
        @test D2[1,2] ≈ ν*E/(1-ν^2) atol=1.0
        @test D2[3,3] ≈ E*(1-ν)/(2*(1-ν^2)) atol=1.0
        @test issymmetric(D2)
    end

    @testset "LinearElastic plane_strain" begin
        E = 1000.0; ν = 0.3
        mat = LinearElastic(E, ν, :plane_strain)
        D = elastic_constants(mat, 2)
        @test size(D) == (3, 3)
        @test issymmetric(D)
        # D[1,1] = E(1-ν)/((1+ν)(1-2ν))
        expected_11 = E*(1-ν)/((1+ν)*(1-2ν))
        @test D[1,1] ≈ expected_11 atol=1e-10
    end

    @testset "LinearElastic 3D" begin
        E = 210e3; ν = 0.25
        mat = LinearElastic(E, ν, :three_d)
        D = elastic_constants(mat, 3)
        @test size(D) == (6, 6)
        @test issymmetric(D)
        λ = ν*E/((1+ν)*(1-2ν))
        μ = E/(2*(1+ν))
        @test D[1,1] ≈ λ + 2μ atol=1e-8
        @test D[1,2] ≈ λ atol=1e-8
        @test D[4,4] ≈ μ atol=1e-8
    end

    @testset "strain_displacement_matrix 2D" begin
        nen = 4; nsd = 2
        # Constant gradient: dN/dx = 1, dN/dy = 2 for each node
        dN_dX = ones(2, nen)
        dN_dX[2, :] .= 2.0
        B0 = strain_displacement_matrix(nsd, nen, dN_dX)

        @test size(B0) == (3, 8)

        # Row 1 (ε_xx): B0[1, 2a-1] = dN_a/dx = 1, B0[1, 2a] = 0
        for a in 1:nen
            @test B0[1, 2a-1] == 1.0
            @test B0[1, 2a]   == 0.0
        end
        # Row 2 (ε_yy): B0[2, 2a-1] = 0, B0[2, 2a] = dN_a/dy = 2
        for a in 1:nen
            @test B0[2, 2a-1] == 0.0
            @test B0[2, 2a]   == 2.0
        end
        # Row 3 (γ_xy): B0[3, 2a-1] = dN_a/dy = 2, B0[3, 2a] = dN_a/dx = 1
        for a in 1:nen
            @test B0[3, 2a-1] == 2.0
            @test B0[3, 2a]   == 1.0
        end
    end

    @testset "strain_displacement_matrix 3D" begin
        nen = 8; nsd = 3
        dN_dX = rand(3, nen)
        B0 = strain_displacement_matrix(nsd, nen, dN_dX)
        @test size(B0) == (6, 24)

        for a in 1:nen
            @test B0[1, 3a-2] == dN_dX[1, a]   # ε_xx
            @test B0[2, 3a-1] == dN_dX[2, a]   # ε_yy
            @test B0[3, 3a]   == dN_dX[3, a]   # ε_zz
        end
    end

end
