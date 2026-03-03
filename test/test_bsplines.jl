using Test
using IGAros
using LinearAlgebra

@testset "BSplines" begin

    # ── find_span ─────────────────────────────────────────────────────────────
    @testset "find_span" begin
        # Open uniform quadratic knot vector: [0,0,0,0.5,1,1,1]
        kv = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        p  = 2; n_basis = 3  # 4 basis functions, indices 0..3

        @test find_span(n_basis, p, 0.0, kv)   == 3   # u=0: clamp to span p+1
        @test find_span(n_basis, p, 1.0, kv)   == 4   # u=1: clamp to n_basis
        @test find_span(n_basis, p, 0.25, kv)  == 3
        @test find_span(n_basis, p, 0.75, kv)  == 4

        # Degree 1, 3 CPs: [0,0,0.5,1,1]  (n_basis = 3-1 = 2)
        kv1 = [0.0, 0.0, 0.5, 1.0, 1.0]
        @test find_span(2, 1, 0.0, kv1)  == 2
        @test find_span(2, 1, 0.25, kv1) == 2
        @test find_span(2, 1, 0.5, kv1)  == 3
        @test find_span(2, 1, 1.0, kv1)  == 3   # right endpoint → last span
    end

    # ── basis_funs ────────────────────────────────────────────────────────────
    @testset "basis_funs" begin
        # Partition of unity: sum of all basis functions = 1
        kv = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        p  = 2
        for u in 0.0:0.1:1.0
            i = find_span(3, p, u, kv)
            N = basis_funs(i, u, p, kv)
            @test sum(N) ≈ 1.0 atol=1e-14
            @test all(N .>= -1e-15)
        end

        # At u=0: only N[1] = 1 (end-interpolation)
        i = find_span(3, p, 0.0, kv)
        N = basis_funs(i, 0.0, p, kv)
        @test N[1] ≈ 1.0 atol=1e-14

        # At u=1: only N[end] = 1
        i = find_span(3, p, 1.0, kv)
        N = basis_funs(i, 1.0, p, kv)
        @test N[end] ≈ 1.0 atol=1e-14

        # p=1 linear: at u=0.5 in span [0,0,1,1], N=[0.5, 0.5]
        kv_lin = [0.0, 0.0, 1.0, 1.0]
        i = find_span(1, 1, 0.5, kv_lin)
        N = basis_funs(i, 0.5, 1, kv_lin)
        @test N ≈ [0.5, 0.5] atol=1e-14
    end

    # ── bspline_basis_and_deriv ───────────────────────────────────────────────
    @testset "bspline_basis_and_deriv" begin
        # Check that row 1 matches basis_funs
        kv = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        p  = 2
        for u in [0.1, 0.3, 0.5, 0.7, 0.9]
            i = find_span(3, p, u, kv)
            N_ref = basis_funs(i, u, p, kv)
            ders  = bspline_basis_and_deriv(i, u, p, 1, kv)
            @test ders[1, :] ≈ N_ref atol=1e-13
        end

        # Derivatives via finite differences
        h = 1e-7
        for u in [0.2, 0.6]
            i  = find_span(3, p, u, kv)
            ih = find_span(3, p, min(u + h, 0.9999), kv)
            ders = bspline_basis_and_deriv(i, u, p, 1, kv)
            Np   = basis_funs(ih, u + h, p, kv)
            N    = basis_funs(i,  u,     p, kv)
            fd_deriv = (Np - N) ./ h
            # Derivative only valid for the support of span i
            @test norm(ders[2, :] .- fd_deriv) < 1e-5
        end

        # Sum of derivatives = 0 (partition of unity ⟹ sum of derivatives = 0)
        for u in [0.2, 0.7]
            i = find_span(3, p, u, kv)
            ders = bspline_basis_and_deriv(i, u, p, 1, kv)
            @test sum(ders[2, :]) ≈ 0.0 atol=1e-12
        end
    end

end
