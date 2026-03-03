using Test
using IGAros

@testset "KnotVectors" begin

    @testset "generate_knot_vector" begin
        # p=1, n=2: [0,0,1,1]
        kv = generate_knot_vector(2, 1)
        @test kv == [0.0, 0.0, 1.0, 1.0]

        # p=2, n=4: [0,0,0,0.5,1,1,1]
        kv = generate_knot_vector(4, 2)
        @test kv ≈ [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]

        # p=1, n=4: [0,0,1/3,2/3,1,1]
        kv = generate_knot_vector(4, 1)
        @test kv ≈ [0.0, 0.0, 1/3, 2/3, 1.0, 1.0]

        # Clamping: first p+1 = 0, last p+1 = 1
        for p in 1:3, n in (p+1):(p+5)
            kv = generate_knot_vector(n, p)
            @test length(kv) == n + p + 1
            @test all(kv[1:p+1] .== 0.0)
            @test all(kv[n+1:end] .== 1.0)
            @test issorted(kv)
        end
    end

    @testset "generate_knot_vectors" begin
        p = [1 1; 1 1]
        n = [2 2; 2 2]
        KV = generate_knot_vectors(2, 2, p, n)
        @test length(KV) == 2
        for pc in 1:2, i in 1:2
            @test KV[pc][i] == [0.0, 0.0, 1.0, 1.0]
        end
    end

    @testset "knot_insertion single" begin
        # Insert u=0.5 into [0,0,1,1] for linear Bézier (n=1, so 2 CPs)
        kv = [0.0, 0.0, 1.0, 1.0]
        Qw = [0.0 1.0; 1.0 1.0]  # 2 CPs, 2 cols (x, w)
        n_new, kv_new, Qw_new = knot_insertion(1, 1, 0.5, kv, Qw)
        @test n_new == 2    # was 1 (n=1 means 2 CPs), now 3
        @test length(kv_new) == 5
        @test kv_new ≈ [0.0, 0.0, 0.5, 1.0, 1.0]
        @test size(Qw_new, 1) == 3
        # Midpoint should be (0+1)/2 = 0.5
        @test Qw_new[2, 1] ≈ 0.5 atol=1e-14
    end

    @testset "krefinement unit square: insert midpoint in ξ" begin
        using LinearAlgebra
        # Single p=1 unit square, insert u=0.5 in direction 1 (ξ)
        nsd = 2; npd = 2; npc = 1
        p = [1 1]
        n = [2 2]
        KV = generate_knot_vectors(npc, npd, p, n)
        B = [0.0 0.0 0.0 1.0
             1.0 0.0 0.0 1.0
             0.0 1.0 0.0 1.0
             1.0 1.0 0.0 1.0]
        P = [[1, 2, 3, 4]]

        kref_data = [[1.0, 1.0, 0.5]]   # patch 1, direction 1, insert u=0.5
        n_new, ncp_new, KV_new, B_new, P_new =
            krefinement(nsd, npd, npc, n, p, KV, B, P, kref_data)

        # Mesh structure checks
        @test n_new[1, :] == [3, 2]     # 3 CPs in ξ, 2 in η
        @test ncp_new == 6
        @test length(KV_new[1][1]) == 5
        @test KV_new[1][1] ≈ [0.0, 0.0, 0.5, 1.0, 1.0]
        @test size(B_new, 1) == 6

        # New midpoint CPs should lie at x=0.5
        xs = sort(unique(round.(B_new[:, 1]; digits=10)))
        @test xs ≈ [0.0, 0.5, 1.0] atol=1e-12

        # Partition of unity and positive Jacobian at every Gauss point
        nel, nnp, nen = patch_metrics(npc, npd, n_new, n_new)
        IEN = build_ien(nsd, npd, npc, n_new, n_new, nel, nnp, nen)
        INC = [build_inc(n_new[1, :])]

        for el in 1:nel[1]
            n0 = INC[1][IEN[1][el, 1]]
            for xi in [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]
                R, _, _, detJ, _ = shape_function(
                    p[1, :], n_new[1, :], KV_new[1], B_new, P_new[1],
                    xi, nen[1], nsd, npd, el, n0, IEN[1], INC[1]
                )
                @test sum(R) ≈ 1.0 atol=1e-13
                @test detJ > 0
            end
        end

        # Reproducing property: ∑ R_a · x_a = x at a sample point in each element
        for el in 1:nel[1]
            n0 = INC[1][IEN[1][el, 1]]
            xi = [0.0, 0.0]
            R, _, _, _, _ = shape_function(
                p[1, :], n_new[1, :], KV_new[1], B_new, P_new[1],
                xi, nen[1], nsd, npd, el, n0, IEN[1], INC[1]
            )
            Xcp = B_new[P_new[1][IEN[1][el, :]], 1:nsd]
            X_comp = Xcp' * R   # nsd vector
            # Physical midpoint of element el is known from KV: elements split at u=0.5
            # Map xi=(0,0) through the parent-to-parametric-to-physical chain
            kv1 = KV_new[1][1]
            a1  = kv1[n0[1]]; b1 = kv1[n0[1]+1]
            kv2 = KV_new[1][2]
            a2  = kv2[n0[2]]; b2 = kv2[n0[2]+1]
            Xi1 = 0.5*(b1 - a1)*0.0 + 0.5*(b1 + a1)
            Xi2 = 0.5*(b2 - a2)*0.0 + 0.5*(b2 + a2)
            X_exact = [Xi1, Xi2]  # For unit square: x = ξ, y = η
            @test X_comp ≈ X_exact atol=1e-13
        end
    end

    @testset "krefinement: insert midpoint in both directions" begin
        nsd = 2; npd = 2; npc = 1
        p = [1 1]
        n = [2 2]
        KV = generate_knot_vectors(npc, npd, p, n)
        B = [0.0 0.0 0.0 1.0
             1.0 0.0 0.0 1.0
             0.0 1.0 0.0 1.0
             1.0 1.0 0.0 1.0]
        P = [[1, 2, 3, 4]]

        kref_data = [[1.0, 1.0, 0.5], [1.0, 2.0, 0.5]]   # insert 0.5 in both directions
        n_new, ncp_new, KV_new, B_new, P_new =
            krefinement(nsd, npd, npc, n, p, KV, B, P, kref_data)

        @test n_new[1, :] == [3, 3]
        @test ncp_new == 9
        @test KV_new[1][1] ≈ [0.0, 0.0, 0.5, 1.0, 1.0]
        @test KV_new[1][2] ≈ [0.0, 0.0, 0.5, 1.0, 1.0]
        @test size(B_new, 1) == 9

        # Partition of unity across all 4 elements
        nel, nnp, nen = patch_metrics(npc, npd, n_new, n_new)
        IEN = build_ien(nsd, npd, npc, n_new, n_new, nel, nnp, nen)
        INC = [build_inc(n_new[1, :])]

        for el in 1:nel[1]
            n0 = INC[1][IEN[1][el, 1]]
            R, _, _, detJ, _ = shape_function(
                p[1, :], n_new[1, :], KV_new[1], B_new, P_new[1],
                [0.0, 0.0], nen[1], nsd, npd, el, n0, IEN[1], INC[1]
            )
            @test sum(R) ≈ 1.0 atol=1e-13
            @test detJ > 0
        end
    end

end
