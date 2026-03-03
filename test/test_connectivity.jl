using Test
using IGAros

@testset "Connectivity" begin

    @testset "nurbs_coords" begin
        # 2×2 grid, 1-based
        n = [2, 2]
        @test nurbs_coords(1, 2, n) == [1, 1]
        @test nurbs_coords(2, 2, n) == [2, 1]
        @test nurbs_coords(3, 2, n) == [1, 2]
        @test nurbs_coords(4, 2, n) == [2, 2]

        # 2×3 grid
        n2 = [2, 3]
        @test nurbs_coords(5, 2, n2) == [1, 3]
        @test nurbs_coords(6, 2, n2) == [2, 3]
    end

    @testset "build_inc" begin
        inc = build_inc([2, 2])
        @test length(inc) == 4
        @test inc[1] == [1, 1]
        @test inc[4] == [2, 2]
    end

    @testset "patch_metrics" begin
        # p=1, n=2×2: nel=1×1=1, nen=2×2=4, nnp=4
        p = reshape([1, 1], 1, 2)
        n = reshape([2, 2], 1, 2)
        nel, nnp, nen = patch_metrics(1, 2, p, n)
        @test nel == [1]
        @test nnp == [4]
        @test nen == [4]

        # p=2, n=4×4: nel=2×2=4, nen=3×3=9, nnp=16
        p2 = reshape([2, 2], 1, 2)
        n2 = reshape([4, 4], 1, 2)
        nel2, nnp2, nen2 = patch_metrics(1, 2, p2, n2)
        @test nel2 == [4]
        @test nnp2 == [16]
        @test nen2 == [9]
    end

    @testset "build_ien p=1 single patch" begin
        # 2×2 mesh with p=1: 1 element, 4 nodes
        p = reshape([1, 1], 1, 2)
        n = reshape([2, 2], 1, 2)
        nel = [1]; nnp = [4]; nen = [4]
        IEN = build_ien(2, 2, 1, p, n, nel, nnp, nen)
        @test size(IEN[1]) == (1, 4)
        # Element contains all 4 nodes
        @test sort(vec(IEN[1])) == [1, 2, 3, 4]
    end

    @testset "build_id" begin
        # 2 CPs, 2 DOFs, constrain CP 1 in DOF 1
        bc = [[1], Int[]]
        neq, ID = build_id(bc, 2, 2)
        @test ID[1, 1] == 0     # constrained
        @test ID[1, 2] != 0
        @test ID[2, 1] != 0
        @test ID[2, 2] != 0
        @test neq == 3
    end

    @testset "build_lm" begin
        p = reshape([1, 1], 1, 2)
        n = reshape([2, 2], 1, 2)
        nel = [1]; nnp = [4]; nen = [4]
        IEN = build_ien(2, 2, 1, p, n, nel, nnp, nen)

        # All free DOFs
        bc = [Int[], Int[]]
        neq, ID = build_id(bc, 2, 4)
        P = [collect(1:4)]
        LM = build_lm([4], 2, 1, [1], ID, IEN, P)

        @test size(LM[1]) == (8, 1)   # 8 dofs × 1 element
        @test all(LM[1][:, 1] .!= 0)  # no constrained DOFs
    end

end
