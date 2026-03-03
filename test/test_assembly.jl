using Test
using IGAros
using LinearAlgebra
using SparseArrays

@testset "Assembly" begin

    # ── Build a single p=1 unit-square patch (bilinear, E=1000, ν=0, plane strain) ──
    function setup_patch_test(; E=1000.0, nu=0.0, p_deg=1)
        nsd = 2; npd = 2; ned = 2; npc = 1
        p = fill(p_deg, 1, 2)
        n = fill(p_deg + 1, 1, 2)   # minimal: n=p+1

        KV = generate_knot_vectors(npc, npd, p, n)

        # Unit square control points [x, y, z(=0), w]
        B = [0.0 0.0 0.0 1.0
             1.0 0.0 0.0 1.0
             0.0 1.0 0.0 1.0
             1.0 1.0 0.0 1.0]
        ncp = size(B, 1)
        P   = [collect(1:ncp)]

        nel, nnp, nen = patch_metrics(npc, npd, p, n)
        IEN = build_ien(nsd, npd, npc, p, n, nel, nnp, nen)
        INC = [build_inc(n[1, :])]

        mat = LinearElastic(E, nu, :plane_strain)

        return (; nsd, npd, ned, npc, p, n, KV, B, P, ncp, nel, nnp, nen, IEN, INC, mat)
    end

    @testset "element stiffness symmetry and PSD" begin
        s = setup_patch_test()
        Ub = zeros(s.ncp, s.nsd)
        n0 = s.INC[1][s.IEN[1][1, 1]]

        Ke, Fe = element_stiffness(
            s.p[1,:], s.n[1,:], s.KV[1], s.P[1], s.B, Ub,
            s.IEN[1], s.INC[1], 1, s.nsd, s.npd, s.nen[1], 2, s.mat, 1.0
        )

        @test Ke ≈ Ke' atol=1e-12             # symmetry
        @test all(eigvals(Ke) .>= -1e-10)     # PSD (rigid body modes allowed)
        @test size(Ke) == (8, 8)
    end

    @testset "patch test: uniaxial tension (ν=0)" begin
        # Single p=1 unit-square patch under uniaxial tension (x-direction).
        # E=1000, ν=0, plane strain.
        # BC: fix u_y on bottom (y=0), fix u_x on left (x=0).
        # Load: unit traction on right face (x=1).
        # Expected: u_x = x/E·t = x/1000 (with t=1)
        s = setup_patch_test(E=1000.0, nu=0.0)

        # Homogeneous BCs:
        #   DOF 2 (y) constrained on segment 1 (bottom: y=0 control points → CPs 1,2)
        #   DOF 1 (x) constrained on segment 4 (left:  x=0 control points → CPs 1,3)
        bc = [[3], [1, 2]]   # DOF1: CP3, DOF2: CPs 1,2 [manually set]
        # Actually let's just set bc from the geometry:
        # Bottom (y=0): CPs 1,2   → constrain DOF 2 (y)
        # Left  (x=0): CPs 1,3   → constrain DOF 1 (x)
        # Fix CP1 in both DOFs to remove rigid body mode.
        bc_per_dof = [Int[1, 3], Int[1, 2]]   # DOF1→{1,3}, DOF2→{1,2}

        neq, ID = build_id(bc_per_dof, s.ned, s.ncp)
        LM = build_lm(s.nen, s.ned, s.npc, s.nel, ID, s.IEN, s.P)

        Ub_zero = zeros(s.ncp, s.nsd)
        K, F_int = build_stiffness_matrix(
            s.npc, s.nsd, s.npd, s.ned, neq,
            s.p, s.n, s.KV, s.P, s.B, Ub_zero,
            s.nen, s.nel, s.IEN, s.INC, LM,
            [s.mat], 2, 1.0
        )

        @test K ≈ Matrix(K') atol=1e-10

        # Apply traction on right face (segment 2, DOF 1, traction=1.0)
        F = zeros(neq)
        F = segment_load(
            s.n[1,:], s.p[1,:], s.KV[1], s.P[1], s.B,
            s.nnp[1], s.nen[1], s.nsd, s.npd, s.ned,
            Int[], 2, ID, F, 1.0, 1.0, 2
        )

        # Solve
        IND = Tuple{Int,Float64}[]  # no non-homo BCs
        K_bc, F_bc = enforce_dirichlet(IND, K, -F)  # Note: K*u = F_ext - F_int
        # For first iteration F_int = 0, so just K*u = F
        K_bc2, F_bc2 = enforce_dirichlet(IND, K, F)
        U = linear_solve(K_bc2, F_bc2)

        # Recover displacements
        Bu, Ub = build_updated_geometry(s.nsd, s.ncp, ID, U, s.B)

        # With ν=0 and uniaxial tension: u_x = x * (traction/E) = x/1000
        # u_y = 0 everywhere
        for A in 1:s.ncp
            x = s.B[A, 1]
            y = s.B[A, 2]
            eq_x = ID[1, A]
            eq_y = ID[2, A]
            ux = (eq_x != 0) ? U[eq_x] : 0.0
            uy = (eq_y != 0) ? U[eq_y] : 0.0
            @test ux ≈ x / 1000.0 atol=1e-10
            @test uy ≈ 0.0        atol=1e-10
        end
    end

    @testset "two-patch conforming patch test (uniaxial tension)" begin
        # Two p=1 patches side by side:
        #   Patch 1: [0, 0.5]×[0,1]   (CPs 1–4)
        #   Patch 2: [0.5, 1]×[0,1]   (CPs 2,5,4,6  — sharing interface CPs 2,4)
        # Global B (6 CPs):
        #   1=(0,0), 2=(0.5,0), 3=(0,1), 4=(0.5,1), 5=(1,0), 6=(1,1)
        # BCs:  u_x=0 at x=0 (CPs 1,3);  u_y=0 at y=0 (CPs 1,2,5)
        # Load: unit traction on right face of patch 2 (x=1)
        # Expected: u_x = x/1000, u_y = 0
        nsd=2; npd=2; ned=2; npc=2
        E=1000.0; nu=0.0
        p = [1 1; 1 1]
        n = [2 2; 2 2]

        KV = generate_knot_vectors(npc, npd, p, n)

        B = [0.0  0.0  0.0  1.0   # CP1
             0.5  0.0  0.0  1.0   # CP2 — interface
             0.0  1.0  0.0  1.0   # CP3
             0.5  1.0  0.0  1.0   # CP4 — interface
             1.0  0.0  0.0  1.0   # CP5
             1.0  1.0  0.0  1.0]  # CP6
        ncp = 6

        P = [[1, 2, 3, 4], [2, 5, 4, 6]]

        nel, nnp, nen = patch_metrics(npc, npd, p, n)
        IEN = build_ien(nsd, npd, npc, p, n, nel, nnp, nen)
        INC = [build_inc(n[1, :]), build_inc(n[2, :])]

        mat = LinearElastic(E, nu, :plane_strain)

        # u_x=0 on left (x=0): CPs 1,3;  u_y=0 on bottom (y=0): CPs 1,2,5
        bc_per_dof = [Int[1, 3], Int[1, 2, 5]]
        neq, ID = build_id(bc_per_dof, ned, ncp)
        LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

        Ub_zero = zeros(ncp, nsd)
        K, _ = build_stiffness_matrix(
            npc, nsd, npd, ned, neq,
            p, n, KV, P, B, Ub_zero,
            nen, nel, IEN, INC, LM,
            [mat, mat], 2, 1.0
        )
        @test K ≈ Matrix(K') atol=1e-10

        # Traction on right face of patch 2 (facet 2, ξ=n[2,1]=2)
        F = zeros(neq)
        F = segment_load(
            n[2, :], p[2, :], KV[2], P[2], B,
            nnp[2], nen[2], nsd, npd, ned,
            Int[], 2, ID, F, 1.0, 1.0, 2
        )

        IND = Tuple{Int,Float64}[]
        K_bc, F_bc = enforce_dirichlet(IND, K, F)
        U = linear_solve(K_bc, F_bc)

        _, Ub = build_updated_geometry(nsd, ncp, ID, U, B)

        for A in 1:ncp
            x  = B[A, 1]
            ux = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x / E atol=1e-10
            @test uy ≈ 0.0   atol=1e-10
        end
    end

    @testset "build_updated_geometry" begin
        # Simple: ncp=2, nsd=2, u=[1.0, 2.0] → Ub=[1,2; 0,0]
        ID = [1 0; 2 0]   # CP1 has DOFs 1,2; CP2 is fully constrained
        B  = [0.0 0.0 0.0 1.0; 1.0 0.0 0.0 1.0]
        U  = [1.0, 2.0]
        Bu, Ub = build_updated_geometry(2, 2, ID, U, B)
        @test Ub[1, :] ≈ [1.0, 2.0]
        @test Ub[2, :] ≈ [0.0, 0.0]
        @test Bu[1, 1] ≈ 1.0; @test Bu[1, 2] ≈ 2.0
        @test Bu[2, 1] ≈ 1.0; @test Bu[2, 2] ≈ 0.0
    end

end
