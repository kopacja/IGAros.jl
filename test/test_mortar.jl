using Test
using IGAros
using LinearAlgebra
using SparseArrays

@testset "Mortar" begin

    # ── Geometry helpers ─────────────────────────────────────────────────────

    @testset "eval_boundary_point" begin
        # Unit interval: p=1, n=2, kv=[0,0,1,1]
        # CPs at (0,0) and (1,0) with unit weight
        kv = [0.0, 0.0, 1.0, 1.0]
        B  = [0.0 0.0 0.0 1.0
              1.0 0.0 0.0 1.0]
        Ps = [1, 2]
        x, dxdξ, R, span = eval_boundary_point(0.5, 1, 2, kv, B, Ps, 2)
        @test x     ≈ [0.5, 0.0] atol=1e-14
        @test dxdξ  ≈ [1.0, 0.0] atol=1e-14
        @test R     ≈ [0.5, 0.5] atol=1e-14
        @test span  == 2

        # Vertical line: CPs at (0,0) and (0,1)
        B2 = [0.0 0.0 0.0 1.0
              0.0 1.0 0.0 1.0]
        x2, dxdξ2, R2, _ = eval_boundary_point(0.25, 1, 2, kv, B2, Ps, 2)
        @test x2    ≈ [0.0, 0.25] atol=1e-14
        @test dxdξ2 ≈ [0.0, 1.0]  atol=1e-14
        @test R2    ≈ [0.75, 0.25] atol=1e-14
    end

    @testset "closest_point_1d on straight line" begin
        # Vertical line x=0, y∈[0,1]:  project (0.1, 0.7) → ξ≈0.7
        kv = [0.0, 0.0, 1.0, 1.0]
        B  = [0.0 0.0 0.0 1.0
              0.0 1.0 0.0 1.0]
        Pm = [1, 2]
        x_s = [0.1, 0.7]
        ξ_m, x_m, _, R_m, _ = closest_point_1d(0.5, x_s, 1, 2, kv, B, Pm, 2)
        @test ξ_m ≈ 0.7          atol=1e-12
        @test x_m ≈ [0.0, 0.7]   atol=1e-12
        @test sum(R_m) ≈ 1.0     atol=1e-14

        # Horizontal line y=0, x∈[0,1]:  project (0.3, 0.5) → ξ≈0.3
        B2 = [0.0 0.0 0.0 1.0
              1.0 0.0 0.0 1.0]
        ξ_m2, x_m2, _, _, _ = closest_point_1d(0.5, [0.3, 0.5], 1, 2, kv, B2, Pm, 2)
        @test ξ_m2 ≈ 0.3         atol=1e-12
        @test x_m2 ≈ [0.3, 0.0]  atol=1e-12

        # Point beyond endpoint: clamps to 1.0
        ξ_m3, x_m3, _, _, _ = closest_point_1d(0.5, [1.5, 0.5], 1, 2, kv, B2, Pm, 2)
        @test ξ_m3 ≈ 1.0         atol=1e-12
    end

    # ── Twin Mortar non-conforming patch test ────────────────────────────────

    @testset "non-conforming patch test (uniaxial tension, ν=0)" begin
        # Two patches with a vertical interface at x = 0.5:
        #   Patch 1: [0, 0.5]×[0,1]  p=1, n=[2,3] → 3 interface nodes
        #   Patch 2: [0.5,1]×[0,1]  p=1, n=[2,4] → 4 interface nodes
        # Non-conforming: 3 vs 4 nodes on the shared interface.
        #
        # BCs:  u_x=0 on left (x=0), u_y=0 on bottom (y=0)
        # Load: unit traction in x on right face of patch 2
        # Expected: u_x = x/E, u_y = 0  (E=1000, ν=0, plane_strain)

        nsd = 2; npd = 2; ned = 2; npc = 2
        E = 1000.0; nu = 0.0

        p = [1 1; 1 1]
        n = [2 3; 2 4]   # patch 1: 2×3, patch 2: 2×4

        KV = generate_knot_vectors(npc, npd, p, n)

        # Patch 1 CPs (global 1..6): [0,0.5]×[0,1], 2×3 grid
        # CP ordering (fast ξ, slow η):
        #   1=(0,0), 2=(0.5,0), 3=(0,0.5), 4=(0.5,0.5), 5=(0,1), 6=(0.5,1)
        B = [
            0.0  0.0  0.0  1.0   # 1
            0.5  0.0  0.0  1.0   # 2
            0.0  0.5  0.0  1.0   # 3
            0.5  0.5  0.0  1.0   # 4
            0.0  1.0  0.0  1.0   # 5
            0.5  1.0  0.0  1.0   # 6
            # Patch 2 CPs (global 7..14): [0.5,1]×[0,1], 2×4 grid
            # η-knots: [0, 0, 1/3, 2/3, 1, 1] → y = 0, 1/3, 2/3, 1
            0.5  0.0        0.0  1.0   # 7
            1.0  0.0        0.0  1.0   # 8
            0.5  1.0/3.0    0.0  1.0   # 9
            1.0  1.0/3.0    0.0  1.0   # 10
            0.5  2.0/3.0    0.0  1.0   # 11
            1.0  2.0/3.0    0.0  1.0   # 12
            0.5  1.0        0.0  1.0   # 13
            1.0  1.0        0.0  1.0   # 14
        ]
        ncp = 14

        P = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14]]

        nel, nnp, nen = patch_metrics(npc, npd, p, n)
        IEN = build_ien(nsd, npd, npc, p, n, nel, nnp, nen)
        INC = [build_inc(n[1, :]), build_inc(n[2, :])]

        mat = LinearElastic(E, nu, :plane_strain)

        # Boundary conditions
        # u_x=0: left face of patch 1 (facet 4, ξ=1): nc[1]=1 → CPs 1,3,5
        # u_y=0: bottom of patch 1 (facet 1, η=1): nc[2]=1 → CPs 1,2
        #        bottom of patch 2 (facet 1, η=1): nc[2]=1 → local {1,2} → global {7,8}
        bc_per_dof = [Int[1, 3, 5], Int[1, 2, 7, 8]]
        neq, ID = build_id(bc_per_dof, ned, ncp)
        LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

        # Standard IGA stiffness
        Ub_zero = zeros(ncp, nsd)
        K, _ = build_stiffness_matrix(
            npc, nsd, npd, ned, neq,
            p, n, KV, P, B, Ub_zero,
            nen, nel, IEN, INC, LM,
            [mat, mat], 2, 1.0
        )
        @test K ≈ K' atol=1e-10

        # Traction on right face of patch 2 (facet 2)
        F = zeros(neq)
        F = segment_load(
            n[2, :], p[2, :], KV[2], P[2], B,
            nnp[2], nen[2], nsd, npd, ned,
            Int[], 2, ID, F, 1.0, 1.0, 2
        )

        # Enforce Dirichlet BCs (no non-homogeneous here)
        IND = Tuple{Int,Float64}[]
        K_bc, F_bc = enforce_dirichlet(IND, K, F)

        # ── Twin Mortar coupling ─────────────────────────────────────────────
        # Interface: right face of patch 1 (facet 2) ↔ left face of patch 2 (facet 4)
        # Both half-passes:
        pairs = [InterfacePair(1, 2, 2, 4),   # patch 1 as slave
                 InterfacePair(2, 4, 1, 2)]    # patch 2 as slave

        Pc = build_interface_cps(pairs, p, n, KV, P, npd, nnp)
        # Expected: right-face CPs of patch 1 = {2,4,6}
        #           left-face  CPs of patch 2 = {7,9,11,13}
        # Total: 7 unique multiplier CPs
        @test length(Pc) == 7
        @test all(∈(Pc), [2, 4, 6, 7, 9, 11, 13])

        nlm = length(Pc)

        C, Z = build_mortar_coupling(
            Pc, pairs, p, n, KV, P, B, ID, nnp,
            ned, nsd, npd, neq, 3, 1.0   # NQUAD=3, epss=1.0
        )
        @test size(C) == (neq, 2 * nlm)
        @test size(Z) == (2 * nlm, 2 * nlm)
        # Z is not symmetric for non-conforming meshes (different integration
        # domains per half-pass), so we only check the diagonal blocks are negative
        # (slave-slave terms: -ε * ∫ R R dΓ < 0)
        @test all(diag(Matrix(Z))[1:nlm]     .<= 0.0)
        @test all(diag(Matrix(Z))[nlm+1:end] .<= 0.0)

        # ── Solve KKT system ─────────────────────────────────────────────────
        U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

        # ── Verify patch-test exactness ──────────────────────────────────────
        for A in 1:ncp
            x_cp = B[A, 1]
            y_cp = B[A, 2]
            ux = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x_cp / E atol=1e-6
            @test uy ≈ 0.0      atol=1e-6
        end
    end

    @testset "non-conforming patch test p=$p_ord (uniaxial tension, ν=0)" for p_ord in 2:4
        nsd = 2; npd = 2; ned = 2; npc = 2
        E = 1000.0; nu = 0.0

        # Single Bezier element in ξ (perpendicular to interface);
        # non-conforming in η: patch 1 has p+2 nodes on interface, patch 2 has p+3.
        n_xi   = p_ord + 1   # minimal single-element patch in ξ
        n_eta1 = p_ord + 2   # patch 1: p+2 interface nodes
        n_eta2 = p_ord + 3   # patch 2: p+3 interface nodes (non-conforming)
        nquad  = p_ord + 1   # sufficient for degree-2p boundary integrals

        p_mat = fill(p_ord, npc, npd)
        n_mat = [n_xi n_eta1; n_xi n_eta2]

        KV   = generate_knot_vectors(npc, npd, p_mat, n_mat)
        ncp1 = n_xi * n_eta1
        ncp2 = n_xi * n_eta2
        ncp  = ncp1 + ncp2

        # Control points (ξ-fast, η-slow ordering):
        #   Patch 1 → [0, 0.5] × [0, 1]
        #   Patch 2 → [0.5, 1] × [0, 1]
        # Bezier knot vector in ξ → x = (i−1)/p_ord * 0.5 gives exact linear map.
        B = zeros(ncp, 4)
        for j in 1:n_eta1, i in 1:n_xi
            r = (j - 1) * n_xi + i
            B[r, 1] = (i - 1) * 0.5 / p_ord
            B[r, 2] = (j - 1) / (n_eta1 - 1)
            B[r, 4] = 1.0
        end
        for j in 1:n_eta2, i in 1:n_xi
            r = ncp1 + (j - 1) * n_xi + i
            B[r, 1] = 0.5 + (i - 1) * 0.5 / p_ord
            B[r, 2] = (j - 1) / (n_eta2 - 1)
            B[r, 4] = 1.0
        end

        P = [collect(1:ncp1), collect(ncp1 + 1 : ncp)]

        nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
        IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
        INC = [build_inc(n_mat[1, :]), build_inc(n_mat[2, :])]
        mat = LinearElastic(E, nu, :plane_strain)

        # BCs: u_x=0 on left face of patch 1 (i=1, ξ=0)
        #      u_y=0 on bottom of both patches (j=1, η=0)
        left_face_1 = [1 + (j - 1) * n_xi for j in 1:n_eta1]
        bottom_1    = collect(1:n_xi)
        bottom_2    = collect(ncp1 + 1 : ncp1 + n_xi)
        bc_per_dof  = [left_face_1, vcat(bottom_1, bottom_2)]

        neq, ID = build_id(bc_per_dof, ned, ncp)
        LM      = build_lm(nen, ned, npc, nel, ID, IEN, P)

        Ub_zero = zeros(ncp, nsd)
        K, _    = build_stiffness_matrix(
            npc, nsd, npd, ned, neq,
            p_mat, n_mat, KV, P, B, Ub_zero,
            nen, nel, IEN, INC, LM,
            [mat, mat], nquad, 1.0
        )
        @test K ≈ K' atol=1e-10

        # Unit traction in x on right face of patch 2 (facet 2, ξ=1)
        F  = zeros(neq)
        F  = segment_load(
            n_mat[2, :], p_mat[2, :], KV[2], P[2], B,
            nnp[2], nen[2], nsd, npd, ned,
            Int[], 2, ID, F, 1.0, 1.0, nquad
        )

        IND        = Tuple{Int,Float64}[]
        K_bc, F_bc = enforce_dirichlet(IND, K, F)

        # Twin Mortar interface coupling
        pairs = [InterfacePair(1, 2, 2, 4),
                 InterfacePair(2, 4, 1, 2)]
        Pc    = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp)

        # Right face of patch 1 (i=n_xi): CPs {n_xi, 2·n_xi, …, n_eta1·n_xi}
        # Left  face of patch 2 (i=1   ): CPs {ncp1+1, ncp1+1+n_xi, …}
        expect_if1 = [j * n_xi for j in 1:n_eta1]
        expect_if2 = [ncp1 + 1 + (j - 1) * n_xi for j in 1:n_eta2]
        @test length(Pc) == n_eta1 + n_eta2
        @test all(∈(Pc), vcat(expect_if1, expect_if2))

        nlm  = length(Pc)
        C, Z = build_mortar_coupling(
            Pc, pairs, p_mat, n_mat, KV, P, B, ID, nnp,
            ned, nsd, npd, neq, nquad, 1.0
        )
        @test size(C) == (neq, 2 * nlm)
        @test size(Z) == (2 * nlm, 2 * nlm)
        @test all(diag(Matrix(Z))[1:nlm]     .<= 0.0)
        @test all(diag(Matrix(Z))[nlm+1:end] .<= 0.0)

        U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

        # Exact solution: u_x = x/E, u_y = 0 at every CP
        for A in 1:ncp
            x_cp = B[A, 1]
            ux   = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy   = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x_cp / E  atol=1e-6
            @test uy ≈ 0.0       atol=1e-6
        end
    end

    @testset "build_interface_cps" begin
        # Conforming interface: both patches share the same CPs (like the 2-patch
        # test in test_assembly.jl). build_interface_cps should return unique union.
        nsd=2; npd=2; npc=2
        p = [1 1; 1 1]
        n = [2 2; 2 2]
        KV = generate_knot_vectors(npc, npd, p, n)
        P = [[1, 2, 3, 4], [2, 5, 4, 6]]
        _, nnp, _ = patch_metrics(npc, npd, p, n)

        # Right face of patch 1 (facet 2): CPs {2,4}
        # Left  face of patch 2 (facet 4): CPs {P[2][1], P[2][3]} = {2,4}
        pairs = [InterfacePair(1, 2, 2, 4), InterfacePair(2, 4, 1, 2)]
        Pc = build_interface_cps(pairs, p, n, KV, P, npd, nnp)
        @test sort(Pc) == [2, 4]   # shared CPs appear once
    end

end
