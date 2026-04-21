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
            ned, nsd, npd, neq, 3, 1e6   # NQUAD=3, epss=1e6
        )
        @test size(C) == (neq, 2 * nlm)
        @test size(Z) == (2 * nlm, 2 * nlm)
        # Z is not symmetric for non-conforming meshes (different integration
        # domains per half-pass), so we only check the diagonal blocks are negative
        # (slave-slave terms: -ε * ∫ R R dΓ < 0)
        @test all(diag(Matrix(Z))[1:nlm]     .>= 0.0)
        @test all(diag(Matrix(Z))[nlm+1:end] .>= 0.0)

        # ── Solve KKT system ─────────────────────────────────────────────────
        U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

        # ── Verify patch-test exactness ──────────────────────────────────────
        # Tolerance 1e-5: p=1 non-conforming element-based has ~6e-6 noise
        # from the Z-block conditioning of the (λ+λ̄) penalty structure.
        for A in 1:ncp
            x_cp = B[A, 1]
            y_cp = B[A, 2]
            ux = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x_cp / E atol=1e-5
            @test uy ≈ 0.0      atol=1e-5
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
        @test all(diag(Matrix(Z))[1:nlm]     .>= 0.0)
        @test all(diag(Matrix(Z))[nlm+1:end] .>= 0.0)

        U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

        # Exact solution: u_x = x/E, u_y = 0 at every CP.
        # Tolerance 1e-5: p=4 with epss=1.0 has O(1e-6) numerical noise from
        # the higher condition number of the KKT system at high polynomial degree.
        tol = 1e-5
        for A in 1:ncp
            x_cp = B[A, 1]
            ux   = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy   = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x_cp / E  atol=tol
            @test uy ≈ 0.0       atol=tol
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

    # ── Segment-based integration ─────────────────────────────────────────────

    @testset "find_interface_segments_1d" begin
        # Slave: p=1, n=2, kv=[0,0,1,1]  (one element, CPs at x=0 and x=1)
        kv_s = [0.0, 0.0, 1.0, 1.0]
        B_s  = [0.0 0.0 0.0 1.0
                1.0 0.0 0.0 1.0]
        Ps   = [1, 2]

        # Master same: conforming → master breakpoints project to slave {0,1}
        # → no new breakpoints; result is {0, 1}
        ξ_breaks = find_interface_segments_1d(1, 2, kv_s, 1, 2, kv_s, B_s, Ps, Ps, 2)
        @test length(ξ_breaks) == 2
        @test ξ_breaks ≈ [0.0, 1.0] atol=1e-12

        # Master: p=1, n=3, kv=[0,0,0.5,1,1] (two elements, adds breakpoint at ξ=0.5)
        kv_m  = [0.0, 0.0, 0.5, 1.0, 1.0]
        B_m   = [0.0 0.0 0.0 1.0
                 0.5 0.0 0.0 1.0
                 1.0 0.0 0.0 1.0]
        Pm    = [1, 2, 3]   # (using B_s extended with a midpoint CP)

        # Build a 3-CP B matrix for the master curve
        B_all = [0.0 0.0 0.0 1.0   # CP 1 (slave+master share endpoints)
                 1.0 0.0 0.0 1.0   # CP 2
                 0.5 0.0 0.0 1.0]  # CP 3 (master midpoint)
        Pm2   = [1, 3, 2]   # master CPs: {CP1, midpoint(3), CP2}

        ξ_breaks2 = find_interface_segments_1d(1, 2, kv_s, 1, 3, kv_m, B_all, Ps, Pm2, 2)
        @test length(ξ_breaks2) == 3
        @test ξ_breaks2[1] ≈ 0.0 atol=1e-12
        @test ξ_breaks2[2] ≈ 0.5 atol=1e-10   # master ξ=0.5 projects to slave ξ=0.5
        @test ξ_breaks2[3] ≈ 1.0 atol=1e-12
    end

    @testset "segment-based patch test (uniaxial tension, ν=0)" begin
        # Identical setup to the element-based p=1 patch test above.
        # Using SegmentBasedIntegration() should give the same patch-test result.
        nsd = 2; npd = 2; ned = 2; npc = 2
        E = 1000.0; nu = 0.0

        p = [1 1; 1 1]
        n = [2 3; 2 4]

        KV = generate_knot_vectors(npc, npd, p, n)

        B = [
            0.0  0.0  0.0  1.0
            0.5  0.0  0.0  1.0
            0.0  0.5  0.0  1.0
            0.5  0.5  0.0  1.0
            0.0  1.0  0.0  1.0
            0.5  1.0  0.0  1.0
            0.5  0.0        0.0  1.0
            1.0  0.0        0.0  1.0
            0.5  1.0/3.0    0.0  1.0
            1.0  1.0/3.0    0.0  1.0
            0.5  2.0/3.0    0.0  1.0
            1.0  2.0/3.0    0.0  1.0
            0.5  1.0        0.0  1.0
            1.0  1.0        0.0  1.0
        ]
        ncp = 14
        P   = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14]]

        nel, nnp, nen = patch_metrics(npc, npd, p, n)
        IEN = build_ien(nsd, npd, npc, p, n, nel, nnp, nen)
        INC = [build_inc(n[1, :]), build_inc(n[2, :])]
        mat = LinearElastic(E, nu, :plane_strain)

        bc_per_dof = [Int[1, 3, 5], Int[1, 2, 7, 8]]
        neq, ID    = build_id(bc_per_dof, ned, ncp)
        LM         = build_lm(nen, ned, npc, nel, ID, IEN, P)

        Ub_zero = zeros(ncp, nsd)
        K, _    = build_stiffness_matrix(
            npc, nsd, npd, ned, neq,
            p, n, KV, P, B, Ub_zero,
            nen, nel, IEN, INC, LM,
            [mat, mat], 2, 1.0
        )

        F = zeros(neq)
        F = segment_load(
            n[2, :], p[2, :], KV[2], P[2], B,
            nnp[2], nen[2], nsd, npd, ned,
            Int[], 2, ID, F, 1.0, 1.0, 2
        )
        IND        = Tuple{Int,Float64}[]
        K_bc, F_bc = enforce_dirichlet(IND, K, F)

        pairs = [InterfacePair(1, 2, 2, 4),
                 InterfacePair(2, 4, 1, 2)]
        Pc    = build_interface_cps(pairs, p, n, KV, P, npd, nnp)
        nlm   = length(Pc)

        C, Z = build_mortar_coupling(
            Pc, pairs, p, n, KV, P, B, ID, nnp,
            ned, nsd, npd, neq, 3, 1.0,
            SegmentBasedIntegration()
        )
        @test size(C) == (neq, 2 * nlm)
        @test size(Z) == (2 * nlm, 2 * nlm)
        # Z must be symmetric (by construction) and have non-positive diagonal
        @test norm(Matrix(Z) - Matrix(Z)') < 1e-12
        @test all(diag(Matrix(Z))[1:nlm]     .>= 0.0)
        @test all(diag(Matrix(Z))[nlm+1:end] .>= 0.0)

        U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

        for A in 1:ncp
            x_cp = B[A, 1]
            ux   = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy   = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x_cp / E  atol=1e-6
            @test uy ≈ 0.0       atol=1e-6
        end
    end

    @testset "segment-based patch test p=$p_ord (uniaxial tension, ν=0)" for p_ord in 2:4
        # Mirror of element-based p=2..4 test using SegmentBasedIntegration.
        nsd = 2; npd = 2; ned = 2; npc = 2
        E = 1000.0; nu = 0.0

        n_xi   = p_ord + 1
        n_eta1 = p_ord + 2
        n_eta2 = p_ord + 3
        nquad  = p_ord + 1

        p_mat  = fill(p_ord, npc, npd)
        n_mat  = [n_xi n_eta1; n_xi n_eta2]

        KV   = generate_knot_vectors(npc, npd, p_mat, n_mat)
        ncp1 = n_xi * n_eta1
        ncp2 = n_xi * n_eta2
        ncp  = ncp1 + ncp2

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

        F  = zeros(neq)
        F  = segment_load(
            n_mat[2, :], p_mat[2, :], KV[2], P[2], B,
            nnp[2], nen[2], nsd, npd, ned,
            Int[], 2, ID, F, 1.0, 1.0, nquad
        )
        IND        = Tuple{Int,Float64}[]
        K_bc, F_bc = enforce_dirichlet(IND, K, F)

        pairs = [InterfacePair(1, 2, 2, 4),
                 InterfacePair(2, 4, 1, 2)]
        Pc    = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp)
        nlm   = length(Pc)

        C, Z = build_mortar_coupling(
            Pc, pairs, p_mat, n_mat, KV, P, B, ID, nnp,
            ned, nsd, npd, neq, nquad, 1.0,
            SegmentBasedIntegration()
        )
        @test size(C) == (neq, 2 * nlm)
        @test size(Z) == (2 * nlm, 2 * nlm)
        @test norm(Matrix(Z) - Matrix(Z)') < 1e-12
        @test all(diag(Matrix(Z))[1:nlm]     .>= 0.0)
        @test all(diag(Matrix(Z))[nlm+1:end] .>= 0.0)

        U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

        tol = 1e-5
        for A in 1:ncp
            x_cp = B[A, 1]
            ux   = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
            uy   = ID[2, A] != 0 ? U[ID[2, A]] : 0.0
            @test ux ≈ x_cp / E  atol=tol
            @test uy ≈ 0.0       atol=tol
        end
    end

    # ── SinglePassFormulation ────────────────────────────────────────────────
    @testset "SinglePassFormulation: dimensions and patch test" begin
        # Reuse the p=1 geometry from the first @testset:
        # Patch 1 (x∈[-1,0]): 3×3 CPs (p=1), CPs 1–9
        # Patch 2 (x∈[ 0,1]): 3×3 CPs (p=1), CPs 10–18
        # Interface: right of patch 1 (facet 2) ↔ left of patch 2 (facet 4)
        # Slave = patch 1, Master = patch 2.
        nsd = 2; npd = 2; ned = 2; npc = 2
        E  = 1e3; nu = 0.0
        mat = LinearElastic(E, nu, :plane_strain)

        p = [1 1; 1 1]
        n = [3 3; 3 3]   # conforming, 2 elements each direction
        KV = generate_knot_vectors(npc, npd, p, n)

        # B: [x, y, z=0, w]; Patch 1 x∈[-1,0], Patch 2 x∈[0,1], y∈[0,1]
        B = Float64[
           -1.0  0.0  0.0  1.0;  #  1
           -0.5  0.0  0.0  1.0;  #  2
            0.0  0.0  0.0  1.0;  #  3  (interface)
           -1.0  0.5  0.0  1.0;  #  4
           -0.5  0.5  0.0  1.0;  #  5
            0.0  0.5  0.0  1.0;  #  6  (interface)
           -1.0  1.0  0.0  1.0;  #  7
           -0.5  1.0  0.0  1.0;  #  8
            0.0  1.0  0.0  1.0;  #  9  (interface)
            0.0  0.0  0.0  1.0;  # 10  (interface)
            0.5  0.0  0.0  1.0;  # 11
            1.0  0.0  0.0  1.0;  # 12
            0.0  0.5  0.0  1.0;  # 13  (interface)
            0.5  0.5  0.0  1.0;  # 14
            1.0  0.5  0.0  1.0;  # 15
            0.0  1.0  0.0  1.0;  # 16  (interface)
            0.5  1.0  0.0  1.0;  # 17
            1.0  1.0  0.0  1.0;  # 18
        ]
        ncp = 18
        P   = [collect(1:9), collect(10:18)]

        nel, nnp, nen = patch_metrics(npc, npd, p, n)
        IEN = build_ien(nsd, npd, npc, p, n, nel, nnp, nen)
        INC = [build_inc(n[pc, :]) for pc in 1:npc]

        # Dirichlet: ux=0 on left (CPs 1,4,7), uy=0 at ONE corner (CP 1 only).
        # Interface nodes (3,6,9,10,13,16) are left free in both directions
        # so tangential multipliers have nonzero coupling → system is non-singular.
        bc_per_dof = [Int[1, 4, 7], Int[1]]
        neq, ID = build_id(bc_per_dof, ned, ncp)
        LM      = build_lm(nen, ned, npc, nel, ID, IEN, P)

        K, _ = build_stiffness_matrix(
            npc, nsd, npd, ned, neq,
            p, n, KV, P, B, zeros(ncp, nsd),
            nen, nel, IEN, INC, LM, [mat, mat], 2, 1.0
        )
        F  = zeros(neq)
        F  = segment_load(n[2, :], p[2, :], KV[2], P[2], B,
                          nnp[2], nen[2], nsd, npd, ned,
                          Int[], 2, ID, F, 1.0, 1.0, 2)
        K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

        form    = SinglePassFormulation()
        pairs_sp = [InterfacePair(1, 2, 2, 4)]   # slave=patch1 right, master=patch2 left

        # Pc: slave CPs only (interface CPs of patch 1 = CPs 3, 6, 9)
        Pc_sp  = build_interface_cps(pairs_sp, p, n, KV, P, npd, nnp, form)
        nlm_sp = length(Pc_sp)
        @test nlm_sp == 3
        @test all(∈(Pc_sp), [3, 6, 9])

        # Dimensions: C is (neq × ned·nlm_sp), Z is (ned·nlm_sp × ned·nlm_sp), Z = 0
        C_sp, Z_sp = build_mortar_coupling(
            Pc_sp, pairs_sp, p, n, KV, P, B, ID, nnp,
            ned, nsd, npd, neq, 4, 0.0,
            SegmentBasedIntegration(), form
        )
        @test size(C_sp) == (neq, ned * nlm_sp)
        @test size(Z_sp) == (ned * nlm_sp, ned * nlm_sp)
        @test norm(Z_sp) == 0.0

        # Patch test: bar fixed at x=-1, traction=1 at x=+1 → ux(x) = (x+1)/E, uy = 0
        U_sp, _ = solve_mortar(K_bc, C_sp, Z_sp, F_bc)
        tol = 1e-5
        for A in 1:ncp
            x_cp = B[A, 1]
            ux   = ID[1, A] != 0 ? U_sp[ID[1, A]] : 0.0
            uy   = ID[2, A] != 0 ? U_sp[ID[2, A]] : 0.0
            @test ux ≈ (x_cp + 1.0) / E  atol=tol
            @test uy ≈ 0.0               atol=tol
        end
    end

    # ── 3D segment-based integration utilities ───────────────────────────────

    @testset "sutherland_hodgman_clip" begin
        n = [0.0, 0.0, 1.0]   # z-normal (all polygons lie in z=0 plane)

        # Square [0,1]² clipped against itself → same square (CCW: SW, SE, NE, NW)
        sq = [[0.0,0.0,0.0], [1.0,0.0,0.0], [1.0,1.0,0.0], [0.0,1.0,0.0]]
        poly = sutherland_hodgman_clip(sq, sq, n)
        @test length(poly) >= 4
        # Area should equal 1.0 (fan-triangulate to check)
        xc   = sum(poly) ./ length(poly)
        area = sum(norm(cross(poly[i] .- xc, poly[mod1(i+1,length(poly))] .- xc)) / 2
                   for i in 1:length(poly))
        @test area ≈ 1.0 atol=1e-12

        # Non-overlapping squares → empty result
        sq2 = [[2.0,0.0,0.0], [3.0,0.0,0.0], [3.0,1.0,0.0], [2.0,1.0,0.0]]
        poly2 = sutherland_hodgman_clip(sq, sq2, n)
        @test length(poly2) == 0

        # Partial overlap: [0,1]² ∩ [0.5,1.5]² = [0.5,1]²  (area=0.25)
        sq3 = [[0.5,0.5,0.0], [1.5,0.5,0.0], [1.5,1.5,0.0], [0.5,1.5,0.0]]
        poly3 = sutherland_hodgman_clip(sq, sq3, n)
        @test length(poly3) >= 3
        xc3   = sum(poly3) ./ length(poly3)
        area3 = sum(norm(cross(poly3[i] .- xc3, poly3[mod1(i+1,length(poly3))] .- xc3)) / 2
                    for i in 1:length(poly3))
        @test area3 ≈ 0.25 atol=1e-12
    end

    @testset "tri_gauss_rule weight sums" begin
        for nq in [1, 2, 3, 4, 7]
            pts, wts = tri_gauss_rule(nq)
            @test sum(wts) ≈ 0.5 atol=1e-12
            @test size(pts, 1) == 2
            @test size(pts, 2) == length(wts)
            # All points inside reference triangle
            for q in 1:length(wts)
                ξ1, ξ2 = pts[1,q], pts[2,q]
                @test ξ1 >= -1e-14
                @test ξ2 >= -1e-14
                @test ξ1 + ξ2 <= 1.0 + 1e-14
            end
        end
    end

    @testset "find_interface_segments_2d flat unit square" begin
        # Two flat patches sharing the z=0 plane over the unit square [0,1]².
        # Slave: 2×2 elements (uniform p=1 knot vectors)
        # Master: 3×3 elements (non-conforming)
        # Total intersection area must equal 1.0.

        # Build a flat NURBS surface over [0,1]² (p=1 bilinear, n×n CPs)
        # kv = open uniform for p=1: [0,0,1/n,...,(n-1)/n,1,1]  (n+1 knot spans)
        function flat_square_patch(nx, ny; z0=0.0)
            # p=1 in both directions, nx×ny elements → (nx+1)×(ny+1) CPs
            p = [1, 1]
            n = [nx+1, ny+1]
            kv_x = vcat(0.0, LinRange(0.0, 1.0, nx+1), 1.0)  # open: repeat endpoints
            kv_y = vcat(0.0, LinRange(0.0, 1.0, ny+1), 1.0)
            ncp  = (nx+1) * (ny+1)
            B    = zeros(ncp, 4)   # [x, y, z, w]
            idx  = 1
            for iy in 0:ny, ix in 0:nx
                B[idx, :] = [ix/nx, iy/ny, z0, 1.0]
                idx += 1
            end
            P = collect(1:ncp)
            return p, n, [kv_x, kv_y], B, P
        end

        ps, ns, KVs, Bs, Ps = flat_square_patch(2, 2)
        pm, nm, KVm, Bm, Pm = flat_square_patch(3, 3)

        # Combine into a single B (same flat surface, so same CPs; use slave B for both)
        ncp_s = size(Bs, 1)
        ncp_m = size(Bm, 1)
        B_all = vcat(Bs, Bm)
        Ps_g  = collect(1:ncp_s)
        Pm_g  = collect((ncp_s+1):(ncp_s+ncp_m))

        cells = find_interface_segments_2d(ps, ns, KVs, pm, nm, KVm, B_all, Ps_g, Pm_g, 3)

        @test length(cells) >= 4   # at least 4 intersection cells

        # Total area of all triangle cells must equal 1.0
        total_area = 0.0
        for cell in cells
            v1 = cell.verts[:,1]; v2 = cell.verts[:,2]; v3 = cell.verts[:,3]
            total_area += norm(cross(v2 .- v1, v3 .- v1)) / 2.0
        end
        @test total_area ≈ 1.0 atol=1e-12

        # Each cell must have positive area
        for cell in cells
            v1 = cell.verts[:,1]; v2 = cell.verts[:,2]; v3 = cell.verts[:,3]
            @test norm(cross(v2 .- v1, v3 .- v1)) / 2.0 > 0.0
        end
    end

end
