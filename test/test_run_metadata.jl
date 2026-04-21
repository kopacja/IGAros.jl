using Test
using TOML
using IGAros

@testset "RunMetadata" begin

    @testset "capture_code_provenance" begin
        code = capture_code_provenance()
        @test haskey(code, "igaros_commit")
        @test haskey(code, "igaros_branch")
        @test haskey(code, "igaros_dirty")
        @test haskey(code, "julia_version")
        @test isa(code["igaros_dirty"], Bool)
        @test code["julia_version"] == string(VERSION)
        # In the repo we run tests from, git must resolve; short SHA is ≥ 7 chars
        @test length(code["igaros_commit"]) >= 7
    end

    @testset "capture_cluster_env off-cluster fallback" begin
        # Clear SLURM env for this test regardless of outer shell state
        cluster = withenv(
            "SLURM_JOB_ID"        => nothing,
            "SLURM_CPUS_PER_TASK" => nothing,
            "SLURM_JOB_PARTITION" => nothing,
            "SLURMD_NODENAME"     => nothing,
        ) do
            capture_cluster_env(wallclock_seconds = 42)
        end
        @test cluster["slurm_jobid"] == 0
        @test cluster["cpus_per_task"] == 0
        @test cluster["partition"] == ""
        @test cluster["wallclock_seconds"] == 42
        @test cluster["threads"] == Threads.nthreads()
        @test !isempty(cluster["host"])
    end

    @testset "capture_cluster_env reads SLURM env" begin
        cluster = withenv(
            "SLURM_JOB_ID"        => "123456",
            "SLURM_CPUS_PER_TASK" => "8",
            "SLURM_JOB_PARTITION" => "express",
            "SLURMD_NODENAME"     => "kraken-m2",
        ) do
            capture_cluster_env(wallclock_seconds = 3600)
        end
        @test cluster["slurm_jobid"] == 123456
        @test cluster["cpus_per_task"] == 8
        @test cluster["partition"] == "express"
        @test cluster["host"] == "kraken-m2"
        @test cluster["wallclock_seconds"] == 3600
    end

    @testset "write_meta_toml round-trip" begin
        mktempdir() do dir
            path = write_meta_toml(
                dir;
                benchmark = "unit_test",
                description = "write_meta_toml round-trip",
                parameters = Dict(
                    "polynomial_orders" => [2, 3, 4],
                    "epsilon_default" => 1.0e-8,
                    "methods" => ["TM-ME", "TM-MS"],
                ),
                outputs = ["factorial.csv", "eps_sweep.csv"],
                wallclock_seconds = 10,
            )
            @test isfile(path)
            @test path == joinpath(dir, "meta.toml")

            data = TOML.parsefile(path)

            @test data["run"]["benchmark"] == "unit_test"
            @test data["run"]["description"] == "write_meta_toml round-trip"
            @test data["run"]["run_id"] == basename(dir)
            @test haskey(data["run"], "date")
            @test haskey(data["run"], "time")

            @test haskey(data, "code")
            @test haskey(data["code"], "igaros_commit")
            @test data["code"]["julia_version"] == string(VERSION)

            @test haskey(data, "cluster")
            @test data["cluster"]["wallclock_seconds"] == 10

            @test data["parameters"]["polynomial_orders"] == [2, 3, 4]
            @test data["parameters"]["epsilon_default"] == 1.0e-8
            @test data["parameters"]["methods"] == ["TM-ME", "TM-MS"]

            @test data["outputs"]["csvs"] == ["factorial.csv", "eps_sweep.csv"]
            @test data["outputs"]["figures_regenerated_by"] == "plot.jl"
        end
    end

    @testset "write_meta_toml accepts extras" begin
        mktempdir() do dir
            write_meta_toml(
                dir;
                benchmark = "with_extras",
                description = "check extras pass-through",
                extras = Dict(
                    "geometry" => Dict("L" => 10.0, "H" => 4.0),
                    "material" => Dict("E" => 100.0, "nu" => 0.0),
                ),
            )
            data = TOML.parsefile(joinpath(dir, "meta.toml"))
            @test data["geometry"]["L"] == 10.0
            @test data["geometry"]["H"] == 4.0
            @test data["material"]["E"] == 100.0
            @test data["material"]["nu"] == 0.0
        end
    end

    @testset "write_meta_toml minimal (empty parameters and outputs)" begin
        mktempdir() do dir
            write_meta_toml(dir; benchmark = "min", description = "minimal")
            data = TOML.parsefile(joinpath(dir, "meta.toml"))
            @test isempty(data["parameters"])
            @test data["outputs"]["csvs"] == String[]
        end
    end

    @testset "write_meta_toml errors on nonexistent run_dir" begin
        missing_dir = joinpath(tempdir(), "definitely_not_there_" * string(time_ns()))
        @test_throws ErrorException write_meta_toml(
            missing_dir;
            benchmark = "x",
            description = "x",
        )
    end

end
