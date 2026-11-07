using JLD2, TOML
@testset "Shared streaming PSD and batch IO" begin
    # Keep diagnostic files; this test never recursively deletes directories.
    folder = mktempdir(; cleanup = false)
    a = (; t = [0.0, 1.0, 2.0], u = [SA[-2.0, 0.0, 0.0, -4.0, 0.0, 0.0], SA[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], SA[2.0, 0.0, 0.0, 4.0, 0.0, 0.0]])
    b = (; t = [0.0, 2.0], u = [SA[-2.0, 1.0, 0.0, 1.0, 0.0, 0.0], SA[2.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
    c = (; t = [0.0], u = [a.u[1]])
    kw = (; detector_m = zeros(3), side_m = 2.0, vlim = 2.0, vgrid = 4)
    mem = forward_psd([a, b, c]; kw..., rate_weights_s = [16.0, 8.0, 0.0])
    for (i, traj) in enumerate((a, b, c))
        path = joinpath(folder, "trajectories_$(lpad(i, 5, '0')).jld2")
        write_trajectory_batch(
            path, [traj]; particle_ids = [i], rate_weights_s = [[16.0, 8.0, 0.0][i]],
            source_density_weights_m3 = [1.0], cell_ids = [i], termination_codes = ["time_limit"]
        )
    end
    open(joinpath(folder, "metadata.toml"), "w") do io
        TOML.print(io, Dict("n_particles" => 3, "species" => "O2+"))
    end
    open(joinpath(folder, "completion.toml"), "w") do io
        TOML.print(io, Dict("complete" => true))
    end
    disk = forward_psd_saved(folder; kw...)
    @test disk.psd == mem.psd
    @test disk.residence_s == mem.residence_s
    @test disk.outside_vlim_residence_s == mem.outside_vlim_residence_s
    @test disk.particle_ids == [1, 2, 3]
    @test disk.retcodes == fill("time_limit", 3)
    @test disk.density_total_m3 == mem.density_total_m3
    for option in (:xy, :xz, :yz)
        @test forward_psd_saved(folder; kw..., option).psd == forward_psd([a, b, c]; kw..., rate_weights_s = [16.0, 8.0, 0.0], option).psd
    end
    sparse = forward_psd_saved(folder; kw..., storage = :sparse)
    @test all(mem.psd[k...] == v for (k, v) in sparse.psd)
    @test length(sparse.psd) == count(>(0), mem.psd)
    acc = ForwardPSDAccumulator(; kw...)
    observed = []
    accumulate_forward_psd!(acc, a; rate_weight_s = 16.0, segment_observer = x -> push!(observed, x))
    @test length(observed) == 2
    @test observed[1].v0_ms == SA[-2.0, 0.0, 0.0]
    @test observed[2].v1_ms == SA[2.0, 0.0, 0.0]
    @test sum(x.t1_s - x.t0_s for x in observed) == acc.residence[1]
    @test all(n == 1 for n in values(acc.unique)) # time samples not independent particles
    @test finish_forward_psd(acc).psd == finish_forward_psd(acc).psd # non-mutating
    firstfile = joinpath(folder, "trajectories_00001.jld2")
    @test_throws ArgumentError write_trajectory_batch(firstfile, [a]; particle_ids = [1], rate_weights_s = [1.0])
    @test_throws ArgumentError foreach_saved_trajectory(_ -> nothing, [firstfile, firstfile])
    @test_throws ArgumentError forward_psd_saved(folder; kw..., species = "H+")
    @test_throws ArgumentError write_trajectory_batch(joinpath(folder, "bad.jld2"), [a]; particle_ids = [1], rate_weights_s = [-1.0])
    @test !isfile(joinpath(folder, "bad.jld2"))
    incomplete = joinpath(folder, "incomplete.jld2")
    jldopen(incomplete, "w") do f
        f["format_version"] = 1
    end
    @test_throws ArgumentError foreach_saved_trajectory(_ -> nothing, [incomplete])
    legacy = joinpath(folder, "legacy.jld2")
    jldopen(legacy, "w") do f
        f["p9/state"] = vcat(permutedims(a.t), reduce(hcat, a.u));f["p9/rate_weight_s1"] = 16.0
    end
    @test forward_psd_saved([legacy]; kw...).psd == forward_psd(a; kw..., rate_weights_s = [16.0]).psd
    compressed = joinpath(folder, "compressed.jld2")
    write_trajectory_batch(compressed, [a]; particle_ids = [10], rate_weights_s = [16.0], compress = true)
    @test forward_psd_saved([compressed]; kw...).psd == forward_psd(a; kw..., rate_weights_s = [16.0]).psd
    println("IO test artifacts: $folder")
end

@testset "Saved component work" begin
    folder = mktempdir(; cleanup = false)
    sp = MarsTP.TP.SpeciesDict["O2+"]; E = 1.0e-5; a = sp.q * E / sp.m
    t = [0.0, 0.1, 0.2, 0.237] # clipped final interval
    tr = (; t, u = [SA[0.5a * s * s, 2.0, 0.0, a * s, 0.0, 0.0] for s in t])
    itp = MarsTP.FieldWorkInterpolators(_ -> SA[E, 0.0, 0.0], _ -> SA[2E, 0.0, 0.0], _ -> SA[-E, 0.0, 0.0])
    w = trajectory_work(tr, itp; power = true)
    @test sum(w.increments[:, 1]) ≈ w.summary.delta_kinetic_eV
    @test w.summary.positive_conv_eV ≈ 2w.summary.total_eV
    @test w.summary.negative_hall_eV ≈ -w.summary.total_eV
    @test w.summary.max_abs_energy_residual_eV < 1.0e-12
    @test cumsum(w.increments[:, 1]) ≈ Electric_field_work_profile(tr, itp).total_eV[2:end]
    @test trajectory_work(tr, itp; mode = :summary).summary == w.summary
    @test trajectory_work(tr, itp; mode = :summary).increments === nothing
    @test w.powers[:, 1] ≈ E * a * t
    one = (; t = [0.0], u = [first(tr.u)])
    @test size(trajectory_work(one, itp).increments) == (0, 3)
    @test trajectory_work(one, itp).summary.total_eV == 0
    for mode in (:steps, :summary)
        path = joinpath(folder, "$(mode).jld2")
        write_trajectory_batch(
            path, [tr, one]; particle_ids = [1, 2], rate_weights_s = [3.0, 0.0],
            work_itp = itp, work_mode = mode, save_power = true, termination_codes = ["time_limit", "zero_rate"]
        )
        records = []
        foreach_saved_trajectory(x -> push!(records, x), path)
        @test records[1].work.summary["total_eV"] ≈ w.summary.total_eV
        @test records[1].work.powers["total"] ≈ w.powers[:, 1]
        @test records[2].work.summary["total_eV"] == 0
        if mode == :steps
            @test records[1].work.steps["hall"] ≈ w.increments[:, 3]
            @test isempty(records[2].work.steps["total"])
            @test records[1].work.steps["total"][end] / diff(t)[end] ≈ E * a * (t[end] + t[end - 1]) / 2
        else
            @test isempty(records[1].work.steps)
        end
    end
    bad = joinpath(folder, "badwork.jld2")
    @test_throws ArgumentError write_trajectory_batch(bad, [tr]; particle_ids = [1], rate_weights_s = [1.0], work_mode = :steps)
    @test !ispath(bad)
    precomputed = joinpath(folder, "precomputed_work.jld2")
    write_trajectory_batch(
        precomputed, [tr]; particle_ids = [1], rate_weights_s = [1.0],
        work_mode = :summary, work_summaries = [w.summary]
    )
    records = []
    foreach_saved_trajectory(x -> push!(records, x), precomputed)
    for (key, value) in pairs(w.summary)
        @test records[1].work.summary[string(key)] ≈ value
    end
    @test isempty(records[1].work.steps)
    @test_throws ArgumentError write_trajectory_batch(
        bad, [tr]; particle_ids = [1], rate_weights_s = [1.0],
        work_mode = :steps, work_summaries = [w.summary]
    )
    @test_throws ArgumentError write_trajectory_batch(
        bad, [tr]; particle_ids = [1], rate_weights_s = [1.0],
        work_mode = :summary, work_summaries = [merge(w.summary, (; total_eV = NaN))]
    )
    @test !ispath(bad)

    packed_path = joinpath(folder, "packed_f32.jld2")
    itp4 = MarsTP.FieldWorkInterpolators(_ -> SA[E, 0.0, 0.0], _ -> SA[2E, 0.0, 0.0], _ -> SA[-0.5E, 0.0, 0.0], _ -> SA[-0.5E, 0.0, 0.0], nothing)
    write_trajectory_batch(
        packed_path, [tr, one]; particle_ids = [1, 2], rate_weights_s = [3.0, 0.0],
        source_density_weights_m3 = [1.5, 0.0], cell_ids = [11, 12],
        work_itp = itp4, work_mode = :steps, save_power = true, termination_codes = ["outer", "zero_rate"],
        packed = true, state_eltype = Float32
    )
    packed_records = []
    foreach_saved_trajectory(x -> push!(packed_records, x), packed_path)
    @test length(packed_records) == 2
    @test packed_records[1].trajectory.t ≈ tr.t
    @test packed_records[1].work.summary["total_eV"] ≈ w.summary.total_eV rtol = 1.0e-6
    @test packed_records[1].work.steps["gradpe"] ≈ -0.5 .* w.increments[:, 1] rtol = 1.0e-5
    @test packed_records[1].work.powers["total"] ≈ w.powers[:, 1] rtol = 1.0e-5
    loaded = load_trajectories(packed_path; species = "O2+", max_points = 10)
    @test length(loaded) == 1
    @test loaded[1]["times_s"] ≈ tr.t

    c1 = joinpath(folder, "chunk1"); c2 = joinpath(folder, "chunk2")
    mkpath(c1); mkpath(c2)
    write_trajectory_batch(joinpath(c1, "trajectories_00001.jld2"), [tr]; particle_ids = [1], rate_weights_s = [4.0], packed = true, state_eltype = Float32)
    write_trajectory_batch(joinpath(c2, "trajectories_00001.jld2"), [tr]; particle_ids = [101], rate_weights_s = [6.0], packed = true, state_eltype = Float32)
    open(joinpath(c1, "metadata.toml"), "w") do io
        TOML.print(io, Dict("n_particles" => 1, "species" => "O2+", "seed" => 20260906, "chunk_id" => 1))
    end
    open(joinpath(c2, "metadata.toml"), "w") do io
        TOML.print(io, Dict("n_particles" => 1, "species" => "O2+", "seed" => 20260906, "chunk_id" => 2))
    end
    open(joinpath(c1, "completion.toml"), "w") do io
        TOML.print(io, Dict("complete" => true, "n_saved_trajectories" => 1))
    end
    open(joinpath(c2, "completion.toml"), "w") do io
        TOML.print(io, Dict("complete" => true, "n_saved_trajectories" => 1))
    end
    pooled = []
    foreach_saved_trajectory(x -> push!(pooled, x), [c1, c2])
    @test length(pooled) == 2
    @test [r.rate_weight_s for r in pooled] ≈ [2.0, 3.0]
    @test_throws ArgumentError foreach_saved_trajectory(_ -> nothing, [c1, c1])
end
