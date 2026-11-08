# Atomic parallel chunk precomputer for `run_incremental_chunks.jl`.
#
# Traces chunk `k` in `$(base_prefix)_chunk_$(k)_tmp`, distills it into
# `$(base_prefix)_chunk_$(k)_paper/paper_analysis.jld2`, removes raw trajectory
# batches, and atomically renames `_tmp` -> `$(base_prefix)_chunk_$(k)` so the
# master incremental runner (`run_incremental_chunks.jl`) picks it up in O(1) time.

include(joinpath(@__DIR__, "..", "monte_carlo_forward_tracing", "monte_carlo_shell.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo_forward_tracing", "analyze_paper_results.jl"))

using .ShellMonteCarlo
using MarsTP, JLD2, TOML, Printf, Random

function run_precompute_worker(
        base_prefix::AbstractString,
        k::Int,
        per_cell::Int,
        cell_stride::Int,
        fields_path::AbstractString,
    )
    cdir = @sprintf("%s_chunk_%02d", base_prefix, k)
    tmp_dir = @sprintf("%s_chunk_%02d_tmp_%d", base_prefix, k, getpid())
    c_paper = @sprintf("%s_chunk_%02d_paper", base_prefix, k)
    c_paper_tmp = @sprintf("%s_chunk_%02d_paper_tmp_%d", base_prefix, k, getpid())

    if isdir(cdir)
        println("Chunk directory $cdir already exists (handled by master or another worker); exiting.")
        return nothing
    end

    chunk_seed = Int(rand(Random.RandomDevice(), UInt64) & 0x7fffffffffffffff)
    cfg = ShellMonteCarlo.Config(;
        altitude_km = 400.0,
        absorption_altitude_km = 200.0,
        cell_stride,
        per_cell,
        seed = chunk_seed,
        random_seed = false,
        tmax = 1000.0,
        dt = 0.1,
        batch_size = 2048,
        fields_path = String(fields_path),
        include_gradpe_work = true,
        save_trajectories = :roi_and_sample,
        save_stride = 5,
        trajectory_sample_size = 20_000,
        packed_trajectories = true,
        trajectory_float32 = true,
        skip_inner_and_trapped = true,
        chunk_id = k,
    )

    println("=== Precomputing Chunk $k -> $tmp_dir (seed=$chunk_seed, per_cell=$per_cell) ===")
    flush(stdout)
    ShellMonteCarlo.run_monte_carlo(tmp_dir, cfg)

    if isdir(cdir)
        println("Master started $cdir while precomputing; cleaning up $tmp_dir and exiting.")
        rm(tmp_dir; recursive = true, force = true)
        return nothing
    end

    println("=== Distilling Chunk $k -> $c_paper_tmp ===")
    flush(stdout)
    run_paper_analysis(tmp_dir, c_paper_tmp; render_plots = false)

    # Remove raw trajectory batches for k >= 2
    if k >= 2
        for fpath in filter(p -> occursin(r"^trajectories_.*\.jld2$", basename(p)), readdir(tmp_dir; join = true))
            rm(fpath; force = true)
        end
    end

    if !isdir(cdir) && !isdir(c_paper)
        mv(c_paper_tmp, c_paper)
        mv(tmp_dir, cdir)
        println("=== Atomically published Chunk $k ($cdir and $c_paper) ===")
    else
        rm(c_paper_tmp; recursive = true, force = true)
        rm(tmp_dir; recursive = true, force = true)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    base_prefix = ARGS[1]
    k = parse(Int, ARGS[2])
    per_cell = parse(Int, ARGS[3])
    cell_stride = parse(Int, ARGS[4])
    fields_path = ARGS[5]
    run_precompute_worker(base_prefix, k, per_cell, cell_stride, fields_path)
end
