# Incremental multi-chunk Monte Carlo runner and statistical convergence analyzer.
#
# Each chunk k = 1..n_chunks uses `chunk_id = k`, which guarantees:
#   1. Disjoint per-cell Xoshiro RNG streams: `seed + (k - 1) * seed_stride + cellid`
#   2. Non-overlapping global particle IDs: `(k - 1) * N_chunk + 1 .. k * N_chunk`
#   3. Automatic 1/K rate-weight averaging when pooling K chunks in `foreach_saved_trajectory`
#
# Usage:
#   julia --threads=8 --project=. examples/forward_tracing/monte_carlo_forward_tracing/run_incremental_chunks.jl [base_prefix] [n_chunks] [per_cell] [cell_stride] [save_mode] [rel_tol]

include("monte_carlo_shell.jl")
include("analyze_paper_results.jl")

using .ShellMonteCarlo
using MarsTP, JLD2, TOML, Printf, Statistics, LinearAlgebra, Random, PyPlot

function _rel_l2_diff(a::AbstractArray, b::AbstractArray; min_val::Float64 = 0.0)
    num = 0.0
    den = 0.0
    n = 0
    @inbounds for i in eachindex(a, b)
        va, vb = Float64(a[i]), Float64(b[i])
        if isfinite(va) && isfinite(vb) && (abs(va) > min_val || abs(vb) > min_val)
            num += (vb - va)^2
            den += 0.5 * (va^2 + vb^2)
            n += 1
        end
    end
    return (n > 0 && den > 0) ? sqrt(num / den) : NaN
end

function compute_chunk_convergence(prev_jld2::AbstractString, curr_jld2::AbstractString)
    return jldopen(prev_jld2, "r") do f1
        jldopen(curr_jld2, "r") do f2
            orbit_def_rel = _rel_l2_diff(log10.(max.(f1["orbit_def"], 1.0)), log10.(max.(f2["orbit_def"], 1.0)); min_val = 2.0)
            orbit_w_conv_rel = _rel_l2_diff(f1["orbit_w_conv_eV"], f2["orbit_w_conv_eV"]; min_val = 5.0)
            orbit_w_hall_rel = _rel_l2_diff(f1["orbit_w_hall_eV"], f2["orbit_w_hall_eV"]; min_val = 5.0)
            tail_def_rel = _rel_l2_diff(log10.(max.(f1["tail_def"], 1.0)), log10.(max.(f2["tail_def"], 1.0)); min_val = 2.0)
            tail_ux_rel = _rel_l2_diff(f1["tail_bulk_vel_ms"][1, :], f2["tail_bulk_vel_ms"][1, :]; min_val = 100.0)
            tail_w_hall_rel = _rel_l2_diff(f1["tail_w_hall_eV"], f2["tail_w_hall_eV"]; min_val = 5.0)
            fig3_def_rel = _rel_l2_diff(log10.(max.(f1["fig3_omni_def"], 1.0)), log10.(max.(f2["fig3_omni_def"], 1.0)); min_val = 2.0)
            Dict(
                "orbit_log_def_rel_l2" => orbit_def_rel,
                "orbit_w_conv_rel_l2" => orbit_w_conv_rel,
                "orbit_w_hall_rel_l2" => orbit_w_hall_rel,
                "tail_log_def_rel_l2" => tail_def_rel,
                "tail_ux_rel_l2" => tail_ux_rel,
                "tail_w_hall_rel_l2" => tail_w_hall_rel,
                "fig3_log_def_rel_l2" => fig3_def_rel,
            )
        end
    end
end

function plot_convergence_history(history::Vector{Dict{String, Any}}, out_png::AbstractString)
    length(history) >= 2 || return nothing
    ks = [Int(h["cumulative_chunks"]) for h in history[2:end]]
    n_tot = [Int(h["cumulative_particles"]) / 1.0e6 for h in history[2:end]]
    fig, ax = plt.subplots(figsize = (7.5, 4.8), layout = "constrained")
    series = [
        ("orbit_log_def_rel_l2", "Circular Orbit log10(DEF)", "o-"),
        ("orbit_w_conv_rel_l2", "Circular Orbit <ε_C>", "s-"),
        ("tail_log_def_rel_l2", "Cross-Tail log10(DEF)", "^-"),
        ("tail_ux_rel_l2", "Cross-Tail Bulk U_x", "d-"),
        ("tail_w_hall_rel_l2", "Cross-Tail <ε_H>", "v--"),
    ]
    for (key, label, style) in series
        vals = [Float64(get(h["metrics"], key, NaN)) * 100.0 for h in history[2:end]]
        ax.plot(n_tot, vals, style; lw = 1.8, ms = 6, label = label)
    end
    ax.set_xlabel("Cumulative Traced Particles (Millions)")
    ax.set_ylabel("Chunk-to-Chunk Relative L2 Change (%)")
    ax.set_title("Incremental Monte Carlo Statistical Convergence")
    ax.set_yscale("log")
    ax.grid(true; which = "both", alpha = 0.3)
    ax.legend(loc = "best", frameon = false)
    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 200)
    plt.close(fig)
    return out_png
end

function run_incremental_chunks(
        base_prefix::AbstractString = "outputs/mc_400km_crustal";
        n_chunks::Int = 3,
        per_cell::Int = 100,
        cell_stride::Int = 1,
        altitude_km::Float64 = 400.0,
        tmax::Float64 = 1000.0,
        dt::Float64 = 0.1,
        save_trajectories::Symbol = :roi_and_sample,
        save_stride::Int = 5,
        skip_inner_and_trapped::Bool = true,
        fields_path::String = MarsTP.data_path("mars_fields_without_crustal.vts"),
        rel_tol::Float64 = 0.0,
        random_seed::Bool = true,
        base_seed::Int = 20260906,
        keep_raw_batches::Bool = true,
        render_plots::Bool = true,
    )
    chunk_dirs = String[]
    chunk_paper_dirs = String[]
    paper_dirs = String[]
    history = Dict{String, Any}[]
    used_seeds = Set{Int}()
    cum_particles = 0

    for k in 1:n_chunks
        cdir = @sprintf("%s_chunk_%02d", base_prefix, k)
        c_paper = @sprintf("%s_chunk_%02d_paper", base_prefix, k)
        pdir = @sprintf("%s_cum_%02d_paper", base_prefix, k)
        comp_path = joinpath(cdir, "completion.toml")
        if isfile(comp_path) && get(TOML.parsefile(comp_path), "complete", false) === true
            existing_meta = TOML.parsefile(joinpath(cdir, "metadata.toml"))
            s_exist = Int(get(existing_meta, "seed", 0))
            s_exist in used_seeds && error("Duplicate seed $s_exist detected in existing chunk $cdir")
            push!(used_seeds, s_exist)
            println("=== Chunk $k/$n_chunks already complete at $cdir (seed=$s_exist); reusing ===")
        else
            ispath(cdir) && rm(cdir; recursive = true, force = true)
            ispath(c_paper) && rm(c_paper; recursive = true, force = true)
            chunk_seed = if random_seed
                s = 0
                while s <= 0 || (s in used_seeds)
                    s = Int(rand(Random.RandomDevice(), UInt64) & 0x7fffffffffffffff)
                end
                s
            else
                base_seed
            end
            push!(used_seeds, chunk_seed)
            cfg = ShellMonteCarlo.Config(;
                altitude_km,
                absorption_altitude_km = 200.0,
                cell_stride,
                per_cell,
                seed = chunk_seed,
                random_seed = false,
                tmax,
                dt,
                batch_size = 2048,
                fields_path,
                include_gradpe_work = true,
                save_trajectories,
                save_stride,
                trajectory_sample_size = 20_000,
                packed_trajectories = true,
                trajectory_float32 = true,
                skip_inner_and_trapped,
                chunk_id = k,
            )
            println("=== Running Chunk $k/$n_chunks -> $cdir (seed=$chunk_seed, chunk_id=$k, per_cell=$per_cell, cell_stride=$cell_stride) ===")
            ShellMonteCarlo.run_monte_carlo(cdir, cfg)
        end
        push!(chunk_dirs, cdir)
        push!(chunk_paper_dirs, c_paper)
        push!(paper_dirs, pdir)
        meta_k = TOML.parsefile(joinpath(cdir, "metadata.toml"))
        comp_k = TOML.parsefile(comp_path)
        cum_particles += Int(meta_k["n_particles"])

        c_paper_jld = joinpath(c_paper, "paper_analysis.jld2")
        has_valid_c_paper = isfile(c_paper_jld) && jldopen(f -> haskey(f, "raw_n_chunks"), c_paper_jld, "r")
        if !has_valid_c_paper
            println("=== Analyzing single chunk $k -> $c_paper ===")
            run_paper_analysis(cdir, c_paper; render_plots = false)
        else
            println("=== Reusing pre-analyzed chunk summary $c_paper ===")
        end

        if !keep_raw_batches
            for fpath in filter(p -> occursin(r"^trajectories_.*\.jld2$", basename(p)), readdir(cdir; join = true))
                rm(fpath; force = true)
            end
        end

        println("=== Merging cumulative chunks 1..$k ($cum_particles total particles) -> $pdir ===")
        merge_paper_analyses(chunk_paper_dirs, pdir; render_plots)

        entry = Dict{String, Any}(
            "cumulative_chunks" => k,
            "cumulative_particles" => cum_particles,
            "chunk_dir" => cdir,
            "paper_dir" => pdir,
            "chunk_elapsed_s" => Float64(get(comp_k, "elapsed_s", 0.0)),
            "chunk_saved_trajectories" => Int(get(comp_k, "n_saved_trajectories", 0)),
        )
        if k >= 2
            prev_jld2 = joinpath(paper_dirs[k - 1], "paper_analysis.jld2")
            curr_jld2 = joinpath(pdir, "paper_analysis.jld2")
            metrics = compute_chunk_convergence(prev_jld2, curr_jld2)
            entry["metrics"] = metrics
            println("  Convergence (1..$(k - 1) vs 1..$k):")
            for (mname, mval) in sort!(collect(metrics); by = first)
                @printf("    %-26s : %6.2f %%\n", mname, 100.0 * mval)
            end
        end
        push!(history, entry)
        open("$(base_prefix)_convergence.toml", "w") do io
            TOML.print(io, Dict("chunks" => history))
        end
        if k >= 2
            plot_convergence_history(history, "$(base_prefix)_convergence.png")
            if rel_tol > 0
                m = entry["metrics"]
                key_vals = [m["orbit_log_def_rel_l2"], m["tail_log_def_rel_l2"], m["tail_ux_rel_l2"]]
                if all(v -> isfinite(v) && v <= rel_tol, key_vals)
                    println("=== Convergence threshold rel_tol=$(rel_tol) reached after $k chunks ($cum_particles particles)! ===")
                    break
                end
            end
        end
    end
    return chunk_dirs, history
end

if abspath(PROGRAM_FILE) == @__FILE__
    base_prefix = length(ARGS) >= 1 ? ARGS[1] : "outputs/mc_400km_crustal"
    n_chunks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    per_cell = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
    cell_stride = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    save_mode = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :roi_and_sample
    rel_tol = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.0
    run_incremental_chunks(
        base_prefix;
        n_chunks,
        per_cell,
        cell_stride,
        save_trajectories = save_mode,
        rel_tol,
    )
end
