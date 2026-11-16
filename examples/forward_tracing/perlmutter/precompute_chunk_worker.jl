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

function _summarize_source_cells_csv(source_cells_csv::AbstractString)
    n_cells = 0
    source_shell_rate_s1 = 0.0
    open(source_cells_csv, "r") do io
        eof(io) && return Dict{String, Any}("n_cells" => 0, "source_shell_rate_s1" => 0.0)
        hdr = strip.(split(readline(io), ','))
        area_idx = findfirst(==("area_m2"), hdr)
        flux_idx = findfirst(==("flux_m2_s"), hdr)
        (area_idx === nothing || flux_idx === nothing) && error("Unexpected source_cells.csv schema in $source_cells_csv: $hdr")
        for line in eachline(io)
            isempty(strip(line)) && continue
            cols = split(line, ',')
            n_cells += 1
            source_shell_rate_s1 += parse(Float64, strip(cols[area_idx])) * parse(Float64, strip(cols[flux_idx]))
        end
    end
    return Dict{String, Any}(
        "n_cells" => n_cells,
        "source_shell_rate_s1" => source_shell_rate_s1,
    )
end

function _summarize_chunk_particles_csv(particles_csv::AbstractString)
    n_saved_csv = 0
    n_outer = 0
    n_inner = 0
    n_other = 0
    saved_rate_s1 = 0.0
    outer_rate_s1 = 0.0
    outer_weighted_work_eV = 0.0
    outer_weighted_work_conv_eV = 0.0
    outer_weighted_work_hall_eV = 0.0
    outer_weighted_work_gradpe_eV = 0.0
    open(particles_csv, "r") do io
        header_line = readline(io)
        hdr = strip.(split(header_line, ','))
        status_idx = findfirst(==("status"), hdr)
        w_idx = something(findfirst(==("rate_weight_s1"), hdr), findfirst(==("weight_s1"), hdr), 0)
        work_idx = something(findfirst(==("work_eV"), hdr), 0)
        wc_idx = something(findfirst(==("work_conv_eV"), hdr), 0)
        wh_idx = something(findfirst(==("work_hall_eV"), hdr), 0)
        wa_idx = something(findfirst(==("work_gradpe_eV"), hdr), 0)
        (status_idx === nothing || w_idx == 0) && error("Unexpected particles.csv schema in $particles_csv: $hdr")
        for line in eachline(io)
            isempty(strip(line)) && continue
            cols = split(line, ',')
            st = strip(cols[status_idx])
            w = parse(Float64, strip(cols[w_idx]))
            n_saved_csv += 1
            saved_rate_s1 += w
            if st == "outer"
                n_outer += 1
                outer_rate_s1 += w
                work_idx > 0 && (outer_weighted_work_eV += w * parse(Float64, strip(cols[work_idx])))
                wc_idx > 0 && (outer_weighted_work_conv_eV += w * parse(Float64, strip(cols[wc_idx])))
                wh_idx > 0 && (outer_weighted_work_hall_eV += w * parse(Float64, strip(cols[wh_idx])))
                wa_idx > 0 && (outer_weighted_work_gradpe_eV += w * parse(Float64, strip(cols[wa_idx])))
            elseif st == "inner"
                n_inner += 1
            else
                n_other += 1
            end
        end
    end
    return Dict{String, Any}(
        "n_saved_csv" => n_saved_csv,
        "n_outer" => n_outer,
        "n_inner" => n_inner,
        "n_other" => n_other,
        "saved_rate_s1" => saved_rate_s1,
        "outer_rate_s1" => outer_rate_s1,
        "outer_weighted_work_eV" => outer_weighted_work_eV,
        "outer_weighted_work_conv_eV" => outer_weighted_work_conv_eV,
        "outer_weighted_work_hall_eV" => outer_weighted_work_hall_eV,
        "outer_weighted_work_gradpe_eV" => outer_weighted_work_gradpe_eV,
        "mean_outer_work_eV" => outer_rate_s1 > 0 ? outer_weighted_work_eV / outer_rate_s1 : NaN,
    )
end

function _write_chunk_summary(
        tmp_dir::AbstractString,
        c_paper_tmp::AbstractString,
        k::Int,
        flux_model::AbstractString,
    )
    meta_path = joinpath(tmp_dir, "metadata.toml")
    comp_path = joinpath(tmp_dir, "completion.toml")
    cells_path = joinpath(tmp_dir, "source_cells.csv")
    csv_path = joinpath(tmp_dir, "particles.csv")
    summary = Dict{String, Any}(
        "chunk_index" => k,
        "flux_model" => String(flux_model),
    )
    if isfile(meta_path)
        summary["metadata"] = TOML.parsefile(meta_path)
    end
    if isfile(comp_path)
        summary["completion"] = TOML.parsefile(comp_path)
    end
    if isfile(cells_path)
        summary["source_cells"] = _summarize_source_cells_csv(cells_path)
    end
    if isfile(csv_path)
        summary["particles"] = _summarize_chunk_particles_csv(csv_path)
    end
    mkpath(c_paper_tmp)
    open(joinpath(c_paper_tmp, "chunk_summary.toml"), "w") do io
        TOML.print(io, summary)
    end
    return summary
end

function run_precompute_worker(
        base_prefix::AbstractString,
        k::Int,
        per_cell::Int,
        cell_stride::Int,
        fields_path::AbstractString,
        flux_model::AbstractString = strip(get(ENV, "FLUX_MODEL", "surface_flux_maxwellian")),
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
        flux_model = String(flux_model),
        chunk_id = k,
    )

    println("=== Precomputing Chunk $k -> $tmp_dir (seed=$chunk_seed, per_cell=$per_cell, flux_model=$flux_model) ===")
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
    _write_chunk_summary(tmp_dir, c_paper_tmp, k, flux_model)

    keep_raw = lowercase(strip(get(ENV, "KEEP_RAW", "false"))) == "true"
    if !keep_raw
        for fpath in readdir(tmp_dir; join = true)
            bname = basename(fpath)
            if occursin(r"^trajectories_.*\.jld2$", bname) ||
               bname in ("particles.csv", "probe_residence.csv", "probe_crossings.csv")
                rm(fpath; force = true)
            end
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
    flux_model = length(ARGS) >= 6 ? String(ARGS[6]) : strip(get(ENV, "FLUX_MODEL", "surface_flux_maxwellian"))
    run_precompute_worker(base_prefix, k, per_cell, cell_stride, fields_path, flux_model)
end

