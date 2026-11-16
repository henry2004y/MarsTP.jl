# Discover completed Perlmutter chunk directories, pool them, generate Figures 3-7,
# and optionally compute chunk-by-chunk statistical convergence.
#
# Usage:
#   julia --threads=64 --project=. examples/forward_tracing/perlmutter/analyze_perlmutter_chunks.jl [base_prefix] [analyze_mode]
#
# Modes (`analyze_mode` / `ANALYZE_MODE`):
#   - "incremental" (default): analyzes cumulative chunks 1..1, 1..2, ..., 1..K and plots convergence
#   - "final"                : analyzes only the full pooled set 1..K into `$(base_prefix)_pooled_paper`

include(joinpath(@__DIR__, "..", "monte_carlo_forward_tracing", "run_incremental_chunks.jl"))

function discover_completed_chunks(base_prefix::AbstractString)
    parent_dir = dirname(abspath(base_prefix))
    base_name = basename(abspath(base_prefix))
    isdir(parent_dir) || error("Parent directory does not exist: $parent_dir")
    pat = Regex("^" * replace(base_name, r"([.^$*+?()\[\]{}|\\])" => s"\\\1") * "_chunk_(\\d+)\$")
    matched = Tuple{Int, String}[]
    for entry in readdir(parent_dir; join = true)
        isdir(entry) || continue
        m = match(pat, basename(entry))
        m === nothing && continue
        comp_path = joinpath(entry, "completion.toml")
        if isfile(comp_path) && get(TOML.parsefile(comp_path), "complete", false) === true
            push!(matched, (parse(Int, m.captures[1]), entry))
        end
    end
    sort!(matched; by = first)
    return [p[2] for p in matched]
end

function analyze_discovered_chunks(base_prefix::AbstractString, mode::String = "incremental")
    chunk_dirs = discover_completed_chunks(base_prefix)
    isempty(chunk_dirs) && error("No completed chunk directories found for prefix: $base_prefix")
    println("Found $(length(chunk_dirs)) completed chunk(s) for $base_prefix:")
    for d in chunk_dirs
        println("  - $d")
    end

    if lowercase(mode) == "final"
        out_dir = "$(base_prefix)_pooled_$(length(chunk_dirs))chunks_paper"
        run_paper_analysis(chunk_dirs, out_dir)
        return out_dir
    end

    history = Dict{String, Any}[]
    chunk_paper_dirs = String[]
    paper_dirs = String[]
    cum_particles = 0
    for (k, cdir) in enumerate(chunk_dirs)
        c_paper = isdir("$(cdir)_paper") ? "$(cdir)_paper" : @sprintf("%s_chunk_%03d_paper", base_prefix, k)
        pdir = @sprintf("%s_cum_%03d_paper", base_prefix, k)
        push!(chunk_paper_dirs, c_paper)
        push!(paper_dirs, pdir)
        meta_k = TOML.parsefile(joinpath(cdir, "metadata.toml"))
        comp_k = TOML.parsefile(joinpath(cdir, "completion.toml"))
        cum_particles += Int(meta_k["n_particles"])

        c_paper_jld = joinpath(c_paper, "paper_analysis.jld2")
        has_valid_c_paper = isfile(c_paper_jld) && jldopen(f -> haskey(f, "raw_n_chunks"), c_paper_jld, "r")
        if !has_valid_c_paper
            println("=== Analyzing single chunk $k -> $c_paper ===")
            run_paper_analysis(cdir, c_paper; render_plots = false)
        else
            println("=== Reusing pre-analyzed single-chunk summary $c_paper ===")
        end

        println("=== Merging cumulative chunks 1..$k ($cum_particles particles) -> $pdir ===")
        merge_paper_analyses(chunk_paper_dirs, pdir; render_plots = true)

        entry = Dict{String, Any}(
            "cumulative_chunks" => k,
            "cumulative_particles" => cum_particles,
            "chunk_dir" => cdir,
            "paper_dir" => pdir,
            "seed" => Int(get(meta_k, "seed", 0)),
            "chunk_id" => Int(get(meta_k, "chunk_id", k)),
            "chunk_elapsed_s" => Float64(get(comp_k, "elapsed_s", 0.0)),
            "chunk_saved_trajectories" => Int(get(comp_k, "n_saved_trajectories", 0)),
        )
        if k >= 2
            metrics = compute_chunk_convergence(
                joinpath(paper_dirs[k - 1], "paper_analysis.jld2"),
                joinpath(pdir, "paper_analysis.jld2"),
            )
            entry["metrics"] = metrics
            println("  Convergence (1..$(k - 1) vs 1..$k):")
            for (mname, mval) in sort!(collect(metrics); by = first)
                @printf("    %-26s : %6.2f %%\n", mname, 100.0 * mval)
            end
        end
        push!(history, entry)
    end

    conv_toml = "$(base_prefix)_convergence.toml"
    open(conv_toml, "w") do io
        TOML.print(io, Dict("chunks" => history))
    end
    if length(history) >= 2
        plot_convergence_history(history, "$(base_prefix)_convergence.png")
    end
    println("Saved convergence report to $conv_toml")

    final_dir = "$(base_prefix)_final_paper"
    isdir(final_dir) && rm(final_dir; recursive = true, force = true)
    cp(last(paper_dirs), final_dir; force = true)
    println("Copied pooled final result to $final_dir")

    chunk_summaries = Dict{String, Any}[]
    for cpdir in chunk_paper_dirs
        spath = joinpath(cpdir, "chunk_summary.toml")
        isfile(spath) && push!(chunk_summaries, TOML.parsefile(spath))
    end
    if !isempty(chunk_summaries)
        n_c = length(chunk_summaries)
        n_launched = 0
        n_saved_csv = 0
        n_outer = 0
        n_inner = 0
        n_other = 0
        source_rates = Float64[]
        outer_rates = Float64[]
        out_w_work = 0.0
        out_w_conv = 0.0
        out_w_hall = 0.0
        out_w_gradpe = 0.0
        out_w_sum = 0.0
        for cs in chunk_summaries
            ms = get(cs, "metadata", nothing)
            sc = get(cs, "source_cells", nothing)
            ps = get(cs, "particles", nothing)
            ms !== nothing && (n_launched += Int(get(ms, "n_particles", 0)))
            sc !== nothing && push!(source_rates, Float64(get(sc, "source_shell_rate_s1", 0.0)))
            ps === nothing && continue
            n_saved_csv += Int(get(ps, "n_saved_csv", 0))
            n_outer += Int(ps["n_outer"])
            n_inner += Int(ps["n_inner"])
            n_other += Int(ps["n_other"])
            r_o = Float64(ps["outer_rate_s1"])
            push!(outer_rates, r_o)
            out_w_work += Float64(ps["outer_weighted_work_eV"])
            out_w_conv += Float64(get(ps, "outer_weighted_work_conv_eV", 0.0))
            out_w_hall += Float64(get(ps, "outer_weighted_work_hall_eV", 0.0))
            out_w_gradpe += Float64(get(ps, "outer_weighted_work_gradpe_eV", 0.0))
            out_w_sum += r_o
        end
        mean_src = isempty(source_rates) ? NaN : sum(source_rates) / length(source_rates)
        mean_out = isempty(outer_rates) ? NaN : sum(outer_rates) / length(outer_rates)
        std_out = length(outer_rates) > 1 ? sqrt(sum(x -> (x - mean_out)^2, outer_rates) / (length(outer_rates) - 1)) : 0.0
        agg = Dict{String, Any}(
            "n_chunks" => n_c,
            "flux_model" => String(get(first(chunk_summaries), "flux_model", "")),
            "n_launched" => n_launched,
            "n_saved_csv" => n_saved_csv,
            "n_outer" => n_outer,
            "n_inner" => n_inner,
            "n_other" => n_other,
            "source_shell_rate_s1" => mean_src,
            "mean_outer_rate_s1" => mean_out,
            "std_outer_rate_s1" => std_out,
            "escape_fraction_rate" => mean_src > 0 ? mean_out / mean_src : NaN,
            "escape_fraction_count" => n_launched > 0 ? n_outer / n_launched : NaN,
            "mean_outer_work_eV" => out_w_sum > 0 ? out_w_work / out_w_sum : NaN,
            "mean_outer_work_conv_eV" => out_w_sum > 0 ? out_w_conv / out_w_sum : NaN,
            "mean_outer_work_hall_eV" => out_w_sum > 0 ? out_w_hall / out_w_sum : NaN,
            "mean_outer_work_gradpe_eV" => out_w_sum > 0 ? out_w_gradpe / out_w_sum : NaN,
            "chunks" => chunk_summaries,
        )
        open(joinpath(final_dir, "ensemble_escape_summary.toml"), "w") do io
            TOML.print(io, agg)
        end
        println("Saved ensemble escape summary to $(joinpath(final_dir, "ensemble_escape_summary.toml"))")
    end

    clean_intermediate = lowercase(strip(get(ENV, "CLEAN_INTERMEDIATE", "false"))) == "true"
    if clean_intermediate && isfile(joinpath(final_dir, "paper_analysis.jld2")) && isfile(conv_toml)
        println("CLEAN_INTERMEDIATE=true: removing intermediate chunk and cumulative directories...")
        for d in vcat(chunk_dirs, chunk_paper_dirs, paper_dirs)
            isdir(d) && rm(d; recursive = true, force = true)
        end
    end

    return final_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    case_name = lowercase(strip(get(ENV, "CASE", "with_crustal")))
    scratch_default = joinpath(get(ENV, "SCRATCH", MarsTP.project_path("outputs")), "MarsTP_outputs", "mc_400km_$(case_name)")
    base_prefix = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "OUT_PREFIX", scratch_default)
    mode = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "ANALYZE_MODE", "incremental")
    analyze_discovered_chunks(base_prefix, mode)
end
