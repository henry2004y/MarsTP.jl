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
        c_paper = @sprintf("%s_chunk_%03d_paper", base_prefix, k)
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
    return last(paper_dirs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    case_name = lowercase(strip(get(ENV, "CASE", "with_crustal")))
    scratch_default = joinpath(get(ENV, "SCRATCH", MarsTP.project_path("outputs")), "MarsTP_outputs", "mc_400km_$(case_name)")
    base_prefix = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "OUT_PREFIX", scratch_default)
    mode = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "ANALYZE_MODE", "incremental")
    analyze_discovered_chunks(base_prefix, mode)
end
