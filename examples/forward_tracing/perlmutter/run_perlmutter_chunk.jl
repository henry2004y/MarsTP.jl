# Perlmutter production runner for a single Monte Carlo chunk or range of chunks.
#
# Supports:
#   - Slurm job arrays via `SLURM_ARRAY_TASK_ID` (each array task runs one independent chunk `k`)
#   - Sequential multi-chunk execution via `CHUNK_START` .. `CHUNK_END`
#   - Both MHD cases: `CASE=with_crustal` or `CASE=without_crustal`
#   - Hardware-entropy CSPRNG (`RandomDevice()`) + SHA-256 256-bit Xoshiro stream derivation
#   - Compact format v3 (`packed = true`, `Float32`, no `t` array, `skip_inner_and_trapped = true`)
#
# Environment / CLI configuration:
#   OUT_PREFIX             : base output path prefix (default: "$SCRATCH/MarsTP_outputs/mc_400km_with_crustal")
#   CASE                   : "with_crustal" (default) or "without_crustal"
#   FIELDS_PATH            : explicit .vts path (overrides CASE if set)
#   ALTITUDE_KM            : launch altitude in km (default: 400.0)
#   ABSORPTION_ALTITUDE_KM : inner absorbing boundary in km (default: 200.0)
#   PER_CELL               : particles per source cell per chunk (default: 100 -> 1,015,300 particles/chunk)
#   CELL_STRIDE            : angular cell stride (default: 1 -> all 10,153 cells)
#   DT_S                   : Boris timestep in seconds (default: 0.1)
#   TMAX_S                 : max flight time in seconds (default: 1000.0)
#   BATCH_SIZE             : particle batch size (default: 4096)
#   SAVE_TRAJECTORIES      : "roi_and_sample" (default), "outer_only", "probes_only", "all", "none"
#   SAVE_STRIDE            : trajectory output stride (default: 5 -> 0.5 s cadence)
#   SAMPLE_SIZE            : unbiased escaping trajectory sample size per chunk (default: 20000)
#   SKIP_INNER_TRAPPED     : skip inner/time_limit particles in CSV and JLD2 (default: true)
#   TRACING_BACKEND        : "cpu" (default) or "cuda"
#   RANDOM_SEED            : draw fresh hardware seed from RandomDevice() (default: true)
#   BASE_SEED              : deterministic seed when RANDOM_SEED=false (default: 20260906)
#   CHUNK_START            : first chunk ID to run (default: SLURM_ARRAY_TASK_ID or 1)
#   CHUNK_END              : last chunk ID to run (default: CHUNK_START)

include(joinpath(@__DIR__, "..", "monte_carlo_forward_tracing", "monte_carlo_shell.jl"))

using .ShellMonteCarlo
using MarsTP, TOML, Printf, Random

function _resolve_fields_path()
    explicit = strip(get(ENV, "FIELDS_PATH", ""))
    !isempty(explicit) && return explicit
    case_name = lowercase(strip(get(ENV, "CASE", "with_crustal")))
    if case_name in ("with_crustal", "crustal", "with")
        p = MarsTP.data_path("mars_fields_with_crustal.vts")
        return isfile(p) ? p : MarsTP.data_path("mars_fields_spherical_from_dat.vts")
    elseif case_name in ("without_crustal", "no_crustal", "without")
        return MarsTP.data_path("mars_fields_without_crustal.vts")
    else
        error("Unknown CASE='$case_name'; expected 'with_crustal' or 'without_crustal'")
    end
end

function main()
    case_name = lowercase(strip(get(ENV, "CASE", "with_crustal")))
    fields_path = _resolve_fields_path()
    isfile(fields_path) || error("MHD fields file not found at $fields_path. Run `julia --project=. scripts/convert_batsrus_mhd.jl` first.")

    scratch_default = joinpath(get(ENV, "SCRATCH", MarsTP.project_path("outputs")), "MarsTP_outputs", "mc_400km_$(case_name)")
    base_prefix = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "OUT_PREFIX", scratch_default)

    array_id = strip(get(ENV, "SLURM_ARRAY_TASK_ID", ""))
    default_start = isempty(array_id) ? 1 : parse(Int, array_id)
    chunk_start = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : parse(Int, get(ENV, "CHUNK_START", string(default_start)))
    chunk_end = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : parse(Int, get(ENV, "CHUNK_END", string(chunk_start)))
    per_cell = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : parse(Int, get(ENV, "PER_CELL", "100"))
    cell_stride = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : parse(Int, get(ENV, "CELL_STRIDE", "1"))

    altitude_km = parse(Float64, get(ENV, "ALTITUDE_KM", "400.0"))
    absorption_altitude_km = parse(Float64, get(ENV, "ABSORPTION_ALTITUDE_KM", "200.0"))
    dt = parse(Float64, get(ENV, "DT_S", "0.1"))
    tmax = parse(Float64, get(ENV, "TMAX_S", "1000.0"))
    batch_size = parse(Int, get(ENV, "BATCH_SIZE", "4096"))
    save_trajectories = Symbol(get(ENV, "SAVE_TRAJECTORIES", "roi_and_sample"))
    save_stride = parse(Int, get(ENV, "SAVE_STRIDE", "5"))
    trajectory_sample_size = parse(Int, get(ENV, "SAMPLE_SIZE", "20000"))
    skip_inner_and_trapped = lowercase(get(ENV, "SKIP_INNER_TRAPPED", "true")) == "true"
    tracing_backend = Symbol(lowercase(get(ENV, "TRACING_BACKEND", "cpu")))
    random_seed = lowercase(get(ENV, "RANDOM_SEED", "true")) == "true"
    base_seed = parse(Int, get(ENV, "BASE_SEED", "20260906"))

    println("==================================================================")
    println("MarsTP Perlmutter Forward Tracing Runner")
    println("  base_prefix            = $base_prefix")
    println("  fields_path            = $fields_path")
    println("  chunk range            = $chunk_start .. $chunk_end")
    println("  per_cell / cell_stride = $per_cell / $cell_stride")
    println("  altitude / absorption  = $altitude_km km / $absorption_altitude_km km")
    println("  dt / tmax              = $dt s / $tmax s")
    println("  save_trajectories      = $save_trajectories (stride=$save_stride, skip_inner_trapped=$skip_inner_and_trapped)")
    println("  tracing_backend        = $tracing_backend (threads=$(Threads.nthreads()))")
    println("  random_seed            = $random_seed")
    println("==================================================================")
    flush(stdout)

    for k in chunk_start:chunk_end
        cdir = @sprintf("%s_chunk_%03d", base_prefix, k)
        comp_path = joinpath(cdir, "completion.toml")
        if isfile(comp_path) && get(TOML.parsefile(comp_path), "complete", false) === true
            println("=== Chunk $k already complete at $cdir; skipping ===")
            flush(stdout)
            continue
        end
        ispath(cdir) && rm(cdir; recursive = true, force = true)
        mkpath(dirname(abspath(cdir)))

        cfg = ShellMonteCarlo.Config(;
            altitude_km,
            absorption_altitude_km,
            per_cell,
            seed = base_seed,
            random_seed,
            cell_stride,
            batch_size,
            dt,
            tmax,
            tracing_backend,
            fields_path,
            include_gradpe_work = true,
            save_trajectories,
            save_stride,
            trajectory_sample_size,
            packed_trajectories = true,
            trajectory_float32 = true,
            skip_inner_and_trapped,
            chunk_id = k,
        )
        println("=== Launching Chunk $k -> $cdir ===")
        flush(stdout)
        ShellMonteCarlo.run_monte_carlo(cdir, cfg)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
