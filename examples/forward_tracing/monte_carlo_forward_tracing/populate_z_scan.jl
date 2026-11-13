# Populate or update the multi-Z cross-tail scan (`tail_z_scan_*`) in an existing
# `paper_analysis.jld2` and render `fig6_z_scan.png` using exact `TransectAccumulators`
# (0.2 R_M sliding detector cubes, 64 logarithmic energy bins, exact sub-step dwell time
# and cumulative work).
#
# Can either:
#   1. Stream from existing raw chunk directory/directories if provided, OR
#   2. Trace a fresh multi-threaded Monte Carlo ensemble directly in memory (skipping disk I/O
#      and only computing work for trajectories intersecting X ∈ [-2.1, -1.9] R_M).
#
# Usage:
#   julia --threads=auto --project=. examples/forward_tracing/monte_carlo_forward_tracing/populate_z_scan.jl PAPER_DIR [PER_CELL] [CELL_STRIDE]

include("monte_carlo_shell.jl")
include("analyze_paper_results.jl")
include("plot_fig7.jl")

using .ShellMonteCarlo
using MarsTP, StaticArrays, LinearAlgebra, JLD2, Printf

function populate_z_scan_in_memory!(
        paper_dir::AbstractString;
        per_cell::Int = 100,
        cell_stride::Int = 1,
        altitude_km::Float64 = 400.0,
        seed::Int = 20261109,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    rm_m, fields_file, energy_edges, tail_y_Rm = jldopen(jld_path, "r") do f
        (
            Float64(f["rm_m"]),
            String(f["fields_file"]),
            collect(Float64, f["energy_edges_eV"]),
            collect(Float64, f["tail_y_Rm"]),
        )
    end

    fields_path = MarsTP.project_path("data", fields_file)
    isfile(fields_path) || error("MHD fields file not found: $fields_path")

    cfg = ShellMonteCarlo.Config(;
        altitude_km,
        absorption_altitude_km = 200.0,
        cell_stride,
        per_cell,
        seed,
        random_seed = false,
        tmax = 1000.0,
        dt = 0.1,
        fields_path,
        include_gradpe_work = true,
    )

    println("=== Initializing source shell and MHD interpolators for $fields_file (per_cell=$per_cell, cell_stride=$cell_stride) ===")
    t0 = time()
    fields = load_mhd_fields(fields_path)
    fields, source, _ = ShellMonteCarlo.validate_geometry(fields.path, fields, cfg)
    param = MarsTP.mhd_param(fields; species = "O2+")
    b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)
    component(which) = begin
        f = load_mhd_fields(fields.path; electric_field = which)
        MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, f.E, fields.r, fields.theta, fields.phi)
    end
    work_itp = MarsTP.FieldWorkInterpolators(
        component(:total), component(:conv), component(:hall),
        cfg.include_gradpe_work ? component(:gradpe) : nothing,
        nothing,
    )
    particles, cells = ShellMonteCarlo.release_particles(fields, source, cfg, seed)
    n_part = length(particles)
    println("  Sampled $n_part source particles across $(length(cells)) cells in $(round(time() - t0; digits = 1)) s")

    tail_side = 0.2 * rm_m
    tail20_zscan_centers = [
        let y_cs = _find_current_sheet_y_Rm(b_itp, -2.0, z0, rm_m)
                [Vec3(-2.0 * rm_m, (y_cs + dy) * rm_m, z0 * rm_m) for dy in tail_y_Rm]
        end for z0 in TAIL_Z_SCAN_RM
    ]
    nt = Threads.nthreads()
    worker_zscan = [[TransectAccumulators(c_vec, tail_side, energy_edges) for c_vec in tail20_zscan_centers] for _ in 1:nt]

    x_lo = -2.105 * rm_m
    x_hi = -1.895 * rm_m
    y_lo = -1.555 * rm_m
    y_hi = 1.555 * rm_m
    z_lo = (minimum(TAIL_Z_SCAN_RM) - 0.105) * rm_m
    z_hi = (maximum(TAIL_Z_SCAN_RM) + 0.105) * rm_m

    t_trace = time()
    n_done = Threads.Atomic{Int}(0)
    n_tail_hits = Threads.Atomic{Int}(0)

    # Partition particles into contiguous chunks per thread to avoid false sharing
    chunk_size = cld(n_part, nt)
    Threads.@threads :static for tid in 1:nt
        q_start = (tid - 1) * chunk_size + 1
        q_end = min(n_part, tid * chunk_size)
        accs_w = worker_zscan[tid]
        for q in q_start:q_end
            p = particles[q]
            if p.W > 0
                r = ShellMonteCarlo.trace_particle(p.x, p.v, param, cfg; compute_step_work = false)
                hist = r.history
                nh = length(hist)
                if nh >= 2
                    # Fast check if trajectory ever enters the X = -2.0 Rm cross-tail slab
                    hits_slab = false
                    last_slab_idx = 0
                    @inbounds for j in 2:nh
                        a = hist[j - 1]
                        b = hist[j]
                        if min(a[2], b[2]) <= x_hi && max(a[2], b[2]) >= x_lo &&
                                min(a[3], b[3]) <= y_hi && max(a[3], b[3]) >= y_lo &&
                                min(a[4], b[4]) <= z_hi && max(a[4], b[4]) >= z_lo
                            hits_slab = true
                            last_slab_idx = j
                        end
                    end
                    if hits_slab
                        Threads.atomic_add!(n_tail_hits, 1)
                        n_keep = min(nh, last_slab_idx + 1)
                        t_vec = [hist[k][1] for k in 1:n_keep]
                        u_vec = [SVector{6, Float64}(hist[k][2], hist[k][3], hist[k][4], hist[k][5], hist[k][6], hist[k][7]) for k in 1:n_keep]
                        tr = (; t = t_vec, u = u_vec)
                        w_full = trajectory_work(tr, work_itp; species = "O2+", mode = cfg.work_mode, power = false)
                        inc = w_full.increments
                        dw_tot = inc[:, 4]
                        dw_conv = inc[:, 1]
                        dw_hall = inc[:, 2]
                        dw_grad = inc[:, 3]
                        for acc_z in accs_w
                            _accumulate_transect!(acc_z, t_vec, u_vec, p.W, p.id, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 20.0, mode = :tail)
                        end
                    end
                end
            end
            done_now = Threads.atomic_add!(n_done, 1) + 1
            if done_now % 100_000 == 0 || done_now == n_part
                el = time() - t_trace
                @printf(
                    "  Progress: %d / %d (%.1f%%) | tail-slab trajectories=%d | elapsed=%.1f s\n",
                    done_now, n_part, 100.0 * done_now / n_part, n_tail_hits[], el
                )
                flush(stdout)
            end
        end
    end

    tail20_zscan_accs = worker_zscan[1]
    for tid in 2:nt, iz in eachindex(tail20_zscan_accs)
        merge_transect!(tail20_zscan_accs[iz], worker_zscan[tid][iz])
    end
    tail20_zscan_res = [finalize_transect(acc, 1) for acc in tail20_zscan_accs]

    b_tail20_zscan_nT = Matrix{Float64}[]
    for c_vec in tail20_zscan_centers
        bmat = zeros(Float64, 4, length(c_vec))
        for (i, c) in enumerate(c_vec)
            bv = Vec3(b_itp(c)) * 1.0e9
            bmat[1:3, i] .= bv
            bmat[4, i] = norm(bv)
        end
        push!(b_tail20_zscan_nT, bmat)
    end

    # Update paper_analysis.jld2 with the new tail_z_scan_* datasets
    tmp_jld = jld_path * ".tmp"
    existing_data = jldopen(jld_path, "r") do f
        Dict{String, Any}(k => f[k] for k in keys(f))
    end
    zscan_y_Rm = [Float64[round(c[2] / rm_m; digits = 4) for c in acc.centers_m] for acc in tail20_zscan_accs]
    zscan_y_cs_Rm = [ys_v[cld(length(ys_v), 2)] for ys_v in zscan_y_Rm]
    existing_data["tail_z_scan_Rm"] = copy(TAIL_Z_SCAN_RM)
    existing_data["tail_z_scan_y_cs_Rm"] = zscan_y_cs_Rm
    existing_data["tail_z_scan_y_Rm"] = zscan_y_Rm
    existing_data["raw_tail20_zscan"] = [_pack_transect_raw(acc) for acc in tail20_zscan_accs]
    existing_data["tail_z_scan_b_nT"] = b_tail20_zscan_nT
    existing_data["tail_z_scan_def"] = [r.def for r in tail20_zscan_res]
    existing_data["tail_z_scan_psd_s3_m6"] = [r.psd for r in tail20_zscan_res]
    existing_data["tail_z_scan_bulk_vel_ms"] = [r.bulk_vel_ms for r in tail20_zscan_res]
    existing_data["tail_z_scan_density_m3"] = [r.density_m3 for r in tail20_zscan_res]
    existing_data["tail_z_scan_w_conv_eV"] = [r.w_conv for r in tail20_zscan_res]
    existing_data["tail_z_scan_w_hall_eV"] = [r.w_hall for r in tail20_zscan_res]
    existing_data["tail_z_scan_w_gradpe_eV"] = [r.w_gradpe for r in tail20_zscan_res]
    existing_data["tail_z_scan_w_total_eV"] = [r.w_total for r in tail20_zscan_res]
    existing_data["tail_z_scan_hits"] = [r.hit_counts for r in tail20_zscan_res]
    existing_data["fig7_rep_zscan"] = [_pack_rep_ions((y_cs, y_cs - 0.5), ys_v, r) for (y_cs, ys_v, r) in zip(zscan_y_cs_Rm, zscan_y_Rm, tail20_zscan_res)]

    jldopen(tmp_jld, "w") do f
        for (k, v) in existing_data
            f[k] = v
        end
    end
    mv(tmp_jld, jld_path; force = true)
    println("Updated $jld_path with 16-level production Z-scan.")

    out_scan = joinpath(paper_dir, "fig7_z_scan.png")
    plot_fig7_z_scan(paper_dir, out_scan)
    return out_scan
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : "outputs/paper_figures_with_crustal"
    pcell = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    cstride = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    populate_z_scan_in_memory!(pdir; per_cell = pcell, cell_stride = cstride)
end
