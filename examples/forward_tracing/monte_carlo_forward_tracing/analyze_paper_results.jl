# Paper-style analysis and figure generation for Mars O2+ forward Monte Carlo runs.
# Reproduces the diagnostics in Testparticle_mars_paper.tex:
#   - Fig. 3: Global trajectories (XZ, XY, YZ) + virtual detector VDF, omni DEF, and energy-pitch angle at (-2, 0, 0) Rm
#   - Fig. 4: Circular virtual spacecraft trajectory at r = 2 Rm (XZ plane, Y = 0, angles 0°..240°):
#             omnidirectional DEF and mean cumulative work <ε_C>, <ε_H>, <ε_A>, <ε_T>
#   - Fig. 5: Representative trajectories, instantaneous power P_ℓ(t), and cumulative work ε_ℓ(t) at 90°, 120°, 150°
#   - Fig. 6: Cross-tail virtual spacecraft trajectory along Y ∈ [-1, 1] Rm at X = -1.75 Rm, Z = +0.75 Rm:
#             magnetic field profile, DEF, bulk velocity (Ux, Uy, Uz), and <ε_C>, <ε_H>, <ε_A>, <ε_T>
#   - Fig. 7: Representative trajectories, power, and cumulative work at current sheet center vs lobe
#
# Usage:
#   julia --threads=auto --project=. examples/forward_tracing/monte_carlo_forward_tracing/analyze_paper_results.jl outputs/mc_400km_crustal_1M outputs/mc_400km_crustal_1M_paper

using MarsTP, StaticArrays, LinearAlgebra, JLD2, TOML, Printf, Statistics, PyPlot

const Vec3 = SVector{3, Float64}
const E_CHARGE = MarsTP.TP.qᵢ
const O2P_MASS = MarsTP.TP.SpeciesDict["O2+"].m

struct TransectAccumulators
    centers_m::Vector{Vec3}
    side_m::Float64
    energy_edges_eV::Vector{Float64}
    energy_centers_eV::Vector{Float64}
    weight_tau::Matrix{Float64}          # (n_det, n_E)
    def_num::Matrix{Float64}             # (n_det, n_E)
    work_conv_num::Matrix{Float64}       # (n_det, n_E)
    work_hall_num::Matrix{Float64}       # (n_det, n_E)
    work_gradpe_num::Matrix{Float64}     # (n_det, n_E)
    work_total_num::Matrix{Float64}      # (n_det, n_E)
    density_tau::Vector{Float64}         # (n_det,)
    vel_tau::Matrix{Float64}             # (3, n_det)
    hit_counts::Vector{Int}              # (n_det,)
    best_pid::Vector{Int}                # (n_det,)
    best_score::Vector{Float64}          # (n_det,)
    best_time_s::Vector{Float64}         # (n_det,)
    best_tr::Vector{Any}                 # (n_det,)
end

function TransectAccumulators(centers_m::Vector{Vec3}, side_m::Float64, energy_edges_eV::Vector{Float64})
    nd = length(centers_m)
    ne = length(energy_edges_eV) - 1
    ec = sqrt.(energy_edges_eV[1:(end - 1)] .* energy_edges_eV[2:end])
    return TransectAccumulators(
        centers_m,
        side_m,
        energy_edges_eV,
        ec,
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd),
        zeros(Float64, 3, nd),
        zeros(Int, nd),
        zeros(Int, nd),
        zeros(Float64, nd),
        zeros(Float64, nd),
        fill(nothing, nd),
    )
end

@inline function _accumulate_transect!(
        acc::TransectAccumulators,
        t_vec,
        u_vec::Vector{SVector{6, Float64}},
        rate_w::Float64,
        pid::Int,
        dw_tot::Vector{Float64},
        dw_conv::Vector{Float64},
        dw_hall::Vector{Float64},
        dw_gradpe::Vector{Float64};
        min_rep_energy_eV::Float64 = 30.0,
    )
    rate_w > 0 || return nothing
    n = length(t_vec)
    n >= 2 || return nothing

    # Fast bounding box of trajectory
    xmin = Inf; xmax = -Inf
    ymin = Inf; ymax = -Inf
    zmin = Inf; zmax = -Inf
    @inbounds for i in 1:n
        u = u_vec[i]
        x, y, z = u[1], u[2], u[3]
        x < xmin && (xmin = x); x > xmax && (xmax = x)
        y < ymin && (ymin = y); y > ymax && (ymax = y)
        z < zmin && (zmin = z); z > zmax && (zmax = z)
    end
    half = 0.5 * acc.side_m
    edges = acc.energy_edges_eV
    ne = length(edges) - 1

    # Check which detectors overlap the trajectory AABB
    @inbounds for d in eachindex(acc.centers_m)
        c = acc.centers_m[d]
        (
            xmax < c[1] - half || xmin > c[1] + half ||
                ymax < c[2] - half || ymin > c[2] + half ||
                zmax < c[3] - half || zmin > c[3] + half
        ) && continue

        lo = c .- half
        hi = c .+ half
        wt_tot = 0.0
        wc_tot = 0.0
        wh_tot = 0.0
        wa_tot = 0.0
        had_hit = false
        best_seg_wt = 0.0
        best_seg_t = 0.0

        for j in 2:n
            ua = u_vec[j - 1]
            ub = u_vec[j]
            xa = Vec3(ua[1], ua[2], ua[3])
            xb = Vec3(ub[1], ub[2], ub[3])
            dwt = dw_tot[j - 1]
            dwc = dw_conv[j - 1]
            dwh = dw_hall[j - 1]
            dwa = dw_gradpe[j - 1]

            hit = MarsTP._forward_cube_interval(xa, xb - xa, lo, hi)
            if hit !== nothing
                s0, s1, _, _ = hit
                ta = t_vec[j - 1]
                tb = t_vec[j]
                tau = (s1 - s0) * (tb - ta)
                if tau > 0
                    had_hit = true
                    sm = 0.5 * (s0 + s1)
                    va = Vec3(ua[4], ua[5], ua[6])
                    vb = Vec3(ub[4], ub[5], ub[6])
                    vm = va + sm * (vb - va)
                    speed = norm(vm)
                    energy_eV = 0.5 * O2P_MASS * speed^2 / E_CHARGE
                    w_tau = rate_w * tau
                    acc.density_tau[d] += w_tau
                    acc.vel_tau[1, d] += w_tau * vm[1]
                    acc.vel_tau[2, d] += w_tau * vm[2]
                    acc.vel_tau[3, d] += w_tau * vm[3]

                    k = searchsortedlast(edges, energy_eV)
                    if k == ne + 1 && energy_eV == edges[end]
                        k = ne
                    end
                    if 1 <= k <= ne
                        et = wt_tot + sm * dwt
                        ec = wc_tot + sm * dwc
                        eh = wh_tot + sm * dwh
                        ea = wa_tot + sm * dwa
                        acc.weight_tau[d, k] += w_tau
                        acc.def_num[d, k] += w_tau * energy_eV * speed
                        acc.work_total_num[d, k] += w_tau * et
                        acc.work_conv_num[d, k] += w_tau * ec
                        acc.work_hall_num[d, k] += w_tau * eh
                        acc.work_gradpe_num[d, k] += w_tau * ea
                        if energy_eV >= min_rep_energy_eV && w_tau > best_seg_wt
                            best_seg_wt = w_tau
                            best_seg_t = ta + sm * (tb - ta)
                        end
                    end
                end
            end
            wt_tot += dwt
            wc_tot += dwc
            wh_tot += dwh
            wa_tot += dwa
        end

        if had_hit
            acc.hit_counts[d] += 1
            if best_seg_wt > acc.best_score[d]
                acc.best_score[d] = best_seg_wt
                acc.best_pid[d] = pid
                acc.best_time_s[d] = best_seg_t
                acc.best_tr[d] = (; t = collect(t_vec), u = copy(u_vec))
            end
        end
    end
    return nothing
end

function finalize_transect(acc::TransectAccumulators)
    nd, ne = size(acc.weight_tau)
    vol = acc.side_m^3
    dE = diff(acc.energy_edges_eV)
    def = zeros(Float64, nd, ne)
    w_conv = fill(NaN, nd, ne)
    w_hall = fill(NaN, nd, ne)
    w_gradpe = fill(NaN, nd, ne)
    w_total = fill(NaN, nd, ne)
    density_m3 = acc.density_tau ./ vol
    bulk_vel_ms = zeros(Float64, 3, nd)

    for d in 1:nd
        if acc.density_tau[d] > 0
            bulk_vel_ms[:, d] .= acc.vel_tau[:, d] ./ acc.density_tau[d]
        end
        for k in 1:ne
            wt = acc.weight_tau[d, k]
            if wt > 0
                def[d, k] = acc.def_num[d, k] / (4pi * vol * dE[k])
                w_conv[d, k] = acc.work_conv_num[d, k] / wt
                w_hall[d, k] = acc.work_hall_num[d, k] / wt
                w_gradpe[d, k] = acc.work_gradpe_num[d, k] / wt
                w_total[d, k] = acc.work_total_num[d, k] / wt
            end
        end
    end
    return (;
        def,
        w_conv,
        w_hall,
        w_gradpe,
        w_total,
        density_m3,
        bulk_vel_ms,
        hit_counts = acc.hit_counts,
        best_pid = acc.best_pid,
        best_time_s = acc.best_time_s,
        best_tr = acc.best_tr,
    )
end

function run_paper_analysis(run_source::Union{AbstractString, AbstractVector{<:AbstractString}}, out_dir::AbstractString)
    ispath(out_dir) && rm(out_dir; recursive = true, force = true)
    mkpath(out_dir)
    run_dirs = run_source isa AbstractString ? [String(run_source)] : collect(String, run_source)
    first_dir = first(run_dirs)
    meta = TOML.parsefile(joinpath(first_dir, "metadata.toml"))
    rm_m = Float64(meta["Rm_m"])
    fields_path = isfile(meta["source_file"]) ? meta["source_file"] : MarsTP.project_path("data", "mars_fields_with_crustal.vts")
    fields = load_mhd_fields(fields_path)
    b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)
    work_itp = build_field_work_interpolators(fields_path; include_extended = true)

    # 1. Detector at (-2, 0, 0) Rm (Paper Fig. 3d-f)
    fig3_center = Vec3(-2.0 * rm_m, 0.0, 0.0)
    fig3_side = 0.2 * rm_m
    energy_edges = 10.0 .^ range(0.0, 4.3; length = 45) # 1 eV to ~20 keV
    fig3_acc = ForwardPSDAccumulator(;
        detector_m = collect(fig3_center),
        side_m = fig3_side,
        vlim = (-500.0, 500.0),
        vgrid = 200,
        velocity_unit = :km_s,
        species = "O2+",
        coordinate_system = meta["coordinate_system"],
        energy_edges_eV = energy_edges,
    )
    b_fig3 = Vec3(b_itp(fig3_center))
    b_hat_fig3 = norm(b_fig3) > 0 ? b_fig3 / norm(b_fig3) : Vec3(1.0, 0.0, 0.0)
    pitch_edges_deg = collect(range(0.0, 180.0; length = 19)) # 10 deg bins
    fig3_epa_wt = zeros(Float64, length(energy_edges) - 1, length(pitch_edges_deg) - 1)

    fig3_observer = seg -> begin
        vm = 0.5 * (seg.v0_ms + seg.v1_ms)
        spd = norm(vm)
        spd > 0 || return nothing
        en_eV = 0.5 * O2P_MASS * spd^2 / E_CHARGE
        pa_deg = acosd(clamp(dot(vm / spd, b_hat_fig3), -1.0, 1.0))
        ke = searchsortedlast(energy_edges, en_eV)
        kp = searchsortedlast(pitch_edges_deg, pa_deg)
        if 1 <= ke <= size(fig3_epa_wt, 1) && 1 <= kp <= size(fig3_epa_wt, 2)
            fig3_epa_wt[ke, kp] += seg.rate_weight_s * (seg.t1_s - seg.t0_s)
        end
        return nothing
    end

    # 2. Circular orbit at r = 2.0 Rm in XZ plane (Y = 0), angles 0°..240° (Paper Fig. 4)
    orbit_angles_deg = collect(0.0:5.0:240.0)
    orbit_centers = [Vec3(2.0 * rm_m * cosd(ang), 0.0, 2.0 * rm_m * sind(ang)) for ang in orbit_angles_deg]
    orbit_side = 0.2 * rm_m
    orbit_acc = TransectAccumulators(orbit_centers, orbit_side, energy_edges)

    # 3. Cross-tail transect along Y ∈ [-1, 1] Rm at X = -1.75 Rm, Z = +0.75 Rm (Paper Fig. 6)
    tail_y_Rm = collect(range(-1.0, 1.0; length = 41))
    tail_x_Rm = -1.75
    tail_z_Rm = 0.75
    tail_centers = [Vec3(tail_x_Rm * rm_m, y * rm_m, tail_z_Rm * rm_m) for y in tail_y_Rm]
    tail_side = 0.2 * rm_m
    tail_acc = TransectAccumulators(tail_centers, tail_side, energy_edges)

    println("Streaming saved trajectories from $(length(run_dirs)) run director$(length(run_dirs) == 1 ? "y" : "ies") ...")
    t_start = time()
    foreach_saved_trajectory(
        length(run_dirs) == 1 ? first_dir : run_dirs; species = "O2+", include_work = true, progress = p -> begin
            if p.batch % 50 == 0
                println("  batch $(p.batch)/$(p.batches), $(p.particles) trajectories ($(round(time() - t_start; digits = 1)) s)")
                flush(stdout)
            end
        end
    ) do rec
        tr = rec.trajectory
        n_seg = length(tr.t) - 1
        n_seg >= 1 || return nothing
        accumulate_forward_psd!(
            fig3_acc,
            tr;
            rate_weight_s = rec.rate_weight_s,
            particle_id = rec.particle_id,
            segment_observer = fig3_observer,
        )
        dw_tot = rec.work !== nothing && haskey(rec.work.steps, "total") ? rec.work.steps["total"] : zeros(n_seg)
        dw_conv = rec.work !== nothing && haskey(rec.work.steps, "conv") ? rec.work.steps["conv"] : zeros(n_seg)
        dw_hall = rec.work !== nothing && haskey(rec.work.steps, "hall") ? rec.work.steps["hall"] : zeros(n_seg)
        dw_grad = rec.work !== nothing && haskey(rec.work.steps, "gradpe") ? rec.work.steps["gradpe"] : zeros(n_seg)

        _accumulate_transect!(orbit_acc, tr.t, tr.u, rec.rate_weight_s, rec.particle_id, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 80.0)
        _accumulate_transect!(tail_acc, tr.t, tr.u, rec.rate_weight_s, rec.particle_id, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 20.0)
    end
    println("Finished streaming in $(round(time() - t_start; digits = 1)) s")

    orbit_res = finalize_transect(orbit_acc)
    tail_res = finalize_transect(tail_acc)
    fig3_xz = finish_forward_psd(fig3_acc; option = :xz)

    # Save numerical results to JLD2
    jldopen(joinpath(out_dir, "paper_analysis.jld2"), "w") do f
        f["energy_edges_eV"] = energy_edges
        f["energy_centers_eV"] = orbit_acc.energy_centers_eV
        f["orbit_angles_deg"] = orbit_angles_deg
        f["orbit_def"] = orbit_res.def
        f["orbit_w_conv_eV"] = orbit_res.w_conv
        f["orbit_w_hall_eV"] = orbit_res.w_hall
        f["orbit_w_gradpe_eV"] = orbit_res.w_gradpe
        f["orbit_w_total_eV"] = orbit_res.w_total
        f["orbit_hits"] = orbit_res.hit_counts
        f["orbit_best_pid"] = orbit_res.best_pid
        f["orbit_best_time_s"] = orbit_res.best_time_s

        f["tail_y_Rm"] = tail_y_Rm
        f["tail_def"] = tail_res.def
        f["tail_bulk_vel_ms"] = tail_res.bulk_vel_ms
        f["tail_density_m3"] = tail_res.density_m3
        f["tail_w_conv_eV"] = tail_res.w_conv
        f["tail_w_hall_eV"] = tail_res.w_hall
        f["tail_w_gradpe_eV"] = tail_res.w_gradpe
        f["tail_w_total_eV"] = tail_res.w_total
        f["tail_hits"] = tail_res.hit_counts
        f["tail_best_pid"] = tail_res.best_pid
        f["tail_best_time_s"] = tail_res.best_time_s

        f["fig3_fxz_s2_m5"] = fig3_xz.psd
        f["fig3_vx_centers_ms"] = fig3_xz.velocity_centers_m_s[1]
        f["fig3_vz_centers_ms"] = fig3_xz.velocity_centers_m_s[2]
        f["fig3_omni_def"] = fig3_xz.omni_def.def
        f["fig3_epa_wt"] = fig3_epa_wt
        f["fig3_pitch_edges_deg"] = pitch_edges_deg
    end

    # Plot Figure 3: Detector at (-2, 0, 0) Rm
    _plot_fig3(fig3_xz, energy_edges, pitch_edges_deg, fig3_epa_wt, joinpath(out_dir, "fig3_detector_m2_0_0.png"))

    # Plot Figure 4: Circular Orbit at 2 Rm (DEF and cumulative work)
    _plot_fig4(orbit_angles_deg, orbit_acc.energy_centers_eV, orbit_res, joinpath(out_dir, "fig4_circular_orbit_work.png"))

    # Plot Figure 6: Cross-tail transect (B field, DEF, Bulk Velocity, cumulative work)
    b_tail_nT = zeros(Float64, 4, length(tail_centers))
    for (i, c) in enumerate(tail_centers)
        bv = Vec3(b_itp(c)) * 1.0e9
        b_tail_nT[1:3, i] .= bv
        b_tail_nT[4, i] = norm(bv)
    end
    _plot_fig6(tail_y_Rm, orbit_acc.energy_centers_eV, b_tail_nT, tail_res, joinpath(out_dir, "fig6_crosstail_work.png"))

    # Plot Figure 5 & 7 representative trajectories if found
    _plot_representative_ions(run_dirs, work_itp, rm_m, orbit_angles_deg, orbit_res, tail_y_Rm, tail_res, out_dir)

    fig_tr, _, _ = plot_trajectory(
        length(run_dirs) == 1 ? first_dir : run_dirs;
        species = "O2+",
        count = 20_000,
        max_points = 300,
        seed = 20260906,
        planes = ("XZ", "XY", "YZ"),
        limits = (-4.2, 4.2),
        boundaries = true,
        output = joinpath(out_dir, "fig3abc_trajectories_20000.png"),
    )
    plt.close(fig_tr)

    println("Paper analysis complete: $out_dir")
    return out_dir
end

function _plot_fig3(fig3_xz, energy_edges, pitch_edges_deg, fig3_epa_wt, out_png)
    fig, axes = plt.subplots(1, 3; figsize = (15, 4.4), constrained_layout = true)
    vx_km = fig3_xz.velocity_centers_m_s[1] ./ 1.0e3
    vz_km = fig3_xz.velocity_centers_m_s[2] ./ 1.0e3
    fxz = replace(permutedims(fig3_xz.psd), 0.0 => NaN)
    im0 = axes[1].pcolormesh(vx_km, vz_km, log10.(fxz); cmap = "turbo", shading = "auto")
    axes[1].set_xlim(-200, 100)
    axes[1].set_ylim(-150, 150)
    axes[1].set_xlabel("Vx (km/s)")
    axes[1].set_ylabel("Vz (km/s)")
    axes[1].set_title("(d) VDF in Vx-Vz at (-2, 0, 0) Rm")
    fig.colorbar(im0, ax = axes[1], label = "log10 F_xz (s² m⁻⁵)")

    ec = fig3_xz.omni_def.energy_centers_eV
    def_val = fig3_xz.omni_def.def
    pos = def_val .> 0
    if any(pos)
        axes[2].loglog(ec[pos], def_val[pos]; color = "tab:blue", lw = 2.0, marker = "o", ms = 3.5)
    end
    axes[2].set_xlabel("Energy (eV)")
    axes[2].set_ylabel("Omni DEF (eV / m² s eV sr)")
    axes[2].set_title("(e) Omnidirectional Energy Spectrum")
    axes[2].grid(true, which = "both", alpha = 0.3)

    pa_centers = 0.5 .* (pitch_edges_deg[1:(end - 1)] .+ pitch_edges_deg[2:end])
    epa_plot = replace(permutedims(fig3_epa_wt), 0.0 => NaN)
    im2 = axes[3].pcolormesh(ec, pa_centers, log10.(epa_plot); cmap = "turbo", shading = "auto")
    axes[3].set_xscale("log")
    axes[3].set_ylim(0, 180)
    axes[3].set_xlabel("Energy (eV)")
    axes[3].set_ylabel("Pitch Angle (deg)")
    axes[3].set_title("(f) Energy-Pitch Angle Distribution")
    fig.colorbar(im2, ax = axes[3], label = "log10 Weighted Residence (s⁻¹·s)")

    fig.savefig(out_png; dpi = 220)
    return plt.close(fig)
end

function _plot_fig4(angles_deg, energy_centers_eV, res, out_png)
    fig, axes = plt.subplots(5, 1; figsize = (10, 12.5), sharex = true, constrained_layout = true)
    def_plot = replace(permutedims(res.def), 0.0 => NaN)
    im1 = axes[1].pcolormesh(angles_deg, energy_centers_eV, log10.(def_plot); cmap = "turbo", shading = "auto")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].set_title("(b) O₂⁺ Omnidirectional DEF along 2 Rm Circular Orbit (XZ Plane, Y=0)")
    fig.colorbar(im1, ax = axes[1], label = "log10 DEF")

    panels = (
        ("(c) Mean Convective Work ⟨ε_C⟩ (eV)", res.w_conv),
        ("(d) Mean Hall Work ⟨ε_H⟩ (eV)", res.w_hall),
        ("(e) Mean Ambipolar Work ⟨ε_A⟩ (eV)", res.w_gradpe),
        ("(f) Mean Total Work ⟨ε_T⟩ (eV)", res.w_total),
    )
    norm_symlog = matplotlib.colors.SymLogNorm(linthresh = 10.0, linscale = 1.0, vmin = -1.0e4, vmax = 1.0e4, base = 10)
    for (idx, (title_str, mat)) in enumerate(panels)
        ax = axes[idx + 1]
        im = ax.pcolormesh(angles_deg, energy_centers_eV, permutedims(mat); cmap = "RdBu_r", norm = norm_symlog, shading = "auto")
        ax.set_yscale("log")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(title_str)
        fig.colorbar(im, ax = ax, label = "Work (eV)")
    end
    axes[end].set_xlabel("Orbital Angle (deg, 0° = +X, 90° = +Z, 180° = -X)")
    fig.savefig(out_png; dpi = 220)
    return plt.close(fig)
end

function _plot_fig6(tail_y_Rm, energy_centers_eV, b_tail_nT, res, out_png)
    fig, axes = plt.subplots(7, 1; figsize = (10, 15.5), sharex = true, constrained_layout = true)
    axes[1].plot(tail_y_Rm, b_tail_nT[1, :]; label = "Bx", lw = 1.8)
    axes[1].plot(tail_y_Rm, b_tail_nT[2, :]; label = "By", lw = 1.8)
    axes[1].plot(tail_y_Rm, b_tail_nT[3, :]; label = "Bz", lw = 1.8)
    axes[1].plot(tail_y_Rm, b_tail_nT[4, :]; label = "|B|", color = "black", lw = 2.0)
    axes[1].axhline(0.0; color = "gray", ls = "--", lw = 0.8)
    axes[1].set_ylabel("B (nT)")
    axes[1].set_title("(b) Magnetic Field across Magnetotail Current Sheet (X = -1.75 Rm, Z = +0.75 Rm)")
    axes[1].legend(loc = "upper right", ncol = 4, frameon = false)

    def_plot = replace(permutedims(res.def), 0.0 => NaN)
    im2 = axes[2].pcolormesh(tail_y_Rm, energy_centers_eV, log10.(def_plot); cmap = "turbo", shading = "auto")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Energy (eV)")
    axes[2].set_title("(c) O₂⁺ Omnidirectional DEF")
    fig.colorbar(im2, ax = axes[2], label = "log10 DEF")

    valid_u = res.density_m3 .> 0
    u_kms = res.bulk_vel_ms ./ 1.0e3
    for (comp, lbl, col) in ((1, "Ux", "tab:blue"), (2, "Uy", "tab:orange"), (3, "Uz", "tab:green"))
        y_vals = fill(NaN, length(tail_y_Rm))
        y_vals[valid_u] .= u_kms[comp, valid_u]
        axes[3].plot(tail_y_Rm, y_vals; label = lbl, color = col, lw = 1.8, marker = "o", ms = 3)
    end
    axes[3].axhline(0.0; color = "gray", ls = "--", lw = 0.8)
    axes[3].set_ylabel("U (km/s)")
    axes[3].set_title("(d) Reconstructed O₂⁺ Bulk Velocity")
    axes[3].legend(loc = "upper right", ncol = 3, frameon = false)

    panels = (
        ("(e) Mean Convective Work ⟨ε_C⟩ (eV)", res.w_conv, 1.0),
        ("(f) Mean Hall Work ⟨ε_H⟩ (eV)", res.w_hall, 1.0),
        ("(g) Mean Ambipolar Work 5×⟨ε_A⟩ (eV)", res.w_gradpe, 5.0),
        ("(h) Mean Total Work ⟨ε_T⟩ (eV)", res.w_total, 1.0),
    )
    norm_symlog = matplotlib.colors.SymLogNorm(linthresh = 10.0, linscale = 1.0, vmin = -2.0e3, vmax = 2.0e3, base = 10)
    for (idx, (title_str, mat, scale)) in enumerate(panels)
        ax = axes[idx + 3]
        im = ax.pcolormesh(tail_y_Rm, energy_centers_eV, permutedims(mat .* scale); cmap = "RdBu_r", norm = norm_symlog, shading = "auto")
        ax.set_yscale("log")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(title_str)
        fig.colorbar(im, ax = ax, label = "Work (eV)")
    end
    axes[end].set_xlabel("Y (Rm)")
    fig.savefig(out_png; dpi = 220)
    return plt.close(fig)
end

function _plot_representative_ions(run_dir, work_itp, rm_m, orbit_angles_deg, orbit_res, tail_y_Rm, tail_res, out_dir)
    # Pick representative particles at 90°, 120°, 150° on orbit, and Y=0 (center) and Y=-0.5 (lobe) on tail
    target_orbit_angles = (90.0, 120.0, 150.0)
    orbit_idxs = [argmin(abs.(orbit_angles_deg .- a)) for a in target_orbit_angles]
    orbit_pids = [orbit_res.best_pid[i] for i in orbit_idxs]
    orbit_times = [orbit_res.best_time_s[i] for i in orbit_idxs]

    tail_targets_y = (0.0, -0.5)
    tail_idxs = [argmin(abs.(tail_y_Rm .- y)) for y in tail_targets_y]
    tail_pids = [tail_res.best_pid[i] for i in tail_idxs]
    tail_times = [tail_res.best_time_s[i] for i in tail_idxs]

    orbit_trs = [orbit_res.best_tr[i] for i in orbit_idxs]
    tail_trs = [tail_res.best_tr[i] for i in tail_idxs]

    # Plot Fig. 5 (3 columns: 90°, 120°, 150°; 3 rows: XZ trajectory, Power, Cumulative Work)
    if all(tr -> tr !== nothing, orbit_trs)
        fig, axes = plt.subplots(3, 3; figsize = (14, 10.5), constrained_layout = true)
        for (col, (ang, pid, t_hit, tr_full)) in enumerate(zip(target_orbit_angles, orbit_pids, orbit_times, orbit_trs))
            k_end = searchsortedlast(tr_full.t, t_hit)
            k_end = clamp(k_end + 1, 2, length(tr_full.t))
            tr = (; t = tr_full.t[1:k_end], u = tr_full.u[1:k_end])
            prof = Electric_field_work_profile(tr, work_itp; species = "O2+")
            xs = [u[1] / rm_m for u in tr.u]
            zs = [u[3] / rm_m for u in tr.u]

            ax_tr = axes[1, col]
            th = range(0, 2pi; length = 180)
            ax_tr.plot(cos.(th), sin.(th); color = "black", lw = 1.2)
            ax_tr.plot(xs, zs; color = "tab:red", lw = 2.0)
            ax_tr.scatter([xs[1]], [zs[1]]; color = "tab:green", s = 35, label = "Launch")
            ax_tr.scatter([xs[end]], [zs[end]]; color = "tab:blue", s = 45, marker = "*", label = "Detector")
            ax_tr.set_xlim(-2.5, 2.5); ax_tr.set_ylim(-1.5, 2.5); ax_tr.set_aspect("equal")
            ax_tr.set_xlabel("X (Rm)"); ax_tr.set_ylabel("Z (Rm)")
            ax_tr.set_title("Orbit $(Int(ang))° (pid=$pid)")
            col == 1 && ax_tr.legend(loc = "lower left", frameon = false)

            ax_p = axes[2, col]
            ax_p.plot(prof.t, prof.power_conv_eV_s; label = "P_C", color = "tab:blue", lw = 1.6)
            ax_p.plot(prof.t, prof.power_hall_eV_s; label = "P_H", color = "tab:orange", lw = 1.6)
            ax_p.plot(prof.t, prof.power_gradpe_eV_s; label = "P_A", color = "tab:green", lw = 1.6)
            ax_p.plot(prof.t, prof.power_total_eV_s; label = "P_T", color = "black", ls = "--", lw = 1.4)
            ax_p.axhline(0.0; color = "gray", lw = 0.7, ls = ":")
            ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("Power (eV/s)")
            col == 1 && ax_p.legend(loc = "best", frameon = false)

            ax_w = axes[3, col]
            ax_w.plot(prof.t, prof.conv_eV; label = "ε_C", color = "tab:blue", lw = 1.8)
            ax_w.plot(prof.t, prof.hall_eV; label = "ε_H", color = "tab:orange", lw = 1.8)
            ax_w.plot(prof.t, prof.gradpe_eV; label = "ε_A", color = "tab:green", lw = 1.8)
            ax_w.plot(prof.t, prof.total_eV; label = "ε_T", color = "black", ls = "--", lw = 1.5)
            ax_w.axhline(0.0; color = "gray", lw = 0.7, ls = ":")
            ax_w.set_xlabel("Time (s)"); ax_w.set_ylabel("Cumulative Work (eV)")
            col == 1 && ax_w.legend(loc = "best", frameon = false)
        end
        fig.savefig(joinpath(out_dir, "fig5_orbit_representative_ions.png"); dpi = 220)
        plt.close(fig)
    end

    # Plot Fig. 7 (2 columns: Center Y=0, Lobe Y=-0.5)
    if all(tr -> tr !== nothing, tail_trs)
        fig, axes = plt.subplots(3, 2; figsize = (10.5, 10.0), constrained_layout = true)
        labels = ("Current Sheet Center (Y = 0 Rm)", "Lobe (Y = -0.5 Rm)")
        for (col, (lbl, pid, t_hit, tr_full)) in enumerate(zip(labels, tail_pids, tail_times, tail_trs))
            k_end = searchsortedlast(tr_full.t, t_hit)
            k_end = clamp(k_end + 1, 2, length(tr_full.t))
            tr = (; t = tr_full.t[1:k_end], u = tr_full.u[1:k_end])
            prof = Electric_field_work_profile(tr, work_itp; species = "O2+")
            xs = [u[1] / rm_m for u in tr.u]
            ys = [u[2] / rm_m for u in tr.u]
            zs = [u[3] / rm_m for u in tr.u]

            ax_tr = axes[1, col]
            th = range(0, 2pi; length = 180)
            ax_tr.plot(cos.(th), sin.(th); color = "black", lw = 1.2)
            ax_tr.plot(xs, zs; color = "tab:red", lw = 2.0, label = "XZ")
            ax_tr.plot(xs, ys; color = "tab:purple", lw = 1.5, ls = "--", label = "XY")
            ax_tr.set_xlim(-2.5, 2.0); ax_tr.set_ylim(-1.5, 2.0); ax_tr.set_aspect("equal")
            ax_tr.set_xlabel("X (Rm)"); ax_tr.set_ylabel("Z / Y (Rm)")
            ax_tr.set_title("$lbl (pid=$pid)")
            col == 1 && ax_tr.legend(loc = "lower left", frameon = false)

            ax_p = axes[2, col]
            ax_p.plot(prof.t, prof.power_conv_eV_s; label = "P_C", color = "tab:blue", lw = 1.6)
            ax_p.plot(prof.t, prof.power_hall_eV_s; label = "P_H", color = "tab:orange", lw = 1.6)
            ax_p.plot(prof.t, prof.power_gradpe_eV_s; label = "P_A", color = "tab:green", lw = 1.6)
            ax_p.plot(prof.t, prof.power_total_eV_s; label = "P_T", color = "black", ls = "--", lw = 1.4)
            ax_p.axhline(0.0; color = "gray", lw = 0.7, ls = ":")
            ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("Power (eV/s)")
            col == 1 && ax_p.legend(loc = "best", frameon = false)

            ax_w = axes[3, col]
            ax_w.plot(prof.t, prof.conv_eV; label = "ε_C", color = "tab:blue", lw = 1.8)
            ax_w.plot(prof.t, prof.hall_eV; label = "ε_H", color = "tab:orange", lw = 1.8)
            ax_w.plot(prof.t, prof.gradpe_eV; label = "ε_A", color = "tab:green", lw = 1.8)
            ax_w.plot(prof.t, prof.total_eV; label = "ε_T", color = "black", ls = "--", lw = 1.5)
            ax_w.axhline(0.0; color = "gray", lw = 0.7, ls = ":")
            ax_w.set_xlabel("Time (s)"); ax_w.set_ylabel("Cumulative Work (eV)")
            col == 1 && ax_w.legend(loc = "best", frameon = false)
        end
        fig.savefig(joinpath(out_dir, "fig7_crosstail_representative_ions.png"); dpi = 220)
        plt.close(fig)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) <= 2
        run_dir = length(ARGS) >= 1 ? ARGS[1] : "outputs/mc_400km_crustal_1M"
        out_dir = length(ARGS) >= 2 ? ARGS[2] : "$(run_dir)_paper"
        run_paper_analysis(run_dir, out_dir)
    else
        # Multiple chunk directories followed by output directory as last argument
        run_paper_analysis(ARGS[1:(end - 1)], ARGS[end])
    end
end
