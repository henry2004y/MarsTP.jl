# Publication-quality script for plotting cumulative electric-field work along O2+
# particle trajectories in the X-Z plane projection from release to the end of tracing:
#   - Dayside (X0 > 0) and Nightside (X0 <= 0) release rows (2 rows x 4 columns)
#   - Also renders the 2x2 full-ensemble (all-release) figure (Convective, Hall, Ambipolar, Total)
#   - Symmetric RdBu_r colormap centered at 0 (SymLogNorm) with one shared colorbar on the right
#   - Supports both MHD cases: with crustal field and without crustal field
#
# Usage:
#   julia --threads=auto --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_trajectory_work_xz.jl [CASE] [N_PARTICLES] [ALTITUDE_KM] [SAMPLING_MODE] [RECOMPUTE]
#
# Arguments:
#   CASE          : "both" (default), "with_crustal", or "without_crustal"
#   N_PARTICLES   : total number of released O2+ particles to trace per case (default: 1200)
#   ALTITUDE_KM   : release altitude in km (default: 400.0)
#   SAMPLING_MODE : "ionosphere" (default, physical O2+ Maxwellian source) or "stationary" (v0 = 0 uniform sphere)
#   RECOMPUTE     : "true" or "false" (default: "false", reuses cached .jld2 if present)

include("monte_carlo_shell.jl")

using .ShellMonteCarlo
using MarsTP
using StaticArrays
using LinearAlgebra
using Random
using JLD2
using Printf
using Statistics
using PyPlot

const Vec3 = SVector{3, Float64}
const E_CHARGE = MarsTP.TP.qᵢ
const O2P_MASS = MarsTP.TP.SpeciesDict["O2+"].m

"""
    trace_particle_work_xz(
        x0::Vec3, v0::Vec3, rate_w::Float64,
        param, ec_itp, eh_itp, ea_itp,
        dt::Float64, tmax::Float64,
        r_inner::Float64, r_outer::Float64;
        save_stride::Int = 5
    )

Fast allocation-free Boris integrator that integrates cumulative electric-field work
(`ε_C`, `ε_H`, `ε_A`, `ε_T` in eV) at every full integration step `dt`, while saving
decimated `(X/Rm, Z/Rm, Y/Rm)` trajectory coordinates and cumulative work profiles
from release (`t = 0`) to termination (`inner`, `outer`, or `time_limit`).
"""
function trace_particle_work_xz(
        x0::Vec3,
        v0::Vec3,
        rate_w::Float64,
        param,
        ec_itp,
        eh_itp,
        ea_itp,
        dt::Float64,
        tmax::Float64,
        r_inner::Float64,
        r_outer::Float64;
        save_stride::Int = 5,
    )
    qm = param.q2m
    ke_factor = 0.5 * O2P_MASS / E_CHARGE
    q_over_eV = MarsTP.TP.SpeciesDict["O2+"].q / MarsTP.TP.eV # +1.0 for O2+
    nsteps = round(Int, tmax / dt)

    x = x0
    v = v0
    t = 0.0
    e0_eV = ke_factor * dot(v0, v0)

    E = Vec3(param.E(x, t))
    B = Vec3(param.B(x, t))
    half = ShellMonteCarlo.BP.update_velocity_boris(v, E, B, -qm * dt / 4)

    cum_wc = 0.0
    cum_wh = 0.0
    cum_wa = 0.0
    cum_wt = 0.0

    max_saved = cld(nsteps, max(save_stride, 1)) + 2
    xs_Rm = Vector{Float32}(undef, max_saved)
    ys_Rm = Vector{Float32}(undef, max_saved)
    zs_Rm = Vector{Float32}(undef, max_saved)
    ts_s  = Vector{Float32}(undef, max_saved)
    wc_eV = Vector{Float32}(undef, max_saved)
    wh_eV = Vector{Float32}(undef, max_saved)
    wa_eV = Vector{Float32}(undef, max_saved)
    wt_eV = Vector{Float32}(undef, max_saved)

    nsave = 1
    xs_Rm[1] = Float32(x0[1] / Rm)
    ys_Rm[1] = Float32(x0[2] / Rm)
    zs_Rm[1] = Float32(x0[3] / Rm)
    ts_s[1]  = 0.0f0
    wc_eV[1] = 0.0f0
    wh_eV[1] = 0.0f0
    wa_eV[1] = 0.0f0
    wt_eV[1] = 0.0f0

    status = :time_limit
    last_saved_step = 0

    @inbounds for step in 1:nsteps
        half = ShellMonteCarlo.BP.update_velocity_boris(half, E, B, qm * dt / 2)
        trial = x + half * dt
        f, flag = ShellMonteCarlo.sphere_stop(x, trial, r_inner, r_outer)
        if f == 0.0 && flag == 2
            status = :inner
            break
        end
        xn = x + f * (trial - x)
        tn = (step - 1 + f) * dt
        dt_step = tn - t
        if dt_step <= 0.0
            break
        end

        r_next = norm(xn)
        query = flag == 4 ? xn * ((r_outer - 1.0e-6) / r_next) :
                flag == 2 ? xn * ((r_inner + 1.0e-6) / r_next) : xn
        En = Vec3(param.E(query, tn))
        Bn = Vec3(param.B(query, tn))
        vn = ShellMonteCarlo.BP.update_velocity_boris(half, En, Bn, qm * (f - 0.5) * dt / 2)

        # Midpoint quadrature for electric-field work increments (eV) over [t, tn]
        x_mid = 0.5 * (x + query)
        v_mid = 0.5 * (v + vn)
        E_tot_mid  = 0.5 * (E + En)
        E_conv_mid = Vec3(ec_itp(x_mid))
        E_hall_mid = Vec3(eh_itp(x_mid))
        E_amb_mid  = Vec3(ea_itp(x_mid))

        cum_wc += q_over_eV * dot(E_conv_mid, v_mid) * dt_step
        cum_wh += q_over_eV * dot(E_hall_mid, v_mid) * dt_step
        cum_wa += q_over_eV * dot(E_amb_mid,  v_mid) * dt_step
        cum_wt += q_over_eV * dot(E_tot_mid,  v_mid) * dt_step

        x = xn
        v = vn
        t = tn
        E = En
        B = Bn

        is_terminal = (flag != 1) || (step == nsteps)
        if (step % save_stride == 0) || is_terminal
            nsave += 1
            xs_Rm[nsave] = Float32(x[1] / Rm)
            ys_Rm[nsave] = Float32(x[2] / Rm)
            zs_Rm[nsave] = Float32(x[3] / Rm)
            ts_s[nsave]  = Float32(t)
            wc_eV[nsave] = Float32(cum_wc)
            wh_eV[nsave] = Float32(cum_wh)
            wa_eV[nsave] = Float32(cum_wa)
            wt_eV[nsave] = Float32(cum_wt)
            last_saved_step = step
        end

        if flag != 1
            status = flag == 2 ? :inner : :outer
            break
        end
    end

    resize!(xs_Rm, nsave)
    resize!(ys_Rm, nsave)
    resize!(zs_Rm, nsave)
    resize!(ts_s,  nsave)
    resize!(wc_eV, nsave)
    resize!(wh_eV, nsave)
    resize!(wa_eV, nsave)
    resize!(wt_eV, nsave)

    ef_eV = ke_factor * dot(v, v)
    dk_eV = ef_eV - e0_eV
    residual_eV = dk_eV - cum_wt
    field_sum_res_eV = cum_wt - (cum_wc + cum_wh + cum_wa)

    return (;
        x0_Rm = Vec3(x0 / Rm),
        is_dayside = x0[1] > 0.0,
        rate_w,
        status,
        end_time_s = t,
        xs_Rm,
        ys_Rm,
        zs_Rm,
        ts_s,
        wc_eV,
        wh_eV,
        wa_eV,
        wt_eV,
        dk_eV,
        residual_eV,
        field_sum_res_eV,
    )
end

"""
    sample_initial_particles(fields, source, n_particles::Int, altitude_km::Float64, sampling_mode::Symbol, seed::Int)

Generate `n_particles` initial O2+ states `(x0, v0, rate_w)` balanced across dayside (`X0 > 0`)
and nightside (`X0 <= 0`) on the spherical release shell at `altitude_km`.
- `:ionosphere`: samples positions uniformly in solid angle on each hemisphere (or weighted by local ionospheric properties) and draws thermal + bulk velocities from the local ionospheric O2+ Maxwellian.
- `:stationary`: uniform spherical release positions with `v0 = (0, 0, 0)` m/s (matching `hemisphere_work.jl`).
"""
function sample_initial_particles(
        source,
        n_particles::Int,
        altitude_km::Float64,
        sampling_mode::Symbol,
        seed::Int,
    )
    rng = Xoshiro(seed)
    r_shell = max(Rm + altitude_km * 1.0e3, Rinner + 1.0e-6)
    kb = MarsTP.TP.kB

    # Draw uniform spherical positions until we have at least n_particles/2 in each hemisphere
    n_half = cld(n_particles, 2)
    day_states = NamedTuple[]
    night_states = NamedTuple[]
    sizehint!(day_states, n_half)
    sizehint!(night_states, n_half)

    while length(day_states) < n_half || length(night_states) < n_half
        mu = 2.0 * rand(rng) - 1.0
        phi = 2.0 * pi * rand(rng)
        s_th = sqrt(max(1.0 - mu^2, 0.0))
        er = Vec3(s_th * cos(phi), s_th * sin(phi), mu)
        is_day = er[1] > 0.0
        if is_day && length(day_states) >= n_half
            continue
        elseif !is_day && length(night_states) >= n_half
            continue
        end

        x0 = r_shell * er
        if sampling_mode == :stationary
            v0 = Vec3(0.0, 0.0, 0.0)
            rate_w = 1.0
        else
            m_prop = ionosphere_properties(source, x0)
            # Draw from the physical local O2+ drifting Maxwellian (temperature_factor = 1.0 for direct physical sample)
            temp_eV = max(kb * m_prop.Ti / E_CHARGE, 0.01)
            vth = sqrt(temp_eV * E_CHARGE / O2P_MASS)
            v0 = Vec3(
                m_prop.Ui[1] + vth * randn(rng),
                m_prop.Ui[2] + vth * randn(rng),
                m_prop.Ui[3] + vth * randn(rng),
            )
            rate_w = max(m_prop.n * norm(m_prop.Ui), 1.0)
        end

        st = (; x0, v0, rate_w)
        if is_day
            push!(day_states, st)
        else
            push!(night_states, st)
        end
    end

    all_states = vcat(day_states[1:n_half], night_states[1:(n_particles - n_half)])
    return all_states
end

function compute_or_load_trajectory_work(
        vts_path::AbstractString,
        cache_jld2::AbstractString;
        n_particles::Int = 1200,
        altitude_km::Float64 = 400.0,
        absorption_altitude_km::Float64 = 200.0,
        dt::Float64 = 0.1,
        tmax::Float64 = 2000.0,
        save_stride::Int = 5,
        sampling_mode::Symbol = :ionosphere,
        seed::Int = 20260905,
        recompute::Bool = false,
        min_flight_steps::Int = 3,
    )
    if !recompute && isfile(cache_jld2)
        valid_cache = jldopen(cache_jld2, "r") do f
            haskey(f, "trajectories") &&
                haskey(f, "altitude_km") &&
                Float64(f["altitude_km"]) ≈ altitude_km &&
                Symbol(f["sampling_mode"]) == sampling_mode &&
                Int(f["n_particles"]) == n_particles
        end
        if valid_cache
            println("Loading cached trajectory work profiles from: ", cache_jld2)
            return jldopen(cache_jld2, "r") do f
                f["trajectories"]
            end
        end
    end

    resolved_vts = MarsTP.resolve_project_path(vts_path)
    isfile(resolved_vts) || error("MHD fields file not found: $resolved_vts")

    cfg0 = ShellMonteCarlo.Config(;
        altitude_km,
        absorption_altitude_km,
        dt,
        tmax,
        fields_path = resolved_vts,
    )

    println("=== Loading MHD fields & electric work interpolators from $(basename(resolved_vts)) ===")
    fields_raw = load_mhd_fields(resolved_vts)
    fields, source, _ = ShellMonteCarlo.validate_geometry(fields_raw.path, fields_raw, cfg0)
    param = MarsTP.mhd_param(fields; species = "O2+")

    component_itp(which::Symbol) = begin
        f = load_mhd_fields(fields.path; electric_field = which)
        MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, f.E, fields.r, fields.theta, fields.phi)
    end
    ec_itp = component_itp(:conv)
    eh_itp = component_itp(:hall)
    ea_itp = component_itp(:gradpe)

    r_inner = Rm + absorption_altitude_km * 1.0e3
    r_outer = Router

    # Oversample slightly if ionosphere mode so that even if some inward-launched ions hit 200 km immediately,
    # we retain exactly n_particles trajectories with non-trivial propagation
    oversample = sampling_mode == :ionosphere ? round(Int, 1.35 * n_particles) : n_particles
    init_states = sample_initial_particles(source, oversample, altitude_km, sampling_mode, seed)

    println("=== Tracing $(length(init_states)) O2+ particles (altitude=$(altitude_km) km, mode=$(sampling_mode), dt=$(dt) s, tmax=$(tmax) s, threads=$(Threads.nthreads())) ===")
    flush(stdout)
    t0 = time()

    raw_results = Vector{Any}(undef, length(init_states))
    Threads.@threads for i in eachindex(init_states)
        st = init_states[i]
        raw_results[i] = trace_particle_work_xz(
            st.x0, st.v0, st.rate_w,
            param, ec_itp, eh_itp, ea_itp,
            dt, tmax, r_inner, r_outer;
            save_stride,
        )
    end

    # Select balanced dayside and nightside trajectories with >= min_flight_steps saved points
    n_half = cld(n_particles, 2)
    day_kept = [r for r in raw_results if r.is_dayside && length(r.xs_Rm) >= min_flight_steps]
    night_kept = [r for r in raw_results if !r.is_dayside && length(r.xs_Rm) >= min_flight_steps]

    if length(day_kept) < n_half
        day_kept = [r for r in raw_results if r.is_dayside && length(r.xs_Rm) >= 2]
    end
    if length(night_kept) < n_half
        night_kept = [r for r in raw_results if !r.is_dayside && length(r.xs_Rm) >= 2]
    end

    trajectories = vcat(
        day_kept[1:min(n_half, length(day_kept))],
        night_kept[1:min(n_particles - n_half, length(night_kept))],
    )

    max_rel_res = maximum(abs(r.residual_eV) / max(abs(r.dk_eV), 1.0) for r in trajectories) * 100.0
    max_sum_res = maximum(abs(r.field_sum_res_eV) / max(abs(r.dk_eV), 1.0) for r in trajectories) * 100.0
    n_out = count(r -> r.status == :outer, trajectories)
    n_in  = count(r -> r.status == :inner, trajectories)
    n_tl  = count(r -> r.status == :time_limit, trajectories)
    @printf(
        "Completed in %.1f s: kept=%d (day=%d, night=%d | outer=%d, inner=%d, time_limit=%d) | max |ΔK - W_tot| = %.3f%% | max |W_tot - ∑W_i| = %.3f%%\n",
        time() - t0, length(trajectories),
        count(r -> r.is_dayside, trajectories), count(r -> !r.is_dayside, trajectories),
        n_out, n_in, n_tl, max_rel_res, max_sum_res
    )

    mkpath(dirname(abspath(cache_jld2)))
    jldopen(cache_jld2, "w") do f
        f["trajectories"] = trajectories
        f["altitude_km"] = altitude_km
        f["sampling_mode"] = String(sampling_mode)
        f["n_particles"] = n_particles
        f["vts_file"] = basename(resolved_vts)
    end
    return trajectories
end

function _build_xz_segments_and_work(trajectories)
    n_segs = sum(max(0, length(r.xs_Rm) - 1) for r in trajectories; init = 0)
    segs = Vector{Matrix{Float64}}(undef, n_segs)
    wc_mid = Vector{Float64}(undef, n_segs)
    wh_mid = Vector{Float64}(undef, n_segs)
    wa_mid = Vector{Float64}(undef, n_segs)
    wt_mid = Vector{Float64}(undef, n_segs)

    idx = 0
    @inbounds for r in trajectories
        xs = r.xs_Rm
        zs = r.zs_Rm
        wc = r.wc_eV
        wh = r.wh_eV
        wa = r.wa_eV
        wt = r.wt_eV
        for i in 1:(length(xs) - 1)
            idx += 1
            mat = Matrix{Float64}(undef, 2, 2)
            mat[1, 1] = Float64(xs[i])
            mat[1, 2] = Float64(zs[i])
            mat[2, 1] = Float64(xs[i + 1])
            mat[2, 2] = Float64(zs[i + 1])
            segs[idx] = mat
            wc_mid[idx] = 0.5 * (Float64(wc[i]) + Float64(wc[i + 1]))
            wh_mid[idx] = 0.5 * (Float64(wh[i]) + Float64(wh[i + 1]))
            wa_mid[idx] = 0.5 * (Float64(wa[i]) + Float64(wa[i + 1]))
            wt_mid[idx] = 0.5 * (Float64(wt[i]) + Float64(wt[i + 1]))
        end
    end
    return segs, (wc_mid, wh_mid, wa_mid, wt_mid)
end

function _style_xz_axis!(ax, altitude_km::Float64; xlim = (-4.0, 4.0), ylim = (-4.0, 4.0))
    bs_mpb(ax, "XZ"; draw_bs = true, draw_mpb = false, xmin = xlim[1], sphere = false, boundary_color = "0.25", boundary_ls = "--", boundary_lw = 1.1)
    bs_mpb(ax, "XZ"; draw_bs = false, draw_mpb = true, xmin = xlim[1], sphere = false, boundary_color = "0.25", boundary_ls = "-", boundary_lw = 1.1)
    plot_mars(ax; radius = 1.0, zorder = 5)

    # Dashed gray circle indicating release sphere altitude
    th = range(0.0, 2pi; length = 360)
    r_rel_Rm = (Rm + altitude_km * 1.0e3) / Rm
    ax.plot(r_rel_Rm .* cos.(th), r_rel_Rm .* sin.(th); color = "0.55", ls = ":", lw = 0.9, zorder = 4)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks(-4:2:4)
    ax.set_yticks(-4:2:4)
    ax.grid(true; alpha = 0.16, lw = 0.6)
    return ax
end

"""
    plot_trajectory_work_2x2(
        trajectories, out_png::AbstractString;
        case_title::AbstractString = "With Crustal Field",
        altitude_km::Float64 = 400.0,
        vmax_eV::Float64 = 2.0e4,
        linthresh_eV::Float64 = 1.0e1,
    )

Plot a 2×2 figure of cumulative electric-field work (`ε_C`, `ε_H`, `ε_A`, `ε_T` in eV)
along O2+ trajectories projected onto the X-Z plane, using a symmetric `RdBu_r`
`SymLogNorm` centered at 0 and one shared colorbar on the right.
"""
function plot_trajectory_work_2x2(
        trajectories,
        out_png::AbstractString;
        case_title::AbstractString = "With Crustal Field",
        altitude_km::Float64 = 400.0,
        vmax_eV::Float64 = 2.0e4,
        linthresh_eV::Float64 = 1.0e1,
    )
    segs, (wc_mid, wh_mid, wa_mid, wt_mid) = _build_xz_segments_and_work(trajectories)
    work_arrays = (wc_mid, wh_mid, wa_mid, wt_mid)
    panel_titles = (
        L"Convective work $\varepsilon_C$",
        L"Hall work $\varepsilon_H$",
        L"Ambipolar work $\varepsilon_A$",
        L"Total work $\varepsilon_T$",
    )
    tags = ("a", "b", "c", "d")

    # Sort segments by |work| ascending in each panel so high-work trajectories render cleanly on top
    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 18
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 18
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4

    fig, axes = plt.subplots(2, 2; figsize = (13.4, 11.8))
    fig.subplots_adjust(left = 0.08, right = 0.865, bottom = 0.08, top = 0.915, wspace = 0.08, hspace = 0.15)
    fig.suptitle("O\$_2^+\$ Cumulative Electric-Field Work ($case_title, $(Int(round(altitude_km))) km release)"; fontsize = 21.5, fontweight = "bold", y = 0.975)

    norm_w = matplotlib.colors.SymLogNorm(
        linthresh = linthresh_eV,
        linscale = 0.8,
        vmin = -vmax_eV,
        vmax = vmax_eV,
        base = 10,
    )

    lc_last = nothing
    for k in 1:4
        row = cld(k, 2)
        col = mod1(k, 2)
        ax = axes[row, col]
        _style_xz_axis!(ax, altitude_km)

        vals = work_arrays[k]
        perm = sortperm(abs.(vals))
        lc = matplotlib.collections.LineCollection(
            segs[perm];
            array = clamp.(vals[perm], -vmax_eV, vmax_eV),
            cmap = "RdBu_r",
            norm = norm_w,
            linewidths = 0.55,
            alpha = 0.82,
            zorder = 3,
            rasterized = true,
        )
        ax.add_collection(lc)
        lc_last = lc

        # Panel label (a)-(d)
        ax.text(
            0.03, 0.95, "($(tags[k]))";
            transform = ax.transAxes, fontsize = 20.5, fontweight = "bold",
            va = "top", ha = "left",
            bbox = Dict("boxstyle" => "square,pad=0.18", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.88),
            zorder = 10,
        )

        # Endpoint work statistics annotation
        end_vals = [
            k == 1 ? Float64(r.wc_eV[end]) :
            k == 2 ? Float64(r.wh_eV[end]) :
            k == 3 ? Float64(r.wa_eV[end]) : Float64(r.wt_eV[end])
            for r in trajectories
        ]
        esc_vals = [
            k == 1 ? Float64(r.wc_eV[end]) :
            k == 2 ? Float64(r.wh_eV[end]) :
            k == 3 ? Float64(r.wa_eV[end]) : Float64(r.wt_eV[end])
            for r in trajectories if r.status == :outer
        ]
        mean_esc = isempty(esc_vals) ? mean(end_vals) : mean(esc_vals)
        max_abs = maximum(abs, end_vals)
        stat_txt = @sprintf("escape mean = %+.0f eV | max |ε| = %.0f eV", mean_esc, max_abs)
        ax.text(
            0.97, 0.04, stat_txt;
            transform = ax.transAxes, fontsize = 13.5, color = "0.15",
            va = "bottom", ha = "right",
            bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "white", "edgecolor" => "0.75", "alpha" => 0.90),
            zorder = 10,
        )

        ax.set_title(panel_titles[k]; fontsize = 20.0, pad = 8)
        row == 1 && ax.set_xticklabels([])
        col == 2 && ax.set_yticklabels([])
        col == 1 && ax.set_ylabel(L"$Z_{\mathrm{MSE}}$ ($R_M$)"; fontsize = 20.0)
        row == 2 && ax.set_xlabel(L"$X_{\mathrm{MSE}}$ ($R_M$)"; fontsize = 20.0)
        ax.tick_params(axis = "both", which = "major", labelsize = 18.0, length = 7.0, width = 1.4)
    end

    # Single shared colorbar on the right spanning both rows
    pos_top = axes[1, 2].get_position()
    pos_bot = axes[2, 2].get_position()
    cax = fig.add_axes([pos_top.x1 + 0.020, pos_bot.y0, 0.022, pos_top.y1 - pos_bot.y0])
    cb = fig.colorbar(lc_last; cax = cax)
    ticks = [-1.0e4, -1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4]
    cb.set_ticks(ticks)
    cb.set_ticklabels([L"$-10^4$", L"$-10^3$", L"$-10^2$", L"$-10^1$", "0", L"$10^1$", L"$10^2$", L"$10^3$", L"$10^4$"])
    cb.ax.tick_params(which = "major", labelsize = 18.0, length = 6.5, width = 1.4)
    cb.ax.tick_params(which = "minor", length = 3.5, width = 1.0)
    cb.set_label(L"Cumulative electric-field work $\varepsilon$ (eV)"; fontsize = 20.5, labelpad = 10)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved 2x2 trajectory work figure to: ", abspath(out_png))
    return out_png
end

"""
    plot_hemisphere_trajectory_work_2x4(
        trajectories, out_png::AbstractString;
        case_title::AbstractString = "With Crustal Field",
        altitude_km::Float64 = 400.0,
        vmax_eV::Float64 = 2.0e4,
        linthresh_eV::Float64 = 1.0e1,
    )

Plot a 2×4 Dayside (`X_0 > 0`) vs. Nightside (`X_0 <= 0`) figure of cumulative
electric-field work (`ε_C`, `ε_H`, `ε_A`, `ε_T` in eV) along O2+ trajectories
projected onto the X-Z plane, with a symmetric `RdBu_r` `SymLogNorm` centered at 0
and one shared colorbar on the right.
"""
function plot_hemisphere_trajectory_work_2x4(
        trajectories,
        out_png::AbstractString;
        case_title::AbstractString = "With Crustal Field",
        altitude_km::Float64 = 400.0,
        vmax_eV::Float64 = 2.0e4,
        linthresh_eV::Float64 = 1.0e1,
    )
    dayside = [r for r in trajectories if r.is_dayside]
    nightside = [r for r in trajectories if !r.is_dayside]

    col_titles = (
        L"Convective work $\varepsilon_C$",
        L"Hall work $\varepsilon_H$",
        L"Ambipolar work $\varepsilon_A$",
        L"Total work $\varepsilon_T$",
    )
    tags = ("a", "b", "c", "d", "e", "f", "g", "h")

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4

    fig, axes = plt.subplots(2, 4; figsize = (22.0, 11.4))
    fig.subplots_adjust(left = 0.058, right = 0.905, bottom = 0.085, top = 0.905, wspace = 0.075, hspace = 0.14)
    fig.suptitle("O\$_2^+\$ Cumulative Electric-Field Work: Dayside vs. Nightside ($case_title, $(Int(round(altitude_km))) km release)"; fontsize = 22.0, fontweight = "bold", y = 0.975)

    norm_w = matplotlib.colors.SymLogNorm(
        linthresh = linthresh_eV,
        linscale = 0.8,
        vmin = -vmax_eV,
        vmax = vmax_eV,
        base = 10,
    )

    lc_last = nothing
    for (row, (group, hemi_name)) in enumerate(((dayside, "Dayside (\$X_0 > 0\$)"), (nightside, "Nightside (\$X_0 \\leq 0\$)")))
        segs, work_arrays = _build_xz_segments_and_work(group)
        for col in 1:4
            idx = (row - 1) * 4 + col
            ax = axes[row, col]
            _style_xz_axis!(ax, altitude_km)

            vals = work_arrays[col]
            if !isempty(vals)
                perm = sortperm(abs.(vals))
                lc = matplotlib.collections.LineCollection(
                    segs[perm];
                    array = clamp.(vals[perm], -vmax_eV, vmax_eV),
                    cmap = "RdBu_r",
                    norm = norm_w,
                    linewidths = 0.58,
                    alpha = 0.84,
                    zorder = 3,
                    rasterized = true,
                )
                ax.add_collection(lc)
                lc_last = lc
            end

            ax.text(
                0.03, 0.95, "($(tags[idx]))";
                transform = ax.transAxes, fontsize = 19.5, fontweight = "bold",
                va = "top", ha = "left",
                bbox = Dict("boxstyle" => "square,pad=0.16", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.88),
                zorder = 10,
            )

            esc_vals = [
                col == 1 ? Float64(r.wc_eV[end]) :
                col == 2 ? Float64(r.wh_eV[end]) :
                col == 3 ? Float64(r.wa_eV[end]) : Float64(r.wt_eV[end])
                for r in group if r.status == :outer
            ]
            end_vals = [
                col == 1 ? Float64(r.wc_eV[end]) :
                col == 2 ? Float64(r.wh_eV[end]) :
                col == 3 ? Float64(r.wa_eV[end]) : Float64(r.wt_eV[end])
                for r in group
            ]
            if !isempty(end_vals)
                mean_esc = isempty(esc_vals) ? mean(end_vals) : mean(esc_vals)
                max_abs = maximum(abs, end_vals)
                stat_txt = @sprintf("esc mean = %+.0f eV | max = %.0f eV", mean_esc, max_abs)
                ax.text(
                    0.97, 0.04, stat_txt;
                    transform = ax.transAxes, fontsize = 12.2, color = "0.15",
                    va = "bottom", ha = "right",
                    bbox = Dict("boxstyle" => "round,pad=0.20", "facecolor" => "white", "edgecolor" => "0.75", "alpha" => 0.90),
                    zorder = 10,
                )
            end

            row == 1 && ax.set_title(col_titles[col]; fontsize = 19.5, pad = 8)
            row == 1 && ax.set_xticklabels([])
            col > 1 && ax.set_yticklabels([])
            col == 1 && ax.set_ylabel("$hemi_name\n\$Z_{\\mathrm{MSE}}\$ (\$R_M\$)"; fontsize = 19.0)
            row == 2 && ax.set_xlabel(L"$X_{\mathrm{MSE}}$ ($R_M$)"; fontsize = 19.5)
            ax.tick_params(axis = "both", which = "major", labelsize = 17.5, length = 7.0, width = 1.4)
        end
    end

    # Single shared colorbar on the right spanning both rows
    pos_top = axes[1, 4].get_position()
    pos_bot = axes[2, 4].get_position()
    cax = fig.add_axes([pos_top.x1 + 0.014, pos_bot.y0, 0.015, pos_top.y1 - pos_bot.y0])
    cb = fig.colorbar(lc_last; cax = cax)
    ticks = [-1.0e4, -1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4]
    cb.set_ticks(ticks)
    cb.set_ticklabels([L"$-10^4$", L"$-10^3$", L"$-10^2$", L"$-10^1$", "0", L"$10^1$", L"$10^2$", L"$10^3$", L"$10^4$"])
    cb.ax.tick_params(which = "major", labelsize = 18.0, length = 6.5, width = 1.4)
    cb.ax.tick_params(which = "minor", length = 3.5, width = 1.0)
    cb.set_label(L"Cumulative electric-field work $\varepsilon$ (eV)"; fontsize = 20.5, labelpad = 10)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved 2x4 hemisphere trajectory work figure to: ", abspath(out_png))
    return out_png
end

function run_trajectory_work_xz_plots(;
        case_mode::Symbol = :both,
        n_particles::Int = 1200,
        altitude_km::Float64 = 400.0,
        sampling_mode::Symbol = :ionosphere,
        recompute::Bool = false,
        vmax_eV::Float64 = 2.0e4,
        linthresh_eV::Float64 = 1.0e1,
    )
    cases = if case_mode == :both
        (
            ("with_crustal", "With Crustal Field", MarsTP.data_path("mars_fields_with_crustal.vts"), joinpath(pwd(), "outputs", "paper_figures_with_crustal")),
            ("without_crustal", "Without Crustal Field", MarsTP.data_path("mars_fields_without_crustal.vts"), joinpath(pwd(), "outputs", "paper_figures_without_crustal")),
        )
    elseif case_mode in (:with_crustal, :with, :crustal)
        (
            ("with_crustal", "With Crustal Field", MarsTP.data_path("mars_fields_with_crustal.vts"), joinpath(pwd(), "outputs", "paper_figures_with_crustal")),
        )
    elseif case_mode in (:without_crustal, :without, :no_crustal)
        (
            ("without_crustal", "Without Crustal Field", MarsTP.data_path("mars_fields_without_crustal.vts"), joinpath(pwd(), "outputs", "paper_figures_without_crustal")),
        )
    else
        error("Unsupported case_mode=$case_mode; expected :both, :with_crustal, or :without_crustal")
    end

    generated = String[]
    for (tag, title_str, vts_path, out_dir) in cases
        mkpath(out_dir)
        alt_int = Int(round(altitude_km))
        cache_jld2 = joinpath(out_dir, "trajectory_work_xz_$(alt_int)km_$(sampling_mode).jld2")
        trajs = compute_or_load_trajectory_work(
            vts_path,
            cache_jld2;
            n_particles,
            altitude_km,
            sampling_mode,
            recompute,
        )

        out_2x2 = joinpath(out_dir, "trajectory_work_xz_2x2_$(alt_int)km.png")
        out_2x4 = joinpath(out_dir, "trajectory_work_xz_dayside_nightside_$(alt_int)km.png")

        plot_trajectory_work_2x2(trajs, out_2x2; case_title = title_str, altitude_km, vmax_eV, linthresh_eV)
        plot_hemisphere_trajectory_work_2x4(trajs, out_2x4; case_title = title_str, altitude_km, vmax_eV, linthresh_eV)
        push!(generated, out_2x2, out_2x4)
    end
    return generated
end

if abspath(PROGRAM_FILE) == @__FILE__
    case_arg = length(ARGS) >= 1 ? Symbol(lowercase(ARGS[1])) : :both
    n_part   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1200
    alt_km   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 400.0
    smode    = length(ARGS) >= 4 ? Symbol(lowercase(ARGS[4])) : :ionosphere
    recomp   = length(ARGS) >= 5 ? (lowercase(ARGS[5]) in ("true", "1", "yes")) : false

    run_trajectory_work_xz_plots(;
        case_mode = case_arg,
        n_particles = n_part,
        altitude_km = alt_km,
        sampling_mode = smode,
        recompute = recomp,
    )
end

