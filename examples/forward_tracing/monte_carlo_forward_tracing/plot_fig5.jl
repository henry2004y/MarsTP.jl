# Publication-quality script for Figure 5 of the Mars O2+ test-particle manuscript:
#   Reorganized 2 rows x 3 columns along circular virtual spacecraft orbit at r = 2.5 Rm (0° to 240° in XZ plane):
#     (a) XZ projection of 20,000 O2+ trajectories (blue-ish colormap), BS (dashed), MPB (solid),
#         Mars disk, and circular virtual spacecraft trajectory at r = 2.5 Rm
#     (b) O2+ PSD (s^3 m^-6) vs Orbit Angle (0°..240°) and Energy (10^0..10^5 eV) on black background
#         with +E / -E hemisphere header bar
#     (c) 1D energy-integrated mean cumulative electric-field work (<ε_T>, <ε_C>, <ε_H>, <ε_A>) vs Orbit Angle
#     (d) 2D Mean Convective Work <ε_C> (eV) vs Orbit Angle and Energy
#     (e) 2D Mean Hall Work <ε_H> (eV) vs Orbit Angle and Energy
#     (f) 2D Mean Ambipolar Work <ε_A> (eV) vs Orbit Angle and Energy
#     (d)-(f) use black background for unoccupied bins and RdBu_r SymLogNorm(linthresh = 10.0 eV)
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig5.jl [PAPER_DIR] [OUTPUT_PNG] [RAW_RUN_DIR]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :paper_style)
    include(joinpath(@__DIR__, "plot_style.jl"))
end

if !isdefined(@__MODULE__, :_load_sampled_trajectories)
    include(joinpath(@__DIR__, "plot_fig3.jl"))
end

function plot_fig5(
        paper_dir::AbstractString = _default_paper_dir(),
        out_png::AbstractString = joinpath(paper_dir, "fig5.png");
        raw_run_dir = nothing,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    d = jldopen(jld_path, "r") do f
        angles = Vector{Float64}(f["orbit_angles_deg"])
        ec = Vector{Float64}(f["energy_centers_eV"])
        edges = Vector{Float64}(f["energy_edges_eV"])
        def = Matrix{Float64}(f["orbit_def"])
        r_orbit = haskey(f, "orbit_radius_Rm") ? Float64(f["orbit_radius_Rm"]) : 2.5

        mass = MarsTP.TP.SpeciesDict["O2+"].m
        qe = MarsTP.TP.qᵢ
        psd = if haskey(f, "orbit_psd_s3_m6")
            Matrix{Float64}(f["orbit_psd_s3_m6"])
        else
            factor = (mass / qe)^2 ./ (2.0 .* (ec') .^ 2)
            def .* factor
        end

        w_conv = Matrix{Float64}(f["orbit_w_conv_eV"])
        w_hall = Matrix{Float64}(f["orbit_w_hall_eV"])
        w_grad = Matrix{Float64}(f["orbit_w_gradpe_eV"])

        ro = f["raw_orbit25"]
        w_tau = Matrix{Float64}(ro["weight_tau"])
        wc_num = Matrix{Float64}(ro["work_conv_num"])
        wh_num = Matrix{Float64}(ro["work_hall_num"])
        wa_num = Matrix{Float64}(ro["work_gradpe_num"])
        wt_num = Matrix{Float64}(ro["work_total_num"])

        n_ang = length(angles)
        wc_1d = fill(NaN, n_ang)
        wh_1d = fill(NaN, n_ang)
        wa_1d = fill(NaN, n_ang)
        wt_1d = fill(NaN, n_ang)
        for j in 1:n_ang
            wt_j = sum(@view w_tau[j, :])
            if wt_j > 0.0
                wc_1d[j] = sum(@view wc_num[j, :]) / wt_j
                wh_1d[j] = sum(@view wh_num[j, :]) / wt_j
                wa_1d[j] = sum(@view wa_num[j, :]) / wt_j
                wt_1d[j] = sum(@view wt_num[j, :]) / wt_j
            end
        end

        (; angles, ec, edges, psd, w_conv, w_hall, w_grad, w_tau, wc_1d, wh_1d, wa_1d, wt_1d, r_orbit)
    end

    paths = _load_sampled_trajectories(paper_dir, raw_run_dir; count = 20_000)

    st = paper_style(18.4)

    fig = plt.figure(figsize = (18.4, 10.6), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.04, wspace = 0.05, hspace = 0.06)
    sfigs = fig.subfigures(2, 1; height_ratios = [1.05, 1.0])
    sfig_top = sfigs[1]
    sfig_bot = sfigs[2]

    axes_top = sfig_top.subplots(1, 3)
    ax_a = axes_top[1]
    ax_b = axes_top[2]
    ax_c = axes_top[3]

    axes_bot = sfig_bot.subplots(1, 3; sharex = true, sharey = true)
    ax_d = axes_bot[1]
    ax_e = axes_bot[2]
    ax_f = axes_bot[3]

    # (a) XZ trajectories + virtual orbit arc (2.5 Rm)
    _draw_blue_trajectories!(ax_a, paths, "XZ")
    bs_mpb(ax_a, "XZ"; draw_bs = true, draw_mpb = false, xmin = -4.0, boundary_color = REF_BLACK, boundary_ls = "--", boundary_lw = 1.1)
    bs_mpb(ax_a, "XZ"; draw_bs = false, draw_mpb = true, xmin = -4.0, boundary_color = REF_BLACK, boundary_ls = "-", boundary_lw = 1.1)
    plot_mars(ax_a; radius = 1.0, zorder = 5)

    ang_arc = range(0.0, 240.0; length = 241)
    r_orb = d.r_orbit
    ax_a.plot(r_orb .* cosd.(ang_arc), r_orb .* sind.(ang_arc); color = COL_ACCENT, lw = 2.6, zorder = 7)
    mark_angles = (0, 60, 120, 180, 240)
    for ang in mark_angles
        xm = r_orb * cosd(ang)
        zm = r_orb * sind(ang)
        ax_a.scatter([xm], [zm]; s = 34, facecolor = COL_ACCENT, edgecolor = "black", lw = 0.6, zorder = 8)
        dx, dz, ha_str, va_str = if ang == 0
            (0.10, 0.0, "left", "center")
        elseif ang == 60
            (0.08, 0.14, "left", "bottom")
        elseif ang == 120
            (-0.08, 0.14, "right", "bottom")
        elseif ang == 180
            (-0.12, 0.0, "right", "center")
        else
            (-0.08, -0.16, "center", "top")
        end
        ax_a.text(xm + dx, zm + dz, "$(ang)°"; fontsize = st.annot, color = "black", ha = ha_str, va = va_str, zorder = 9)
    end
    ax_a.text(2.8, -1.0, "Virtual\ntrajectory"; color = COL_ACCENT, fontweight = "bold", fontsize = st.annot, ha = "center", va = "center")
    ax_a.set_xlim(-4.0, 4.0)
    ax_a.set_ylim(-3.0, 4.0)
    ax_a.set_aspect("equal")
    ax_a.set_xticks(-4:2:4)
    ax_a.set_yticks(-3:1:4)
    ax_a.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
    ax_a.set_xlabel("\$X_{MSE}\$ (\$R_M\$)"; fontsize = st.label)
    ax_a.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = st.label)
    ax_a.text(0.03, 0.95, "(a)"; transform = ax_a.transAxes, fontsize = st.panel, fontweight = "bold", va = "top", ha = "left")

    # (b) PSD vs Orbit Angle and Energy
    ax_b.set_facecolor(PAPER_BAD_COLOR)
    cmap_b = paper_cmap_continuous()
    psd_plot = copy(permutedims(d.psd))
    psd_plot[psd_plot .<= 0.0] .= NaN
    norm_b = matplotlib.colors.LogNorm(vmin = 1.0e-20, vmax = 1.0e-4)
    im_b = ax_b.pcolormesh(d.angles, d.ec, psd_plot; cmap = cmap_b, norm = norm_b, shading = "auto", rasterized = true)
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, 240.0)
    ax_b.set_ylim(1.0e0, 1.0e5)
    ax_b.set_xticks(0:60:240)
    ax_b.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
    ax_b.tick_params(axis = "y", which = "minor", length = st.min_sz, width = st.min_w)
    ax_b.set_xlabel("Orbit Angle (deg)"; fontsize = st.label)
    ax_b.set_ylabel("Energy (eV)"; fontsize = st.label)
    ax_b.text(0.03, 0.95, "(b)"; transform = ax_b.transAxes, color = "white", fontsize = st.panel, fontweight = "bold", va = "top", ha = "left")

    ax_b.annotate(
        "Beam", xy = (96.0, 1.4e4), xytext = (122.0, 3.6e4),
        color = "white", fontsize = st.annot, ha = "left", va = "center",
        arrowprops = Dict("arrowstyle" => "->", "color" => "white", "lw" => 1.6),
    )
    ax_b.annotate(
        "Low-energy\nions", xy = (182.0, 1.0e1), xytext = (152.0, 1.2e1),
        color = "white", fontsize = st.annot_sm, ha = "right", va = "center",
        arrowprops = Dict("arrowstyle" => "->", "color" => "white", "lw" => 1.6),
    )

    # Top +E / -E hemisphere bar above panel b
    ax_b.plot([0.0, 240.0], [1.04, 1.04]; color = REF_BLACK, lw = 1.1, transform = ax_b.get_xaxis_transform(), clip_on = false)
    for xv in (0.0, 180.0, 240.0)
        ax_b.plot([xv, xv], [1.01, 1.07]; color = REF_BLACK, lw = 1.1, transform = ax_b.get_xaxis_transform(), clip_on = false)
    end
    ax_b.text(90.0, 1.08, "+E hemisphere"; transform = ax_b.get_xaxis_transform(), fontsize = st.annot, ha = "center", va = "bottom", clip_on = false)
    ax_b.text(210.0, 1.08, "−E"; transform = ax_b.get_xaxis_transform(), fontsize = st.annot, ha = "center", va = "bottom", clip_on = false)

    cb_b = sfig_top.colorbar(im_b; ax = ax_b, location = "right", fraction = 0.048, pad = 0.03)
    style_colorbar!(cb_b, st; label = "PSD (s\$^3\$ m\$^{-6}\$)")

    # (c) 1D Energy-integrated mean cumulative work vs Orbit Angle
    ax_c.axhline(0.0; color = REF_GRAY, ls = ":", lw = 1.2)
    ax_c.axvline(180.0; color = REF_GRAY, ls = "--", lw = 1.2)
    ax_c.plot(d.angles, d.wt_1d; color = COL_TOTAL, ls = "--", lw = st.lw_total, label = "\$\\langle\\varepsilon_T\\rangle\$ (Total)", zorder = 5)
    ax_c.plot(d.angles, d.wc_1d; color = COL_CONV, lw = st.lw_curve, label = "\$\\langle\\varepsilon_C\\rangle\$ (Conv)", zorder = 4)
    ax_c.plot(d.angles, d.wh_1d; color = COL_HALL, lw = st.lw_curve, label = "\$\\langle\\varepsilon_H\\rangle\$ (Hall)", zorder = 4)
    ax_c.plot(d.angles, d.wa_1d; color = COL_AMB, lw = st.lw_curve, label = "\$\\langle\\varepsilon_A\\rangle\$ (Amb)", zorder = 4)
    ax_c.set_yscale("symlog"; linthresh = 10.0, linscale = 0.8)
    ax_c.set_xlim(0.0, 240.0)
    ax_c.set_ylim(-1.0e3, 2.5e4)
    ax_c.set_xticks(0:60:240)
    ax_c.set_yticks([-1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4])
    ax_c.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
    ax_c.set_xlabel("Orbit Angle (deg)"; fontsize = st.label)
    ax_c.set_ylabel("Mean cumulative work (eV)"; fontsize = st.label)
    ax_c.grid(true; which = "major", alpha = 0.25)
    ax_c.legend(loc = "lower left", fontsize = st.legend, frameon = true, facecolor = "white", edgecolor = "0.78", framealpha = 0.92)
    ax_c.text(0.03, 0.95, "(c)"; transform = ax_c.transAxes, fontsize = st.panel, fontweight = "bold", va = "top", ha = "left", bbox = Dict("boxstyle" => "square,pad=0.12", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.78), zorder = 10)
    ax_c.text(176.0, 1.7e4, "+E"; fontsize = st.annot, color = "0.35", ha = "right", va = "center")
    ax_c.text(184.0, 1.7e4, "−E"; fontsize = st.annot, color = "0.35", ha = "left", va = "center")

    # (d)-(f) 2D Mean cumulative work spectrograms (RdBu_r, SymLogNorm linthresh = 10.0, dark-gray unoccupied bins)
    cmap_w = paper_cmap_diverging()
    norm_w = matplotlib.colors.SymLogNorm(linthresh = 10.0, linscale = 0.8, vmin = -1.0e4, vmax = 1.0e4, base = 10)
    occ_perm = permutedims(d.w_tau .> 0.0)

    work_panels = (
        (ax_d, d.w_conv, "(d)", "\$\\langle\\varepsilon_C\\rangle\$ (Convective)", true),
        (ax_e, d.w_hall, "(e)", "\$\\langle\\varepsilon_H\\rangle\$ (Hall)", false),
        (ax_f, d.w_grad, "(f)", "\$\\langle\\varepsilon_A\\rangle\$ (Ambipolar)", false),
    )
    im_w = nothing
    for (ax, mat, tag, sym_lbl, show_y) in work_panels
        ax.set_facecolor(PAPER_BAD_COLOR)
        mat_plot = copy(permutedims(mat))
        mat_plot[.!occ_perm .| .!isfinite.(mat_plot)] .= NaN
        im_w = ax.pcolormesh(d.angles, d.ec, mat_plot; cmap = cmap_w, norm = norm_w, shading = "auto", rasterized = true)
        ax.set_yscale("log")
        ax.set_xlim(0.0, 240.0)
        ax.set_ylim(1.0e0, 1.0e5)
        ax.set_xticks(0:60:240)
        ax.set_yticks([1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5])
        ax.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        ax.tick_params(axis = "y", which = "minor", length = st.min_sz, width = st.min_w)
        ax.set_xlabel("Orbit Angle (deg)"; fontsize = st.label)
        if show_y
            ax.set_ylabel("Energy (eV)"; fontsize = st.label)
        else
            ax.tick_params(labelleft = false)
        end
        ax.text(0.03, 0.94, tag; transform = ax.transAxes, color = "white", fontsize = st.panel, fontweight = "bold", va = "top", ha = "left", zorder = 10)
        ax.text(0.96, 0.93, sym_lbl; transform = ax.transAxes, color = "white", fontsize = st.annot, va = "top", ha = "right", zorder = 10)
    end

    cb_w = sfig_bot.colorbar(im_w; ax = collect(axes_bot), location = "right", fraction = 0.022, pad = 0.02)
    cb_w.set_ticks([-1.0e4, -1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4])
    style_colorbar!(cb_w, st; label = "Mean cumulative work \$\\langle\\varepsilon_\\ell\\rangle\$ (eV)")

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 5 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : _default_paper_dir()
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig5.png")
    rdir = length(ARGS) >= 3 ? ARGS[3] : nothing
    plot_fig5(pdir, out; raw_run_dir = rdir)
end
