# Publication-quality script for Figure 4 of the Mars O2+ test-particle manuscript:
#   Top row:
#     (a) XZ projection of 20,000 O2+ trajectories (blue-ish colormap), BS, MPB, Mars disk,
#         and circular virtual spacecraft trajectory at r = 2.5 Rm (0° to 240°)
#     (b) O2+ PSD (s^3 m^-6) vs Orbit Angle (0°..240°) and Energy (10^0..10^5 eV) on black background,
#         with +E / -E hemisphere header bar and "Beam" / "Low-energy ions" annotations
#   Bottom two rows (2x2 grid):
#     (c) Mean Convective Work <ε_C> (eV)
#     (d) Mean Hall Work <ε_H> (eV)
#     (e) Mean Ambipolar Work <ε_A> (eV)
#     (f) Mean Total Work <ε_T> (eV)
#     on black background with shared bwr SymLogNorm(-10^4, 10^4) colorbar
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig4.jl [PAPER_DIR] [OUTPUT_PNG] [RAW_RUN_DIR]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :_load_sampled_trajectories)
    include(joinpath(@__DIR__, "plot_fig3.jl"))
end

function plot_fig4(
        paper_dir::AbstractString = joinpath(pwd(), "outputs", "perlmutter_runs", "mc_400km_without_crustal_cum_005_paper"),
        out_png::AbstractString = joinpath(paper_dir, "fig4.png");
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
            # Exact conversion from DEF [eV / (m^2 s eV sr)] to PSD [s^3 m^-6]: PSD = DEF * (m/e)^2 / (2 E_eV^2)
            factor = (mass / qe)^2 ./ (2.0 .* (ec') .^ 2)
            def .* factor
        end

        w_conv = Matrix{Float64}(f["orbit_w_conv_eV"])
        w_hall = Matrix{Float64}(f["orbit_w_hall_eV"])
        w_grad = Matrix{Float64}(f["orbit_w_gradpe_eV"])
        w_tot = Matrix{Float64}(f["orbit_w_total_eV"])
        (; angles, ec, edges, psd, w_conv, w_hall, w_grad, w_tot, r_orbit)
    end

    paths = _load_sampled_trajectories(paper_dir, raw_run_dir; count = 20_000)

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 14
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 13.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 13.5

    fig = plt.figure(figsize = (12.8, 13.2), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.04, wspace = 0.04, hspace = 0.04)
    gs_main = fig.add_gridspec(2, 1; height_ratios = [1.0, 1.25])
    gs_top = gs_main[1, 1].subgridspec(1, 2)
    gs_bot = gs_main[2, 1].subgridspec(2, 2)

    ax_a = fig.add_subplot(gs_top[1, 1])
    ax_b = fig.add_subplot(gs_top[1, 2])
    ax_c = fig.add_subplot(gs_bot[1, 1])
    ax_d = fig.add_subplot(gs_bot[1, 2])
    ax_e = fig.add_subplot(gs_bot[2, 1])
    ax_f = fig.add_subplot(gs_bot[2, 2])

    # (a) XZ trajectories + red virtual orbit arc
    _draw_blue_trajectories!(ax_a, paths, "XZ")
    bs_mpb(ax_a, "XZ"; draw_bs = true, draw_mpb = false, xmin = -4.0, boundary_color = "black", boundary_ls = "-", boundary_lw = 1.0)
    bs_mpb(ax_a, "XZ"; draw_bs = false, draw_mpb = true, xmin = -4.0, boundary_color = "black", boundary_ls = "-", boundary_lw = 1.0)
    plot_mars(ax_a; radius = 1.0, zorder = 5)

    ang_arc = range(0.0, 240.0; length = 241)
    r_orb = d.r_orbit
    ax_a.plot(r_orb .* cosd.(ang_arc), r_orb .* sind.(ang_arc); color = "#d6192b", lw = 2.1, zorder = 7)
    mark_angles = (0, 60, 120, 180, 240)
    for ang in mark_angles
        xm = r_orb * cosd(ang)
        zm = r_orb * sind(ang)
        ax_a.scatter([xm], [zm]; s = 26, color = "#d6192b", zorder = 8)
        dx, dz, ha_str, va_str = if ang == 0
            (0.08, 0.0, "left", "center")
        elseif ang == 60
            (0.06, 0.12, "left", "bottom")
        elseif ang == 120
            (-0.06, 0.12, "right", "bottom")
        elseif ang == 180
            (-0.10, 0.0, "right", "center")
        else
            (-0.06, -0.14, "center", "top")
        end
        ax_a.text(xm + dx, zm + dz, "$(ang)°"; fontsize = 14.5, color = "black", ha = ha_str, va = va_str, zorder = 9)
    end
    ax_a.text(2.8, -0.8, "Virtual\ntrajectory"; color = "#d6192b", fontsize = 14, ha = "center", va = "center")
    ax_a.set_xlim(-4.0, 4.0)
    ax_a.set_ylim(-3.0, 4.0)
    ax_a.set_aspect("equal")
    ax_a.set_xticks(-4:2:4)
    ax_a.set_yticks(-3:1:4)
    ax_a.tick_params(labelsize = 13.5)
    ax_a.set_xlabel("\$X_{MSE}\$ (\$R_M\$)"; fontsize = 16)
    ax_a.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 16)
    ax_a.text(0.03, 0.95, "a"; transform = ax_a.transAxes, fontsize = 18, fontweight = "bold", va = "top", ha = "left")

    # (b) PSD vs Orbit Angle and Energy
    ax_b.set_facecolor("black")
    psd_plot = copy(permutedims(d.psd))
    psd_plot[psd_plot .<= 0.0] .= NaN
    norm_b = matplotlib.colors.LogNorm(vmin = 1.0e-20, vmax = 1.0e-4)
    im_b = ax_b.pcolormesh(d.angles, d.ec, psd_plot; cmap = "turbo", norm = norm_b, shading = "auto", rasterized = true)
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, 240.0)
    ax_b.set_ylim(1.0e0, 1.0e5)
    ax_b.set_xticks(0:60:240)
    ax_b.tick_params(labelsize = 13.5)
    ax_b.set_xlabel("Orbit Angle (deg)"; fontsize = 16)
    ax_b.set_ylabel("Energy (eV)"; fontsize = 16)
    ax_b.text(0.03, 0.95, "b"; transform = ax_b.transAxes, color = "white", fontsize = 18, fontweight = "bold", va = "top", ha = "left")

    # Annotations on panel b
    ax_b.annotate(
        "Beam", xy = (96.0, 1.4e4), xytext = (122.0, 3.6e4),
        color = "white", fontsize = 14.5, ha = "left", va = "center",
        arrowprops = Dict("arrowstyle" => "->", "color" => "white", "lw" => 1.6),
    )
    ax_b.annotate(
        "Low-energy ions", xy = (182.0, 1.0e1), xytext = (154.0, 1.0e1),
        color = "white", fontsize = 14.5, ha = "right", va = "center",
        arrowprops = Dict("arrowstyle" => "->", "color" => "white", "lw" => 1.6),
    )

    # Top +E / -E hemisphere bar above panel b
    ax_b.plot([0.0, 240.0], [1.04, 1.04]; color = "black", lw = 1.1, transform = ax_b.get_xaxis_transform(), clip_on = false)
    for xv in (0.0, 180.0, 240.0)
        ax_b.plot([xv, xv], [1.01, 1.07]; color = "black", lw = 1.1, transform = ax_b.get_xaxis_transform(), clip_on = false)
    end
    ax_b.text(90.0, 1.08, "+E hemisphere"; transform = ax_b.get_xaxis_transform(), fontsize = 14.5, ha = "center", va = "bottom", clip_on = false)
    ax_b.text(210.0, 1.08, "−E hemisphere"; transform = ax_b.get_xaxis_transform(), fontsize = 14.5, ha = "center", va = "bottom", clip_on = false)

    # (c)-(f) Mean cumulative work 2x2 grid
    norm_w = matplotlib.colors.SymLogNorm(linthresh = 1.0, linscale = 0.8, vmin = -1.0e4, vmax = 1.0e4, base = 10)
    work_panels = (
        (ax_c, d.w_conv, "c", "\$\\langle\\varepsilon_C\\rangle\$", false, true),
        (ax_d, d.w_hall, "d", "\$\\langle\\varepsilon_H\\rangle\$", false, false),
        (ax_e, d.w_grad, "e", "\$\\langle\\varepsilon_A\\rangle\$", true, true),
        (ax_f, d.w_tot, "f", "\$\\langle\\varepsilon_T\\rangle\$", true, false),
    )
    im_w = nothing
    for (ax, mat, tag, sym_lbl, show_x, show_y) in work_panels
        ax.set_facecolor("black")
        im_w = ax.pcolormesh(d.angles, d.ec, permutedims(mat); cmap = "bwr", norm = norm_w, shading = "auto", rasterized = true)
        ax.set_yscale("log")
        ax.set_xlim(0.0, 240.0)
        ax.set_ylim(1.0e0, 1.0e5)
        ax.set_xticks(0:60:240)
        ax.set_yticks([1.0e0, 1.0e2, 1.0e4])
        ax.tick_params(labelsize = 13.5)
        if show_x
            ax.set_xlabel("Orbit Angle (deg)"; fontsize = 16)
        else
            ax.tick_params(labelbottom = false)
        end
        if show_y
            ax.set_ylabel("Energy (eV)"; fontsize = 16)
        else
            ax.tick_params(labelleft = false)
        end
        ax.text(0.03, 0.93, tag; transform = ax.transAxes, color = "white", fontsize = 18, fontweight = "bold", va = "top", ha = "left")
        ax.text(0.96, 0.92, sym_lbl; transform = ax.transAxes, color = "white", fontsize = 18.5, va = "top", ha = "right")
    end

    fig.canvas.draw()
    fig.set_layout_engine("none")

    cbar_b_ax = ax_b.inset_axes([1.035, 0.0, 0.042, 1.0])
    cb_b = fig.colorbar(im_b; cax = cbar_b_ax)
    cb_b.ax.tick_params(labelsize = 13)
    cb_b.set_label("PSD (s\$^3\$ m\$^{-6}\$)"; fontsize = 15)

    pos_d = ax_d.get_position()
    pos_f = ax_f.get_position()
    y_top_rel = (pos_d.y1 - pos_f.y0) / pos_f.height
    cbar_w_ax = ax_f.inset_axes([1.035, 0.0, 0.042, y_top_rel])
    cb_w = fig.colorbar(im_w; cax = cbar_w_ax)
    cb_w.set_ticks([-1.0e4, -1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4])
    cb_w.ax.tick_params(labelsize = 13)
    cb_w.set_label("Mean cumulative work \$\\langle\\varepsilon\\rangle\$ (eV)"; fontsize = 15)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 4 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "outputs", "perlmutter_runs", "mc_400km_without_crustal_cum_005_paper")
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig4.png")
    rdir = length(ARGS) >= 3 ? ARGS[3] : nothing
    plot_fig4(pdir, out; raw_run_dir = rdir)
end

