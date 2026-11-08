# Publication-quality script for Figure 6 of the Mars O2+ test-particle manuscript:
#   4 rows x 2 columns across the magnetotail current sheet (X_MSE = -2 R_M, Z_MSE = +0.75 R_M, Y_MSE ∈ [-1, 1] R_M):
#     (a) YZ projection of O2+ trajectories (blue-ish colormap), Mars disk, and red virtual trajectory line
#     (b) Magnetic field components B_x (red) and |B_t| (black) in nT
#     (c) O2+ PSD (s^3 m^-6) vs Y_MSE and Energy (10^1..10^4 eV) with horizontal inset colorbar
#     (d) Bulk velocity components U_x (black), U_y (blue), U_z (orange), and U_t (green dashed) in km/s
#     (e) Mean Convective Work <ε_C> (eV)
#     (f) Mean Hall Work <ε_H> (eV)
#     (g) Mean Ambipolar Work 5 × <ε_A> (eV)
#     (h) Mean Total Work <ε_T> (eV)
#     with linear bwr [-300, 300] eV colorbar
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig6.jl [PAPER_DIR] [OUTPUT_PNG] [RAW_RUN_DIR]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :_load_sampled_trajectories)
    include(joinpath(@__DIR__, "plot_fig3.jl"))
end

function plot_fig6(
        paper_dir::AbstractString = _default_paper_dir(),
        out_png::AbstractString = joinpath(paper_dir, "fig6.png");
        raw_run_dir = nothing,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    d = jldopen(jld_path, "r") do f
        y_Rm = Vector{Float64}(f["tail_y_Rm"])
        ec = Vector{Float64}(f["energy_centers_eV"])
        edges = Vector{Float64}(f["energy_edges_eV"])
        def = Matrix{Float64}(f["tail_def"])
        u_ms = Matrix{Float64}(f["tail_bulk_vel_ms"])
        dens_m3 = Vector{Float64}(f["tail_density_m3"])
        w_conv = Matrix{Float64}(f["tail_w_conv_eV"])
        w_hall = Matrix{Float64}(f["tail_w_hall_eV"])
        w_grad = Matrix{Float64}(f["tail_w_gradpe_eV"])
        w_tot = Matrix{Float64}(f["tail_w_total_eV"])
        tail_x_Rm = haskey(f, "tail_x_Rm") ? Float64(f["tail_x_Rm"]) : -2.0
        tail_z_Rm = haskey(f, "tail_z_Rm") ? Float64(f["tail_z_Rm"]) : 0.75
        ff = haskey(f, "fields_file") ? String(f["fields_file"]) : "mars_fields_without_crustal.vts"

        mass = MarsTP.TP.SpeciesDict["O2+"].m
        qe = MarsTP.TP.qᵢ
        psd = if haskey(f, "tail_psd_s3_m6")
            Matrix{Float64}(f["tail_psd_s3_m6"])
        else
            factor = (mass / qe)^2 ./ (2.0 .* (ec') .^ 2)
            def .* factor
        end

        b_nT = if haskey(f, "tail_b_nT")
            Matrix{Float64}(f["tail_b_nT"])
        else
            vts_file = isfile(MarsTP.data_path(ff)) ? MarsTP.data_path(ff) : MarsTP.data_path("mars_fields_without_crustal.vts")
            fields = load_mhd_fields(vts_file)
            b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)
            bmat = zeros(Float64, 4, length(y_Rm))
            for (i, yv) in enumerate(y_Rm)
                bv = SVector{3, Float64}(b_itp(SVector{3, Float64}(tail_x_Rm * Rm, yv * Rm, tail_z_Rm * Rm))) * 1.0e9
                bmat[1:3, i] .= bv
                bmat[4, i] = norm(bv)
            end
            bmat
        end

        (; y_Rm, ec, edges, psd, b_nT, u_ms, dens_m3, w_conv, w_hall, w_grad, w_tot, tail_x_Rm, tail_z_Rm)
    end

    paths = _load_sampled_trajectories(paper_dir, raw_run_dir; count = 20_000)

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 18.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 18.5
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.5
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.5
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["xtick.minor.size"] = 4.5
    PyPlot.matplotlib.rcParams["ytick.minor.size"] = 4.5
    PyPlot.matplotlib.rcParams["xtick.minor.width"] = 1.1
    PyPlot.matplotlib.rcParams["ytick.minor.width"] = 1.1

    fig, axes = plt.subplots(4, 2; figsize = (13.4, 12.6), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.03, h_pad = 0.03, wspace = 0.03, hspace = 0.03)

    ax_a = axes[1, 1]
    ax_b = axes[1, 2]
    ax_c = axes[2, 1]
    ax_d = axes[2, 2]
    ax_e = axes[3, 1]
    ax_f = axes[3, 2]
    ax_g = axes[4, 1]
    ax_h = axes[4, 2]

    # (a) YZ trajectories + Mars disk + Virtual trajectory line
    _draw_blue_trajectories!(ax_a, paths, "YZ")
    plot_mars(ax_a; radius = 1.0, zorder = 5)
    ax_a.plot([-1.0, 1.0], [d.tail_z_Rm, d.tail_z_Rm]; color = "#d6192b", lw = 2.2, zorder = 8)
    ax_a.text(
        0.0, d.tail_z_Rm + 0.14, "Virtual trajectory";
        color = "#d6192b", fontsize = 16.5, ha = "center", va = "bottom",
        bbox = Dict("boxstyle" => "square,pad=0.12", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.78),
        zorder = 9,
    )
    ax_a.set_xlim(-2.0, 2.0)
    ax_a.set_ylim(0.0, 2.0)
    ax_a.set_aspect("equal")
    ax_a.set_xticks(-2:1:2)
    ax_a.set_yticks(0.0:0.5:2.0)
    ax_a.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
    ax_a.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 19)
    ax_a.set_title("\$X_{MSE} = −2\\,R_M\$"; fontsize = 20, pad = 7)
    ax_a.text(0.02, 0.93, "a"; transform = ax_a.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left", zorder = 10)

    # (b) Magnetic field (nT)
    ax_b.axhline(0.0; color = "0.65", lw = 0.8)
    ax_b.plot(d.y_Rm, d.b_nT[1, :]; color = "red", lw = 1.9, label = "\$B_x\$")
    ax_b.plot(d.y_Rm, d.b_nT[4, :]; color = "black", lw = 1.9, label = "\$|B_t|\$")
    ax_b.set_xlim(-1.0, 1.0)
    ax_b.set_ylim(-15.0, 17.0)
    ax_b.set_xticks(-1.0:0.5:1.0)
    ax_b.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
    ax_b.set_ylabel("Magnetic field (nT)"; fontsize = 19)
    ax_b.legend(loc = "center right", frameon = false, fontsize = 17)
    ax_b.text(0.03, 0.93, "b"; transform = ax_b.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left")

    # (c) PSD vs Y_MSE and Energy with horizontal inset colorbar
    psd_plot = copy(permutedims(d.psd))
    psd_plot[psd_plot .<= 0.0] .= NaN
    norm_c = matplotlib.colors.LogNorm(vmin = 1.0e-12, vmax = 1.0e-7)
    im_c = ax_c.pcolormesh(d.y_Rm, d.ec, psd_plot; cmap = "turbo", norm = norm_c, shading = "auto", rasterized = true)
    ax_c.set_yscale("log")
    ax_c.set_xlim(-1.0, 1.0)
    ax_c.set_ylim(1.0e1, 1.0e4)
    ax_c.set_xticks(-1.0:0.5:1.0)
    ax_c.tick_params(axis = "both", which = "major", labelbottom = false, labelsize = 18.5, length = 7.5, width = 1.4)
    ax_c.tick_params(axis = "y", which = "minor", length = 4.5, width = 1.1)
    ax_c.set_ylabel("Energy (eV)"; fontsize = 19)
    ax_c.text(0.03, 0.93, "c"; transform = ax_c.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left")

    cax_c = ax_c.inset_axes([0.20, 0.81, 0.75, 0.068])
    cb_c = fig.colorbar(im_c; cax = cax_c, orientation = "horizontal")
    cb_c.ax.xaxis.set_ticks_position("bottom")
    cb_c.ax.tick_params(which = "major", labelsize = 14.5, length = 4.5, width = 1.1, pad = 1.5)
    cb_c.ax.tick_params(which = "minor", length = 2.5, width = 0.9)
    cb_c.ax.set_title("PSD (s\$^3\$ m\$^{-6}\$)"; fontsize = 15.5, pad = 3.0)

    # (d) Bulk velocity (km/s)
    ax_d.axhline(0.0; color = "0.65", lw = 0.8)
    valid_u = d.dens_m3 .> 0.0
    u_kms = fill(NaN, 4, length(d.y_Rm))
    for i in eachindex(d.y_Rm)
        if valid_u[i]
            u_kms[1:3, i] .= d.u_ms[:, i] ./ 1.0e3
            u_kms[4, i] = norm(@view u_kms[1:3, i])
        end
    end
    ax_d.plot(d.y_Rm, u_kms[1, :]; color = "black", lw = 1.9, ls = "-", label = "\$U_x\$")
    ax_d.plot(d.y_Rm, u_kms[2, :]; color = "#0072B2", lw = 1.9, ls = "-", label = "\$U_y\$")
    ax_d.plot(d.y_Rm, u_kms[3, :]; color = "#D55E00", lw = 1.9, ls = "-", label = "\$U_z\$")
    ax_d.plot(d.y_Rm, u_kms[4, :]; color = "#009E73", lw = 1.9, ls = "--", label = "\$U_t\$")
    ax_d.set_xlim(-1.0, 1.0)
    ax_d.set_ylim(-42.0, 52.0)
    ax_d.set_xticks(-1.0:0.5:1.0)
    ax_d.tick_params(axis = "both", which = "major", labelbottom = false, labelsize = 18.5, length = 7.5, width = 1.4)
    ax_d.set_ylabel("Bulk velocity (km s\$^{-1}\$)"; fontsize = 19)
    ax_d.legend(loc = "center left", bbox_to_anchor = (1.01, 0.5), frameon = false, fontsize = 17)
    ax_d.text(0.03, 0.93, "d"; transform = ax_d.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left")

    # (e)-(h) Mean cumulative work 2x2 grid on linear [-300, 300] eV bwr scale
    work_panels = (
        (ax_e, d.w_conv, 1.0, "e", "\$\\langle\\varepsilon_C\\rangle\$", false, true),
        (ax_f, d.w_hall, 1.0, "f", "\$\\langle\\varepsilon_H\\rangle\$", false, false),
        (ax_g, d.w_grad, 5.0, "g", "\$5 \\times \\langle\\varepsilon_A\\rangle\$", true, true),
        (ax_h, d.w_tot, 1.0, "h", "\$\\langle\\varepsilon_T\\rangle\$", true, false),
    )
    im_w = nothing
    for (ax, mat, scale, tag, sym_lbl, show_x, show_y) in work_panels
        im_w = ax.pcolormesh(d.y_Rm, d.ec, permutedims(mat .* scale); cmap = "bwr", vmin = -300.0, vmax = 300.0, shading = "auto", rasterized = true)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(1.0e1, 1.0e4)
        ax.set_xticks(-1.0:0.5:1.0)
        ax.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
        ax.tick_params(axis = "y", which = "minor", length = 4.5, width = 1.1)
        if show_x
            ax.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 19.5)
        else
            ax.tick_params(labelbottom = false)
        end
        if show_y
            ax.set_ylabel("Energy (eV)"; fontsize = 19)
        else
            ax.tick_params(labelleft = false)
        end
        ax.text(0.03, 0.93, tag; transform = ax.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left")
        ax.text(0.96, 0.92, sym_lbl; transform = ax.transAxes, fontsize = 21, va = "top", ha = "right")
    end

    fig.canvas.draw()
    fig.set_layout_engine("none")
    pos_f = ax_f.get_position()
    pos_h = ax_h.get_position()
    y_top_rel = (pos_f.y1 - pos_h.y0) / pos_h.height
    cbar_w_ax = ax_h.inset_axes([1.035, 0.0, 0.042, y_top_rel])
    cb_w = fig.colorbar(im_w; cax = cbar_w_ax)
    cb_w.set_ticks(-300:100:300)
    cb_w.ax.tick_params(which = "major", labelsize = 17, length = 6.0, width = 1.3)
    cb_w.set_label("Mean cumulative work (eV)"; fontsize = 19)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 6 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : _default_paper_dir()
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig6.png")
    rdir = length(ARGS) >= 3 ? ARGS[3] : nothing
    plot_fig6(pdir, out; raw_run_dir = rdir)
end

