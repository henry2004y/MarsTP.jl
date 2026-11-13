# Publication-quality script for Figure 7 of the Mars O2+ test-particle manuscript:
#   4 rows x 2 columns across the magnetotail current sheet (X_MSE = -2 R_M, Z_MSE = +0.75 R_M, Y_MSE ∈ [-1, 1] R_M):
#     (a) YZ projection of O2+ trajectories (blue-ish colormap), Mars disk, and red virtual trajectory line
#     (b) Magnetic field components B_x (red) and |B_t| (black) in nT
#     (c) O2+ PSD (s^3 m^-6) vs Y_MSE and Energy (10^1..10^4 eV) with horizontal inset colorbar on black background
#     (d) Bulk velocity components U_x (black), U_y (blue), U_z (orange), and U_t (green dashed) in km/s
#     (e) Mean Convective Work <ε_C> (eV)
#     (f) Mean Hall Work <ε_H> (eV)
#     (g) Mean Ambipolar Work 5 × <ε_A> (eV)
#     (h) Mean Total Work <ε_T> (eV)
#     on black background with RdBu_r [-300, 300] eV colorbar
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig7.jl [PAPER_DIR] [OUTPUT_PNG] [RAW_RUN_DIR]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :_load_sampled_trajectories)
    include(joinpath(@__DIR__, "plot_fig3.jl"))
end

function plot_fig7(
        paper_dir::AbstractString = _default_paper_dir(),
        out_png::AbstractString = joinpath(paper_dir, "fig7.png");
        raw_run_dir = nothing,
        z_Rm = nothing,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    d = jldopen(jld_path, "r") do f
        y_Rm = Vector{Float64}(f["tail_y_Rm"])
        ec = Vector{Float64}(f["energy_centers_eV"])
        edges = Vector{Float64}(f["energy_edges_eV"])
        tail_x_Rm = haskey(f, "tail_x_Rm") ? Float64(f["tail_x_Rm"]) : -2.0
        ff = haskey(f, "fields_file") ? String(f["fields_file"]) : "mars_fields_without_crustal.vts"
        mass = MarsTP.TP.SpeciesDict["O2+"].m
        qe = MarsTP.TP.qᵢ

        if z_Rm !== nothing && haskey(f, "tail_z_scan_Rm")
            z_list = Vector{Float64}(f["tail_z_scan_Rm"])
            iz = argmin(abs.(z_list .- Float64(z_Rm)))
            tail_z_Rm = z_list[iz]
            if haskey(f, "tail_z_scan_y_Rm")
                y_Rm = Vector{Float64}(f["tail_z_scan_y_Rm"][iz])
            end
            def = Matrix{Float64}(f["tail_z_scan_def"][iz])
            psd = Matrix{Float64}(f["tail_z_scan_psd_s3_m6"][iz])
            b_nT = Matrix{Float64}(f["tail_z_scan_b_nT"][iz])
            u_ms = Matrix{Float64}(f["tail_z_scan_bulk_vel_ms"][iz])
            dens_m3 = Vector{Float64}(f["tail_z_scan_density_m3"][iz])
            w_conv = Matrix{Float64}(f["tail_z_scan_w_conv_eV"][iz])
            w_hall = Matrix{Float64}(f["tail_z_scan_w_hall_eV"][iz])
            w_grad = Matrix{Float64}(f["tail_z_scan_w_gradpe_eV"][iz])
            w_tot = Matrix{Float64}(f["tail_z_scan_w_total_eV"][iz])
            occ_mask = psd .> 0.0
        else
            def = Matrix{Float64}(f["tail_def"])
            u_ms = Matrix{Float64}(f["tail_bulk_vel_ms"])
            dens_m3 = Vector{Float64}(f["tail_density_m3"])
            w_conv = Matrix{Float64}(f["tail_w_conv_eV"])
            w_hall = Matrix{Float64}(f["tail_w_hall_eV"])
            w_grad = Matrix{Float64}(f["tail_w_gradpe_eV"])
            w_tot = Matrix{Float64}(f["tail_w_total_eV"])
            tail_z_Rm = haskey(f, "tail_z_Rm") ? Float64(f["tail_z_Rm"]) : 0.75

            psd = if haskey(f, "tail_psd_s3_m6")
                Matrix{Float64}(f["tail_psd_s3_m6"])
            else
                factor = (mass / qe)^2 ./ (2.0 .* (ec') .^ 2)
                def .* factor
            end

            occ_mask = if haskey(f, "raw_tail20")
                Matrix{Float64}(f["raw_tail20"]["weight_tau"]) .> 0.0
            else
                psd .> 0.0
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
        end

        (; y_Rm, ec, edges, psd, occ_mask, b_nT, u_ms, dens_m3, w_conv, w_hall, w_grad, w_tot, tail_x_Rm, tail_z_Rm)
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

    y_min_plot = first(d.y_Rm)
    y_max_plot = last(d.y_Rm)
    y_mid_plot = 0.5 * (y_min_plot + y_max_plot)
    y_ticks_plot = round.(collect(range(y_min_plot, y_max_plot; length = 5)); digits = 2)

    # (a) YZ trajectories + Mars disk + Virtual trajectory line
    _draw_blue_trajectories!(ax_a, paths, "YZ")
    plot_mars(ax_a; radius = 1.0, zorder = 5)
    ax_a.plot([y_min_plot, y_max_plot], [d.tail_z_Rm, d.tail_z_Rm]; color = "#d6192b", lw = 2.2, zorder = 8)
    ax_a.text(
        y_mid_plot, d.tail_z_Rm + 0.14, "Virtual trajectory";
        color = "#d6192b", fontsize = 16.5, ha = "center", va = "bottom",
        bbox = Dict("boxstyle" => "square,pad=0.12", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.78),
        zorder = 9,
    )
    ax_a.set_xlim(-2.0, 2.0)
    if d.tail_z_Rm < 0.2
        ax_a.set_ylim(-1.0, 1.5)
        ax_a.set_yticks(-1.0:0.5:1.5)
    else
        ax_a.set_ylim(0.0, 2.0)
        ax_a.set_yticks(0.0:0.5:2.0)
    end
    ax_a.set_aspect("equal")
    ax_a.set_xticks(-2:1:2)
    ax_a.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
    ax_a.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 19)
    ax_a.set_title("\$X_{MSE} = −2\\,R_M\$"; fontsize = 20, pad = 7)
    ax_a.text(0.02, 0.93, "a"; transform = ax_a.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left", zorder = 10)

    # (b) Magnetic field (nT)
    ax_b.axhline(0.0; color = "0.65", lw = 0.8)
    ax_b.plot(d.y_Rm, d.b_nT[1, :]; color = "red", lw = 1.9, label = "\$B_x\$")
    ax_b.plot(d.y_Rm, d.b_nT[4, :]; color = "black", lw = 1.9, label = "\$|B_t|\$")
    ax_b.set_xlim(y_min_plot, y_max_plot)
    ax_b.set_ylim(-15.0, 17.0)
    ax_b.set_xticks(y_ticks_plot)
    ax_b.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
    ax_b.set_ylabel("Magnetic field (nT)"; fontsize = 19)
    ax_b.legend(loc = "center right", frameon = false, fontsize = 17)
    ax_b.text(0.03, 0.93, "b"; transform = ax_b.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left")

    # (c) PSD vs Y_MSE and Energy on black background with horizontal inset colorbar
    ax_c.set_facecolor("black")
    cmap_c = matplotlib.cm.get_cmap("turbo").copy()
    cmap_c.set_bad(color = "black")
    psd_plot = copy(permutedims(d.psd))
    psd_plot[psd_plot .<= 0.0] .= NaN
    norm_c = matplotlib.colors.LogNorm(vmin = 1.0e-12, vmax = 1.0e-7)
    im_c = ax_c.pcolormesh(d.y_Rm, d.ec, psd_plot; cmap = cmap_c, norm = norm_c, shading = "auto", rasterized = true)
    ax_c.set_yscale("log")
    ax_c.set_xlim(y_min_plot, y_max_plot)
    ax_c.set_ylim(1.0e1, 1.0e4)
    ax_c.set_xticks(y_ticks_plot)
    ax_c.tick_params(axis = "both", which = "major", labelbottom = false, labelsize = 18.5, length = 7.5, width = 1.4)
    ax_c.tick_params(axis = "y", which = "minor", length = 4.5, width = 1.1)
    ax_c.set_ylabel("Energy (eV)"; fontsize = 19)
    ax_c.text(0.03, 0.93, "c"; transform = ax_c.transAxes, color = "white", fontsize = 21, fontweight = "bold", va = "top", ha = "left", zorder = 10)

    cax_c = ax_c.inset_axes([0.20, 0.80, 0.75, 0.068])
    cb_c = fig.colorbar(im_c; cax = cax_c, orientation = "horizontal")
    cb_c.ax.xaxis.set_ticks_position("bottom")
    cb_c.ax.tick_params(which = "major", labelsize = 14.5, colors = "white", length = 4.5, width = 1.1, pad = 2.0)
    cb_c.ax.tick_params(which = "minor", colors = "white", length = 2.5, width = 0.9)
    cb_c.outline.set_edgecolor("white")
    cb_c.ax.set_title("PSD (s\$^3\$ m\$^{-6}\$)"; fontsize = 15.5, color = "white", pad = 4.0)

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
    ax_d.set_xlim(y_min_plot, y_max_plot)
    ax_d.set_ylim(-42.0, 52.0)
    ax_d.set_xticks(y_ticks_plot)
    ax_d.tick_params(axis = "both", which = "major", labelbottom = false, labelsize = 18.5, length = 7.5, width = 1.4)
    ax_d.set_ylabel("Bulk velocity (km s\$^{-1}\$)"; fontsize = 19)
    ax_d.legend(loc = "center left", bbox_to_anchor = (1.01, 0.5), frameon = false, fontsize = 17)
    ax_d.text(0.03, 0.93, "d"; transform = ax_d.transAxes, fontsize = 21, fontweight = "bold", va = "top", ha = "left")

    # (e)-(h) Mean cumulative work 2x2 grid on black background with RdBu_r [-300, 300] eV scale
    cmap_w = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap_w.set_bad(color = "black")
    occ_perm = permutedims(d.occ_mask)
    work_panels = (
        (ax_e, d.w_conv, 1.0, "e", "\$\\langle\\varepsilon_C\\rangle\$", false, true),
        (ax_f, d.w_hall, 1.0, "f", "\$\\langle\\varepsilon_H\\rangle\$", false, false),
        (ax_g, d.w_grad, 5.0, "g", "\$5 \\times \\langle\\varepsilon_A\\rangle\$", true, true),
        (ax_h, d.w_tot, 1.0, "h", "\$\\langle\\varepsilon_T\\rangle\$", true, false),
    )
    im_w = nothing
    for (ax, mat, scale, tag, sym_lbl, show_x, show_y) in work_panels
        ax.set_facecolor("black")
        mat_plot = copy(permutedims(mat .* scale))
        mat_plot[.!occ_perm .| .!isfinite.(mat_plot)] .= NaN
        im_w = ax.pcolormesh(d.y_Rm, d.ec, mat_plot; cmap = cmap_w, vmin = -300.0, vmax = 300.0, shading = "auto", rasterized = true)
        ax.set_yscale("log")
        ax.set_xlim(y_min_plot, y_max_plot)
        ax.set_ylim(1.0e1, 1.0e4)
        ax.set_xticks(y_ticks_plot)
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
        ax.text(0.03, 0.93, tag; transform = ax.transAxes, color = "white", fontsize = 21, fontweight = "bold", va = "top", ha = "left", zorder = 10)
        ax.text(0.96, 0.92, sym_lbl; transform = ax.transAxes, color = "white", fontsize = 21, va = "top", ha = "right", zorder = 10)
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
    println("Saved Figure 7 to ", abspath(out_png))
    return out_png
end

function plot_fig7_z_scan(
        paper_dir::AbstractString = _default_paper_dir(),
        out_png::AbstractString = joinpath(paper_dir, "fig7_z_scan.png");
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    s = jldopen(jld_path, "r") do f
        haskey(f, "tail_z_scan_Rm") || error("tail_z_scan_Rm not found in $jld_path")
        y_Rm_default = Vector{Float64}(f["tail_y_Rm"])
        ec = Vector{Float64}(f["energy_centers_eV"])
        z_list = Vector{Float64}(f["tail_z_scan_Rm"])
        y_list = haskey(f, "tail_z_scan_y_Rm") ? [Vector{Float64}(v) for v in f["tail_z_scan_y_Rm"]] : [copy(y_Rm_default) for _ in z_list]
        y_cs_list = haskey(f, "tail_z_scan_y_cs_Rm") ? Vector{Float64}(f["tail_z_scan_y_cs_Rm"]) : [v[cld(length(v), 2)] for v in y_list]
        b_list = [Matrix{Float64}(m) for m in f["tail_z_scan_b_nT"]]
        psd_list = [Matrix{Float64}(m) for m in f["tail_z_scan_psd_s3_m6"]]
        dens_list = [Vector{Float64}(v) for v in f["tail_z_scan_density_m3"]]
        ff = haskey(f, "fields_file") ? String(f["fields_file"]) : "mars_fields_with_crustal.vts"
        (; y_list, y_cs_list, ec, z_list, b_list, psd_list, dens_list, ff)
    end

    vts_file = isfile(MarsTP.data_path(s.ff)) ? MarsTP.data_path(s.ff) : MarsTP.data_path("mars_fields_with_crustal.vts")
    fields = load_mhd_fields(vts_file)
    b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)

    y_grid = collect(range(-1.5, 1.5; length = 121))
    z_grid = collect(range(-1.0, 1.5; length = 101))
    bx_map = zeros(length(z_grid), length(y_grid))
    for (iz, zv) in enumerate(z_grid), (iy, yv) in enumerate(y_grid)
        bv = b_itp(SVector(-2.0 * Rm, yv * Rm, zv * Rm)) .* 1.0e9
        bx_map[iz, iy] = bv[1]
    end

    ny = length(first(s.y_list))
    nz = length(s.z_list)
    dens_map = zeros(nz, ny)
    y_mesh = zeros(nz, ny)
    z_mesh = zeros(nz, ny)
    for iz in 1:nz
        dens_map[iz, :] .= s.dens_list[iz] .* 1.0e-6
        y_mesh[iz, :] .= s.y_list[iz]
        z_mesh[iz, :] .= s.z_list[iz]
    end
    dens_map[dens_map .<= 0.0] .= NaN

    nrows_grid = cld(nz, 2)
    fig = plt.figure(figsize = (19.0, 2.5 * nrows_grid + 3.2), layout = "constrained")
    sfigs = fig.subfigures(2, 1; height_ratios = [1.25, 0.85 * nrows_grid])
    axes_top = sfigs[1].subplots(1, 2)
    axes_grid = sfigs[2].subplots(nrows_grid, 4)

    case_lbl = occursin("without_crustal", lowercase(s.ff)) ? "WITHOUT CRUSTAL" : "WITH CRUSTAL"

    ax_map_b = axes_top[1]
    im_bx = ax_map_b.pcolormesh(y_grid, z_grid, bx_map; cmap = "RdBu_r", vmin = -16.0, vmax = 16.0, shading = "auto", rasterized = true)
    ax_map_b.contour(y_grid, z_grid, bx_map; levels = [0.0], colors = ["black"], linewidths = [2.2])
    for (iz, z0) in enumerate(s.z_list)
        yv = s.y_list[iz]
        ax_map_b.plot([first(yv), last(yv)], [z0, z0]; ls = "--", lw = 1.0, color = "gold", alpha = 0.85)
    end
    ax_map_b.set_xlim(-1.5, 1.5)
    ax_map_b.set_ylim(-1.0, 1.5)
    ax_map_b.set_aspect("equal")
    ax_map_b.set_title("$case_lbl: \$B_x(Y, Z)\$ & \$B_x=0\$ at \$X = -2\\,R_M\$"; fontsize = 13, fontweight = "bold")
    ax_map_b.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 11)
    ax_map_b.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 11)
    sfigs[1].colorbar(im_bx; ax = ax_map_b, label = "\$B_x\$ (nT)")

    ax_map_n = axes_top[2]
    ax_map_n.set_facecolor("black")
    if any(!isnan, dens_map)
        vmax_n = maximum(filter(!isnan, vec(dens_map)))
        vmin_n = max(vmax_n * 1.0e-3, 1.0e-3)
        im_n = ax_map_n.pcolormesh(y_mesh, z_mesh, dens_map; cmap = "turbo", norm = matplotlib.colors.LogNorm(vmin = vmin_n, vmax = vmax_n), shading = "auto", rasterized = true)
        sfigs[1].colorbar(im_n; ax = ax_map_n, label = "\$n_{\\mathrm{O}_2^+}\$ (cm\$^{-3}\$)")
    end
    ax_map_n.contour(y_grid, z_grid, bx_map; levels = [0.0], colors = ["white"], linewidths = [2.2])
    for (iz, z0) in enumerate(s.z_list)
        yv = s.y_list[iz]
        ax_map_n.plot([first(yv), last(yv)], [z0, z0]; ls = "--", lw = 1.0, color = "white", alpha = 0.75)
    end
    ax_map_n.set_xlim(minimum(y_mesh), maximum(y_mesh))
    ax_map_n.set_ylim(minimum(s.z_list), maximum(s.z_list))
    ax_map_n.set_aspect("equal")
    ax_map_n.set_title("$case_lbl: Production O\$_2^+\$ Number Density \$n(Y, Z)\$ at \$X = -2\\,R_M\$"; fontsize = 13, fontweight = "bold")
    ax_map_n.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 11)
    ax_map_n.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 11)

    norm_c = matplotlib.colors.LogNorm(vmin = 1.0e-12, vmax = 1.0e-7)
    for (idx, z0) in enumerate(s.z_list)
        row = div(idx - 1, 2) + 1
        col_pair = mod(idx - 1, 2)
        c_b = 2 * col_pair + 1
        c_s = 2 * col_pair + 2

        ax_b = axes_grid[row, c_b]
        ax_s = axes_grid[row, c_s]

        yv = s.y_list[idx]
        y_cs = s.y_cs_list[idx]
        yticks_v = round.(collect(range(first(yv), last(yv); length = 5)); digits = 2)
        bmat = s.b_list[idx]
        ax_b.axhline(0.0; color = "0.65", lw = 0.8)
        ax_b.axvline(y_cs; color = "0.45", ls = "--", lw = 1.1)
        ax_b.plot(yv, bmat[1, :]; color = "red", lw = 1.6, label = "\$B_x\$")
        ax_b.plot(yv, bmat[4, :]; color = "black", lw = 1.6, label = "\$|B_t|\$")
        ax_b.set_xlim(first(yv), last(yv))
        ax_b.set_xticks(yticks_v)
        ax_b.set_ylim(-15.0, 17.0)
        ax_b.set_ylabel("B (nT)"; fontsize = 9.5)
        ax_b.tick_params(labelsize = 9)
        ax_b.set_title(@sprintf("\$Z = %+.2f\\,R_M\$ (\$Y_{CS}=%+.2f\$)", z0, y_cs); fontsize = 10.5, fontweight = "bold")
        if idx == 1
            ax_b.legend(loc = "upper right", frameon = false, fontsize = 8.5)
        end
        if row == nrows_grid
            ax_b.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 10)
        end

        psd_plot = copy(permutedims(s.psd_list[idx]))
        psd_plot[psd_plot .<= 0.0] .= NaN
        ax_s.set_facecolor("black")
        ax_s.pcolormesh(yv, s.ec, psd_plot; cmap = "turbo", norm = norm_c, shading = "auto", rasterized = true)
        ax_s.axvline(y_cs; color = "white", ls = "--", lw = 1.1, alpha = 0.85)
        ax_s.set_yscale("log")
        ax_s.set_xlim(first(yv), last(yv))
        ax_s.set_xticks(yticks_v)
        ax_s.set_ylim(1.0e1, 1.0e4)
        ax_s.set_ylabel("Energy (eV)"; fontsize = 9.5)
        ax_s.tick_params(labelsize = 9)
        ax_s.set_title(@sprintf("\$Z = %+.2f\\,R_M\$: PSD (s\$^3\$ m\$^{-6}\$)", z0); fontsize = 10.5, fontweight = "bold")
        if row == nrows_grid
            ax_s.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 10)
        end
    end

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 220, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 7 Z-scan to ", abspath(out_png))
    return out_png
end

const plot_fig6_z_scan = plot_fig7_z_scan

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : _default_paper_dir()
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig7.png")
    rdir = length(ARGS) >= 3 && !isempty(ARGS[3]) ? ARGS[3] : nothing
    zval = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : nothing
    plot_fig7(pdir, out; raw_run_dir = rdir, z_Rm = zval)
end
