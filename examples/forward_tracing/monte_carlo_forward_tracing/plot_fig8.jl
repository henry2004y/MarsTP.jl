# Publication-quality script for Figure 8 of the Mars O2+ test-particle manuscript:
#   4-panel Crustal-Field Sensitivity Summary ("Robustness vs. Tail Warping"):
#     (a) Plume & Wake O2+ PSD (s^3 m^-6) along circular r = 2.5 Rm XZ orbit WITH crustal fields
#     (b) 1D energy-integrated mean cumulative work vs Orbit Angle WITH crustal fields
#         (plus dotted baseline without-crustal total work curve and global 1.0 Rm work-share comparison)
#     (c) Warped magnetotail current sheet & O2+ number density n(Y, Z) at X = -2.0 Rm WITH crustal fields,
#         overlaid with the MHD B_x(Y, Z) = 0 contour and Z = +0.75 Rm transect (Y_CS = +0.36 Rm)
#     (d) Cross-tail Hall work <ε_H>(Y, E) at X = -2.0 Rm, Z = +0.75 Rm WITH crustal fields,
#         highlighting shifted B_x = 0 crossing (Y_CS = +0.36 Rm) vs unmagnetized symmetry plane (Y = 0)
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig8.jl [WITH_CRUSTAL_DIR] [OUTPUT_PNG] [WITHOUT_CRUSTAL_DIR] [VTS_WITH_CRUSTAL]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

function _compute_orbit_work_1d(jld_path::AbstractString)
    return jldopen(jld_path, "r") do f
        angles = Vector{Float64}(f["orbit_angles_deg"])
        ec = Vector{Float64}(f["energy_centers_eV"])
        psd = Matrix{Float64}(f["orbit_psd_s3_m6"])
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
        (; angles, ec, psd, wc_1d, wh_1d, wa_1d, wt_1d)
    end
end

function _shell_1Rm_shares(alt_jld::AbstractString)
    isfile(alt_jld) || return nothing
    return jldopen(alt_jld, "r") do f
        hw = f["hist_w_tau"]
        hc = f["hist_wc_tau"]
        hh = f["hist_wh_tau"]
        ha = f["hist_wa_tau"]
        w = sum(@view hw[4, 2:end, :, :])
        wc = sum(@view hc[4, 2:end, :, :]) / w
        wh = sum(@view hh[4, 2:end, :, :]) / w
        wa = sum(@view ha[4, 2:end, :, :]) / w
        wt = wc + wh + wa
        (wc / wt * 100.0, wh / wt * 100.0, wa / wt * 100.0)
    end
end

function plot_fig8(
        with_crustal_dir::AbstractString = joinpath(pwd(), "outputs", "paper_figures_with_crustal"),
        out_png::AbstractString = joinpath(with_crustal_dir, "fig8.png");
        without_crustal_dir::AbstractString = joinpath(pwd(), "outputs", "paper_figures_without_crustal"),
        vts_with_crustal::AbstractString = MarsTP.data_path("mars_fields_with_crustal.vts"),
    )
    wc_jld = joinpath(with_crustal_dir, "paper_analysis.jld2")
    nc_jld = joinpath(without_crustal_dir, "paper_analysis.jld2")
    isfile(wc_jld) || error("Missing with-crustal paper_analysis.jld2: $wc_jld")
    isfile(nc_jld) || error("Missing without-crustal paper_analysis.jld2: $nc_jld")

    orb_wc = _compute_orbit_work_1d(wc_jld)
    orb_nc = _compute_orbit_work_1d(nc_jld)
    sh_wc = _shell_1Rm_shares(joinpath(with_crustal_dir, "altitude_energy_maps_025_050_075_100Rm.jld2"))
    sh_nc = _shell_1Rm_shares(joinpath(without_crustal_dir, "altitude_energy_maps_025_050_075_100Rm.jld2"))

    tail_z = jldopen(wc_jld, "r") do f
        haskey(f, "tail_z_scan_Rm") || error("tail_z_scan_Rm missing in $wc_jld")
        z_list = Vector{Float64}(f["tail_z_scan_Rm"])
        y_list = [Vector{Float64}(v) for v in f["tail_z_scan_y_Rm"]]
        y_cs_list = haskey(f, "tail_z_scan_y_cs_Rm") ? Vector{Float64}(f["tail_z_scan_y_cs_Rm"]) : [v[cld(length(v), 2)] for v in y_list]
        dens_list = [Vector{Float64}(v) for v in f["tail_z_scan_density_m3"]]
        iz075 = argmin(abs.(z_list .- 0.75))
        y_075 = y_list[iz075]
        ycs_075 = y_cs_list[iz075]
        wh_075 = Matrix{Float64}(f["tail_z_scan_w_hall_eV"][iz075])
        psd_075 = Matrix{Float64}(f["tail_z_scan_psd_s3_m6"][iz075])
        bx_075 = Matrix{Float64}(f["tail_z_scan_b_nT"][iz075])[1, :]
        ec_z = Vector{Float64}(f["energy_centers_eV"])
        (; z_list, y_list, y_cs_list, dens_list, iz075, y_075, ycs_075, wh_075, psd_075, bx_075, ec_z)
    end

    # Evaluate B_x(Y, Z) at X = -2.0 Rm for the B_x = 0 current-sheet contour
    fields = load_mhd_fields(vts_with_crustal)
    b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)
    y_grid_b = collect(range(-1.45, 1.45; length = 117))
    z_grid_b = collect(range(-0.7, 0.9; length = 81))
    bx_yz = zeros(Float64, length(z_grid_b), length(y_grid_b))
    for (iz, zv) in enumerate(z_grid_b), (iy, yv) in enumerate(y_grid_b)
        bv = SVector{3, Float64}(b_itp(SVector(-2.0 * Rm, yv * Rm, zv * Rm))) .* 1.0e9
        bx_yz[iz, iy] = bv[1]
    end

    # Interpolate Z-scan density onto a clean rectangular (Y, Z) mesh so warped corridor edges are crisp
    nz = length(tail_z.z_list)
    y_rect = collect(range(-1.35, 1.45; length = 113))
    dens_rect = fill(NaN, nz, length(y_rect))
    for iz in 1:nz
        yv = tail_z.y_list[iz]
        dv = tail_z.dens_list[iz] .* 1.0e-6 # cm^-3
        for (iy, yq) in enumerate(y_rect)
            if first(yv) <= yq <= last(yv)
                # Linear interpolation inside the local cross-tail scan strip [Y_CS - 1, Y_CS + 1]
                k = searchsortedlast(yv, yq)
                k = clamp(k, 1, length(yv) - 1)
                t = (yq - yv[k]) / max(yv[k + 1] - yv[k], 1.0e-12)
                dval = (1.0 - t) * dv[k] + t * dv[k + 1]
                dens_rect[iz, iy] = dval > 0.0 ? dval : NaN
            end
        end
    end

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4

    fig, axes = plt.subplots(2, 2; figsize = (15.4, 11.4), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.04, wspace = 0.06, hspace = 0.06)

    ax_a = axes[1, 1]
    ax_b = axes[1, 2]
    ax_c = axes[2, 1]
    ax_d = axes[2, 2]

    # (a) 2.5 Rm XZ Orbit PSD with crustal fields
    ax_a.set_facecolor("black")
    cmap_a = matplotlib.cm.get_cmap("turbo").copy()
    cmap_a.set_bad(color = "black")
    psd_a = copy(permutedims(orb_wc.psd))
    psd_a[psd_a .<= 0.0] .= NaN
    im_a = ax_a.pcolormesh(orb_wc.angles, orb_wc.ec, psd_a; cmap = cmap_a, norm = matplotlib.colors.LogNorm(vmin = 1.0e-20, vmax = 1.0e-4), shading = "auto", rasterized = true)
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.0, 240.0)
    ax_a.set_ylim(1.0e0, 1.0e5)
    ax_a.set_xticks(0:60:240)
    ax_a.set_xlabel("Orbit Angle (deg)"; fontsize = 19)
    ax_a.set_ylabel("Energy (eV)"; fontsize = 19)
    ax_a.set_title("With crustal fields: \$r = 2.5\\,R_M\$ XZ orbit PSD"; fontsize = 19, pad = 8)
    ax_a.text(0.03, 0.94, "(a)"; transform = ax_a.transAxes, color = "white", fontsize = 20.5, fontweight = "bold", va = "top", ha = "left", zorder = 10)
    ax_a.axvline(180.0; color = "white", ls = "--", lw = 1.2, alpha = 0.75, zorder = 7)
    ax_a.text(173.0, 5.5e4, "+E"; color = "0.90", fontsize = 15.0, ha = "right", va = "top", zorder = 10)
    ax_a.text(187.0, 5.5e4, "−E"; color = "0.90", fontsize = 15.0, ha = "left", va = "top", zorder = 10)

    cb_a = fig.colorbar(im_a; ax = ax_a, fraction = 0.046, pad = 0.025)
    cb_a.ax.tick_params(which = "major", labelsize = 16, length = 5.5, width = 1.3)
    cb_a.ax.tick_params(which = "minor", length = 3.2, width = 1.0)
    cb_a.set_label("PSD (s\$^3\$ m\$^{-6}\$)"; fontsize = 18)

    # (b) 1D Orbital Work with crustal fields vs baseline total work
    ax_b.axhline(0.0; color = "0.50", ls = ":", lw = 1.2)
    ax_b.axvline(180.0; color = "0.45", ls = "--", lw = 1.2)
    ax_b.plot(orb_nc.angles, orb_nc.wt_1d; color = "0.52", ls = ":", lw = 2.4, label = "\$\\langle\\varepsilon_T\\rangle\$ (No crustal)", zorder = 3)
    ax_b.plot(orb_wc.angles, orb_wc.wt_1d; color = "black", ls = "--", lw = 2.5, label = "\$\\langle\\varepsilon_T\\rangle\$ (Crustal)", zorder = 5)
    ax_b.plot(orb_wc.angles, orb_wc.wc_1d; color = "#0072B2", lw = 2.2, label = "\$\\langle\\varepsilon_C\\rangle\$", zorder = 4)
    ax_b.plot(orb_wc.angles, orb_wc.wh_1d; color = "#D55E00", lw = 2.2, label = "\$\\langle\\varepsilon_H\\rangle\$", zorder = 4)
    ax_b.plot(orb_wc.angles, orb_wc.wa_1d; color = "#009E73", lw = 2.2, label = "\$\\langle\\varepsilon_A\\rangle\$", zorder = 4)
    ax_b.set_yscale("symlog"; linthresh = 10.0, linscale = 0.8)
    ax_b.set_xlim(0.0, 240.0)
    ax_b.set_ylim(-1.0e3, 3.5e5)
    ax_b.set_xticks(0:60:240)
    ax_b.set_yticks([-1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5])
    ax_b.set_xlabel("Orbit Angle (deg)"; fontsize = 19)
    ax_b.set_ylabel("Mean cumulative work (eV)"; fontsize = 19)
    ax_b.set_title("Global & plume work hierarchy preserved"; fontsize = 19, pad = 8)
    ax_b.grid(true; which = "major", alpha = 0.25)
    ax_b.legend(loc = "lower left", ncol = 2, fontsize = 13.5, frameon = true, facecolor = "white", edgecolor = "0.78", framealpha = 0.94, columnspacing = 0.8, handletextpad = 0.4)
    ax_b.text(0.03, 0.94, "(b)"; transform = ax_b.transAxes, fontsize = 20.5, fontweight = "bold", va = "top", ha = "left")

    if sh_wc !== nothing && sh_nc !== nothing
        badge = @sprintf(
            "1.0 \$R_M\$ shell shares:\nWith crustal: C %.0f%% | H %.0f%% | A %.0f%%\nNo crustal:   C %.0f%% | H %.0f%% | A %.0f%%\n~69%% lower ROI flux (trapping)",
            sh_wc[1], sh_wc[2], sh_wc[3], sh_nc[1], sh_nc[2], sh_nc[3],
        )
        ax_b.text(
            0.97, 0.95, badge;
            transform = ax_b.transAxes, fontsize = 13.0, va = "top", ha = "right",
            bbox = Dict("boxstyle" => "round,pad=0.28", "facecolor" => "white", "edgecolor" => "0.72", "alpha" => 0.93),
            zorder = 8,
        )
    end

    # (c) Warped Magnetotail Density n(Y, Z) & B_x = 0 Current-Sheet Contour at X = -2.0 Rm
    ax_c.set_facecolor("0.15")
    cmap_c = matplotlib.cm.get_cmap("turbo").copy()
    cmap_c.set_bad(color = "0.15")
    valid_d = filter(isfinite, vec(dens_rect))
    vmax_n = maximum(valid_d)
    vmin_n = max(vmax_n * 1.0e-3, 1.0e-3)
    im_c = ax_c.pcolormesh(y_rect, tail_z.z_list, dens_rect; cmap = cmap_c, norm = matplotlib.colors.LogNorm(vmin = vmin_n, vmax = vmax_n), shading = "auto", rasterized = true)
    ax_c.contour(y_grid_b, z_grid_b, bx_yz; levels = [0.0], colors = ["white"], linewidths = [2.6], zorder = 6)
    ax_c.axvline(0.0; color = "0.80", ls = ":", lw = 1.4, zorder = 5)
    ax_c.plot([first(tail_z.y_075), last(tail_z.y_075)], [0.75, 0.75]; color = "#d6192b", ls = "--", lw = 2.4, zorder = 7)
    ax_c.scatter([tail_z.ycs_075], [0.75]; s = 90, facecolor = "#ffd700", edgecolor = "black", lw = 1.2, zorder = 9)
    ax_c.annotate(
        "\$B_x=0\$ at \$Y_{CS}=$(@sprintf("%+.2f", tail_z.ycs_075))\\,R_M\$";
        xy = (tail_z.ycs_075, 0.75), xytext = (tail_z.ycs_075 - 0.1, 0.56),
        color = "white", fontsize = 14.5, fontweight = "bold", ha = "right", va = "top",
        bbox = Dict("boxstyle" => "round,pad=0.20", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.7),
        arrowprops = Dict("arrowstyle" => "->", "color" => "#ffd700", "lw" => 1.8),
        zorder = 10,
    )
    ax_c.text(0.04, -0.48, "\$Y = 0\$"; color = "0.90", fontsize = 14.5, ha = "left", va = "center", zorder = 8)
    ax_c.set_xlim(-1.35, 1.45)
    ax_c.set_ylim(-0.62, 0.85)
    ax_c.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 19)
    ax_c.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 19)
    ax_c.set_title("Warped tail CS (\$B_x = 0\$) at \$X = −2.0\\,R_M\$"; fontsize = 19, pad = 8)
    ax_c.text(0.03, 0.94, "(c)"; transform = ax_c.transAxes, color = "white", fontsize = 20.5, fontweight = "bold", va = "top", ha = "left", zorder = 10)

    cb_c = fig.colorbar(im_c; ax = ax_c, fraction = 0.046, pad = 0.025)
    cb_c.ax.tick_params(which = "major", labelsize = 16, length = 5.5, width = 1.3)
    cb_c.ax.tick_params(which = "minor", length = 3.2, width = 1.0)
    cb_c.set_label("\$n_{\\mathrm{O}_2^+}\$ (cm\$^{-3}\$)"; fontsize = 18)

    # (d) Cross-tail Hall work <ε_H>(Y, E) across the warped current sheet at X = -2.0 Rm, Z = +0.75 Rm
    ax_d.set_facecolor("black")
    cmap_d = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap_d.set_bad(color = "black")
    wh_plot = copy(permutedims(tail_z.wh_075))
    occ_d = permutedims(tail_z.psd_075 .> 0.0)
    wh_plot[.!occ_d .| .!isfinite.(wh_plot)] .= NaN
    im_d = ax_d.pcolormesh(tail_z.y_075, tail_z.ec_z, wh_plot; cmap = cmap_d, vmin = -300.0, vmax = 300.0, shading = "auto", rasterized = true)
    ax_d.axvline(tail_z.ycs_075; color = "#ffd700", ls = "--", lw = 2.0, zorder = 8)
    ax_d.axvline(0.0; color = "0.80", ls = ":", lw = 1.5, zorder = 7)
    ax_d.set_yscale("log")
    ax_d.set_xlim(first(tail_z.y_075), last(tail_z.y_075))
    ax_d.set_ylim(1.0e1, 1.0e4)
    ax_d.set_xticks(round.(collect(range(first(tail_z.y_075), last(tail_z.y_075); length = 5)); digits = 2))
    ax_d.set_xlabel("\$Y_{MSE}\$ (\$R_M\$)"; fontsize = 19)
    ax_d.set_ylabel("Energy (eV)"; fontsize = 19)
    ax_d.set_title("Asymmetric Hall work at \$Z = +0.75\\,R_M\$"; fontsize = 19, pad = 8)
    ax_d.text(0.03, 0.94, "(d)"; transform = ax_d.transAxes, color = "white", fontsize = 20.5, fontweight = "bold", va = "top", ha = "left", zorder = 10)
    ax_d.text(0.96, 0.94, "\$\\langle\\varepsilon_H\\rangle\$"; transform = ax_d.transAxes, color = "white", fontsize = 20.5, va = "top", ha = "right", zorder = 10)
    ax_d.text(tail_z.ycs_075 + 0.04, 4.5e3, "\$Y_{CS}=+0.36\$"; color = "#ffd700", fontsize = 14.5, fontweight = "bold", va = "top", ha = "left", zorder = 9)
    ax_d.text(-0.04, 4.5e3, "\$Y=0\$"; color = "0.90", fontsize = 14.5, va = "top", ha = "right", zorder = 9)

    cb_d = fig.colorbar(im_d; ax = ax_d, fraction = 0.046, pad = 0.025)
    cb_d.set_ticks(-300:100:300)
    cb_d.ax.tick_params(which = "major", labelsize = 16, length = 5.5, width = 1.3)
    cb_d.set_label("Mean Hall work \$\\langle\\varepsilon_H\\rangle\$ (eV)"; fontsize = 18)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 8 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    wc_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "outputs", "paper_figures_with_crustal")
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(wc_dir, "fig8.png")
    nc_dir = length(ARGS) >= 3 ? ARGS[3] : joinpath(pwd(), "outputs", "paper_figures_without_crustal")
    vts_wc = length(ARGS) >= 4 ? ARGS[4] : MarsTP.data_path("mars_fields_with_crustal.vts")
    plot_fig8(wc_dir, out; without_crustal_dir = nc_dir, vts_with_crustal = vts_wc)
end
