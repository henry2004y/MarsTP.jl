# Publication-quality script for Figure 4 of the Mars O2+ test-particle manuscript:
#   7-panel Global Altitude Energy & Electric-Field Work Synthesis:
#     Top 2x2 grid (a)-(d): Residence-weighted mean O2+ energy <E>(Lon, Lat) at altitudes
#       0.25, 0.50, 0.75, and 1.00 Rm (848, 1695, 2542, and 3390 km) on LogNorm turbo scale
#     Bottom 1x3 grid (e)-(g): Spatial electric-field work maps at 0.75 Rm (2542 km):
#       (e) Convective work <ε_C>, (f) Hall work <ε_H>, (g) Ambipolar work <ε_A>
#       on diverging RdBu_r SymLogNorm(linthresh = 10.0 eV) scale
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig4.jl [PAPER_DIR] [OUTPUT_PNG] [VTS_PATH]

if !isdefined(@__MODULE__, :compute_altitude_energy_maps)
    include(joinpath(@__DIR__, "plot_altitude_energy_maps.jl"))
end

using Statistics
using Printf
using PyPlot

function _default_paper_dir_fig4()
    p1 = joinpath(pwd(), "outputs", "paper_figures_without_crustal")
    isfile(joinpath(p1, "altitude_energy_maps_025_050_075_100Rm.jld2")) && return p1
    return joinpath(pwd(), "outputs", "paper_figures_with_crustal")
end

function plot_fig4(
        paper_dir::AbstractString = _default_paper_dir_fig4(),
        out_png::AbstractString = joinpath(paper_dir, "fig4.png");
        vts_path::AbstractString = "",
    )
    resolved_vts = if !isempty(vts_path) && isfile(vts_path)
        String(vts_path)
    elseif occursin("with_crustal", lowercase(basename(abspath(paper_dir)))) && !occursin("without_crustal", lowercase(basename(abspath(paper_dir))))
        MarsTP.data_path("mars_fields_with_crustal.vts")
    else
        MarsTP.data_path("mars_fields_without_crustal.vts")
    end

    cache_jld2 = joinpath(paper_dir, "altitude_energy_maps_025_050_075_100Rm.jld2")
    isfile(cache_jld2) || error("Missing cached altitude maps JLD2: $cache_jld2")

    data = compute_altitude_energy_maps(;
        vts_path = resolved_vts,
        cache_jld2 = cache_jld2,
        recompute = false,
    )

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4

    fig = plt.figure(figsize = (16.6, 13.8), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.05, wspace = 0.04, hspace = 0.06)
    sfigs = fig.subfigures(2, 1; height_ratios = [1.55, 0.72])
    sfig_top = sfigs[1]
    sfig_bot = sfigs[2]

    # Top 2x2: Residence-weighted mean energy <E> across 0.25, 0.50, 0.75, 1.00 Rm
    axes_top = sfig_top.subplots(2, 2; sharex = true, sharey = true)
    vmin_eV = 1.0e-1
    vmax_eV = 1.0e4
    norm_e = matplotlib.colors.LogNorm(vmin = vmin_eV, vmax = vmax_eV)
    cmap_e = matplotlib.cm.get_cmap("turbo").copy()
    cmap_e.set_bad(color = "0.15")

    tags_e = ("a", "b", "c", "d")
    im_e = nothing
    for k in 1:4
        row = cld(k, 2)
        col = mod1(k, 2)
        ax = axes_top[row, col]
        alt_km = Float64(data.altitudes_km[k])
        alt_lbl = _format_altitude_label(alt_km)
        mat_smooth = _smooth_and_upsample_log_energy(data.mean_energy_tau_eV[k, :, :]; upsample = 4)
        for i in eachindex(mat_smooth)
            if isfinite(mat_smooth[i]) && mat_smooth[i] > 0.0
                mat_smooth[i] = clamp(mat_smooth[i], vmin_eV, vmax_eV)
            else
                mat_smooth[i] = NaN
            end
        end

        im_e = ax.imshow(
            mat_smooth;
            extent = (0.0, 360.0, -90.0, 90.0),
            origin = "lower",
            aspect = "auto",
            cmap = cmap_e,
            norm = norm_e,
            interpolation = "bicubic",
            rasterized = true,
        )

        ax.scatter([0.0], [0.0]; marker = "*", s = 220, facecolor = "white", edgecolor = "black", lw = 1.2, zorder = 6, clip_on = false)
        if k == 1
            ax.text(
                10.0, 19.0, "Subsolar";
                fontsize = 16.5, va = "center", ha = "left",
                bbox = Dict("boxstyle" => "square,pad=0.18", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.90),
                zorder = 6,
            )
        end

        wc_all = sum(@view data.hist_wc_tau[k, 2:end, :, :])
        wh_all = sum(@view data.hist_wh_tau[k, 2:end, :, :])
        wa_all = sum(@view data.hist_wa_tau[k, 2:end, :, :])
        wt_all = wc_all + wh_all + wa_all

        valid_e = filter(x -> isfinite(x) && x > 0.0, vec(data.mean_energy_tau_eV[k, :, :]))
        if !isempty(valid_e) && abs(wt_all) > 0.0
            e_med = median(valid_e)
            e_p90 = quantile(valid_e, 0.90)
            stat_str = @sprintf(
                "med=%.1f eV, P90=%.0f eV | C:%.0f%% H:%.0f%% A:%.0f%%",
                e_med, e_p90, 100.0 * wc_all / wt_all, 100.0 * wh_all / wt_all, 100.0 * wa_all / wt_all,
            )
            ax.text(
                0.98, 0.05, stat_str;
                transform = ax.transAxes, fontsize = 13.8, color = "white",
                va = "bottom", ha = "right",
                bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.72),
                zorder = 7,
            )
        end

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:60:360)
        ax.set_yticks(-90:30:90)
        ax.set_title("($(tags_e[k])) O\$_2^+\$ mean energy at $alt_lbl"; fontsize = 19.5, pad = 7)
        col == 1 && ax.set_ylabel("Latitude (°)"; fontsize = 19.5)
        row == 2 && ax.set_xlabel("Longitude (°)"; fontsize = 19.5)
    end

    cb_e = sfig_top.colorbar(im_e; ax = vec(axes_top), location = "right", fraction = 0.020, pad = 0.018)
    cb_e.ax.tick_params(which = "major", labelsize = 17.5, length = 6.0, width = 1.3)
    cb_e.ax.tick_params(which = "minor", length = 3.5, width = 1.0)
    cb_e.set_label("Residence-weighted mean energy \$\\langle E \\rangle\$ (eV)"; fontsize = 19.5, labelpad = 8)

    # Bottom 1x3: Spatial work partition at 0.75 Rm (2542 km, k = 3) on RdBu_r SymLogNorm(linthresh = 10.0 eV)
    axes_bot = sfig_bot.subplots(1, 3; sharex = true, sharey = true)
    k_ref = argmin(abs.(data.altitudes_km .- 0.75 * (Rm / 1.0e3)))
    alt_ref_lbl = _format_altitude_label(Float64(data.altitudes_km[k_ref]))

    vmin_w = -1.0e3
    vmax_w = 1.0e4
    linthresh_w = 10.0
    norm_w = matplotlib.colors.SymLogNorm(linthresh = linthresh_w, linscale = 0.8, vmin = vmin_w, vmax = vmax_w, base = 10)
    cmap_w = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap_w.set_bad(color = "0.15")

    w_tot_all = sum(@view data.hist_w_tau[k_ref, 2:end, :, :])
    wc_sum = sum(@view data.hist_wc_tau[k_ref, 2:end, :, :])
    wh_sum = sum(@view data.hist_wh_tau[k_ref, 2:end, :, :])
    wa_sum = sum(@view data.hist_wa_tau[k_ref, 2:end, :, :])
    wt_shell_eV = w_tot_all > 0.0 ? (wc_sum + wh_sum + wa_sum) / w_tot_all : 1.0

    comp_specs = (
        (1, data.mean_w_conv_eV, wc_sum, "e", "\$\\langle\\varepsilon_C\\rangle\$ (Convective)"),
        (2, data.mean_w_hall_eV, wh_sum, "f", "\$\\langle\\varepsilon_H\\rangle\$ (Hall)"),
        (3, data.mean_w_amb_eV,  wa_sum, "g", "\$\\langle\\varepsilon_A\\rangle\$ (Ambipolar)"),
    )
    im_w = nothing
    for (col, cube, h_sum, tag, title_lbl) in comp_specs
        ax = axes_bot[col]
        raw_mat = cube[k_ref, :, :]
        mat_smooth = _smooth_and_upsample_symlog_work(raw_mat; upsample = 4, linthresh = linthresh_w)
        for i in eachindex(mat_smooth)
            if isfinite(mat_smooth[i])
                mat_smooth[i] = clamp(mat_smooth[i], vmin_w, vmax_w)
            else
                mat_smooth[i] = NaN
            end
        end

        im_w = ax.imshow(
            mat_smooth;
            extent = (0.0, 360.0, -90.0, 90.0),
            origin = "lower",
            aspect = "auto",
            cmap = cmap_w,
            norm = norm_w,
            interpolation = "bicubic",
            rasterized = true,
        )

        ax.scatter([0.0], [0.0]; marker = "*", s = 180, facecolor = "white", edgecolor = "black", lw = 1.1, zorder = 6, clip_on = false)
        shell_mean = w_tot_all > 0.0 ? (h_sum / w_tot_all) : 0.0
        share_pct = 100.0 * shell_mean / wt_shell_eV
        stat_str = @sprintf("shell mean = %+.1f eV (%.0f%%)", shell_mean, share_pct)
        ax.text(
            0.97, 0.06, stat_str;
            transform = ax.transAxes, fontsize = 13.5, color = "white",
            va = "bottom", ha = "right",
            bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.74),
            zorder = 7,
        )

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:90:360)
        ax.set_yticks(-90:30:90)
        ax.set_title("($tag) $title_lbl at 0.75 \$R_M\$"; fontsize = 17.5, pad = 7)
        ax.set_xlabel("Longitude (°)"; fontsize = 19.5)
        col == 1 && ax.set_ylabel("Latitude (°)"; fontsize = 19.5)
    end

    cb_w = sfig_bot.colorbar(im_w; ax = vec(axes_bot), location = "right", fraction = 0.020, pad = 0.018)
    cb_w.set_ticks([-1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4])
    cb_w.set_ticklabels(["−10\$^3\$", "−10\$^2\$", "−10\$^1\$", "0", "10\$^1\$", "10\$^2\$", "10\$^3\$", "10\$^4\$"])
    cb_w.ax.tick_params(which = "major", labelsize = 17.5, length = 6.0, width = 1.3)
    cb_w.ax.tick_params(which = "minor", length = 3.5, width = 1.0)
    cb_w.set_label("Residence-weighted work \$\\langle\\varepsilon_\\ell\\rangle\$ (eV)"; fontsize = 19.5, labelpad = 8)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 4 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : _default_paper_dir_fig4()
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig4.png")
    vts = length(ARGS) >= 3 ? ARGS[3] : ""
    plot_fig4(pdir, out; vts_path = vts)
end
