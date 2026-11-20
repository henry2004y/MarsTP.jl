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

if !isdefined(@__MODULE__, :paper_style)
    include(joinpath(@__DIR__, "plot_style.jl"))
end

function _default_paper_dir_fig4()
    p1 = joinpath(pwd(), "outputs", "paper_figures_without_crustal")
    isfile(joinpath(p1, "altitude_energy_maps_025_050_075_100Rm.jld2")) && return p1
    return joinpath(pwd(), "outputs", "paper_figures_with_crustal")
end

function _annotate_hemispheres_and_daynight!(
        ax;
        line_color::AbstractString = "white",
        line_alpha::Real = 0.55,
        fontsize::Real = 13.0,
    )
    # Reference lines: equator (lat = 0°, +E / -E hemisphere boundary) and terminators (lon = 90°, 270°)
    ax.axhline(0.0; color = line_color, linestyle = "--", linewidth = 0.95, alpha = line_alpha, zorder = 5)
    ax.axvline(90.0; color = line_color, linestyle = "--", linewidth = 0.95, alpha = line_alpha, zorder = 5)
    ax.axvline(270.0; color = line_color, linestyle = "--", linewidth = 0.95, alpha = line_alpha, zorder = 5)

    lbl_bbox = Dict("boxstyle" => "round,pad=0.16", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.45)

    # Dayside / Nightside notation near top of each panel
    ax.text(45.0, 77.0, "Dayside"; fontsize = fontsize, color = "white", va = "center", ha = "center", bbox = lbl_bbox, zorder = 7)
    ax.text(180.0, 77.0, "Nightside"; fontsize = fontsize, color = "white", va = "center", ha = "center", bbox = lbl_bbox, zorder = 7)
    ax.text(315.0, 77.0, "Dayside"; fontsize = fontsize, color = "white", va = "center", ha = "center", bbox = lbl_bbox, zorder = 7)

    # +E / -E hemisphere notation on the right side (clear of Subsolar label and bottom stat box)
    ax.text(352.0, 40.0, "+E hem."; fontsize = fontsize, color = "white", va = "center", ha = "right", bbox = lbl_bbox, zorder = 7)
    ax.text(352.0, -40.0, "−E hem."; fontsize = fontsize, color = "white", va = "center", ha = "right", bbox = lbl_bbox, zorder = 7)
    return nothing
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

    st = paper_style(16.6)

    fig = plt.figure(figsize = (16.6, 12.5), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.03, h_pad = 0.02, wspace = 0.03, hspace = 0.02)
    sfigs = fig.subfigures(2, 1; height_ratios = [1.44, 0.72])
    sfig_top = sfigs[1]
    sfig_bot = sfigs[2]

    # Top 2x2: Residence-weighted mean energy <E> across 0.25, 0.50, 0.75, 1.00 Rm
    # Use a 2x3 GridSpec so the right colorbar spans the full height of both rows.
    gs_top = sfig_top.add_gridspec(2, 3; width_ratios = [1.0, 1.0, 0.045], hspace = 0.06)
    ax_t11 = sfig_top.add_subplot(gs_top[1, 1])
    ax_t12 = sfig_top.add_subplot(gs_top[1, 2]; sharex = ax_t11, sharey = ax_t11)
    ax_t21 = sfig_top.add_subplot(gs_top[2, 1]; sharex = ax_t11, sharey = ax_t11)
    ax_t22 = sfig_top.add_subplot(gs_top[2, 2]; sharex = ax_t11, sharey = ax_t11)
    axes_top = [ax_t11 ax_t12; ax_t21 ax_t22]
    cax_e = sfig_top.add_subplot(gs_top.new_subplotspec((0, 2); rowspan = 2, colspan = 1))
    vmin_eV = 1.0e-1
    vmax_eV = 1.0e4
    norm_e = matplotlib.colors.LogNorm(vmin = vmin_eV, vmax = vmax_eV)
    cmap_e = paper_cmap_continuous()

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

        _annotate_hemispheres_and_daynight!(ax; line_color = REF_ON_DARK, line_alpha = 0.65, fontsize = st.annot_sm)

        ax.scatter([0.0], [0.0]; marker = "*", s = 220, facecolor = "white", edgecolor = "black", lw = 1.2, zorder = 6, clip_on = false)
        if k == 1
            ax.text(
                10.0, 19.0, "Subsolar";
                fontsize = st.annot, va = "center", ha = "left",
                bbox = Dict("boxstyle" => "square,pad=0.18", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.9),
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
            e_p90 = quantile(valid_e, 0.9)
            stat_str = @sprintf(
                "med=%.1f eV, P90=%.0f eV | C:%.0f%% H:%.0f%% A:%.0f%%",
                e_med, e_p90, 100.0 * wc_all / wt_all, 100.0 * wh_all / wt_all, 100.0 * wa_all / wt_all,
            )
            ax.text(
                0.98, 0.05, stat_str;
                transform = ax.transAxes, fontsize = st.annot_sm, color = "white",
                va = "bottom", ha = "right",
                bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.72),
                zorder = 7,
            )
        end

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:60:360)
        ax.set_yticks(-90:30:90)
        ax.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        row == 1 && ax.tick_params(labelbottom = false)
        col == 2 && ax.tick_params(labelleft = false)
        ax.set_title("($(tags_e[k])) O\$_2^+\$ mean energy at $alt_lbl"; fontsize = st.title, pad = 7)
        col == 1 && ax.set_ylabel("Latitude (°)"; fontsize = st.label)
        row == 2 && ax.set_xlabel("Longitude (°)"; fontsize = st.label)
    end

    cb_e = sfig_top.colorbar(im_e; cax = cax_e)
    style_colorbar!(cb_e, st; label = "Residence-weighted mean energy \$\\langle E \\rangle\$ (eV)", labelpad = 8)

    # Bottom 1x3: Spatial work partition at 0.75 Rm (2542 km, k = 3) on RdBu_r SymLogNorm(linthresh = 10.0 eV)
    # Match colorbar physical width with the top GridSpec (0.045/2.045 ≈ 0.0675/3.0675).
    gs_bot = sfig_bot.add_gridspec(1, 4; width_ratios = [1.0, 1.0, 1.0, 0.0675])
    ax_b1 = sfig_bot.add_subplot(gs_bot[1, 1])
    ax_b2 = sfig_bot.add_subplot(gs_bot[1, 2]; sharex = ax_b1, sharey = ax_b1)
    ax_b3 = sfig_bot.add_subplot(gs_bot[1, 3]; sharex = ax_b1, sharey = ax_b1)
    axes_bot = [ax_b1, ax_b2, ax_b3]
    cax_w = sfig_bot.add_subplot(gs_bot[1, 4])
    k_ref = argmin(abs.(data.altitudes_km .- 0.75 * (Rm / 1.0e3)))
    alt_ref_lbl = _format_altitude_label(Float64(data.altitudes_km[k_ref]))

    vmin_w = -1.0e3
    vmax_w = 1.0e4
    linthresh_w = 10.0
    norm_w = matplotlib.colors.SymLogNorm(linthresh = linthresh_w, linscale = 0.8, vmin = vmin_w, vmax = vmax_w, base = 10)
    cmap_w = paper_cmap_diverging()

    w_tot_all = sum(@view data.hist_w_tau[k_ref, 2:end, :, :])
    wc_sum = sum(@view data.hist_wc_tau[k_ref, 2:end, :, :])
    wh_sum = sum(@view data.hist_wh_tau[k_ref, 2:end, :, :])
    wa_sum = sum(@view data.hist_wa_tau[k_ref, 2:end, :, :])
    wt_shell_eV = w_tot_all > 0.0 ? (wc_sum + wh_sum + wa_sum) / w_tot_all : 1.0

    comp_specs = (
        (1, data.mean_w_conv_eV, wc_sum, "e", "\$\\langle\\varepsilon_C\\rangle\$ (Convective)"),
        (2, data.mean_w_hall_eV, wh_sum, "f", "\$\\langle\\varepsilon_H\\rangle\$ (Hall)"),
        (3, data.mean_w_amb_eV, wa_sum, "g", "\$\\langle\\varepsilon_A\\rangle\$ (Ambipolar)"),
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

        _annotate_hemispheres_and_daynight!(ax; line_color = REF_GRAY, line_alpha = 0.65, fontsize = round(st.annot_sm * 0.85; digits = 1))

        ax.scatter([0.0], [0.0]; marker = "*", s = 180, facecolor = "white", edgecolor = "black", lw = 1.1, zorder = 6, clip_on = false)
        shell_mean = w_tot_all > 0.0 ? (h_sum / w_tot_all) : 0.0
        share_pct = 100.0 * shell_mean / wt_shell_eV
        stat_str = @sprintf("shell mean = %+.1f eV (%.0f%%)", shell_mean, share_pct)
        ax.text(
            0.97, 0.06, stat_str;
            transform = ax.transAxes, fontsize = round(st.annot_sm * 0.94; digits = 1), color = "white",
            va = "bottom", ha = "right",
            bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.74),
            zorder = 7,
        )

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:90:360)
        ax.set_yticks(-90:30:90)
        ax.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        col > 1 && ax.tick_params(labelleft = false)
        ax.set_title("($tag) $title_lbl at 0.75 \$R_M\$"; fontsize = st.title_sm, pad = 7)
        ax.set_xlabel("Longitude (°)"; fontsize = st.label)
        col == 1 && ax.set_ylabel("Latitude (°)"; fontsize = st.label)
    end

    cb_w = sfig_bot.colorbar(im_w; cax = cax_w)
    cb_w.set_ticks([-1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4])
    cb_w.set_ticklabels(["−10\$^3\$", "−10\$^2\$", "−10\$^1\$", "0", "10\$^1\$", "10\$^2\$", "10\$^3\$", "10\$^4\$"])
    style_colorbar!(cb_w, st; label = "Residence-weighted work \$\\langle\\varepsilon_\\ell\\rangle\$ (eV)", labelpad = 8)

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
