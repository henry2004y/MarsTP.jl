# ==============================================================================
# Unified 1D Analytical Benchmark: Source Altitude Sensitivity & TP-MHD Discrepancy
# ==============================================================================
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/forward_tracing/tp_mhd_discrepancy_benchmark.jl
#
# Generates the 3-panel figure `examples/forward_tracing/images/tp_mhd_discrepancy_6panel.png`
# (or custom path via `TP_MHD_BENCHMARK_OUTPUT`).

using MarsTP
using StaticArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using PyPlot

if !isdefined(@__MODULE__, :paper_style)
    include(joinpath(@__DIR__, "monte_carlo_forward_tracing", "plot_style.jl"))
end

function run_unified_tp_mhd_benchmark(;
        N_p_col::Int = 15_000,
        output_png::AbstractString = get(
            ENV,
            "TP_MHD_BENCHMARK_OUTPUT",
            joinpath(@__DIR__, "images", "tp_mhd_discrepancy_6panel.png"),
        ),
        save_csv_dir::Union{Nothing, AbstractString} = nothing,
    )
    mkpath(dirname(abspath(output_png)))
    col = AnalyticalOutflowColumn()

    println("1. Running Mechanism 1: Fluid Pressure Closure vs. Kinetic Streaming (Parallel B)...")
    p2 = benchmark_parallel_gradpi(col; N_p = 4_000)

    println("2. Running Mechanism 2: Ionospheric O2+ Source & h0 Sensitivity across 200, 400, 600 km...")
    ob_200 = trace_outflow_column(col; z0_km = 200.0, b_geometry = :oblique, e_mode = :full, sample_mode = :flux, N_p = N_p_col, seed = 101)
    ob_400 = trace_outflow_column(col; z0_km = 400.0, b_geometry = :oblique, e_mode = :full, sample_mode = :flux, N_p = N_p_col, seed = 102)
    ob_600 = trace_outflow_column(col; z0_km = 600.0, b_geometry = :oblique, e_mode = :full, sample_mode = :flux, N_p = N_p_col, seed = 103)

    z_km = ob_200.z_km
    z_m = z_km .* 1.0e3
    idx_800 = findfirst(>=(800.0), z_km)

    prod_above_200 = column_mhd_flux(col, col.z_max) - column_mhd_flux(col, 200.0e3)
    prod_above_300 = column_mhd_flux(col, col.z_max) - column_mhd_flux(col, 300.0e3)
    prod_above_400 = column_mhd_flux(col, col.z_max) - column_mhd_flux(col, 400.0e3)
    prod_above_600 = column_mhd_flux(col, col.z_max) - column_mhd_flux(col, 600.0e3)
    println("\n=== Ionospheric O2+ Source Confinement Summary ===")
    h_src_grid = range(150.0e3, 400.0e3; length = 251)
    s_src_grid = [column_source_rate(col, z) for z in h_src_grid]
    s_peak_idx = argmax(s_src_grid)
    @printf(
        "  Peak S_net = %.2f x10^6 m^-3 s^-1 at h_peak = %.0f km\n",
        s_src_grid[s_peak_idx] / 1.0e6,
        h_src_grid[s_peak_idx] / 1.0e3
    )
    @printf(
        "  Fraction of (>200 km) S_net below 300 km : %.2f%% (above 300 km: %.2f%%)\n",
        100.0 * (1.0 - prod_above_300 / prod_above_200), 100.0 * prod_above_300 / prod_above_200
    )
    @printf(
        "  Fraction of (>200 km) S_net below 400 km : %.2f%% (above 400 km: %.3f%%)\n",
        100.0 * (1.0 - prod_above_400 / prod_above_200), 100.0 * prod_above_400 / prod_above_200
    )
    @printf(
        "  Fraction of (>200 km) S_net below 600 km : %.4f%% (above 600 km: %.4f%%)\n",
        100.0 * (1.0 - prod_above_600 / prod_above_200), 100.0 * prod_above_600 / prod_above_200
    )
    @printf(
        "  MHD Flux at 200, 400, 600, 800 km (x10^11 m^-2 s^-1): %.2f, %.2f, %.2f, %.2f\n",
        column_mhd_flux(col, 200.0e3) / 1.0e11,
        column_mhd_flux(col, 400.0e3) / 1.0e11,
        column_mhd_flux(col, 600.0e3) / 1.0e11,
        column_mhd_flux(col, 800.0e3) / 1.0e11
    )
    @printf(
        "  MHD Density at 200, 400, 600, 800 km (x10^8 m^-3)   : %.2f, %.2f, %.2f, %.2f\n",
        column_mhd_density(col, 200.0e3) / 1.0e8,
        column_mhd_density(col, 400.0e3) / 1.0e8,
        column_mhd_density(col, 600.0e3) / 1.0e8,
        column_mhd_density(col, 800.0e3) / 1.0e8
    )
    @printf(
        "  MHD U_h at 200, 400, 600, 800 km (km/s)            : %.2f, %.2f, %.2f, %.2f\n",
        column_mhd_velocity(col, 200.0e3) / 1.0e3,
        column_mhd_velocity(col, 400.0e3) / 1.0e3,
        column_mhd_velocity(col, 600.0e3) / 1.0e3,
        column_mhd_velocity(col, 800.0e3) / 1.0e3
    )

    println("\n=== Parallel-Field Mechanism 1 Summary (h = 200 -> 800 km) ===")
    @printf(
        "  MHD Fluid Closure (-∇P_e - ∇P_i) : U_h(200)=%4.2f km/s -> U_h(800)=%4.2f km/s\n",
        p2.u_mhd_km_s[1], p2.u_mhd_km_s[end]
    )
    @printf(
        "  TP Free-Streaming (E_∥ = 0)      : U_h(200)=%4.2f km/s -> U_h(800)=%4.2f km/s\n",
        p2.u_free_km_s[1], p2.u_free_km_s[end]
    )
    @printf(
        "  TP with E_A = -∇P_e/(qn) only    : U_h(200)=%4.2f km/s -> U_h(800)=%4.2f km/s\n",
        p2.u_amb_km_s[1], p2.u_amb_km_s[end]
    )

    println("\n=== Oblique Draped Summary at h_c = 800 km (g = 0, matching Eq. 5) ===")
    @printf(
        "  MHD Exact                        : n = %5.2f x10^8 m^-3 | U_h = %5.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1\n",
        column_mhd_density(col, 800.0e3) / 1.0e8,
        column_mhd_velocity(col, 800.0e3) / 1.0e3,
        column_mhd_flux(col, 800.0e3) / 1.0e11
    )
    for (lbl, res) in (("h0 = 200 km", ob_200), ("h0 = 400 km", ob_400), ("h0 = 600 km", ob_600))
        @printf(
            "  Oblique Draped %-11s       : n = %5.2f x10^8 m^-3 | U_h = %5.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1 (esc = %5.1f%%) | <ε_T> = %.2f eV (C/H/A = %.1f%% / %.1f%% / %.1f%%)\n",
            lbl,
            res.n_m3[idx_800] / 1.0e8, res.u_z_m_s[idx_800] / 1.0e3, res.flux_m2_s[idx_800] / 1.0e11,
            100.0 * res.escape_fraction,
            res.shares_800.total_eV, res.shares_800.conv_pct, res.shares_800.hall_pct, res.shares_800.amb_pct
        )
    end

    if save_csv_dir !== nothing
        mkpath(save_csv_dir)
        writedlm(
            joinpath(save_csv_dir, "case1_parallel.csv"),
            hcat(p2.z_km, p2.u_mhd_km_s, p2.u_free_km_s, p2.u_amb_km_s), ','
        )
        profiles_mat = hcat(
            z_km,
            [column_mhd_density(col, z) / 1.0e8 for z in z_m],
            [column_mhd_velocity(col, z) / 1.0e3 for z in z_m],
            [column_mhd_flux(col, z) / 1.0e11 for z in z_m],
            ob_200.n_m3 ./ 1.0e8, ob_200.flux_m2_s ./ 1.0e11,
            ob_400.n_m3 ./ 1.0e8, ob_400.flux_m2_s ./ 1.0e11,
            ob_600.n_m3 ./ 1.0e8, ob_600.flux_m2_s ./ 1.0e11,
        )
        writedlm(joinpath(save_csv_dir, "case2_oblique_profiles.csv"), profiles_mat, ',')
    end

    # ==========================================================================
    # Render 3-Panel Publication Figure (2 Top + 1 Centered Bottom)
    # ==========================================================================
    st = paper_style(14.4)

    fig = PyPlot.plt.figure(figsize = (14.4, 10.4), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.05, h_pad = 0.06, wspace = 0.06, hspace = 0.08)
    gs_outer = fig.add_gridspec(2, 1; height_ratios = [1.05, 0.95])
    gs_top = gs_outer[1].subgridspec(1, 2; wspace = 0.22)
    gs_bot = gs_outer[2].subgridspec(1, 3; width_ratios = [0.08, 0.84, 0.08], wspace = 0.0)

    # --------------------------------------------------------------------------
    # Panel (a): Mechanism 1 — Fluid Pressure Closure vs. Kinetic Streaming
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs_top[1, 1])
    ax_a.plot(p2.z_km, p2.u_mhd_km_s; color = COL_TOTAL, lw = 2.8, label = raw"MHD Fluid Closure $U_{h,\mathrm{MHD}}(h)$")
    ax_a.plot(p2.z_km, p2.u_amb_km_s; color = COL_AMB, lw = 2.6, label = raw"Kinetic TP with $\vec{E}_A = -\nabla P_e/(qn)$")
    ax_a.plot(p2.z_km, p2.u_free_km_s; color = SET2_BLUE, lw = 2.4, ls = "--", label = raw"Kinetic TP Free-Streaming ($E_\parallel = 0$)")
    idx_700 = findfirst(>=(700.0), p2.z_km)
    u_mhd_700 = p2.u_mhd_km_s[idx_700]
    u_amb_700 = p2.u_amb_km_s[idx_700]
    ax_a.annotate(
        "";
        xy = (700.0, u_mhd_700),
        xytext = (700.0, u_amb_700),
        arrowprops = Dict("arrowstyle" => "<->", "color" => REF_GRAY, "lw" => 1.5, "shrinkA" => 1, "shrinkB" => 1),
    )
    ax_a.annotate(
        "Fluid scalar\n\$P_i\$ closure";
        xy = (696.0, 0.5 * (u_mhd_700 + u_amb_700)),
        xytext = (585.0, 2.38),
        ha = "center", va = "center", fontsize = st.annot_sm, color = "#222222",
        bbox = Dict("boxstyle" => "round,pad=0.3", "facecolor" => "#f8f9fa", "edgecolor" => "0.72", "alpha" => 0.94),
        arrowprops = Dict("arrowstyle" => "->", "color" => REF_GRAY, "lw" => 1.1, "connectionstyle" => "arc3,rad=0.15"),
    )
    ax_a.set_xlim(200.0, 800.0)
    ax_a.set_ylim(1.75, 4.22)
    ax_a.set_xlabel(raw"Altitude $h$ (km)"; fontsize = st.label)
    ax_a.set_ylabel(raw"Bulk Outflow Speed $U_h$ (km/s)"; fontsize = st.label)
    ax_a.set_title("(a) Fluid Pressure Closure vs. Kinetic Streaming"; fontsize = st.title_sm, fontweight = "bold", pad = 8.0)
    ax_a.tick_params(axis = "both", which = "major", labelsize = st.tick)
    ax_a.grid(true; alpha = 0.25)
    ax_a.legend(; loc = "upper left", fontsize = st.legend, framealpha = 0.94)

    # --------------------------------------------------------------------------
    # Panel (b): Mechanism 2 — Ionospheric O2+ Source & h0 Sensitivity
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs_top[1, 2])
    z_fine_km = collect(range(150.0, 900.0; length = 300))
    z_fine_m = z_fine_km .* 1.0e3
    n_mhd_prof = [column_mhd_density(col, z) / 1.0e8 for z in z_fine_m]
    s_net_prof = [column_source_rate(col, z) / 1.0e6 for z in z_fine_m]

    ax_b_rhs = ax_b.twinx()
    ax_b_rhs.fill_between(z_fine_km, 0.0, s_net_prof; color = SET2_TAN, alpha = 0.32)
    ax_b_rhs.plot(z_fine_km, s_net_prof; color = REF_GRAY, lw = 1.6, ls = "--", alpha = 0.85)
    ax_b_rhs.set_ylim(0.0, 5.2)
    ax_b_rhs.set_ylabel(raw"Net Source $S_{\mathrm{net}}$ ($10^6\ \mathrm{m^{-3}\,s^{-1}}$)"; fontsize = st.label, color = "0.35")
    ax_b_rhs.tick_params(axis = "y", colors = "0.35", labelsize = st.tick)

    ax_b.set_zorder(ax_b_rhs.get_zorder() + 1)
    ax_b.patch.set_visible(false)

    m200 = (200.0 .<= z_km .<= 900.0) .& (ob_200.n_m3 .> 0.0)
    m400 = (400.0 .<= z_km .<= 900.0) .& (ob_400.n_m3 .> 0.0)
    m600 = (600.0 .<= z_km .<= 900.0) .& (ob_600.n_m3 .> 0.0)
    gmhd = @sprintf("%.2f", column_mhd_flux(col, 800.0e3) / 1.0e11)
    g200 = @sprintf("%.2f", ob_200.flux_m2_s[idx_800] / 1.0e11)
    g400 = @sprintf("%.2f", ob_400.flux_m2_s[idx_800] / 1.0e11)
    g600 = @sprintf("%.2f", ob_600.flux_m2_s[idx_800] / 1.0e11)

    for z0_mark in (200.0, 400.0, 600.0)
        ax_b.axvline(z0_mark; color = REF_GRAY, ls = ":", lw = 1.0)
    end
    ax_b.plot(z_fine_km, n_mhd_prof; color = COL_TOTAL, lw = 3.0, ls = "-", zorder = 3, label = "MHD \$n_{\\mathrm{MHD}}(h)\$ (\$\\Gamma_{800}=$(gmhd)\$)")
    for (mask_j, res_j, clr_j, ls_j, zo_j, lbl_j) in (
            (m200, ob_200, SET2_BLUE, "-", 4, "TP \$h_0=200\$ km (\$\\Gamma_{800}=$(g200)\$)"),
            (m400, ob_400, SET2_TEAL, "--", 5, "TP \$h_0=400\$ km (\$\\Gamma_{800}=$(g400)\$)"),
            (m600, ob_600, SET2_ORANGE, "-.", 6, "TP \$h_0=600\$ km (\$\\Gamma_{800}=$(g600)\$)"),
        )
        zk_sub = z_km[mask_j]
        nk_sub = res_j.n_m3[mask_j] ./ 1.0e8
        ax_b.plot(zk_sub, nk_sub; color = clr_j, lw = 2.6, ls = ls_j, zorder = zo_j, label = lbl_j)
        ax_b.plot([zk_sub[1]], [nk_sub[1]]; marker = "o", ls = "none", color = clr_j, markersize = 7.5, markeredgecolor = "black", markeredgewidth = 0.7, zorder = 7)
    end
    ax_b_rhs.annotate(
        "Ionospheric\n\$S_{\\mathrm{net}} = P - L\$";
        xy = (258.0, column_source_rate(col, 258.0e3) / 1.0e6),
        xytext = (450.0, 0.55),
        ha = "center", va = "center", color = "0.25", fontsize = st.annot_sm, fontweight = "bold",
        bbox = Dict("boxstyle" => "round,pad=0.28", "facecolor" => "#f8f5ef", "edgecolor" => SET2_TAN, "alpha" => 0.94),
        arrowprops = Dict("arrowstyle" => "->", "color" => REF_GRAY, "lw" => 1.1, "connectionstyle" => "arc3,rad=-0.18"),
    )
    ax_b.set_xlim(150.0, 900.0)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.05, 16.0)
    ax_b.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ax_b.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax_b.minorticks_off()
    ax_b.set_xlabel(raw"Altitude $h$ (km)"; fontsize = st.label)
    ax_b.set_ylabel(raw"Ion Density $n$ ($10^8\ \mathrm{m^{-3}}$)"; fontsize = st.label)
    ax_b.set_title(raw"(b) $\mathrm{O}_2^+$ Source Layer & $h_0$ Sensitivity (Oblique $\vec{B}$)"; fontsize = st.title_sm, fontweight = "bold", pad = 8.0)
    ax_b.tick_params(axis = "both", which = "major", labelsize = st.tick)
    ax_b.grid(true; alpha = 0.25)
    ax_b.legend(;
        loc = "upper right", fontsize = round(st.legend * 0.96; digits = 1), framealpha = 0.94,
        title = raw"$\Gamma_{800}$ ($10^{11}\ \mathrm{m^{-2}\,s^{-1}}$)", title_fontsize = round(st.legend * 0.96; digits = 1)
    )

    # --------------------------------------------------------------------------
    # Panel (c): Sensitivity of Electric-Field Work at 800 km to Launch Altitude
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs_bot[1, 2])
    runs = (ob_200, ob_400, ob_600)
    z0_vals = (200, 400, 600)
    x_pos = collect(0.0:1.0:2.0)
    bar_w = 0.24

    conv_pcts = [r.shares_800.conv_pct for r in runs]
    hall_pcts = [r.shares_800.hall_pct for r in runs]
    amb_pcts = [r.shares_800.amb_pct for r in runs]

    b_c = ax_c.bar(x_pos .- bar_w, conv_pcts; width = bar_w, color = COL_CONV, edgecolor = "black", linewidth = 0.7, label = raw"Convective $\langle\varepsilon_C\rangle$")
    b_h = ax_c.bar(x_pos, hall_pcts; width = bar_w, color = COL_HALL, edgecolor = "black", linewidth = 0.7, label = raw"Hall $\langle\varepsilon_H\rangle$")
    b_a = ax_c.bar(x_pos .+ bar_w, amb_pcts; width = bar_w, color = COL_AMB, edgecolor = "black", linewidth = 0.7, label = raw"Ambipolar $\langle\varepsilon_A\rangle$")

    for (bars, vals) in ((b_c, conv_pcts), (b_h, hall_pcts), (b_a, amb_pcts))
        for (bar, val) in zip(bars, vals)
            cx = bar.get_x() + 0.5 * bar.get_width()
            lbl_str = val < 3.0 ? @sprintf("%.1f%%", val) : @sprintf("%.0f%%", val)
            ax_c.text(cx, val + 1.6, lbl_str; ha = "center", va = "bottom", fontsize = st.annot)
        end
    end

    xtick_labels = [
        @sprintf("\$h_0 = %d\$ km\n(\$\\langle\\varepsilon_T\\rangle = %.2f\\ \\mathrm{eV}\$)", z0v, r.shares_800.total_eV)
            for (z0v, r) in zip(z0_vals, runs)
    ]
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(xtick_labels; fontsize = st.annot)
    ax_c.tick_params(axis = "y", which = "major", labelsize = st.tick)
    ax_c.set_ylim(0.0, 114.0)
    ax_c.set_ylabel("Share of Total Work at 800 km (%)"; fontsize = st.label)
    ax_c.set_title(raw"(c) Cumulative Electric-Field Work Shares at 800 km vs. $h_0$"; fontsize = st.title_sm, fontweight = "bold", pad = 8.0)
    ax_c.grid(true; axis = "y", alpha = 0.25)
    ax_c.legend(; loc = "upper center", ncol = 3, fontsize = round(st.legend * 1.06; digits = 1), framealpha = 0.94)

    fig.savefig(output_png; dpi = 300, bbox_inches = "tight", pad_inches = 0.10)
    PyPlot.plt.close(fig)
    println("Saved 3-panel benchmark figure to: ", abspath(output_png))
    return (; col, p2, ob_200, ob_400, ob_600, output_png)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_unified_tp_mhd_benchmark()
end
