# ==============================================================================
# Unified 1D Analytical Benchmark: Source Altitude Sensitivity & TP-MHD Discrepancy
# ==============================================================================
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/forward_tracing/tp_mhd_discrepancy_benchmark.jl
#
# Generates the 6-panel figure `examples/forward_tracing/images/tp_mhd_discrepancy_6panel.png`
# (or custom path via `TP_MHD_BENCHMARK_OUTPUT`).

using MarsTP
using StaticArrays
using LinearAlgebra
using Printf
using DelimitedFiles
using PyPlot

function run_unified_tp_mhd_benchmark(;
        N_p_weight::Int = 40_000,
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

    println("1. Running Mechanism 1: Boundary Volume-Maxwellian vs. Flux-Maxwellian Weighting...")
    c1 = benchmark_weighting_vdf(col; N_p = N_p_weight)

    println("2. Running Oblique Draped Column (B = Bx(z) x_hat + Bz0 z_hat) across z0 = 200, 300, 400 km...")
    ob_200 = trace_outflow_column(col; z0_km = 200.0, b_geometry = :oblique, e_mode = :full, sample_mode = :volume_full, N_p = N_p_col, seed = 101)
    ob_300 = trace_outflow_column(col; z0_km = 300.0, b_geometry = :oblique, e_mode = :full, sample_mode = :volume_full, N_p = N_p_col, seed = 102)
    ob_400 = trace_outflow_column(col; z0_km = 400.0, b_geometry = :oblique, e_mode = :full, sample_mode = :volume_full, N_p = N_p_col, seed = 103)
    # Also record the 2D (v_y, v_z) VDF at z_c = 600 km under convective E_C = -U x B in the draped field,
    # where upward acceleration comes entirely from the polarization drift v_y = -(U_z / Omega_c) dU_z/dz < 0:
    ob_vdf = trace_outflow_column(col; z0_km = 400.0, b_geometry = :horizontal, e_mode = :conv_only, sample_mode = :volume_full, N_p = N_p_col, seed = 105, record_vdf_600 = true)
    ob_closure = trace_outflow_column(
        col;
        z0_km = 200.0, b_geometry = :oblique, e_mode = :full, sample_mode = :flux,
        volumetric_source = true, include_gradpi = true, flr_scale = 0.08,
        N_p = 4_000, seed = 104,
    )

    println("3. Running Parallel-Field Acceleration & Progressive Closure (Missing -grad(P_i) & Volumetric Source)...")
    par_ec   = trace_outflow_column(col; z0_km = 200.0, b_geometry = :parallel, e_mode = :conv_only, sample_mode = :volume_upward, N_p = N_p_col, seed = 201)
    par_ea   = trace_outflow_column(col; z0_km = 200.0, b_geometry = :parallel, e_mode = :conv_amb,  sample_mode = :volume_upward, N_p = N_p_col, seed = 202)
    par_400  = trace_outflow_column(col; z0_km = 400.0, b_geometry = :parallel, e_mode = :conv_amb,  sample_mode = :volume_upward, N_p = N_p_col, seed = 203)
    par_full = trace_outflow_column(
        col;
        z0_km = 200.0, b_geometry = :parallel, e_mode = :full, sample_mode = :flux,
        volumetric_source = true, include_gradpi = true, N_p = N_p_col, seed = 204,
    )

    z_km = ob_200.z_km
    z_m = z_km .* 1.0e3
    idx_600 = findfirst(>=(600.0), z_km)

    println("\n=== Summary at z_c = 600 km (g = 0, matching Manuscript Eq. 5) ===")
    @printf("  MHD Exact                        : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1\n",
            column_mhd_density(col, 600.0e3) / 1.0e8,
            column_mhd_velocity(col, 600.0e3) / 1.0e3,
            column_mhd_flux(col, 600.0e3) / 1.0e11)
    @printf("  Oblique Draped z0 = 200 km       : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1 | W_C/W_H/W_A = %4.1f%% / %4.1f%% / %4.1f%%\n",
            ob_200.n_m3[idx_600] / 1.0e8, ob_200.u_z_m_s[idx_600] / 1.0e3, ob_200.flux_m2_s[idx_600] / 1.0e11,
            ob_200.shares_600.conv_pct, ob_200.shares_600.hall_pct, ob_200.shares_600.amb_pct)
    @printf("  Oblique Draped z0 = 300 km       : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1 | W_C/W_H/W_A = %4.1f%% / %4.1f%% / %4.1f%%\n",
            ob_300.n_m3[idx_600] / 1.0e8, ob_300.u_z_m_s[idx_600] / 1.0e3, ob_300.flux_m2_s[idx_600] / 1.0e11,
            ob_300.shares_600.conv_pct, ob_300.shares_600.hall_pct, ob_300.shares_600.amb_pct)
    @printf("  Oblique Draped z0 = 400 km       : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1 | W_C/W_H/W_A = %4.1f%% / %4.1f%% / %4.1f%%\n",
            ob_400.n_m3[idx_600] / 1.0e8, ob_400.u_z_m_s[idx_600] / 1.0e3, ob_400.flux_m2_s[idx_600] / 1.0e11,
            ob_400.shares_600.conv_pct, ob_400.shares_600.hall_pct, ob_400.shares_600.amb_pct)
    @printf("  Oblique Progressive Closure      : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s | Flux = %5.2f x10^11 m^-2 s^-1\n",
            ob_closure.n_m3[idx_600] / 1.0e8, ob_closure.u_z_m_s[idx_600] / 1.0e3, ob_closure.flux_m2_s[idx_600] / 1.0e11)
    @printf("  Parallel B (z0=200 km, E=0)      : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s\n",
            par_ec.n_m3[idx_600] / 1.0e8, par_ec.u_z_m_s[idx_600] / 1.0e3)
    @printf("  Parallel B (z0=200 km, E_A)      : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s\n",
            par_ea.n_m3[idx_600] / 1.0e8, par_ea.u_z_m_s[idx_600] / 1.0e3)
    @printf("  Parallel B (z0=400 km, E_A)      : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s\n",
            par_400.n_m3[idx_600] / 1.0e8, par_400.u_z_m_s[idx_600] / 1.0e3)
    @printf("  Parallel B (Full Fluid Closure)  : n = %5.2f x10^8 m^-3 | U_z = %4.2f km/s\n",
            par_full.n_m3[idx_600] / 1.0e8, par_full.u_z_m_s[idx_600] / 1.0e3)

    if save_csv_dir !== nothing
        mkpath(save_csv_dir)
        writedlm(joinpath(save_csv_dir, "case1_vdf.csv"), hcat(c1.v_km_s, c1.f_mhd, c1.f_vol, c1.f_flux), ',')
        profiles_mat = hcat(
            z_km,
            [column_mhd_density(col, z) / 1.0e8 for z in z_m],
            [column_mhd_velocity(col, z) / 1.0e3 for z in z_m],
            [column_mhd_flux(col, z) / 1.0e11 for z in z_m],
            par_ec.n_m3 ./ 1.0e8, par_ec.u_z_m_s ./ 1.0e3,
            par_ea.n_m3 ./ 1.0e8, par_ea.u_z_m_s ./ 1.0e3,
            par_400.n_m3 ./ 1.0e8, par_400.u_z_m_s ./ 1.0e3,
            par_full.n_m3 ./ 1.0e8, par_full.u_z_m_s ./ 1.0e3,
            ob_200.n_m3 ./ 1.0e8, ob_200.flux_m2_s ./ 1.0e11,
            ob_300.n_m3 ./ 1.0e8, ob_300.flux_m2_s ./ 1.0e11,
            ob_400.n_m3 ./ 1.0e8, ob_400.flux_m2_s ./ 1.0e11,
            ob_closure.n_m3 ./ 1.0e8, ob_closure.flux_m2_s ./ 1.0e11,
        )
        writedlm(joinpath(save_csv_dir, "case23_profiles.csv"), profiles_mat, ',')
        writedlm(joinpath(save_csv_dir, "case3_vdf600.csv"), ob_vdf.vdf_600, ',')
        writedlm(joinpath(save_csv_dir, "case3_vaxes.csv"), hcat(ob_vdf.vy_centers_km_s, ob_vdf.vz_centers_km_s), ',')
    end

    # ==========================================================================
    # Render 2x3 (6-Panel) Publication Figure
    # ==========================================================================
    fig, axes = PyPlot.plt.subplots(2, 3; figsize = (18.0, 10.5), layout = "constrained")

    # --------------------------------------------------------------------------
    # Panel (a): 1D Oblique Draped MHD Background & Volumetric Source Profiles
    # --------------------------------------------------------------------------
    ax_a = axes[1, 1]
    z_fine_km = collect(range(150.0, 800.0; length = 260))
    z_fine_m = z_fine_km .* 1.0e3
    n_mhd_prof = [column_mhd_density(col, z) / 1.0e8 for z in z_fine_m]
    u_mhd_prof = [column_mhd_velocity(col, z) / 1.0e3 for z in z_fine_m]
    s_net_prof = [column_source_rate(col, z) / 1.0e6 for z in z_fine_m]
    bx_prof_nT = [column_magnetic_field(col, z)[1] * 1.0e9 for z in z_fine_m]

    ax_a.fill_between(z_fine_km, 0.0, s_net_prof; color = "#f4a261", alpha = 0.28, label = raw"$S_{\mathrm{net}}(z) = P - L$ ($10^6\ \mathrm{m^{-3}s^{-1}}$)")
    ax_a.plot(z_fine_km, n_mhd_prof; color = "black", lw = 2.4, label = raw"$n_{\mathrm{MHD}}(z)$ ($10^8\ \mathrm{m^{-3}}$)")
    ax_a.plot(z_fine_km, u_mhd_prof; color = "#1f77b4", lw = 2.2, ls = "--", label = raw"$U_{z,\mathrm{MHD}}(z)$ ($\mathrm{km/s}$)")
    ax_a.plot(z_fine_km, bx_prof_nT ./ 3.0; color = "#9467bd", lw = 2.0, ls = "-.", label = raw"$B_x(z)/3$ ($\mathrm{nT}$, $B_z=5\ \mathrm{nT}$)")
    for z0_mark in (200.0, 300.0, 400.0)
        ax_a.axvline(z0_mark; color = "0.5", ls = ":", lw = 1.1)
    end
    ax_a.set_xlim(150.0, 800.0)
    ax_a.set_ylim(0.0, 10.5)
    ax_a.set_xlabel("Altitude z (km)"; fontsize = 11)
    ax_a.set_ylabel("MHD Moment / Field Magnitude"; fontsize = 11)
    ax_a.set_title(raw"(a) 1D Oblique Draped MHD Background & Source $S_{\mathrm{net}}(z)$"; fontsize = 12, fontweight = "bold")
    ax_a.grid(true; alpha = 0.25)
    ax_a.legend(; loc = "upper right", fontsize = 8.8, framealpha = 0.92)

    # --------------------------------------------------------------------------
    # Panel (b): Mechanism 1 — Boundary Weighting Bias in Residence-Time PSD
    # --------------------------------------------------------------------------
    ax_b = axes[1, 2]
    ax_b.plot(c1.v_km_s, c1.f_mhd ./ 1.0e5; color = "black", lw = 2.6, label = raw"MHD Exact Maxwellian $f_{\mathrm{MHD}}(v_z)$")
    ax_b.plot(c1.v_km_s, c1.f_vol ./ 1.0e5; color = "#d62728", lw = 2.4, ls = "--", label = raw"TP Volume Weight $w_i \propto g(\vec{V})$ (Eqs. 2–4): $\propto g/v_z$")
    ax_b.plot(c1.v_km_s, c1.f_flux ./ 1.0e5; color = "#1f77b4", lw = 2.6, ls = ":", label = raw"TP Flux Weight $w_i \propto v_z g(\vec{V})$: Exact Match")
    ax_b.set_xlim(0.0, 2.8)
    ax_b.set_ylim(0.0, 14.8)
    ax_b.set_xlabel(raw"Vertical Velocity $v_z$ (km/s)"; fontsize = 11)
    ax_b.set_ylabel(raw"Reconstructed VDF $f(v_z)$ ($10^5\ \mathrm{m^{-4}\,s}$)"; fontsize = 11)
    ax_b.set_title("(b) Boundary Weighting Bias in Residence-Time PSD (Eq. 6)"; fontsize = 12, fontweight = "bold")
    ax_b.grid(true; alpha = 0.25)
    ax_b.legend(; loc = "upper right", fontsize = 8.8, framealpha = 0.92)

    # --------------------------------------------------------------------------
    # Panel (c): Mechanism 2 — Parallel Field & Progressive Bulk Speed Closure
    # --------------------------------------------------------------------------
    ax_c = axes[1, 3]
    ax_c.plot(z_fine_km, u_mhd_prof; color = "black", lw = 2.8, label = raw"MHD Exact $U_{z,\mathrm{MHD}}(z)$")
    m200 = z_km .>= 200.0
    m400 = z_km .>= 400.0
    ax_c.plot(z_km[m200], par_ec.u_z_m_s[m200] ./ 1.0e3; color = "#1f77b4", lw = 2.2, ls = "--", label = raw"TP $z_0=200\ \mathrm{km}$ ($E_\parallel = 0$, no $-\nabla P$)")
    ax_c.plot(z_km[m200], par_ea.u_z_m_s[m200] ./ 1.0e3; color = "#ff7f0e", lw = 2.2, label = raw"TP $z_0=200\ \mathrm{km}$ (with $\vec{E}_A$, missing $-\nabla P_i$)")
    ax_c.plot(z_km[m400], par_400.u_z_m_s[m400] ./ 1.0e3; color = "#2ca02c", lw = 2.2, label = raw"TP $z_0=400\ \mathrm{km}$ (with $\vec{E}_A$, missing $-\nabla P_i$)")
    ax_c.plot(z_km[m200], par_full.u_z_m_s[m200] ./ 1.0e3; color = "#d62728", lw = 2.6, ls = ":", label = raw"TP Volumetric $S_{\mathrm{net}}$ + Full Fluid Force ($-\nabla P_i$)")
    ax_c.set_xlim(150.0, 800.0)
    ax_c.set_ylim(0.0, 4.5)
    ax_c.set_xlabel("Altitude z (km)"; fontsize = 11)
    ax_c.set_ylabel(raw"Bulk Outflow Speed $U_z$ (km/s)"; fontsize = 11)
    ax_c.set_title(raw"(c) Parallel $\vec{B}$: Missing $-\nabla P_i$ Closure vs. Full Fluid Force"; fontsize = 12, fontweight = "bold")
    ax_c.grid(true; alpha = 0.25)
    ax_c.legend(; loc = "upper left", fontsize = 8.6, framealpha = 0.92)

    # --------------------------------------------------------------------------
    # Panel (d): Mechanism 3 — Oblique Draped Field Density n(z) & z0 Sensitivity
    # --------------------------------------------------------------------------
    ax_d = axes[2, 1]
    m300 = z_km .>= 300.0
    g200 = round(ob_200.flux_m2_s[idx_600] / 1.0e11; digits = 1)
    g300 = round(ob_300.flux_m2_s[idx_600] / 1.0e11; digits = 1)
    g400 = round(ob_400.flux_m2_s[idx_600] / 1.0e11; digits = 1)
    ax_d.plot(z_fine_km, n_mhd_prof; color = "black", lw = 2.8, label = raw"MHD Exact $n_{\mathrm{MHD}}(z)$ ($\Gamma_{600}=13.8\times 10^{11}\ \mathrm{m^{-2}s^{-1}}$)")
    ax_d.plot(z_km[m200], ob_200.n_m3[m200] ./ 1.0e8; color = "#1f77b4", lw = 2.2, label = "TP \$z_0 = 200\\ \\mathrm{km}\$ (\$\\Gamma_{600} = $(g200)\\times 10^{11}\\ \\mathrm{m^{-2}s^{-1}}\$)")
    ax_d.plot(z_km[m300], ob_300.n_m3[m300] ./ 1.0e8; color = "#ff7f0e", lw = 2.2, label = "TP \$z_0 = 300\\ \\mathrm{km}\$ (\$\\Gamma_{600} = $(g300)\\times 10^{11}\\ \\mathrm{m^{-2}s^{-1}}\$)")
    ax_d.plot(z_km[m400], ob_400.n_m3[m400] ./ 1.0e8; color = "#2ca02c", lw = 2.2, label = "TP \$z_0 = 400\\ \\mathrm{km}\$ (\$\\Gamma_{600} = $(g400)\\times 10^{11}\\ \\mathrm{m^{-2}s^{-1}}\$)")
    ax_d.plot(z_km[m200], ob_closure.n_m3[m200] ./ 1.0e8; color = "#d62728", lw = 2.6, ls = ":", label = raw"Progressive Closure ($S_{\mathrm{net}}$ + Flux Wt + Small-$r_L$)")
    ax_d.set_xlim(150.0, 800.0)
    ax_d.set_ylim(0.0, 10.5)
    ax_d.set_xlabel("Altitude z (km)"; fontsize = 11)
    ax_d.set_ylabel(raw"Ion Density $n$ ($10^8\ \mathrm{m^{-3}}$)"; fontsize = 11)
    ax_d.set_title(raw"(d) Oblique Draped $\vec{B}$: FLR Loss, $z_0$ Sensitivity & Closure"; fontsize = 12, fontweight = "bold")
    ax_d.grid(true; alpha = 0.25)
    ax_d.legend(; loc = "upper right", fontsize = 8.4, framealpha = 0.92)

    # --------------------------------------------------------------------------
    # Panel (e): 2D Non-Maxwellian VDF at z_c = 600 km (Polarization Drift & Ring-Beam)
    # --------------------------------------------------------------------------
    ax_e = axes[2, 2]
    vdf_max = maximum(ob_vdf.vdf_600)
    vdf_log = [v > 0 ? log10(max(v / vdf_max, 1.0e-3)) : -3.0 for v in ob_vdf.vdf_600]
    # Note: ob_vdf.vdf_600 has rows = vy, cols = vz; transpose so x-axis = vy, y-axis = vz
    im = ax_e.pcolormesh(
        ob_vdf.vy_centers_km_s,
        ob_vdf.vz_centers_km_s,
        permutedims(vdf_log);
        cmap = "viridis",
        vmin = -2.5,
        vmax = 0.0,
        shading = "auto",
    )
    ef_600 = column_electric_fields(col, 600.0e3)
    uz_600_km = ef_600.U_mhd[3] / 1.0e3
    vth_km = column_vth(col) / 1.0e3
    theta_c = range(0.0, 2.0 * π; length = 120)
    # Local MHD Maxwellian in 1D vertical outflow column is centered at (v_y = 0, v_z = U_z(600 km))
    ax_e.plot(vth_km .* cos.(theta_c), uz_600_km .+ vth_km .* sin.(theta_c); color = "white", lw = 1.8, ls = "--", label = raw"MHD Maxwellian ($1\,v_{\mathrm{th}},\,2\,v_{\mathrm{th}}$)")
    ax_e.plot(2.0 * vth_km .* cos.(theta_c), uz_600_km .+ 2.0 * vth_km .* sin.(theta_c); color = "white", lw = 1.4, ls = ":")
    ax_e.scatter([0.0], [uz_600_km]; color = "#ff3333", marker = "*", s = 130, zorder = 5, label = raw"MHD Bulk $\vec{U}_{\mathrm{MHD}}(600\ \mathrm{km})$")
    vdf_sum = sum(ob_vdf.vdf_600)
    vy_tp_mean = sum(ob_vdf.vdf_600[iy, iz] * ob_vdf.vy_centers_km_s[iy] for iy in eachindex(ob_vdf.vy_centers_km_s), iz in eachindex(ob_vdf.vz_centers_km_s)) / vdf_sum
    vz_tp_mean = sum(ob_vdf.vdf_600[iy, iz] * ob_vdf.vz_centers_km_s[iz] for iy in eachindex(ob_vdf.vy_centers_km_s), iz in eachindex(ob_vdf.vz_centers_km_s)) / vdf_sum
    ax_e.scatter([vy_tp_mean], [vz_tp_mean]; color = "#00ffff", marker = "o", s = 70, zorder = 5, label = "TP Mean Drift \$($(round(vy_tp_mean; digits=2)), $(round(vz_tp_mean; digits=2)))\\ \\mathrm{km/s}\$")
    ax_e.set_xlim(-3.2, 2.2)
    ax_e.set_ylim(0.5, 6.2)
    ax_e.set_xlabel(raw"$v_y$ (km/s)"; fontsize = 11)
    ax_e.set_ylabel(raw"$v_z$ (km/s)"; fontsize = 11)
    ax_e.set_title(raw"(e) Non-Maxwellian VDF at $z_c = 600\ \mathrm{km}$ ($z_0 = 400\ \mathrm{km}$)"; fontsize = 12, fontweight = "bold")
    ax_e.legend(; loc = "lower left", fontsize = 8.6, facecolor = "0.15", edgecolor = "white", labelcolor = "white", framealpha = 0.85)
    fig.colorbar(im; ax = ax_e, label = raw"$\log_{10}[f_{\mathrm{TP}}(v_y, v_z) / f_{\max}]$", fraction = 0.046, pad = 0.02)

    # --------------------------------------------------------------------------
    # Panel (f): Robustness of Electric-Field Work Partition (W_C, W_H, W_A) Across z0
    # --------------------------------------------------------------------------
    ax_f = axes[2, 3]
    # Plot cumulative work profiles vs altitude for z0 = 200, 300, 400 km and show share bars in inset/legend
    for (res, ls_style, lw_val, z_lbl) in (
            (ob_200, "-",  2.3, "200 km"),
            (ob_300, "--", 2.0, "300 km"),
            (ob_400, ":",  2.2, "400 km"),
        )
        mask = z_km .>= parse(Float64, first(split(z_lbl)))
        ax_f.plot(z_km[mask], res.w_conv_eV[mask]; color = "#1f77b4", ls = ls_style, lw = lw_val,
                  label = z_lbl == "200 km" ? raw"Convective $\langle W_C \rangle$" : "")
        ax_f.plot(z_km[mask], res.w_hall_eV[mask]; color = "#ff7f0e", ls = ls_style, lw = lw_val,
                  label = z_lbl == "200 km" ? raw"Hall $\langle W_H \rangle$" : "")
        ax_f.plot(z_km[mask], res.w_amb_eV[mask];  color = "#2ca02c", ls = ls_style, lw = lw_val,
                  label = z_lbl == "200 km" ? raw"Ambipolar $\langle W_A \rangle$" : "")
    end
    # Add dummy lines for z0 line styles
    ax_f.plot(Float64[], Float64[]; color = "0.3", ls = "-",  lw = 2.2, label = raw"$z_0 = 200\ \mathrm{km}$ (" * @sprintf("%.0f%% / %.0f%% / %.0f%%)", ob_200.shares_600.conv_pct, ob_200.shares_600.hall_pct, ob_200.shares_600.amb_pct))
    ax_f.plot(Float64[], Float64[]; color = "0.3", ls = "--", lw = 2.0, label = raw"$z_0 = 300\ \mathrm{km}$ (" * @sprintf("%.0f%% / %.0f%% / %.0f%%)", ob_300.shares_600.conv_pct, ob_300.shares_600.hall_pct, ob_300.shares_600.amb_pct))
    ax_f.plot(Float64[], Float64[]; color = "0.3", ls = ":",  lw = 2.2, label = raw"$z_0 = 400\ \mathrm{km}$ (" * @sprintf("%.0f%% / %.0f%% / %.0f%%)", ob_400.shares_600.conv_pct, ob_400.shares_600.hall_pct, ob_400.shares_600.amb_pct))

    ax_f.set_xlim(180.0, 800.0)
    ax_f.set_ylim(-0.05, 1.55)
    ax_f.set_xlabel("Altitude z (km)"; fontsize = 11)
    ax_f.set_ylabel("Mean Cumulative Electric Work (eV)"; fontsize = 11)
    ax_f.set_title(raw"(f) Robustness of $\vec{E}_C, \vec{E}_H, \vec{E}_A$ Work Shares to $z_0$"; fontsize = 12, fontweight = "bold")
    ax_f.grid(true; alpha = 0.25)
    ax_f.legend(; loc = "upper left", fontsize = 8.4, ncol = 1, framealpha = 0.92)

    fig.savefig(output_png; dpi = 200)
    PyPlot.plt.close(fig)
    println("Saved 6-panel benchmark figure to: ", abspath(output_png))
    return (; col, c1, ob_200, ob_300, ob_400, ob_closure, par_ec, par_ea, par_400, par_full, output_png)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_unified_tp_mhd_benchmark()
end

