# Quantitative & visual comparison of the outward surface-flux pilot
# ("surface_flux_maxwellian") against:
#   1. The 40.6M bulk-speed baseline (outputs/paper_figures_without_crustal)
#   2. The 3D multi-species BATS-R-US MHD moments (data/mars_fields_without_crustal.vts)
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/compare_pilot_surface_flux.jl

using MarsTP
using JLD2
using TOML
using StaticArrays
using LinearAlgebra
using Statistics
using Printf
using PyPlot

function _shell_area_avg_mhd_density(m_o2p, r, theta, phi, rk::Float64, lat_edges::Vector{Float64}, lon_centers::Vector{Float64})
    n_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, m_o2p.n, r, theta, phi)
    nlat = length(lat_edges) - 1
    nlon = length(lon_centers)
    lat_centers = [0.5 * (lat_edges[j] + lat_edges[j + 1]) for j in 1:nlat]
    num = 0.0
    den = 0.0
    for j in 1:nlat
        dcos = max(sind(lat_edges[j + 1]) - sind(lat_edges[j]), 1.0e-12)
        th_c = deg2rad(90.0 - clamp(lat_centers[j], -89.999, 89.999))
        for i in 1:nlon
            ph_c = mod2pi(deg2rad(lon_centers[i]))
            pos = MarsTP.TP.sph2cart(rk, th_c, ph_c)
            nv = max(Float64(n_itp(pos)), 0.0)
            num += nv * dcos
            den += dcos
        end
    end
    return (num / den) / 1.0e6 # cm^-3
end

function _shell_area_avg_tp_density(dens_m3::AbstractMatrix{<:Real}, lat_edges::Vector{Float64})
    nlat, nlon = size(dens_m3)
    num = 0.0
    den = 0.0
    for j in 1:nlat
        dcos = max(sind(lat_edges[j + 1]) - sind(lat_edges[j]), 1.0e-12)
        for i in 1:nlon
            nv = Float64(dens_m3[j, i])
            if isfinite(nv)
                num += nv * dcos
            end
            den += dcos
        end
    end
    return (num / den) / 1.0e6 # cm^-3
end

function _extract_1d_work(raw_dict)
    w_tau = Matrix{Float64}(raw_dict["weight_tau"])
    wc_num = Matrix{Float64}(raw_dict["work_conv_num"])
    wh_num = Matrix{Float64}(raw_dict["work_hall_num"])
    wa_num = Matrix{Float64}(raw_dict["work_gradpe_num"])
    wt_num = Matrix{Float64}(raw_dict["work_total_num"])
    n = size(w_tau, 1)
    wc = fill(NaN, n)
    wh = fill(NaN, n)
    wa = fill(NaN, n)
    wt = fill(NaN, n)
    for j in 1:n
        s = sum(@view w_tau[j, :])
        if s > 0.0
            wc[j] = sum(@view wc_num[j, :]) / s
            wh[j] = sum(@view wh_num[j, :]) / s
            wa[j] = sum(@view wa_num[j, :]) / s
            wt[j] = sum(@view wt_num[j, :]) / s
        end
    end
    return (; wc, wh, wa, wt)
end

function _escape_rate_from_csv(particles_csv::AbstractString)
    isfile(particles_csv) || return (; n_outer = 0, outer_rate_s1 = 0.0, mean_outer_work_eV = NaN)
    n_outer = 0
    outer_rate = 0.0
    work_sum = 0.0
    open(particles_csv, "r") do io
        eof(io) && return (; n_outer = 0, outer_rate_s1 = 0.0, mean_outer_work_eV = NaN)
        hdr = split(readline(io), ',')
        idx_w = something(findfirst(==("rate_weight_s1"), hdr), findfirst(==("weight_s1"), hdr))
        idx_st = findfirst(==("status"), hdr)
        idx_wk = findfirst(==("work_eV"), hdr)
        for line in eachline(io)
            isempty(strip(line)) && continue
            p = split(line, ',')
            if p[idx_st] == "outer"
                w = parse(Float64, p[idx_w])
                wk = parse(Float64, p[idx_wk])
                n_outer += 1
                outer_rate += w
                work_sum += w * wk
            end
        end
    end
    return (; n_outer, outer_rate_s1 = outer_rate, mean_outer_work_eV = outer_rate > 0 ? work_sum / outer_rate : NaN)
end

function run_comparison(;
        base_paper_dir::AbstractString = joinpath(pwd(), "outputs", "paper_figures_without_crustal"),
        pilot_paper_dir::AbstractString = joinpath(pwd(), "outputs", "pilot_surface_flux_without_crustal_paper"),
        pilot_chunk_dir::AbstractString = joinpath(pwd(), "outputs", "pilot_surface_flux_without_crustal_chunk_01"),
        vts_path::AbstractString = MarsTP.data_path("mars_fields_without_crustal.vts"),
        out_png::AbstractString = joinpath(pilot_paper_dir, "pilot_vs_baseline_mhd_comparison.png"),
    )
    base_jld = joinpath(base_paper_dir, "paper_analysis.jld2")
    pilot_jld = joinpath(pilot_paper_dir, "paper_analysis.jld2")
    base_alt_jld = joinpath(base_paper_dir, "altitude_energy_maps_025_050_075_100Rm.jld2")
    pilot_alt_jld = joinpath(pilot_paper_dir, "altitude_energy_maps_025_050_075_100Rm.jld2")

    for p in (base_jld, pilot_jld, base_alt_jld, pilot_alt_jld)
        isfile(p) || error("Missing required analysis file: $p")
    end

    moments = load_mhd_moments(vts_path)
    n_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, moments.O2plus.n, moments.r, moments.theta, moments.phi)
    ux_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, moments.O2plus.v[1, :, :, :], moments.r, moments.theta, moments.phi)
    uy_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, moments.O2plus.v[2, :, :, :], moments.r, moments.theta, moments.phi)
    uz_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, moments.O2plus.v[3, :, :, :], moments.r, moments.theta, moments.phi)
    t_itp  = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, moments.O2plus.t, moments.r, moments.theta, moments.phi)
    kB_over_q = MarsTP.TP.kB / MarsTP.TP.qᵢ

    # Exact 400 km total injection rate for old bulk-speed source
    A400 = calculate_surface_areas(Rm + 400.0e3, moments.theta, moments.phi)
    src400 = load_ionosphere_source(vts_path; altitude_km = 400.0)
    mass_o2p = MarsTP.TP.SpeciesDict["O2+"].m
    q400_bulk = 0.0
    q400_sflux = 0.0
    for j in 1:(length(moments.theta) - 1), k in 1:(length(moments.phi) - 1)
        th0, th1 = moments.theta[j:(j + 1)]
        ph0, ph1 = moments.phi[k:(k + 1)]
        mu = 0.5 * (cos(th0) + cos(th1))
        ph = 0.5 * (ph0 + ph1)
        er = SVector{3, Float64}(sqrt(1.0 - mu^2) * cos(ph), sqrt(1.0 - mu^2) * sin(ph), mu)
        mp = ionosphere_properties(src400, er)
        area = (Rm + 400.0e3)^2 * (cos(th0) - cos(th1)) * (ph1 - ph0)
        q400_bulk += mp.n * norm(mp.Ui) * area
        sig = mp.Ti > 0 ? sqrt(MarsTP.TP.kB * mp.Ti / mass_o2p) : 0.0
        if mp.n > 0 && sig > 0
            q400_sflux += maxwellian_surface_flux(mp.n, dot(mp.Ui, er), sig) * area
        end
    end

    # Load baseline & pilot paper_analysis.jld2
    read_paper(path) = jldopen(path, "r") do f
        (;
            n_chunks = Int(f["raw_n_chunks"]),
            fig3_n_cm3 = Float64(f["fig3_density_m3"]) / 1.0e6,
            fig3_u_kms = Vector{Float64}(f["fig3_u_bulk_ms"]) ./ 1.0e3,
            fig3_T_eV = Float64(f["fig3_T_eV"]),
            angles = Vector{Float64}(f["orbit_angles_deg"]),
            orbit_hits = Vector{Int}(f["orbit_hits"]),
            orbit_work = _extract_1d_work(f["raw_orbit25"]),
            y_Rm = Vector{Float64}(f["tail_y_Rm"]),
            tail_n_cm3 = Vector{Float64}(f["tail_density_m3"]) ./ 1.0e6,
            tail_u_kms = Matrix{Float64}(f["tail_bulk_vel_ms"]) ./ 1.0e3,
            tail_hits = Vector{Int}(f["tail_hits"]),
            tail_work = _extract_1d_work(f["raw_tail20"]),
        )
    end
    base = read_paper(base_jld)
    pilot = read_paper(pilot_jld)

    read_alt(path) = jldopen(path, "r") do f
        (;
            altitudes_km = Vector{Float64}(f["altitudes_km"]),
            lat_edges = Vector{Float64}(f["lat_edges"]),
            lon_centers = Vector{Float64}(f["lon_centers"]),
            density_m3 = Array{Float64, 3}(f["density_m3"]),
            mean_energy_tau_eV = Array{Float64, 3}(f["mean_energy_tau_eV"]),
            hist_w_tau = Array{Float64, 4}(f["hist_w_tau"]),
            hist_wc_tau = Array{Float64, 4}(f["hist_wc_tau"]),
            hist_wh_tau = Array{Float64, 4}(f["hist_wh_tau"]),
            hist_wa_tau = Array{Float64, 4}(f["hist_wa_tau"]),
            total_particles = Int(f["total_particles"]),
        )
    end
    alt_base = read_alt(base_alt_jld)
    alt_pilot = read_alt(pilot_alt_jld)

    # Global injection / escape / ROI efficiency summary
    pilot_meta = TOML.parsefile(joinpath(pilot_chunk_dir, "metadata.toml"))
    pilot_comp = TOML.parsefile(joinpath(pilot_chunk_dir, "completion.toml"))
    pilot_esc = _escape_rate_from_csv(joinpath(pilot_chunk_dir, "particles.csv"))
    pilot_n_part = Int(pilot_meta["n_particles"])
    base_n_part = 10_153 * 500 * base.n_chunks # 40,612,000

    sc = pilot_comp["status_counts"]
    println("==========================================================================")
    println("1. 400 km Injection Rate, Boundary Fate & ROI Hit Efficiency (without_crustal)")
    println("==========================================================================")
    @printf("  Old bulk-speed (n|U|A, 40.612M traj)  : Q_inj = %.3e s^-1 | orbit hits/traj = %.3f | tail hits/traj = %.3f\n",
            q400_bulk, sum(base.orbit_hits) / base_n_part, sum(base.tail_hits) / base_n_part)
    @printf("  New surface-flux (exact, %d traj): Q_inj = %.3e s^-1 | orbit hits/traj = %.3f | tail hits/traj = %.3f\n",
            pilot_n_part, Float64(pilot_meta["source_total_rate_s1"]),
            sum(pilot.orbit_hits) / pilot_n_part, sum(pilot.tail_hits) / pilot_n_part)
    @printf("    Pilot termination counts : inner=%d (%.1f%%), outer=%d (%.1f%%), time_limit=%d (%.1f%%) | elapsed=%.1f s\n",
            sc["inner"], 100.0 * sc["inner"] / pilot_n_part,
            sc["outer"], 100.0 * sc["outer"] / pilot_n_part,
            sc["time_limit"], 100.0 * sc["time_limit"] / pilot_n_part,
            Float64(pilot_comp["elapsed_s"]))
    @printf("    Pilot outer-boundary escape rate: %.3e s^-1 (%.2f%% of Q_inj) | <work>_outer = %.1f eV\n",
            pilot_esc.outer_rate_s1, 100.0 * pilot_esc.outer_rate_s1 / Float64(pilot_meta["source_total_rate_s1"]),
            pilot_esc.mean_outer_work_eV)

    # Detector at (-2, 0, 0) Rm (Fig. 3d)
    p_f3 = SVector{3, Float64}(-2.0 * Rm, 0.0, 0.0)
    mhd_f3_n = Float64(n_itp(p_f3)) / 1.0e6
    mhd_f3_u = [Float64(ux_itp(p_f3)), Float64(uy_itp(p_f3)), Float64(uz_itp(p_f3))] ./ 1.0e3
    mhd_f3_T = Float64(t_itp(p_f3)) * kB_over_q
    println("\n==========================================================================")
    println("2. Virtual Detector at (-2, 0, 0) R_M (Fig. 3d)")
    println("==========================================================================")
    @printf("  MHD        : n = %7.3f cm^-3 | U = (%6.2f, %6.2f, %6.2f) km/s | T_i = %6.2f eV\n",
            mhd_f3_n, mhd_f3_u[1], mhd_f3_u[2], mhd_f3_u[3], mhd_f3_T)
    @printf("  Old Weight : n = %7.3f cm^-3 | U = (%6.2f, %6.2f, %6.2f) km/s | T_i = %6.2f eV\n",
            base.fig3_n_cm3, base.fig3_u_kms[1], base.fig3_u_kms[2], base.fig3_u_kms[3], base.fig3_T_eV)
    @printf("  New Weight : n = %7.3f cm^-3 | U = (%6.2f, %6.2f, %6.2f) km/s | T_i = %6.2f eV\n",
            pilot.fig3_n_cm3, pilot.fig3_u_kms[1], pilot.fig3_u_kms[2], pilot.fig3_u_kms[3], pilot.fig3_T_eV)

    # Global altitude shells (Fig. 4)
    println("\n==========================================================================")
    println("3. Global Altitude Shells: 0.25, 0.50, 0.75, 1.00 R_M (Fig. 4)")
    println("==========================================================================")
    n_mhd_shell = zeros(Float64, 4)
    n_old_shell = zeros(Float64, 4)
    n_new_shell = zeros(Float64, 4)
    r_shells_Rm = Float64[0.25, 0.50, 0.75, 1.00]
    for k in 1:4
        rk = Rm + r_shells_Rm[k] * Rm
        n_mhd_shell[k] = _shell_area_avg_mhd_density(moments.O2plus, moments.r, moments.theta, moments.phi, rk, alt_base.lat_edges, alt_base.lon_centers)
        n_old_shell[k] = _shell_area_avg_tp_density(alt_base.density_m3[k, :, :], alt_base.lat_edges)
        n_new_shell[k] = _shell_area_avg_tp_density(alt_pilot.density_m3[k, :, :], alt_pilot.lat_edges)

        metrics = map((alt_base, alt_pilot)) do d
            wt = sum(@view d.hist_w_tau[k, 2:end, :, :])
            wc = sum(@view d.hist_wc_tau[k, 2:end, :, :])
            wh = sum(@view d.hist_wh_tau[k, 2:end, :, :])
            wa = sum(@view d.hist_wa_tau[k, 2:end, :, :])
            wsum = wc + wh + wa
            ve = filter(x -> isfinite(x) && x > 0.0, vec(d.mean_energy_tau_eV[k, :, :]))
            (;
                med_e = isempty(ve) ? NaN : median(ve),
                wt_eV = wt > 0 ? wsum / wt : NaN,
                c_pct = wsum != 0 ? 100.0 * wc / wsum : NaN,
                h_pct = wsum != 0 ? 100.0 * wh / wsum : NaN,
                a_pct = wsum != 0 ? 100.0 * wa / wsum : NaN,
            )
        end
        ob, np = metrics[1], metrics[2]
        @printf("  Shell %.2f R_M (%.0f km):\n", r_shells_Rm[k], alt_base.altitudes_km[k])
        @printf("    <n>_shell [cm^-3] : MHD = %7.2f | Old = %7.2f (%.2fx MHD) | New = %7.2f (%.2fx MHD)\n",
                n_mhd_shell[k], n_old_shell[k], n_old_shell[k] / n_mhd_shell[k],
                n_new_shell[k], n_new_shell[k] / n_mhd_shell[k])
        @printf("    Old (508k tr)     : med<E>=%6.1f eV | <eps_T>=%6.1f eV | C/H/A = %5.1f%% / %5.1f%% / %5.1f%%\n",
                ob.med_e, ob.wt_eV, ob.c_pct, ob.h_pct, ob.a_pct)
        @printf("    New (%.0fk tr)     : med<E>=%6.1f eV | <eps_T>=%6.1f eV | C/H/A = %5.1f%% / %5.1f%% / %5.1f%%\n",
                alt_pilot.total_particles / 1e3, np.med_e, np.wt_eV, np.c_pct, np.h_pct, np.a_pct)
    end

    # Circular orbit at r = 2.5 Rm (Fig. 5c)
    println("\n==========================================================================")
    println("4. Circular Virtual Orbit r = 2.5 R_M (Fig. 5c) Work Partition")
    println("==========================================================================")
    for ang_t in (90.0, 120.0, 150.0, 180.0, 210.0)
        ia = argmin(abs.(base.angles .- ang_t))
        @printf("  Angle %5.1f°:\n", base.angles[ia])
        @printf("    Old (40.6M): <eps_T>=%7.1f | C=%7.1f H=%6.1f A=%5.1f eV (hits=%d)\n",
                base.orbit_work.wt[ia], base.orbit_work.wc[ia], base.orbit_work.wh[ia], base.orbit_work.wa[ia], base.orbit_hits[ia])
        @printf("    New (254k) : <eps_T>=%7.1f | C=%7.1f H=%6.1f A=%5.1f eV (hits=%d)\n",
                pilot.orbit_work.wt[ia], pilot.orbit_work.wc[ia], pilot.orbit_work.wh[ia], pilot.orbit_work.wa[ia], pilot.orbit_hits[ia])
    end

    # Cross-tail transect at X = -2.0 Rm, Z = +0.75 Rm (Fig. 7)
    mhd_tail_n = [Float64(n_itp(SVector{3, Float64}(-2.0 * Rm, y * Rm, 0.75 * Rm))) / 1.0e6 for y in base.y_Rm]
    mhd_tail_ux = [Float64(ux_itp(SVector{3, Float64}(-2.0 * Rm, y * Rm, 0.75 * Rm))) / 1.0e3 for y in base.y_Rm]
    println("\n==========================================================================")
    println("5. Cross-Tail Transect X = -2.0 R_M, Z = +0.75 R_M (Fig. 7)")
    println("==========================================================================")
    for (lbl, y_t) in (("Center Y=0.0", 0.0), ("Lobe Y=-0.5", -0.5), ("Lobe Y=+0.5", +0.5))
        iy = argmin(abs.(base.y_Rm .- y_t))
        @printf("  %s R_M:\n", lbl)
        @printf("    n  [cm^-3] : MHD = %6.3f | Old = %6.3f | New = %6.3f\n",
                mhd_tail_n[iy], base.tail_n_cm3[iy], pilot.tail_n_cm3[iy])
        @printf("    Ux [km/s]  : MHD = %6.1f | Old = %6.1f | New = %6.1f\n",
                mhd_tail_ux[iy], base.tail_u_kms[1, iy], pilot.tail_u_kms[1, iy])
        @printf("    Old work   : <eps_T>=%6.1f | C=%6.1f (%5.1f%%) H=%6.1f (%5.1f%%) A=%5.1f eV\n",
                base.tail_work.wt[iy], base.tail_work.wc[iy], 100.0 * base.tail_work.wc[iy] / base.tail_work.wt[iy],
                base.tail_work.wh[iy], 100.0 * base.tail_work.wh[iy] / base.tail_work.wt[iy], base.tail_work.wa[iy])
        @printf("    New work   : <eps_T>=%6.1f | C=%6.1f (%5.1f%%) H=%6.1f (%5.1f%%) A=%5.1f eV (hits=%d)\n",
                pilot.tail_work.wt[iy], pilot.tail_work.wc[iy], 100.0 * pilot.tail_work.wc[iy] / pilot.tail_work.wt[iy],
                pilot.tail_work.wh[iy], 100.0 * pilot.tail_work.wh[iy] / pilot.tail_work.wt[iy], pilot.tail_work.wa[iy], pilot.tail_hits[iy])
    end

    # ==========================================================================
    # 2x2 Comparison Figure
    # ==========================================================================
    fig, axes = PyPlot.plt.subplots(2, 2; figsize = (13.6, 9.6), layout = "constrained")

    # (a) Shell-averaged density vs altitude (0.25 .. 1.00 Rm)
    ax_a = axes[1, 1]
    ax_a.plot(r_shells_Rm, n_mhd_shell; color = "black", marker = "o", lw = 2.4, ms = 6, label = "3D MHD")
    ax_a.plot(r_shells_Rm, n_old_shell; color = "#d62728", marker = "s", ls = "--", lw = 2.2, ms = 6, label = raw"Old Weight ($n|\vec{U}|A$)")
    ax_a.plot(r_shells_Rm, n_new_shell; color = "#1f77b4", marker = "^", ls = "-", lw = 2.4, ms = 6.5, label = "New Weight (Surface Flux)")
    ax_a.set_yscale("log")
    ax_a.set_xticks(r_shells_Rm)
    ax_a.set_xlabel(raw"Shell Altitude ($R_M$)"; fontsize = 11)
    ax_a.set_ylabel(raw"Shell-Averaged $n_{\mathrm{O}_2^+}$ ($\mathrm{cm^{-3}}$)"; fontsize = 11)
    ax_a.set_title("(a) Global Shell Density vs. 3D MHD"; fontsize = 12, fontweight = "bold")
    ax_a.grid(true; which = "both", alpha = 0.25)
    ax_a.legend(; loc = "upper right", fontsize = 9.5, framealpha = 0.92)

    # (b) Cross-tail density n(Y) vs MHD
    ax_b = axes[1, 2]
    ax_b.plot(base.y_Rm, mhd_tail_n; color = "black", lw = 2.4, label = "3D MHD")
    ax_b.plot(base.y_Rm, base.tail_n_cm3; color = "#d62728", ls = "--", lw = 2.2, label = "Old Weight (40.6M)")
    ax_b.plot(pilot.y_Rm, pilot.tail_n_cm3; color = "#1f77b4", lw = 2.2, label = "New Surface Flux (254k)")
    ax_b.set_xlim(-1.0, 1.0)
    ax_b.set_xlabel(raw"$Y_{MSE}$ ($R_M$)"; fontsize = 11)
    ax_b.set_ylabel(raw"$n_{\mathrm{O}_2^+}$ ($\mathrm{cm^{-3}}$)"; fontsize = 11)
    ax_b.set_title(raw"(b) Cross-Tail Density ($X = -2.0\,R_M, Z = +0.75\,R_M$)"; fontsize = 12, fontweight = "bold")
    ax_b.grid(true; alpha = 0.25)
    ax_b.legend(; loc = "upper right", fontsize = 9.5, framealpha = 0.92)
    f3_txt = @sprintf("(-2,0,0) Rm detector n:\nMHD: %.2f | Old: %.2f | New: %.2f cm^-3", mhd_f3_n, base.fig3_n_cm3, pilot.fig3_n_cm3)
    ax_b.text(0.03, 0.96, f3_txt; transform = ax_b.transAxes, va = "top", ha = "left", fontsize = 8.6,
              bbox = Dict("boxstyle" => "round,pad=0.3", "facecolor" => "#f8f9fa", "edgecolor" => "0.75", "alpha" => 0.92))

    # (c) Circular orbit work profiles (0 .. 215 deg)
    ax_c = axes[2, 1]
    m_orb = base.angles .<= 215.0
    for (vals_o, vals_n, clr, lbl) in (
            (base.orbit_work.wt, pilot.orbit_work.wt, "black", raw"$\langle\varepsilon_T\rangle$"),
            (base.orbit_work.wc, pilot.orbit_work.wc, "#1f77b4", raw"$\langle\varepsilon_C\rangle$"),
            (base.orbit_work.wh, pilot.orbit_work.wh, "#ff7f0e", raw"$\langle\varepsilon_H\rangle$"),
            (base.orbit_work.wa, pilot.orbit_work.wa, "#2ca02c", raw"$\langle\varepsilon_A\rangle$"),
        )
        ax_c.plot(base.angles[m_orb], vals_o[m_orb]; color = clr, ls = "--", lw = 1.9, alpha = 0.85, label = "$lbl Old")
        ax_c.plot(pilot.angles[m_orb], vals_n[m_orb]; color = clr, ls = "-", lw = 2.2, label = "$lbl New")
    end
    ax_c.axhline(0.0; color = "0.5", lw = 0.8)
    ax_c.set_xlim(0.0, 215.0)
    ax_c.set_xlabel("Orbit Angle (deg)"; fontsize = 11)
    ax_c.set_ylabel("Mean Cumulative Work (eV)"; fontsize = 11)
    ax_c.set_title(raw"(c) $r = 2.5\,R_M$ Orbit Work: Old (dashed) vs. New (solid)"; fontsize = 12, fontweight = "bold")
    ax_c.grid(true; alpha = 0.25)
    ax_c.legend(; loc = "upper left", ncol = 2, fontsize = 8.2, framealpha = 0.92)

    # (d) Cross-tail work profiles at X = -2.0 Rm, Z = +0.75 Rm
    ax_d = axes[2, 2]
    for (vals_o, vals_n, clr, lbl) in (
            (base.tail_work.wt, pilot.tail_work.wt, "black", raw"$\langle\varepsilon_T\rangle$"),
            (base.tail_work.wc, pilot.tail_work.wc, "#1f77b4", raw"$\langle\varepsilon_C\rangle$"),
            (base.tail_work.wh, pilot.tail_work.wh, "#ff7f0e", raw"$\langle\varepsilon_H\rangle$"),
            (base.tail_work.wa, pilot.tail_work.wa, "#2ca02c", raw"$\langle\varepsilon_A\rangle$"),
        )
        ax_d.plot(base.y_Rm, vals_o; color = clr, ls = "--", lw = 1.9, alpha = 0.85, label = "$lbl Old")
        ax_d.plot(pilot.y_Rm, vals_n; color = clr, ls = "-", lw = 2.2, label = "$lbl New")
    end
    ax_d.axhline(0.0; color = "0.5", lw = 0.8)
    ax_d.set_xlim(-1.0, 1.0)
    ax_d.set_xlabel(raw"$Y_{MSE}$ ($R_M$)"; fontsize = 11)
    ax_d.set_ylabel("Mean Cumulative Work (eV)"; fontsize = 11)
    ax_d.set_title(raw"(d) Cross-Tail Work ($X = -2.0\,R_M, Z = +0.75\,R_M$)"; fontsize = 12, fontweight = "bold")
    ax_d.grid(true; alpha = 0.25)
    ax_d.legend(; loc = "upper center", ncol = 2, fontsize = 8.2, framealpha = 0.92)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 250)
    PyPlot.plt.close(fig)
    println("\nSaved comparison figure to: ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
