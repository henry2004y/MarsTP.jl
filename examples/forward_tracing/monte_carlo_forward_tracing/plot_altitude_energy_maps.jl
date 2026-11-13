# Weight-averaged O2+ particle energy maps and electric-field work partition
# histograms at altitudes 0.25, 0.50, 0.75, and 1.00 Rm from forward Monte Carlo tracing.
#
# Accumulates 4D (altitude, energy_bin, lat, lon) residence-time weighted histograms:
#   - Residence weight:           ∑ w_i * τ_i
#   - Kinetic energy sum:         ∑ w_i * τ_i * E_i
#   - Convective work sum:        ∑ w_i * τ_i * ε_{i,C}
#   - Hall work sum:              ∑ w_i * τ_i * ε_{i,H}
#   - Ambipolar (-∇Pe) work sum:  ∑ w_i * τ_i * ε_{i,A}
#   - Total electric work sum:    ∑ w_i * τ_i * ε_{i,T}
#
# Uses r_outer = 3.0 Rm and tmax = 1000 s to capture returning particles above 1.00 Rm.
#
# Usage:
#   julia --threads=auto --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_altitude_energy_maps.jl [OUT_PNG] [VTS_PATH] [PER_CELL] [N_CHUNKS] [RECOMPUTE]

include("monte_carlo_shell.jl")

using .ShellMonteCarlo
using MarsTP
using StaticArrays
using LinearAlgebra
using JLD2
using Printf
using Statistics
using PyPlot

const Vec3 = SVector{3, Float64}
const E_CHARGE = MarsTP.TP.qᵢ
const O2P_MASS = MarsTP.TP.SpeciesDict["O2+"].m
const DEFAULT_ALTITUDES_KM = Float64[0.25, 0.50, 0.75, 1.00] .* (Rm / 1.0e3)

@inline function _lon_lat_bins(pos::Vec3, nlon::Int, nlat::Int)
    r = norm(pos)
    r > 0.0 || return 1, 1
    lat_deg = asind(clamp(pos[3] / r, -1.0, 1.0))
    lon_deg = mod(rad2deg(atan(pos[2], pos[1])), 360.0)
    ilat = clamp(floor(Int, (lat_deg + 90.0) / 180.0 * nlat) + 1, 1, nlat)
    ilon = clamp(floor(Int, lon_deg / 360.0 * nlon) + 1, 1, nlon)
    return ilat, ilon
end

@inline function _energy_bin_index(energy_eV::Float64, log10_emin::Float64, inv_dlog10_e::Float64, ne::Int)
    energy_eV > 0.0 || return 0
    idx = floor(Int, (log10(energy_eV) - log10_emin) * inv_dlog10_e) + 1
    return (1 <= idx <= ne) ? idx : 0
end

"""
    trace_particle_altitude_shells!(
        hist_w_tau, hist_e_tau, hist_wc_tau, hist_wh_tau, hist_wa_tau, hist_wt_tau,
        vr_tau_sum, hits_sum, w_flux_sum, e_flux_sum,
        x0::Vec3, v0::Vec3, rate_w::Float64,
        param, ec_itp, eh_itp, ea_itp,
        dt::Float64, nsteps::Int,
        r_inner::Float64, r_outer::Float64,
        r_targets::NTuple{4, Float64}, half_dr::Float64,
        nlon::Int, nlat::Int,
        log10_emin::Float64, inv_dlog10_e::Float64, ne::Int
    )

Fast allocation-free Boris integrator that tracks cumulative electric-field work
(`ε_C, ε_H, ε_A, ε_T` in eV) along each trajectory and accumulates 4D
`(altitude, energy_bin, lat, lon)` residence-time weighted histograms in-place.
"""
function trace_particle_altitude_shells!(
        hist_w_tau::Array{Float64, 4},
        hist_e_tau::Array{Float64, 4},
        hist_wc_tau::Array{Float64, 4},
        hist_wh_tau::Array{Float64, 4},
        hist_wa_tau::Array{Float64, 4},
        hist_wt_tau::Array{Float64, 4},
        vr_tau_sum::Array{Float64, 3},
        hits_sum::Array{Int, 3},
        w_flux_sum::Array{Float64, 3},
        e_flux_sum::Array{Float64, 3},
        x0::Vec3,
        v0::Vec3,
        rate_w::Float64,
        param,
        ec_itp,
        eh_itp,
        ea_itp,
        dt::Float64,
        nsteps::Int,
        r_inner::Float64,
        r_outer::Float64,
        r_targets::NTuple{4, Float64},
        half_dr::Float64,
        nlon::Int,
        nlat::Int,
        log10_emin::Float64,
        inv_dlog10_e::Float64,
        ne::Int,
    )
    rate_w > 0.0 || return nothing
    qm = param.q2m
    ke_factor = 0.5 * O2P_MASS / E_CHARGE
    q_over_eV = MarsTP.TP.SpeciesDict["O2+"].q / MarsTP.TP.eV # +1.0 for O2+

    x = x0
    v = v0
    t = 0.0
    r_curr = norm(x)

    # Record initial state if shell 1 is the launch shell itself
    if abs(r_curr - r_targets[1]) <= 10.0
        ilat0, ilon0 = _lon_lat_bins(x0, nlon, nlat)
        e0_eV = ke_factor * dot(v0, v0)
        @inbounds begin
            w_flux_sum[1, ilat0, ilon0] += rate_w
            e_flux_sum[1, ilat0, ilon0] += rate_w * e0_eV
        end
    end

    E = Vec3(param.E(x, t))
    B = Vec3(param.B(x, t))
    half = ShellMonteCarlo.BP.update_velocity_boris(v, E, B, -qm * dt / 4)

    r_lo_1 = r_targets[1] - half_dr; r_hi_1 = r_targets[1] + half_dr
    r_lo_2 = r_targets[2] - half_dr; r_hi_2 = r_targets[2] + half_dr
    r_lo_3 = r_targets[3] - half_dr; r_hi_3 = r_targets[3] + half_dr
    r_lo_4 = r_targets[4] - half_dr; r_hi_4 = r_targets[4] + half_dr
    r_lo_tup = (r_lo_1, r_lo_2, r_lo_3, r_lo_4)
    r_hi_tup = (r_hi_1, r_hi_2, r_hi_3, r_hi_4)

    cum_wc = 0.0
    cum_wh = 0.0
    cum_wa = 0.0
    cum_wt = 0.0

    @inbounds for step in 1:nsteps
        half = ShellMonteCarlo.BP.update_velocity_boris(half, E, B, qm * dt / 2)
        trial = x + half * dt
        f, flag = ShellMonteCarlo.sphere_stop(x, trial, r_inner, r_outer)
        if f == 0.0 && flag == 2
            break
        end
        xn = x + f * (trial - x)
        tn = (step - 1 + f) * dt
        dt_step = tn - t
        if dt_step <= 0.0
            break
        end

        r_next = norm(xn)
        query = flag == 4 ? xn * ((r_outer - 1.0e-6) / r_next) :
                flag == 2 ? xn * ((r_inner + 1.0e-6) / r_next) : xn
        En = Vec3(param.E(query, tn))
        Bn = Vec3(param.B(query, tn))
        vn = ShellMonteCarlo.BP.update_velocity_boris(half, En, Bn, qm * (f - 0.5) * dt / 2)

        # Midpoint quadrature for electric-field work increments (eV) over [t, tn]
        x_mid_step = 0.5 * (x + query)
        v_mid_step = 0.5 * (v + vn)
        E_tot_mid = 0.5 * (E + En)
        E_conv_mid = Vec3(ec_itp(x_mid_step))
        E_hall_mid = Vec3(eh_itp(x_mid_step))
        E_amb_mid  = Vec3(ea_itp(x_mid_step))

        dwc = q_over_eV * dot(E_conv_mid, v_mid_step) * dt_step
        dwh = q_over_eV * dot(E_hall_mid, v_mid_step) * dt_step
        dwa = q_over_eV * dot(E_amb_mid,  v_mid_step) * dt_step
        dwt = q_over_eV * dot(E_tot_mid,  v_mid_step) * dt_step

        r_min_seg = min(r_curr, r_next)
        r_max_seg = max(r_curr, r_next)
        dr_seg = r_next - r_curr

        # Check all 4 altitude shells for residence-time overlap and upward surface crossing
        for k in 1:4
            r_lo = r_lo_tup[k]
            r_hi = r_hi_tup[k]
            if r_max_seg >= r_lo && r_min_seg <= r_hi
                if abs(dr_seg) <= 1.0e-3
                    ds = 1.0
                    sm = 0.5
                else
                    s_a = clamp((r_lo - r_curr) / dr_seg, 0.0, 1.0)
                    s_b = clamp((r_hi - r_curr) / dr_seg, 0.0, 1.0)
                    ds = abs(s_b - s_a)
                    sm = 0.5 * (s_a + s_b)
                end
                tau = ds * dt_step
                if tau > 0.0
                    xm = x + sm * (xn - x)
                    vm = v + sm * (vn - v)
                    em_eV = ke_factor * dot(vm, vm)
                    rm_norm = norm(xm)
                    vr_m = rm_norm > 0.0 ? dot(vm, xm) / rm_norm : 0.0
                    ilat, ilon = _lon_lat_bins(xm, nlon, nlat)
                    ke = _energy_bin_index(em_eV, log10_emin, inv_dlog10_e, ne)

                    wc_m = cum_wc + sm * dwc
                    wh_m = cum_wh + sm * dwh
                    wa_m = cum_wa + sm * dwa
                    wt_m = cum_wt + sm * dwt

                    wt = rate_w * tau
                    if 1 <= ke <= ne
                        hist_w_tau[k, ke, ilat, ilon]  += wt
                        hist_e_tau[k, ke, ilat, ilon]  += wt * em_eV
                        hist_wc_tau[k, ke, ilat, ilon] += wt * wc_m
                        hist_wh_tau[k, ke, ilat, ilon] += wt * wh_m
                        hist_wa_tau[k, ke, ilat, ilon] += wt * wa_m
                        hist_wt_tau[k, ke, ilat, ilon] += wt * wt_m
                    end

                    vr_tau_sum[k, ilat, ilon] += wt * vr_m
                    hits_sum[k, ilat, ilon] += 1
                end
            end

            # Upward exact crossing of sphere r = r_targets[k]
            rk = r_targets[k]
            if r_curr < rk && r_next >= rk && dr_seg > 0.0
                sc = clamp((rk - r_curr) / dr_seg, 0.0, 1.0)
                xc = x + sc * (xn - x)
                vc = v + sc * (vn - v)
                ec_eV = ke_factor * dot(vc, vc)
                ilat_c, ilon_c = _lon_lat_bins(xc, nlon, nlat)
                w_flux_sum[k, ilat_c, ilon_c] += rate_w
                e_flux_sum[k, ilat_c, ilon_c] += rate_w * ec_eV
            end
        end

        cum_wc += dwc
        cum_wh += dwh
        cum_wa += dwa
        cum_wt += dwt

        x = xn
        v = vn
        t = tn
        r_curr = r_next
        E = En
        B = Bn
        if flag != 1
            break
        end
    end
    return nothing
end

function compute_altitude_energy_maps(;
        vts_path::AbstractString = MarsTP.data_path("mars_fields_with_crustal.vts"),
        cache_jld2::AbstractString = joinpath(pwd(), "outputs", "paper_figures_with_crustal", "altitude_energy_maps_025_050_075_100Rm.jld2"),
        source_altitude_km::Float64 = 400.0,
        altitudes_km::Vector{Float64} = copy(DEFAULT_ALTITUDES_KM),
        shell_half_width_km::Float64 = 0.025 * (Rm / 1.0e3),
        r_outer_Rm::Float64 = 3.0,
        nlon::Int = 120,
        nlat::Int = 60,
        n_energy_bins::Int = 50,
        emin_eV::Float64 = 1.0e-1,
        emax_eV::Float64 = 1.0e4,
        per_cell::Int = 50,
        cell_stride::Int = 1,
        n_chunks::Int = 1,
        tmax::Float64 = 1000.0,
        dt::Float64 = 0.1,
        base_seed::Int = 20260906,
        recompute::Bool = false,
    )
    if !recompute && isfile(cache_jld2)
        has_work_hist = jldopen(f -> haskey(f, "hist_wc_tau") && haskey(f, "energy_edges_eV") && collect(Float64, f["altitudes_km"]) ≈ altitudes_km, cache_jld2, "r")
        if has_work_hist
            println("Loading cached altitude energy & work maps from: ", cache_jld2)
            return jldopen(cache_jld2, "r") do f
                (;
                    altitudes_km = Vector{Float64}(f["altitudes_km"]),
                    shell_half_width_km = Float64(f["shell_half_width_km"]),
                    r_outer_Rm = Float64(f["r_outer_Rm"]),
                    lon_edges = Vector{Float64}(f["lon_edges"]),
                    lat_edges = Vector{Float64}(f["lat_edges"]),
                    lon_centers = Vector{Float64}(f["lon_centers"]),
                    lat_centers = Vector{Float64}(f["lat_centers"]),
                    energy_edges_eV = Vector{Float64}(f["energy_edges_eV"]),
                    energy_centers_eV = Vector{Float64}(f["energy_centers_eV"]),
                    hist_w_tau = Array{Float64, 4}(f["hist_w_tau"]),
                    hist_e_tau = Array{Float64, 4}(f["hist_e_tau"]),
                    hist_wc_tau = Array{Float64, 4}(f["hist_wc_tau"]),
                    hist_wh_tau = Array{Float64, 4}(f["hist_wh_tau"]),
                    hist_wa_tau = Array{Float64, 4}(f["hist_wa_tau"]),
                    hist_wt_tau = Array{Float64, 4}(f["hist_wt_tau"]),
                    mean_energy_tau_eV = Array{Float64, 3}(f["mean_energy_tau_eV"]),
                    mean_energy_flux_eV = Array{Float64, 3}(f["mean_energy_flux_eV"]),
                    mean_w_conv_eV = Array{Float64, 3}(f["mean_w_conv_eV"]),
                    mean_w_hall_eV = Array{Float64, 3}(f["mean_w_hall_eV"]),
                    mean_w_amb_eV = Array{Float64, 3}(f["mean_w_amb_eV"]),
                    mean_w_tot_eV = Array{Float64, 3}(f["mean_w_tot_eV"]),
                    mhd_mean_energy_eV = Array{Float64, 3}(f["mhd_mean_energy_eV"]),
                    density_m3 = Array{Float64, 3}(f["density_m3"]),
                    mean_vr_ms = Array{Float64, 3}(f["mean_vr_ms"]),
                    hits = Array{Int, 3}(f["hits"]),
                    total_particles = Int(f["total_particles"]),
                    vts_file = String(f["vts_file"]),
                )
            end
        end
    end

    length(altitudes_km) == 4 || error("Expected 4 target altitudes, got $(length(altitudes_km))")
    resolved_vts = MarsTP.resolve_project_path(vts_path)
    isfile(resolved_vts) || error("VTS file not found: $resolved_vts")

    cfg0 = ShellMonteCarlo.Config(;
        altitude_km = source_altitude_km,
        absorption_altitude_km = 200.0,
        cell_stride,
        per_cell,
        seed = base_seed,
        random_seed = false,
        tmax,
        dt,
        fields_path = resolved_vts,
    )

    println("=== Loading MHD fields & work interpolators from $resolved_vts ===")
    fields_raw = load_mhd_fields(resolved_vts)
    fields, source, _ = ShellMonteCarlo.validate_geometry(fields_raw.path, fields_raw, cfg0)
    param = MarsTP.mhd_param(fields; species = "O2+")

    component_itp(which::Symbol) = begin
        f = load_mhd_fields(fields.path; electric_field = which)
        MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, f.E, fields.r, fields.theta, fields.phi)
    end
    ec_itp = component_itp(:conv)
    eh_itp = component_itp(:hall)
    ea_itp = component_itp(:gradpe)

    r_inner = Rm + cfg0.absorption_altitude_km * 1.0e3
    max_alt_km = maximum(altitudes_km)
    r_outer = min(Router, max(r_outer_Rm * Rm, Rm + (max_alt_km + 350.0) * 1.0e3))
    r_targets = ntuple(k -> Rm + altitudes_km[k] * 1.0e3, 4)
    half_dr = shell_half_width_km * 1.0e3
    nsteps = round(Int, tmax / dt)

    log10_emin = log10(emin_eV)
    log10_emax = log10(emax_eV)
    energy_edges_eV = 10.0 .^ collect(range(log10_emin, log10_emax; length = n_energy_bins + 1))
    energy_centers_eV = sqrt.(energy_edges_eV[1:(end - 1)] .* energy_edges_eV[2:end])
    inv_dlog10_e = n_energy_bins / (log10_emax - log10_emin)

    nt = Threads.nthreads()
    hist_w_workers  = [zeros(Float64, 4, n_energy_bins, nlat, nlon) for _ in 1:nt]
    hist_e_workers  = [zeros(Float64, 4, n_energy_bins, nlat, nlon) for _ in 1:nt]
    hist_wc_workers = [zeros(Float64, 4, n_energy_bins, nlat, nlon) for _ in 1:nt]
    hist_wh_workers = [zeros(Float64, 4, n_energy_bins, nlat, nlon) for _ in 1:nt]
    hist_wa_workers = [zeros(Float64, 4, n_energy_bins, nlat, nlon) for _ in 1:nt]
    hist_wt_workers = [zeros(Float64, 4, n_energy_bins, nlat, nlon) for _ in 1:nt]

    vr_tau_workers = [zeros(Float64, 4, nlat, nlon) for _ in 1:nt]
    hits_workers   = [zeros(Int, 4, nlat, nlon) for _ in 1:nt]
    w_flux_workers = [zeros(Float64, 4, nlat, nlon) for _ in 1:nt]
    e_flux_workers = [zeros(Float64, 4, nlat, nlon) for _ in 1:nt]

    total_particles = 0
    for chunk_id in 1:n_chunks
        chunk_seed = base_seed + (chunk_id - 1) * 100_003
        cfg_k = ShellMonteCarlo.Config(;
            altitude_km = source_altitude_km,
            absorption_altitude_km = 200.0,
            cell_stride,
            per_cell,
            seed = chunk_seed,
            random_seed = false,
            tmax,
            dt,
            fields_path = resolved_vts,
            chunk_id,
        )
        particles, cells = ShellMonteCarlo.release_particles(fields, source, cfg_k, chunk_seed)
        n_part = length(particles)
        total_particles += n_part
        println("=== Tracing chunk $chunk_id/$n_chunks ($n_part particles across $(length(cells)) cells, $nt threads, r_outer=$(round(r_outer/Rm; digits=2)) Rm / $(round((r_outer - Rm)/1e3; digits=0)) km, tmax=$(Int(round(tmax))) s) ===")
        flush(stdout)

        t_chunk = time()
        n_done = Threads.Atomic{Int}(0)
        chunk_size = cld(n_part, nt)

        Threads.@threads :static for tid in 1:nt
            q_start = (tid - 1) * chunk_size + 1
            q_end = min(n_part, tid * chunk_size)
            hw  = hist_w_workers[tid]
            he  = hist_e_workers[tid]
            hwc = hist_wc_workers[tid]
            hwh = hist_wh_workers[tid]
            hwa = hist_wa_workers[tid]
            hwt = hist_wt_workers[tid]
            vrt_arr = vr_tau_workers[tid]
            h_arr   = hits_workers[tid]
            wf_arr  = w_flux_workers[tid]
            ef_arr  = e_flux_workers[tid]

            for q in q_start:q_end
                p = particles[q]
                if p.W > 0.0
                    trace_particle_altitude_shells!(
                        hw, he, hwc, hwh, hwa, hwt,
                        vrt_arr, h_arr, wf_arr, ef_arr,
                        p.x, p.v, Float64(p.W),
                        param, ec_itp, eh_itp, ea_itp,
                        dt, nsteps, r_inner, r_outer,
                        r_targets, half_dr, nlon, nlat,
                        log10_emin, inv_dlog10_e, n_energy_bins,
                    )
                end
                done_now = Threads.atomic_add!(n_done, 1) + 1
                if done_now % 200_000 == 0 || done_now == n_part
                    @printf("  Chunk %d/%d progress: %d / %d (%.1f%%) | elapsed = %.1f s\n",
                            chunk_id, n_chunks, done_now, n_part, 100.0 * done_now / n_part, time() - t_chunk)
                    flush(stdout)
                end
            end
        end
    end

    hist_w_tau  = hist_w_workers[1]
    hist_e_tau  = hist_e_workers[1]
    hist_wc_tau = hist_wc_workers[1]
    hist_wh_tau = hist_wh_workers[1]
    hist_wa_tau = hist_wa_workers[1]
    hist_wt_tau = hist_wt_workers[1]
    vr_tau_tot  = vr_tau_workers[1]
    hits_tot    = hits_workers[1]
    w_flux_tot  = w_flux_workers[1]
    e_flux_tot  = e_flux_workers[1]

    for tid in 2:nt
        hist_w_tau  .+= hist_w_workers[tid]
        hist_e_tau  .+= hist_e_workers[tid]
        hist_wc_tau .+= hist_wc_workers[tid]
        hist_wh_tau .+= hist_wh_workers[tid]
        hist_wa_tau .+= hist_wa_workers[tid]
        hist_wt_tau .+= hist_wt_workers[tid]
        vr_tau_tot  .+= vr_tau_workers[tid]
        hits_tot    .+= hits_workers[tid]
        w_flux_tot  .+= w_flux_workers[tid]
        e_flux_tot  .+= e_flux_workers[tid]
    end

    lon_edges = collect(range(0.0, 360.0; length = nlon + 1))
    lat_edges = collect(range(-90.0, 90.0; length = nlat + 1))
    lon_centers = [0.5 * (lon_edges[i] + lon_edges[i + 1]) for i in 1:nlon]
    lat_centers = [0.5 * (lat_edges[j] + lat_edges[j + 1]) for j in 1:nlat]

    mean_energy_tau_eV  = fill(NaN, 4, nlat, nlon)
    mean_energy_flux_eV = fill(NaN, 4, nlat, nlon)
    mean_w_conv_eV      = fill(NaN, 4, nlat, nlon)
    mean_w_hall_eV      = fill(NaN, 4, nlat, nlon)
    mean_w_amb_eV       = fill(NaN, 4, nlat, nlon)
    mean_w_tot_eV       = fill(NaN, 4, nlat, nlon)
    density_m3          = zeros(Float64, 4, nlat, nlon)
    mean_vr_ms          = fill(NaN, 4, nlat, nlon)
    mhd_mean_energy_eV  = fill(NaN, 4, nlat, nlon)

    dphi = deg2rad(360.0 / nlon)
    scale_chunks = 1.0 / max(n_chunks, 1)

    for k in 1:4
        rk = r_targets[k]
        r_lo = rk - half_dr
        r_hi = rk + half_dr
        shell_r3 = (r_hi^3 - r_lo^3) / 3.0
        src_k = MarsTP.IonosphereSource(source.n, source.temperature, source.velocity, rk)

        for ilat in 1:nlat
            th_lo = deg2rad(90.0 - lat_edges[ilat + 1])
            th_hi = deg2rad(90.0 - lat_edges[ilat])
            dcos = max(cos(th_lo) - cos(th_hi), 1.0e-12)
            vol_cell = shell_r3 * dcos * dphi

            lat_c = lat_centers[ilat]
            th_c = deg2rad(90.0 - clamp(lat_c, -89.999, 89.999))

            for ilon in 1:nlon
                wt = sum(@view hist_w_tau[k, :, ilat, ilon])
                if wt > 0.0
                    mean_energy_tau_eV[k, ilat, ilon] = sum(@view hist_e_tau[k, :, ilat, ilon]) / wt
                    mean_w_conv_eV[k, ilat, ilon]     = sum(@view hist_wc_tau[k, :, ilat, ilon]) / wt
                    mean_w_hall_eV[k, ilat, ilon]     = sum(@view hist_wh_tau[k, :, ilat, ilon]) / wt
                    mean_w_amb_eV[k, ilat, ilon]      = sum(@view hist_wa_tau[k, :, ilat, ilon]) / wt
                    mean_w_tot_eV[k, ilat, ilon]      = sum(@view hist_wt_tau[k, :, ilat, ilon]) / wt
                    mean_vr_ms[k, ilat, ilon]         = vr_tau_tot[k, ilat, ilon] / wt
                    density_m3[k, ilat, ilon]         = (wt * scale_chunks) / vol_cell
                end
                wf = w_flux_tot[k, ilat, ilon]
                if wf > 0.0
                    mean_energy_flux_eV[k, ilat, ilon] = e_flux_tot[k, ilat, ilon] / wf
                end

                ph_c = mod2pi(deg2rad(lon_centers[ilon]))
                pos_c = MarsTP.TP.sph2cart(rk, th_c, ph_c)
                p_mhd = ionosphere_properties(src_k, pos_c)
                u2 = dot(p_mhd.Ui, p_mhd.Ui)
                e_bulk_eV = 0.5 * O2P_MASS * u2 / E_CHARGE
                e_therm_eV = 1.5 * MarsTP.TP.kB * max(p_mhd.Ti, 0.0) / E_CHARGE
                mhd_mean_energy_eV[k, ilat, ilon] = e_bulk_eV + e_therm_eV
            end
        end
    end

    mkpath(dirname(abspath(cache_jld2)))
    jldopen(cache_jld2, "w") do f
        f["altitudes_km"]        = altitudes_km
        f["shell_half_width_km"] = shell_half_width_km
        f["r_outer_Rm"]          = r_outer / Rm
        f["lon_edges"]           = lon_edges
        f["lat_edges"]           = lat_edges
        f["lon_centers"]         = lon_centers
        f["lat_centers"]         = lat_centers
        f["energy_edges_eV"]     = energy_edges_eV
        f["energy_centers_eV"]   = energy_centers_eV
        f["hist_w_tau"]          = hist_w_tau
        f["hist_e_tau"]          = hist_e_tau
        f["hist_wc_tau"]         = hist_wc_tau
        f["hist_wh_tau"]         = hist_wh_tau
        f["hist_wa_tau"]         = hist_wa_tau
        f["hist_wt_tau"]         = hist_wt_tau
        f["mean_energy_tau_eV"]  = mean_energy_tau_eV
        f["mean_energy_flux_eV"] = mean_energy_flux_eV
        f["mean_w_conv_eV"]      = mean_w_conv_eV
        f["mean_w_hall_eV"]      = mean_w_hall_eV
        f["mean_w_amb_eV"]       = mean_w_amb_eV
        f["mean_w_tot_eV"]       = mean_w_tot_eV
        f["mhd_mean_energy_eV"]  = mhd_mean_energy_eV
        f["density_m3"]          = density_m3
        f["mean_vr_ms"]          = mean_vr_ms
        f["hits"]                = hits_tot
        f["total_particles"]     = total_particles
        f["vts_file"]            = basename(resolved_vts)
    end
    println("Saved altitude energy & work histogram dataset to: ", abspath(cache_jld2))

    return (;
        altitudes_km,
        shell_half_width_km,
        r_outer_Rm = r_outer / Rm,
        lon_edges,
        lat_edges,
        lon_centers,
        lat_centers,
        energy_edges_eV,
        energy_centers_eV,
        hist_w_tau,
        hist_e_tau,
        hist_wc_tau,
        hist_wh_tau,
        hist_wa_tau,
        hist_wt_tau,
        mean_energy_tau_eV,
        mean_energy_flux_eV,
        mean_w_conv_eV,
        mean_w_hall_eV,
        mean_w_amb_eV,
        mean_w_tot_eV,
        mhd_mean_energy_eV,
        density_m3,
        mean_vr_ms,
        hits = hits_tot,
        total_particles,
        vts_file = basename(resolved_vts),
    )
end

"""
    _smooth_and_upsample_log_energy(mat::AbstractMatrix{<:Real}; upsample::Int = 4)

Inpaint isolated 1-pixel statistical speckles (pixels with >= 5 valid 8-neighbors)
in `log10(E)` while preserving contiguous physical cavities, then bilinearly
interpolate (with periodic longitude wrapping) onto an `upsample ×` finer grid.
"""
function _smooth_and_upsample_log_energy(mat::AbstractMatrix{<:Real}; upsample::Int = 4)
    nlat, nlon = size(mat)
    log_m = fill(NaN, nlat, nlon)
    for j in 1:nlat, i in 1:nlon
        v = Float64(mat[j, i])
        if isfinite(v) && v > 0.0
            log_m[j, i] = log10(v)
        end
    end

    # Pass 1: Fill isolated 1-pixel NaN speckles surrounded by >= 5 valid neighbors
    filled = copy(log_m)
    for j in 1:nlat, i in 1:nlon
        if !isfinite(log_m[j, i])
            s = 0.0
            c = 0
            for dj in -1:1, di in -1:1
                (dj == 0 && di == 0) && continue
                jj = clamp(j + dj, 1, nlat)
                ii = mod1(i + di, nlon)
                v = log_m[jj, ii]
                if isfinite(v)
                    s += v
                    c += 1
                end
            end
            if c >= 5
                filled[j, i] = s / c
            end
        end
    end

    upsample <= 1 && return 10.0 .^ filled

    # Pass 2: Bilinear interpolation in log10(E) onto (upsample * nlat, upsample * nlon)
    nlat_u = nlat * upsample
    nlon_u = nlon * upsample
    out = fill(NaN, nlat_u, nlon_u)

    for ju in 1:nlat_u
        # Map fine pixel center to coarse grid index in [1, nlat]
        y = (ju - 0.5) / upsample + 0.5
        j0 = clamp(floor(Int, y), 1, nlat)
        j1 = clamp(j0 + 1, 1, nlat)
        fy = clamp(y - j0, 0.0, 1.0)

        for iu in 1:nlon_u
            x = (iu - 0.5) / upsample + 0.5
            i0 = floor(Int, x)
            fx = x - i0
            i0_w = mod1(i0, nlon)
            i1_w = mod1(i0 + 1, nlon)

            v00 = filled[j0, i0_w]
            v10 = filled[j0, i1_w]
            v01 = filled[j1, i0_w]
            v11 = filled[j1, i1_w]

            w00 = (1.0 - fy) * (1.0 - fx)
            w10 = (1.0 - fy) * fx
            w01 = fy * (1.0 - fx)
            w11 = fy * fx

            w_valid = 0.0
            s_valid = 0.0
            if isfinite(v00); w_valid += w00; s_valid += w00 * v00; end
            if isfinite(v10); w_valid += w10; s_valid += w10 * v10; end
            if isfinite(v01); w_valid += w01; s_valid += w01 * v01; end
            if isfinite(v11); w_valid += w11; s_valid += w11 * v11; end

            if w_valid >= 0.5
                out[ju, iu] = 10.0^(s_valid / w_valid)
            end
        end
    end
    return out
end

"""
    _smooth_and_upsample_symlog_work(mat::AbstractMatrix{<:Real}; upsample::Int = 4, linthresh::Float64 = 1.0)

Inpaint isolated 1-pixel statistical speckles and bilinearly upsample signed
cumulative work maps `<ε_ℓ>` in symmetric-log (`symlog`) space:
`ψ(ε) = sign(ε) * log10(1 + |ε| / linthresh)`.
"""
function _smooth_and_upsample_symlog_work(mat::AbstractMatrix{<:Real}; upsample::Int = 4, linthresh::Float64 = 1.0)
    nlat, nlon = size(mat)
    psi_m = fill(NaN, nlat, nlon)
    for j in 1:nlat, i in 1:nlon
        v = Float64(mat[j, i])
        if isfinite(v)
            psi_m[j, i] = sign(v) * log10(1.0 + abs(v) / linthresh)
        end
    end

    filled = copy(psi_m)
    for j in 1:nlat, i in 1:nlon
        if !isfinite(psi_m[j, i])
            s = 0.0
            c = 0
            for dj in -1:1, di in -1:1
                (dj == 0 && di == 0) && continue
                jj = clamp(j + dj, 1, nlat)
                ii = mod1(i + di, nlon)
                v = psi_m[jj, ii]
                if isfinite(v)
                    s += v
                    c += 1
                end
            end
            if c >= 5
                filled[j, i] = s / c
            end
        end
    end

    inv_symlog(p::Float64) = isfinite(p) ? sign(p) * linthresh * (10.0^abs(p) - 1.0) : NaN
    upsample <= 1 && return inv_symlog.(filled)

    nlat_u = nlat * upsample
    nlon_u = nlon * upsample
    out = fill(NaN, nlat_u, nlon_u)

    for ju in 1:nlat_u
        y = (ju - 0.5) / upsample + 0.5
        j0 = clamp(floor(Int, y), 1, nlat)
        j1 = clamp(j0 + 1, 1, nlat)
        fy = clamp(y - j0, 0.0, 1.0)

        for iu in 1:nlon_u
            x = (iu - 0.5) / upsample + 0.5
            i0 = floor(Int, x)
            fx = x - i0
            i0_w = mod1(i0, nlon)
            i1_w = mod1(i0 + 1, nlon)

            v00 = filled[j0, i0_w]
            v10 = filled[j0, i1_w]
            v01 = filled[j1, i0_w]
            v11 = filled[j1, i1_w]

            w00 = (1.0 - fy) * (1.0 - fx)
            w10 = (1.0 - fy) * fx
            w01 = fy * (1.0 - fx)
            w11 = fy * fx

            w_valid = 0.0
            s_valid = 0.0
            if isfinite(v00); w_valid += w00; s_valid += w00 * v00; end
            if isfinite(v10); w_valid += w10; s_valid += w10 * v10; end
            if isfinite(v01); w_valid += w01; s_valid += w01 * v01; end
            if isfinite(v11); w_valid += w11; s_valid += w11 * v11; end

            if w_valid >= 0.5
                out[ju, iu] = inv_symlog(s_valid / w_valid)
            end
        end
    end
    return out
end

function _format_altitude_label(alt_km::Real)
    alt_Rm = Float64(alt_km) * 1.0e3 / Rm
    if isapprox(alt_Rm * 4.0, round(alt_Rm * 4.0); atol = 0.02) && alt_Rm >= 0.2
        return @sprintf("%.2g R\$_M\$ (%.0f km)", alt_Rm, alt_km)
    else
        return @sprintf("%d km", Int(round(alt_km)))
    end
end

"""
    plot_altitude_work_histograms(data, out_png)

Plot the energy-binned electric-field work histograms (`<ε_C(E)>, <ε_H(E)>, <ε_A(E)>, <ε_T(E)>`)
and residence-time energy distributions at the 4 target altitudes
in a publication-quality 2×2 layout using a colorblind-safe palette (no red-green pair).
"""
function plot_altitude_work_histograms(data, out_png::AbstractString)
    ec = data.energy_centers_eV
    edges = data.energy_edges_eV
    ne = length(ec)

    # Colorblind-safe palette (avoiding red + green):
    #   Total:      Black      (#000000)
    #   Convective: Blue       (#0072B2)
    #   Hall:       Orange     (#D55E00)
    #   Ambipolar:  Purple     (#882255)
    col_tot  = "black"
    col_conv = "#0072B2"
    col_hall = "#D55E00"
    col_amb  = "#882255"

    fig, axes = plt.subplots(2, 2; figsize = (15.4, 11.2), layout = "constrained", sharex = true)
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.05, wspace = 0.04, hspace = 0.05)

    tags = ("a", "b", "c", "d")

    for k in 1:4
        row = cld(k, 2)
        col = mod1(k, 2)
        ax = axes[row, col]
        alt_km = data.altitudes_km[k]
        alt_lbl = _format_altitude_label(alt_km)

        # Sum across all (lat, lon) bins for each energy bin ke
        w_bin  = [sum(@view data.hist_w_tau[k, ke, :, :])  for ke in 1:ne]
        wc_bin = [sum(@view data.hist_wc_tau[k, ke, :, :]) for ke in 1:ne]
        wh_bin = [sum(@view data.hist_wh_tau[k, ke, :, :]) for ke in 1:ne]
        wa_bin = [sum(@view data.hist_wa_tau[k, ke, :, :]) for ke in 1:ne]
        wt_bin = [sum(@view data.hist_wt_tau[k, ke, :, :]) for ke in 1:ne]

        # Exclude legacy clamped underflow in bin 1 (E < 0.1 eV) so the PDF shows the true distribution on [0.1, 10^4] eV
        w_bin_plot = copy(w_bin)
        if length(w_bin_plot) >= 3 && w_bin_plot[1] > 1.8 * w_bin_plot[2]
            w_bin_plot[1] = 0.0
        end
        w_tot_shell = sum(w_bin_plot)
        pdf_logE = w_tot_shell > 0 ? (w_bin_plot ./ w_tot_shell) : zeros(ne)

        # Full shell-integrated ensemble work partition across ALL particles (no arbitrary E cutoff)
        w_all  = sum(@view w_bin_plot[2:end])
        wc_all = w_all > 0 ? sum(@view wc_bin[2:end]) / w_all : 0.0
        wh_all = w_all > 0 ? sum(@view wh_bin[2:end]) / w_all : 0.0
        wa_all = w_all > 0 ? sum(@view wa_bin[2:end]) / w_all : 0.0
        wt_all = wc_all + wh_all + wa_all

        # Filter bins with significant occupancy (>= 1e-3 of peak bin weight to suppress sparse 2-particle tail noise)
        min_w = 1.0e-3 * max(maximum(w_bin_plot), 1.0e-30)
        mean_wc = [ke >= 2 && w_bin[ke] >= min_w ? wc_bin[ke] / w_bin[ke] : NaN for ke in 1:ne]
        mean_wh = [ke >= 2 && w_bin[ke] >= min_w ? wh_bin[ke] / w_bin[ke] : NaN for ke in 1:ne]
        mean_wa = [ke >= 2 && w_bin[ke] >= min_w ? wa_bin[ke] / w_bin[ke] : NaN for ke in 1:ne]
        mean_wt = [ke >= 2 && w_bin[ke] >= min_w ? wt_bin[ke] / w_bin[ke] : NaN for ke in 1:ne]

        # Background shaded residence-weight distribution on twin axis
        ax_pdf = ax.twinx()
        ax_pdf.fill_between(ec, 0.0, pdf_logE; color = "0.72", alpha = 0.32, step = "mid", zorder = 1)
        ax_pdf.set_ylim(0.0, max(maximum(pdf_logE) * 1.25, 0.05))
        ax_pdf.tick_params(axis = "y", labelsize = 18.0, colors = "0.38", length = 6.5, width = 1.3)
        if col == 2
            ax_pdf.set_ylabel("Residence weight fraction"; fontsize = 20.5, color = "0.35", labelpad = 8)
        else
            ax_pdf.set_yticklabels([])
        end

        ax.set_zorder(ax_pdf.get_zorder() + 1)
        ax.patch.set_visible(false)
        ax.axhline(0.0; color = "0.45", ls = ":", lw = 1.3)

        ax.plot(ec, mean_wt; color = col_tot,  lw = 2.8, ls = "--", label = L"$\langle\varepsilon_T\rangle$ (Total)")
        ax.plot(ec, mean_wc; color = col_conv, lw = 2.6, label = L"$\langle\varepsilon_C\rangle$ (Conv)")
        ax.plot(ec, mean_wh; color = col_hall, lw = 2.6, label = L"$\langle\varepsilon_H\rangle$ (Hall)")
        ax.plot(ec, mean_wa; color = col_amb,  lw = 2.6, label = L"$\langle\varepsilon_A\rangle$ (Amb)")

        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh = 1.0)
        ax.set_xlim(first(edges), last(edges))
        ax.set_ylim(-5.0e2, 2.0e4)
        ax.set_title("($(tags[k])) Work vs. Energy at $alt_lbl"; fontsize = 21.5, pad = 9)
        col == 1 && ax.set_ylabel("Mean cumulative work (eV)"; fontsize = 21)
        row == 2 && ax.set_xlabel("Particle energy E (eV)"; fontsize = 21)
        ax.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
        ax.tick_params(axis = "x", which = "minor", length = 4.5, width = 1.1)
        ax.grid(true; which = "major", alpha = 0.28)

        if k == 1
            ax.legend(loc = "upper left", fontsize = 16.0, frameon = true, facecolor = "white", edgecolor = "0.75", framealpha = 0.94)
        end

        if alt_km > 420.0 && abs(wt_all) > 0.05
            pc = 100.0 * wc_all / wt_all
            ph = 100.0 * wh_all / wt_all
            pa = 100.0 * wa_all / wt_all
            box_txt = @sprintf("Shell ensemble mean:\n\$\\langle\\varepsilon_T\\rangle = %.1f\$ eV\nC: %.0f%% | H: %.0f%% | A: %.0f%%", wt_all, pc, ph, pa)
            loc_y, va_str = k == 1 ? (0.06, "bottom") : (0.95, "top")
            loc_x, ha_str = k == 1 ? (0.96, "right")  : (0.04, "left")
            ax.text(
                loc_x, loc_y, box_txt;
                transform = ax.transAxes, fontsize = 16.0, va = va_str, ha = ha_str,
                bbox = Dict("boxstyle" => "round,pad=0.28", "facecolor" => "white", "edgecolor" => "0.7", "alpha" => 0.93),
                zorder = 8,
            )
        else
            box_txt = @sprintf("Launch shell (\$400\\pm 25\$ km):\n\$\\langle\\varepsilon_C\\rangle = %.2f\$ eV\n\$\\langle\\varepsilon_H\\rangle = %.2f\$ eV\n\$\\langle\\varepsilon_A\\rangle = %.2f\$ eV", wc_all, wh_all, wa_all)
            ax.text(
                0.96, 0.06, box_txt;
                transform = ax.transAxes, fontsize = 15.5, va = "bottom", ha = "right",
                bbox = Dict("boxstyle" => "round,pad=0.28", "facecolor" => "white", "edgecolor" => "0.7", "alpha" => 0.93),
                zorder = 8,
            )
        end
    end

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved altitude work partition histograms to: ", abspath(out_png))
    return out_png
end

"""
    plot_altitude_electric_field_maps(
        altitudes_km::AbstractVector{<:Real},
        out_png::AbstractString;
        vts_path::AbstractString = MarsTP.data_path("mars_fields_with_crustal.vts"),
        vmin_mVm::Float64 = 1.0e-4,
        vmax_mVm::Float64 = 1.0e1,
        nlon::Int = 240,
        nlat::Int = 120,
    )

Plot a 3×4 grid of `(Longitude, Latitude)` electric-field magnitude maps (in mV/m)
across the 4 target altitudes:
  - Row 1 (a–d): Convective electric field `|E_C| = |-u × B|`
  - Row 2 (e–h): Hall electric field `|E_H| = |J × B / (e n_e)|`
  - Row 3 (i–l): Ambipolar electric field `|E_A| = |-∇P_e / (e n_e)|`
"""
function plot_altitude_electric_field_maps(
        altitudes_km::AbstractVector{<:Real},
        out_png::AbstractString;
        vts_path::AbstractString = MarsTP.data_path("mars_fields_with_crustal.vts"),
        vmin_mVm::Float64 = 1.0e-4,
        vmax_mVm::Float64 = 1.0e1,
        nlon::Int = 240,
        nlat::Int = 120,
    )
    length(altitudes_km) == 4 || error("Expected 4 altitudes, got $(length(altitudes_km))")
    resolved_vts = MarsTP.resolve_project_path(vts_path)
    isfile(resolved_vts) || error("VTS file not found: $resolved_vts")

    work_itp = build_field_work_interpolators(resolved_vts; include_extended = true)
    itps = (work_itp.conv, work_itp.hall, work_itp.gradpe)
    comp_labels = (L"$|\mathbf{E}_C|$", L"$|\mathbf{E}_H|$", L"$|\mathbf{E}_A|$")
    comp_names = ("Convective", "Hall", "Ambipolar")

    lon_edges = collect(range(0.0, 360.0; length = nlon + 1))
    lat_edges = collect(range(-90.0, 90.0; length = nlat + 1))
    lon_centers = 0.5 .* (lon_edges[1:(end - 1)] .+ lon_edges[2:end])
    lat_centers = 0.5 .* (lat_edges[1:(end - 1)] .+ lat_edges[2:end])

    # Evaluate |E_C|, |E_H|, |E_A| (in mV/m) on each spherical shell
    efield_maps = zeros(Float64, 3, 4, nlat, nlon)
    for k in 1:4
        r_m = Rm + Float64(altitudes_km[k]) * 1.0e3
        for (ilat, lat_deg) in enumerate(lat_centers)
            clat = cosd(lat_deg)
            slat = sind(lat_deg)
            for (ilon, lon_deg) in enumerate(lon_centers)
                pos = Vec3(r_m * clat * cosd(lon_deg), r_m * clat * sind(lon_deg), r_m * slat)
                for c in 1:3
                    ev_mVm = Vec3(itps[c](pos)) * 1.0e3
                    efield_maps[c, k, ilat, ilon] = max(norm(ev_mVm), 1.0e-12)
                end
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

    fig, axes = plt.subplots(3, 4; figsize = (22.4, 12.2))
    fig.subplots_adjust(left = 0.062, right = 0.905, bottom = 0.085, top = 0.935, wspace = 0.115, hspace = 0.185)

    norm_c = matplotlib.colors.LogNorm(vmin = vmin_mVm, vmax = vmax_mVm)
    cmap = matplotlib.cm.get_cmap("turbo").copy()
    cmap.set_bad(color = "0.15")

    tags = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
    im_last = nothing

    for row in 1:3, col in 1:4
        idx = (row - 1) * 4 + col
        ax = axes[row, col]
        alt_km = Float64(altitudes_km[col])
        alt_lbl = _format_altitude_label(alt_km)
        raw_mat = efield_maps[row, col, :, :]
        mat_disp = clamp.(raw_mat, vmin_mVm, vmax_mVm)

        im_last = ax.imshow(
            mat_disp;
            extent = [0.0, 360.0, -90.0, 90.0],
            origin = "lower",
            aspect = "auto",
            cmap = cmap,
            norm = norm_c,
            interpolation = "bicubic",
            rasterized = true,
        )

        # Subsolar marker at (0°, 0°)
        ax.scatter([0.0], [0.0]; marker = "*", s = 170, color = "white", edgecolors = "black", linewidths = 1.1, zorder = 6, clip_on = false)
        if row == 1 && col == 1
            ax.text(
                10.0, 18.0, "Subsolar";
                fontsize = 15.5, va = "center", ha = "left",
                bbox = Dict("boxstyle" => "square,pad=0.18", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.90),
                zorder = 6,
            )
        end

        vals = vec(raw_mat)
        e_med = median(vals)
        e_p90 = quantile(vals, 0.90)
        stat_str = @sprintf("med = %.2g | P90 = %.2g mV/m", e_med, e_p90)
        ax.text(
            0.98, 0.05, stat_str;
            transform = ax.transAxes, fontsize = 13.5, color = "white",
            va = "bottom", ha = "right",
            bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.68),
            zorder = 7,
        )

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:60:360)
        ax.set_yticks(-90:30:90)
        row < 3 && ax.set_xticklabels([])
        col > 1 && ax.set_yticklabels([])
        ax.tick_params(axis = "both", which = "major", labelsize = 17.5, length = 7.0, width = 1.4)
        ax.set_title("($(tags[idx])) $(comp_labels[row]) at $alt_lbl"; fontsize = 19.0, pad = 8)
        col == 1 && ax.set_ylabel("$(comp_names[row])\nLatitude (°)"; fontsize = 19.5)
        row == 3 && ax.set_xlabel("Longitude (°)"; fontsize = 20.0)
    end

    pos_top = axes[1, 4].get_position()
    pos_bot = axes[3, 4].get_position()
    cax = fig.add_axes([pos_top.x1 + 0.012, pos_bot.y0, 0.014, pos_top.y1 - pos_bot.y0])
    cb = fig.colorbar(im_last; cax = cax)
    cb.ax.tick_params(which = "major", labelsize = 18.5, length = 6.5, width = 1.4)
    cb.ax.tick_params(which = "minor", length = 3.8, width = 1.1)
    cb.set_label(L"Electric field magnitude $|\mathbf{E}|$ (mV/m)"; fontsize = 21.0, labelpad = 10)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved altitude electric-field maps to: ", abspath(out_png))
    return out_png
end

"""
    plot_altitude_work_maps(
        data,
        out_png::AbstractString;
        vmin_eV::Float64 = -1.0e3,
        vmax_eV::Float64 = 1.0e4,
        linthresh_eV::Float64 = 1.0,
    )

Plot a 3×4 grid of `(Longitude, Latitude)` residence-time weighted cumulative
electric-field work maps (in eV) across the 4 target altitudes:
  - Row 1 (a–d): Convective work `<ε_C>`
  - Row 2 (e–h): Hall work `<ε_H>`
  - Row 3 (i–l): Ambipolar (`-∇P_e`) work `<ε_A>`
Uses a symmetric logarithmic (`SymLogNorm`) colorbar to show both positive
acceleration (`> 0 eV`) and negative deceleration (`< 0 eV`).
"""
function plot_altitude_work_maps(
        data,
        out_png::AbstractString;
        vmin_eV::Float64 = -1.0e3,
        vmax_eV::Float64 = 1.0e4,
        linthresh_eV::Float64 = 1.0,
    )
    work_cubes = (data.mean_w_conv_eV, data.mean_w_hall_eV, data.mean_w_amb_eV)
    hist_cubes = (data.hist_wc_tau, data.hist_wh_tau, data.hist_wa_tau)
    comp_labels = (L"$\langle\varepsilon_C\rangle$", L"$\langle\varepsilon_H\rangle$", L"$\langle\varepsilon_A\rangle$")
    comp_names = ("Convective", "Hall", "Ambipolar")

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 17.5
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.0
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4

    fig, axes = plt.subplots(3, 4; figsize = (22.4, 12.2))
    fig.subplots_adjust(left = 0.062, right = 0.905, bottom = 0.085, top = 0.935, wspace = 0.115, hspace = 0.185)

    norm_c = matplotlib.colors.SymLogNorm(
        linthresh = linthresh_eV,
        linscale = 0.5,
        vmin = vmin_eV,
        vmax = vmax_eV,
        base = 10,
    )
    cmap = matplotlib.cm.get_cmap("turbo").copy()
    cmap.set_bad(color = "0.15")

    tags = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
    im_last = nothing

    for row in 1:3, col in 1:4
        idx = (row - 1) * 4 + col
        ax = axes[row, col]
        alt_km = Float64(data.altitudes_km[col])
        alt_lbl = _format_altitude_label(alt_km)
        raw_mat = work_cubes[row][col, :, :]
        mat_smooth = _smooth_and_upsample_symlog_work(raw_mat; upsample = 4, linthresh = linthresh_eV)
        for i in eachindex(mat_smooth)
            if isfinite(mat_smooth[i])
                mat_smooth[i] = clamp(mat_smooth[i], vmin_eV, vmax_eV)
            else
                mat_smooth[i] = NaN
            end
        end

        im_last = ax.imshow(
            mat_smooth;
            extent = [0.0, 360.0, -90.0, 90.0],
            origin = "lower",
            aspect = "auto",
            cmap = cmap,
            norm = norm_c,
            interpolation = "bicubic",
            rasterized = true,
        )

        # Subsolar marker at (0°, 0°)
        ax.scatter([0.0], [0.0]; marker = "*", s = 170, color = "white", edgecolors = "black", linewidths = 1.1, zorder = 6, clip_on = false)
        if row == 1 && col == 1
            ax.text(
                10.0, 18.0, "Subsolar";
                fontsize = 15.5, va = "center", ha = "left",
                bbox = Dict("boxstyle" => "square,pad=0.18", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.90),
                zorder = 6,
            )
        end

        # Full shell ensemble mean for this work component (excluding underflow bin 1)
        w_tot = sum(@view data.hist_w_tau[col, 2:end, :, :])
        w_comp = sum(@view hist_cubes[row][col, 2:end, :, :])
        shell_mean = w_tot > 0.0 ? (w_comp / w_tot) : 0.0

        valid_w = filter(isfinite, vec(raw_mat))
        if !isempty(valid_w)
            w_med = median(valid_w)
            w_p90 = quantile(valid_w, 0.90)
            stat_str = @sprintf("mean=%.1f | med=%.1f, P90=%.0f eV", shell_mean, w_med, w_p90)
            ax.text(
                0.98, 0.05, stat_str;
                transform = ax.transAxes, fontsize = 13.0, color = "white",
                va = "bottom", ha = "right",
                bbox = Dict("boxstyle" => "round,pad=0.22", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.68),
                zorder = 7,
            )
        end

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:60:360)
        ax.set_yticks(-90:30:90)
        row < 3 && ax.set_xticklabels([])
        col > 1 && ax.set_yticklabels([])
        ax.tick_params(axis = "both", which = "major", labelsize = 17.5, length = 7.0, width = 1.4)
        ax.set_title("($(tags[idx])) $(comp_labels[row]) at $alt_lbl"; fontsize = 19.0, pad = 8)
        col == 1 && ax.set_ylabel("$(comp_names[row])\nLatitude (°)"; fontsize = 19.5)
        row == 3 && ax.set_xlabel("Longitude (°)"; fontsize = 20.0)
    end

    pos_top = axes[1, 4].get_position()
    pos_bot = axes[3, 4].get_position()
    cax = fig.add_axes([pos_top.x1 + 0.012, pos_bot.y0, 0.014, pos_top.y1 - pos_bot.y0])
    cb = fig.colorbar(im_last; cax = cax)
    cb.set_ticks([-1.0e3, -1.0e2, -1.0e1, -1.0e0, 0.0, 1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4])
    cb.set_ticklabels([L"$-10^3$", L"$-10^2$", L"$-10^1$", L"$-10^0$", "0", L"$10^0$", L"$10^1$", L"$10^2$", L"$10^3$", L"$10^4$"])
    cb.ax.tick_params(which = "major", labelsize = 18.0, length = 6.5, width = 1.4)
    cb.ax.tick_params(which = "minor", length = 3.5, width = 1.0)
    cb.set_label(L"Residence-weighted electric work $\langle\varepsilon_\ell\rangle$ (eV)"; fontsize = 21.0, labelpad = 10)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved altitude electric-field work maps to: ", abspath(out_png))
    return out_png
end

function plot_altitude_energy_maps(
        out_png::AbstractString = joinpath(pwd(), "outputs", "paper_figures_with_crustal", "altitude_energy_maps_025_050_075_100Rm.png");
        vts_path::AbstractString = MarsTP.data_path("mars_fields_with_crustal.vts"),
        cache_jld2::AbstractString = joinpath(dirname(abspath(out_png)), replace(basename(out_png), r"\.png$"i => ".jld2")),
        altitudes_km::Vector{Float64} = copy(DEFAULT_ALTITUDES_KM),
        shell_half_width_km::Float64 = 0.025 * (Rm / 1.0e3),
        per_cell::Int = 50,
        n_chunks::Int = 1,
        r_outer_Rm::Float64 = 3.0,
        tmax::Float64 = 1000.0,
        recompute::Bool = false,
        weighting::Symbol = :residence, # :residence (<E>_τ), :flux (<E>_flux), or :mhd
        vmin_eV::Float64 = 1.0e-1,
        vmax_eV::Float64 = 1.0e4,
    )
    data = compute_altitude_energy_maps(;
        vts_path,
        cache_jld2,
        altitudes_km,
        shell_half_width_km,
        per_cell,
        n_chunks,
        r_outer_Rm,
        tmax,
        recompute,
    )

    energy_maps = if weighting == :residence
        data.mean_energy_tau_eV
    elseif weighting == :flux
        data.mean_energy_flux_eV
    elseif weighting == :mhd
        data.mhd_mean_energy_eV
    else
        error("Unsupported weighting=$weighting; expected :residence, :flux, or :mhd")
    end

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 20
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 20
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 20
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 8.0
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 8.0
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.5
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.5

    # Use explicit subplots_adjust so we can position the colorbar to match
    # the exact vertical span [pos_bot.y0, pos_top.y1] of the 2-row panel spines
    # (excluding the top titles and bottom x-axis labels).
    fig, axes = plt.subplots(2, 2; figsize = (15.8, 9.8))
    fig.subplots_adjust(left = 0.075, right = 0.885, bottom = 0.095, top = 0.925, wspace = 0.085, hspace = 0.165)

    norm_c = matplotlib.colors.LogNorm(vmin = vmin_eV, vmax = vmax_eV)
    cmap = matplotlib.cm.get_cmap("turbo").copy()
    cmap.set_bad(color = "0.15")

    im_last = nothing
    tags = ("a", "b", "c", "d")

    for k in 1:4
        row = cld(k, 2)
        col = mod1(k, 2)
        ax = axes[row, col]
        alt_km = data.altitudes_km[k]
        alt_lbl = _format_altitude_label(alt_km)
        mat_smooth = _smooth_and_upsample_log_energy(energy_maps[k, :, :]; upsample = 4)
        for i in eachindex(mat_smooth)
            if isfinite(mat_smooth[i]) && mat_smooth[i] > 0.0
                mat_smooth[i] = clamp(mat_smooth[i], vmin_eV, vmax_eV)
            else
                mat_smooth[i] = NaN
            end
        end

        im_last = ax.imshow(
            mat_smooth;
            extent = (0.0, 360.0, -90.0, 90.0),
            origin = "lower",
            aspect = "auto",
            cmap = cmap,
            norm = norm_c,
            interpolation = "bicubic",
            rasterized = true,
        )

        # Subsolar marker at (0°, 0°)
        ax.scatter([0.0], [0.0]; marker = "*", s = 240, facecolor = "white", edgecolor = "black", lw = 1.2, zorder = 6, clip_on = false)
        if k == 1
            ax.text(
                10.0, 18.0, "Subsolar";
                fontsize = 18.0, va = "center", ha = "left",
                bbox = Dict("boxstyle" => "square,pad=0.2", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.90),
                zorder = 6,
            )
        end

        # Full shell ensemble work shares (across all energies, excluding underflow bin 1)
        wc_all = sum(@view data.hist_wc_tau[k, 2:end, :, :])
        wh_all = sum(@view data.hist_wh_tau[k, 2:end, :, :])
        wa_all = sum(@view data.hist_wa_tau[k, 2:end, :, :])
        wt_all = wc_all + wh_all + wa_all

        valid_e = filter(x -> isfinite(x) && x > 0.0, vec(energy_maps[k, :, :]))
        if !isempty(valid_e)
            e_med = median(valid_e)
            e_p90 = quantile(valid_e, 0.90)
            stat_str = if alt_km > 420.0 && abs(wt_all) > 0.0
                @sprintf("med=%.1f eV, P90=%.0f eV | C:%.0f%% H:%.0f%% A:%.0f%%",
                         e_med, e_p90, 100*wc_all/wt_all, 100*wh_all/wt_all, 100*wa_all/wt_all)
            else
                @sprintf("med = %.1f eV | P90 = %.0f eV", e_med, e_p90)
            end
            ax.text(
                0.98, 0.05, stat_str;
                transform = ax.transAxes, fontsize = 14.5, color = "white",
                va = "bottom", ha = "right",
                bbox = Dict("boxstyle" => "round,pad=0.24", "facecolor" => "black", "edgecolor" => "none", "alpha" => 0.68),
                zorder = 7,
            )
        end

        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:60:360)
        ax.set_yticks(-90:30:90)
        row == 1 && ax.set_xticklabels([])
        col == 2 && ax.set_yticklabels([])
        ax.tick_params(axis = "both", which = "major", labelsize = 20, length = 8.0, width = 1.5)
        ax.set_title("($(tags[k])) O\$_2^+\$ mean energy at $alt_lbl"; fontsize = 22.0, pad = 9)
        col == 1 && ax.set_ylabel("Latitude (°)"; fontsize = 22.5)
        row == 2 && ax.set_xlabel("Longitude (°)"; fontsize = 22.5)
    end

    pos_top = axes[1, 2].get_position()
    pos_bot = axes[2, 2].get_position()
    cax = fig.add_axes([pos_top.x1 + 0.016, pos_bot.y0, 0.018, pos_top.y1 - pos_bot.y0])
    cb = fig.colorbar(im_last; cax = cax)
    cb.ax.tick_params(which = "major", labelsize = 20, length = 6.5, width = 1.4)
    cb.ax.tick_params(which = "minor", length = 3.8, width = 1.1)
    cbar_label = weighting == :residence ?
        "Residence-weighted mean energy \$\\langle E \\rangle\$ (eV)" :
        weighting == :flux ?
        "Flux-weighted mean energy \$\\langle E \\rangle_{\\mathrm{flux}}\$ (eV)" :
        "MHD mean ion energy \$\\frac{1}{2}m U^2 + \\frac{3}{2}k_B T_i\$ (eV)"
    cb.set_label(cbar_label; fontsize = 22.5, labelpad = 10)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved altitude energy maps figure to: ", abspath(out_png))

    hist_png = replace(out_png, r"\.png$"i => "_work_histograms.png")
    plot_altitude_work_histograms(data, hist_png)
    work_maps_png = replace(out_png, r"\.png$"i => "_work_maps.png")
    plot_altitude_work_maps(data, work_maps_png)
    efield_png = replace(out_png, r"\.png$"i => "_efield_maps.png")
    plot_altitude_electric_field_maps(data.altitudes_km, efield_png; vts_path)
    return out_png, hist_png, work_maps_png, efield_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    out_png = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "outputs", "paper_figures_with_crustal", "altitude_energy_maps_025_050_075_100Rm.png")
    vts_path = length(ARGS) >= 2 ? ARGS[2] : MarsTP.data_path("mars_fields_with_crustal.vts")
    per_cell = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
    n_chunks = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    recompute = length(ARGS) >= 5 ? (lowercase(ARGS[5]) in ("true", "1", "yes")) : false

    jld_file = joinpath(dirname(abspath(out_png)), "altitude_energy_maps_025_050_075_100Rm.jld2")
    plot_altitude_energy_maps(
        out_png;
        vts_path,
        cache_jld2 = jld_file,
        altitudes_km = copy(DEFAULT_ALTITUDES_KM),
        shell_half_width_km = 0.025 * (Rm / 1.0e3),
        per_cell,
        n_chunks,
        r_outer_Rm = 3.0,
        recompute,
    )
end
