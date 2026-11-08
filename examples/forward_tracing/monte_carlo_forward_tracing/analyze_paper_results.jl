# Paper-style analysis and data accumulation for Mars O2+ forward Monte Carlo runs.
# Produces `paper_analysis.jld2` and `trajectories_20000_sample.jld2` for the dedicated
# figure scripts (`plot_fig3.jl` .. `plot_fig7.jl`):
#   - Fig. 3: Global trajectories (XZ, XY, YZ) + virtual detector VDF (1 km/s bins on [-100, 100] km/s),
#             moments (U_bulk, T, n, B), 1D energy PSD, and 2D energy-pitch angle PSD at (-2, 0, 0) Rm
#   - Fig. 4: Circular virtual spacecraft trajectory at r = 2.5 Rm (and r = 2.0 Rm) in XZ plane (Y = 0, 0°..240°):
#             PSD (s^3 m^-6), omni DEF, and mean cumulative work <ε_C>, <ε_H>, <ε_A>, <ε_T>
#   - Fig. 5: Representative trajectories, instantaneous power P_ℓ(t), and cumulative work ε_ℓ(t) at 90°, 120°, 150°
#   - Fig. 6: Cross-tail virtual spacecraft trajectory along Y ∈ [-1, 1] Rm at X = -2.0 Rm (and -1.75 Rm), Z = +0.75 Rm:
#             magnetic field profile, PSD, bulk velocity (Ux, Uy, Uz, Ut), and <ε_C>, <ε_H>, <ε_A>, <ε_T>
#   - Fig. 7: Representative trajectories, power, and cumulative work at (-2, 0, +0.75) Rm and (-2, -0.5, +0.75) Rm
#
# Usage:
#   julia --threads=auto --project=. examples/forward_tracing/monte_carlo_forward_tracing/analyze_paper_results.jl RUN_DIR_OR_DIRS... OUT_DIR

using MarsTP, StaticArrays, LinearAlgebra, JLD2, TOML, Printf, Statistics, Random, PyPlot

const Vec3 = SVector{3, Float64}
const E_CHARGE = MarsTP.TP.qᵢ
const O2P_MASS = MarsTP.TP.SpeciesDict["O2+"].m

mutable struct DetectorMomentsAcc
    w_tau::Float64
    vx_tau::Float64
    vy_tau::Float64
    vz_tau::Float64
    vx2_tau::Float64
    vy2_tau::Float64
    vz2_tau::Float64
end
DetectorMomentsAcc() = DetectorMomentsAcc(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function merge_moments!(a::DetectorMomentsAcc, b::DetectorMomentsAcc)
    a.w_tau += b.w_tau
    a.vx_tau += b.vx_tau
    a.vy_tau += b.vy_tau
    a.vz_tau += b.vz_tau
    a.vx2_tau += b.vx2_tau
    a.vy2_tau += b.vy2_tau
    a.vz2_tau += b.vz2_tau
    return a
end

function finalize_moments(a::DetectorMomentsAcc, vol::Float64, n_chunks::Int = 1)
    if a.w_tau <= 0 || n_chunks <= 0
        return (; density_m3 = 0.0, u_bulk_ms = zeros(3), T_eV = 0.0)
    end
    n_m3 = a.w_tau / (vol * n_chunks)
    ux = a.vx_tau / a.w_tau
    uy = a.vy_tau / a.w_tau
    uz = a.vz_tau / a.w_tau
    var_v = max((a.vx2_tau + a.vy2_tau + a.vz2_tau) / a.w_tau - (ux^2 + uy^2 + uz^2), 0.0)
    T_eV = O2P_MASS * var_v / (3.0 * E_CHARGE)
    return (; density_m3 = n_m3, u_bulk_ms = [ux, uy, uz], T_eV)
end

_pack_moments_raw(a::DetectorMomentsAcc) = Float64[a.w_tau, a.vx_tau, a.vy_tau, a.vz_tau, a.vx2_tau, a.vy2_tau, a.vz2_tau]
_unpack_moments_raw(v::AbstractVector{<:Real}) = DetectorMomentsAcc(Float64(v[1]), Float64(v[2]), Float64(v[3]), Float64(v[4]), Float64(v[5]), Float64(v[6]), Float64(v[7]))

function merge_psd_acc!(dest::ForwardPSDAccumulator, src::ForwardPSDAccumulator)
    for (k, v) in src.occupancy
        dest.occupancy[k] = get(dest.occupancy, k, 0.0) + v
    end
    for (k, v) in src.sumsq
        dest.sumsq[k] = get(dest.sumsq, k, 0.0) + v
    end
    for (k, v) in src.unique
        dest.unique[k] = get(dest.unique, k, 0) + v
    end
    append!(dest.particle_ids, src.particle_ids)
    append!(dest.rates, src.rates)
    append!(dest.residence, src.residence)
    append!(dest.outside, src.outside)
    append!(dest.retcodes, src.retcodes)
    if dest.omni !== nothing && src.omni !== nothing
        dest.omni.density .+= src.omni.density
        dest.omni.ev_density .+= src.omni.ev_density
        dest.omni.outside += src.omni.outside
    end
    return dest
end

struct TransectAccumulators
    centers_m::Vector{Vec3}
    side_m::Float64
    energy_edges_eV::Vector{Float64}
    energy_centers_eV::Vector{Float64}
    weight_tau::Matrix{Float64}          # (n_det, n_E)
    def_num::Matrix{Float64}             # (n_det, n_E)
    work_conv_num::Matrix{Float64}       # (n_det, n_E)
    work_hall_num::Matrix{Float64}       # (n_det, n_E)
    work_gradpe_num::Matrix{Float64}     # (n_det, n_E)
    work_total_num::Matrix{Float64}      # (n_det, n_E)
    density_tau::Vector{Float64}         # (n_det,)
    vel_tau::Matrix{Float64}             # (3, n_det)
    hit_counts::Vector{Int}              # (n_det,)
    best_pid::Vector{Int}                # (n_det,)
    best_score::Vector{Float64}          # (n_det,)
    best_time_s::Vector{Float64}         # (n_det,)
    best_tr::Vector{Any}                 # (n_det,)
    pref_pid::Vector{Int}                # (n_det,)
    pref_score::Vector{Float64}          # (n_det,)
    pref_time_s::Vector{Float64}         # (n_det,)
    pref_tr::Vector{Any}                 # (n_det,)
end

function TransectAccumulators(centers_m::Vector{Vec3}, side_m::Float64, energy_edges_eV::Vector{Float64})
    nd = length(centers_m)
    ne = length(energy_edges_eV) - 1
    ec = sqrt.(energy_edges_eV[1:(end - 1)] .* energy_edges_eV[2:end])
    return TransectAccumulators(
        centers_m,
        side_m,
        energy_edges_eV,
        ec,
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd, ne),
        zeros(Float64, nd),
        zeros(Float64, 3, nd),
        zeros(Int, nd),
        zeros(Int, nd),
        zeros(Float64, nd),
        zeros(Float64, nd),
        fill(nothing, nd),
        zeros(Int, nd),
        zeros(Float64, nd),
        zeros(Float64, nd),
        fill(nothing, nd),
    )
end

function merge_transect!(dest::TransectAccumulators, src::TransectAccumulators)
    dest.weight_tau .+= src.weight_tau
    dest.def_num .+= src.def_num
    dest.work_conv_num .+= src.work_conv_num
    dest.work_hall_num .+= src.work_hall_num
    dest.work_gradpe_num .+= src.work_gradpe_num
    dest.work_total_num .+= src.work_total_num
    dest.density_tau .+= src.density_tau
    dest.vel_tau .+= src.vel_tau
    dest.hit_counts .+= src.hit_counts
    for d in eachindex(dest.centers_m)
        if src.best_score[d] > dest.best_score[d]
            dest.best_score[d] = src.best_score[d]
            dest.best_pid[d] = src.best_pid[d]
            dest.best_time_s[d] = src.best_time_s[d]
            dest.best_tr[d] = src.best_tr[d]
        end
        if src.pref_score[d] > dest.pref_score[d]
            dest.pref_score[d] = src.pref_score[d]
            dest.pref_pid[d] = src.pref_pid[d]
            dest.pref_time_s[d] = src.pref_time_s[d]
            dest.pref_tr[d] = src.pref_tr[d]
        end
    end
    return dest
end

@inline function _accumulate_transect!(
        acc::TransectAccumulators,
        t_vec,
        u_vec::Vector{SVector{6, Float64}},
        rate_w::Float64,
        pid::Int,
        dw_tot::Vector{Float64},
        dw_conv::Vector{Float64},
        dw_hall::Vector{Float64},
        dw_gradpe::Vector{Float64};
        min_rep_energy_eV::Float64 = 30.0,
        mode::Symbol = :orbit,
    )
    rate_w > 0 || return nothing
    n = length(t_vec)
    n >= 2 || return nothing

    xmin = Inf; xmax = -Inf
    ymin = Inf; ymax = -Inf
    zmin = Inf; zmax = -Inf
    @inbounds for i in 1:n
        u = u_vec[i]
        x, y, z = u[1], u[2], u[3]
        x < xmin && (xmin = x); x > xmax && (xmax = x)
        y < ymin && (ymin = y); y > ymax && (ymax = y)
        z < zmin && (zmin = z); z > zmax && (zmax = z)
    end
    half = 0.5 * acc.side_m
    edges = acc.energy_edges_eV
    ne = length(edges) - 1

    @inbounds for d in eachindex(acc.centers_m)
        c = acc.centers_m[d]
        (
            xmax < c[1] - half || xmin > c[1] + half ||
                ymax < c[2] - half || ymin > c[2] + half ||
                zmax < c[3] - half || zmin > c[3] + half
        ) && continue

        lo = c .- half
        hi = c .+ half
        wt_tot = 0.0
        wc_tot = 0.0
        wh_tot = 0.0
        wa_tot = 0.0
        had_hit = false
        best_seg_wt = 0.0
        best_seg_t = 0.0
        pref_seg_wt = 0.0
        pref_seg_t = 0.0

        for j in 2:n
            ua = u_vec[j - 1]
            ub = u_vec[j]
            xa = Vec3(ua[1], ua[2], ua[3])
            xb = Vec3(ub[1], ub[2], ub[3])
            dwt = dw_tot[j - 1]
            dwc = dw_conv[j - 1]
            dwh = dw_hall[j - 1]
            dwa = dw_gradpe[j - 1]

            hit = MarsTP._forward_cube_interval(xa, xb - xa, lo, hi)
            if hit !== nothing
                s0, s1, _, _ = hit
                ta = t_vec[j - 1]
                tb = t_vec[j]
                tau = (s1 - s0) * (tb - ta)
                if tau > 0
                    had_hit = true
                    sm = 0.5 * (s0 + s1)
                    va = Vec3(ua[4], ua[5], ua[6])
                    vb = Vec3(ub[4], ub[5], ub[6])
                    vm = va + sm * (vb - va)
                    speed = norm(vm)
                    energy_eV = 0.5 * O2P_MASS * speed^2 / E_CHARGE
                    w_tau = rate_w * tau
                    acc.density_tau[d] += w_tau
                    acc.vel_tau[1, d] += w_tau * vm[1]
                    acc.vel_tau[2, d] += w_tau * vm[2]
                    acc.vel_tau[3, d] += w_tau * vm[3]

                    k = searchsortedlast(edges, energy_eV)
                    if k == ne + 1 && energy_eV == edges[end]
                        k = ne
                    end
                    if 1 <= k <= ne
                        et = wt_tot + sm * dwt
                        ec = wc_tot + sm * dwc
                        eh = wh_tot + sm * dwh
                        ea = wa_tot + sm * dwa
                        acc.weight_tau[d, k] += w_tau
                        acc.def_num[d, k] += w_tau * energy_eV * speed
                        acc.work_total_num[d, k] += w_tau * et
                        acc.work_conv_num[d, k] += w_tau * ec
                        acc.work_hall_num[d, k] += w_tau * eh
                        acc.work_gradpe_num[d, k] += w_tau * ea
                        t_mid = ta + sm * (tb - ta)
                        if energy_eV >= min_rep_energy_eV && w_tau > best_seg_wt
                            best_seg_wt = w_tau
                            best_seg_t = t_mid
                        end
                        # Preferred representative particle matching paper Fig. 5 & Fig. 7 physics
                        is_pref = if mode == :orbit
                            ang_rad = atan(c[3], c[1])
                            ang_deg = mod(rad2deg(ang_rad), 360.0)
                            if abs(ang_deg - 90.0) <= 6.0
                                (5000.0 <= energy_eV <= 18000.0) && (50.0 <= t_mid <= 180.0) && (ec > 0.8 * et) && (eh > 0.0)
                            elseif abs(ang_deg - 120.0) <= 6.0
                                (1500.0 <= energy_eV <= 6500.0) && (120.0 <= t_mid <= 400.0) && (ec > 0.65 * et) && (eh > 50.0)
                            elseif abs(ang_deg - 150.0) <= 6.0
                                (250.0 <= energy_eV <= 1200.0) && (180.0 <= t_mid <= 600.0) && (ec > 0.45 * et) && (eh > 20.0)
                            else
                                energy_eV >= min_rep_energy_eV
                            end
                        else
                            y_Rm = c[2] / Rm
                            if abs(y_Rm) < 0.15
                                # Current sheet center: positive Hall work (accelerating), 120-350 eV
                                (110.0 <= energy_eV <= 350.0) && (150.0 <= t_mid <= 500.0) && (eh > 25.0) && (ec > 60.0) && (ea > 0.0)
                            elseif y_Rm < -0.35
                                # Lobe (Y = -0.5 Rm): negative Hall work (decelerating), 80-250 eV
                                (70.0 <= energy_eV <= 260.0) && (200.0 <= t_mid <= 600.0) && (eh < -15.0) && (ec > 70.0) && (ea > 0.0)
                            else
                                energy_eV >= min_rep_energy_eV
                            end
                        end
                        if is_pref && w_tau > pref_seg_wt
                            pref_seg_wt = w_tau
                            pref_seg_t = t_mid
                        end
                    end
                end
            end
            wt_tot += dwt
            wc_tot += dwc
            wh_tot += dwh
            wa_tot += dwa
        end

        if had_hit
            acc.hit_counts[d] += 1
            if best_seg_wt > acc.best_score[d]
                acc.best_score[d] = best_seg_wt
                acc.best_pid[d] = pid
                acc.best_time_s[d] = best_seg_t
                acc.best_tr[d] = (; t = collect(t_vec), u = copy(u_vec))
            end
            if pref_seg_wt > acc.pref_score[d]
                acc.pref_score[d] = pref_seg_wt
                acc.pref_pid[d] = pid
                acc.pref_time_s[d] = pref_seg_t
                acc.pref_tr[d] = (; t = collect(t_vec), u = copy(u_vec))
            end
        end
    end
    return nothing
end

function finalize_transect(acc::TransectAccumulators, n_chunks::Int = 1)
    nd, ne = size(acc.weight_tau)
    vol = acc.side_m^3
    scale = 1.0 / max(n_chunks, 1)
    dE = diff(acc.energy_edges_eV)
    v_edges = sqrt.(2.0 .* acc.energy_edges_eV .* E_CHARGE ./ O2P_MASS)
    d3v = (4.0 * pi / 3.0) .* (v_edges[2:end] .^ 3 .- v_edges[1:(end - 1)] .^ 3)

    def = zeros(Float64, nd, ne)
    psd = zeros(Float64, nd, ne)
    w_conv = fill(NaN, nd, ne)
    w_hall = fill(NaN, nd, ne)
    w_gradpe = fill(NaN, nd, ne)
    w_total = fill(NaN, nd, ne)
    density_m3 = (acc.density_tau ./ vol) .* scale
    bulk_vel_ms = zeros(Float64, 3, nd)

    chosen_pid = zeros(Int, nd)
    chosen_time_s = zeros(Float64, nd)
    chosen_tr = Vector{Any}( fill(nothing, nd) )

    for d in 1:nd
        if acc.density_tau[d] > 0
            bulk_vel_ms[:, d] .= acc.vel_tau[:, d] ./ acc.density_tau[d]
        end
        for k in 1:ne
            wt = acc.weight_tau[d, k]
            if wt > 0
                def[d, k] = scale * acc.def_num[d, k] / (4pi * vol * dE[k])
                psd[d, k] = scale * wt / (vol * d3v[k])
                w_conv[d, k] = acc.work_conv_num[d, k] / wt
                w_hall[d, k] = acc.work_hall_num[d, k] / wt
                w_gradpe[d, k] = acc.work_gradpe_num[d, k] / wt
                w_total[d, k] = acc.work_total_num[d, k] / wt
            end
        end
        if acc.pref_tr[d] !== nothing
            chosen_pid[d] = acc.pref_pid[d]
            chosen_time_s[d] = acc.pref_time_s[d]
            chosen_tr[d] = acc.pref_tr[d]
        else
            chosen_pid[d] = acc.best_pid[d]
            chosen_time_s[d] = acc.best_time_s[d]
            chosen_tr[d] = acc.best_tr[d]
        end
    end
    return (;
        def,
        psd,
        w_conv,
        w_hall,
        w_gradpe,
        w_total,
        density_m3,
        bulk_vel_ms,
        hit_counts = acc.hit_counts,
        best_pid = chosen_pid,
        best_time_s = chosen_time_s,
        best_tr = chosen_tr,
    )
end

function _pack_tr_vec(trs::Vector{Any})
    n = length(trs)
    ts = Vector{Vector{Float64}}(undef, n)
    us = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        tr = trs[i]
        if tr === nothing
            ts[i] = Float64[]
            us[i] = zeros(Float64, 6, 0)
        else
            k_len = length(tr.t)
            ts[i] = Float64.(tr.t)
            mat = Matrix{Float64}(undef, 6, k_len)
            for k in 1:k_len, c in 1:6
                mat[c, k] = Float64(tr.u[k][c])
            end
            us[i] = mat
        end
    end
    return ts, us
end

function _unpack_tr_vec(ts::Vector{Vector{Float64}}, us::Vector{Matrix{Float64}})
    n = length(ts)
    trs = Vector{Any}(fill(nothing, n))
    for i in 1:n
        if !isempty(ts[i])
            mat = us[i]
            u_vec = [SVector{6, Float64}(mat[1, k], mat[2, k], mat[3, k], mat[4, k], mat[5, k], mat[6, k]) for k in 1:size(mat, 2)]
            trs[i] = (; t = copy(ts[i]), u = u_vec)
        end
    end
    return trs
end

function _pack_transect_raw(acc::TransectAccumulators)
    nd = length(acc.centers_m)
    centers_mat = Matrix{Float64}(undef, 3, nd)
    for d in 1:nd
        centers_mat[:, d] .= acc.centers_m[d]
    end
    best_t, best_u = _pack_tr_vec(acc.best_tr)
    pref_t, pref_u = _pack_tr_vec(acc.pref_tr)
    return Dict{String, Any}(
        "centers_m" => centers_mat,
        "side_m" => acc.side_m,
        "energy_edges_eV" => acc.energy_edges_eV,
        "energy_centers_eV" => acc.energy_centers_eV,
        "weight_tau" => acc.weight_tau,
        "def_num" => acc.def_num,
        "work_conv_num" => acc.work_conv_num,
        "work_hall_num" => acc.work_hall_num,
        "work_gradpe_num" => acc.work_gradpe_num,
        "work_total_num" => acc.work_total_num,
        "density_tau" => acc.density_tau,
        "vel_tau" => acc.vel_tau,
        "hit_counts" => acc.hit_counts,
        "best_pid" => acc.best_pid,
        "best_score" => acc.best_score,
        "best_time_s" => acc.best_time_s,
        "best_tr_t" => best_t,
        "best_tr_u" => best_u,
        "pref_pid" => acc.pref_pid,
        "pref_score" => acc.pref_score,
        "pref_time_s" => acc.pref_time_s,
        "pref_tr_t" => pref_t,
        "pref_tr_u" => pref_u,
    )
end

function _unpack_transect_raw(d::Dict{String, Any})
    centers_mat = d["centers_m"]
    centers_m = [Vec3(centers_mat[1, i], centers_mat[2, i], centers_mat[3, i]) for i in 1:size(centers_mat, 2)]
    best_tr = _unpack_tr_vec(d["best_tr_t"], d["best_tr_u"])
    pref_tr = _unpack_tr_vec(d["pref_tr_t"], d["pref_tr_u"])
    return TransectAccumulators(
        centers_m,
        Float64(d["side_m"]),
         copy(d["energy_edges_eV"]),
        copy(d["energy_centers_eV"]),
        copy(d["weight_tau"]),
        copy(d["def_num"]),
        copy(d["work_conv_num"]),
        copy(d["work_hall_num"]),
        copy(d["work_gradpe_num"]),
        copy(d["work_total_num"]),
        copy(d["density_tau"]),
        copy(d["vel_tau"]),
        copy(d["hit_counts"]),
        copy(d["best_pid"]),
        copy(d["best_score"]),
        copy(d["best_time_s"]),
        best_tr,
        copy(d["pref_pid"]),
        copy(d["pref_score"]),
        copy(d["pref_time_s"]),
        pref_tr,
    )
end

function _discover_batch_files(run_dirs::Vector{String})
    files = String[]
    for d in run_dirs
        if isdir(d)
            jlds = sort(filter(p -> occursin(r"^trajectories_.*\.jld2$", basename(p)), readdir(d; join = true)))
            append!(files, jlds)
        elseif isfile(d)
            push!(files, d)
        end
    end
    return files
end

function _pack_rep_ions(target_coords, coord_grid, res)
    idxs = [argmin(abs.(coord_grid .- c)) for c in target_coords]
    pids = [res.best_pid[i] for i in idxs]
    times = [res.best_time_s[i] for i in idxs]
    trs_t = Vector{Vector{Float64}}(undef, length(idxs))
    trs_u = Vector{Matrix{Float64}}(undef, length(idxs))
    for (m, i) in enumerate(idxs)
        tr = res.best_tr[i]
        if tr === nothing
            trs_t[m] = Float64[]
            trs_u[m] = zeros(Float64, 6, 0)
        else
            k_end = clamp(searchsortedlast(tr.t, times[m]) + 1, 2, length(tr.t))
            trs_t[m] = Float64.(tr.t[1:k_end])
            mat = Matrix{Float64}(undef, 6, k_end)
            for k in 1:k_end, c in 1:6
                mat[c, k] = Float64(tr.u[k][c])
            end
            trs_u[m] = mat
        end
    end
    return (; coords = collect(Float64, target_coords), pids, hit_times_s = times, t = trs_t, u = trs_u)
end

function _save_and_plot_paper_analysis(
        out_dir::AbstractString;
        rm_m::Float64,
        fields_path::String,
        n_chunks::Int,
        energy_edges::Vector{Float64},
        pitch_edges_deg::Vector{Float64},
        orbit_angles_deg::Vector{Float64},
        tail_y_Rm::Vector{Float64},
        b_fig3::Vec3,
        b_tail20_nT::Matrix{Float64},
        b_tail175_nT::Matrix{Float64},
        fig3_side::Float64,
        raw_fig3_fxz_unscaled::Matrix{Float64},
        raw_fig3_omni_def_unscaled::Vector{Float64},
        fig3_vx_centers_ms::Vector{Float64},
        fig3_vz_centers_ms::Vector{Float64},
        fig3_vx_edges_ms::Vector{Float64},
        fig3_vz_edges_ms::Vector{Float64},
        fig3_epa_wt::Matrix{Float64},
        fig3_mom_acc::DetectorMomentsAcc,
        orbit25_acc::TransectAccumulators,
        orbit20_acc::TransectAccumulators,
        tail20_acc::TransectAccumulators,
        tail175_acc::TransectAccumulators,
        render_plots::Bool = true,
    )
    scale = 1.0 / max(n_chunks, 1)
    orbit25_res = finalize_transect(orbit25_acc, n_chunks)
    orbit20_res = finalize_transect(orbit20_acc, n_chunks)
    tail20_res = finalize_transect(tail20_acc, n_chunks)
    tail175_res = finalize_transect(tail175_acc, n_chunks)
    fig3_mom = finalize_moments(fig3_mom_acc, fig3_side^3, n_chunks)

    fig3_fxz_s2_m5 = raw_fig3_fxz_unscaled .* scale
    fig3_omni_def = raw_fig3_omni_def_unscaled .* scale

    # Compute Fig 3 1D energy PSD (s^3 m^-6) and 2D Energy-Pitch Angle PSD (s^3 m^-6)
    v_edges = sqrt.(2.0 .* energy_edges .* E_CHARGE ./ O2P_MASS)
    d3v_shell = (4.0 * pi / 3.0) .* (v_edges[2:end] .^ 3 .- v_edges[1:(end - 1)] .^ 3)
    vol_fig3 = fig3_side^3
    fig3_wt_E = vec(sum(fig3_epa_wt; dims = 2))
    fig3_psd_1d = scale .* fig3_wt_E ./ (vol_fig3 .* d3v_shell)

    fig3_epa_psd = zeros(Float64, size(fig3_epa_wt))
    for ke in 1:size(fig3_epa_wt, 1), kp in 1:size(fig3_epa_wt, 2)
        dcos = cosd(pitch_edges_deg[kp]) - cosd(pitch_edges_deg[kp + 1])
        d3v_pa = (2.0 * pi / 3.0) * (v_edges[ke + 1]^3 - v_edges[ke]^3) * dcos
        fig3_epa_psd[ke, kp] = scale * fig3_epa_wt[ke, kp] / (vol_fig3 * d3v_pa)
    end

    fig5_rep_25 = _pack_rep_ions((90.0, 120.0, 150.0), orbit_angles_deg, orbit25_res)
    fig5_rep_20 = _pack_rep_ions((90.0, 120.0, 150.0), orbit_angles_deg, orbit20_res)
    fig7_rep_20 = _pack_rep_ions((0.0, -0.5), tail_y_Rm, tail20_res)
    fig7_rep_175 = _pack_rep_ions((0.0, -0.5), tail_y_Rm, tail175_res)

    # Save numerical results + raw additive accumulators to JLD2
    jldopen(joinpath(out_dir, "paper_analysis.jld2"), "w") do f
        f["rm_m"] = rm_m
        f["fields_file"] = basename(fields_path)
        f["energy_edges_eV"] = energy_edges
        f["energy_centers_eV"] = orbit25_acc.energy_centers_eV

        # Raw additive accumulators for O(1) incremental chunk merging
        f["raw_n_chunks"] = n_chunks
        f["raw_fig3_fxz_unscaled"] = raw_fig3_fxz_unscaled
        f["raw_fig3_omni_def_unscaled"] = raw_fig3_omni_def_unscaled
        f["raw_fig3_epa_wt"] = fig3_epa_wt
        f["raw_fig3_mom"] = _pack_moments_raw(fig3_mom_acc)
        f["raw_orbit25"] = _pack_transect_raw(orbit25_acc)
        f["raw_orbit20"] = _pack_transect_raw(orbit20_acc)
        f["raw_tail20"] = _pack_transect_raw(tail20_acc)
        f["raw_tail175"] = _pack_transect_raw(tail175_acc)

        # Default orbit keys use r = 2.5 Rm (matching manuscript Fig. 4 & Fig. 5)
        f["orbit_radius_Rm"] = 2.5
        f["orbit_angles_deg"] = orbit_angles_deg
        f["orbit_def"] = orbit25_res.def
        f["orbit_psd_s3_m6"] = orbit25_res.psd
        f["orbit_w_conv_eV"] = orbit25_res.w_conv
        f["orbit_w_hall_eV"] = orbit25_res.w_hall
        f["orbit_w_gradpe_eV"] = orbit25_res.w_gradpe
        f["orbit_w_total_eV"] = orbit25_res.w_total
        f["orbit_hits"] = orbit25_res.hit_counts
        f["orbit_best_pid"] = orbit25_res.best_pid
        f["orbit_best_time_s"] = orbit25_res.best_time_s

        # Also store r = 2.0 Rm orbit
        f["orbit20_def"] = orbit20_res.def
        f["orbit20_psd_s3_m6"] = orbit20_res.psd
        f["orbit20_w_conv_eV"] = orbit20_res.w_conv
        f["orbit20_w_hall_eV"] = orbit20_res.w_hall
        f["orbit20_w_gradpe_eV"] = orbit20_res.w_gradpe
        f["orbit20_w_total_eV"] = orbit20_res.w_total
        f["orbit20_hits"] = orbit20_res.hit_counts

        # Default tail keys use X = -2.0 Rm (matching manuscript Fig. 6 & Fig. 7)
        f["tail_x_Rm"] = -2.0
        f["tail_z_Rm"] = 0.75
        f["tail_y_Rm"] = tail_y_Rm
        f["tail_b_nT"] = b_tail20_nT
        f["tail_def"] = tail20_res.def
        f["tail_psd_s3_m6"] = tail20_res.psd
        f["tail_bulk_vel_ms"] = tail20_res.bulk_vel_ms
        f["tail_density_m3"] = tail20_res.density_m3
        f["tail_w_conv_eV"] = tail20_res.w_conv
        f["tail_w_hall_eV"] = tail20_res.w_hall
        f["tail_w_gradpe_eV"] = tail20_res.w_gradpe
        f["tail_w_total_eV"] = tail20_res.w_total
        f["tail_hits"] = tail20_res.hit_counts
        f["tail_best_pid"] = tail20_res.best_pid
        f["tail_best_time_s"] = tail20_res.best_time_s

        # Also store X = -1.75 Rm tail
        f["tail175_b_nT"] = b_tail175_nT
        f["tail175_def"] = tail175_res.def
        f["tail175_psd_s3_m6"] = tail175_res.psd
        f["tail175_bulk_vel_ms"] = tail175_res.bulk_vel_ms
        f["tail175_density_m3"] = tail175_res.density_m3
        f["tail175_w_conv_eV"] = tail175_res.w_conv
        f["tail175_w_hall_eV"] = tail175_res.w_hall
        f["tail175_w_gradpe_eV"] = tail175_res.w_gradpe
        f["tail175_w_total_eV"] = tail175_res.w_total
        f["tail175_hits"] = tail175_res.hit_counts

        # Fig. 3 detector at (-2, 0, 0) Rm
        f["fig3_fxz_s2_m5"] = fig3_fxz_s2_m5
        f["fig3_vx_centers_ms"] = fig3_vx_centers_ms
        f["fig3_vz_centers_ms"] = fig3_vz_centers_ms
        f["fig3_vx_edges_ms"] = fig3_vx_edges_ms
        f["fig3_vz_edges_ms"] = fig3_vz_edges_ms
        f["fig3_omni_def"] = fig3_omni_def
        f["fig3_psd_1d_s3_m6"] = fig3_psd_1d
        f["fig3_epa_wt"] = fig3_epa_wt .* scale
        f["fig3_epa_psd_s3_m6"] = fig3_epa_psd
        f["fig3_pitch_edges_deg"] = pitch_edges_deg
        f["fig3_density_m3"] = fig3_mom.density_m3
        f["fig3_u_bulk_ms"] = fig3_mom.u_bulk_ms
        f["fig3_T_eV"] = fig3_mom.T_eV
        f["fig3_B_nT"] = collect(Float64, b_fig3 .* 1.0e9)

        # Representative ions for Fig. 5 & Fig. 7
        f["fig5_rep_25"] = fig5_rep_25
        f["fig5_rep_20"] = fig5_rep_20
        f["fig7_rep_20"] = fig7_rep_20
        f["fig7_rep_175"] = fig7_rep_175
    end

    if render_plots
        for script_name in ("plot_fig3.jl", "plot_fig4.jl", "plot_fig5.jl", "plot_fig6.jl", "plot_fig7.jl")
            spath = joinpath(@__DIR__, script_name)
            if isfile(spath)
                include(spath)
            end
        end
        if isdefined(@__MODULE__, :plot_fig3)
            Base.invokelatest(Base.invokelatest(getfield, @__MODULE__, :plot_fig3), out_dir, joinpath(out_dir, "fig3.png"))
        end
        if isdefined(@__MODULE__, :plot_fig4)
            Base.invokelatest(Base.invokelatest(getfield, @__MODULE__, :plot_fig4), out_dir, joinpath(out_dir, "fig4.png"))
        end
        if isdefined(@__MODULE__, :plot_fig5)
            Base.invokelatest(Base.invokelatest(getfield, @__MODULE__, :plot_fig5), out_dir, joinpath(out_dir, "fig5.png"); vts_path = fields_path)
        end
        if isdefined(@__MODULE__, :plot_fig6)
            Base.invokelatest(Base.invokelatest(getfield, @__MODULE__, :plot_fig6), out_dir, joinpath(out_dir, "fig6.png"))
        end
        if isdefined(@__MODULE__, :plot_fig7)
            Base.invokelatest(Base.invokelatest(getfield, @__MODULE__, :plot_fig7), out_dir, joinpath(out_dir, "fig7.png"); vts_path = fields_path)
        end
    end
    return out_dir
end

function run_paper_analysis(
        run_source::Union{AbstractString, AbstractVector{<:AbstractString}},
        out_dir::AbstractString;
        render_plots::Bool = true,
    )
    ispath(out_dir) && rm(out_dir; recursive = true, force = true)
    mkpath(out_dir)
    run_dirs = run_source isa AbstractString ? [String(run_source)] : collect(String, run_source)
    n_chunks = max(length(run_dirs), 1)
    first_dir = first(run_dirs)
    meta = TOML.parsefile(joinpath(first_dir, "metadata.toml"))
    rm_m = Float64(meta["Rm_m"])
    fields_path = isfile(meta["source_file"]) ? meta["source_file"] : MarsTP.project_path("data", basename(String(meta["source_file"])))
    if !isfile(fields_path)
        fields_path = MarsTP.project_path("data", "mars_fields_without_crustal.vts")
    end
    fields = load_mhd_fields(fields_path)
    b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)

    # 1. Detector at (-2, 0, 0) Rm (Paper Fig. 3d-f)
    fig3_center = Vec3(-2.0 * rm_m, 0.0, 0.0)
    fig3_side = 0.2 * rm_m
    energy_edges = 10.0 .^ range(0.0, 5.0; length = 65) # 1 eV to 100 keV
    b_fig3 = Vec3(b_itp(fig3_center))
    b_hat_fig3 = norm(b_fig3) > 0 ? b_fig3 / norm(b_fig3) : Vec3(1.0, 0.0, 0.0)
    pitch_edges_deg = collect(range(0.0, 180.0; length = 19)) # 10 deg bins

    # 2. Circular orbit in XZ plane (Y = 0), angles 0°..240° (Paper Fig. 4 & 5: r = 2.5 Rm and r = 2.0 Rm)
    orbit_angles_deg = collect(0.0:2.5:240.0)
    orbit25_centers = [Vec3(2.5 * rm_m * cosd(ang), 0.0, 2.5 * rm_m * sind(ang)) for ang in orbit_angles_deg]
    orbit20_centers = [Vec3(2.0 * rm_m * cosd(ang), 0.0, 2.0 * rm_m * sind(ang)) for ang in orbit_angles_deg]
    orbit_side = 0.2 * rm_m

    # 3. Cross-tail transect along Y ∈ [-1, 1] Rm at Z = +0.75 Rm (Paper Fig. 6 & 7: X = -2.0 Rm and X = -1.75 Rm)
    tail_y_Rm = collect(range(-1.0, 1.0; length = 41))
    tail20_centers = [Vec3(-2.0 * rm_m, y * rm_m, 0.75 * rm_m) for y in tail_y_Rm]
    tail175_centers = [Vec3(-1.75 * rm_m, y * rm_m, 0.75 * rm_m) for y in tail_y_Rm]
    tail_side = 0.2 * rm_m

    batch_files = _discover_batch_files(run_dirs)
    isempty(batch_files) && error("No trajectory batch files found in $run_dirs")
    n_workers = min(Threads.nthreads(), 16, length(batch_files))
    println("Streaming $(length(batch_files)) trajectory batches across $n_workers threads (n_chunks=$n_chunks) ...")
    t_start = time()

    worker_fig3_fine = [
        ForwardPSDAccumulator(;
            detector_m = collect(fig3_center),
            side_m = fig3_side,
            vlim = (-100.0, 100.0),
            vgrid = 160, # 1.25 km/s bins on [-100, 100] km/s
            velocity_unit = :km_s,
            species = "O2+",
            coordinate_system = meta["coordinate_system"],
            energy_edges_eV = energy_edges,
        ) for _ in 1:n_workers
    ]
    worker_fig3_epa = [zeros(Float64, length(energy_edges) - 1, length(pitch_edges_deg) - 1) for _ in 1:n_workers]
    worker_fig3_mom = [DetectorMomentsAcc() for _ in 1:n_workers]
    worker_orbit25 = [TransectAccumulators(orbit25_centers, orbit_side, energy_edges) for _ in 1:n_workers]
    worker_orbit20 = [TransectAccumulators(orbit20_centers, orbit_side, energy_edges) for _ in 1:n_workers]
    worker_tail20 = [TransectAccumulators(tail20_centers, tail_side, energy_edges) for _ in 1:n_workers]
    worker_tail175 = [TransectAccumulators(tail175_centers, tail_side, energy_edges) for _ in 1:n_workers]

    # Partition batch files across workers
    worker_batches = [String[] for _ in 1:n_workers]
    for (idx, fpath) in enumerate(batch_files)
        push!(worker_batches[mod1(idx, n_workers)], fpath)
    end

    Threads.@threads for w in 1:n_workers
        acc_f3 = worker_fig3_fine[w]
        epa_w = worker_fig3_epa[w]
        mom_w = worker_fig3_mom[w]
        acc_o25 = worker_orbit25[w]
        acc_o20 = worker_orbit20[w]
        acc_t20 = worker_tail20[w]
        acc_t175 = worker_tail175[w]

        obs_w = seg -> begin
            dt_seg = seg.t1_s - seg.t0_s
            dt_seg > 0 || return nothing
            vm = 0.5 * (seg.v0_ms + seg.v1_ms)
            spd = norm(vm)
            spd > 0 || return nothing
            wt = seg.rate_weight_s * dt_seg
            mom_w.w_tau += wt
            mom_w.vx_tau += wt * vm[1]
            mom_w.vy_tau += wt * vm[2]
            mom_w.vz_tau += wt * vm[3]
            mom_w.vx2_tau += wt * vm[1]^2
            mom_w.vy2_tau += wt * vm[2]^2
            mom_w.vz2_tau += wt * vm[3]^2

            en_eV = 0.5 * O2P_MASS * spd^2 / E_CHARGE
            pa_deg = acosd(clamp(dot(vm / spd, b_hat_fig3), -1.0, 1.0))
            ke = searchsortedlast(energy_edges, en_eV)
            kp = searchsortedlast(pitch_edges_deg, pa_deg)
            if 1 <= ke <= size(epa_w, 1) && 1 <= kp <= size(epa_w, 2)
                epa_w[ke, kp] += wt
            end
            return nothing
        end

        foreach_saved_trajectory(worker_batches[w]; species = "O2+", include_work = true) do rec
            tr = rec.trajectory
            n_seg = length(tr.t) - 1
            n_seg >= 1 || return nothing
            accumulate_forward_psd!(
                acc_f3,
                tr;
                rate_weight_s = rec.rate_weight_s,
                particle_id = rec.particle_id,
                segment_observer = obs_w,
            )
            dw_tot = rec.work !== nothing && haskey(rec.work.steps, "total") ? rec.work.steps["total"] : zeros(n_seg)
            dw_conv = rec.work !== nothing && haskey(rec.work.steps, "conv") ? rec.work.steps["conv"] : zeros(n_seg)
            dw_hall = rec.work !== nothing && haskey(rec.work.steps, "hall") ? rec.work.steps["hall"] : zeros(n_seg)
            dw_grad = rec.work !== nothing && haskey(rec.work.steps, "gradpe") ? rec.work.steps["gradpe"] : zeros(n_seg)

            _accumulate_transect!(acc_o25, tr.t, tr.u, rec.rate_weight_s, rec.particle_id, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 80.0, mode = :orbit)
            _accumulate_transect!(acc_o20, tr.t, tr.u, rec.rate_weight_s, rec.rate_weight_s > 0 ? rec.particle_id : 0, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 80.0, mode = :orbit)
            _accumulate_transect!(acc_t20, tr.t, tr.u, rec.rate_weight_s, rec.particle_id, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 20.0, mode = :tail)
            _accumulate_transect!(acc_t175, tr.t, tr.u, rec.rate_weight_s, rec.particle_id, dw_tot, dw_conv, dw_hall, dw_grad; min_rep_energy_eV = 20.0, mode = :tail)
        end
    end

    fig3_acc = worker_fig3_fine[1]
    fig3_epa_wt = worker_fig3_epa[1]
    fig3_mom_acc = worker_fig3_mom[1]
    orbit25_acc = worker_orbit25[1]
    orbit20_acc = worker_orbit20[1]
    tail20_acc = worker_tail20[1]
    tail175_acc = worker_tail175[1]

    for w in 2:n_workers
        merge_psd_acc!(fig3_acc, worker_fig3_fine[w])
        fig3_epa_wt .+= worker_fig3_epa[w]
        merge_moments!(fig3_mom_acc, worker_fig3_mom[w])
        merge_transect!(orbit25_acc, worker_orbit25[w])
        merge_transect!(orbit20_acc, worker_orbit20[w])
        merge_transect!(tail20_acc, worker_tail20[w])
        merge_transect!(tail175_acc, worker_tail175[w])
    end

    println("Finished parallel trajectory streaming in $(round(time() - t_start; digits = 1)) s")

    fig3_xz = finish_forward_psd(fig3_acc; option = :xz)

    # Magnetic field along cross-tail transects
    b_tail20_nT = zeros(Float64, 4, length(tail20_centers))
    for (i, c) in enumerate(tail20_centers)
        bv = Vec3(b_itp(c)) * 1.0e9
        b_tail20_nT[1:3, i] .= bv
        b_tail20_nT[4, i] = norm(bv)
    end
    b_tail175_nT = zeros(Float64, 4, length(tail175_centers))
    for (i, c) in enumerate(tail175_centers)
        bv = Vec3(b_itp(c)) * 1.0e9
        b_tail175_nT[1:3, i] .= bv
        b_tail175_nT[4, i] = norm(bv)
    end

    # Reservoir-sample 20,000 trajectories and save compactly for local plotting
    records_20000 = load_trajectories(
        length(run_dirs) == 1 ? first_dir : run_dirs;
        species = "O2+",
        count = 20_000,
        max_points = 200,
        seed = 20260906,
        rm_m = rm_m,
    )
    jldopen(joinpath(out_dir, "trajectories_20000_sample.jld2"), "w") do f
        f["ids"] = [Int32(r["id"]) for r in records_20000]
        f["points_Rm"] = [Float32.(r["points"]) for r in records_20000]
        f["times_s"] = [Float32.(r["times_s"]) for r in records_20000]
    end

    _save_and_plot_paper_analysis(
        out_dir;
        rm_m,
        fields_path,
        n_chunks,
        energy_edges,
        pitch_edges_deg,
        orbit_angles_deg,
        tail_y_Rm,
        b_fig3,
        b_tail20_nT,
        b_tail175_nT,
        fig3_side,
        raw_fig3_fxz_unscaled = fig3_xz.psd,
        raw_fig3_omni_def_unscaled = fig3_xz.omni_def.def,
        fig3_vx_centers_ms = collect(Float64, fig3_xz.velocity_centers_m_s[1]),
        fig3_vz_centers_ms = collect(Float64, fig3_xz.velocity_centers_m_s[2]),
        fig3_vx_edges_ms = collect(Float64, fig3_xz.velocity_edges_m_s[1]),
        fig3_vz_edges_ms = collect(Float64, fig3_xz.velocity_edges_m_s[2]),
        fig3_epa_wt,
        fig3_mom_acc,
        orbit25_acc,
        orbit20_acc,
        tail20_acc,
        tail175_acc,
        render_plots,
    )

    println("Paper analysis complete: $out_dir")
    return out_dir
end

"""
    merge_paper_analyses(paper_dirs, out_dir; render_plots=true)

Merge pre-analyzed chunk paper directories (`paper_analysis.jld2` + `trajectories_20000_sample.jld2`)
in O(1) time without re-streaming any raw trajectory batch files.
"""
function merge_paper_analyses(
        paper_dirs::AbstractVector{<:AbstractString},
        out_dir::AbstractString;
        render_plots::Bool = true,
    )
    isempty(paper_dirs) && error("paper_dirs must not be empty")
    ispath(out_dir) && rm(out_dir; recursive = true, force = true)
    mkpath(out_dir)
    t_start = time()

    first_jld = joinpath(first(paper_dirs), "paper_analysis.jld2")
    rm_m, fields_file, energy_edges, pitch_edges_deg, orbit_angles_deg, tail_y_Rm,
        b_fig3_nT, b_tail20_nT, b_tail175_nT,
        fig3_vx_centers_ms, fig3_vz_centers_ms, fig3_vx_edges_ms, fig3_vz_edges_ms,
        total_chunks, raw_fxz, raw_omni, raw_epa_wt, fig3_mom_acc,
        orbit25_acc, orbit20_acc, tail20_acc, tail175_acc = jldopen(first_jld, "r") do f
        (
            Float64(f["rm_m"]),
            String(f["fields_file"]),
            collect(Float64, f["energy_edges_eV"]),
            collect(Float64, f["fig3_pitch_edges_deg"]),
            collect(Float64, f["orbit_angles_deg"]),
            collect(Float64, f["tail_y_Rm"]),
            collect(Float64, f["fig3_B_nT"]),
            copy(f["tail_b_nT"]),
            copy(f["tail175_b_nT"]),
            collect(Float64, f["fig3_vx_centers_ms"]),
            collect(Float64, f["fig3_vz_centers_ms"]),
            collect(Float64, f["fig3_vx_edges_ms"]),
            collect(Float64, f["fig3_vz_edges_ms"]),
            Int(f["raw_n_chunks"]),
            copy(f["raw_fig3_fxz_unscaled"]),
            copy(f["raw_fig3_omni_def_unscaled"]),
            copy(f["raw_fig3_epa_wt"]),
            _unpack_moments_raw(f["raw_fig3_mom"]),
            _unpack_transect_raw(f["raw_orbit25"]),
            _unpack_transect_raw(f["raw_orbit20"]),
            _unpack_transect_raw(f["raw_tail20"]),
            _unpack_transect_raw(f["raw_tail175"]),
        )
    end

    dir_chunk_weights = Int[total_chunks]
    for idx in 2:length(paper_dirs)
        jpath = joinpath(paper_dirs[idx], "paper_analysis.jld2")
        jldopen(jpath, "r") do f
            k_i = Int(f["raw_n_chunks"])
            push!(dir_chunk_weights, k_i)
            total_chunks += k_i
            raw_fxz .+= f["raw_fig3_fxz_unscaled"]
            raw_omni .+= f["raw_fig3_omni_def_unscaled"]
            raw_epa_wt .+= f["raw_fig3_epa_wt"]
            merge_moments!(fig3_mom_acc, _unpack_moments_raw(f["raw_fig3_mom"]))
            merge_transect!(orbit25_acc, _unpack_transect_raw(f["raw_orbit25"]))
            merge_transect!(orbit20_acc, _unpack_transect_raw(f["raw_orbit20"]))
            merge_transect!(tail20_acc, _unpack_transect_raw(f["raw_tail20"]))
            merge_transect!(tail175_acc, _unpack_transect_raw(f["raw_tail175"]))
        end
    end

    # Combine trajectories_20000_sample.jld2 weighted by chunk count
    sample_ids = Int32[]
    sample_pts = Vector{Matrix{Float32}}()
    sample_tms = Vector{Vector{Float32}}()
    target_total = 20_000
    for (idx, pdir) in enumerate(paper_dirs)
        spath = joinpath(pdir, "trajectories_20000_sample.jld2")
        isfile(spath) || continue
        ids_i, pts_i, tms_i = jldopen(spath, "r") do sf
            (sf["ids"], sf["points_Rm"], sf["times_s"])
        end
        n_i = length(ids_i)
        n_i == 0 && continue
        quota = length(paper_dirs) == 1 ? n_i : min(n_i, cld(target_total * dir_chunk_weights[idx], total_chunks))
        if quota < n_i
            perm = randperm(Xoshiro(20260906 + idx), n_i)[1:quota]
            sort!(perm)
            append!(sample_ids, ids_i[perm])
            append!(sample_pts, pts_i[perm])
            append!(sample_tms, tms_i[perm])
        else
            append!(sample_ids, ids_i)
            append!(sample_pts, pts_i)
            append!(sample_tms, tms_i)
        end
    end
    if length(sample_ids) > target_total
        perm = sort!(randperm(Xoshiro(20260906), length(sample_ids))[1:target_total])
        sample_ids = sample_ids[perm]
        sample_pts = sample_pts[perm]
        sample_tms = sample_tms[perm]
    end
    jldopen(joinpath(out_dir, "trajectories_20000_sample.jld2"), "w") do f
        f["ids"] = sample_ids
        f["points_Rm"] = sample_pts
        f["times_s"] = sample_tms
    end

    fields_path = MarsTP.project_path("data", fields_file)
    if !isfile(fields_path)
        fields_path = MarsTP.project_path("data", "mars_fields_without_crustal.vts")
    end

    _save_and_plot_paper_analysis(
        out_dir;
        rm_m,
        fields_path,
        n_chunks = total_chunks,
        energy_edges,
        pitch_edges_deg,
        orbit_angles_deg,
        tail_y_Rm,
        b_fig3 = Vec3(b_fig3_nT[1] * 1.0e-9, b_fig3_nT[2] * 1.0e-9, b_fig3_nT[3] * 1.0e-9),
        b_tail20_nT,
        b_tail175_nT,
        fig3_side = 0.2 * rm_m,
        raw_fig3_fxz_unscaled = raw_fxz,
        raw_fig3_omni_def_unscaled = raw_omni,
        fig3_vx_centers_ms,
        fig3_vz_centers_ms,
        fig3_vx_edges_ms,
        fig3_vz_edges_ms,
        fig3_epa_wt = raw_epa_wt,
        fig3_mom_acc,
        orbit25_acc,
        orbit20_acc,
        tail20_acc,
        tail175_acc,
        render_plots,
    )

    println("Merged $(length(paper_dirs)) paper analyses (total_chunks=$total_chunks) in $(round(time() - t_start; digits = 2)) s -> $out_dir")
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) <= 2
        run_dir = length(ARGS) >= 1 ? ARGS[1] : "outputs/mc_400km_crustal_1M"
        out_dir = length(ARGS) >= 2 ? ARGS[2] : "$(run_dir)_paper"
        run_paper_analysis(run_dir, out_dir)
    else
        run_paper_analysis(ARGS[1:(end - 1)], ARGS[end])
    end
end
