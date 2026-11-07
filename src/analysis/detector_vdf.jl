function load_detector_vdf(path)
    data = load(resolve_project_path(path))
    f = data["f2d_xz"]
    positive = f[isfinite.(f) .& (f .> 0)]
    return (; vx_km = data["vx_km"], vz_km = data["vz_km"], f2d_xz = f,
        max_f = isempty(positive) ? NaN : maximum(positive),
        positive_count = length(positive), units = get(data, "units", "unknown"))
end

function plot_vx_vz(vdf; output = nothing, color_min = nothing, color_max = nothing)
    f = copy(vdf.f2d_xz)
    valid = isfinite.(f) .& (f .> 0)
    logs = fill(isnothing(color_min) ? -20.0 : Float64(color_min), size(f))
    logs[valid] .= log10.(f[valid])
    cmax = isnothing(color_max) ? maximum(logs[valid]; init = -8.0) : Float64(color_max)
    cmin = isnothing(color_min) ? cmax - 8.0 : Float64(color_min)
    fig, ax = PyPlot.subplots(figsize = (9, 7))
    hm = ax.pcolormesh(
        vdf.vx_km,
        vdf.vz_km,
        permutedims(logs);
        cmap = "turbo",
        vmin = cmin,
        vmax = cmax,
        shading = "auto",
    )
    ax.set_xlabel("Vx [km/s]")
    ax.set_ylabel("Vz [km/s]")
    ax.set_title("Backtraced O2+ f(Vx,Vz)")
    cb = fig.colorbar(hm, ax = ax)
    cb.set_label("log10 f")
    if !isnothing(output)
        mkpath(dirname(abspath(output)))
        fig.savefig(output, dpi = 180, bbox_inches = "tight")
    end
    return fig
end

"""
    vdf_moments(vdf)

Compute number density `n` [m^-3] and bulk velocity components `(Vx_kms, Vz_kms)` [km/s]
from a 2D `Vx-Vz` detector VDF (`s^2 m^-5` or `s^2 km^-5`) using TestParticle's
`velocity_moments`.
"""
function vdf_moments(vdf)
    vx = Float64.(vdf.vx_km)
    vz = Float64.(vdf.vz_km)
    dvx = length(vx) > 1 ? step(range(first(vx), last(vx), length = length(vx))) : 1.0
    dvz = length(vz) > 1 ? step(range(first(vz), last(vz), length = length(vz))) : dvx
    units = hasproperty(vdf, :units) ? String(vdf.units) : "s^2 m^-5"
    # TestParticle.velocity_moments expects A in [s^2/km^5] (which is 1e15 * [s^2/m^5]).
    scale = occursin("km", units) ? 1.0 : 1.0e15
    A_km = Float64.(vdf.f2d_xz) .* scale
    mx = TP.velocity_moments(vx, A_km; dv = sqrt(dvx * dvz))
    mz = TP.velocity_moments(vz, permutedims(A_km); dv = sqrt(dvx * dvz))
    Vx_kms = mx.n > 0 ? mx.nV / mx.n : 0.0
    Vz_kms = mz.n > 0 ? mz.nV / mz.n : 0.0
    return (; n_m3 = mx.n, nVx = mx.nV, nVz = mz.nV, Vx_kms, Vz_kms)
end

function vdf_moments(f3d::AbstractArray{<:Real, 3}, axes::NTuple{3, AbstractVector}; species = "O2+")
    vx, vy, vz = Float64.(axes[1]), Float64.(axes[2]), Float64.(axes[3])
    dvx = length(vx) > 1 ? (last(vx) - first(vx)) / (length(vx) - 1) : 1.0
    dvy = length(vy) > 1 ? (last(vy) - first(vy)) / (length(vy) - 1) : 1.0
    dvz = length(vz) > 1 ? (last(vz) - first(vz)) / (length(vz) - 1) : 1.0
    f2d_xz = dropdims(sum(Float64.(f3d); dims = 2); dims = 2) .* dvy
    res = vdf_moments((; vx_km = vx ./ 1.0e3, vz_km = vz ./ 1.0e3, f2d_xz = f2d_xz, units = "s^2 m^-5"))
    f2d_yz = dropdims(sum(Float64.(f3d); dims = 1); dims = 1) .* dvx
    res_y = vdf_moments((; vx_km = vy ./ 1.0e3, vz_km = vz ./ 1.0e3, f2d_xz = f2d_yz, units = "s^2 m^-5"))
    V = SVector{3, Float64}(res.Vx_kms * 1.0e3, res_y.Vx_kms * 1.0e3, res.Vz_kms * 1.0e3)
    return (; n = res.n_m3, V = V, Vx_kms = res.Vx_kms, Vy_kms = res_y.Vx_kms, Vz_kms = res.Vz_kms)
end

"""
    refine_backtrace_config(config::BacktraceConfig, f3d::AbstractArray{<:Real, 3};
        dv_kms=1.0, dvy_kms=dv_kms, margin_kms=20.0, relthresh=1e-5)

Refine a coarse `BacktraceConfig` velocity window around the populated region of `f3d`
using `TestParticle.refine_vdf_window`.
"""
function refine_backtrace_config(
        config::BacktraceConfig,
        f3d::AbstractArray{<:Real, 3};
        dv_kms::Real = 1.0,
        dvy_kms::Real = dv_kms,
        margin_kms::Real = 20.0,
        relthresh::Real = 1.0e-5,
    )
    axes = velocity_axes(config)
    vx_km = axes.vx ./ 1.0e3
    vy_km = axes.vy ./ 1.0e3
    vz_km = axes.vz ./ 1.0e3
    bounds = (
        (config.vx_min_kms, config.vx_max_kms),
        (config.vy_min_kms, config.vy_max_kms),
        (config.vz_min_kms, config.vz_max_kms),
    )
    v0 = (config.vx_min_kms, config.vy_min_kms, config.vz_min_kms)
    dv = (Float64(dv_kms), Float64(dvy_kms), Float64(dv_kms))
    vx_new, vy_new, vz_new = TP.refine_vdf_window(
        f3d, vx_km, vy_km, vz_km, v0, dv;
        margin = Float64(margin_kms), relthresh, bounds,
    )
    return BacktraceConfig(;
        detector_Rm = config.detector_Rm,
        species = config.species,
        ionosphere_altitude_km = config.ionosphere_altitude_km,
        include_ionosphere = config.include_ionosphere,
        mhd_file = config.mhd_file,
        source_file = config.source_file,
        vx_min_kms = first(vx_new),
        vx_max_kms = last(vx_new),
        vy_min_kms = first(vy_new),
        vy_max_kms = last(vy_new),
        vz_min_kms = first(vz_new),
        vz_max_kms = last(vz_new),
        dv_kms = Float64(dv_kms),
        dvy_kms = Float64(dvy_kms),
        tspan = config.tspan,
        solver = config.solver,
        dt = config.dt,
        safety = config.safety,
        energy_edges_eV = config.energy_edges_eV,
        include_work = config.include_work,
        stream_vy = config.stream_vy,
        checkpoint_file = config.checkpoint_file,
        output_file = config.output_file,
        progress_interval = config.progress_interval,
    )
end
