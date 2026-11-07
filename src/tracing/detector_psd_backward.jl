Base.@kwdef struct BacktraceConfig
    detector_Rm::SVector{3, Float64} = SA[-1.5, 0.0, 1.0]
    species::String = "O2+"
    ionosphere_altitude_km::Float64 = 400.0
    include_ionosphere::Bool = true
    mhd_file::String = data_path("mars_fields_spherical_from_dat.vts")
    source_file::String = data_path("O2plus_source_rates.mat")
    vx_min_kms::Float64 = -500.0
    vx_max_kms::Float64 = 500.0
    vy_min_kms::Float64 = -500.0
    vy_max_kms::Float64 = 500.0
    vz_min_kms::Float64 = -500.0
    vz_max_kms::Float64 = 500.0
    dv_kms::Float64 = 1.0
    dvy_kms::Float64 = 1.0
    tspan::Tuple{Float64, Float64} = (0.0, -500.0)
    solver::Symbol = :boris
    dt::Float64 = -0.2
    safety::Float64 = 0.005
    # Optional 4pi-averaged DEF, accumulated before vy projection, eV edges.
    energy_edges_eV::Union{Nothing, Vector{Float64}} = nothing
    include_work::Bool = false
    stream_vy::Bool = true
    checkpoint_file::String = project_path("output", "backtrace_checkpoint.jld2")
    output_file::String = project_path("output", "backtrace_vdf.jld2")
    progress_interval::Int = 10000
end

velocity_axes(c::BacktraceConfig) = (
    vx = collect((c.vx_min_kms * 1.0e3):(c.dv_kms * 1.0e3):(c.vx_max_kms * 1.0e3)),
    vy = collect((c.vy_min_kms * 1.0e3):(c.dvy_kms * 1.0e3):(c.vy_max_kms * 1.0e3)),
    vz = collect((c.vz_min_kms * 1.0e3):(c.dv_kms * 1.0e3):(c.vz_max_kms * 1.0e3)),
)

# First entry into the inner sphere or exit from the outer sphere along a
# Boris position segment. Also catches a segment passing through the sphere.
function _backtrace_crossing(a, b, inner, outer)
    d = b - a
    A = dot(d, d)
    A == 0 && return (1.0, 1)
    ra = norm(a)
    tolerance = 64eps(max(inner, outer))
    radial_drift = dot(a, d)
    abs(ra - inner) <= tolerance && radial_drift < 0 && return (0.0, 2)
    # A nonzero tangent segment leaves the convex outer sphere immediately,
    # even though its radial derivative at the initial endpoint is zero.
    abs(ra - outer) <= tolerance && radial_drift >= 0 && return (0.0, 4)
    best, status = 1.0, 1
    for (radius, flag) in ((inner, 2), (outer, 4))
        # Avoid cancellation between two squared planetary-scale radii.
        B, C = 2radial_drift, (ra - radius) * (ra + radius)
        disc = B^2 - 4A * C
        disc < 0 && continue
        q = -0.5 * (B + copysign(sqrt(disc), B))
        roots = q == 0 ? (-B / (2A), -B / (2A)) : (q / A, C / q)
        for fraction in roots
            0 <= fraction <= best || continue
            slope = dot(a + fraction * d, d)
            ((flag == 2 && slope < 0) || (flag == 4 && slope > 0)) || continue
            best, status = fraction, flag
        end
    end
    return best, status
end

function _volume_source(q_itp, gitm, mass, position, velocity)
    q = q_itp(position)
    isfinite(q) && q >= 0 || error("Invalid volume production at $position: $q")
    q == 0 && return 0.0
    Tn = _neutral_temperature(gitm, position)
    isfinite(Tn) && Tn > 0 || error("Invalid neutral temperature at $position: $Tn")
    return q * VDF.Maxwellian(sqrt(2 * TP.kB * Tn / mass); u0 = SA[0.0, 0.0, 0.0])(velocity)
end

# Zero-allocation tuple variant returning (count, f1, f2) in ascending order.
@inline function _shell_crossings_tuple(a, b, radius, maxfraction = 1.0)
    d = b - a
    A = dot(d, d)
    A == 0 && return (0, 0.0, 0.0)
    B = 2dot(a, d)
    C = (norm(a) - radius) * (norm(a) + radius)
    disc = B^2 - 4A * C
    tolerance_disc = 64eps(Float64) * max(B^2, abs(4A * C), 1.0)
    disc < -tolerance_disc && return (0, 0.0, 0.0)
    center = -B / (2A)
    if abs(disc) <= tolerance_disc
        0 <= center <= maxfraction &&
            error("Unresolved tangency to the ideal ionosphere sheet; use a finite-thickness model")
        return (0, 0.0, 0.0)
    end
    q = -0.5 * (B + copysign(sqrt(disc), B))
    r1, r2 = q / A, C / q
    if r1 > r2
        r1, r2 = r2, r1
    end
    tolerance = 64eps(Float64)
    v1 = -tolerance <= r1 <= maxfraction + tolerance
    v2 = -tolerance <= r2 <= maxfraction + tolerance
    if v1 && v2
        return (2, clamp(r1, 0.0, maxfraction), clamp(r2, 0.0, maxfraction))
    elseif v1
        return (1, clamp(r1, 0.0, maxfraction), 0.0)
    elseif v2
        return (1, clamp(r2, 0.0, maxfraction), 0.0)
    else
        return (0, 0.0, 0.0)
    end
end

# Roots on the accepted part of a position segment, including both directions.
# A previously counted endpoint is deduplicated by event time in _trace_sources.
function _shell_crossings(a, b, radius, maxfraction = 1.0)
    n, f1, f2 = _shell_crossings_tuple(a, b, radius, maxfraction)
    n == 0 && return Float64[]
    n == 1 && return [f1]
    return [f1, f2]
end

# F*g/|v dot er|: (m^-2 s^-1)*(s^3 m^-3)/(m s^-1) = s^3 m^-6.
function _surface_increment(source, position, velocity)
    vr = abs(dot(velocity, position / norm(position)))
    vr > sqrt(eps(Float64)) * max(norm(velocity), 1.0) ||
        error("Unresolved grazing ionosphere crossing; no radial-speed floor is applied")
    value = source.flux * source.g / vr
    isfinite(value) && value >= 0 || error("Invalid ionosphere sheet contribution")
    return value
end

# Streaming quadrature over the entire trajectory down to the fixed 200 km
# boundary. Source-shell events split the volume quadrature but do not terminate.
function _trace_sources(
        position, velocity, param, config, volume_source, ionosphere;
        outer = Router, work_itp = nothing, segment_observer = nothing
    )
    Rinner < norm(position) < outer ||
        throw(ArgumentError("Detector must lie strictly between 200 km and the outer boundary"))
    shell = _ionosphere_radius(config.ionosphere_altitude_km)
    config.include_ionosphere && abs(norm(position) - shell) <= 1.0e-7 &&
        throw(ArgumentError("Detector on a zero-thickness source sheet is ambiguous; move it off the sheet"))
    previous_v = Ref(velocity)
    work_eV = zeros(3) # forward-time gain: total, convection, Hall
    charge = TP.SpeciesDict[config.species].q
    previous = Ref(position)
    previous_t = Ref(config.tspan[1])
    previous_q = Ref(volume_source(position, velocity))
    volume, sheet_f, flux = Ref(0.0), Ref(0.0), Ref(0.0)
    hits, last_hit_time = Ref(0), Ref(NaN)
    status = Ref(1)
    endpoint, end_velocity = Ref(position), Ref(velocity)
    end_time = Ref(config.tspan[1])
    function boundary(u, p, t)
        if !all(isfinite, u[1:3]) || !isfinite(t)
            status[] = 3
            return true
        end
        step = t - previous_t[]
        step < 0 || error("Backtrace time must decrease")
        r = SVector{3, Float64}(u[1:3])
        # 0.24 callbacks expose node velocity. The displacement gives the
        # actual drift/half-step velocity, also when the endpoint is outside
        # the table and its synchronized velocity is NaN. Check the crossing
        # before evaluating fields at the safe accepted part of this segment.
        vhalf = (r - previous[]) / step
        fraction, flag = _backtrace_crossing(previous[], r, Rinner, outer)
        n_events, ev1, ev2 = config.include_ionosphere ?
            _shell_crossings_tuple(previous[], r, shell, fraction) : (0, 0.0, 0.0)
        n_nodes = n_events + 1
        nodes_tuple = n_events == 0 ? ((fraction, false), (0.0, false), (0.0, false)) :
            n_events == 1 ? ((ev1, true), (fraction, false), (0.0, false)) :
            ((ev1, true), (ev2, true), (fraction, false))
        last_fraction, last_q = 0.0, previous_q[]
        last_x, last_v = previous[], previous_v[]
        for idx in 1:n_nodes
            f, is_sheet = nodes_tuple[idx]
            x = previous[] + f * (r - previous[])
            hit_t = previous_t[] + f * step
            safe_x = x / norm(x) * clamp(norm(x), Rinner + 1.0e-8, outer - 1.0e-8)
            v = BorisPushers.update_velocity(vhalf, safe_x, (f - 0.5) * step, hit_t, p, BorisPushers.Boris())
            all(isfinite, v) || error("Invalid synchronized velocity at $safe_x")
            q = volume_source(safe_x, v)
            volume[] += 0.5 * (last_q + q) * abs((f - last_fraction) * step)
            if work_itp !== nothing
                mid_x, mid_v = (last_x + safe_x) / 2, (last_v + v) / 2
                forward_dt = -(f - last_fraction) * step
                sw1 = _work_one(work_itp.total, mid_x, mid_v, charge, forward_dt) / TP.eV
                sw2 = _work_one(work_itp.conv, mid_x, mid_v, charge, forward_dt) / TP.eV
                sw3 = _work_one(work_itp.hall, mid_x, mid_v, charge, forward_dt) / TP.eV
                work_eV[1] += sw1
                work_eV[2] += sw2
                work_eV[3] += sw3
                segment_observer !== nothing && segment_observer(
                    last_x, safe_x,
                    previous_t[] + last_fraction * step, hit_t, [sw1, sw2, sw3]
                )
            end
            last_x, last_v = safe_x, v
            last_fraction, last_q = f, q
            if is_sheet
                # Nanometre position roundoff near a shared endpoint must not
                # count the same crossing again. Distinct resolved returns count.
                time_tol = 1.0e-7 / max(norm(v), 1.0) + 64eps(max(abs(hit_t), 1.0))
                if !isfinite(last_hit_time[]) || abs(hit_t - last_hit_time[]) > time_tol
                    source = ionosphere(x, v)
                    sheet_f[] += _surface_increment(source, x, v)
                    flux[] = source.flux
                    hits[] += 1
                    last_hit_time[] = hit_t
                end
            end
            endpoint[], end_velocity[], end_time[] = x, v, hit_t
        end
        if flag != 1
            status[] = flag
            return true
        end
        previous[], previous_t[], previous_q[] = r, t, last_q
        previous_v[] = last_v
        return false
    end
    prob = TP.TraceProblem(vcat(position, velocity), config.tspan, param)
    if config.solver == :boris
        steps = ceil(Int, abs((config.tspan[2] - config.tspan[1]) / config.dt))
        dt = (config.tspan[2] - config.tspan[1]) / steps
        TP.solve(
            prob, BorisPushers.Boris(); dt, isoutside = boundary, maxiters = steps + 1,
            save_start = false, save_end = false, save_everystep = false
        )
    else
        TP.solve(
            prob, BorisPushers.AdaptiveBoris(; safety = config.safety); isoutside = boundary,
            save_start = false, save_end = false, save_everystep = false
        )
    end
    if status[] == 1 && !isapprox(end_time[], config.tspan[2]; atol = 1.0e-7, rtol = 1.0e-10)
        error("Backtrace stopped before the requested time without a boundary status")
    end
    return (;
        volume = volume[], ionosphere = sheet_f[], flux = flux[], crossings = hits[], status = status[],
        position = endpoint[], velocity = end_velocity[], time = end_time[], work_eV,
        delta_kinetic_eV = TP.SpeciesDict[config.species].m * (dot(velocity, velocity) - dot(end_velocity[], end_velocity[])) / (2TP.eV),
    )
end

function _validate_backtrace(config)
    config.energy_edges_eV === nothing || _OmniDEFAccumulator(config.energy_edges_eV, config.species)
    radius = _ionosphere_radius(config.ionosphere_altitude_km)
    all(isfinite, config.detector_Rm) && Rinner < norm(config.detector_Rm) * Rm < Router ||
        throw(ArgumentError("Detector must lie strictly above 200 km and below 4 Rm"))
    config.include_ionosphere && abs(norm(config.detector_Rm) * Rm - radius) <= 1.0e-7 &&
        throw(ArgumentError("Detector must not lie exactly on the zero-thickness source sheet"))
    config.solver in (:boris, :adaptive) || throw(ArgumentError("solver must be :boris or :adaptive"))
    all(isfinite, config.tspan) && config.tspan[2] < config.tspan[1] ||
        throw(ArgumentError("tspan must be finite and decreasing"))
    isfinite(config.dt) && config.dt < 0 || throw(ArgumentError("dt must be finite and negative"))
    isfinite(config.safety) && config.safety > 0 || throw(ArgumentError("safety must be positive"))
    for (lo, hi, dv) in (
            (config.vx_min_kms, config.vx_max_kms, config.dv_kms),
            (config.vy_min_kms, config.vy_max_kms, config.dvy_kms),
            (config.vz_min_kms, config.vz_max_kms, config.dv_kms),
        )
        all(isfinite, (lo, hi, dv)) && lo <= hi && dv > 0 ||
            throw(ArgumentError("Velocity axes require finite min <= max and positive spacing"))
    end
    return nothing
end

function run_backtrace_vdf(config::BacktraceConfig = BacktraceConfig())
    config.species == "O2+" || error("Only O2+ source-rate backtracing is configured")
    _validate_backtrace(config)
    mkpath(dirname(config.output_file))
    fields = load_mhd_fields(config.mhd_file; electric_field = :total)
    work_itp = config.include_work ? build_field_work_interpolators(config.mhd_file) : nothing
    source = load_o2plus_source_rates(config.source_file; fill_invalid = false)
    ionosphere_source = config.include_ionosphere ? load_ionosphere_source(
            config.mhd_file;
            altitude_km = config.ionosphere_altitude_km
        ) : nothing
    assert_same_grid(source, fields)
    gitm = load_gitm()
    param = mhd_param(fields; species = config.species)
    q_itp = TP.build_interpolator(
        TP.StructuredGrid, source.production_density,
        fields.r, fields.theta, fields.phi
    )
    axes = velocity_axes(config)
    nx, ny, nz = length(axes.vx), length(axes.vy), length(axes.vz)
    total = nx * ny * nz
    total > 100_000_000 && !config.stream_vy &&
        error("Use stream_vy=true for very large 3D velocity grids")

    position = config.detector_Rm .* Rm
    mass = TP.SpeciesDict[config.species].m
    volume_source(p, v) = _volume_source(q_itp, gitm, mass, p, v)
    sheet_source(p, v) = ionosphere_distribution(ionosphere_source, p, v; mass)
    omni_acc = config.energy_edges_eV === nothing ? nothing : _OmniDEFAccumulator(config.energy_edges_eV, config.species)
    f2d = zeros(Float64, nx, nz)
    f2d_volume = zeros(Float64, nx, nz)
    f2d_ionosphere = zeros(Float64, nx, nz)
    # Work arrays use (vx, vy, vz, component); disabled runs allocate no 3D grid.
    work3d_eV = config.include_work ? fill(NaN, nx, ny, nz, 3) : nothing
    delta_kinetic3d_eV = config.include_work ? fill(NaN, nx, ny, nz) : nothing
    status3d = config.include_work ? zeros(Int, nx, ny, nz) : nothing
    psd3d = config.include_work ? zeros(nx, ny, nz) : nothing
    work_numerator = config.include_work ? zeros(nx, nz, 3) : nothing
    status_counts = zeros(Int, 5)
    ionosphere_crossings = zeros(Int, nx, nz)
    model = "thin_shell_source_v1"
    status_labels = ["unused", "time_limit", "inner_200km", "nonfinite", "outer"]
    units = "s^2 m^-5"
    source_units = (;
        f3d = "s^3 m^-6", volume_rate = "m^-3 s^-1",
        surface_rate = "m^-2 s^-1", velocity_pdf = "s^3 m^-3", source_term = "s^2 m^-6",
    )
    done = Threads.Atomic{Int}(0)

    for (iy, vy) in pairs(axes.vy)
        slice = zeros(Float64, nx, nz)
        sheet_slice = zeros(Float64, nx, nz)
        flags = zeros(Int, nx, nz)
        Threads.@threads for index in eachindex(slice)
            I = CartesianIndices(slice)[index]
            velocity = SA[axes.vx[I[1]], vy, axes.vz[I[2]]]
            local result = _trace_sources(
                position, velocity, param, config, volume_source,
                sheet_source; outer = last(fields.r), work_itp
            )
            slice[I], sheet_slice[I], flags[I] = result.volume, result.ionosphere, result.status
            if config.include_work
                ix, iz = Tuple(I)
                status3d[ix, iy, iz] = result.status
                if result.status != 3
                    work3d_eV[ix, iy, iz, :] .= result.work_eV
                    delta_kinetic3d_eV[ix, iy, iz] = result.delta_kinetic_eV
                    psd3d[ix, iy, iz] = result.volume + result.ionosphere
                    work_numerator[ix, iz, :] .+= psd3d[ix, iy, iz] .* result.work_eV .* (config.dvy_kms * 1.0e3)
                end
            end
            ionosphere_crossings[I] += result.crossings
            n = Threads.atomic_add!(done, 1) + 1
            config.progress_interval > 0 && n % config.progress_interval == 0 &&
                println("backtrace progress: $n / $total")
        end
        if omni_acc !== nothing
            any(==(3), flags) && error("Cannot form DEF from failed backtrace velocities")
            _def_vdf!(
                omni_acc, reshape(slice .+ sheet_slice, nx, 1, nz),
                (axes.vx, [vy], axes.vz), (config.dv_kms * 1.0e3, config.dvy_kms * 1.0e3, config.dv_kms * 1.0e3)
            )
        end
        f2d_volume .+= slice .* (config.dvy_kms * 1.0e3)
        f2d_ionosphere .+= sheet_slice .* (config.dvy_kms * 1.0e3)
        f2d .= f2d_volume .+ f2d_ionosphere
        for flag in 1:4
            status_counts[flag + 1] += count(==(flag), flags)
        end
        if config.stream_vy
            jldsave(
                config.checkpoint_file; omni_def = omni_acc === nothing ? nothing : detector_omni_def(omni_acc), work3d_eV, delta_kinetic3d_eV, status3d, psd3d, work_numerator, completed_iy = iy, f2d_xz = f2d, f2d_volume, f2d_ionosphere,
                status_counts, status_labels, ionosphere_crossings, model, units, source_units, vx_km = axes.vx ./ 1.0e3, vy_km = axes.vy ./ 1.0e3,
                vz_km = axes.vz ./ 1.0e3, config
            )
        end
    end

    mean_work2d_eV = config.include_work ? map(
            (w, f) -> f > 0 ? w / f : NaN,
            work_numerator, repeat(reshape(f2d, nx, nz, 1), 1, 1, 3)
        ) : nothing
    result = (;
        omni_def = omni_acc === nothing ? nothing : detector_omni_def(omni_acc), work3d_eV, delta_kinetic3d_eV, status3d, psd3d, mean_work2d_eV,
        work_components = ("total", "convection", "Hall"), work_units = "eV",
        work_definition = "Forward-time gain from backtrace endpoint to detector; 2D mean weighted by full-path PSD over vy, not birth-to-detector work",
        vx_km = axes.vx ./ 1.0e3, vy_km = axes.vy ./ 1.0e3,
        vz_km = axes.vz ./ 1.0e3, f2d_xz = f2d, f2d_volume, f2d_ionosphere, status_counts, config,
        status_labels, ionosphere_crossings, model, source_units,
        ionosphere_flux_definition = "F=n*norm(Ui) [m^-2 s^-1]; each crossing adds F*g/abs(v dot er)",
        units,
    )
    jldsave(config.output_file; result...)
    return result
end
