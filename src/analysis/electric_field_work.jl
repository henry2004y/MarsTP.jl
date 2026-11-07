struct FieldWorkInterpolators
    total::Any
    conv::Any
    hall::Any
    gradpe::Any
    res::Any
end

FieldWorkInterpolators(total, conv, hall) =
    FieldWorkInterpolators(total, conv, hall, nothing, nothing)

function _try_load_electric_itp(mhd_file, selector)
    try
        f = load_mhd_fields(mhd_file; electric_field = selector)
        return TP.build_interpolator(TP.StructuredGrid, f.E, f.r, f.theta, f.phi)
    catch
        return nothing
    end
end

function build_field_work_interpolators(
        mhd_file = data_path("mars_fields_spherical_from_dat.vts");
        include_extended::Bool = true,
    )
    total = load_mhd_fields(mhd_file; electric_field = :total)
    conv = load_mhd_fields(mhd_file; electric_field = :conv)
    hall = load_mhd_fields(mhd_file; electric_field = :hall)
    gradpe = include_extended ? _try_load_electric_itp(mhd_file, :gradpe) : nothing
    res = include_extended ? _try_load_electric_itp(mhd_file, :res) : nothing
    return FieldWorkInterpolators(
        TP.build_interpolator(TP.StructuredGrid, total.E, total.r, total.theta, total.phi),
        TP.build_interpolator(TP.StructuredGrid, conv.E, conv.r, conv.theta, conv.phi),
        TP.build_interpolator(TP.StructuredGrid, hall.E, hall.r, hall.theta, hall.phi),
        gradpe,
        res,
    )
end

function _cartesian_velocity_to_spherical(v, p)
    return TP.cart2sphvec(v[1], v[2], v[3], p[1], p[2], p[3])
end

function _work_one(itp, p, v, q, dt)
    E = itp(p)
    all(isfinite, E) || error("Nonfinite electric field in work integral")
    # StructuredGrid vector interpolators already return Cartesian components.
    return q * dot(E, v) * dt
end

function _work_trajectory(sol)
    if hasproperty(sol, :t)
        traj = sol
    elseif hasproperty(sol, :u) && length(sol.u) == 1
        return _work_trajectory(only(sol.u))
    else
        throw(ArgumentError("Expected one trajectory; use particle_field_work for multiple particles"))
    end
    length(traj.t) == length(traj.u) && !isempty(traj.t) ||
        throw(ArgumentError("Trajectory must have matching nonempty time and state arrays"))
    all(isfinite, traj.t) || throw(ArgumentError("Nonfinite trajectory times"))
    if length(traj.t) >= 3 && traj.t[end] == traj.t[end - 1]
        if traj.u[end] == traj.u[end - 2]
            traj = (; t = @view(traj.t[1:(end - 2)]), u = @view(traj.u[1:(end - 2)]))
        elseif traj.u[end] == traj.u[end - 1]
            traj = (; t = @view(traj.t[1:(end - 1)]), u = @view(traj.u[1:(end - 1)]))
        end
    end
    dt = diff(traj.t)
    (all(>(0), dt) || all(<(0), dt)) ||
        throw(ArgumentError("Trajectory times must be strictly monotonic"))
    all(u -> length(u) >= 6 && all(isfinite, u[1:6]), traj.u) ||
        throw(ArgumentError("States must contain finite Cartesian position and velocity"))
    return traj
end

"""
    field_work(sol, itp; species="O2+")

Signed total, convection and Hall work along one existing trajectory, in J and eV.
Uses midpoint quadrature of q E⋅v dt. Positions and velocities must be Cartesian
SI values; fields returned by `itp` must be Cartesian V/m. Decreasing times
retain their sign. A single-member ensemble is accepted; larger ensembles
must use `particle_field_work`. Does not write files.
"""
function field_work(sol, itp::FieldWorkInterpolators; species = "O2+")
    traj = _work_trajectory(sol)
    q = TP.SpeciesDict[species].q
    total_J = 0.0
    conv_J = 0.0
    hall_J = 0.0
    gradpe_J = 0.0
    res_J = 0.0
    has_ext = itp.gradpe !== nothing || itp.res !== nothing
    for i in 1:(length(traj.t) - 1)
        u0, u1 = traj.u[i], traj.u[i + 1]
        p = SA[0.5 * (u0[1] + u1[1]), 0.5 * (u0[2] + u1[2]), 0.5 * (u0[3] + u1[3])]
        v = SA[0.5 * (u0[4] + u1[4]), 0.5 * (u0[5] + u1[5]), 0.5 * (u0[6] + u1[6])]
        dt = traj.t[i + 1] - traj.t[i]
        all(isfinite, p) && all(isfinite, v) || error("Nonfinite trajectory in work integral")
        total_J += _work_one(itp.total, p, v, q, dt)
        conv_J += _work_one(itp.conv, p, v, q, dt)
        hall_J += _work_one(itp.hall, p, v, q, dt)
        itp.gradpe !== nothing && (gradpe_J += _work_one(itp.gradpe, p, v, q, dt))
        itp.res !== nothing && (res_J += _work_one(itp.res, p, v, q, dt))
    end
    base = (; total_J, conv_J, hall_J,
        total_eV = total_J / TP.eV, conv_eV = conv_J / TP.eV, hall_eV = hall_J / TP.eV)
    has_ext || return base
    return (; base..., gradpe_J, res_J,
        gradpe_eV = gradpe_J / TP.eV, res_eV = res_J / TP.eV)
end

"""
    Electric_field_work_profile(sol, itp; species="O2+")

Return time-resolved signed work and instantaneous power on the same trajectory.
`total_eV`, `conv_eV`, `hall_eV` are cumulative work, starting at zero.
`power_total_eV_s`, `power_conv_eV_s`, `power_hall_eV_s` are q E⋅v at saved states.
`kinetic_eV` and `delta_kinetic_eV` use nonrelativistic kinetic energy.
`energy_residual_eV = delta_kinetic_eV - total_eV` checks quadrature/integration
accuracy when only electromagnetic forces act. `field_sum_residual_eV` checks
whether total work equals convection plus Hall work. Totals are in `summary`.
Invalid fields/states raise an error instead of silently contributing zero.
Coarse saved trajectories can underestimate work; check saving cadence and dt.
"""
function Electric_field_work_profile(sol, itp::FieldWorkInterpolators; species = "O2+")
    traj = _work_trajectory(sol)
    sp = TP.SpeciesDict[species]
    n = length(traj.t)
    has_ext = itp.gradpe !== nothing || itp.res !== nothing
    work = zeros(n, 5)
    power = zeros(n, 5)
    kinetic = zeros(n)
    fields = (itp.total, itp.conv, itp.hall, itp.gradpe, itp.res)
    for i in 1:n
        u = traj.u[i]
        p, v = SA[u[1], u[2], u[3]], SA[u[4], u[5], u[6]]
        kinetic[i] = 0.5 * sp.m * dot(v, v) / TP.eV
        for j in 1:5
            fields[j] === nothing && continue
            power[i, j] = _work_one(fields[j], p, v, sp.q, 1.0) / TP.eV
        end
        if i > 1
            a = traj.u[i - 1]
            pm = 0.5 * (p + SA[a[1], a[2], a[3]])
            vm = 0.5 * (v + SA[a[4], a[5], a[6]])
            dt = traj.t[i] - traj.t[i - 1]
            for j in 1:5
                fields[j] === nothing && continue
                work[i, j] = work[i - 1, j] + _work_one(fields[j], pm, vm, sp.q, dt) / TP.eV
            end
        end
    end
    total_eV, conv_eV, hall_eV, gradpe_eV, res_eV =
        work[:, 1], work[:, 2], work[:, 3], work[:, 4], work[:, 5]
    delta_kinetic_eV = kinetic .- first(kinetic)
    energy_residual_eV = delta_kinetic_eV .- total_eV
    field_sum_residual_eV = total_eV .- conv_eV .- hall_eV .- gradpe_eV .- res_eV
    if !has_ext
        summary = (; total_eV = last(total_eV), conv_eV = last(conv_eV),
            hall_eV = last(hall_eV), delta_kinetic_eV = last(delta_kinetic_eV),
            energy_residual_eV = last(energy_residual_eV),
            field_sum_residual_eV = last(field_sum_residual_eV))
        return (; t = collect(traj.t), total_eV, conv_eV, hall_eV,
            power_total_eV_s = power[:, 1], power_conv_eV_s = power[:, 2],
            power_hall_eV_s = power[:, 3], kinetic_eV = kinetic, delta_kinetic_eV,
            energy_residual_eV, field_sum_residual_eV, summary)
    end
    summary = (; total_eV = last(total_eV), conv_eV = last(conv_eV),
        hall_eV = last(hall_eV), gradpe_eV = last(gradpe_eV), res_eV = last(res_eV),
        delta_kinetic_eV = last(delta_kinetic_eV),
        energy_residual_eV = last(energy_residual_eV),
        field_sum_residual_eV = last(field_sum_residual_eV))
    return (; t = collect(traj.t), total_eV, conv_eV, hall_eV, gradpe_eV, res_eV,
        power_total_eV_s = power[:, 1], power_conv_eV_s = power[:, 2],
        power_hall_eV_s = power[:, 3], power_gradpe_eV_s = power[:, 4],
        power_res_eV_s = power[:, 5], kinetic_eV = kinetic, delta_kinetic_eV,
        energy_residual_eV, field_sum_residual_eV, summary)
end

"""
    particle_field_work(solutions, itp; species="O2+", profiles=false, threaded=false)

Analyze every particle in a TestParticle ensemble or a vector of trajectories
(including the vector of single-member ensembles returned by `trace_forward`).
Returns results in input order, using `field_work` or, if `profiles=true`,
`Electric_field_work_profile`. Build interpolators once and reuse them. Threaded mode
requires thread-safe, read-only interpolators. No files are saved.
"""
function particle_field_work(solutions, itp::FieldWorkInterpolators;
        species = "O2+", profiles = false, threaded = false)
    hasproperty(solutions, :t) &&
        throw(ArgumentError("Use field_work or Electric_field_work_profile for a single trajectory"))
    trajectories = solutions isa AbstractVector ? solutions : solutions.u
    analyze = profiles ? Electric_field_work_profile : field_work
    result = Vector{Any}(undef, length(trajectories))
    if threaded
        Threads.@threads for i in eachindex(trajectories)
            result[i] = analyze(trajectories[i], itp; species)
        end
    else
        for i in eachindex(trajectories)
            result[i] = analyze(trajectories[i], itp; species)
        end
    end
    return result
end

"""Compatibility alias for [`Electric_field_work_profile`](@ref)."""
const field_work_profile = Electric_field_work_profile


"""
    trajectory_work(sol, itp; species="O2+", mode=:steps, power=false)

Midpoint signed work on supplied intervals, in eV. N states give N-1 increments.
Use full integration cadence before thinning. Summary mode uses constant memory
unless endpoint power is requested. Positive/negative sums depend on cadence.
"""
function trajectory_work(sol,itp;species="O2+",mode=:steps,power=false)
    mode in (:steps,:summary) || throw(ArgumentError("Invalid work mode"))
    tr=_work_trajectory(sol);sp=TP.SpeciesDict[species];n=length(tr.t)
    fields=(itp.total,itp.conv,itp.hall)
    increments=mode==:steps ? zeros(n-1,3) : nothing
    powers=power ? zeros(n,3) : nothing
    totals=zeros(3);positive=zeros(3);negative=zeros(3)
    kinetic(u)=sp.m*sum(abs2,u[4:6])/(2TP.eV)
    k0=kinetic(first(tr.u));k1=kinetic(last(tr.u));maxresidual=0.
    for i in 1:n
        b=tr.u[i]
        if power
            for j in 1:3
                powers[i,j]=_work_one(fields[j],SA[b[1],b[2],b[3]],SA[b[4],b[5],b[6]],sp.q,1.)/TP.eV
            end
        end
        i==1 && continue
        a=tr.u[i-1]
        pm=SA[(a[1]+b[1])/2,(a[2]+b[2])/2,(a[3]+b[3])/2]
        vm=SA[(a[4]+b[4])/2,(a[5]+b[5])/2,(a[6]+b[6])/2]
        for j in 1:3
            dw=_work_one(fields[j],pm,vm,sp.q,tr.t[i]-tr.t[i-1])/TP.eV
            isfinite(dw) || error("Nonfinite work")
            totals[j]+=dw;positive[j]+=max(dw,0);negative[j]+=min(dw,0)
            increments===nothing || (increments[i-1,j]=dw)
        end
        maxresidual=max(maxresidual,abs(kinetic(b)-k0-totals[1]))
    end
    summary=(;total_eV=totals[1],conv_eV=totals[2],hall_eV=totals[3],
        positive_total_eV=positive[1],positive_conv_eV=positive[2],positive_hall_eV=positive[3],
        negative_total_eV=negative[1],negative_conv_eV=negative[2],negative_hall_eV=negative[3],
        initial_kinetic_eV=k0,final_kinetic_eV=k1,delta_kinetic_eV=k1-k0,
        energy_residual_eV=k1-k0-totals[1],max_abs_energy_residual_eV=maxresidual,
        field_sum_residual_eV=totals[1]-totals[2]-totals[3])
    return (;summary,increments,powers)
end

"""
    adiabatic_work_profile(sol, param; relativistic=false)

Integrate TestParticle v0.26's gyro-averaged guiding-center / adiabatic work rates
(`TP.get_work_rates` and `TP.get_work_channels`) along a trajectory `sol` using
midpoint quadrature, returning cumulative work [eV] and instantaneous power [eV/s]
for the parallel (`E_parallel`), Fermi (curvature drift), grad-B, local betatron,
and combined betatron (`grad + betatron_local`) channels.
"""
function adiabatic_work_profile(sol, fields::MHDFields; species = "O2+", relativistic::Bool = false)
    return adiabatic_work_profile(sol, mhd_param(fields; species); relativistic)
end

function adiabatic_work_profile(sol, param; relativistic::Bool = false)
    traj = _work_trajectory(sol)
    n = length(traj.t)
    rates_eV_s = zeros(n, 4)
    work_eV = zeros(n, 4)
    kappa = zeros(n)
    for i in 1:n
        u = SVector{6, Float64}(traj.u[i][1:6])
        r_W = TP.get_work_rates(u, param, traj.t[i]; relativistic)
        kappa[i] = TP.get_adiabaticity(u, param, traj.t[i])
        for j in 1:4
            rates_eV_s[i, j] = r_W[j] / TP.eV
        end
        if i > 1
            u_prev = SVector{6, Float64}(traj.u[i - 1][1:6])
            u_mid = 0.5 * (u_prev + u)
            t_mid = 0.5 * (traj.t[i - 1] + traj.t[i])
            dt = traj.t[i] - traj.t[i - 1]
            r_mid_W = TP.get_work_rates(u_mid, param, t_mid; relativistic)
            for j in 1:4
                work_eV[i, j] = work_eV[i - 1, j] + r_mid_W[j] * dt / TP.eV
            end
        end
    end
    parallel_eV = work_eV[:, 1]
    fermi_eV = work_eV[:, 2]
    grad_eV = work_eV[:, 3]
    betatron_local_eV = work_eV[:, 4]
    betatron_eV = grad_eV .+ betatron_local_eV
    total_adiabatic_eV = parallel_eV .+ fermi_eV .+ betatron_eV
    summary = (;
        parallel_eV = last(parallel_eV),
        fermi_eV = last(fermi_eV),
        grad_eV = last(grad_eV),
        betatron_local_eV = last(betatron_local_eV),
        betatron_eV = last(betatron_eV),
        total_adiabatic_eV = last(total_adiabatic_eV),
    )
    return (;
        t = collect(traj.t),
        parallel_eV,
        fermi_eV,
        grad_eV,
        betatron_local_eV,
        betatron_eV,
        total_adiabatic_eV,
        kappa,
        power_parallel_eV_s = rates_eV_s[:, 1],
        power_fermi_eV_s = rates_eV_s[:, 2],
        power_grad_eV_s = rates_eV_s[:, 3],
        power_betatron_local_eV_s = rates_eV_s[:, 4],
        power_betatron_eV_s = rates_eV_s[:, 3] .+ rates_eV_s[:, 4],
        summary,
    )
end
