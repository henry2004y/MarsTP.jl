# One device work item per particle. Boundary checks precede endpoint field
# evaluation, so an overshooting Boris drift never queries outside the mesh.
using KernelAbstractions: @kernel, @index

@kernel function _bounded_boris_kernel!(history, times, counts, flags, diagnostics, initial,
        param, work_fields, charge_eV, mass_eV, t0, dt, nsteps, inner, outer, save_every)
    i = @index(Global, Linear)
    x = SVector(initial[i,1], initial[i,2], initial[i,3])
    v = SVector(initial[i,4], initial[i,5], initial[i,6])
    t = t0
    count = 1
    radius = norm(x)
    tolerance = 64eps(max(inner,outer))
    radial_velocity = dot(x,v)
    # A reconstructed shell point can lie a few ulps on either side. Classify
    # its physical direction instead of absorbing an outward launch by roundoff.
    flag = radius < inner-tolerance || (radius <= inner+tolerance && radial_velocity < 0) ? 2 :
           radius > outer+tolerance || (radius >= outer-tolerance && radial_velocity > 0) ? 4 : 1
    totals = SVector(0.,0.,0.)
    positive = totals
    negative = totals
    k0 = mass_eV*dot(v,v)/2
    maxresidual = 0.
    maxgyro = 0.
    times[i,1] = t
    for j in 1:3
        history[i,j,1] = x[j]
        history[i,j+3,1] = v[j]
    end
    query0 = norm(x) >= outer ? x*((outer-1e-6)/norm(x)) :
             norm(x) <= inner ? x*((inner+1e-6)/norm(x)) : x
    half = flag == 1 ? BorisPushers.update_velocity_half(v,query0,dt,t,param,BorisPushers.Boris()) : v
    for step in 1:(flag == 1 ? nsteps : 0)
        query0 = norm(x) >= outer ? x*((outer-1e-6)/norm(x)) :
                 norm(x) <= inner ? x*((inner+1e-6)/norm(x)) : x
        maxgyro = max(maxgyro,abs(param.q2m)*norm(param.B(query0,t))*dt)
        next_half = BorisPushers.update_velocity(half,query0,dt,t,param,BorisPushers.Boris())
        trial = x + next_half*dt
        if !all(isfinite,trial) || !all(isfinite,next_half)
            flag = 5
            break
        end
        fraction, event = _backtrace_crossing(x,trial,inner,outer)
        if event != 1 && fraction == 0
            flag = event
            break
        end
        xn = x + fraction*(trial-x)
        tn = t0 + (step-1+fraction)*dt
        if event != 1 && tn <= t
            flag = event
            break
        end
        # Evaluate on the covered side of the boundary only. Saved position is
        # the actual intersection; this guard is solely for floating point roundoff.
        query = event == 4 ? xn*((outer-1e-6)/norm(xn)) :
                event == 2 ? xn*((inner+1e-6)/norm(xn)) : xn
        vn = BorisPushers.update_velocity(next_half,query,(fraction-0.5)*dt,tn,param,BorisPushers.Boris())
        if !all(isfinite,vn)
            flag = 5
            break
        end
        if work_fields !== nothing
            midpoint = (x+xn)/2
            velocity = (v+vn)/2
            dwork = SVector(dot(work_fields[1](midpoint),velocity),
                dot(work_fields[2](midpoint),velocity),dot(work_fields[3](midpoint),velocity)) * (charge_eV*(tn-t))
            if !all(isfinite,dwork)
                flag = 5
                break
            end
            totals += dwork
            positive += max.(dwork,0.)
            negative += min.(dwork,0.)
            maxresidual = max(maxresidual,abs(mass_eV*dot(vn,vn)/2-k0-totals[1]))
        end
        x, v, t, half = xn, vn, tn, next_half
        flag = event
        if event != 1 || step == nsteps || (save_every > 0 && step % save_every == 0)
            count += 1
            times[i,count] = t
            for j in 1:3
                history[i,j,count] = x[j]
                history[i,j+3,count] = v[j]
            end
        end
        event != 1 && break
    end
    counts[i] = count
    flags[i] = flag
    for j in 1:3
        diagnostics[i,j] = totals[j]
        diagnostics[i,j+3] = positive[j]
        diagnostics[i,j+6] = negative[j]
    end
    diagnostics[i,10] = maxresidual
    diagnostics[i,11] = maxgyro
end

"""
    trace_forward_bounded(initial_states; config, fields=nothing,
        backend=KA.CPU(), inner_radius_m=Rm+200e3, outer_radius_m=Router,
        save_every=1, work_itp=nothing, throw_on_failure=true)

Fixed-step Boris tracing with per-particle absorption and escape on CPU or GPU.
Cartesian states use m and m/s; times use s. Each device thread stops at the
first sphere intersection along a Boris drift, including a transit through the
inner sphere with both drift endpoints outside. Boundary velocities are
synchronized using the clipped fraction of the step, as in the CPU shell model.

Returns a vector of trajectories with `t`, `u`, `status` (:inner, :outer,
:time_limit, :numerical_failure), `retcode`, and `species`. Every successful
trajectory includes its exact termination endpoint.
Finite initial states already inside the absorption sphere or beyond the escape
sphere stop immediately with :inner or :outer, without evaluating fields.
`save_every=0` saves only initial/final states; otherwise save every N integration steps. Use bounded
particle batches for long histories. Sparse histories cannot recover detector
residence or postprocessed work on omitted steps. The interval must contain an
integer number of steps. This implements spherical boundaries, not arbitrary
Julia callbacks, detector events, or source sampling.
With `work_itp=FieldWorkInterpolators(...)`, each accepted integration segment
accumulates signed total, convection and Hall work on the device, in eV, using
midpoint q E dot v dt. The returned `work` summary includes positive/negative
totals and kinetic-energy closure, independent of output thinning. Without
work interpolators, `work` is nothing. `maxgyro` is the maximum step gyro angle.
"""
function trace_forward_bounded(initial_states;
        config::ForwardTraceConfig=ForwardTraceConfig(solver=:boris),
        fields=nothing, backend=KA.CPU(), inner_radius_m=Rm+200e3,
        outer_radius_m=Router, save_every::Integer=1, work_itp=nothing, throw_on_failure=true)
    config.solver == :boris || throw(ArgumentError("Bounded tracing requires solver=:boris"))
    backend isa KA.Backend || throw(ArgumentError("Expected a KernelAbstractions backend"))
    t0,t1 = config.tspan
    all(isfinite,config.tspan) && t1 > t0 && isfinite(config.dt) && config.dt > 0 ||
        throw(ArgumentError("Finite increasing tspan and positive dt required"))
    save_every >= 0 || throw(ArgumentError("save_every must be nonnegative"))
    isfinite(inner_radius_m) && isfinite(outer_radius_m) && 0 < inner_radius_m < outer_radius_m ||
        throw(ArgumentError("Invalid boundary radii"))
    ratio = (t1-t0)/config.dt
    nsteps = round(Int,ratio)
    nsteps > 0 && isapprox(ratio,nsteps;rtol=0,atol=1e-8) ||
        throw(ArgumentError("tspan must contain an integer number of dt steps"))
    states = [SVector{6,Float64}(s) for s in initial_states]
    isempty(states) && return []
    all(s->all(isfinite,s),states) || throw(ArgumentError("Nonfinite initial state"))
    fields = fields === nothing ? load_mhd_fields(config.mhd_file;electric_field=:total) : fields
    # Log-spaced axes can differ from the defining radius by a few ulps.
    # Allow 16 ulps for exp(log(radius)) reconstruction. Event/start queries
    # are guarded on the covered side; this is not physical extrapolation.
    first(fields.r)-16eps(first(fields.r)) <= inner_radius_m < outer_radius_m <= last(fields.r)+16eps(last(fields.r)) ||
        throw(ArgumentError("Field mesh must cover both boundary spheres"))
    p = mhd_param(fields;species=config.species)
    # Upstream adaptation is reused, including spherical interpolation storage.
    param = BorisPushers.adapt_params(p, backend)
    work_fields = work_itp === nothing ? nothing :
        map(f->backend isa KA.CPU ? f : BorisPushers.adapt_field_to_gpu(f,backend),
            (work_itp.total,work_itp.conv,work_itp.hall))
    sp = TP.SpeciesDict[config.species]
    initial = [states[i][j] for i in eachindex(states), j in 1:6]
    device_initial = KA.allocate(backend,Float64,size(initial))
    copyto!(device_initial,initial)
    slots = save_every == 0 ? 2 : fld(nsteps,save_every)+2
    history = KA.allocate(backend,Float64,(length(states),6,slots))
    times = KA.allocate(backend,Float64,(length(states),slots))
    counts = KA.allocate(backend,Int,(length(states),))
    flags = KA.allocate(backend,Int,(length(states),))
    diagnostics = KA.allocate(backend,Float64,(length(states),11))
    _bounded_boris_kernel!(backend,64)(history,times,counts,flags,diagnostics,device_initial,param,
        work_fields,sp.q/TP.eV,sp.m/TP.eV,
        t0,config.dt,nsteps,Float64(inner_radius_m),Float64(outer_radius_m),Int(save_every);
        ndrange=length(states))
    KA.synchronize(backend)
    h,ts,ns,fs = Array(history),Array(times),Array(counts),Array(flags)
    diag = Array(diagnostics)
    labels = (:time_limit,:inner,:unused,:outer,:numerical_failure)
    throw_on_failure && any(==(5),fs) && error("Bounded tracing numerical failure for particles $(findall(==(5),fs))")
    return map(eachindex(states)) do i
        u = [SVector{6,Float64}(h[i,:,j]) for j in 1:ns[i]]
        k0 = sp.m*sum(abs2,states[i][4:6])/(2TP.eV)
        k1 = sp.m*sum(abs2,last(u)[4:6])/(2TP.eV)
        work = work_itp === nothing ? nothing :
            (;total_eV=diag[i,1],conv_eV=diag[i,2],hall_eV=diag[i,3],
              positive_total_eV=diag[i,4],positive_conv_eV=diag[i,5],positive_hall_eV=diag[i,6],
              negative_total_eV=diag[i,7],negative_conv_eV=diag[i,8],negative_hall_eV=diag[i,9],
              initial_kinetic_eV=k0,final_kinetic_eV=k1,delta_kinetic_eV=k1-k0,
              energy_residual_eV=k1-k0-diag[i,1],max_abs_energy_residual_eV=diag[i,10],
              field_sum_residual_eV=diag[i,1]-diag[i,2]-diag[i,3])
        (;t=ts[i,1:ns[i]],u,
        status=labels[fs[i]],retcode=fs[i]==5 ? TP.ReturnCode.Failure : TP.ReturnCode.Success,
        species=config.species,work,maxgyro=diag[i,11])
    end
end
