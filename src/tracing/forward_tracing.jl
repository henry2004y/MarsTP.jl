Base.@kwdef struct ForwardTraceConfig
    species::String = "O2+"
    mhd_file::String = data_path("mars_fields_spherical_from_dat.vts")
    tspan::Tuple{Float64, Float64} = (0.0, 500.0)
    solver::Symbol = :adaptive
    dt::Float64 = 0.2
    safety::Float64 = 1 / 32
end

"""
    trace_forward(initial_states; config, fields=nothing, backend=nothing, saveat=())

Cartesian states are `[x,y,z,vx,vy,vz]` in m and m/s. Pass cached `MHDFields`
to avoid repeated VTK I/O. A hardware `backend` (e.g. `CUDA.CUDABackend()`)
uses the TestParticle 0.24 ensemble kernel and requires `solver=:boris`.
The return value preserves MarsTP's vector of single-member ensembles.

Absorption below 200 km altitude and radial field-domain exits stop at the last accepted in-domain state with
`ReturnCode.Terminated`. Nonfinite saved states raise an error.
Finite initial positions below 200 km or outside the radial mesh terminate at the initial time
without querying or extrapolating fields. Other particles continue normally.
Use `trace_forward_bounded` for per-particle spherical absorption and escape.
CPU backtracing/source accumulation is separate from this forward API.
"""
function trace_forward(initial_states; config::ForwardTraceConfig = ForwardTraceConfig(),
        fields=nothing, backend=nothing, saveat=())
    config.solver in (:boris, :adaptive) || throw(ArgumentError("Unknown solver"))
    backend === nothing || backend isa KA.Backend || throw(ArgumentError("Expected a KernelAbstractions backend"))
    backend === nothing || config.solver == :boris || throw(ArgumentError("GPU/backend tracing requires solver=:boris"))
    all(isfinite, config.tspan) && config.tspan[2] > config.tspan[1] ||
        throw(ArgumentError("Forward tspan must be finite and increasing"))
    config.solver != :boris || isfinite(config.dt) && config.dt > 0 ||
        throw(ArgumentError("Forward Boris dt must be positive and finite"))
    states = [SVector{6,Float64}(s) for s in initial_states]
    isempty(states) && return []
    all(s -> all(isfinite,s), states) || throw(ArgumentError("Nonfinite initial state"))
    fields = fields === nothing ? load_mhd_fields(config.mhd_file; electric_field = :total) : fields
    original_states = states
    lower_radius = max(Rinner,first(fields.r))
    active = findall(s -> lower_radius <= norm(s[1:3]) <= last(fields.r),states)
    states = states[active]
    param = mhd_param(fields; species = config.species)
    # Capture scalar radii only, so the predicate can execute on a GPU.
    # Check position before velocity: endpoint field interpolation may be NaN
    # after a finite drift has left the mesh. Interior NaNs remain failures.
    rmin, rmax = lower_radius, last(fields.r)
    isoutside = let lo=rmin, hi=rmax
        (u,p,t) -> begin
            radius = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
            isfinite(radius) && (radius < lo || radius > hi)
        end
    end
    solver = config.solver == :boris ? BorisPushers.Boris() : BorisPushers.AdaptiveBoris(; safety = config.safety)
    boris_ext = Base.get_extension(TP, :TestParticleBorisPushersExt)
    sols = if isempty(states)
        []
    elseif backend === nothing
        map(states) do state
            prob = TP.TraceProblem(state, config.tspan, param)
            # Do not save the rejected trial node (its velocity may be NaN).
            # Restore both time and state, matching the device rollback policy.
            condition = (u,t,integrator) -> isoutside(u,integrator.p,t)
            stop = integrator -> begin
                # SciML can save/interpolate a trial step before discrete
                # callbacks. Discard only output belonging to that rejected
                # step, then save the restored endpoint exactly once.
                while !isempty(integrator.sol.t) && last(integrator.sol.t) >= integrator.tprev
                    pop!(integrator.sol.t)
                    pop!(integrator.sol.u)
                end
                integrator.saveiter = length(integrator.sol.t)
                integrator.u = integrator.uprev
                integrator.t = integrator.tprev
                TP.terminate!(integrator)
            end
            callback = TP.DiscreteCallback(condition,stop;save_positions=(false,true))
            # TraceProblem.solve supplies its own callback keyword; use the
            # same upstream ODE adapter directly so our controlled rollback runs.
            ode_prob = boris_ext._boris_problem(prob)
            dt = config.solver == :boris ? config.dt : boris_ext._boris_initial_dt(prob,solver)
            TP.solve(ode_prob,solver;dt,saveat,callback,dense=false,
                save_everystep=isempty(saveat))
        end
    else
        prob = TP.TraceProblem(first(states), config.tspan, param)
        # Upstream bulk input layout is (particles, 6), not (6, particles).
        u0 = Matrix{Float64}(undef, length(states), 6)
        for i in eachindex(states), j in 1:6
            u0[i,j] = states[i][j]
        end
        TP.solve(prob, solver, backend; dt=config.dt, trajectories=length(states),
            u0, saveat, isoutside, save_everystep=isempty(saveat)).u
    end
    for (i,sol) in enumerate(sols)
        sol.retcode in (TP.ReturnCode.Success, TP.ReturnCode.Terminated) || error("Forward particle $i failed: $(sol.retcode)")
        all(s -> all(isfinite,s), sol.u) || error("Forward particle $i has nonfinite states, possibly left the field domain")
    end
    accepted = Dict(zip(active,sols))
    return map(eachindex(original_states)) do i
        sol = if haskey(accepted,i)
            accepted[i]
        else
            state = original_states[i]
            prob = boris_ext._boris_problem(TP.TraceProblem(state,config.tspan,param))
            t = [config.tspan[1]]
            u = [state]
            TP.build_solution(prob,solver,t,u;interp=TP.LinearInterpolation(t,u),
                retcode=TP.ReturnCode.Terminated)
        end
        TP.EnsembleSolution([sol],0.0,true)
    end
end
