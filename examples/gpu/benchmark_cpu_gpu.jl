#=
Run from MarsTP.jl: julia --threads=4 --project=. examples/gpu/benchmark_cpu_gpu.jl
Float64, static fields, nonrelativistic dynamics, identical saved cadence.
Times cover the complete warmed solve, including allocation, field adaptation,
GPU transfers and host solution construction; exclude field file I/O and JIT.
=#
using MarsTP, TestParticle, CUDA, KernelAbstractions, StaticArrays
using Random, Statistics, Dates, TOML, SHA, LinearAlgebra
const TP = TestParticle
const BP = MarsTP.BorisPushers
const KA = KernelAbstractions

function timed_solve(f; repeats=3)
    f() # warm the exact particle count and dispatch
    times = Float64[]
    last = nothing
    for _ in 1:repeats
        GC.gc()
        CUDA.synchronize()
        t = time_ns()
        last = f()
        CUDA.synchronize()
        push!(times, (time_ns()-t)*1e-9)
    end
    return last, times
end

function differences(a,b)
    @assert length(a) == length(b)
    dx, dv = 0., 0.
    for (sa,sb) in zip(a,b)
        @assert sa.retcode == sb.retcode == TP.ReturnCode.Success
        @assert length(sa.u) == length(sb.u)
        @assert sa.t ≈ sb.t
        for (ua,ub) in zip(sa.u,sb.u)
            @assert all(isfinite,ua) && all(isfinite,ub)
            dx = max(dx,norm(ua[1:3]-ub[1:3]))
            dv = max(dv,norm(ua[4:6]-ub[4:6]))
        end
    end
    return (;position_m=dx, velocity_m_s=dv)
end

function uniform_problem(states; dt=.01)
    p = TP.prepare(; E = _ -> SA[0.,0.,0.], B = _ -> SA[0.,0.,1e-8], species=TP.Proton)
    pf = (prob,ctx) -> TP.remake(prob;u0=states[ctx.sim_id])
    return TP.TraceProblem(first(states),(0.,10.),p;prob_func=pf)
end

function physics_checks(gpu)
    state = SA[0.,0.,0.,1e5,0.,2e4]
    prob = uniform_problem([state])
    sp = TP.Proton
    omega = sp.q*1e-8/sp.m
    errors = Float64[]
    energy = 0.
    for dt in (.02,.01)
        sol = only(TP.solve(prob,BP.Boris(),gpu;dt,trajectories=1,saveat=.1).u)
        expected = SA[1e5/omega*sin(omega*10),1e5/omega*(cos(omega*10)-1),2e5]
        push!(errors,norm(sol.u[end][1:3]-expected))
        energy = maximum(abs(sum(abs2,u[4:6])/sum(abs2,state[4:6])-1) for u in sol.u)
        @assert sol.u[2][5] < 0 # positive proton, +Bz, initial +Vx
        @assert energy < 1e-10
    end
    @assert errors[2] < .3errors[1]
    zero_p = TP.prepare(; E = _->SA[0.,0.,0.], B = _->SA[0.,0.,0.], species=TP.Proton)
    zero_prob = TP.TraceProblem(state,(0.,1.),zero_p)
    zero_sol = only(TP.solve(zero_prob,BP.Boris(),gpu;dt=.01,saveat=.1).u)
    @assert zero_sol.u[end] ≈ SA[1e5,0.,2e4,1e5,0.,2e4]
    return Dict("energy_relative_error"=>energy,"position_error_dt_002_m"=>errors[1],
        "position_error_dt_001_m"=>errors[2],"zero_field_passed"=>true,
        "gyration_direction_passed"=>true,"gyroperiod_s"=>2pi/omega)
end

function main()
    @assert CUDA.functional() "CUDA is unavailable"
    CUDA.allowscalar(false)
    gpu = CUDA.CUDABackend()
    rng = Xoshiro(20261008)
    out = joinpath(MarsTP.project_path("outputs"),"gpu_benchmark_"*Dates.format(now(),"yyyymmdd_HHMMSS"))
    mkpath(out)
    metadata = Dict{String,Any}("julia"=>string(VERSION),"TestParticle"=>string(pkgversion(TP)),
        "CUDA"=>string(pkgversion(CUDA)),"KernelAbstractions"=>string(pkgversion(KA)),
        "MarsTP_commit"=>strip(read(`git -C $(MarsTP.project_path()) rev-parse HEAD`,String)),
        "gpu"=>CUDA.name(CUDA.device()),"cpu"=>Sys.cpu_info()[1].model,
        "threads"=>Threads.nthreads(),"precision"=>"Float64","seed"=>20261008,
        "timing"=>"median of 3 warmed complete solves, includes upload/download/output; excludes VTK I/O and JIT",
        "uniform_dt_s"=>.01,"uniform_tspan_s"=>[0.,10.],"uniform_saveat_s"=>1.,
        "mhd_dt_s"=>.001,"mhd_tspan_s"=>[0.,1.],"mhd_saveat_s"=>.1,
        "mars_radius_m"=>Rm,"mhd_species"=>"O2+","electric_field"=>"total",
        "coordinate_basis"=>"Cartesian positions/velocities, same axes as input MHD; spherical grid uses colatitude/radians",
        "physics"=>physics_checks(gpu))
    rows = Vector{Dict{String,Any}}()
    println("Physics checks passed; loading MHD once..."); flush(stdout)
    load_seconds = @elapsed fields = load_mhd_fields()
    metadata["field_io_seconds"] = load_seconds
    metadata["field_path"] = fields.path
    metadata["field_sha256"] = open(io->bytes2hex(sha256(io)),fields.path)
    metadata["grid_size"] = collect(size(fields.B))
    cfg = ForwardTraceConfig(species="O2+",solver=:boris,dt=.001,tspan=(0.,1.))
    for n in (100,1000,10000)
        uniform_states = [SA[0.,0.,0.,1e5*cos(a),1e5*sin(a),2e4] for a in 2pi*rand(rng,n)]
        prob = uniform_problem(uniform_states)
        mhd_states = [SA[1.5Rm,.1Rm,.4Rm,1e4*randn(rng),1e4*randn(rng),1e4*randn(rng)] for _ in 1:n]
        for kind in ("uniform","MarsTP_MHD")
            results = Dict{String,Any}()
            for mode in ("cpu_serial","cpu_threads","gpu")
                f = if kind == "uniform"
                    alg = mode == "cpu_serial" ? TP.EnsembleSerial() : mode == "cpu_threads" ? TP.EnsembleThreads() : gpu
                    () -> TP.solve(prob,BP.Boris(),alg;dt=.01,trajectories=n,saveat=1.).u
                else
                    backend = mode == "cpu_serial" ? nothing : mode == "cpu_threads" ? KA.CPU() : gpu
                    () -> [only(s.u) for s in trace_forward(mhd_states;config=cfg,fields,backend,saveat=.1)]
                end
                sols,times = timed_solve(f)
                results[mode] = sols
                push!(rows,Dict("case"=>kind,"particles"=>n,"mode"=>mode,
                    "seconds"=>times,"median_seconds"=>median(times)))
                println(kind," n=",n," ",mode," median=",round(median(times);digits=6)," s"); flush(stdout)
            end
            d = differences(results["cpu_serial"],results["gpu"])
            @assert d.position_m < .01 && d.velocity_m_s < .01
            metadata["$(kind)_$(n)_agreement"] = Dict("max_position_m"=>d.position_m,"max_velocity_m_s"=>d.velocity_m_s)
            threaded = differences(results["cpu_serial"],results["cpu_threads"])
            @assert threaded.position_m < .01 && threaded.velocity_m_s < .01
            println("Agreement: ",d); flush(stdout)
            open(joinpath(out,"results.toml"),"w") do io
                TOML.print(io,merge(metadata,Dict("measurements"=>rows)))
            end
        end
    end
    open(joinpath(out,"timings.csv"),"w") do io
        println(io,"case,particles,mode,median_seconds,trial_1_s,trial_2_s,trial_3_s")
        for r in rows
            println(io,join([r["case"],r["particles"],r["mode"],r["median_seconds"],r["seconds"]...],","))
        end
    end
    println("RESULT_DIRECTORY=",out)
end
main()
