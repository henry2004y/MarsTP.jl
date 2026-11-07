using KernelAbstractions, LinearAlgebra
@testset "Forward backend contract and spherical interpolation" begin
    r = collect(range(Rinner,Router;length=8))
    theta = collect(range(0.,pi;length=9))
    phi = collect(range(0.,2pi;length=17))
    E = zeros(3,length(r),length(theta),length(phi))
    B = similar(E)
    for k in eachindex(phi),j in eachindex(theta),i in eachindex(r)
        B[:,i,j,k] .= [0.,0.,1e-8]
    end
    MarsTP._rotate_vectors_to_spherical!(B,r,theta,phi)
    fields = MHDFields(r,theta,phi,E,B,:total,"synthetic_uniform_Bz")
    cfg = ForwardTraceConfig(species="H+",solver=:boris,dt=.01,tspan=(0.,1.))
    states = [SA[1.5Rm,.1Rm,.4Rm,1e4,0.,0.],SA[1.5Rm,.1Rm,.4Rm,0.,1e4,0.]]
    serial = trace_forward(states;config=cfg,fields,saveat=.1)
    parallel = trace_forward(states;config=cfg,fields,backend=KernelAbstractions.CPU(),saveat=.1)
    @test length(serial) == length(parallel) == 2
    for (a,b) in zip(serial,parallel)
        @test length(a.u) == length(b.u) == 1
        sa,sb = only(a.u),only(b.u)
        @test sa.t ≈ sb.t
        @test sa.u ≈ sb.u
        @test maximum(abs(sum(abs2,s[4:6])/1e8-1) for s in sb.u) < 1e-10
    end
    @test_throws ArgumentError trace_forward(states;fields,backend=KernelAbstractions.CPU())
    @test_throws ArgumentError trace_forward(states;fields,config=ForwardTraceConfig(solver=:other))
    outside = only(only(trace_forward([SA[0.,0.,0.,1.,0.,0.]];fields,config=cfg)).u)
    @test outside.retcode == MarsTP.TP.ReturnCode.Terminated
    @test outside.t == [0.]
    @test isempty(trace_forward([];fields,config=cfg))
    for backend in (KernelAbstractions.CPU(),)
        exit = only(only(trace_forward([SA[Router-1,0.,0.,1e5,0.,0.]];
            fields,config=cfg,backend)).u)
        @test exit.retcode == MarsTP.TP.ReturnCode.Terminated
        @test all(u -> all(isfinite,u), exit.u)
        @test norm(last(exit.u)[1:3]) <= Router
        @test last(exit.t) < cfg.tspan[2]
    end
end

@testset "0.24 backtrace accepts boundary before invalid endpoint fields" begin
    # 0.24 synchronizes the node velocity before the boundary callback. At an
    # out-of-grid node that velocity is NaN, while the drift position is valid.
    E = x -> Rinner <= norm(x) <= Router ? SA[0.,0.,0.] : SA[NaN,NaN,NaN]
    B = x -> Rinner <= norm(x) <= Router ? SA[0.,0.,1e-9] : SA[NaN,NaN,NaN]
    p = MarsTP.TP.prepare(; E=E, B=B, species=MarsTP.TP.SpeciesDict["O2+"])
    for solver in (:boris,:adaptive)
        cfg = BacktraceConfig(include_ionosphere=false,solver=solver,dt=-.2,tspan=(0.,-1.),safety=.001)
        for (z,vz,status,radius) in ((Router-100.,-1000.,4,Router),(Rinner+100.,1000.,2,Rinner))
            result = MarsTP._trace_sources(SA[0.,0.,z],SA[0.,0.,vz],p,cfg,(x,v)->3.,nothing)
            @test result.status == status
            @test result.time ≈ -.1 atol=1e-8
            @test result.position[3] ≈ radius
            @test result.velocity ≈ SA[0.,0.,vz]
            @test result.volume ≈ .3 atol=1e-8
        end
    end
end
