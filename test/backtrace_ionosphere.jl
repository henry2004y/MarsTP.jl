using MarsTP, StaticArrays, Test, LinearAlgebra
const TP = MarsTP.TP
@testset "MHD ionosphere definition and units" begin
    source = IonosphereSource(_->2., _->1000., (_->-3., _->4., _->0.), Rm+400e3)
    p = ionosphere_properties(source, SA[1.,0.,0.])
    @test p.flux == 10
    @test p.Ui == SA[-3.,4.,0.]
    sp = TP.SpeciesDict["O2+"]
    vth = sqrt(2TP.kB*1000/sp.m)
    distribution=ionosphere_distribution(source,SA[1.,0.,0.],p.Ui)
    @test distribution.g ≈ 1/(pi^1.5*vth^3)
    @test distribution.f ≈ 2distribution.g
    @test MarsTP._surface_increment(distribution,SA[1.,0.,0.],SA[-3.,4.,0.]) ≈ 10distribution.g/3
    @test_throws ErrorException MarsTP._surface_increment(distribution,SA[1.,0.,0.],SA[0.,4.,0.])
    @test_throws ArgumentError MarsTP._ionosphere_radius(199.)
    @test_throws ArgumentError MarsTP._ionosphere_radius(801.)
    @test MarsTP._ionosphere_radius(200.) == Rinner
    @test MarsTP._ionosphere_radius(800.) == Rm+800e3
    @test BacktraceConfig().ionosphere_altitude_km == 400.
end
@testset "Thin sheet crossings and fixed inner boundary" begin
    param = TP.prepare(; E = _->SA[0.,0.,0.], B = _->SA[0.,0.,1e-9], species=TP.SpeciesDict["O2+"])
    source=(p,v)->(;g=7.,flux=8.)
    # The shell is crossed at t=-1, integration continues to the same 200 km
    # inner boundary for every chosen source altitude and both solvers.
    for alg in (:boris,:adaptive), h in (200.,400.,800.)
        flight=h-200+1
        cfg=BacktraceConfig(ionosphere_altitude_km=h,solver=alg,dt=-.2,tspan=(0.,-flight-1))
        r=Rm+h*1e3
        result=MarsTP._trace_sources(SA[0.,0.,r+1000],SA[0.,0.,1000.],param,cfg,(p,v)->3.,source)
        @test result.status == 2
        @test result.time ≈ -flight atol=1e-7
        @test norm(result.position) ≈ Rinner
        @test result.volume ≈ 3flight atol=1e-6
        @test result.ionosphere ≈ 56/1000
        @test result.crossings == 1
        @test result.velocity ≈ SA[0.,0.,1000.]
    end
    cfg=BacktraceConfig(tspan=(0.,-.93),dt=-.2)
    result=MarsTP._trace_sources(SA[0.,0.,2Rm],SA[0.,0.,1000.],param,cfg,(p,v)->3.,source)
    @test result.status == 1
    @test result.time ≈ -.93
    @test result.volume ≈ 2.79
    @test result.ionosphere == 0
    result=MarsTP._trace_sources(SA[0.,0.,Router-100],SA[0.,0.,-1000.],param,cfg,(p,v)->3.,source)
    @test result.status == 4
    @test result.time ≈ -.1 atol=1e-8
    @test result.volume ≈ .3 atol=1e-8
    @test MarsTP._backtrace_crossing(SA[2.,0.,0.],SA[-2.,0.,0.],1.,4.) == (.25,2)
    @test MarsTP._shell_crossings(SA[2.,0.,0.],SA[-2.,0.,0.],1.) ≈ [.25,.75]
    @test_throws ErrorException MarsTP._shell_crossings(SA[2.,1.,0.],SA[-2.,1.,0.],1.)
    # Below the sheet, an inward backtrace sees volume only.
    cfg=BacktraceConfig(tspan=(0.,-101.),dt=-.2,detector_Rm=SA[0.,0.,(Rm+300e3)/Rm])
    @test isnothing(MarsTP._validate_backtrace(cfg))
    result=MarsTP._trace_sources(cfg.detector_Rm*Rm,SA[0.,0.,1000.],param,cfg,(p,v)->3.,source)
    @test result.status == 2
    @test result.volume ≈ 300
    @test result.ionosphere == 0
    @test result.crossings == 0
    # Outward backtracing from below the sheet may acquire a source contribution.
    result=MarsTP._trace_sources(cfg.detector_Rm*Rm,SA[0.,0.,-1000.],param,cfg,(p,v)->3.,source)
    @test result.status == 1
    @test result.ionosphere ≈ 56/1000
    @test result.crossings == 1
    # Turning off the sheet still permits reaching 200 km.
    cfg=BacktraceConfig(tspan=(0.,-202.),include_ionosphere=false)
    result=MarsTP._trace_sources(SA[0.,0.,Rm+401e3],SA[0.,0.,1000.],param,cfg,(p,v)->3.,source)
    @test result.status == 2
    @test result.volume ≈ 603
    @test result.ionosphere == 0
    @test result.crossings == 0
end
@testset "Repeated oblique crossings and partial-step volume" begin
    param=TP.prepare(; E = _->SA[0.,0.,0.], B = _->SA[1e-9,0.,0.], species=TP.SpeciesDict["O2+"])
    z=Rm+300e3
    speed=1e6
    cfg=BacktraceConfig(tspan=(0.,-4Rm/speed),dt=-.1)
    r=MarsTP._trace_sources(SA[2Rm,0.,z],SA[speed,0.,0.],param,cfg,(p,v)->3.,(p,v)->(;g=7.,flux=8.))
    @test r.status==1
    @test r.crossings==2
    vr=speed*sqrt(1-(z/(Rm+400e3))^2)
    @test r.ionosphere ≈ 2*56/vr rtol=1e-10
    @test r.volume ≈ 3*4Rm/speed
end
@testset "Synchronized sheet velocity with electric acceleration" begin
    sp=TP.SpeciesDict["O2+"]
    acceleration=100.
    param=TP.prepare(; E = _->SA[0.,0.,acceleration*sp.m/sp.q], B = _->SA[0.,0.,1e-9], species=sp)
    radius=Rm+400e3
    exact_time=(-1000+sqrt(1000^2-2acceleration*1000))/acceleration
    errors=Float64[]
    for dt in (-.1,-.05)
        cfg=BacktraceConfig(dt=dt,tspan=(0.,-2.))
        hit_velocities=Float64[]
        function source(p,v)
            push!(hit_velocities,v[3])
            return (;g=1.,flux=1.)
        end
        r=MarsTP._trace_sources(SA[0.,0.,radius+1000],SA[0.,0.,1000.],param,cfg,(p,v)->0.,source)
        @test r.status==1
        @test r.crossings==1
        @test r.velocity[3] ≈ 1000+acceleration*r.time atol=1e-7
        @test r.ionosphere ≈ 1/only(hit_velocities)
        push!(errors,abs((only(hit_velocities)-1000)/acceleration-exact_time))
    end
    @test errors[2] < errors[1]
    @test errors[2] < 1e-4
end
