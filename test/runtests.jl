using MarsTP, StaticArrays, Test
include("monte_carlo_weight.jl")
include("forward_psd.jl")
include("forward_backend.jl")
include("forward_domain_gpu.jl")
include("forward_bounded.jl")
include("detector_omni_def.jl")
include("trajectory_io.jl")

@testset "Electric work: Cartesian frame and signed energy" begin
    fields = MarsTP.FieldWorkInterpolators(_ -> SA[1., 0., 0.],
        _ -> SA[2., 0., 0.], _ -> SA[-1., 0., 0.])
    traj = (; t = [0., 1.], u = [SA[0., 2., 0., 3., 0., 0.], SA[3., 2., 0., 3., 0., 0.]])
    w = field_work(traj, fields)
    @test w.total_eV ≈ 3
    @test w.conv_eV ≈ 6
    @test w.hall_eV ≈ -3
    @test field_work((; u = [traj]), fields) == w
    @test field_work_profile === Electric_field_work_profile
    p = Electric_field_work_profile(traj, fields)
    @test p.total_eV ≈ [0, 3]
    @test p.conv_eV ≈ [0, 6]
    @test p.hall_eV ≈ [0, -3]
    @test p.power_conv_eV_s ≈ [6, 6]
    @test p.power_hall_eV_s ≈ [-3, -3]
    @test p.energy_residual_eV ≈ [0, -3] # imposed constant velocity, not a dynamical solution
    @test p.field_sum_residual_eV ≈ [0, 0] atol=1e-12
    reverse_traj = (; t = reverse(traj.t), u = reverse(traj.u))
    @test field_work(reverse_traj, fields).total_eV ≈ -3
    @test Electric_field_work_profile(reverse_traj, fields).total_eV ≈ [0, -3]
    single = (; t = [0.], u = [traj.u[1]])
    @test field_work(single, fields).total_eV == 0
    @test Electric_field_work_profile(single, fields).total_eV == [0]
    @test_throws ArgumentError field_work((; u=[traj,traj]),fields)
    expected = [w,w]
    @test particle_field_work([traj,traj],fields) == expected
    @test particle_field_work((;u=[traj,traj]),fields;threaded=true) == expected
    @test particle_field_work([(;u=[traj]),(;u=[traj])],fields) == expected
    profiles = particle_field_work([traj,traj],fields;profiles=true,threaded=true)
    @test all(x -> x.summary == p.summary, profiles)
    @test isempty(particle_field_work([],fields))
    @test_throws ArgumentError particle_field_work(traj,fields)
    @test_throws ArgumentError field_work((;t=[0.,0.],u=traj.u),fields)
    @test_throws ArgumentError field_work((;t=Float64[],u=[]),fields)
    @test_throws ArgumentError field_work((;t=[0.,NaN],u=traj.u),fields)
    @test_throws ArgumentError field_work((;t=[0.],u=[SA[0.,0.,0.,NaN,0.,0.]]),fields)
    invalid = MarsTP.FieldWorkInterpolators(_ -> SA[NaN,0.,0.], fields.conv, fields.hall)
    @test_throws ErrorException field_work(traj,invalid)
    @test_throws ErrorException Electric_field_work_profile(traj,invalid)
end

@testset "Uniform electric acceleration: energy closure" begin
    sp = MarsTP.TP.SpeciesDict["O2+"]
    efield = 1e-5
    acceleration = sp.q * efield / sp.m
    t = [0.,0.1,0.4,1.]
    u = [SA[0.5acceleration*s^2, 2., 0., acceleration*s, 0., 0.] for s in t]
    fields = MarsTP.FieldWorkInterpolators(_ -> SA[efield,0.,0.],
        _ -> SA[0.8efield,0.,0.], _ -> SA[0.2efield,0.,0.])
    p = Electric_field_work_profile((;t,u),fields)
    @test p.total_eV ≈ p.delta_kinetic_eV
    @test p.conv_eV ≈ 0.8p.total_eV
    @test p.hall_eV ≈ 0.2p.total_eV
    @test maximum(abs,p.energy_residual_eV) < 1e-12
end

include("backtrace_ionosphere.jl")

@testset "Backtrace work physical sign" begin
    for sign in (-1.,1.), solver in (:boris,:adaptive)
        E=SA[0.,0.,sign*1e-5]
        fields=MarsTP.FieldWorkInterpolators(_->E,_->2E,_->-E)
        param=MarsTP.TP.prepare(;E=_->E,B=_->SA[0.,0.,1e-9],species=MarsTP.TP.SpeciesDict["O2+"])
        cfg=BacktraceConfig(include_ionosphere=false,solver=solver,dt=-.03,tspan=(0.,-.97))
        observed=zeros(3)
        observe=(a,b,ta,tb,dw)->begin
            @test tb < ta
            observed .+= dw
        end
        r=MarsTP._trace_sources(SA[0.,0.,2Rm],SA[0.,0.,1e4],param,cfg,(p,v)->0.,nothing;work_itp=fields,segment_observer=observe)
        @test observed ≈ r.work_eV
        @test sign*r.work_eV[1]>0
        @test r.work_eV[1] ≈ r.delta_kinetic_eV rtol=1e-8
        @test r.work_eV[2] ≈ 2r.work_eV[1]
        @test r.work_eV[3] ≈ -r.work_eV[1]
    end
end

include("visualization.jl")
