@testset "Forward detector PSD" begin
    beam = (; t=[0.,2.], u=[SA[-2.,0.,0.,2.,0.,0.], SA[2.,0.,0.,2.,0.,0.]])
    cfg = (; detector_m=[0.,0.,0.], side_m=2., vlim=4., vgrid=4,
        species="O2+", rate_weights_s=[24.]) # Q=n*u*L^2, n=3
    r = forward_psd([beam]; cfg...)
    @test r.residence_s ≈ [1.]
    @test r.density_total_m3 ≈ 3.
    @test r.density_in_range_m3 ≈ 3.
    @test r.density_outside_vlim_m3 == 0
    @test sum(r.psd)*2^3 ≈ 3.
    @test r.psd[4,3,3] ≈ 3/8
    @test forward_psd(beam; cfg...).psd == r.psd
    @test forward_psd((;u=[beam]); cfg...).psd == r.psd
    @test forward_psd([(;u=[beam])]; cfg...).psd == r.psd
    @test forward_psd([beam]; cfg..., vlim=.004, velocity_unit=:km_s).psd == r.psd
    @test forward_psd([beam,beam]; cfg..., rate_weights_s=[12.,12.]).psd ≈ r.psd
    @test forward_psd([beam]; cfg..., rate_weights_s=[0.]).density_total_m3 == 0
    @test forward_psd([]; cfg..., rate_weights_s=[]).density_total_m3 == 0

    # Different axis ranges/bin widths detect transposition and marginalization errors.
    a = (;t=[0.,1.],u=[SA[-.1,0.,0.,-3.,-2.,-1.], SA[.1,0.,0.,3.,4.,7.]])
    kw = (;cfg...,vlim=((-4.,4.),(-3.,6.),(-2.,10.)),vgrid=(4,3,6))
    xyz = forward_psd([a]; kw...)
    for (option, omitted, kept) in (("Vx-Vy",3,(1,2)),("Vy-Vz",1,(2,3)),("Vx-Vz",2,(1,3)))
        p = forward_psd([a]; kw..., option)
        dv = diff(xyz.all_velocity_edges_m_s[omitted])[1]
        @test p.psd ≈ dropdims(sum(xyz.psd;dims=omitted);dims=omitted)*dv
        @test p.axes == map(k -> (:vx,:vy,:vz)[k],kept)
        @test sum(p.psd)*prod(diff(e)[1] for e in p.velocity_edges_m_s) ≈ 3.
        @test p.units == "s^2 m^-5"
    end
    # Velocity edges subdivide the segment even though both positions are inside.
    ramp = (;t=[0.,2.],u=[SA[0.,0.,0.,-2.,0.,0.],SA[0.,0.,0.,2.,0.,0.]])
    p = forward_psd(ramp;cfg..., vlim=1.,vgrid=2,rate_weights_s=[8.])
    @test p.residence_s ≈ [2.]
    @test p.outside_vlim_residence_s ≈ [1.]
    @test p.density_in_range_m3 ≈ 1.
    @test p.density_outside_vlim_m3 ≈ 1.
    @test p.psd[:,2,2] ≈ [.5,.5]
    # Omitted-axis range is respected in 2D too.
    @test sum(forward_psd(ramp;cfg...,vlim=1.,vgrid=2,option=:yz).psd) ≈ 3.
    @test forward_psd(beam;cfg...,vlim=1.).density_outside_vlim_m3 ≈ 3.
    @test forward_psd(beam;cfg...,vlim=2.).density_in_range_m3 ≈ 3. # final edge included
    face = (;t=beam.t,u=[SA[-2.,1.,0.,2.,0.,0.],SA[2.,1.,0.,2.,0.,0.]])
    @test forward_psd(face;cfg...).density_total_m3 == 0 # upper cube face excluded
    tangent = (;t=beam.t,u=[SA[-2.,0.,0.,2.,2.,0.],SA[0.,2.,0.,2.,2.,0.]])
    @test forward_psd(tangent;cfg...).density_total_m3 == 0 # point contact
    back = (;t=[0.,1.,2.],u=[beam.u[1],beam.u[2],beam.u[1]])
    @test forward_psd(back;cfg...).residence_s ≈ [1.] # all re-entries
    @test forward_psd((;t=[0.],u=[beam.u[1]]);cfg...).density_total_m3 == 0
    @test_throws ArgumentError forward_psd((;t=[1.,0.],u=beam.u);cfg...)
    @test_throws ArgumentError forward_psd((;t=[0.,0.],u=beam.u);cfg...)
    @test_throws ArgumentError forward_psd((;t=[0.,NaN],u=beam.u);cfg...)
    @test_throws ArgumentError forward_psd(beam;cfg...,rate_weights_s=[-1.])
    @test_throws ArgumentError forward_psd(beam;cfg...,rate_weights_s=[])
    @test_throws ArgumentError forward_psd(beam;cfg...,vlim=(-1.,-2.))
    @test_throws ArgumentError forward_psd(beam;cfg...,vgrid=0)
    @test_throws ArgumentError forward_psd(beam;cfg...,option="slice")
    @test_throws ArgumentError forward_psd(beam;cfg...,species="invalid")
    @test_throws ArgumentError forward_psd(beam;cfg...,side_m=0.)
    @test_throws ArgumentError forward_psd(merge(beam,(;retcode=MarsTP.TP.ReturnCode.MaxIters));cfg...)
end

@testset "Boris saved velocity and PSD convergence" begin
    TP = MarsTP.TP
    BP = MarsTP.BorisPushers
    sp = TP.SpeciesDict["O2+"]
    # Constant acceleration parallel to B: v_x=t and x=t^2/2 analytically.
    param = TP.prepare(; E = _->SA[sp.m/sp.q,0.,0.], B = _->SA[1e-7,0.,0.], species=sp)
    prob = TP.TraceProblem(SA[0.,0.,0.,0.,0.,0.], (0.,2.), param)
    kw = (;detector_m=[0.,0.,0.],side_m=2.,vlim=3.,vgrid=6,rate_weights_s=[8.])
    errors = Float64[]
    for dt in (.2,.1,.05)
        sol = TP.solve(prob,BP.Boris();dt)
        traj = sol
        @test all(isapprox(u[4],t;atol=1e-12) for (u,t) in zip(traj.u,traj.t))
        p = forward_psd([sol];kw...)
        @test p.psd == forward_psd(sol;kw...).psd
        push!(errors,abs(p.density_total_m3-sqrt(2)))
        @test_throws ArgumentError forward_psd(sol;kw...,species="H+")
    end
    @test errors[3] < errors[2] < errors[1]
    sol = TP.solve(prob,BP.AdaptiveBoris(;safety=.001))
    @test all(isapprox(u[4],t;atol=1e-10) for (u,t) in zip(sol.u,sol.t))
    @test forward_psd(sol;kw...).density_total_m3 ≈ sqrt(2) rtol=1e-3
end
