using Test
include("monte_carlo_shell.jl")
using .ShellMonteCarlo, MarsTP, StaticArrays, LinearAlgebra, Random
const MC=ShellMonteCarlo
@testset "Source sampling and detector geometry" begin
    er=SA[1.,0.,0.];U=SA[-1000.,200.,300.];sigma=1000.
    rng=Xoshiro(19)
    for _ in 1:100
        v=MC.outward_velocity(rng,U,sigma,er)
        @test dot(v,er)>0
        @test MC.log_importance(v,U,sigma,1.,er)≈0 atol=1e-14
    end
    @test exp(MC.log_normal_tail(0.))≈0.5
    @test MC.log_normal_tail(10.)≈-53.23128515051247 atol=1e-7
    lo=SA[-1.,-1.,-1.];hi=-lo
    @test MC.cube_segment(SA[-2.,0.,0.],SA[2.,0.,0.],lo,hi)==(.25,.75,-1,1)
    @test MC.cube_segment(SA[-2.,2.,0.],SA[2.,2.,0.],lo,hi)===nothing
    @test MC.cube_segment(SA[0.,0.,0.],SA[2.,0.,0.],lo,hi)==(0.,.5,0,1)
    a=SA[Rm+500e3-1e-9,0.,0.]
    @test MC.sphere_stop(a,a-SA[1.,0.,0.],Rm+500e3,Router)==(0.,2)
    @test MC.sphere_stop(a,a+SA[1.,0.,0.],Rm+500e3,Router)==(1.,1)
end

@testset "Reservoir shell area and rate normalization" begin
    radius=Rm+500e3
    fields=MHDFields([radius,Router],[0.,pi/2,pi],[0.,pi,2pi],zeros(3,2,3,3),zeros(3,2,3,3),:total,"")
    source=IonosphereSource(x->1e6,x->1000.,(x->0.,x->0.,x->0.),radius)
    c=MC.Config(per_cell=5000,flux_model="reservoir_maxwellian_rate")
    particles,cells=MC.release_particles(fields,source,c)
    @test length(particles)==20_000
    @test sum(a.area for a in cells)≈4pi*radius^2
    @test any(p->p.W==0,particles)
    @test all(p->p.W==0 || dot(p.x,p.v)>0,particles)
    sigma=sqrt(MarsTP.TP.kB*1000/MarsTP.TP.SpeciesDict["O2+"].m)
    analytic=1e6*4pi*radius^2*sigma/sqrt(2pi)
    @test abs(sum(p.W for p in particles)/analytic-1)<.05
    for i in 1:4
        @test sum(p.density_weight for p in particles if p.cellid==i)≈1e6
    end
end

@testset "Boris transport, boundaries, residence" begin
    TP=MarsTP.TP
    species=TP.SpeciesDict["O2+"]
    E=(x,t)->SA[0.,0.,0.];B=(x,t)->SA[0.,0.,0.]
    p=TP.prepare(;E,B,species)
    x=SA[Rm+500e3,0.,0.];v=SA[1e4,0.,0.]
    c=MC.Config(tmax=1.,detector=x+SA[5000.,0.,0.],side=2000.)
    r=MC.trace_particle(x,v,p,c)
    @test r.x≈x+v
    @test r.v≈v
    @test sum(a[2]-a[1] for a in r.residence)≈.2 atol=1e-12
    @test length(r.events)==2
    @test r.residual==0
    # A positively charged ion initially along +x rotates toward -y in +Bz.
    B=(x,t)->SA[0.,0.,1e-8]
    p=TP.prepare(;E,B,species)
    r=MC.trace_particle(x,v,p,c)
    @test r.v[2]<0
    @test norm(r.v)≈norm(v) rtol=1e-12
    omega=species.q/species.m*1e-8
    @test norm(r.v-SA[cos(omega),-sin(omega),0.]*norm(v))/norm(v)<1e-5
    # Full orbit returns through the 200 km absorbing sphere; no out-of-domain read.
    r=MC.trace_particle(x,v,p,MC.Config(tmax=300.))
    @test r.status=="inner"
    @test norm(r.x)≈Rm+200e3 atol=1e-7
    zero_field=(x,t)->SA[0.,0.,0.]
    param=TP.prepare(;E=zero_field,B=zero_field,species)
    inward=MC.trace_particle(x,SA[-1000.,0.,0.],param,MC.Config(dt=1.,tmax=301.))
    @test inward.status=="inner"
    @test inward.time≈300. atol=1e-9
    @test norm(inward.x)≈Rm+200e3 atol=1e-7
end
