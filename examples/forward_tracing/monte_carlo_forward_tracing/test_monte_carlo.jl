using Test
include("monte_carlo_shell.jl")
using .ShellMonteCarlo, MarsTP, StaticArrays, LinearAlgebra, Random
const MC=ShellMonteCarlo

@testset "Production batch GPU adapter, work, and detector records" begin
    r=collect(range(Rinner,Router;length=8));th=collect(range(0.,pi;length=9));ph=collect(range(0.,2pi;length=17))
    E=zeros(3,length(r),length(th),length(ph));E[1,:,:,:].=1e-5
    MarsTP._rotate_vectors_to_spherical!(E,r,th,ph)
    fields=MHDFields(r,th,ph,E,zero(E),:total,"uniform_Ex")
    interp(A)=MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid,A,r,th,ph)
    itp=MarsTP.FieldWorkInterpolators(interp(E),interp(2E),interp(-E))
    param=MarsTP.mhd_param(fields;species="O2+")
    starts=[(SA[Rinner+1000.,0.,0.],SA[-1000.,0.,0.]),
            (SA[Router-150.,0.,0.],SA[1000.,0.,0.]),
            (SA[Rinner+2000.,0.,0.],SA[0.,0.,0.]),
            (SA[Rm+500e3,0.,0.],SA[100.,0.,0.])]
    particles=[(;x,v,W=i==4 ? 0. : 1.) for (i,(x,v)) in enumerate(starts)]
    options=(;dt=.3,tmax=1.2,detector=SA[Rinner+500.,0.,0.],side=200.)
    cpu=MC.trace_batch(particles,fields,param,MC.Config(;options...),itp)
    backends=get(ENV,"MARSTP_TEST_CUDA","false")=="true" ? (:kernel_cpu,:cuda) : (:kernel_cpu,)
    for backend in backends
        gpu=MC.trace_batch(particles,fields,param,MC.Config(;options...,tracing_backend=backend),itp)
        @test getproperty.(gpu,:status)==["inner","outer","time_limit","zero_rate"]
        for (a,b) in zip(cpu,gpu)
            @test a.time ≈ b.time atol=1e-9
            @test a.x ≈ b.x atol=1e-6
            @test a.v ≈ b.v atol=1e-7
            @test a.maxgyro ≈ b.maxgyro
            for key in keys(a.work_summary)
                @test getproperty(a.work_summary,key) ≈ getproperty(b.work_summary,key) atol=1e-8
            end
            @test length(a.residence)==length(b.residence)
            @test length(a.events)==length(b.events)
            for (x,y) in zip(a.residence,b.residence);@test x ≈ y atol=1e-6;end
            for (x,y) in zip(a.events,b.events);@test x ≈ y atol=1e-6;end
        end
        @test length(gpu[1].events)==2
    end
end
@testset "Bulk-speed shell source without sign selection" begin
    radius=Rm+500e3
    fields=MHDFields([radius,Router],[0.,pi/2,pi],[0.,pi,2pi],zeros(3,2,3,3),zeros(3,2,3,3),:total,"")
    U=SA[-1000.,200.,300.]
    source=IonosphereSource(x->1e6,x->1000.,(x->U[1],x->U[2],x->U[3]),radius)
    c=MC.Config(per_cell=1000)
    @test c.flux_model=="n_bulk_speed_maxwellian"
    particles,cells=MC.release_particles(fields,source,c)
    @test sum(p.W for p in particles) ≈ 1e6*norm(U)*4pi*radius^2
    @test all(p->p.W>0,particles)
    @test any(p->dot(p.x,p.v)<0,particles)
    @test any(p->dot(p.x,p.v)>0,particles)
    for cell in cells
        localp=filter(p->p.cellid==cell.cellid,particles)
        @test sum(p.W for p in localp) ≈ cell.flux*cell.area
        @test all(p->isapprox(p.W,cell.area*norm(U)*p.density_weight),localp)
    end
    sp=MarsTP.TP.SpeciesDict["O2+"]
    zero_field=(x,t)->SA[0.,0.,0.]
    p=first(filter(p->dot(p.x,p.v)<0,particles))
    r=MC.trace_particle(p.x,p.v,MarsTP.TP.prepare(;E=zero_field,B=zero_field,species=sp),MC.Config(tmax=1.))
    @test r.status=="time_limit" && r.time≈1.
    @test norm(r.x)<norm(p.x)
end
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


@testset "Monte Carlo work matches saved interval diagnostics" begin
    sp=MarsTP.TP.SpeciesDict["O2+"]
    E=SA[1e-5,0.,0.]
    param=MarsTP.TP.prepare(;E=(x,t)->E,B=(x,t)->zero(E),species=sp)
    itp=MarsTP.FieldWorkInterpolators(_->E,_->2E,_->-E)
    for x in (SA[Rm+500e3,0.,0.],SA[Router-150.,0.,0.])
        r=MC.trace_particle(x,SA[1000.,0.,0.],param,MC.Config(dt=.1,tmax=1.))
        tr=(;t=[a[1] for a in r.history],u=[a[2:7] for a in r.history])
        w=trajectory_work(tr,itp)
        @test w.summary.total_eV ≈ r.work
        @test w.summary.energy_residual_eV ≈ r.residual atol=1e-12
        @test w.summary.conv_eV ≈ 2r.work
        @test w.summary.hall_eV ≈ -r.work
        if x[1]>Router-1000
            @test r.status=="outer"
            @test 0<diff(tr.t)[end]<.1
        else
            @test abs(r.residual)<1e-10
        end
    end
end
