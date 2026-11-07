module ShellMonteCarlo

using MarsTP, StaticArrays, LinearAlgebra, Random, JLD2, ReadVTK, Dates, SHA, TOML, Printf
import KernelAbstractions as KA
import CUDA
const TP = MarsTP.TP
const BP = MarsTP.BorisPushers
const Vec = SVector{3,Float64}
const Row = SVector{7,Float64}

Base.@kwdef struct Config
    dt::Float64 = 0.1
    tmax::Float64 = 500.0
    altitude_km::Float64 = 500.0 # source altitude
    absorption_altitude_km::Float64 = 200.0
    per_cell::Int = 100
    seed::Int = 20260906
    detector::Vec = Vec(1,0,2)*Rm
    side::Float64 = 0.2Rm
    # native cell edges, no duplicate longitude seam or duplicate pole areas
    cell_stride::Int = 1
    batch_size::Int = 256
    flux_model::String = "n_bulk_speed_maxwellian"
    sampling_temperature_factor::Float64 = 4.0
    compress_trajectories::Bool = false
    work_mode::Symbol = :steps
    save_power::Bool = false
    tracing_backend::Symbol = :cpu # :cpu, :cuda, or :kernel_cpu (validation)
end

function _zero_result(p,work_itp)
    tr=(;t=[0.],u=[SVector{6,Float64}(p.x...,p.v...)])
    w=trajectory_work(tr,work_itp;species="O2+",mode=:summary).summary
    return (;history=[Row(0.,p.x...,p.v...)],residence=Vector{Float64}[],events=Vector{Float64}[],
        status="zero_rate",time=0.,x=p.x,v=p.v,work=0.,dK=0.,residual=0.,maxgyro=0.,work_summary=w)
end

# Use every accepted GPU step for the existing piecewise-linear detector model.
# Field interpolation and work accumulation stay on the device; detector
# crossings, residence, and serialization are computed on the host.
function _detector_records(tr,c)
    residence=Vector{Float64}[];events=Vector{Float64}[]
    lo,hi=c.detector.-c.side/2,c.detector.+c.side/2
    for j in 2:length(tr.t)
        a,b=tr.u[j-1],tr.u[j];ta,tb=tr.t[j-1],tr.t[j]
        xa,xb=Vec(a[1:3]),Vec(b[1:3]);va,vb=Vec(a[4:6]),Vec(b[4:6])
        hit=cube_segment(xa,xb,lo,hi)
        hit===nothing && continue
        s0,s1,entry,exit=hit
        x0,x1=xa+s0*(xb-xa),xa+s1*(xb-xa)
        v0,v1=va+s0*(vb-va),va+s1*(vb-va)
        t0,t1=ta+s0*(tb-ta),ta+s1*(tb-ta)
        push!(residence,[t0,t1,x0...,x1...,v0...,v1...])
        entry!=0 && s0>=0 && push!(events,[t0,entry,1,x0...,v0...])
        exit!=0 && s1>0 && push!(events,[t1,exit,-1,x1...,v1...])
    end
    return residence,events
end

function trace_batch(particles,fields,param,c,work_itp)
    results=Vector{Any}(undef,length(particles))
    if c.tracing_backend==:cpu
        Threads.@threads for q in eachindex(particles)
            p=particles[q]
            if p.W>0
                r=trace_particle(p.x,p.v,param,c)
                tr=(;t=[a[1] for a in r.history],u=[a[2:7] for a in r.history])
                w=trajectory_work(tr,work_itp;species="O2+",mode=:summary).summary
                results[q]=merge(r,(;work_summary=w))
            else
                results[q]=_zero_result(p,work_itp)
            end
        end
    else
        backend=c.tracing_backend==:cuda ? CUDA.CUDABackend() :
            c.tracing_backend==:kernel_cpu ? KA.CPU() : error("Unknown tracing backend")
        active=findall(p->p.W>0,particles)
        cfg=ForwardTraceConfig(species="O2+",solver=:boris,dt=c.dt,tspan=(0.,c.tmax))
        sols=trace_forward_bounded([SVector{6,Float64}(particles[q].x...,particles[q].v...) for q in active];
            config=cfg,fields,backend,inner_radius_m=Rm+c.absorption_altitude_km*1e3,
            outer_radius_m=Router,save_every=1,work_itp)
        for (q,tr) in zip(active,sols)
            residence,events=_detector_records(tr,c)
            w=tr.work;laststate=last(tr.u)
            results[q]=(;history=[Row(t,u...) for (t,u) in zip(tr.t,tr.u)],residence,events,
                status=String(tr.status),time=last(tr.t),x=Vec(laststate[1:3]),v=Vec(laststate[4:6]),
                work=w.total_eV,dK=w.delta_kinetic_eV,residual=w.energy_residual_eV,
                maxgyro=tr.maxgyro,work_summary=w)
        end
        for q in eachindex(particles)
            particles[q].W>0 || (results[q]=_zero_result(particles[q],work_itp))
        end
    end
    return results
end

# log P(Z>a), stable even for strongly inward drift. erfc is in Julia's libm.
function log_normal_tail(a)
    if a<10
        return log(0.5*ccall((:erfc,Base.Math.libm),Cdouble,(Cdouble,),a/sqrt(2)))
    end
    z=inv(a*a)
    return -a*a/2-log(a)-log(2pi)/2+log(1-z+3z^2-15z^3+105z^4-945z^5)
end

function log_importance(v,U,sigma,factor,er)
    factor>0 || error("sampling temperature factor must be positive")
    ss=sigma*sqrt(factor)
    d2=sum(abs2,v-U)
    # g+/gs+ = (g/gs) P_s(vr>0)/P(vr>0), with different local radial normals.
    return 1.5log(factor)+0.5d2*(inv(ss^2)-inv(sigma^2))+
        log_normal_tail(-dot(U,er)/ss)-log_normal_tail(-dot(U,er)/sigma)
end

# Stable positive-tail standard normal, Robert exponential rejection for a>0.
function positive_normal(rng, a)
    if a <= 0
        while true
            z=randn(rng)
            z>a && return z
        end
    end
    alpha=(a+sqrt(a*a+4))/2
    while true
        z=a+randexp(rng)/alpha
        rand(rng) <= exp(-0.5*(z-alpha)^2) && return z
    end
end

function outward_velocity(rng, U, sigma, er)
    t1=normalize(cross(abs(er[3])<0.9 ? Vec(0,0,1) : Vec(1,0,0),er))
    t2=cross(er,t1)
    ur=dot(U,er)
    vr=ur+sigma*positive_normal(rng,-ur/sigma)
    vr>0 || error("Unresolved outward velocity")
    return vr*er+(dot(U,t1)+sigma*randn(rng))*t1+(dot(U,t2)+sigma*randn(rng))*t2
end

# Slab intersection. Face code: -1/+1=x-/x+, -2/+2=y-/y+, -3/+3=z-/z+.
cube_segment(a,b,lo,hi)=MarsTP._forward_cube_interval(a,b-a,lo,hi)

function sphere_stop(a,b,inner,outer)
    d=b-a
    # A grazing launch can turn inward within the first Boris drift. Treat it
    # as immediate absorption, including sub-micrometre initial radius roundoff.
    abs(norm(a)-inner)<1e-6 && dot(a,d)<0 && return 0.0,2
    norm(a)>=inner-1e-6 || error("Previous state below absorbing boundary")
    A=dot(d,d)
    A==0 && return 1.0,1
    best,flag=1.0,1
    for (radius,status) in ((inner,2),(outer,4))
        B=2dot(a,d);C=(norm(a)-radius)*(norm(a)+radius)
        disc=B^2-4A*C
        disc<0 && continue
        q=-.5*(B+copysign(sqrt(disc),B))
        roots=q==0 ? (0.0,0.0) : (q/A,C/q)
        for f in roots
            0<=f<=best || continue
            slope=dot(a+f*d,d)
            ((status==2 && slope<0)||(status==4 && slope>0)) || continue
            best,flag=f,status
        end
    end
    return best,flag
end

function trace_particle(x0,v0,param,c::Config; keep_history=true)
    qm=param.q2m; charge=TP.SpeciesDict["O2+"].q; mass=TP.SpeciesDict["O2+"].m
    inner=Rm+c.absorption_altitude_km*1e3
    lo,hi=c.detector.-c.side/2,c.detector.+c.side/2
    x,v,t=x0,v0,0.0
    E,B=Vec(param.E(x,t)),Vec(param.B(x,t))
    half=BP.update_velocity_half(v,E,B,-qm*c.dt/4)
    history=Row[Row(t,x...,v...)]
    # residence: t0,t1, x0[3],x1[3],v0[3],v1[3]; events: t,face,direction,x[3],v[3]
    residence=Vector{Float64}[];events=Vector{Float64}[]
    work=0.0; status="time_limit";maxgyro=0.0
    nsteps=round(Int,c.tmax/c.dt)
    for step in 1:nsteps
        maxgyro=max(maxgyro,abs(qm)*norm(B)*c.dt)
        half=BP.update_velocity_half(half,E,B,qm*c.dt/2)
        trial=x+half*c.dt
        all(isfinite,trial) && all(isfinite,half) || error("Nonfinite particle state")
        f,flag=sphere_stop(x,trial,inner,Router)
        if f==0 && flag==2
            status="inner"
            break
        end
        xn=x+f*(trial-x);tn=(step-1+f)*c.dt
        # One micrometre inward ONLY for field evaluation of an outer event;
        # saved trajectory endpoint remains on the exact geometric sphere.
        query=flag==4 ? xn*((Router-1e-6)/norm(xn)) : xn
        En,Bn=Vec(param.E(query,tn)),Vec(param.B(query,tn))
        vn=BP.update_velocity_half(half,En,Bn,qm*(f-0.5)*c.dt/2)
        all(isfinite,vn) && all(isfinite,En) && all(isfinite,Bn) ||
            error("Nonfinite fields/velocity at t=$tn, r=$(norm(xn)), flag=$flag, query_radius=$(norm(query)), E=$En, B=$Bn")
        hit=cube_segment(x,xn,lo,hi)
        if hit!==nothing
            s0,s1,entry,exit=hit
            # Synchronize velocity at clipped detector endpoints; no staggered velocities saved.
            function at(s)
                xx=x+s*(xn-x);tt=t+s*(tn-t)
                vv=v+s*(vn-v) # Same saved-endpoint interpolation as forward_psd
                return xx,vv,tt
            end
            xa,va,ta=at(s0);xb,vb,tb=at(s1)
            push!(residence,[ta,tb,xa...,xb...,va...,vb...])
            # Entry at a shared endpoint belongs to the following interior segment;
            # exit belongs to the preceding interior segment. Zero-length hits excluded.
            entry!=0 && s0>=0 && push!(events,[ta,entry,1,xa...,va...])
            exit!=0 && s1>0 && push!(events,[tb,exit,-1,xb...,vb...])
        end
        work += charge*dot(Vec(param.E((x+xn)/2,(t+tn)/2)),(v+vn)/2)*(tn-t)/TP.eV
        x,v,t,E,B=xn,vn,tn,En,Bn
        keep_history && push!(history,Row(t,x...,v...))
        if flag!=1
            status=flag==2 ? "inner" : "outer"
            break
        end
    end
    !keep_history && push!(history,Row(t,x...,v...))
    dK=mass*(dot(v,v)-dot(v0,v0))/(2TP.eV)
    return (;history,residence,events,status,time=t,x,v,work,dK,residual=dK-work,maxgyro)
end

function validate_geometry(path,fields,c)
    vtk=VTKFile(path)
    dims=ReadVTK.get_wholeextent(vtk.xml_file)[1]
    pts=reshape(ReadVTK.get_points(vtk),3,dims...)
    errors=zeros(3)
    scales=(1.0,1e3,Rm)
    # Validate the actual stored Cartesian mesh against the package's analytic axes.
    for i in unique([1,div(dims[1],2),dims[1]]),j in 1:dims[2],k in 1:dims[3]
        r,th,ph=fields.r[i],fields.theta[j],fields.phi[k]
        expected=r*Vec(sin(th)*cos(ph),sin(th)*sin(ph),cos(th))
        for q in 1:3
            errors[q]=max(errors[q],norm(scales[q]*Vec(pts[:,i,j,k])-expected))
        end
    end
    q=argmin(errors);maxerr=errors[q]
    maxerr<1e3 || error("Stored VTK mesh disagrees with expected spherical mesh under m/km/Rm scales: $errors m")
    # Read radial axes from stored points rather than recreating the lower radius.
    # The file lower radius is 1.058892815 Rm, slightly different from Rinner/Rm.
    r=[norm(Vec(pts[:,i,1,1]))*scales[q] for i in 1:dims[1]]
    all(diff(r).>0) || error("Nonmonotonic radial mesh")
    r[1] <= Rm+c.absorption_altitude_km*1e3 < Rm+c.altitude_km*1e3 <= r[end] || error("Mesh must cover absorption and source radii")
    mesherr=0.0
    for i in 1:dims[1],j in 1:dims[2],k in 1:dims[3]
        th,ph=fields.theta[j],fields.phi[k]
        expected=r[i]*Vec(sin(th)*cos(ph),sin(th)*sin(ph),cos(th))
        mesherr=max(mesherr,norm(scales[q]*Vec(pts[:,i,j,k])-expected))
    end
    mesherr<5.0 || error("Mesh is not separable in the expected spherical angles")
    actual=MHDFields(r,fields.theta,fields.phi,fields.E,fields.B,fields.electric_field,fields.path)
    pd=get_point_data(vtk)
    scalar(A)=TP.build_interpolator(TP.StructuredGrid,A,r,fields.theta,fields.phi)
    n=MarsTP._read_array(pd,"n_O^2^p [m^-3]",dims...)
    T=MarsTP._read_array(pd,"T_O^2^p [K]",dims...)
    U=MarsTP._read_array(pd,"U_O^2^p [m/s]",dims...)
    source=IonosphereSource(scalar(n),scalar(T),ntuple(k->scalar(Array(U[k,:,:,:])),3),Rm+c.altitude_km*1e3)
    meta=Dict("dimensions"=>collect(dims),"max_original_analytic_mesh_error_m"=>maxerr,
        "max_actual_mesh_error_m"=>mesherr,"r_min_Rm"=>r[1]/Rm,
        "radial_axis"=>"norm of stored VTK points times coordinate scale, not reconstructed logspace",
        "VTK_points_to_m"=>scales[q],"point_fields"=>collect(keys(pd)))
    return actual,source,meta
end

function release_particles(fields,source,c)
    c.flux_model in ("n_bulk_speed_maxwellian","n_speed_outward_maxwellian","reservoir_maxwellian_rate") || error("Unsupported flux model")
    radius=source.radius
    particles=NamedTuple[]
    cells=NamedTuple[]
    mass=TP.SpeciesDict["O2+"].m
    cellid=0
    for j in 1:length(fields.theta)-1,k in 1:length(fields.phi)-1
        cellid+=1
        (cellid-1)%c.cell_stride==0 || continue
        th0,th1=fields.theta[j:j+1];ph0,ph1=fields.phi[k:k+1]
        area=radius^2*(cos(th0)-cos(th1))*(ph1-ph0)
        # One midpoint flux per native angular cell; uniform area samples within it.
        mu=(cos(th0)+cos(th1))/2;ph=(ph0+ph1)/2
        er=Vec(sqrt(1-mu^2)*cos(ph),sqrt(1-mu^2)*sin(ph),mu)
        m=ionosphere_properties(source,er)
        push!(cells,(;cellid,j,k,area,flux=m.flux,n=m.n,T=m.Ti,U=m.Ui,
            lon=rad2deg(ph),lat=asind(mu)))
        if c.flux_model in ("n_bulk_speed_maxwellian","reservoir_maxwellian_rate")
            # Follow maxwellian_source.jl: one midpoint patch, N UNTRUNCATED draws.
            # Bulk-speed rates retain both velocity signs; reservoir is an explicit legacy mode.
            rng=Xoshiro(c.seed+cellid)
            sampled=sample_maxwellian_source(c.per_cell;position_m=radius*er,
                bulk_velocity_m_s=m.Ui,temperature_ev=TP.kB*m.Ti/1.602176634e-19,
                weights=MonteCarloWeight(source_number_density_m3=m.n,
                    sampling_temperature_factor=c.sampling_temperature_factor),
                normal=er,area_m2=area,rng,
                flux_model=c.flux_model=="n_bulk_speed_maxwellian" ? :bulk_speed : :reservoir)
            for q in 1:c.per_cell
                state=sampled.initial_states[q]
                x=Vec(state[1:3]);v=Vec(state[4:6])
                id=(cellid-1)*c.per_cell+q
                W=m.n>0 ? sampled.rate_weights_s[q] : 0.0
                density_weight=m.n>0 ? sampled.density_weights_m3[q] : 0.0
                logw=log(sampled.importance_weights[q])
                push!(particles,(;id,cellid,x,v,logw,W,flux=m.flux,area,density_weight))
            end
            continue
        end
        local_samples=NamedTuple[]
        sigma=sqrt(TP.kB*m.Ti/mass)
        for q in 1:c.per_cell
            id=(cellid-1)*c.per_cell+q
            rng=Xoshiro(c.seed+id) # independent of thread scheduling / batches
            mu=cos(th1)+rand(rng)*(cos(th0)-cos(th1))
            ph=ph0+rand(rng)*(ph1-ph0)
            er=Vec(sqrt(1-mu^2)*cos(ph),sqrt(1-mu^2)*sin(ph),mu)
            x=radius*er
            v=outward_velocity(rng,m.Ui,sigma*sqrt(c.sampling_temperature_factor),er)
            logw=log_importance(v,m.Ui,sigma,c.sampling_temperature_factor,er)
            push!(local_samples,(;id,cellid,x,v,logw))
        end
        maxlog=maximum(p.logw for p in local_samples)
        denominator=sum(exp(p.logw-maxlog) for p in local_samples)
        for p in local_samples
            W=m.flux*area*exp(p.logw-maxlog)/denominator
            push!(particles,(;p...,W,flux=m.flux,area,density_weight=m.n*exp(p.logw-maxlog)/denominator))
        end
    end
    return particles,cells
end

"""Run the 500 km shell experiment in bounded batches and save synchronized SI states."""
function run_monte_carlo(out,c=Config())
    c.dt>0 && c.tmax>0 && c.per_cell>0 && c.cell_stride>0 || error("Invalid configuration")
    0 <= c.absorption_altitude_km < c.altitude_km || error("Absorption altitude must be below source altitude")
    c.batch_size>0 || error("batch_size must be positive")
    c.tracing_backend in (:cpu,:cuda,:kernel_cpu) || error("Unknown tracing backend")
    c.tracing_backend!=:cuda || CUDA.functional() || error("CUDA backend requested but unavailable")
    isapprox(c.tmax/c.dt,round(c.tmax/c.dt);atol=1e-8) || error("tmax must be a multiple of dt")
    ispath(out) && error("Output exists; choose a new run directory: $out")
    mkpath(out)
    fields=load_mhd_fields()
    fields,source,geometry=validate_geometry(fields.path,fields,c)
    param=MarsTP.mhd_param(fields;species="O2+")
    c.work_mode in (:steps,:summary) || error("work_mode must be :steps or :summary")
    # Components must use the actual VTK radial axis, just like the integrator.
    component(which)=begin
        f=load_mhd_fields(fields.path;electric_field=which)
        TP.build_interpolator(TP.StructuredGrid,f.E,fields.r,fields.theta,fields.phi)
    end
    work_itp=MarsTP.FieldWorkInterpolators(component(:total),component(:conv),component(:hall))
    particles,cells=release_particles(fields,source,c)
    if c.flux_model!="n_bulk_speed_maxwellian"
        all(p->p.W==0 || dot(p.v,p.x)>0,particles) || error("Positive-rate non-outward release")
    end
    totalrate=sum(p.W for p in particles)
    area=sum(a.area for a in cells)
    c.cell_stride==1 && !isapprox(area,4pi*source.radius^2;rtol=1e-12) && error("Shell area mismatch")
    meta=Dict("model"=>"steady_shell_mc_v1","species"=>"O2+","Rm_m"=>Rm,
        "particle_mass_kg"=>TP.SpeciesDict["O2+"].m,"particle_charge_C"=>TP.SpeciesDict["O2+"].q,
        "inner_altitude_km"=>c.absorption_altitude_km,"source_altitude_km"=>c.altitude_km,
        "outer_radius_Rm"=>Router/Rm,"detector_Rm"=>collect(c.detector/Rm),
        "cube_side_Rm"=>c.side/Rm,"cube_volume_m3"=>c.side^3,"cube_face_area_m2"=>c.side^2,
        "dt_s"=>c.dt,"max_flight_time_s"=>c.tmax,"per_cell"=>c.per_cell,"seed"=>c.seed,
        "cell_stride"=>c.cell_stride,"n_particles"=>length(particles),"n_cells"=>length(cells),
        "flux_model"=>c.flux_model,"source_total_rate_s1"=>totalrate,"sampled_area_m2"=>area,
        "volume_production"=>false,"electric_field"=>"static total E from VTS",
        "coordinate_system"=>"native MHD Cartesian axes; MSO/MSE provenance not established",
        "velocity_sampling"=>"broadened conditional outward drifting Maxwellian; per-cell self-normalized g+/gs+ importance weights",
        "sampling_temperature_factor"=>c.sampling_temperature_factor,
        "weight_formula"=>"W_i = F_cell A_cell exp(log_g_over_gs_i) / sum_cell(exp(log_g_over_gs))",
        "weight_reference"=>"MarsASPEN.jl/src/monte_carlo_weight.jl local checkout; density normalization adapted to source rate",
        "sampling_caveat"=>"10 particles/cell with self-normalized importance weights has finite-sample bias; inspect effective sample sizes",
        "weight_unit"=>"s^-1","position_unit"=>"m","velocity_unit"=>"m s^-1",
        "phase_space_density_unit"=>"s^3 m^-6","projection_unit"=>"s^2 m^-5",
        "source_file"=>fields.path,"source_size_bytes"=>filesize(fields.path),
        "source_sha256"=>open(sha256,fields.path)|>bytes2hex,
        "julia_version"=>string(VERSION),"TestParticle_version"=>string(pkgversion(TP)),
        "git_commit"=>get(ENV,"MC_GIT_COMMIT","unavailable; set MC_GIT_COMMIT at launch"),
        "git_status"=>get(ENV,"MC_GIT_STATUS","unavailable; set MC_GIT_STATUS at launch"),
        "geometry"=>geometry,"created_utc"=>string(now(UTC)),
        "steady_state_note"=>"weights are injection rates; residence gives contribution up to max flight age, no extra division by tmax",
        "trajectory_columns"=>["time_s","x_m","y_m","z_m","vx_ms","vy_ms","vz_ms"])
    meta["work_mode"]=String(c.work_mode)
    meta["tracing_backend"]=String(c.tracing_backend)
    meta["work_accumulation"]=c.tracing_backend==:cpu ? "host" : "device, every accepted Boris segment"
    meta["detector_processing"]="host, full integration cadence"
    meta["work_method"]="midpoint q E dot v dt; full integration cadence"
    meta["save_power"]=c.save_power
    meta["batch_size"]=c.batch_size
    meta["detector_interpolation"]="piecewise linear saved endpoints, shared with forward_psd"
    meta["compress_trajectories"]=c.compress_trajectories
    meta["n_positive_rate_particles"]=count(p->p.W>0,particles)
    rate_mode=c.flux_model in ("n_bulk_speed_maxwellian","reservoir_maxwellian_rate")
    if rate_mode
        meta["model"]="steady_reservoir_mc_v2"
        meta["velocity_sampling"]="sample_maxwellian_source: untruncated Cartesian Maxwellian at Ts=4Ti; local midpoint patch"
        meta["position_sampling"]="area-centroid angular midpoint per native cell, as local patch approximation"
        meta["random_stream"]="Xoshiro(seed+cell_id), all N draws in original order"
        meta["weight_formula"]="Q_i = n A max(dot(v_i,er),0) (g_i/gs_i) / N_all_draws"
        meta["density_weight_formula"]="Wn_i = n w_i / sum_cell(w); diagnostic source density shares, not used by detector estimator"
        meta["weight_reference"]="examples/forward_tracing/monte_carlo_forward_tracing/README.md and src/tracing/monte_carlo_weight.jl"
        meta["sampling_caveat"]="No rate self-normalization; inward samples have Q=0 and are retained without propagation"
        meta["source_flux_column_note"]="source_flux_m2_s is n*norm(U) diagnostic only; rate weights use individual outward radial speed"
    end
    if c.flux_model=="n_bulk_speed_maxwellian"
        meta["model"]="steady_bulk_speed_mc_v3"
        meta["velocity_sampling"]="untruncated Cartesian Maxwellian at Ts=sampling_temperature_factor*Ti; no radial-sign selection"
        meta["weight_formula"]="Q_i = n norm(U_bulk) A w_i / sum_cell(w), w=g/gs"
        meta["sampling_caveat"]="Self-normalized importance estimates have finite-sample bias; patch rates sum exactly to n norm(U_bulk) A"
        meta["source_flux_column_note"]="source_flux_m2_s = n norm(U_bulk), prescribed injection flux, not net radial surface flux"
        meta["inner_boundary_note"]="inward launch velocities retain source rate and propagate below the source until the 200 km absorbing boundary, outer boundary, or time limit"
    end
    open(joinpath(out,"metadata.toml"),"w") do io; TOML.print(io,meta);end
    cp(MarsTP.project_path("Project.toml"),joinpath(out,"Project.snapshot.toml"))
    cp(MarsTP.project_path("Manifest.toml"),joinpath(out,"Manifest.snapshot.toml"))
    cp(@__FILE__,joinpath(out,"monte_carlo_shell.snapshot.jl"))
    cp(MarsTP.project_path("src","analysis","electric_field_work.jl"),joinpath(out,"electric_field_work.snapshot.jl"))
    cp(MarsTP.project_path("src","tracing","trajectory_io.jl"),joinpath(out,"trajectory_io.snapshot.jl"))
    cp(MarsTP.project_path("src","tracing","forward_bounded.jl"),joinpath(out,"forward_bounded.snapshot.jl"))
    if rate_mode
        cp(MarsTP.project_path("src","tracing","monte_carlo_weight.jl"),joinpath(out,"monte_carlo_weight.snapshot.jl"))
        cp(MarsTP.project_path("examples","forward_tracing","monte_carlo_forward_tracing","README.md"),joinpath(out,"detector_3d_psd.snapshot.md"))
    end
    open(joinpath(out,"source_cells.csv"),"w") do io
        println(io,"cell_id,theta_index,phi_index,longitude_deg,latitude_deg,area_m2,flux_m2_s,n_m3,Ti_K,ux_ms,uy_ms,uz_ms")
        for a in cells
            println(io,join((a.cellid,a.j,a.k,a.lon,a.lat,a.area,a.flux,a.n,a.T,a.U...),','))
        end
    end
    println("START particles=$(length(particles)), cells=$(length(cells)), rate=$totalrate s^-1, threads=$(Threads.nthreads())")
    flush(stdout)
    summary=open(joinpath(out,"particles.csv"),"w")
    resio=open(joinpath(out,"probe_residence.csv"),"w")
    eventio=open(joinpath(out,"probe_crossings.csv"),"w")
    weight_name=rate_mode ? "rate_weight_s1" : "weight_s1"
    println(summary,"particle_id,cell_id,$weight_name,source_flux_m2_s,source_area_m2,x0_m,y0_m,z0_m,vx0_ms,vy0_ms,vz0_ms,status,end_time_s,xend_m,yend_m,zend_m,vxend_ms,vyend_ms,vzend_ms,work_eV,deltaK_eV,residual_eV,max_gyro_angle_rad,probe_residence_s,log_importance,source_density_weight_m3,work_conv_eV,work_hall_eV,field_sum_residual_eV")
    println(resio,"particle_id,$weight_name,t0_s,t1_s,x0_m,y0_m,z0_m,x1_m,y1_m,z1_m,vx0_ms,vy0_ms,vz0_ms,vx1_ms,vy1_ms,vz1_ms")
    println(eventio,"particle_id,$weight_name,face_flux_m2_s,time_s,face,direction,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms")
    start=time();counts=Dict("inner"=>0,"outer"=>0,"time_limit"=>0,"zero_rate"=>0)
    try
        for bstart in 1:c.batch_size:length(particles)
            indices=bstart:min(bstart+c.batch_size-1,length(particles))
            results=trace_batch(particles[indices],fields,param,c,work_itp)
            batchid=div(bstart-1,c.batch_size)+1
            trajectories=[(;t=[a[1] for a in r.history],u=[a[2:7] for a in r.history]) for r in results]
            write_trajectory_batch(joinpath(out,@sprintf("trajectories_%05d.jld2",batchid)),trajectories;
                particle_ids=[particles[i].id for i in indices],rate_weights_s=[particles[i].W for i in indices],
                source_density_weights_m3=[particles[i].density_weight for i in indices],
                cell_ids=[particles[i].cellid for i in indices],termination_codes=[r.status for r in results],
                species="O2+",coordinate_system=meta["coordinate_system"],compress=c.compress_trajectories,
                work_itp,work_mode=c.work_mode,save_power=c.save_power,
                work_summaries=c.work_mode==:summary && !c.save_power ? [r.work_summary for r in results] : nothing)
            for (q,i) in enumerate(indices)
                p=particles[i];r=results[q];counts[r.status]+=1
                dwell=sum((a[2]-a[1] for a in r.residence);init=0.0)
                println(summary,join((p.id,p.cellid,p.W,p.flux,p.area,p.x...,p.v...,r.status,r.time,r.x...,r.v...,r.work,r.dK,r.residual,r.maxgyro,dwell,p.logw,p.density_weight,r.work_summary.conv_eV,r.work_summary.hall_eV,r.work_summary.field_sum_residual_eV),','))
                for a in r.residence;println(resio,join((p.id,p.W,a...),','));end
                for a in r.events;println(eventio,join((p.id,p.W,p.W/c.side^2,a...),','));end
            end
            flush(summary);flush(resio);flush(eventio)
            println("PROGRESS $(last(indices))/$(length(particles)) elapsed=$(round(time()-start;digits=1)) s $counts")
            flush(stdout)
        end
    finally
        close(summary);close(resio);close(eventio)
    end
    open(joinpath(out,"completion.toml"),"w") do io
        TOML.print(io,Dict("complete"=>true,"elapsed_s"=>time()-start,"status_counts"=>counts))
    end
    println("COMPLETE $out")
    return out
end

main(out,c=Config())=run_monte_carlo(out,c)

end # module

if abspath(PROGRAM_FILE)==@__FILE__
    using .ShellMonteCarlo
    out=length(ARGS)>0 ? ARGS[1] : error("Usage: julia --project=. monte_carlo_shell.jl OUTPUT [cell_stride] [tmax] [dt] [per_cell] [flux_model] [compress] [backend] [work_mode] [batch_size]")
    stride=length(ARGS)>1 ? parse(Int,ARGS[2]) : 1
    tmax=length(ARGS)>2 ? parse(Float64,ARGS[3]) : 500.0
    dt=length(ARGS)>3 ? parse(Float64,ARGS[4]) : 0.1
    per_cell=length(ARGS)>4 ? parse(Int,ARGS[5]) : 100
    flux_model=length(ARGS)>5 ? ARGS[6] : "n_bulk_speed_maxwellian"
    compress_trajectories=length(ARGS)>6 ? parse(Bool,ARGS[7]) : false
    tracing_backend=length(ARGS)>7 ? Symbol(ARGS[8]) : :cpu
    work_mode=length(ARGS)>8 ? Symbol(ARGS[9]) : :steps
    batch_size=length(ARGS)>9 ? parse(Int,ARGS[10]) : 1024
    ShellMonteCarlo.main(out,ShellMonteCarlo.Config(;cell_stride=stride,tmax,dt,per_cell,flux_model,compress_trajectories,batch_size,tracing_backend,work_mode))
end
