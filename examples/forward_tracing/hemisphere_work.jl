using MarsTP, StaticArrays, LinearAlgebra, Random
const TP = MarsTP.TP
const BP = MarsTP.BorisPushers
function main()
    rng = Xoshiro(20260905)
    positions = map(1:1000) do _
        mu, phi = 2rand(rng)-1, 2pi*rand(rng)
        d = SA[sqrt(1-mu^2)*cos(phi),sqrt(1-mu^2)*sin(phi),mu]
        (Rm+800e3) * normalize(d)
    end
    hemisphere = get(ENV, "WORK_HEMISPHERE", "dayside")
    hemisphere in ("dayside", "all") || error("WORK_HEMISPHERE must be dayside or all")
    ids = hemisphere == "all" ? collect(eachindex(positions)) : findall(p->p[1]>0,positions)
    fields = load_mhd_fields()
    param = MarsTP.mhd_param(fields;species="O2+")
    itp = build_field_work_interpolators()
    lock_output = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    println(stderr,"Analyzing $(length(ids)) $hemisphere particles, 800 km, dt=0.1 s, threads=$(Threads.nthreads())")
    Threads.@threads for id in ids
        status = Ref("time_limit")
        function outside(u,p,t)
            all(isfinite,u[1:3]) || error("Nonfinite position for particle $id")
            r=norm(SA[u[1],u[2],u[3]])
            if r<Rinner || r>Router
                status[]=r<Rinner ? "inner" : "outer"
                return true
            end
            all(isfinite,u) || error("Nonfinite in-domain velocity")
            return false
        end
        prob=TP.TraceProblem(vcat(positions[id],SA[0.,0.,0.]),(0.,20000.),param)
        sol = nothing
        p = nothing
        actual_dt = 0.1
        for dt in (0.1, 0.05, 0.025, 0.0125)
            status[] = "time_limit"
            sol=TP.solve(prob,BP.Boris();dt,isoutside=outside,
                maxiters=ceil(Int,20000/dt)+1)
            p=Electric_field_work_profile(sol,itp)
            actual_dt = dt
            abs(p.summary.energy_residual_eV)/max(abs(p.summary.delta_kinetic_eV),1.) <= 1e-3 && break
        end
        abs(p.summary.energy_residual_eV)/max(abs(p.summary.delta_kinetic_eV),1.) <= 1e-3 ||
            error("Energy closure exceeds 0.1% for particle $id")
        # Integrate all saved steps; decimate only the graphical representation.
        selected=unique(vcat(collect(1:10:length(p.t)),length(p.t)))
        lock(lock_output) do
            print("{\"id\":",id,",\"dt\":",actual_dt,",\"status\":\"",status[],"\",\"summary\":[",
                p.summary.conv_eV,',',p.summary.hall_eV,',',p.summary.delta_kinetic_eV,',',
                p.summary.energy_residual_eV,"],\"points\":[")
            for (j,i) in enumerate(selected)
                j>1 && print(',')
                u=sol.u[i]
                print('[',u[1]/Rm,',',u[3]/Rm,',',p.conv_eV[i],',',p.hall_eV[i],',',p.total_eV[i],']')
            end
            println("]}")
            flush(stdout)
        end
        count=Threads.atomic_add!(done,1)+1
        count%50==0 && println(stderr,"Work profiles completed: $count / $(length(ids))")
    end
end
main()
