# Actual total-field trajectory; signed work components evaluated on that trajectory.
# Output goes directly to Python, with no trajectory file.
using MarsTP, StaticArrays, LinearAlgebra
const TP = MarsTP.TP
const BP = MarsTP.BorisPushers
function main()
    fields = load_mhd_fields()
    param = MarsTP.mhd_param(fields; species = "O2+")
    itps = build_field_work_interpolators()
    species = TP.SpeciesDict["O2+"]
    p0 = (Rm + 800e3) * normalize(SA[1.0, 0.0, 1.0])
    println(stderr, "Initial position [RM] = ", p0/Rm, "; O2+, v0=0, altitude=800 km")
    for dt in (0.1, 0.05, 0.025)
        status = Ref("time_limit")
        function outside(u, p, t)
            all(isfinite, u[1:3]) || error("Nonfinite trajectory position")
            r = norm(SA[u[1],u[2],u[3]])
            if r < Rinner || r > Router
                status[] = r < Rinner ? "inner" : "outer"
                return true
            end
            all(isfinite,u) || error("Nonfinite in-domain velocity")
            return false
        end
        prob = TP.TraceProblem(vcat(p0, SA[0.,0.,0.]), (0.,20000.), param)
        sol = TP.solve(prob, BP.Boris(); dt, isoutside=outside,
            maxiters=ceil(Int,20000/dt)+1)
        if length(sol.t) >= 3 && sol.t[end] == sol.t[end-1] && sol.u[end] == sol.u[end-2]
            resize!(sol.t, length(sol.t) - 2)
            resize!(sol.u, length(sol.u) - 2)
        end
        work = field_work(sol, itps)
        kinetic = u -> 0.5species.m * sum(abs2, u[4:6]) / TP.eV
        delta = kinetic(sol.u[end]) - kinetic(sol.u[1])
        println(stderr, "dt=$dt, boundary=$(status[]), last_valid_t=$(sol.t[end]), K=$delta eV, Wconv=$(work.conv_eV), Whall=$(work.hall_eV), Wtotal=$(work.total_eV), residual=$(delta-work.total_eV) eV")
        dt == 0.025 || continue
        cumulative = zeros(3)
        powers = [species.q * dot(f(p0), SA[0.,0.,0.]) / TP.eV for f in (itps.conv,itps.hall,itps.total)]
        println("t,x,y,z,K,Wconv,Whall,Wtotal,Pconv,Phall")
        println(join([0.; p0/Rm; 0.; cumulative; powers[1:2]], ','))
        for i in 2:length(sol.t)
            a,b = sol.u[i-1],sol.u[i]
            p = SA[(a[1]+b[1])/2,(a[2]+b[2])/2,(a[3]+b[3])/2]
            v = SA[(a[4]+b[4])/2,(a[5]+b[5])/2,(a[6]+b[6])/2]
            for (j,f) in enumerate((itps.conv,itps.hall,itps.total))
                cumulative[j] += MarsTP._work_one(f,p,v,species.q,sol.t[i]-sol.t[i-1])/TP.eV
            end
            if i % 10 == 0 || i == length(sol.t)
                pos = SA[b[1],b[2],b[3]]
                vel = SA[b[4],b[5],b[6]]
                pc = species.q*dot(itps.conv(pos),vel)/TP.eV
                ph = species.q*dot(itps.hall(pos),vel)/TP.eV
                println(join([sol.t[i]; pos/Rm; kinetic(b); cumulative; pc; ph], ','))
            end
        end
        @assert isapprox(cumulative[1],work.conv_eV;rtol=1e-10)
        @assert isapprox(cumulative[2],work.hall_eV;rtol=1e-10)
        @assert abs(delta-work.total_eV)/max(abs(delta),1.) < 1e-3
    end
end
main()
