using MarsTP, StaticArrays, LinearAlgebra, Random
const TP = MarsTP.TP
const BP = MarsTP.BorisPushers

# Callback records positions at integer times and detects radial exits.
# TestParticle 0.24 synchronizes velocity before this callback, so check the
# position/boundary before requiring a finite in-domain velocity.
function trace(v0, param, rin, rout, dt; limit=500.0)
    p0 = SA[0.0, 0.0, 2Rm]
    previous, previous_t = Ref(p0), Ref(0.0)
    points, times = [p0 / Rm], [0.0]
    status = Ref("time_limit")
    steps = Ref(0)
    stride = round(Int, 0.5 / abs(dt))
    function boundary(u, p, t)
        all(isfinite, u[1:3]) || error("Nonfinite backtrace position at t=$t")
        r = SA[u[1], u[2], u[3]]
        radius = norm(r)
        if radius < rin || radius > rout
            inner = radius < rin
            target = inner ? rin : rout
            status[] = inner ? "inner" : "outer"
            a, b = 0.0, 1.0
            for _ in 1:45
                m = (a+b)/2
                rm = norm(previous[] + m*(r-previous[]))
                crossed = inner ? rm < target : rm > target
                if crossed; b=m; else; a=m; end
            end
            f = (a+b)/2
            push!(points, (previous[] + f*(r-previous[]))/Rm)
            push!(times, previous_t[] + f*(t-previous_t[]))
            return true
        end
        all(isfinite,u) || error("Nonfinite in-domain velocity at t=$t")
        steps[] += 1
        if steps[] % stride == 0 || t <= -limit + abs(dt)/2
            push!(points, r/Rm); push!(times, t)
        end
        previous[] = r; previous_t[] = t
        return false
    end
    prob = TP.TraceProblem(vcat(p0, v0), (0.0, -limit), param)
    TP.solve(prob, BP.Boris(); dt, isoutside=boundary,
        save_everystep=false, save_start=false, save_end=false,
        maxiters=ceil(Int, limit/abs(dt))+1)
    @assert all(diff(times) .< 0)
    @assert status[] != "time_limit" || abs(times[end]+limit) < 1e-7
    return (;status=status[], points, times)
end

function main()
    fields = load_mhd_fields(;electric_field=:total)
    param = MarsTP.mhd_param(fields;species="O2+")
    rng = Xoshiro(20260905)
    n = parse(Int, get(ENV, "PARTICLE_COUNT", "5000"))
    n > 0 || error("PARTICLE_COUNT must be positive")
    velocities = map(1:n) do _
        speed = (10 + 190rand(rng))*1e3
        angle = 2pi*rand(rng)
        SA[speed*cos(angle), 0.0, speed*sin(angle)]
    end
    println(stderr, "Julia $VERSION; TestParticle $(pkgversion(TP)); O2+; total E; Rm=$Rm m")
    println(stderr, "Single-particle pilot")
    trace(velocities[1], param, first(fields.r), last(fields.r), -0.05)
    for (i, v0) in enumerate(velocities)
        coarse = trace(v0, param, first(fields.r), last(fields.r), -0.1)
        fine = trace(v0, param, first(fields.r), last(fields.r), -0.05)
        # Compare shared 0.5 s samples, excluding interpolated boundary endpoints.
        nc = min(length(coarse.points), length(fine.points))-1
        separation = maximum(norm(coarse.points[k]-fine.points[k]) for k in 1:nc)
        print("{\"id\":",i,",\"speed_kms\":",norm(v0)/1e3,
            ",\"v0_kms\":[",join(v0/1e3,','),"],\"status\":\"",fine.status,
            "\",\"coarse_status\":\"",coarse.status,"\",\"max_separation_Rm\":",separation,
            ",\"endpoint_difference_Rm\":",norm(coarse.points[end]-fine.points[end]),
            ",\"time_difference_s\":",abs(coarse.times[end]-fine.times[end]),
            ",\"times_s\":[",join(fine.times,','),"],\"points\":[")
        for (k,r) in enumerate(fine.points)
            k>1 && print(',')
            print('[',join(r,','),']')
        end
        println("]}"); flush(stdout)
        i%100 == 0 && println(stderr,"Completed $i/$n, including step-halving check")
    end
end
main()
