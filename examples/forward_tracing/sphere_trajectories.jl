# Stream trajectories to Python over stdout; no simulation files are written.
using MarsTP, StaticArrays, LinearAlgebra, Random
const TP = MarsTP.TP
const BP = MarsTP.BorisPushers

function trace_particle(p0, param, dt, limit)
    previous = Ref(p0)
    previous_t = Ref(0.0)
    endpoint = Ref(p0)
    hit_t = Ref(0.0)
    status = Ref("time_limit")
    function boundary(u, p, t)
        r = SA[u[1], u[2], u[3]]
        if !all(isfinite, u[1:3])
            status[] = "nonfinite"
            return true
        end
        radius = norm(r)
        if radius < Rinner || radius > Router
            target = radius < Rinner ? Rinner : Router
            status[] = radius < Rinner ? "inner" : "outer"
            # Intersect the last integration segment with the boundary sphere.
            a, b = 0.0, 1.0
            for _ in 1:45
                m = (a + b) / 2
                rm = norm(previous[] + m * (r - previous[]))
                crossed = target == Rinner ? rm < target : rm > target
                if crossed
                    b = m
                else
                    a = m
                end
            end
            fraction = (a + b) / 2
            endpoint[] = previous[] + fraction * (r - previous[])
            hit_t[] = previous_t[] + fraction * (t - previous_t[])
            return true
        end
        all(isfinite, u) || error("Nonfinite in-domain velocity")
        previous[] = r
        previous_t[] = t
        return false
    end
    prob = TP.TraceProblem(vcat(p0, SA[0.0, 0.0, 0.0]), (0.0, limit), param)
    sol = TP.solve(
        prob, BP.Boris(); dt, isoutside = boundary,
        saveat = 5dt, maxiters = ceil(Int, limit / dt) + 1
    )
    if length(sol.t) >= 2 && sol.t[end] == sol.t[end - 1]
        resize!(sol.t, length(sol.t) - 2)
        resize!(sol.u, length(sol.u) - 2)
    end
    points = [SA[u[1], u[2], u[3]] / Rm for u in sol.u]
    if status[] in ("inner", "outer")
        push!(points, endpoint[] / Rm)
    else
        hit_t[] = sol.t[end]
    end
    return (; status = status[], time = hit_t[], points)
end

function main()
    n = parse(Int, get(ENV, "PARTICLE_COUNT", "1000"))
    dt = parse(Float64, get(ENV, "TRACE_DT", "0.1"))
    altitude = parse(Float64, get(ENV, "RELEASE_ALTITUDE_KM", "800"))
    release_radius = Rm + altitude * 1.0e3
    Rinner <= release_radius < Router || error("Release altitude outside field domain")
    limit = parse(Float64, get(ENV, "TRACE_LIMIT", "20000"))
    rng = Xoshiro(20260905)
    fields = load_mhd_fields()
    param = MarsTP.mhd_param(fields; species = "O2+")
    # Uniform solid angle: cos(theta) uniform, not theta uniform.
    positions = map(1:n) do _
        mu = 2rand(rng) - 1
        phi = 2pi * rand(rng)
        direction = SA[sqrt(1 - mu^2) * cos(phi), sqrt(1 - mu^2) * sin(phi), mu]
        # 10 nanometres inside the tabulated field domain avoids roundoff NaNs.
        direction / norm(direction) * max(release_radius, fields.r[1] + 1.0e-8)
    end
    println(stderr, "Loaded fields; tracing $n O2+ particles at $altitude km, dt=$dt s, guard=$limit s, threads=$(Threads.nthreads())")
    efield, bfield = param.E, param.B
    println(stderr, "Total E at (3,0,0) RM [V/m]: ", efield(SA[3Rm, 0.0, 0.0], 0.0))
    println(stderr, "B at (3,0,0) RM [T]: ", bfield(SA[3Rm, 0.0, 0.0], 0.0))
    dayside = filter(p -> p[1] > 0, positions)
    outward = count(p -> dot(efield(p, 0.0), p) > 0, dayside)
    println(stderr, "X>0 release points with outward initial electric acceleration: $outward / $(length(dayside))")
    output_lock = ReentrantLock()
    completed = Threads.Atomic{Int}(0)
    return Threads.@threads for i in 1:n
        result = trace_particle(positions[i], param, dt, limit)
        buf = IOBuffer()
        print(
            buf, "{\"id\":", i, ",\"status\":\"", result.status,
            "\",\"time\":", result.time, ",\"points\":["
        )
        for (j, r) in enumerate(result.points)
            j > 1 && print(buf, ',')
            print(buf, '[', r[1], ',', r[2], ',', r[3], ']')
        end
        println(buf, "]}")
        bytes = take!(buf)
        lock(output_lock) do
            write(stdout, bytes)
            flush(stdout)
        end
        done = Threads.atomic_add!(completed, 1) + 1
        done % 100 == 0 && println(stderr, "Completed $done / $n")
    end
end
main()
