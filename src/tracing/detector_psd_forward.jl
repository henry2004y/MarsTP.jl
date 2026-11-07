"""
    forward_psd(solutions; detector_m, side_m, vlim, vgrid, species="O2+",
                rate_weights_s, option="3D", velocity_unit=:m_s,
                coordinate_system="unspecified", storage=:dense, energy_edges_eV=nothing)

Steady-source, finite-volume velocity PSD from saved forward trajectories.
`solutions` accepts `trace_forward` output, a TestParticle ensemble, a vector of
trajectories, or one trajectory with `t` and `u` ([x,y,z,vx,vy,vz,...]). Saved
positions/velocities/times must be synchronized Cartesian m, m/s, s. The cube
is axis-aligned in that same frame; `detector_m` is its center, `side_m` its side.

`rate_weights_s` is required: one nonnegative physical particle rate per input
trajectory, in particles/s, already including source Monte Carlo normalization.
Density or dimensionless weights cannot substitute for rates. All trajectories
must belong to `species`; known TestParticle mass/charge parameters are checked.

`vlim` is a positive symmetric limit, `(min,max)` shared by all axes, or three
`(min,max)` pairs. `vgrid` is the number of bins (positive integer), or three
bin counts. `velocity_unit=:m_s` (default) or `:km_s` applies ONLY to `vlim`.
Returned velocity centers/edges are always m/s; returned PSD is always SI.
`option` is "3D", "Vx-Vy", "Vy-Vz", or "Vx-Vz" (also :xyz/:xy/:yz/:xz).
Two-dimensional output integrates the omitted axis over its specified range.

Returns `psd`, `axes`, `velocity_centers_m_s`, `velocity_edges_m_s`, `units`,
`density_in_range_m3`, `density_outside_vlim_m3`, `density_total_m3`, and
per-trajectory residence times and saved termination codes. Array dimension
order follows `axes`. 3D units are s^3 m^-6, 2D units are s^2 m^-5.

Accumulates Q*dt within the cube AND velocity bin. Each saved segment is linear
in position and velocity and is split at cube faces and velocity-bin edges.
Bins and cube use [lower,upper), including the final velocity upper edge.
This handles crossings with both endpoints outside. Curvature lost between
saved samples cannot be recovered: check timestep/save-cadence convergence.
Only saved time intervals contribute; no extrapolation or extra time/sample
normalization is performed. Failed solver return codes and invalid data error.
No tracing, field loading, plotting, or file writing is performed.
`storage=:sparse` optionally returns a Dict with one-based bin tuple keys.
This function and `forward_psd_saved` share `ForwardPSDAccumulator`.
With `energy_edges_eV`, `omni_def` additionally contains direction-averaged DEF
[eV/(m^2 s eV sr)] from trajectory residence, independent of `vlim`/`vgrid`
and `option`. Without energy edges it is `nothing`. Use `detector_omni_def`
for a standalone energy spectrum without allocating velocity PSD bins.
"""
function forward_psd(
        solutions; detector_m, side_m, vlim, vgrid,
        species = "O2+", rate_weights_s, option = "3D", velocity_unit = :m_s,
        coordinate_system = "unspecified", storage = :dense, energy_edges_eV = nothing
    )
    acc = ForwardPSDAccumulator(; detector_m, side_m, vlim, vgrid, species, velocity_unit, coordinate_system, energy_edges_eV)
    trajectories = hasproperty(solutions, :t) ? (solutions,) :
        solutions isa AbstractVector ? solutions : solutions.u
    rate_weights_s isa AbstractVector || rate_weights_s isa Tuple ||
        throw(ArgumentError("rate_weights_s must contain one rate per trajectory"))
    length(trajectories) == length(rate_weights_s) ||
        throw(ArgumentError("One rate_weights_s entry is required per trajectory"))
    for (i, traj) in enumerate(trajectories)
        accumulate_forward_psd!(acc, traj; rate_weight_s = rate_weights_s[i], particle_id = i)
    end
    return finish_forward_psd(acc; option, storage)
end

function _psd_edges(vlim, vgrid, velocity_unit)
    scale = velocity_unit == :m_s ? 1.0 : velocity_unit == :km_s ? 1000.0 :
        throw(ArgumentError("velocity_unit must be :m_s or :km_s"))
    limits = vlim isa Real ? ntuple(_ -> (-vlim, vlim), 3) :
        length(vlim) == 2 && all(x -> x isa Real, vlim) ? ntuple(_ -> vlim, 3) : vlim
    counts = vgrid isa Integer ? (vgrid, vgrid, vgrid) : vgrid
    length(limits) == 3 && length(counts) == 3 || throw(ArgumentError("Expected three axes"))
    return ntuple(3) do k
        counts[k] isa Integer && !(counts[k] isa Bool) && counts[k] > 0 ||
            throw(ArgumentError("vgrid must specify positive integer bin counts"))
        lim = limits[k]
        length(lim) == 2 || throw(ArgumentError("Each vlim must be (min,max)"))
        lo, hi = Float64(lim[1]) * scale, Float64(lim[2]) * scale
        isfinite(lo) && isfinite(hi) && hi > lo && isfinite(hi - lo) ||
            throw(ArgumentError("Velocity limits must be finite and increasing"))
        e = collect(range(lo, hi; length = counts[k] + 1))
        all(>(0), diff(e)) || throw(ArgumentError("Velocity bins are too narrow"))
        e
    end
end

@inline function _finite_psd_state6(u)
    return length(u) >= 6 &&
        @inbounds(
        isfinite(u[1]) && isfinite(u[2]) && isfinite(u[3]) &&
            isfinite(u[4]) && isfinite(u[5]) && isfinite(u[6])
    )
end

@inline function _is_strictly_increasing(t)
    n = length(t)
    @inbounds for i in 1:(n - 1)
        t[i + 1] > t[i] || return false
    end
    return true
end

function _psd_trajectory(sol, sp)
    while !hasproperty(sol, :t)
        hasproperty(sol, :u) && length(sol.u) == 1 ||
            throw(ArgumentError("Expected a trajectory or single-member ensemble"))
        sol = only(sol.u)
    end
    length(sol.t) == length(sol.u) && !isempty(sol.t) ||
        throw(ArgumentError("Trajectory needs matching nonempty times and states"))
    all(isfinite, sol.t) && all(_finite_psd_state6, sol.u) ||
        throw(ArgumentError("Trajectory has invalid times or Cartesian states"))
    if hasproperty(sol, :retcode)
        sol.retcode in (TP.ReturnCode.Success, TP.ReturnCode.Terminated) ||
            throw(ArgumentError("Failed trajectory: $(sol.retcode)"))
    end
    if hasproperty(sol, :prob) && hasproperty(sol.prob, :p)
        p = sol.prob.p
        if hasproperty(p, :q2m) && hasproperty(p, :m)
            isapprox(p.q2m, sp.q / sp.m; rtol = 1.0e-10) && isapprox(p.m, sp.m; rtol = 1.0e-10) ||
                throw(ArgumentError("Trajectory species does not match requested species"))
        elseif p isa Tuple && length(p) >= 4
            isapprox(p[1], sp.q / sp.m; rtol = 1.0e-10) && isapprox(p[2], sp.m; rtol = 1.0e-10) ||
                throw(ArgumentError("Trajectory species does not match requested species"))
        end
    end
    return sol
end
