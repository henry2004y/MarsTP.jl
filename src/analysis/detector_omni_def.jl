# Nonrelativistic, 4pi direction-averaged DEF. All internal velocities are m/s.
mutable struct _OmniDEFAccumulator
    edges::Vector{Float64}
    mass::Float64
    species::String
    density::Vector{Float64}
    ev_density::Vector{Float64}
    outside::Float64
end

function _OmniDEFAccumulator(edges, species)
    haskey(TP.SpeciesDict, species) || throw(ArgumentError("Unknown species: $species"))
    e = Float64.(collect(edges))
    length(e) >= 2 && all(isfinite, e) && first(e) >= 0 &&
        all(x -> isfinite(x) && x > 0, diff(e)) ||
        throw(ArgumentError("energy_edges_eV must be finite, nonnegative and strictly increasing"))
    return _OmniDEFAccumulator(e, TP.SpeciesDict[species].m, String(species), zeros(length(e) - 1), zeros(length(e) - 1), 0.0)
end

_def_bin(e, E) = E < first(e) || E > last(e) ? 0 : min(searchsortedlast(e, E), length(e) - 1)

"""Finish a shared DEF accumulator without modifying it."""
function detector_omni_def(a::_OmniDEFAccumulator)
    widths = diff(a.edges)
    spectrum = a.ev_density ./ (4pi .* widths)
    dn = a.density ./ widths
    all(isfinite, spectrum) && all(isfinite, dn) && isfinite(a.outside) || error("DEF accumulation overflow")
    return (;
        def = spectrum, energy_edges_eV = copy(a.edges), energy_centers_eV = a.edges[1:(end - 1)] .+ widths ./ 2,
        energy_widths_eV = widths, dn_dE_m3_eV = dn, density_per_bin_m3 = copy(a.density),
        density_in_range_m3 = sum(a.density), density_outside_energy_range_m3 = a.outside,
        density_total_m3 = sum(a.density) + a.outside, species = a.species,
        units = "eV/(m^2 s eV sr)", solid_angle_sr = 4pi,
        definition = "4pi direction-averaged DEF; no isotropy assumption; nonrelativistic energy",
        energy_bin_convention = "[lower,upper), including final upper edge",
    )
end

# Adaptive Simpson integration after exact energy-edge splitting. The integrand
# is |v|^3; split also at its minimum to handle velocity reversal at zero speed.
function _def_integral(f, a, b; rtol = 1.0e-9)
    fa, fm, fb = f(a), f((a + b) / 2), f(b)
    all(isfinite, (fa, fm, fb)) || error("DEF integrand overflow")
    s = (b - a) * (fa + 4fm + fb) / 6
    tol = max(rtol * abs(s), floatmin(Float64))
    function refine(a, b, fa, fm, fb, s, tol, depth)
        m = (a + b) / 2
        fl, fr = f((a + m) / 2), f((m + b) / 2)
        sl = (m - a) * (fa + 4fl + fm) / 6; sr = (b - m) * (fm + 4fr + fb) / 6
        all(isfinite, (sl, sr)) || error("DEF quadrature overflow")
        abs(sl + sr - s) <= 15tol && return sl + sr + (sl + sr - s) / 15
        depth > 0 || error("DEF quadrature did not converge; refine saved trajectory cadence")
        return refine(a, m, fa, fl, fm, sl, tol / 2, depth - 1) + refine(m, b, fm, fr, fb, sr, tol / 2, depth - 1)
    end
    return refine(a, b, fa, fm, fb, s, tol, 24)
end

function _def_segment!(acc::_OmniDEFAccumulator, v0, v1, dt, density_rate)
    dt > 0 && isfinite(dt) && isfinite(density_rate) && density_rate >= 0 || throw(ArgumentError("Invalid DEF segment"))
    density_rate == 0 && return acc
    dv = v1 - v0; A = dot(dv, dv); B = 2dot(v0, dv); C = dot(v0, v0)
    alpha = acc.mass / (2TP.eV)
    all(isfinite, (A, B, C, alpha)) || error("DEF energy overflow")
    cuts = [0.0, 1.0]
    if A > 0
        vertex = -B / (2A)
        0 < vertex < 1 && push!(cuts, vertex)
        for edge in acc.edges
            c = C - edge / alpha
            disc = B^2 - 4A * c
            disc < 0 && continue
            q = -0.5 * (B + copysign(sqrt(disc), B))
            roots = q == 0 ? (-B / (2A),) : (q / A, c / q)
            for t in roots
                0 < t < 1 && push!(cuts, t)
            end
        end
    end
    sort!(unique!(cuts))
    for j in 1:(length(cuts) - 1)
        lo, hi = cuts[j], cuts[j + 1]
        E = alpha * sum(abs2, v0 + ((lo + hi) / 2) * dv)
        k = _def_bin(acc.edges, E)
        dn = density_rate * dt * (hi - lo)
        if k == 0
            acc.outside += dn
        else
            acc.density[k] += dn
            acc.ev_density[k] += density_rate * dt * alpha * _def_integral(t -> norm(v0 + t * dv)^3, lo, hi)
        end
    end
    return acc
end

"""
    detector_omni_def(solutions; detector_m, side_m, rate_weights_s,
                      energy_edges_eV, species="O2+")

Compute 4pi direction-averaged differential energy flux directly from forward
trajectories: sum(Q*integral(E*v*dt))/(4pi*Vdet*DeltaE). E is eV, speed m/s,
Q particles/s, and output `def` has units eV/(m^2 s eV sr). No isotropy is
assumed. Synchronized saved states are Cartesian m, m/s with increasing seconds.
Spatial and energy crossings are split on each piecewise-linear saved segment;
Ev is integrated within each energy bin, not replaced by its bin-center value.
No velocity cube or angular distribution is required. Lost sub-step curvature
requires saved-cadence convergence. Reports density outside the energy range.
"""
function detector_omni_def(solutions; detector_m, side_m, rate_weights_s, energy_edges_eV, species = "O2+")
    a = _OmniDEFAccumulator(energy_edges_eV, species)
    length(detector_m) == 3 && all(isfinite, detector_m) && isfinite(side_m) && side_m > 0 ||
        throw(ArgumentError("Invalid detector cube"))
    center = SVector{3, Float64}(detector_m);side = Float64(side_m)
    lower, upper = center .- side / 2, center .+ side / 2
    all(isfinite, lower) && all(isfinite, upper) && all(upper .> lower) && isfinite(side^3) && side^3 > 0 ||
        throw(ArgumentError("Unrepresentable detector cube"))
    trajectories = hasproperty(solutions, :t) ? (solutions,) : solutions isa AbstractVector ? solutions : solutions.u
    rate_weights_s isa Union{AbstractVector, Tuple} && length(trajectories) == length(rate_weights_s) ||
        throw(ArgumentError("One rate is required per trajectory"))
    for (traj, rate) in zip(trajectories, rate_weights_s)
        isfinite(rate) && rate >= 0 || throw(ArgumentError("Invalid particle rate"))
        tr = _psd_trajectory(traj, TP.SpeciesDict[species])
        for j in 1:(length(tr.t) - 1)
            dt = Float64(tr.t[j + 1]) - Float64(tr.t[j])
            isfinite(dt) && dt > 0 || throw(ArgumentError("Forward times must strictly increase"))
            uj = tr.u[j]; uj1 = tr.u[j + 1]
            x = SVector{3, Float64}(uj[1], uj[2], uj[3]); dx = SVector{3, Float64}(uj1[1], uj1[2], uj1[3]) - x
            v = SVector{3, Float64}(uj[4], uj[5], uj[6]); dv = SVector{3, Float64}(uj1[4], uj1[5], uj1[6]) - v
            all(isfinite, dx) && all(isfinite, dv) || error("Segment overflow")
            hit = _forward_cube_interval(x, dx, lower, upper)
            hit === nothing && continue
            lo, hi = hit[1:2]
            _def_segment!(a, v + lo * dv, v + hi * dv, dt * (hi - lo), rate / side^3)
        end
    end
    return merge(detector_omni_def(a), (; method = :trajectory_residence, detector_m = center, side_m = side))
end

function _def_vdf!(a::_OmniDEFAccumulator, f, axes, widths)
    length(axes) == 3 && length(widths) == 3 || throw(ArgumentError("Three velocity axes and widths required"))
    size(f) == Tuple(length.(axes)) || throw(ArgumentError("PSD shape must match velocity axes"))
    for k in 1:3
        !isempty(axes[k]) && all(isfinite, axes[k]) && all(>(0), diff(axes[k])) || throw(ArgumentError("Invalid velocity axis"))
        widths[k] isa Real ? (isfinite(widths[k]) && widths[k] > 0 || throw(ArgumentError("Invalid cell width"))) :
            (length(widths[k]) == length(axes[k]) && all(x -> isfinite(x)&&x > 0, widths[k]) || throw(ArgumentError("Invalid cell widths")))
    end
    all(x -> isfinite(x)&&x >= 0, f) || throw(ArgumentError("PSD must be finite and nonnegative"))
    for I in CartesianIndices(f)
        v = SVector{3, Float64}(ntuple(k -> axes[k][I[k]], 3))
        dn = f[I] * prod(widths[k] isa Real ? widths[k] : widths[k][I[k]] for k in 1:3)
        E = a.mass * dot(v, v) / (2TP.eV)
        isfinite(E) && isfinite(dn) || error("DEF VDF overflow")
        k = _def_bin(a.edges, E)
        if k == 0
            a.outside += dn
        else
            a.density[k] += dn; a.ev_density[k] += dn * E * norm(v)
        end
    end
    return a
end

"""
    detector_omni_def(psd3d, velocity_axes_m_s; velocity_cell_widths_m_s,
                      energy_edges_eV, species="O2+")

Bin a Cartesian 3D detector PSD [s^3 m^-6] into omnidirectional DEF using
f*dV*E*v/(4pi*DeltaE). Axes and widths are tuples in m/s, ordered (vx,vy,vz).
Each width is a scalar or per-axis vector. This is rectangular velocity-node
quadrature: each full cell's density is assigned to its node's energy. Check
velocity-grid convergence, especially for narrow energy bins. A 2D projection
cannot recover omni DEF. Finite velocity-domain coverage must be checked;
uncomputed directions are not extrapolated or renormalized to 4pi coverage.
"""
function detector_omni_def(f::AbstractArray{<:Real, 3}, axes; velocity_cell_widths_m_s, energy_edges_eV, species = "O2+")
    a = _OmniDEFAccumulator(energy_edges_eV, species)
    _def_vdf!(a, f, axes, velocity_cell_widths_m_s)
    return merge(detector_omni_def(a), (; method = :velocity_node_quadrature))
end
