"""
    ForwardPSDAccumulator(; detector_m, side_m, vlim, vgrid, species="O2+",
                          velocity_unit=:m_s, coordinate_system="unspecified",
                          energy_edges_eV=nothing)

Incremental, sparse 3D occupancy accumulator for `forward_psd`. Feed complete
trajectories with `accumulate_forward_psd!`, then call `finish_forward_psd`.
Uses exactly the same piecewise-linear saved-state estimator as `forward_psd`.
Memory scales with occupied velocity bins and trajectory diagnostics, not all
saved trajectory states. `vgrid` is bin count, not bin width.
Optional `energy_edges_eV` accumulates an independent direction-averaged energy
spectrum from residence segments, returned as `omni_def` by the finish call.
"""
mutable struct ForwardPSDAccumulator
    detector_m::SVector{3, Float64}
    side_m::Float64
    edges::NTuple{3, Vector{Float64}}
    species::String
    coordinate_system::String
    occupancy::Dict{NTuple{3, Int}, Float64}
    sumsq::Dict{NTuple{3, Int}, Float64}
    unique::Dict{NTuple{3, Int}, Int}
    particle_ids::Vector{Int}
    rates::Vector{Float64}
    residence::Vector{Float64}
    outside::Vector{Float64}
    retcodes::Vector{String}
    omni::Union{Nothing, _OmniDEFAccumulator}
end

function ForwardPSDAccumulator(;
        detector_m, side_m, vlim, vgrid, species = "O2+",
        velocity_unit = :m_s, coordinate_system = "unspecified", energy_edges_eV = nothing
    )
    haskey(TP.SpeciesDict, species) || throw(ArgumentError("Unknown species: $species"))
    length(detector_m) == 3 && all(isfinite, detector_m) || throw(ArgumentError("Invalid detector center"))
    isfinite(side_m) && side_m > 0 || throw(ArgumentError("side_m must be positive"))
    center = SVector{3, Float64}(detector_m); side = Float64(side_m)
    lower, upper = center .- side / 2, center .+ side / 2
    all(isfinite, lower) && all(isfinite, upper) && all(upper .> lower) &&
        isfinite(side^3) && side^3 > 0 || throw(ArgumentError("Unrepresentable cube"))
    edges = _psd_edges(vlim, vgrid, velocity_unit)
    for k in 1:3
        all(w -> isfinite(w * side^3) && w * side^3 > 0, diff(edges[k])) ||
            throw(ArgumentError("Unrepresentable phase-space volume"))
    end
    return ForwardPSDAccumulator(
        center, side, edges, String(species), String(coordinate_system),
        Dict{NTuple{3, Int}, Float64}(), Dict{NTuple{3, Int}, Float64}(), Dict{NTuple{3, Int}, Int}(),
        Int[], Float64[], Float64[], Float64[], String[],
        energy_edges_eV === nothing ? nothing : _OmniDEFAccumulator(energy_edges_eV, species)
    )
end

# The single shared spatial clipping kernel: cube [lower,upper).
@inline function _forward_cube_interval(x, dx, lower, upper)
    xb = x + dx
    if (max(x[1], xb[1]) < lower[1] || min(x[1], xb[1]) >= upper[1]) ||
            (max(x[2], xb[2]) < lower[2] || min(x[2], xb[2]) >= upper[2]) ||
            (max(x[3], xb[3]) < lower[3] || min(x[3], xb[3]) >= upper[3])
        return nothing
    end
    lo, hi = 0.0, 1.0; entry, exit = 0, 0
    for k in 1:3
        if dx[k] == 0
            lower[k] <= x[k] < upper[k] || return nothing
        else
            p, q = (lower[k] - x[k]) / dx[k], (upper[k] - x[k]) / dx[k]
            fp, fq = -k, k
            if p > q
                p, q = q, p;fp, fq = fq, fp
            end
            if p >= lo
                lo = p;entry = fp
            end
            if q <= hi
                hi = q;exit = fq
            end
            hi > lo || return nothing
        end
    end
    return lo, hi, entry, exit
end

"""
    accumulate_forward_psd!(acc, trajectory; rate_weight_s, particle_id=...,
                            segment_observer=nothing)

Add one complete trajectory (s, m, m/s), with physical source rate in s^-1.
Use one call per independent source draw so repeated visits are combined for
effective-sample diagnostics. `segment_observer`, if supplied, receives each
clipped residence segment with synchronized linearly interpolated velocities,
times, positions, and entry/exit face codes (±1=X, ±2=Y, ±3=Z, 0=no crossing).
The observer does not change the estimator. Discard the accumulator if a call
throws; partial accumulation is not rolled back.
"""
function accumulate_forward_psd!(
        acc::ForwardPSDAccumulator, wrapped;
        rate_weight_s, particle_id = length(acc.rates) + 1, segment_observer = nothing
    )
    rate = Float64(rate_weight_s)
    isfinite(rate) && rate >= 0 || throw(ArgumentError("Invalid particle rate"))
    particle_id isa Integer && particle_id > 0 || throw(ArgumentError("Invalid particle ID"))
    traj = _psd_trajectory(wrapped, TP.SpeciesDict[acc.species])
    lower, upper = acc.detector_m .- acc.side_m / 2, acc.detector_m .+ acc.side_m / 2
    edges = acc.edges;dims = length.(edges) .- 1
    localbins = nothing; residence = 0.0;outside = 0.0;cuts = nothing
    @inbounds for j in 1:(length(traj.t) - 1)
        a, b = traj.u[j], traj.u[j + 1]
        ta, tb = Float64(traj.t[j]), Float64(traj.t[j + 1]);dt = tb - ta
        isfinite(dt) && dt > 0 || throw(ArgumentError("Forward times must strictly increase"))
        x = SVector{3, Float64}(a[1], a[2], a[3]);dx = SVector{3, Float64}(b[1], b[2], b[3]) - x
        v = SVector{3, Float64}(a[4], a[5], a[6]);dv = SVector{3, Float64}(b[4], b[5], b[6]) - v
        all(isfinite, dx) && all(isfinite, dv) || throw(ArgumentError("Segment overflow"))
        hit = _forward_cube_interval(x, dx, lower, upper)
        hit === nothing && continue
        lo, hi, entry, exit = hit
        residence += dt * (hi - lo)
        if acc.omni !== nothing
            _def_segment!(acc.omni, v + lo * dv, v + hi * dv, dt * (hi - lo), rate / acc.side_m^3)
        end
        if segment_observer !== nothing
            segment_observer(
                (;
                    particle_id = Int(particle_id), rate_weight_s = rate,
                    t0_s = ta + lo * dt, t1_s = ta + hi * dt, x0_m = x + lo * dx, x1_m = x + hi * dx,
                    v0_ms = v + lo * dv, v1_ms = v + hi * dv, entry_face = entry, exit_face = exit,
                )
            )
        end
        if cuts === nothing
            cuts = Float64[lo, hi]
        else
            empty!(cuts);push!(cuts, lo, hi)
        end
        for k in 1:3
            dv[k] == 0 && continue
            v0, v1 = v[k] + lo * dv[k], v[k] + hi * dv[k]
            for n in searchsortedfirst(edges[k], min(v0, v1)):searchsortedlast(edges[k], max(v0, v1))
                alpha = (edges[k][n] - v[k]) / dv[k]
                lo < alpha < hi && push!(cuts, alpha)
            end
        end
        sort!(cuts)
        for n in 1:(length(cuts) - 1)
            duration = dt * (cuts[n + 1] - cuts[n]);duration > 0 || continue
            vm = v + ((cuts[n] + cuts[n + 1]) / 2) * dv
            bins = ntuple(3) do k
                value = vm[k]
                value < first(edges[k]) || value > last(edges[k]) ? 0 :
                    min(searchsortedlast(edges[k], value), dims[k])
            end
            if any(==(0), bins)
                outside += duration
            elseif rate > 0
                amount = rate * duration
                acc.occupancy[bins] = get(acc.occupancy, bins, 0.0) + amount
                if localbins === nothing
                    localbins = Dict{NTuple{3, Int}, Float64}(bins => amount)
                else
                    localbins[bins] = get(localbins, bins, 0.0) + amount
                end
            end
        end
    end
    if localbins !== nothing
        for (bin, amount) in localbins
            acc.sumsq[bin] = get(acc.sumsq, bin, 0.0) + amount^2
            acc.unique[bin] = get(acc.unique, bin, 0) + 1
        end
    end
    push!(acc.particle_ids, particle_id);push!(acc.rates, rate);push!(acc.residence, residence);push!(acc.outside, outside)
    push!(acc.retcodes, hasproperty(traj, :retcode) ? string(traj.retcode) : "unavailable")
    return acc
end

"""
    finish_forward_psd(acc; option="3D", storage=:dense)

Normalize accumulated 3D occupancy once and return a PSD. 2D options integrate
over the entire omitted velocity axis within `vlim`. `storage=:sparse` returns
`psd` as a Dict of one-based bin tuples to SI values; `:dense` preserves the
original `forward_psd` array API. Calling this does not modify the accumulator.
"""
function finish_forward_psd(acc::ForwardPSDAccumulator; option = "3D", storage = :dense)
    key = lowercase(replace(string(option), "v" => "", "V" => "", "-" => ""))
    kept = key in ("3d", "xyz") ? (1, 2, 3) : key == "xy" ? (1, 2) : key == "yz" ? (2, 3) : key == "xz" ? (1, 3) :
        throw(ArgumentError("option must be 3D, Vx-Vy, Vy-Vz, or Vx-Vz"))
    storage in (:dense, :sparse) || throw(ArgumentError("storage must be :dense or :sparse"))
    edges = acc.edges;widths = diff.(edges);dims = length.(widths)
    occupancy = Dict{NTuple{length(kept), Int}, Float64}()
    for (bin, amount) in acc.occupancy
        index = map(k -> bin[k], kept)
        occupancy[index] = get(occupancy, index, 0.0) + amount
    end
    psd = storage == :dense ? zeros(map(k -> dims[k], kept)) : Dict{NTuple{length(kept), Int}, Float64}()
    for (index, amount) in occupancy
        volume = acc.side_m^3 * prod(widths[kept[j]][index[j]] for j in eachindex(kept))
        isfinite(volume) && volume > 0 || throw(ArgumentError("Unrepresentable phase-space volume"))
        value = amount / volume
        isfinite(value) || throw(ArgumentError("PSD accumulation overflow"))
        if storage == :dense
            psd[index...] = value
        else
            psd[index] = value
        end
    end
    total = sum(acc.rates .* acc.residence) / acc.side_m^3
    outside = sum(acc.rates .* acc.outside) / acc.side_m^3
    inside = sum(values(acc.occupancy); init = 0.0) / acc.side_m^3
    all(isfinite, (total, outside, inside)) || throw(ArgumentError("PSD accumulation overflow"))
    return (;
        omni_def = acc.omni === nothing ? nothing : detector_omni_def(acc.omni), psd, axes = map(k -> (:vx, :vy, :vz)[k], kept), storage,
        velocity_centers_m_s = map(k -> (edges[k][1:(end - 1)] + edges[k][2:end]) / 2, kept),
        velocity_edges_m_s = map(k -> edges[k], kept), all_velocity_edges_m_s = edges,
        units = length(kept) == 3 ? "s^3 m^-6" : "s^2 m^-5", density_in_range_m3 = inside,
        density_outside_vlim_m3 = outside, density_total_m3 = total,
        residence_s = copy(acc.residence), outside_vlim_residence_s = copy(acc.outside),
        retcodes = copy(acc.retcodes), particle_ids = copy(acc.particle_ids), rate_weights_s = copy(acc.rates),
        species = acc.species, detector_m = acc.detector_m, side_m = acc.side_m,
        coordinate_system = acc.coordinate_system, interpolation = :piecewise_linear,
    )
end
