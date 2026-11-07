"""
    write_trajectory_batch(path, trajectories; particle_ids, rate_weights_s,
        source_density_weights_m3=nothing, cell_ids=nothing,
        termination_codes=nothing, species="O2+", coordinate_system="unspecified",
        compress=false, work_itp=nothing, work_mode=:none, save_power=false,
        work_summaries=nothing)

Write one bounded batch of synchronized Cartesian trajectories to JLD2.
Never overwrites an existing path. Each p<ID>/state is Float64 7×N with rows
t,x,y,z,vx,vy,vz (s,m,m/s); rate_weight_s1 is a physical source rate (s^-1).
Compression is optional. A `complete=true` marker is written only after all
records succeed. Call this after each tracing batch, then release that batch;
do not first collect the entire 50 GB ensemble. Failed partial files are kept
for diagnosis and rejected by the reader. Input accepts trajectories with t/u.
Set work_mode=:steps or :summary and pass work_itp to enable format v2 diagnostics.
Steps stores p<ID>/work/dW_{total,conv,hall}_eV with N-1 intervals; summary
stores signed totals, positive/negative sums, endpoint kinetic energy and closure.
Optional endpoint powers use N samples. Supply full cadence trajectories: this
writer cannot recover work on omitted integration steps. :none preserves v1.
`work_summaries` accepts device-computed summaries in :summary mode without
power output. They must represent all accepted integration steps, including
the partial terminal step; this bypasses host work recomputation.
"""
function _has_uniform_interior_dt(t::AbstractVector{<:Real})
    n = length(t)
    n <= 2 && return true, (n == 2 ? Float64(t[2] - t[1]) : 0.0)
    t0 = Float64(t[1])
    dt = Float64(t[2] - t[1])
    (isfinite(dt) && dt > 0.0) || return false, NaN
    tol = 1.0e-7 * max(1.0, abs(Float64(t[end]) - t0))
    @inbounds for j in 2:(n - 1)
        abs(Float64(t[j]) - (t0 + (j - 1) * dt)) <= tol || return false, NaN
    end
    return true, dt
end

function _reconstruct_packed_times(t0::Float64, dt::Float64, tend::Float64, n::Int)
    t = Vector{Float64}(undef, n)
    n == 0 && return t
    t[1] = t0
    @inbounds for j in 2:(n - 1)
        t[j] = t0 + (j - 1) * dt
    end
    n >= 2 && (t[n] = tend)
    return t
end

function write_trajectory_batch(
        path, trajectories; particle_ids, rate_weights_s,
        source_density_weights_m3 = nothing, cell_ids = nothing, termination_codes = nothing,
        species = "O2+", coordinate_system = "unspecified", compress = false,
        work_itp = nothing, work_mode = :none, save_power = false, work_summaries = nothing,
        work_results = nothing, packed::Bool = false, state_eltype::Type{<:AbstractFloat} = Float64
    )
    work_mode in (:none, :summary, :steps) || throw(ArgumentError("Invalid work mode"))
    state_eltype in (Float32, Float64) || throw(ArgumentError("state_eltype must be Float32 or Float64"))
    work_mode == :none && save_power && throw(ArgumentError("Power requires work mode"))
    work_summaries === nothing || (work_mode == :summary && !save_power && length(work_summaries) == length(trajectories)) ||
        throw(ArgumentError("Precomputed work requires summary mode, no power, and matching batch length"))
    work_results === nothing || (work_mode != :none && length(work_results) == length(trajectories)) ||
        throw(ArgumentError("Precomputed work_results requires active work mode and matching batch length"))
    work_mode != :none && work_itp === nothing && work_summaries === nothing && work_results === nothing && throw(ArgumentError("Work interpolators required"))
    if work_summaries !== nothing
        required = (
            :total_eV, :conv_eV, :hall_eV, :positive_total_eV, :positive_conv_eV, :positive_hall_eV,
            :negative_total_eV, :negative_conv_eV, :negative_hall_eV, :initial_kinetic_eV, :final_kinetic_eV,
            :delta_kinetic_eV, :energy_residual_eV, :max_abs_energy_residual_eV, :field_sum_residual_eV,
        )
        all(w -> all(k -> hasproperty(w, k) && isfinite(getproperty(w, k)), required), work_summaries) ||
            throw(ArgumentError("Invalid precomputed work summary"))
    end
    ispath(path) && throw(ArgumentError("Refusing to overwrite $path"))
    haskey(TP.SpeciesDict, species) || throw(ArgumentError("Unknown species"))
    n = length(trajectories)
    n > 0 && length(particle_ids) == n && length(rate_weights_s) == n || throw(ArgumentError("Batch length mismatch"))
    all(i -> i isa Integer && i > 0, particle_ids) && length(unique(particle_ids)) == n ||
        throw(ArgumentError("Particle IDs must be unique positive integers"))
    all(q -> isfinite(q) && q >= 0, rate_weights_s) || throw(ArgumentError("Invalid particle rates"))
    for values in (source_density_weights_m3, cell_ids, termination_codes)
        values === nothing || length(values) == n || throw(ArgumentError("Optional batch length mismatch"))
    end
    source_density_weights_m3 === nothing || all(q -> isfinite(q) && q >= 0, source_density_weights_m3) ||
        throw(ArgumentError("Invalid source density weights"))
    cell_ids === nothing || all(i -> i isa Integer && i > 0, cell_ids) || throw(ArgumentError("Invalid cell IDs"))
    # Validate before opening, so bad input does not create an apparently valid batch.
    sp_obj = TP.SpeciesDict[species]
    unwrapped = [_psd_trajectory(wrapped, sp_obj) for wrapped in trajectories]
    for traj in unwrapped
        _is_strictly_increasing(traj.t) || throw(ArgumentError("Forward times must strictly increase"))
    end
    mkpath(dirname(abspath(path)))
    jldopen(path, "w"; compress) do file
        if packed
            file["format_version"] = 3
            file["packed"] = true
            file["state_eltype"] = String(nameof(state_eltype))
            if work_mode != :none
                file["work_mode"] = String(work_mode)
                file["work_method"] = work_summaries === nothing ? "midpoint q E dot v dt on supplied intervals" :
                    "midpoint q E dot v dt on all accepted integration intervals (precomputed)"
                file["work_unit"] = "eV"
                file["power_unit"] = "eV s^-1"
            end
            file["particle_ids"] = Int32.(particle_ids)
            file["species"] = String(species)
            file["coordinate_system"] = String(coordinate_system)
            file["state_units"] = ["m", "m", "m", "m s^-1", "m s^-1", "m s^-1"]
            file["time_unit"] = "s"
            file["weight_unit"] = "s^-1"
            file["rate_weights_s1"] = Float64.(rate_weights_s)
            source_density_weights_m3 === nothing || (file["source_density_weights_m3"] = Float64.(source_density_weights_m3))
            cell_ids === nothing || (file["cell_ids"] = Int32.(cell_ids))
            codes = String[termination_codes === nothing ? (hasproperty(unwrapped[i], :retcode) ? string(unwrapped[i].retcode) : "unavailable") : string(termination_codes[i]) for i in 1:n]
            file["termination_codes"] = codes

            point_offsets = Vector{Int32}(undef, n + 1)
            point_offsets[1] = 1
            t_start_s = Vector{Float64}(undef, n)
            dt_step_s = Vector{Float64}(undef, n)
            t_end_s = Vector{Float64}(undef, n)
            all_uniform = true
            for i in 1:n
                ti = unwrapped[i].t
                ni = length(ti)
                point_offsets[i + 1] = point_offsets[i] + Int32(ni)
                t_start_s[i] = Float64(ti[1])
                t_end_s[i] = Float64(ti[end])
                ok, dti = _has_uniform_interior_dt(ti)
                dt_step_s[i] = dti
                all_uniform &= ok
            end
            total_pts = Int(point_offsets[end] - 1)
            states = Matrix{state_eltype}(undef, 6, total_pts)
            for i in 1:n
                ui = unwrapped[i].u
                p0 = Int(point_offsets[i]) - 1
                @inbounds for j in eachindex(ui)
                    uj = ui[j]
                    col = p0 + j
                    states[1, col] = state_eltype(uj[1])
                    states[2, col] = state_eltype(uj[2])
                    states[3, col] = state_eltype(uj[3])
                    states[4, col] = state_eltype(uj[4])
                    states[5, col] = state_eltype(uj[5])
                    states[6, col] = state_eltype(uj[6])
                end
            end
            file["point_offsets"] = point_offsets
            file["t_start_s"] = t_start_s
            file["dt_step_s"] = dt_step_s
            file["t_end_s"] = t_end_s
            if !all_uniform
                times_fallback = Vector{Float64}(undef, total_pts)
                for i in 1:n
                    ti = unwrapped[i].t
                    p0 = Int(point_offsets[i]) - 1
                    @inbounds for j in eachindex(ti)
                        times_fallback[p0 + j] = Float64(ti[j])
                    end
                end
                file["times_s"] = times_fallback
            end
            file["states"] = states

            if work_mode != :none
                w_list = [
                    work_results !== nothing ? work_results[i] :
                        work_summaries === nothing ? trajectory_work(unwrapped[i], work_itp; species, mode = work_mode, power = save_power) :
                        (; summary = work_summaries[i], increments = nothing, powers = nothing) for i in 1:n
                ]
                sum_keys = collect(keys(w_list[1].summary))
                file["work_summary_keys"] = String.(sum_keys)
                for k in sum_keys
                    file["work/summary/$(string(k))"] = Float64[Float64(getproperty(w_list[i].summary, k)) for i in 1:n]
                end
                all_names = ("total", "conv", "hall", "gradpe", "resistive")
                n_comp = w_list[1].increments !== nothing ? size(w_list[1].increments, 2) :
                    (
                        w_list[1].powers !== nothing ? size(w_list[1].powers, 2) :
                        (hasproperty(w_list[1].summary, :res_eV) ? 5 : (hasproperty(w_list[1].summary, :gradpe_eV) ? 4 : 3))
                    )
                comp_names = String[all_names[j] for j in 1:n_comp]
                file["work_components"] = comp_names
                if work_mode == :steps
                    interval_offsets = Vector{Int32}(undef, n + 1)
                    interval_offsets[1] = 1
                    for i in 1:n
                        ni = max(0, length(unwrapped[i].t) - 1)
                        interval_offsets[i + 1] = interval_offsets[i] + Int32(ni)
                    end
                    total_int = Int(interval_offsets[end] - 1)
                    dW_mat = Matrix{state_eltype}(undef, n_comp, total_int)
                    for i in 1:n
                        inc = w_list[i].increments
                        i0 = Int(interval_offsets[i]) - 1
                        ni = size(inc, 1)
                        @inbounds for r in 1:ni, c in 1:n_comp
                            dW_mat[c, i0 + r] = state_eltype(inc[r, c])
                        end
                    end
                    file["work/interval_offsets"] = interval_offsets
                    file["work/dW_eV"] = dW_mat
                end
                if save_power
                    P_mat = Matrix{state_eltype}(undef, n_comp, total_pts)
                    for i in 1:n
                        pw = w_list[i].powers
                        p0 = Int(point_offsets[i]) - 1
                        ni = size(pw, 1)
                        @inbounds for r in 1:ni, c in 1:n_comp
                            P_mat[c, p0 + r] = state_eltype(pw[r, c])
                        end
                    end
                    file["work/power_eV_s"] = P_mat
                end
            end
            file["complete"] = true
        else
            file["format_version"] = work_mode == :none ? 1 : 2
            if work_mode != :none
                file["work_mode"] = String(work_mode)
                file["work_method"] = work_summaries === nothing ? "midpoint q E dot v dt on supplied intervals" :
                    "midpoint q E dot v dt on all accepted integration intervals (precomputed)"
                file["work_unit"] = "eV"
                file["power_unit"] = "eV s^-1"
                file["work_components"] = ["total", "conv", "hall"]
            end
            file["particle_ids"] = Int.(particle_ids)
            file["species"] = String(species)
            file["coordinate_system"] = String(coordinate_system)
            file["state_units"] = ["s", "m", "m", "m", "m s^-1", "m s^-1", "m s^-1"]
            file["weight_unit"] = "s^-1"
            for i in 1:n
                traj = unwrapped[i]
                state = Matrix{Float64}(undef, 7, length(traj.t))
                @inbounds for j in eachindex(traj.t)
                    state[1, j] = traj.t[j]
                    uj = traj.u[j]
                    state[2, j] = uj[1];state[3, j] = uj[2];state[4, j] = uj[3]
                    state[5, j] = uj[4];state[6, j] = uj[5];state[7, j] = uj[6]
                end
                pgrp = JLD2.Group(file, "p$(particle_ids[i])")
                pgrp["state"] = state
                pgrp["rate_weight_s1"] = Float64(rate_weights_s[i])
                source_density_weights_m3 === nothing || (pgrp["source_density_weight_m3"] = Float64(source_density_weights_m3[i]))
                cell_ids === nothing || (pgrp["cell_id"] = Int(cell_ids[i]))
                code = termination_codes === nothing ? (hasproperty(traj, :retcode) ? string(traj.retcode) : "unavailable") : string(termination_codes[i])
                pgrp["termination_code"] = code
                if work_mode != :none
                    w = work_results !== nothing ? work_results[i] :
                        work_summaries === nothing ? trajectory_work(traj, work_itp; species, mode = work_mode, power = save_power) :
                        (; summary = work_summaries[i], increments = nothing, powers = nothing)
                    wgrp = JLD2.Group(pgrp, "work")
                    sumgrp = JLD2.Group(wgrp, "summary")
                    for (key, value) in pairs(w.summary)
                        sumgrp[string(key)] = value
                    end
                    for (j, name) in enumerate(("total", "conv", "hall", "gradpe", "resistive"))
                        w.increments === nothing || j > size(w.increments, 2) || (wgrp["dW_$(name)_eV"] = w.increments[:, j])
                        w.powers === nothing || j > size(w.powers, 2) || (wgrp["power_$(name)_eV_s"] = w.powers[:, j])
                    end
                end
            end
            file["complete"] = true
        end
    end
    return abspath(path)
end

"""
    foreach_saved_trajectory(callback, directory_or_files; species="O2+", progress=nothing)

Read one trajectory at a time and call callback(record), where record contains
particle_id, rate_weight_s, trajectory (t/u), source_density_weight_m3, cell_id,
and termination_code, plus work (nothing for older files). Work contains summary,
steps and powers dictionaries. Supports both write_trajectory_batch format and the
original p<ID>/state legacy batches. A directory must have completion.toml with
complete=true and metadata.toml with matching n_particles/species. Explicit
file lists support partial-run inspection; completeness is then per new batch.
Legacy individual files have no completion marker, reported as legacy format.
The callback must consume the state immediately to keep memory bounded.
"""
function foreach_saved_trajectory(callback, source; species = "O2+", progress = nothing, include_work::Bool = true)
    haskey(TP.SpeciesDict, species) || throw(ArgumentError("Unknown species"))
    expected = nothing
    if source isa AbstractString && isdir(source)
        completion = TOML.parsefile(joinpath(source, "completion.toml"))
        get(completion, "complete", false) === true || throw(ArgumentError("Incomplete trajectory run"))
        meta = TOML.parsefile(joinpath(source, "metadata.toml"))
        meta["species"] == species || throw(ArgumentError("Saved species mismatch"))
        expected = Int(get(completion, "n_saved_trajectories", get(meta, "n_saved_trajectories", meta["n_particles"])))
        files = sort(filter(p -> occursin(r"^trajectories_\d+\.jld2$", basename(p)), readdir(source; join = true)))
    else
        files = source isa AbstractString ? [source] : collect(source)
    end
    isempty(files) && throw(ArgumentError("No trajectory batches"))
    seen = Set{Int}();count = 0
    for (batch, path) in enumerate(files)
        jldopen(path, "r") do file
            modern = haskey(file, "format_version")
            format_ver = 0
            work_mode_str = ""
            work_mode_sym = :none
            if modern
                format_ver = file["format_version"]
                format_ver in (1, 2, 3) || throw(ArgumentError("Unsupported trajectory format"))
                haskey(file, "complete") && file["complete"] === true || throw(ArgumentError("Incomplete batch: $path"))
                file["species"] == species || throw(ArgumentError("Saved species mismatch"))
                file["weight_unit"] == "s^-1" || throw(ArgumentError("Saved rate unit mismatch"))
                if include_work && (format_ver == 2 || (format_ver == 3 && haskey(file, "work_mode")))
                    file["work_unit"] == "eV" || throw(ArgumentError("Invalid work unit"))
                    work_mode_str = file["work_mode"]
                    work_mode_str in ("steps", "summary") || throw(ArgumentError("Invalid saved work mode"))
                    work_mode_sym = Symbol(work_mode_str)
                end
            end
            if modern && format_ver == 3
                ids = file["particle_ids"]
                point_offsets = file["point_offsets"]
                t_start_s = file["t_start_s"]
                dt_step_s = file["dt_step_s"]
                t_end_s = file["t_end_s"]
                times_fallback = haskey(file, "times_s") ? file["times_s"] : nothing
                states = file["states"]
                rates = file["rate_weights_s1"]
                dens_weights = haskey(file, "source_density_weights_m3") ? file["source_density_weights_m3"] : nothing
                cids = haskey(file, "cell_ids") ? file["cell_ids"] : nothing
                codes = file["termination_codes"]
                has_work = include_work && haskey(file, "work_mode")
                sum_keys = has_work ? String.(file["work_summary_keys"]) : String[]
                sum_vecs = has_work ? Dict(k => file["work/summary/$k"] for k in sum_keys) : Dict{String, Vector{Float64}}()
                comp_names = has_work ? String.(file["work_components"]) : String[]
                int_offsets = (has_work && work_mode_str == "steps") ? file["work/interval_offsets"] : nothing
                dW_mat = (has_work && work_mode_str == "steps") ? file["work/dW_eV"] : nothing
                P_mat = (has_work && haskey(file, "work/power_eV_s")) ? file["work/power_eV_s"] : nothing
                for (i, raw_pid) in enumerate(ids)
                    pid = Int(raw_pid)
                    pid > 0 && !(pid in seen) || throw(ArgumentError("Invalid or duplicate saved particle ID $pid"))
                    push!(seen, pid)
                    rate = Float64(rates[i])
                    isfinite(rate) && rate >= 0 || throw(ArgumentError("Invalid saved rate"))
                    p0 = Int(point_offsets[i]); p1 = Int(point_offsets[i + 1]) - 1
                    npts = p1 - p0 + 1
                    npts >= 1 || throw(ArgumentError("Empty trajectory"))
                    t_vec = if times_fallback !== nothing && !isfinite(dt_step_s[i])
                        Float64[Float64(times_fallback[j]) for j in p0:p1]
                    else
                        _reconstruct_packed_times(Float64(t_start_s[i]), Float64(dt_step_s[i]), Float64(t_end_s[i]), npts)
                    end
                    _is_strictly_increasing(t_vec) || throw(ArgumentError("Invalid saved time sequence"))
                    u_vec = Vector{SVector{6, Float64}}(undef, npts)
                    @inbounds for j in 1:npts
                        col = p0 + j - 1
                        s1 = Float64(states[1, col]);s2 = Float64(states[2, col]);s3 = Float64(states[3, col])
                        s4 = Float64(states[4, col]);s5 = Float64(states[5, col]);s6 = Float64(states[6, col])
                        (isfinite(s1)&&isfinite(s2)&&isfinite(s3)&&isfinite(s4)&&isfinite(s5)&&isfinite(s6)) ||
                            throw(ArgumentError("Trajectory has invalid Cartesian states"))
                        u_vec[j] = SVector{6, Float64}(s1, s2, s3, s4, s5, s6)
                    end
                    trajectory = (; t = t_vec, u = u_vec)
                    code = String(codes[i])
                    code in ("unavailable", "Success", "Terminated", "inner", "outer", "time_limit", "zero_rate") ||
                        throw(ArgumentError("Failed/unknown saved termination code: $code"))
                    work = nothing
                    if has_work
                        summary = Dict{String, Float64}(k => Float64(sum_vecs[k][i]) for k in sum_keys)
                        all(isfinite, values(summary)) || throw(ArgumentError("Nonfinite work summary"))
                        steps = Dict{String, Vector{Float64}}();powers = Dict{String, Vector{Float64}}()
                        if work_mode_str == "steps" && dW_mat !== nothing && int_offsets !== nothing
                            i0 = Int(int_offsets[i]); i1 = Int(int_offsets[i + 1]) - 1
                            nint = i1 - i0 + 1
                            nint == npts - 1 || throw(ArgumentError("Invalid work intervals"))
                            for (c, name) in enumerate(comp_names)
                                a = Float64[Float64(dW_mat[c, col]) for col in i0:i1]
                                all(isfinite, a) || throw(ArgumentError("Invalid work intervals"))
                                steps[name] = a
                            end
                        end
                        if P_mat !== nothing
                            for (c, name) in enumerate(comp_names)
                                a = Float64[Float64(P_mat[c, col]) for col in p0:p1]
                                all(isfinite, a) || throw(ArgumentError("Invalid power samples"))
                                powers[name] = a
                            end
                        end
                        work = (; summary, steps, powers, mode = work_mode_sym)
                    end
                    callback(
                        (;
                            particle_id = pid, rate_weight_s = rate, trajectory,
                            source_density_weight_m3 = dens_weights === nothing ? nothing : Float64(dens_weights[i]),
                            cell_id = cids === nothing ? nothing : Int(cids[i]),
                            termination_code = code, work, legacy_format = false,
                        )
                    )
                    count += 1
                end
            else
                ids = modern ? file["particle_ids"] : sort([parse(Int, k[2:end]) for k in keys(file) if occursin(r"^p\d+$", k)])
                for pid in ids
                    pid > 0 && !(pid in seen) || throw(ArgumentError("Invalid or duplicate saved particle ID $pid"))
                    push!(seen, pid)
                    prefix = "p$pid";state = file["$prefix/state"]
                    size(state, 1) == 7 || throw(ArgumentError("Expected 7×N saved state"))
                    ratekey = haskey(file, "$prefix/rate_weight_s1") ? "rate_weight_s1" : "weight_s1"
                    rate = Float64(file["$prefix/$ratekey"])
                    isfinite(rate) && rate >= 0 || throw(ArgumentError("Invalid saved rate"))
                    !isempty(state) && all(isfinite, state) || throw(ArgumentError("Trajectory has invalid times or Cartesian states"))
                    t_view = view(state, 1, :)
                    _is_strictly_increasing(t_view) || throw(ArgumentError("Invalid saved time sequence"))
                    u_vec = [@inbounds SVector{6, Float64}(state[2, j], state[3, j], state[4, j], state[5, j], state[6, j], state[7, j]) for j in axes(state, 2)]
                    trajectory = (; t = t_view, u = u_vec)
                    code = haskey(file, "$prefix/termination_code") ? file["$prefix/termination_code"] : "unavailable"
                    code in ("unavailable", "Success", "Terminated", "inner", "outer", "time_limit", "zero_rate") ||
                        throw(ArgumentError("Failed/unknown saved termination code: $code"))
                    work = nothing
                    if include_work && modern && format_ver == 2
                        sum_grp = file["$prefix/work/summary"]
                        summary = Dict(k => sum_grp[k] for k in keys(sum_grp))
                        all(isfinite, values(summary)) || throw(ArgumentError("Nonfinite work summary"))
                        steps = Dict{String, Vector{Float64}}();powers = Dict{String, Vector{Float64}}()
                        for name in ("total", "conv", "hall", "gradpe", "resistive")
                            key = "$prefix/work/dW_$(name)_eV"
                            if work_mode_str == "steps" && (name in ("total", "conv", "hall") || haskey(file, key))
                                a = file[key]
                                length(a) == length(trajectory.t) - 1 && all(isfinite, a) || throw(ArgumentError("Invalid work intervals"))
                                steps[name] = a
                            end
                            key = "$prefix/work/power_$(name)_eV_s"
                            if haskey(file, key)
                                a = file[key]
                                length(a) == length(trajectory.t) && all(isfinite, a) || throw(ArgumentError("Invalid power samples"))
                                powers[name] = a
                            end
                        end
                        work = (; summary, steps, powers, mode = work_mode_sym)
                    end
                    callback(
                        (;
                            particle_id = Int(pid), rate_weight_s = rate, trajectory,
                            source_density_weight_m3 = haskey(file, "$prefix/source_density_weight_m3") ? file["$prefix/source_density_weight_m3"] : nothing,
                            cell_id = haskey(file, "$prefix/cell_id") ? file["$prefix/cell_id"] : nothing,
                            termination_code = code, work, legacy_format = !modern,
                        )
                    )
                    count += 1
                end
            end
        end
        progress === nothing || progress((; batch, batches = length(files), particles = count, path))
    end
    expected === nothing || count == expected || throw(ArgumentError("Saved particle count mismatch: $count != $expected"))
    return count
end

"""
    forward_psd_saved(directory_or_files; detector_m, side_m, vlim, vgrid,
        species="O2+", option="3D", velocity_unit=:m_s,
        coordinate_system="unspecified", storage=:dense, progress=nothing,
        energy_edges_eV=nothing)

Streaming equivalent of forward_psd. Reads source rates directly from each
saved trajectory and uses the same accumulator, cube clipping, interpolation,
velocity-bin splitting, and SI normalization. No MHD input or tracing required.
With `energy_edges_eV`, also returns `omni_def` from the same streamed residence
segments, independent of the Cartesian velocity-bin limits.
"""
function forward_psd_saved(
        source; detector_m, side_m, vlim, vgrid, species = "O2+",
        option = "3D", velocity_unit = :m_s, coordinate_system = "unspecified", storage = :dense, progress = nothing, energy_edges_eV = nothing
    )
    acc = ForwardPSDAccumulator(; detector_m, side_m, vlim, vgrid, species, velocity_unit, coordinate_system, energy_edges_eV)
    foreach_saved_trajectory(source; species, progress, include_work = false) do record
        accumulate_forward_psd!(acc, record.trajectory; rate_weight_s = record.rate_weight_s, particle_id = record.particle_id)
        acc.retcodes[end] = record.termination_code
    end
    return finish_forward_psd(acc; option, storage)
end
