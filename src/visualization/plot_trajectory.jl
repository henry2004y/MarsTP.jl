# Plot forward/backward trajectory files in 2D or 3D using PyPlot.

function _folder_metadata(folder::AbstractString)
    toml_path = joinpath(folder, "metadata.toml")
    if isfile(toml_path)
        return TOML.parsefile(toml_path)
    end
    json_path = joinpath(folder, "metadata.json")
    if isfile(json_path)
        return Dict{String, Any}(JSON.parsefile(json_path))
    end
    return Dict{String, Any}()
end

function _subsample_indices(n::Int, max_points::Int)
    m = min(max_points, n)
    m >= n && return collect(1:n)
    # Match floor/int linspace indexing on 0:(n-1), converted to 1-based indices
    raw = [floor(Int, x) + 1 for x in range(0.0, n - 1, length = m)]
    return unique(raw)
end

function _extract_state_time_xyz(state::AbstractMatrix)
    if size(state, 1) == 7 && size(state, 2) != 7
        # Native Julia JLD2 7 x N layout (row 1 = t, rows 2:4 = x, y, z)
        return collect(Float64, @view state[1, :]), Matrix{Float64}((@view state[2:4, :])')
    elseif size(state, 2) == 7 && size(state, 1) != 7
        # N x 7 layout
        return collect(Float64, @view state[:, 1]), Matrix{Float64}(@view state[:, 2:4])
    elseif size(state, 1) == 7 && size(state, 2) == 7
        # Ambiguous 7x7: check whether row 1 or col 1 is strictly monotonic time
        dt_row = diff(vec(@view state[1, :]))
        if all(>(0), dt_row) || all(<(0), dt_row)
            return collect(Float64, @view state[1, :]), Matrix{Float64}((@view state[2:4, :])')
        else
            return collect(Float64, @view state[:, 1]), Matrix{Float64}(@view state[:, 2:4])
        end
    else
        throw(ArgumentError("Expected 7xN or Nx7 SI state matrix, got $(size(state))"))
    end
end

function _extract_subsampled_state_time_xyz(state::AbstractMatrix, max_points::Int)
    if size(state, 1) == 7 && size(state, 2) != 7
        ix = _subsample_indices(size(state, 2), max_points)
        m = length(ix)
        t = Vector{Float64}(undef, m)
        xyz = Matrix{Float64}(undef, m, 3)
        @inbounds for (r, c) in enumerate(ix)
            t[r] = state[1, c]
            xyz[r, 1] = state[2, c]
            xyz[r, 2] = state[3, c]
            xyz[r, 3] = state[4, c]
        end
        return t, xyz
    else
        t_all, xyz_all = _extract_state_time_xyz(state)
        ix = _subsample_indices(length(t_all), max_points)
        return t_all[ix], xyz_all[ix, :]
    end
end

"""
    load_trajectories(source; species=nothing, assume_species=nothing, count=5000,
                      max_points=300, seed=20260906, position_unit=nothing,
                      rm_m=Rm, particle_ids=nothing)

Reservoir-sample trajectories from `.jld2` batches or `.jsonl` files, returning
a `Vector{Dict{String, Any}}` with positions scaled to planetary radii (`Rm`).
"""
function load_trajectories(
        source;
        species = nothing,
        assume_species = nothing,
        count::Integer = 5000,
        max_points::Integer = 300,
        seed::Integer = 20260906,
        position_unit = nothing,
        rm_m::Real = Rm,
        particle_ids = nothing,
    )
    if count < 1 || max_points < 2 || !isfinite(rm_m) || rm_m <= 0
        throw(ArgumentError("count >= 1, max_points >= 2, rm_m > 0 required"))
    end
    src_path = String(source)
    wanted = if particle_ids !== nothing
        Set{Int}(Int.(particle_ids))
    elseif isdir(src_path) && isfile(joinpath(src_path, "plot_sample_ids.toml"))
        Set{Int}(Int.(TOML.parsefile(joinpath(src_path, "plot_sample_ids.toml"))["particle_ids"]))
    else
        nothing
    end
    files = if isdir(src_path)
        jld_files = sort(filter(p -> occursin(r"^trajectories_.*\.jld2$", basename(p)), readdir(src_path; join = true)))
        if !isempty(jld_files)
            jld_files
        elseif isfile(joinpath(src_path, "trajectories.jsonl"))
            [joinpath(src_path, "trajectories.jsonl")]
        else
            String[]
        end
    elseif isfile(src_path)
        [src_path]
    else
        String[]
    end
    isempty(files) && throw(ArgumentError("No trajectory files found (PSD-only files contain no paths)"))

    rng = MersenneTwister(Int(seed))
    selected = Dict{String, Any}[]
    eligible = 0

    function consider!(item::Dict{String, Any})
        if wanted !== nothing && !(item["id"] in wanted)
            return
        end
        raw_sp = get(item, "species", nothing)
        known = (raw_sp === nothing || isempty(String(raw_sp))) ? assume_species : String(raw_sp)
        if species !== nothing && known === nothing
            throw(ArgumentError("Species metadata missing; supply assume_species explicitly"))
        end
        if species !== nothing && known != String(species)
            return
        end
        item["species"] = known === nothing ? "unspecified" : String(known)
        eligible += 1
        if length(selected) < count
            push!(selected, item)
        else
            j = rand(rng, 1:eligible)
            if j <= count
                selected[j] = item
            end
        end
        return nothing
    end

    for path in files
        meta = _folder_metadata(dirname(abspath(path)))
        ext = lowercase(splitext(path)[2])
        if ext in (".jld2", ".h5", ".hdf5")
            if position_unit !== nothing && String(position_unit) != "m"
                throw(ArgumentError("MarsTP state format has fixed SI position units (m)"))
            end
            jldopen(path, "r") do f
                if haskey(f, "format_version") && (!haskey(f, "complete") || f["complete"] !== true)
                    throw(ArgumentError("Incomplete trajectory batch: $path"))
                end
                kind = haskey(f, "species") ? String(f["species"]) : get(meta, "species", nothing)
                frame = haskey(f, "coordinate_system") ? String(f["coordinate_system"]) : String(get(meta, "coordinate_system", "unspecified"))
                found = false
                if haskey(f, "format_version") && f["format_version"] == 3
                    ids = f["particle_ids"]
                    point_offsets = f["point_offsets"]
                    t_start_s = f["t_start_s"]
                    dt_step_s = f["dt_step_s"]
                    t_end_s = f["t_end_s"]
                    times_fallback = haskey(f, "times_s") ? f["times_s"] : nothing
                    states = f["states"]
                    known = (kind === nothing || isempty(String(kind))) ? assume_species : String(kind)
                    if species !== nothing && known === nothing
                        throw(ArgumentError("Species metadata missing; supply assume_species explicitly"))
                    end
                    found = !isempty(ids)
                    if !(species !== nothing && known != String(species))
                        for (i, raw_pid) in enumerate(ids)
                            pid = Int(raw_pid)
                            if wanted !== nothing && !(pid in wanted)
                                continue
                            end
                            p0 = Int(point_offsets[i])
                            p1 = Int(point_offsets[i + 1]) - 1
                            npts = p1 - p0 + 1
                            npts < 2 && continue
                            eligible += 1
                            take_idx = if length(selected) < count
                                length(selected) + 1
                            else
                                j = rand(rng, 1:eligible)
                                j <= count ? j : 0
                            end
                            if take_idx > 0
                                ix = _subsample_indices(npts, Int(max_points))
                                m = length(ix)
                                t_sub = Vector{Float64}(undef, m)
                                xyz_sub = Matrix{Float64}(undef, m, 3)
                                t0 = Float64(t_start_s[i])
                                dt = Float64(dt_step_s[i])
                                tend = Float64(t_end_s[i])
                                @inbounds for (r, k) in enumerate(ix)
                                    col = p0 + k - 1
                                    t_sub[r] = if times_fallback !== nothing && !isfinite(dt)
                                        Float64(times_fallback[col])
                                    elseif k == npts
                                        tend
                                    elseif k == 1
                                        t0
                                    else
                                        t0 + (k - 1) * dt
                                    end
                                    xyz_sub[r, 1] = Float64(states[1, col])
                                    xyz_sub[r, 2] = Float64(states[2, col])
                                    xyz_sub[r, 3] = Float64(states[3, col])
                                end
                                item = Dict{String, Any}(
                                    "path" => path,
                                    "id" => pid,
                                    "species" => known === nothing ? "unspecified" : String(known),
                                    "coordinate_system" => frame,
                                    "unit" => "m",
                                    "times_s" => t_sub,
                                    "points" => xyz_sub,
                                )
                                if take_idx > length(selected)
                                    push!(selected, item)
                                else
                                    selected[take_idx] = item
                                end
                            end
                        end
                    end
                else
                    for key in keys(f)
                        if startswith(key, "p") && all(isdigit, SubString(key, 2)) && haskey(f, "$key/state")
                            found = true
                            pid = parse(Int, SubString(key, 2))
                            if wanted !== nothing && !(pid in wanted)
                                continue
                            end
                            rec_species = haskey(f, "$key/species") ? String(f["$key/species"]) : kind
                            known = (rec_species === nothing || isempty(String(rec_species))) ? assume_species : String(rec_species)
                            if species !== nothing && known === nothing
                                throw(ArgumentError("Species metadata missing; supply assume_species explicitly"))
                            end
                            if species !== nothing && known != String(species)
                                continue
                            end
                            state = f["$key/state"]
                            if !(state isa AbstractMatrix) || !(size(state, 1) == 7 || size(state, 2) == 7)
                                throw(ArgumentError("Expected 7xN or Nx7 SI state: $path:$key"))
                            end
                            npts = size(state, 1) == 7 && size(state, 2) != 7 ? size(state, 2) : size(state, 1)
                            npts < 2 && continue
                            eligible += 1
                            if length(selected) < count
                                t_sub, xyz_sub = _extract_subsampled_state_time_xyz(state, Int(max_points))
                                push!(
                                    selected, Dict{String, Any}(
                                        "path" => path,
                                        "id" => pid,
                                        "species" => known === nothing ? "unspecified" : String(known),
                                        "coordinate_system" => frame,
                                        "unit" => "m",
                                        "times_s" => t_sub,
                                        "points" => xyz_sub,
                                    )
                                )
                            else
                                j = rand(rng, 1:eligible)
                                if j <= count
                                    t_sub, xyz_sub = _extract_subsampled_state_time_xyz(state, Int(max_points))
                                    selected[j] = Dict{String, Any}(
                                        "path" => path,
                                        "id" => pid,
                                        "species" => known === nothing ? "unspecified" : String(known),
                                        "coordinate_system" => frame,
                                        "unit" => "m",
                                        "times_s" => t_sub,
                                        "points" => xyz_sub,
                                    )
                                end
                            end
                        end
                    end
                end
                found || throw(ArgumentError("No saved trajectories in $path; PSD arrays cannot reconstruct paths"))
            end
        elseif ext == ".jsonl"
            for line in eachline(path)
                isempty(strip(line)) && continue
                r = Dict{String, Any}(JSON.parse(line))
                unit_raw = position_unit !== nothing ? position_unit : get(r, "position_unit", get(meta, "position_unit", nothing))
                unit = unit_raw === nothing ? "" : String(unit_raw)
                unit in ("m", "km", "Rm") || throw(ArgumentError("JSONL needs explicit position_unit: m, km or Rm"))
                pts_raw = r["points"]
                times_raw = Float64.(r["times_s"])
                length(pts_raw) == length(times_raw) && length(times_raw) >= 2 && all(p -> length(p) == 3, pts_raw) ||
                    throw(ArgumentError("Expected N x 3 points and matching times_s"))
                pts = Matrix{Float64}(undef, length(times_raw), 3)
                for (i, row) in enumerate(pts_raw)
                    pts[i, 1] = Float64(row[1])
                    pts[i, 2] = Float64(row[2])
                    pts[i, 3] = Float64(row[3])
                end
                ix = _subsample_indices(length(times_raw), Int(max_points))
                r["points"] = pts[ix, :]
                r["times_s"] = times_raw[ix]
                consider!(
                    Dict{String, Any}(
                        "path" => path,
                        "id" => Int(r["id"]),
                        "species" => get(r, "species", get(meta, "species", nothing)),
                        "coordinate_system" => String(get(r, "coordinate_system", get(meta, "coordinate_system", "unspecified"))),
                        "unit" => unit,
                        "raw" => r,
                    )
                )
            end
        else
            throw(ArgumentError("Unsupported trajectory format: $ext"))
        end
    end

    isempty(selected) && throw(ArgumentError("No trajectories match the requested species"))

    result = Vector{Dict{String, Any}}(undef, length(selected))
    for (idx, item) in enumerate(selected)
        times, xyz = if haskey(item, "times_s")
            item["times_s"]::Vector{Float64}, item["points"]::Matrix{Float64}
        else
            r = item["raw"]
            r["times_s"]::Vector{Float64}, r["points"]::Matrix{Float64}
        end
        all(isfinite, xyz) && all(isfinite, times) || throw(ArgumentError("Nonfinite trajectory"))
        _is_strictly_monotonic(times) || throw(ArgumentError("Trajectory times must be strictly monotonic"))
        unit = String(item["unit"])
        scale = unit == "m" ? (1.0 / Float64(rm_m)) : (unit == "km" ? (1000.0 / Float64(rm_m)) : 1.0)
        result[idx] = Dict{String, Any}(
            "id" => Int(item["id"]),
            "source" => String(item["path"]),
            "points" => scale == 1.0 ? xyz : xyz .* scale,
            "times_s" => times,
            "species" => String(item["species"]),
            "coordinate_system" => String(item["coordinate_system"]),
        )
    end
    return result
end

const _PLANE_AXES = Dict(
    "XY" => (1, 2),
    "XZ" => (1, 3),
    "YZ" => (2, 3),
)

"""
    plot_trajectory(source; planes=("XZ", "XY", "YZ"), limits=(-4.2, 4.2),
                    boundaries=true, x_slice=nothing, output=nothing, load_options...)

Plot sampled trajectories in 2D (`"XZ"`, `"XY"`, `"YZ"`) or `"3D"` using PyPlot.
Returns `(fig, axes, records)`.
"""
function plot_trajectory(
        source;
        planes = ("XZ", "XY", "YZ"),
        limits = (-4.2, 4.2),
        boundaries::Bool = true,
        x_slice = nothing,
        output = nothing,
        load_options...,
    )
    records = load_trajectories(source; load_options...)
    plane_list = planes isa AbstractString ? [uppercase(String(planes))] : [uppercase(String(p)) for p in planes]
    if isempty(plane_list) || any(p -> !(p in ("XY", "XZ", "YZ", "3D")), plane_list)
        throw(ArgumentError("planes must be chosen from XY, XZ, YZ, 3D"))
    end
    if length(limits) != 2 || !all(isfinite, limits) || limits[1] >= limits[2]
        throw(ArgumentError("limits must be finite and increasing"))
    end
    lims = (Float64(limits[1]), Float64(limits[2]))
    frames = unique(r["coordinate_system"] for r in records)
    length(frames) > 1 && throw(ArgumentError("Cannot overlay trajectories with different coordinate frames"))

    fig = PyPlot.plt.figure(figsize = (5 * length(plane_list), 5), layout = "constrained")
    axes = Any[]
    cmap = PyPlot.plt.get_cmap("turbo")
    nrec = length(records)
    colors = [cmap(nrec == 1 ? 0.5 : 0.05 + 0.9 * (i - 1) / (nrec - 1)) for i in 1:nrec]

    for (k, plane) in enumerate(plane_list)
        ax = plane == "3D" ?
            fig.add_subplot(1, length(plane_list), k; projection = "3d") :
            fig.add_subplot(1, length(plane_list), k)
        push!(axes, ax)
        plot_mars_context(ax, plane; boundaries = boundaries, xmin = lims[1], x_slice = x_slice)
        if plane == "3D"
            for (r, color) in zip(records, colors)
                p = r["points"]::Matrix{Float64}
                ax.plot(p[:, 1], p[:, 2], p[:, 3]; color = color, lw = 0.5, alpha = 0.4)
            end
        else
            i, j = _PLANE_AXES[plane]
            segments = Matrix{Float64}[
                let p = r["points"]::Matrix{Float64}
                    seg = Matrix{Float64}(undef, size(p, 1), 2)
                    @inbounds for k in 1:size(p, 1)
                        seg[k, 1] = p[k, i]
                        seg[k, 2] = p[k, j]
                    end
                    seg
                end for r in records
            ]
            lc = PyPlot.matplotlib.collections.LineCollection(
                segments;
                colors = colors,
                linewidths = 0.5,
                alpha = 0.4,
                zorder = 2,
            )
            ax.add_collection(lc)
        end
        if plane == "3D"
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_zlim(lims)
            ax.set_xlabel("X (Rm)")
            ax.set_ylabel("Y (Rm)")
            ax.set_zlabel("Z (Rm)")
            ax.set_box_aspect((1, 1, 1))
        else
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("$(plane[1]) (Rm)")
            ax.set_ylabel("$(plane[2]) (Rm)")
            ax.set_aspect("equal")
        end
        title_str = (plane == "YZ" && x_slice !== nothing) ?
            @sprintf("%s (boundaries: X=%g Rm)", plane, Float64(x_slice)) : plane
        ax.set_title(title_str)
    end
    species_str = join(sort(unique(r["species"] for r in records)), ", ")
    fig.suptitle("$species_str; $(only(frames))")
    if output !== nothing
        out_path = String(output)
        lowercase(splitext(out_path)[2]) == ".png" || throw(ArgumentError("Output must be PNG"))
        mkpath(dirname(abspath(out_path)))
        fig.savefig(out_path; dpi = 180)
    end
    return fig, axes, records
end
