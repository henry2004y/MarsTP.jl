# Canonical analysis: all three probes use the same MarsTP accumulator as forward_psd.
# julia --project=. analyze_saved_probes.jl ORIGINAL_RUN NEW_OUTPUT
using MarsTP, JLD2, TOML, LinearAlgebra

function analyze_saved_probes(run, out)
    ispath(out) && error("Choose a new output directory: $out")
    meta = TOML.parsefile(joinpath(run, "metadata.toml"))
    names = ("probe_1_0_2", "probe_0_0_2", "probe_m1p5_0_1")
    centers = ([1.0, 0.0, 2.0], [0.0, 0.0, 2.0], [-1.5, 0.0, 1.0])
    accs = [
        ForwardPSDAccumulator(;
            detector_m = c * meta["Rm_m"], side_m = meta["cube_side_Rm"] * meta["Rm_m"],
            vlim = (-500.0, 500.0), vgrid = 200, velocity_unit = :km_s, species = meta["species"],
            coordinate_system = meta["coordinate_system"]
        ) for c in centers
    ]
    mkpath(out)
    handles = IOStream[];counts = zeros(Int, 3);currents = [zeros(3) for _ in 1:3]
    vmin = [fill(Inf, 3) for _ in 1:3];vmax = [fill(-Inf, 3) for _ in 1:3]
    for (i, name) in enumerate(names)
        mkpath(joinpath(out, name))
        io = open(joinpath(out, name, "probe_residence.csv"), "w");push!(handles, io)
        println(io, "particle_id,rate_weight_s1,t0_s,t1_s,x0_m,y0_m,z0_m,x1_m,y1_m,z1_m,vx0_ms,vy0_ms,vz0_ms,vx1_ms,vy1_ms,vz1_ms,entry_face,exit_face")
    end
    observers = ntuple(3) do i
        io = handles[i]
        side3 = accs[i].side_m^3
        s -> begin
            counts[i] += 1
            currents[i] .+= s.rate_weight_s * (s.t1_s - s.t0_s) * (s.v0_ms + s.v1_ms) / 2 / side3
            vmin[i] .= min.(vmin[i], s.v0_ms, s.v1_ms);vmax[i] .= max.(vmax[i], s.v0_ms, s.v1_ms)
            println(io, join((s.particle_id, s.rate_weight_s, s.t0_s, s.t1_s, s.x0_m..., s.x1_m..., s.v0_ms..., s.v1_ms..., s.entry_face, s.exit_face), ','))
        end
    end
    start = time()
    try
        foreach_saved_trajectory(
            run; species = meta["species"], include_work = false, progress = p -> begin
                if p.batch % 25 == 0
                    println("$(p.batch)/$(p.batches) batches, $(p.particles) particles, segments=$counts, $(round(time() - start; digits = 1)) s")
                    flush(stdout)
                end
            end
        ) do record
            for i in 1:3
                accumulate_forward_psd!(
                    accs[i], record.trajectory; rate_weight_s = record.rate_weight_s,
                    particle_id = record.particle_id, segment_observer = observers[i]
                )
                accs[i].retcodes[end] = record.termination_code
            end
        end
    finally
        foreach(close, handles)
    end
    for (i, name) in enumerate(names)
        acc = accs[i];r = finish_forward_psd(acc; storage = :sparse)
        bins = sort(collect(keys(r.psd)));dv = 5000.0
        xyz = isempty(bins) ? zeros(Int, 3, 0) : reduce(hcat, collect.(bins)) .- 1
        values = [r.psd[b] for b in bins]
        xy = finish_forward_psd(acc; option = :xy).psd
        xz = finish_forward_psd(acc; option = :xz).psd
        for density in (sum(values) * dv^3, sum(xy) * dv^2, sum(xz) * dv^2)
            isapprox(density, r.density_in_range_m3; rtol = 1.0e-11) || error("Projection normalization mismatch")
        end
        r.density_outside_vlim_m3 == 0 || error("Probe velocities outside ±500 km/s")
        dwell = acc.rates .* acc.residence;hit = findall(>(0), dwell)
        neff = isempty(hit) ? 0.0 : sum(dwell)^2 / sum(abs2, dwell)
        folder = joinpath(out, name)
        jldopen(joinpath(folder, "library_psd.jld2"), "w") do f
            f["indices_xyz"] = xyz;f["f3d_s3_m6"] = values
            f["fxy_s2_m5"] = xy;f["fxz_s2_m5"] = xz
            f["number_per_nonzero_bin"] = [acc.occupancy[b] for b in bins]
            f["unique_particles_per_nonzero_bin"] = [acc.unique[b] for b in bins]
            f["effective_particles_per_nonzero_bin"] = [acc.occupancy[b]^2 / acc.sumsq[b] for b in bins]
            for (k, axis) in enumerate(("vx", "vy", "vz"))
                f["$(axis)_edges_ms"] = acc.edges[k]
            end
            f["shape_xyz"] = [200, 200, 200];f["dv_ms"] = dv
            f["particle_ids"] = acc.particle_ids[hit];f["rate_weights_s"] = acc.rates[hit]
            f["residence_s"] = acc.residence[hit]
            f["complete"] = true
        end
        info = copy(meta)
        merge!(
            info, Dict(
                "detector_Rm" => centers[i], "source_run_directory" => abspath(run),
                "number_density_m3" => r.density_total_m3, "number_density_cm3" => r.density_total_m3 / 1.0e6,
                "density_outside_vlim_m3" => r.density_outside_vlim_m3, "unique_probe_particles" => length(hit),
                "probe_effective_particles" => neff, "residence_segments" => counts[i], "nonzero_3d_bins" => length(bins),
                "velocity_min_kms" => vmin[i] / 1.0e3, "velocity_max_kms" => vmax[i] / 1.0e3,
                "volume_averaged_number_flux_vector_m2_s" => currents[i],
                "estimator" => "MarsTP.ForwardPSDAccumulator, shared by forward_psd and forward_psd_saved",
                "interpolation" => "piecewise_linear_saved_endpoints", "velocity_bin_width_kms" => 5.0,
                "velocity_edges_kms" => [-500.0, 500.0], "plot_limit_kms" => 300.0, "grid_shape" => [200, 200, 200]
            )
        )
        open(joinpath(folder, "library_summary.toml"), "w") do io
            TOML.print(io, info)
        end
        println("$name: density=$(r.density_total_m3 / 1.0e6) cm^-3; hits=$(length(hit)); Neff=$neff")
    end
    open(joinpath(out, "analysis_complete.toml"), "w") do io
        TOML.print(io, Dict("complete" => true, "elapsed_s" => time() - start, "particles" => length(first(accs).rates)))
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    analyze_saved_probes(ARGS[1], ARGS[2])
end
