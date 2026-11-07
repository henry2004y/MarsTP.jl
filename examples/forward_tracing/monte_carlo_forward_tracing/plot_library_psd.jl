# Plot/export PSD computed by MarsTP analyze_saved_probes.jl using PyPlot.
using MarsTP
using JLD2
using JSON
using TOML
using Printf
using PyPlot

include(joinpath(dirname(@__DIR__), "analyze_monte_carlo.jl"))

function plot_one(folder::AbstractString)
    meta = TOML.parsefile(joinpath(folder, "library_summary.toml"))
    data = jldopen(joinpath(folder, "library_psd.jld2"), "r") do f
        f["complete"] === true || error("Incomplete library_psd.jld2 in $folder")
        (;
            fxy = Matrix{Float64}(f["fxy_s2_m5"]),
            fxz = Matrix{Float64}(f["fxz_s2_m5"]),
            edge = Vector{Float64}(f["vx_edges_ms"]),
            dv = Float64(f["dv_ms"]),
        )
    end
    edges = (data.edge, data.edge, data.edge)
    draw_psd_panels(
        joinpath(folder, "probe_psd_projections"),
        edges, data.fxy, data.fxz,
        "Reduced PSD (s^2 m^-5)",
        "Full velocity integration; $(data.dv / 1000.0) km/s bins; maximum flight age $(meta["max_flight_time_s"]) s",
        meta["unique_probe_particles"], meta["probe_effective_particles"], meta,
    )
    @printf("%s: %.9g cm^-3; library PSD plotted\n", basename(folder), Float64(meta["number_density_m3"]) / 1.0e6)
end

function main(args = ARGS)
    isempty(args) && error("Usage: julia --project=. plot_library_psd.jl LIBRARY_OUTPUT_ROOT")
    root = abspath(args[1])
    if isfile(joinpath(root, "library_psd.jld2"))
        plot_one(root)
    else
        for name in ("probe_1_0_2", "probe_0_0_2", "probe_m1p5_0_1")
            sub = joinpath(root, name)
            isdir(sub) && plot_one(sub)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

