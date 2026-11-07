# Re-bin or plot finite-volume O2+ Monte Carlo residence PSD in SI units using MarsTP and PyPlot.
using MarsTP
using JLD2
using JSON
using TOML
using Printf
using PyPlot

function draw_psd_panels(path_stem, edges, a, b, label, subtitle, hits, neff, meta)
    pos_a = a[a .> 0.0]
    pos_b = b[b .> 0.0]
    positive = vcat(pos_a, pos_b)
    norm = if !isempty(positive)
        vmin, vmax = minimum(positive), maximum(positive)
        vmax <= vmin && ((vmin, vmax) = (vmin / 2.0, vmax * 2.0))
        PyPlot.matplotlib.colors.LogNorm(vmin = vmin, vmax = vmax)
    else
        nothing
    end

    fig, axes = PyPlot.plt.subplots(1, 2; figsize = (10.4, 4.8), layout = "constrained")
    im = nothing
    for (ax, matrix, ydim, panel) in zip(axes, (a, b), (2, 3), ("a", "b"))
        masked = copy(matrix')
        masked[masked .<= 0.0] .= NaN
        im = ax.pcolormesh(edges[1] ./ 1.0e3, edges[ydim] ./ 1.0e3, masked; cmap = "viridis", norm = norm, rasterized = true)
        ax.set(xlabel = "v_x (km/s)", ylabel = ydim == 2 ? "v_y (km/s)" : "v_z (km/s)", aspect = "equal")
        ax.set_title("($panel) " * (ydim == 2 ? "Vx-Vy" : "Vx-Vz"); loc = "left")
        ax.axhline(0.0; lw = 0.5, c = "gray", alpha = 0.4)
        ax.axvline(0.0; lw = 0.5, c = "gray", alpha = 0.4)
        if haskey(meta, "plot_limit_kms") && meta["plot_limit_kms"] !== nothing
            lim = Float64(meta["plot_limit_kms"])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        end
    end
    !isempty(positive) && fig.colorbar(im; ax = collect(axes), label = label, shrink = 0.78)
    probe = join(meta["detector_Rm"], ", ")
    fig.suptitle("O2+, probe ($probe) R_M, cube side $(meta["cube_side_Rm"]) R_M\n$subtitle")
    fig.savefig(path_stem * ".png"; dpi = 190, bbox_inches = "tight", pad_inches = 0.12)
    PyPlot.plt.close(fig)
end

function main(args = ARGS)
    isempty(args) && error("Usage: julia --project=. analyze_monte_carlo.jl RUN_DIR [dv_kms] [vmax_kms]")
    run_dir = abspath(args[1])
    dv_kms = length(args) >= 2 ? parse(Float64, args[2]) : 10.0
    vmax_kms = length(args) >= 3 ? parse(Float64, args[3]) : 200.0

    meta = TOML.parsefile(joinpath(run_dir, "metadata.toml"))
    detector_m = Float64.(meta["detector_Rm"]) .* Float64(meta["Rm_m"])
    side_m = Float64(meta["cube_side_Rm"]) * Float64(meta["Rm_m"])
    vgrid = round(Int, 2.0 * vmax_kms / dv_kms)

    res = forward_psd_saved(
        run_dir;
        detector_m = detector_m,
        side_m = side_m,
        vlim = vmax_kms,
        vgrid = vgrid,
        velocity_unit = :km_s,
    )
    edges = res.all_velocity_edges_m_s
    dv = step(edges[1])
    f3d = res.psd
    fxy = dropdims(sum(f3d; dims = 3); dims = 3) .* dv
    fxz = dropdims(sum(f3d; dims = 2); dims = 2) .* dv
    hits = count(>(0.0), res.residence_s)
    dwell = res.rate_weights_s .* res.residence_s
    neff = sum(dwell) > 0.0 ? sum(dwell)^2 / sum(abs2, dwell) : 0.0

    draw_psd_panels(
        joinpath(run_dir, "probe_psd_projections"),
        edges, fxy, fxz, "Reduced PSD (s^2 m^-5)",
        "Velocity-integrated projections; $(dv_kms) km/s bins",
        hits, neff, meta,
    )
    println("Saved PSD projections in ", run_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

