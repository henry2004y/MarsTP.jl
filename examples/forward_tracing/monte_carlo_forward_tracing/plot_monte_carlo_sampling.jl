# Three-panel 2D O2+ importance-sampling illustration using MarsTP and PyPlot.
using MarsTP
using LinearAlgebra
using Random
using Printf
using PyPlot

function sample_demo(count = 100_000, seed = 20260907)
    count >= 2 || throw(ArgumentError("At least two Maxwellian samples are required"))
    n = 5.0e6
    mass = MarsTP.TP.SpeciesDict["O2+"].m
    qe = MarsTP.TP.eV
    temperature_ev = 10.0
    bulk = [-10.0e3, 0.0]
    factor = 4.0
    area = 1.0
    sigma = sqrt(qe * temperature_ev / mass)
    rng = MersenneTwister(seed)
    vel = Matrix{Float64}(undef, count, 2)
    for i in 1:count
        vel[i, 1] = bulk[1] + sigma * sqrt(factor) * randn(rng)
        vel[i, 2] = bulk[2] + sigma * sqrt(factor) * randn(rng)
    end
    d2 = (vel[:, 1] .- bulk[1]).^2 .+ (vel[:, 2] .- bulk[2]).^2
    importance = factor .* exp.(-d2 ./ (2.0 * sigma^2) .* (1.0 - 1.0 / factor))
    density_weight = n .* importance ./ sum(importance)
    rate_weight = area * norm(bulk) .* density_weight
    analytic_rate = n * area * norm(bulk)
    neff = sum(importance)^2 / sum(abs2, importance)
    return (; n, sigma, bulk, vel, density_weight, rate_weight, analytic_rate, neff, count)
end

function main(args = ARGS)
    out_png = length(args) >= 1 ? abspath(args[1]) : joinpath(@__DIR__, "monte_carlo_sampling.png")
    d = sample_demo()
    fig, axes = PyPlot.plt.subplots(1, 3; figsize = (15, 5.1), layout = "constrained")
    radius = ceil(maximum(abs, d.vel .- d.bulk') / 5000.0) * 5000.0
    xlim = (d.bulk[1] - radius, d.bulk[1] + radius)
    ylim = (-radius, radius)
    x = collect(range(xlim[1], xlim[2], length = 361))
    y = collect(range(ylim[1], ylim[2], length = 361))
    f2 = [d.n / (2.0 * pi * d.sigma^2) * exp(-((xi - d.bulk[1])^2 + yj^2) / (2.0 * d.sigma^2)) for yj in y, xi in x]

    im0 = axes[1].pcolormesh(x ./ 1000.0, y ./ 1000.0, f2; cmap = "turbo", shading = "auto")
    axes[1].set_title("(a) Analytic Maxwellian"; loc = "left")
    fig.colorbar(im0; ax = axes[1], label = "f_xy (s^2 m^-5)", shrink = 0.79)

    for (ax, weights, title, label) in zip(
        axes[2:3],
        (d.density_weight, d.rate_weight),
        ("(b) MC density weight", "(c) MC flux (rate) weight"),
        ("W_{n,i} (m^-3)", "Q_i (s^-1)"),
    )
        order = sortperm(weights)
        im = ax.scatter(d.vel[order, 1] ./ 1000.0, d.vel[order, 2] ./ 1000.0;
            c = weights[order], s = 1, cmap = "turbo", linewidths = 0, rasterized = true)
        ax.set_title(title; loc = "left")
        fig.colorbar(im; ax = ax, label = label, shrink = 0.79)
    end
    for ax in axes
        ax.set(xlim = xlim ./ 1000.0, ylim = ylim ./ 1000.0, xlabel = "v_x (km/s)", ylabel = "v_y (km/s)", aspect = "equal")
    end
    fig.suptitle("O2+: n = 5 cm^-3, U_x = -10 km/s, T = 10 eV, v_z = 0\n$(d.count) Maxwellian samples; T_s = 4T")
    mkpath(dirname(out_png))
    fig.savefig(out_png; dpi = 220, bbox_inches = "tight")
    PyPlot.plt.close(fig)
    @printf("Saved %s (Neff = %.1f)\n", out_png, d.neff)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

