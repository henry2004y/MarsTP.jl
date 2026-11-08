# Publication-quality script for Figure 2 of the Mars O2+ test-particle manuscript:
#   (a) O2+ number density at 400 km altitude
#   (b) O2+ bulk speed at 400 km altitude
#   (c) O2+ temperature at 400 km altitude
#   (d) Local O2+ Maxwellian f_xz at Longitude = 300°, Latitude = -60°
#   (e) Sampled Monte Carlo particle-rate weights w_i in (v_x, v_z)
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig2.jl [OUTPUT_PNG] [VTS_OR_RUN_PATH] [SAMPLING_MODE]
#
# SAMPLING_MODE options:
#   - "maxwellian_source" : Uses MarsTP.sample_maxwellian_source (Ts = 4*Ti Gaussian importance sampling)
#   - "uniform_cube"      : Uses uniform cube sampling on [-3.8*vth, +3.8*vth]^3 (matching legacy Fig. 2e)
#   - "saved_particles"   : Reads saved initial conditions from particles.csv in VTS_OR_RUN_PATH

using MarsTP
using StaticArrays
using LinearAlgebra
using Random
using Printf
using PyPlot

function _read_source_cells_csv(csv_path::AbstractString)
    lines = readlines(csv_path)
    length(lines) > 1 || error("Empty source_cells.csv: $csv_path")
    cells = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        p = split(line, ',')
        push!(
            cells, (;
                cell_id = parse(Int, p[1]),
                theta_index = parse(Int, p[2]),
                phi_index = parse(Int, p[3]),
                lon = parse(Float64, p[4]),
                lat = parse(Float64, p[5]),
                area = parse(Float64, p[6]),
                flux = parse(Float64, p[7]),
                n = parse(Float64, p[8]),
                Ti = parse(Float64, p[9]),
                ux = parse(Float64, p[10]),
                uy = parse(Float64, p[11]),
                uz = parse(Float64, p[12]),
            )
        )
    end
    return cells
end

function _read_cell_particles_csv(csv_path::AbstractString, target_cell_id::Int)
    vx = Float64[]
    vy = Float64[]
    vz = Float64[]
    w = Float64[]
    open(csv_path, "r") do io
        eof(io) && return vx, vy, vz, w
        readline(io) # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            p = split(line, ',')
            cid = parse(Int, p[2])
            if cid == target_cell_id
                push!(w, parse(Float64, p[3]))
                push!(vx, parse(Float64, p[9]))
                push!(vy, parse(Float64, p[10]))
                push!(vz, parse(Float64, p[11]))
            end
        end
    end
    return vx, vy, vz, w
end

function plot_fig2(
        out_png::AbstractString = joinpath(pwd(), "outputs", "paper_figures", "fig2.png");
        vts_path::AbstractString = MarsTP.data_path("mars_fields_without_crustal.vts"),
        run_dir::AbstractString = "",
        sampling_mode::Symbol = :maxwellian_source,
        altitude_km::Real = 400.0,
        sample_lon_deg::Real = 300.0,
        sample_lat_deg::Real = -60.0,
        n_samples::Int = 5000,
        seed::Int = 20260907,
    )
    if isdir(vts_path) && isempty(run_dir)
        run_dir = vts_path
        vts_path = MarsTP.data_path("mars_fields_without_crustal.vts")
    end
    src = load_ionosphere_source(vts_path; altitude_km = Float64(altitude_km))
    r_shell = max(Rm + Float64(altitude_km) * 1.0e3, Rinner + 1.0e-8)

    cells_csv = !isempty(run_dir) ? joinpath(run_dir, "source_cells.csv") : ""
    if !isempty(cells_csv) && isfile(cells_csv)
        cells = _read_source_cells_csv(cells_csv)
        nth = maximum(c.theta_index for c in cells)
        nph = maximum(c.phi_index for c in cells)
        lons = zeros(Float64, nph)
        lats = zeros(Float64, nth)
        n_map = fill(NaN, nth, nph)
        u_map = fill(NaN, nth, nph)
        t_map = fill(NaN, nth, nph)
        for c in cells
            # theta_index 1 is North Pole (+90°), nth is South Pole (-90°); flip row order so lat increases
            ilat = nth - c.theta_index + 1
            ilon = c.phi_index
            lons[ilon] = c.lon
            lats[ilat] = c.lat
            n_map[ilat, ilon] = max(c.n, 1.0e5)
            u_map[ilat, ilon] = hypot(c.ux, c.uy, c.uz) / 1.0e3
            t_map[ilat, ilon] = max(c.Ti, 50.0)
        end
    else
        lons = collect(0.0:1.5:360.0)
        lats = collect(-90.0:1.5:90.0)
        nlon, nlat = length(lons), length(lats)

        n_map = zeros(Float64, nlat, nlon)
        u_map = zeros(Float64, nlat, nlon)
        t_map = zeros(Float64, nlat, nlon)

        for (ilat, lat) in enumerate(lats), (ilon, lon) in enumerate(lons)
            theta = deg2rad(90.0 - clamp(lat, -89.999, 89.999))
            phi = mod2pi(deg2rad(lon))
            pos = MarsTP.TP.sph2cart(r_shell, theta, phi)
            p = ionosphere_properties(src, pos)
            n_map[ilat, ilon] = max(p.n, 1.0e5)
            u_map[ilat, ilon] = hypot(p.Ui[1], p.Ui[2], p.Ui[3]) / 1.0e3 # km/s
            t_map[ilat, ilon] = max(p.Ti, 50.0)
        end
    end

    # Local Maxwellian and Monte Carlo sampling at (Lon = 300°, Lat = -60°)
    theta0 = deg2rad(90.0 - Float64(sample_lat_deg))
    phi0 = mod2pi(deg2rad(Float64(sample_lon_deg)))
    pos0 = collect(MarsTP.TP.sph2cart(r_shell, theta0, phi0))
    er0 = SVector{3, Float64}(pos0 ./ norm(pos0))
    p0 = ionosphere_properties(src, pos0)

    mass = MarsTP.TP.SpeciesDict["O2+"].m
    kb = MarsTP.TP.kB
    vth = sqrt(kb * p0.Ti / mass)
    ux0, uy0, uz0 = p0.Ui[1], p0.Ui[2], p0.Ui[3]

    span = 3.8 * vth
    vx_grid = collect(range(ux0 - span, ux0 + span; length = 241))
    vz_grid = collect(range(uz0 - span, uz0 + span; length = 241))
    fxz_exact = [
        (p0.n / (2.0 * pi * vth^2)) * exp(-((vx - ux0)^2 + (vz - uz0)^2) / (2.0 * vth^2))
        for vz in vz_grid, vx in vx_grid
    ]

    # Area element of one cell on 400 km grid for physical rate weight scale ~1.4e19 s^-1
    dtheta = pi / 90.0
    dphi = 2pi / 180.0
    area_cell = r_shell^2 * sin(theta0) * dtheta * dphi * 16.0
    q_total = p0.n * hypot(ux0, uy0, uz0) * area_cell

    particles_csv = !isempty(run_dir) ? joinpath(run_dir, "particles.csv") : ""
    if sampling_mode == :saved_particles && !isempty(cells_csv) && isfile(cells_csv) && isfile(particles_csv)
        cells = _read_source_cells_csv(cells_csv)
        best_idx = argmin([hypot(c.lon - sample_lon_deg, c.lat - sample_lat_deg) for c in cells])
        target_cell = cells[best_idx]
        vx_s, vy_s, vz_s, w_i = _read_cell_particles_csv(particles_csv, target_cell.cell_id)
    elseif sampling_mode == :maxwellian_source
        rng = MersenneTwister(seed)
        sampled = sample_maxwellian_source(
            n_samples;
            position_m = SVector{3, Float64}(pos0),
            bulk_velocity_m_s = p0.Ui,
            temperature_ev = kb * p0.Ti / MarsTP.TP.eV,
            weights = MonteCarloWeight(
                source_number_density_m3 = p0.n,
                sampling_temperature_factor = 4.0,
            ),
            normal = er0,
            area_m2 = area_cell,
            rng = rng,
            flux_model = :bulk_speed,
        )
        vx_s = [Float64(s[4]) for s in sampled.initial_states]
        vy_s = [Float64(s[5]) for s in sampled.initial_states]
        vz_s = [Float64(s[6]) for s in sampled.initial_states]
        w_i = collect(Float64, sampled.rate_weights_s)
    else
        # Uniform velocity cube importance proposal matching reference Fig. 2e
        rng = MersenneTwister(seed)
        vx_s = Vector{Float64}(undef, n_samples)
        vy_s = Vector{Float64}(undef, n_samples)
        vz_s = Vector{Float64}(undef, n_samples)
        g_raw = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            vx_s[i] = ux0 + (2.0 * rand(rng) - 1.0) * span
            vy_s[i] = uy0 + (2.0 * rand(rng) - 1.0) * span
            vz_s[i] = uz0 + (2.0 * rand(rng) - 1.0) * span
            d2 = (vx_s[i] - ux0)^2 + (vy_s[i] - uy0)^2 + (vz_s[i] - uz0)^2
            g_raw[i] = exp(-d2 / (2.0 * vth^2))
        end
        w_i = q_total .* (g_raw ./ sum(g_raw))
    end

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 16
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 15.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 15.5

    fig = plt.figure(figsize = (14.2, 11.4), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.04, wspace = 0.05, hspace = 0.04)
    gs_main = fig.add_gridspec(1, 2; width_ratios = [1.05, 1.0])
    gs_left = gs_main[1, 1].subgridspec(3, 1)
    gs_right = gs_main[1, 2].subgridspec(2, 1)

    ax_a = fig.add_subplot(gs_left[1, 1])
    ax_b = fig.add_subplot(gs_left[2, 1])
    ax_c = fig.add_subplot(gs_left[3, 1])
    ax_d = fig.add_subplot(gs_right[1, 1])
    ax_e = fig.add_subplot(gs_right[2, 1])

    # (a) Density
    im_a = ax_a.pcolormesh(
        lons, lats, n_map;
        cmap = "turbo", norm = matplotlib.colors.LogNorm(vmin = 1.0e6, vmax = 1.0e9),
        shading = "auto", rasterized = true,
    )
    ax_a.scatter([0.0], [0.0]; marker = "*", s = 230, facecolor = "white", edgecolor = "black", lw = 1.1, zorder = 6, clip_on = false)
    ax_a.text(
        10.0, 22.0, "Subsolar";
        fontsize = 15.5, va = "center", ha = "left",
        bbox = Dict("boxstyle" => "square,pad=0.2", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.9),
        zorder = 6,
    )
    ax_a.set_title("(a) O\$_2^+\$ density at $(Int(round(altitude_km))) km"; fontsize = 18, pad = 7)
    cax_a = ax_a.inset_axes([1.03, 0.0, 0.038, 1.0])
    cb_a = fig.colorbar(im_a; cax = cax_a)
    cb_a.ax.tick_params(labelsize = 15)
    cb_a.set_label("Number density (m\$^{-3}\$)"; fontsize = 16.5)

    # (b) Speed
    im_b = ax_b.pcolormesh(
        lons, lats, u_map;
        cmap = "turbo", vmin = 0.0, vmax = 30.0,
        shading = "auto", rasterized = true,
    )
    ax_b.set_title("(b) O\$_2^+\$ speed at $(Int(round(altitude_km))) km"; fontsize = 18, pad = 7)
    cax_b = ax_b.inset_axes([1.03, 0.0, 0.038, 1.0])
    cb_b = fig.colorbar(im_b; cax = cax_b)
    cb_b.ax.tick_params(labelsize = 15)
    cb_b.set_label("Speed (km s\$^{-1}\$)"; fontsize = 16.5)

    # (c) Temperature
    im_c = ax_c.pcolormesh(
        lons, lats, t_map;
        cmap = "turbo", norm = matplotlib.colors.LogNorm(vmin = 1.0e2, vmax = 1.0e7),
        shading = "auto", rasterized = true,
    )
    ax_c.set_title("(c) O\$_2^+\$ temperature at $(Int(round(altitude_km))) km"; fontsize = 18, pad = 7)
    ax_c.set_xlabel("Longitude (°)"; fontsize = 17.5)
    cax_c = ax_c.inset_axes([1.03, 0.0, 0.038, 1.0])
    cb_c = fig.colorbar(im_c; cax = cax_c)
    cb_c.ax.tick_params(labelsize = 15)
    cb_c.set_label("Temperature (K)"; fontsize = 16.5)

    for ax in (ax_a, ax_b, ax_c)
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks(0:60:360)
        ax.set_yticks(-90:30:90)
        ax.tick_params(labelsize = 15.5)
        ax.set_ylabel("Latitude (°)"; fontsize = 17.5)
    end

    # (d) Local O2+ Maxwellian
    im_d = ax_d.pcolormesh(
        vx_grid ./ 1.0e3, vz_grid ./ 1.0e3, fxz_exact;
        cmap = "turbo", vmin = 0.0, vmax = maximum(fxz_exact),
        shading = "auto", rasterized = true,
    )
    ax_d.set_aspect("equal")
    ax_d.tick_params(labelsize = 15.5)
    ax_d.set_xlabel("\$v_x\$ (km s\$^{-1}\$)"; fontsize = 17.5)
    ax_d.set_ylabel("\$v_z\$ (km s\$^{-1}\$)"; fontsize = 17.5)
    ax_d.set_title("(d) Local O\$_2^+\$ Maxwellian\nLongitude = $(Int(round(sample_lon_deg)))°, Latitude = $(Int(round(sample_lat_deg)))°"; fontsize = 18, pad = 8)
    n_exp = floor(Int, log10(p0.n))
    n_mant = p0.n / 10.0^n_exp
    t_exp = floor(Int, log10(p0.Ti))
    t_mant = p0.Ti / 10.0^t_exp
    info_txt = @sprintf(
        "\$n = %.2f \\times 10^{%d}\$ m\$^{-3}\$\n\$V = (%.2f, %.4f, %.2f)\$ km/s\n\$T = %.2f \\times 10^{%d}\$ K",
        n_mant, n_exp, ux0 / 1.0e3, uy0 / 1.0e3, uz0 / 1.0e3, t_mant, t_exp,
    )
    ax_d.text(0.04, 0.05, info_txt; transform = ax_d.transAxes, color = "white", fontsize = 15, va = "bottom", ha = "left")
    cax_d = ax_d.inset_axes([1.035, 0.0, 0.045, 1.0])
    cb_d = fig.colorbar(im_d; cax = cax_d)
    cb_d.ax.tick_params(labelsize = 15)
    cb_d.set_label("\$f_{xz}\$ (s\$^2\$ m\$^{-5}\$)"; fontsize = 16.5)

    # (e) Particle-rate weights
    ord = sortperm(w_i)
    sc_e = ax_e.scatter(
        vx_s[ord] ./ 1.0e3, vz_s[ord] ./ 1.0e3;
        c = w_i[ord], cmap = "turbo", vmin = 0.0, vmax = maximum(w_i),
        s = 9, marker = "s", linewidths = 0, rasterized = true,
    )
    ax_e.set_xlim(extrema(vx_grid) ./ 1.0e3)
    ax_e.set_ylim(extrema(vz_grid) ./ 1.0e3)
    ax_e.set_aspect("equal")
    ax_e.tick_params(labelsize = 15.5)
    ax_e.set_xlabel("\$v_x\$ (km s\$^{-1}\$)"; fontsize = 17.5)
    ax_e.set_ylabel("\$v_z\$ (km s\$^{-1}\$)"; fontsize = 17.5)
    ax_e.set_title("(e) Particle-rate weights \$w_i\$"; fontsize = 18, pad = 8)
    cax_e = ax_e.inset_axes([1.035, 0.0, 0.045, 1.0])
    cb_e = fig.colorbar(sc_e; cax = cax_e)
    cb_e.ax.tick_params(labelsize = 15)
    cb_e.set_label("\$w_i\$ (s\$^{-1}\$)"; fontsize = 16.5)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 2 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "outputs", "paper_figures", "fig2.png")
    vts = length(ARGS) >= 2 ? ARGS[2] : MarsTP.data_path("mars_fields_without_crustal.vts")
    mode = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :maxwellian_source
    plot_fig2(out; vts_path = vts, sampling_mode = mode)
end

