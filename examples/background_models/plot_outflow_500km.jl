# Compare O2+ and O+ outward and speed flux at 500 km using MarsTP and PyPlot.
using MarsTP
using LinearAlgebra
using Printf
using PyPlot

const OUT = length(ARGS) >= 1 ? abspath(ARGS[1]) : project_path("outputs", "outflow_500km")
mkpath(OUT)

moments = load_mhd_moments()
r, theta, phi = moments.r, moments.theta, moments.phi
scalar(A) = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, A, r, theta, phi)

species_itp = Dict(
    "O2+" => (n = scalar(moments.O2plus.n), u = ntuple(k -> scalar(Array(moments.O2plus.v[k, :, :, :])), 3)),
    "O+" => (n = scalar(moments.Oplus.n), u = ntuple(k -> scalar(Array(moments.Oplus.v[k, :, :, :])), 3)),
)

lons = collect(-180.0:2.0:180.0)
lats = collect(-90.0:2.0:90.0)
r_shell = Rm + 500.0e3

outward_flux = Dict{String, Matrix{Float64}}()
speed_flux = Dict{String, Matrix{Float64}}()

for sp in ("O2+", "O+")
    itp = species_itp[sp]
    out_m = zeros(Float64, length(lats), length(lons))
    spd_m = zeros(Float64, length(lats), length(lons))
    for (ilat, lat) in enumerate(lats), (ilon, lon) in enumerate(lons)
        th = deg2rad(90.0 - lat)
        ph = mod2pi(deg2rad(lon))
        x = collect(MarsTP.TP.sph2cart(r_shell, th, ph))
        er = x ./ norm(x)
        n_val = max(itp.n(x), 0.0)
        u_vec = [itp.u[1](x), itp.u[2](x), itp.u[3](x)]
        ur = dot(u_vec, er)
        out_m[ilat, ilon] = n_val * max(ur, 0.0) / 1.0e4
        spd_m[ilat, ilon] = n_val * norm(u_vec) / 1.0e4
    end
    outward_flux[sp] = out_m
    speed_flux[sp] = spd_m
end

for (flux_dict, stem, label) in (
    (outward_flux, "outflow_500km", "Outward flux, n max(U_r, 0) (cm^-2 s^-1)"),
    (speed_flux, "speed_flux_500km", "Flux magnitude, n |U| (cm^-2 s^-1)"),
)
    vmax = max(maximum(flux_dict["O2+"][lats .> -90.0, :]), maximum(flux_dict["O+"][lats .> -90.0, :]))
    norm = PyPlot.matplotlib.colors.Normalize(0.0, vmax)
    fig, axes = PyPlot.plt.subplots(1, 2; figsize = (10.4, 3.8), layout = "constrained", sharex = true, sharey = true)
    im = nothing
    for (i, sp) in enumerate(("O2+", "O+"))
        ax = axes[i]
        im = ax.pcolormesh(lons, lats, flux_dict[sp]; cmap = "turbo", norm = norm, shading = "nearest", rasterized = true)
        ax.set(xlim = (-180, 180), ylim = (-90, 90), xticks = [-180, -90, 0, 90, 180], yticks = [-90, -45, 0, 45, 90], xlabel = "MSO longitude (deg)")
        ax.set_title("($(Char('a' + i - 1))) $sp, 500 km"; loc = "left")
        ax.set_aspect("equal")
    end
    axes[1].set_ylabel("MSO latitude (deg)")
    fig.colorbar(im; ax = collect(axes), location = "bottom", fraction = 0.1, pad = 0.07, aspect = 45, label = label)
    out_png = joinpath(OUT, "$stem.png")
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight", pad_inches = 0.12)
    PyPlot.plt.close(fig)
    println("Saved ", out_png)
end

