# Compare MHD O2+ flux and Ti on 200/400 km shells in native MSO angles using PyPlot.
using MarsTP
using JSON
using Printf
using PyPlot

const IMAGE = joinpath(@__DIR__, "images")
mkpath(IMAGE)

lons = collect(-180.0:2.0:180.0)
lats = collect(-90.0:2.0:90.0)
heights_km = (200.0, 400.0)

flux_maps = zeros(Float64, length(heights_km), length(lats), length(lons))
ti_maps = zeros(Float64, length(heights_km), length(lats), length(lons))

for (ih, h) in enumerate(heights_km)
    src = load_ionosphere_source(; altitude_km = h)
    r = Rm + h * 1.0e3
    for (ilat, lat) in enumerate(lats), (ilon, lon) in enumerate(lons)
        theta = deg2rad(90.0 - lat)
        phi = mod2pi(deg2rad(lon))
        pos = MarsTP.TP.sph2cart(r, theta, phi)
        props = ionosphere_properties(src, pos)
        flux_maps[ih, ilat, ilon] = props.flux / 1.0e4
        ti_maps[ih, ilat, ilon] = props.Ti
    end
end

fig, axes = PyPlot.plt.subplots(2, 2; figsize = (183 / 25.4, 145 / 25.4), layout = "constrained", sharex = true, sharey = true)
for (col, (values, label, title)) in enumerate((
    (flux_maps, "n_i |U_i| (cm^-2 s^-1)", "O2+ flux"),
    (ti_maps, "T_i (K)", "O2+ ion temperature"),
))
    valid = values[:, lats .> -90.0, :]
    pos_vals = valid[valid .> 0.0]
    vmin = 10.0^floor(log10(minimum(pos_vals)))
    vmax = 10.0^ceil(log10(maximum(pos_vals)))
    norm = PyPlot.matplotlib.colors.LogNorm(vmin = vmin, vmax = vmax)
    im = nothing
    for (row, h) in enumerate(heights_km)
        ax = axes[row, col]
        slice = copy(values[row, :, :])
        slice[slice .<= 0.0] .= NaN
        im = ax.pcolormesh(lons, lats, slice; norm = norm, cmap = "turbo", shading = "nearest", rasterized = true)
        ax.set(xlim = (-180, 180), ylim = (-90, 90), xticks = [-180, -90, 0, 90, 180], yticks = [-90, -45, 0, 45, 90])
        letter = Char('a' + 2 * (row - 1) + (col - 1))
        ax.set_title(row == 1 ? "($letter) $(Int(h)) km | $title" : "($letter) $(Int(h)) km"; loc = "left")
        col == 1 && ax.set_ylabel("MSO latitude (deg)")
        row == 2 && ax.set_xlabel("MSO longitude (deg)")
    end
    fig.colorbar(im; ax = collect(axes[:, col]), location = "bottom", shrink = 0.9, pad = 0.06, label = label)
end

out_png = joinpath(IMAGE, "flux_Ti_200_400km.png")
fig.savefig(out_png; dpi = 350)
PyPlot.plt.close(fig)
println("Saved ", out_png)

