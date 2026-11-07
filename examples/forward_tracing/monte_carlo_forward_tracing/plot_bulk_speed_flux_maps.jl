# Reproduce the six bulk-speed flux maps from sampler CSV or directly from MarsTP using PyPlot.
using MarsTP
using LinearAlgebra
using PyPlot

const HERE = @__DIR__
const OUTPUT = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(HERE, "bulk_speed_flux_200_400_600km.png")

moments = load_mhd_moments()
r, theta, phi = moments.r, moments.theta, moments.phi
scalar(A) = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, A, r, theta, phi)

species_itp = Dict(
    "O2+" => (n = scalar(moments.O2plus.n), u = ntuple(k -> scalar(Array(moments.O2plus.v[k, :, :, :])), 3)),
    "O+" => (n = scalar(moments.Oplus.n), u = ntuple(k -> scalar(Array(moments.Oplus.v[k, :, :, :])), 3)),
)

lons = collect(-180.0:2.0:180.0)
lats = collect(-90.0:2.0:90.0)
heights = (200.0, 400.0, 600.0)
values = fill(NaN, 3, 2, length(lats), length(lons))

for (row, h) in enumerate(heights), (col, sp) in enumerate(("O2+", "O+"))
    itp = species_itp[sp]
    r_shell = max(Rm + h * 1.0e3, Rinner + 1.0e-8)
    for (ilat, lat) in enumerate(lats), (ilon, lon) in enumerate(lons)
        lat == -90.0 && continue
        th = deg2rad(90.0 - lat)
        ph = mod2pi(deg2rad(lon))
        x = collect(MarsTP.TP.sph2cart(r_shell, th, ph))
        n_val = max(itp.n(x), 0.0)
        u_vec = [itp.u[1](x), itp.u[2](x), itp.u[3](x)]
        v = n_val * norm(u_vec) / 1.0e4
        v > 0.0 && (values[row, col, ilat, ilon] = v)
    end
end

valid = filter(x -> isfinite(x) && x > 0.0, vec(values))
norm = PyPlot.matplotlib.colors.LogNorm(minimum(valid), maximum(valid))

fig, axes = PyPlot.plt.subplots(3, 2; figsize = (9, 9), layout = "constrained", sharex = true, sharey = true)
fig.suptitle("Flux magnitude, n |V|"; fontsize = 14)
im = nothing
for (row, h) in enumerate(heights), (col, sp) in enumerate(("O2+", "O+"))
    ax = axes[row, col]
    im = ax.pcolormesh(lons, lats, values[row, col, :, :]; norm = norm, cmap = "turbo", shading = "nearest", rasterized = true)
    ax.set(xlim = (-180, 180), ylim = (-90, 90), xticks = [-180, -90, 0, 90, 180], yticks = [-90, -45, 0, 45, 90], aspect = "equal")
    letter = Char('a' + (row - 1) * 2 + (col - 1))
    ax.set_title("($letter) $sp, $(Int(h)) km"; loc = "left")
    col == 1 && ax.set_ylabel("MSO latitude (deg)")
    row == 3 && ax.set_xlabel("MSO longitude (deg)")
end
fig.colorbar(im; ax = collect(vec(axes)), location = "bottom", fraction = 0.05, pad = 0.035, aspect = 45, label = "n |V| (cm^-2 s^-1)")
mkpath(dirname(OUTPUT))
fig.savefig(OUTPUT; dpi = 300, bbox_inches = "tight", pad_inches = 0.12)
PyPlot.plt.close(fig)
println("Saved ", OUTPUT)

