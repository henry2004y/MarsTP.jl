# Three altitudes by two species, one shared log scale for flux magnitude using PyPlot.
using MarsTP
using LinearAlgebra
using PyPlot

const HEIGHTS = (200.0, 400.0, 600.0)
const OUT = length(ARGS) >= 1 ? abspath(ARGS[1]) : project_path("outputs", "speed_log_200_400_600km")
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
speed = zeros(Float64, length(HEIGHTS), 2, length(lats), length(lons))

for (row, h) in enumerate(HEIGHTS), (col, sp) in enumerate(("O2+", "O+"))
    itp = species_itp[sp]
    r_shell = max(Rm + h * 1.0e3, Rinner + 1.0e-8)
    for (ilat, lat) in enumerate(lats), (ilon, lon) in enumerate(lons)
        th = deg2rad(90.0 - lat)
        ph = mod2pi(deg2rad(lon))
        x = collect(MarsTP.TP.sph2cart(r_shell, th, ph))
        n_val = max(itp.n(x), 0.0)
        u_vec = [itp.u[1](x), itp.u[2](x), itp.u[3](x)]
        speed[row, col, ilat, ilon] = n_val * norm(u_vec) / 1.0e4
    end
end

valid = speed[:, :, lats .> -90.0, :]
pos_vals = valid[valid .> 0.0]
norm_log = PyPlot.matplotlib.colors.LogNorm(minimum(pos_vals), maximum(pos_vals))

fig, axes = PyPlot.plt.subplots(3, 2; figsize = (9, 9), layout = "constrained", sharex = true, sharey = true)
fig.suptitle("Flux magnitude, n |V|"; fontsize = 14)
im = nothing
for (row, h) in enumerate(HEIGHTS), (col, sp) in enumerate(("O2+", "O+"))
    ax = axes[row, col]
    slice = copy(speed[row, col, :, :])
    slice[slice .<= 0.0] .= NaN
    im = ax.pcolormesh(lons, lats, slice; norm = norm_log, cmap = "turbo", shading = "nearest", rasterized = true)
    ax.set(xlim = (-180, 180), ylim = (-90, 90), xticks = [-180, -90, 0, 90, 180], yticks = [-90, -45, 0, 45, 90], aspect = "equal")
    letter = Char('a' + (row - 1) * 2 + (col - 1))
    ax.set_title("($letter) $sp, $(Int(h)) km"; loc = "left")
    col == 1 && ax.set_ylabel("MSO latitude (deg)")
    row == 3 && ax.set_xlabel("MSO longitude (deg)")
end
fig.colorbar(im; ax = collect(vec(axes)), location = "bottom", fraction = 0.05, pad = 0.035, aspect = 45,
    label = "n |V| (cm^-2 s^-1)")
out_png = joinpath(OUT, "speed_200_400_600km.png")
fig.savefig(out_png; dpi = 300, bbox_inches = "tight", pad_inches = 0.12)
PyPlot.plt.close(fig)
println("Saved ", out_png)

