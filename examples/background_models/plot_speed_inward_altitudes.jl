# Three altitudes by four columns, separate shared scales for speed/inward flux using PyPlot.
using MarsTP
using LinearAlgebra
using PyPlot

const HEIGHTS = (200.0, 400.0, 600.0)
const OUT = length(ARGS) >= 1 ? abspath(ARGS[1]) : project_path("outputs", "speed_inward_200_400_600km")
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
inward = fill(NaN, length(HEIGHTS), 2, length(lats), length(lons))

for (row, h) in enumerate(HEIGHTS), (col, sp) in enumerate(("O2+", "O+"))
    itp = species_itp[sp]
    r_shell = max(Rm + h * 1.0e3, Rinner + 1.0e-8)
    for (ilat, lat) in enumerate(lats), (ilon, lon) in enumerate(lons)
        th = deg2rad(90.0 - lat)
        ph = mod2pi(deg2rad(lon))
        x = collect(MarsTP.TP.sph2cart(r_shell, th, ph))
        er = x ./ norm(x)
        n_val = max(itp.n(x), 0.0)
        u_vec = [itp.u[1](x), itp.u[2](x), itp.u[3](x)]
        speed[row, col, ilat, ilon] = n_val * norm(u_vec) / 1.0e4
        ur = dot(u_vec, er)
        if ur < 0.0
            inward[row, col, ilat, ilon] = n_val * ur / 1.0e4
        end
    end
end

norm_spd = PyPlot.matplotlib.colors.Normalize(0.0, maximum(speed[:, :, lats .> -90.0, :]))
inw_valid = filter(isfinite, vec(inward[:, :, lats .> -90.0, :]))
norm_inw = PyPlot.matplotlib.colors.Normalize(minimum(inw_valid), 0.0)

fig, axes = PyPlot.plt.subplots(3, 4; figsize = (18, 9), layout = "constrained", sharex = true, sharey = true)
im_spd = nothing
im_inw = nothing
for (row, h) in enumerate(HEIGHTS)
    for (col, sp) in enumerate(("O2+", "O+"))
        ax_s = axes[row, col]
        im_spd = ax_s.pcolormesh(lons, lats, speed[row, col, :, :]; norm = norm_spd, cmap = "turbo", shading = "nearest", rasterized = true)
        ax_s.set(xlim = (-180, 180), ylim = (-90, 90), aspect = "equal")
        ax_s.set_title("$sp |V|, $(Int(h)) km"; loc = "left")

        ax_i = axes[row, col + 2]
        im_inw = ax_i.pcolormesh(lons, lats, inward[row, col, :, :]; norm = norm_inw, cmap = "turbo", shading = "nearest", rasterized = true)
        ax_i.set(xlim = (-180, 180), ylim = (-90, 90), aspect = "equal")
        ax_i.set_title("$sp inward, $(Int(h)) km"; loc = "left")
    end
    axes[row, 1].set_ylabel("MSO latitude (deg)")
end
for c in 1:4
    axes[3, c].set_xlabel("MSO longitude (deg)")
end
fig.colorbar(im_spd; ax = collect(vec(axes[:, 1:2])), location = "bottom", fraction = 0.05, pad = 0.035, label = "n |V| (cm^-2 s^-1)")
fig.colorbar(im_inw; ax = collect(vec(axes[:, 3:4])), location = "bottom", fraction = 0.05, pad = 0.035, label = "n V_r, V_r < 0 (cm^-2 s^-1)")
out_png = joinpath(OUT, "speed_inward_200_400_600km.png")
fig.savefig(out_png; dpi = 300, bbox_inches = "tight", pad_inches = 0.12)
PyPlot.plt.close(fig)
println("Saved ", out_png)

