# Full-path forward-time energy gains, PSD-weighted over Vy using PyPlot.
using MarsTP
using DelimitedFiles
using JSON
using PyPlot

function read_named_csv(path)
    lines = readlines(path)
    header = split(strip(lines[1]), ",")
    data = readdlm(IOBuffer(join(lines[2:end], "\n")), ',', Float64)
    return Dict(String(header[i]) => data[:, i] for i in eachindex(header))
end

out = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "data")
a = read_named_csv(joinpath(out, "energy_xz.csv"))
vx = sort(unique(a["vx_kms"]))
vz = sort(unique(a["vz_kms"]))
shape = (length(vz), length(vx))
f = reshape(a["f_s2_m5"], length(vx), length(vz))'
values = [reshape(a[k], length(vx), length(vz))' ./ 1000.0 for k in ("conv_eV", "hall_eV", "total_eV")]
limit = maximum(maximum(abs, filter(isfinite, vec(v))) for v in values)

fig, axs = PyPlot.plt.subplots(1, 3; figsize = (183 / 25.4, 82 / 25.4), layout = "constrained", sharey = true)
norm = PyPlot.matplotlib.colors.SymLogNorm(linthresh = 0.001, linscale = 1.0, vmin = -limit, vmax = limit, base = 10)
im = nothing
for (i, (ax, v, title, letter)) in enumerate(zip(axs, values, ("Convection", "Hall", "Total"), ("a", "b", "c")))
    masked = copy(v)
    masked[(f .<= 0.0) .| .!isfinite.(v)] .= NaN
    im = ax.pcolormesh(vx, vz, masked; cmap = "coolwarm", norm = norm, shading = "nearest")
    ax.set(aspect = "equal", xlabel = "MSO v_x (km/s)", title = title)
    ax.text(0.0, 1.06, letter; transform = ax.transAxes, fontweight = "bold")
end
axs[1].set_ylabel("MSO v_z (km/s)")
fig.suptitle("O2+: full-path energy gain, PSD-weighted over v_y\nPreliminary velocity grid"; fontsize = 10)
fig.colorbar(im; ax = collect(axs), location = "bottom", shrink = 0.8, pad = 0.08, label = "Mean forward-time energy gain (keV)")

image_dir = joinpath(@__DIR__, "images")
mkpath(image_dir)
image = joinpath(image_dir, "energy_gain_xz_preliminary.png")
fig.savefig(image; dpi = 350)
PyPlot.plt.close(fig)
println("Saved ", image)

