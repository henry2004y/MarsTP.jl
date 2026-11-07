# Plot the marginal VDF f_xz(vx, vz) using MarsTP and PyPlot.
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
image_dir = joinpath(@__DIR__, "images")
mkpath(image_dir)

a = read_named_csv(joinpath(out, "vdf.csv"))
vx = sort(unique(a["vx_kms"]))
vz = sort(unique(a["vz_kms"]))
f = reshape(a["f_total_s2_m5"], length(vx), length(vz))'
positive = f[f .> 0.0]
upper = 10.0^ceil(log10(maximum(positive)))
lower = max(10.0^floor(log10(minimum(positive))), upper / 1.0e8)

fig, ax = PyPlot.plt.subplots(figsize = (183 / 25.4, 165 / 25.4), layout = "constrained")
masked = copy(f)
masked[masked .<= 0.0] .= NaN
im = ax.pcolormesh(vx, vz, masked; cmap = "turbo", norm = PyPlot.matplotlib.colors.LogNorm(lower, upper), shading = "nearest", rasterized = true)
ax.set(xlim = (minimum(vx) - 5.0, maximum(vx) + 5.0), ylim = (minimum(vz) - 5.0, maximum(vz) + 5.0), aspect = "equal",
    xlabel = "MSO v_x (km/s)", ylabel = "MSO v_z (km/s)")
ax.set_title("O2+ at (0, 0, 2 R_M), preliminary grid\nf_xz = int f dv_y"; fontsize = 11, pad = 10)
fig.colorbar(im; ax = ax, pad = 0.025, shrink = 0.83, label = "f_xz (s^2 m^-5)")
ax.axhline(0.0; color = "#444444", alpha = 0.22, lw = 0.6)
ax.axvline(0.0; color = "#444444", alpha = 0.22, lw = 0.6)

out_png = joinpath(image_dir, "vdf_xz_preliminary.png")
fig.savefig(out_png; dpi = 350)
PyPlot.plt.close(fig)
println("Saved ", out_png)

