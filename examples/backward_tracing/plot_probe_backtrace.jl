# Plot probe backtrace trajectories from trajectories.jsonl using MarsTP and PyPlot.
using MarsTP
using JSON
using PyPlot

data_file = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(project_path("outputs"), "trajectories.jsonl")
records = [JSON.parse(line) for line in eachline(data_file) if startswith(strip(line), "{")]

norm = PyPlot.matplotlib.colors.Normalize(10.0, 200.0)
cmap = PyPlot.plt.get_cmap("turbo")
colors = [cmap(norm(Float64(r["speed_kms"]))) for r in records]
paths = [ [Float64(pt[1]) Float64(pt[3])] for r in records for pt in (r["points"],) ]
paths_mat = [ [Float64(p[1]) Float64(p[3]) for p in r["points"]] for r in records ]
seg_list = [ hcat([Float64(p[1]) for p in r["points"]], [Float64(p[3]) for p in r["points"]]) for r in records ]

fig, ax = PyPlot.plt.subplots(figsize = (183 / 25.4, 172 / 25.4), layout = "constrained")
bs_mpb(ax, "XZ"; draw_mpb = false, sphere = false, boundary_color = "#444444", boundary_ls = "--", boundary_lw = 0.8)
bs_mpb(ax, "XZ"; draw_bs = false, sphere = false, boundary_color = "#444444", boundary_ls = ":", boundary_lw = 0.9)
plot_mars(ax; radius = 1.0, alpha = 0.85, zorder = 3)
for radius in (Rinner / Rm, 4.0)
    ax.add_patch(PyPlot.matplotlib.patches.Circle((0.0, 0.0), radius; fill = false, ls = "--", lw = 0.8, color = "#999999"))
end
lc = PyPlot.matplotlib.collections.LineCollection(seg_list; colors = colors, linewidths = 0.35, alpha = 0.28, rasterized = true)
ax.add_collection(lc)
ax.scatter([0.0], [2.0]; marker = "*", s = 190, c = "crimson", edgecolors = "white", zorder = 5)
ax.set(xlim = (-4.2, 4.2), ylim = (-4.2, 4.2), aspect = "equal", xlabel = "X / R_M", ylabel = "Z / R_M")
ax.grid(alpha = 0.15)
sm = PyPlot.matplotlib.cm.ScalarMappable(norm = norm, cmap = cmap)
fig.colorbar(sm; ax = ax, shrink = 0.8, label = "Initial speed (km/s)", pad = 0.025)

image_dir = joinpath(@__DIR__, "images")
mkpath(image_dir)
out_png = joinpath(image_dir, "probe_backtrace_xz_5000.png")
fig.savefig(out_png; dpi = 400)
PyPlot.plt.close(fig)
println("Saved ", out_png)

