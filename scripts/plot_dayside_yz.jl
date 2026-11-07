# YZ projection of full trajectories selected by dayside initial position using MarsTP and PyPlot.
using MarsTP
using JSON
using PyPlot

const ROOT = project_path()
const PREVIEW = get(ENV, "YZ_PREVIEW", project_path("outputs", "dayside_yz.png"))
mkpath(dirname(abspath(PREVIEW)))

cmd = setenv(
    `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$ROOT $(project_path("examples", "forward_tracing", "sphere_trajectories.jl"))`,
    "RELEASE_ALTITUDE_KM" => "800", "PARTICLE_COUNT" => "1000", "TRACE_DT" => "0.1",
)
records = [JSON.parse(line) for line in eachline(cmd) if startswith(strip(line), "{") && JSON.parse(line)["points"][1][1] > 0.0]
paths = [hcat([Float64(p[2]) for p in r["points"]], [Float64(p[3]) for p in r["points"]]) for r in records]

fig, ax = PyPlot.plt.subplots(figsize = (7.3, 7.0), layout = "constrained")
plot_mars(ax; texture = false, facecolor = "#dddddd", edgecolor = "#888888", lw = 0.8, alpha = 0.55, zorder = 1)
lc = PyPlot.matplotlib.collections.LineCollection(paths; colors = "#16749a", linewidths = 0.65, alpha = 0.45, zorder = 3)
ax.add_collection(lc)
starts_y = [p[1, 1] for p in paths]
starts_z = [p[1, 2] for p in paths]
ax.scatter(starts_y, starts_z; c = "#333333", s = 4, alpha = 0.35, zorder = 4)
ax.set(xlim = (-4.15, 4.15), ylim = (-4.15, 4.15), aspect = "equal", xlabel = "Y / R_M", ylabel = "Z / R_M", title = "Dayside O2+")
ax.grid(alpha = 0.12)
fig.savefig(PREVIEW; dpi = 170)
PyPlot.plt.close(fig)
println("Preview: ", PREVIEW)

