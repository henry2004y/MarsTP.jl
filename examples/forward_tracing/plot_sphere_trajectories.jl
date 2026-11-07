# Run sphere_trajectories.jl and preview XZ projections using MarsTP and PyPlot.
using MarsTP
using JSON
using PyPlot

const ROOT = project_path()
const PREVIEW = get(ENV, "TRAJECTORY_PREVIEW", joinpath(@__DIR__, "images", "trajectories_xz_legacy.png"))
const ALTITUDE = parse(Float64, get(ENV, "RELEASE_ALTITUDE_KM", "800"))
mkpath(dirname(abspath(PREVIEW)))

cmd = `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$ROOT $(joinpath(@__DIR__, "sphere_trajectories.jl"))`
records = [JSON.parse(line) for line in eachline(cmd) if startswith(strip(line), "{")]

colors = Dict("inner" => "#b46924", "outer" => "#16749a", "time_limit" => "#9b418e")
groups = (
    records,
    [r for r in records if r["points"][1][1] > 0.0],
    [r for r in records if r["points"][1][1] <= 0.0],
)

fig, axes = PyPlot.plt.subplots(1, 3; figsize = (18, 6.5), layout = "constrained")
for (ax, selected, title) in zip(axes, groups, ("All O2+", "Dayside O2+", "Nightside O2+"))
    lim = 4.15
    bs_mpb(ax, "XZ"; draw_mpb = false, sphere = false, boundary_color = "#454545", boundary_ls = "--", boundary_lw = 1.0)
    bs_mpb(ax, "XZ"; draw_bs = false, sphere = false, boundary_color = "#454545", boundary_ls = ":", boundary_lw = 1.2)
    plot_mars(ax; zorder = 1, alpha = 0.6)
    for (status, color) in colors
        paths = [hcat([Float64(p[1]) for p in r["points"]], [Float64(p[3]) for p in r["points"]]) for r in selected if r["status"] == status]
        isempty(paths) && continue
        lc = PyPlot.matplotlib.collections.LineCollection(paths; colors = color, linewidths = 0.45, alpha = 0.32, zorder = 3)
        ax.add_collection(lc)
    end
    starts_x = [Float64(r["points"][1][1]) for r in selected]
    starts_z = [Float64(r["points"][1][3]) for r in selected]
    ax.scatter(starts_x, starts_z; s = 3, c = "#333333", alpha = 0.25, zorder = 4)
    angle = range(0.0, 2pi, length = 500)
    radius = (3390.0 + ALTITUDE) / 3390.0
    ax.plot(radius .* cos.(angle), radius .* sin.(angle); color = "#999999", lw = 0.7)
    ax.set(xlim = (-lim, lim), ylim = (-lim, lim), aspect = "equal", xlabel = "X / R_M", ylabel = "Z / R_M", title = title)
    ax.grid(alpha = 0.12)
end
fig.savefig(PREVIEW; dpi = 160)
PyPlot.plt.close(fig)
println("Preview: ", PREVIEW)

