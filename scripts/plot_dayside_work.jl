# Signed work on dayside trajectories using MarsTP and PyPlot.
using MarsTP
using JSON
using PyPlot

const ROOT = project_path()
const PREVIEW = get(ENV, "DAYSIDE_WORK_PREVIEW", project_path("outputs", "dayside_work.png"))
mkpath(dirname(abspath(PREVIEW)))

cmd = `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$ROOT $(joinpath(@__DIR__, "dayside_work.jl"))`
records = [JSON.parse(line) for line in eachline(cmd) if startswith(strip(line), "{")]

summaries = reduce(vcat, [Float64.(r["summary"])' for r in records])
arrays = [reduce(vcat, [Float64.(pt)' for pt in r["points"]]) for r in records]
limit = maximum(maximum(abs, @view a[:, 3:end]) for a in arrays) / 1000.0
norm = PyPlot.matplotlib.colors.SymLogNorm(linthresh = 0.1, vmin = -limit, vmax = limit, base = 10)

fig, axes = PyPlot.plt.subplots(1, 3; figsize = (17, 6), layout = "constrained")
lines_lc = nothing
for (ax, column, title) in zip(axes[1:2], (3, 4), ("Convection work", "Hall work"))
    bs_mpb(ax, "XZ"; sphere = false, boundary_color = "0.4")
    plot_mars(ax; alpha = 0.5, zorder = 1)
    segs = Matrix{Float64}[]
    vals = Float64[]
    for a in arrays, i in 1:(size(a, 1) - 1)
        push!(segs, [a[i, 1] a[i, 2]; a[i + 1, 1] a[i + 1, 2]])
        push!(vals, 0.5 * (a[i, column] + a[i + 1, column]) / 1000.0)
    end
    lines_lc = PyPlot.matplotlib.collections.LineCollection(segs; array = vals, cmap = "RdBu_r", norm = norm, linewidths = 0.6, alpha = 0.85, zorder = 3)
    ax.add_collection(lines_lc)
    ax.set(xlim = (-4.15, 4.15), ylim = (-4.15, 4.15), aspect = "equal", xlabel = "X / R_M", ylabel = "Z / R_M", title = title)
end
fig.colorbar(lines_lc; ax = collect(axes[1:2]), location = "bottom", label = "Cumulative signed work (keV)", shrink = 0.9, pad = 0.03)

ax = axes[3]
for (status, marker, color, label) in (
    ("outer", "o", "#16749a", "Outer boundary"),
    ("inner", "^", "#b46924", "Inner boundary"),
    ("time_limit", "s", "#9b418e", "Time limit"),
)
    m = [r["status"] == status for r in records]
    any(m) && ax.scatter(summaries[m, 1] ./ 1000.0, summaries[m, 2] ./ 1000.0; s = 18, alpha = 0.6, marker = marker, c = color, label = label)
end
ax.axhline(0.0; c = "0.6", lw = 0.7)
ax.axvline(0.0; c = "0.6", lw = 0.7)
ax.set_xscale("symlog"; linthresh = 0.1)
ax.set_yscale("symlog"; linthresh = 0.1)
ax.set(xlabel = "Final convection work (keV)", ylabel = "Final Hall work (keV)", title = "Per-particle net work")
ax.legend(frameon = false, fontsize = 9)
for a in axes
    a.grid(alpha = 0.12)
end
fig.savefig(PREVIEW; dpi = 170)
PyPlot.plt.close(fig)
println("Preview: ", PREVIEW)

